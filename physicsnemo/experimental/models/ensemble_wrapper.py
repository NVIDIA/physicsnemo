# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
physicsnemo/experimental/models/ensemble_wrapper.py

Model-agnostic ensemble wrapper for uncertainty quantification.

Motivation
----------
PhysicsNeMo 25.08 introduced ensemble-based confidence estimation as a
Jupyter notebook workflow in ``physicsnemo-cfd``, scoped to the DoMINO
automotive aerodynamics NIM. This module promotes that pattern to a
model-agnostic, reusable utility in the core library: any
``physicsnemo.Module`` can be wrapped for ensemble-based uncertainty
quantification in two lines.

Usage
-----

.. code-block:: python

    from physicsnemo.experimental.models.ensemble_wrapper import EnsembleWrapper

    # Train N models with different seeds using standard PhysicsNeMo loops
    models = [train_my_model(seed=i) for i in range(5)]

    # Wrap for uncertainty-aware inference
    ensemble = EnsembleWrapper(models)

    # Drop-in forward (returns mean — compatible with any existing code)
    mean = ensemble(x)

    # Uncertainty-aware inference
    result = ensemble.predict_with_uncertainty(x)
    print(result.mean.shape)          # same as single-model output
    print(result.std.shape)           # epistemic uncertainty estimate
    print(result.predictions.shape)   # (N, *output_shape) all member outputs

    # Load from saved checkpoints
    ensemble = EnsembleWrapper.from_checkpoints(
        model_cls=MyModel,
        checkpoint_paths=["ckpt_0.pt", "ckpt_1.pt", "ckpt_2.pt"],
        **model_init_kwargs,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Union

import torch
import torch.nn as nn

from physicsnemo import Module
from physicsnemo.models.meta import ModelMetaData


# ---------------------------------------------------------------------------
# Metadata  (required by MOD-001 / physicsnemo.Module convention)
# ---------------------------------------------------------------------------


@dataclass
class EnsembleWrapperMeta(ModelMetaData):
    r"""Metadata for ``EnsembleWrapper``.

    Attributes
    ----------
    name : str
        Human-readable model name.
    jit : bool
        Whether the model supports TorchScript JIT compilation.
        ``False`` because member models may not individually support JIT.
    cuda_graphs : bool
        Whether the model supports CUDA graphs. ``False`` by default;
        set to ``True`` only when all member models support CUDA graphs.
    amp_cpu : bool
        Whether the model supports Automatic Mixed Precision on CPU.
    amp_gpu : bool
        Whether the model supports Automatic Mixed Precision on GPU.
    """

    name: str = "EnsembleWrapper"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = True
    amp_gpu: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnsemblePrediction:
    r"""
    Output of ``EnsembleWrapper.predict_with_uncertainty()``.

    Attributes
    ----------
    mean : torch.Tensor
        Element-wise mean over all ensemble members,
        shape :math:`(B, \ldots)`.
    std : torch.Tensor
        Element-wise standard deviation over all ensemble members —
        an estimate of epistemic uncertainty — shape :math:`(B, \ldots)`.
    predictions : torch.Tensor
        Stacked raw outputs from all :math:`N` ensemble members,
        shape :math:`(N, B, \ldots)`.

    Notes
    -----
    The standard deviation here captures **epistemic** (model) uncertainty
    arising from different weight initialisations and training trajectories.
    It does not capture aleatoric (data) uncertainty. For a decomposed
    estimate, consider combining ``EnsembleWrapper`` with members that
    themselves output predictive distributions (e.g. models with a learned
    variance head).
    """

    mean: torch.Tensor
    std: torch.Tensor
    predictions: torch.Tensor


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnsembleWrapper(Module):
    r"""
    Model-agnostic ensemble wrapper for uncertainty quantification.

    Wraps a collection of independently trained ``physicsnemo.Module``
    instances. At inference, each member is queried and the wrapper
    returns the element-wise mean (via ``forward``) and standard deviation
    (via ``predict_with_uncertainty``) across member outputs.

    The wrapper itself is a ``physicsnemo.Module`` (following rule MOD-001),
    so it supports PhysicsNeMo checkpointing, versioning, and the model
    registry. Member models are stored as a ``torch.nn.ModuleList`` so
    their parameters are included in ``state_dict`` and moved together
    with ``.to(device)``.

    ``forward`` returns only the mean prediction so that ``EnsembleWrapper``
    is a **drop-in replacement** for any single model — existing training
    loops, metrics, and inference scripts require no changes.

    Parameters
    ----------
    models : list of Module
        Pre-trained ``physicsnemo.Module`` instances forming the ensemble.
        All members must accept the same input signature and return tensors
        of the same shape.

    Raises
    ------
    ValueError
        If ``models`` is empty.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn import FullyConnected
    >>> from physicsnemo.experimental.models.ensemble_wrapper import EnsembleWrapper
    >>>
    >>> # Build and (notionally) train 5 members with different seeds
    >>> members = [FullyConnected(in_features=4, out_features=1) for _ in range(5)]
    >>> ensemble = EnsembleWrapper(members)
    >>>
    >>> x = torch.randn(32, 4)
    >>>
    >>> # Drop-in forward: returns mean, shape (32, 1)
    >>> mean = ensemble(x)
    >>>
    >>> # Uncertainty-aware forward
    >>> result = ensemble.predict_with_uncertainty(x)
    >>> result.mean.shape, result.std.shape
    (torch.Size([32, 1]), torch.Size([32, 1]))

    See Also
    --------
    EnsemblePrediction : Output dataclass for ``predict_with_uncertainty``.
    EnsembleWrapper.from_checkpoints : Construct from saved checkpoint files.
    """

    def __init__(self, models: List[Module]) -> None:
        if len(models) == 0:
            raise ValueError(
                "EnsembleWrapper requires at least one member model. "
                "Received an empty list."
            )
        super().__init__(meta=EnsembleWrapperMeta())
        # Store members as ModuleList so parameters are properly tracked
        # and the ensemble can be moved with .to(device) in one call.
        self.members: nn.ModuleList = nn.ModuleList(models)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_members(self) -> int:
        r"""Number of models in the ensemble."""
        return len(self.members)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Run ensemble inference and return the mean prediction.

        Returns only the mean so that ``EnsembleWrapper`` is a drop-in
        replacement for a single model. For the full uncertainty estimate
        use ``predict_with_uncertainty``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, \ldots)`.

        Returns
        -------
        torch.Tensor
            Element-wise mean over all ensemble members,
            shape :math:`(B, \ldots)`.
        """
        return self.predict_with_uncertainty(x).mean

    # ------------------------------------------------------------------
    # Uncertainty-aware inference
    # ------------------------------------------------------------------

    def predict_with_uncertainty(self, x: torch.Tensor) -> EnsemblePrediction:
        r"""
        Run ensemble inference and return mean, std, and all member outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, \ldots)`.

        Returns
        -------
        EnsemblePrediction
            Dataclass with fields:

            - ``mean`` — element-wise mean, shape :math:`(B, \ldots)`
            - ``std``  — element-wise standard deviation (epistemic
              uncertainty), shape :math:`(B, \ldots)`
            - ``predictions`` — stacked member outputs,
              shape :math:`(N, B, \ldots)`
        """
        # Stack outputs from all N members along a new leading dimension.
        # Shape: (N, B, *output_shape)
        predictions = torch.stack([member(x) for member in self.members], dim=0)

        return EnsemblePrediction(
            mean=predictions.mean(dim=0),
            # correction=0: population std (MLE) — ensures std=0 for N=1,
            # avoids NaN from Bessel's correction when ensemble size is 1.
            std=predictions.std(dim=0, correction=0),
            predictions=predictions,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        model_cls: Type[Module],
        checkpoint_paths: List[Union[str, Path]],
        map_location: Union[str, torch.device, None] = None,
        **model_kwargs,
    ) -> "EnsembleWrapper":
        r"""
        Construct an ``EnsembleWrapper`` from saved checkpoint files.

        Each checkpoint must have been saved with
        ``torch.save(model.state_dict(), path)`` or via PhysicsNeMo's
        built-in checkpoint utilities.

        Parameters
        ----------
        model_cls : type
            The ``physicsnemo.Module`` subclass used for each member.
        checkpoint_paths : list of str or Path
            Paths to the saved ``state_dict`` files, one per ensemble member.
        map_location : str, torch.device, or None, optional
            Passed directly to ``torch.load``. Useful for loading
            GPU-trained checkpoints on CPU. Default: ``None``.
        **model_kwargs
            Keyword arguments forwarded to ``model_cls.__init__``.

        Returns
        -------
        EnsembleWrapper
            Ensemble with one member per checkpoint.

        Raises
        ------
        FileNotFoundError
            If any of the provided checkpoint paths does not exist.

        Examples
        --------
        >>> ensemble = EnsembleWrapper.from_checkpoints(
        ...     model_cls=FullyConnected,
        ...     checkpoint_paths=["ckpt_0.pt", "ckpt_1.pt", "ckpt_2.pt"],
        ...     map_location="cpu",
        ...     in_features=4,
        ...     out_features=1,
        ... )
        """
        models = []
        for path in checkpoint_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"EnsembleWrapper.from_checkpoints: checkpoint not found at '{path}'."
                )
            model = model_cls(**model_kwargs)
            state = torch.load(path, map_location=map_location, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        return cls(models)
