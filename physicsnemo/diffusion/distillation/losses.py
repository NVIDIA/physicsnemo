# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

r"""Train diffusion models with distillation losses in your own training loop."""

import importlib
from collections.abc import Mapping
from typing import Any

import torch
from jaxtyping import Float, Shaped
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.base import DiffusionExtraTimes
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    LinearGaussianNoiseScheduler,
)

from .base import LossMap

_METHODS = {
    "sCM": ("scm._fastgen", "_SCMLoss"),
    "MeanFlow": ("meanflow._fastgen", "_MeanFlowLoss"),
    "DMD2": ("dmd2._fastgen", "_DMD2Loss"),
    "f-distill": ("f_distill._fastgen", "_FDistillLoss"),
}


class DistillationLoss:
    r"""Compute sCM, MeanFlow, DMD2, or f-distill losses for a diffusion student.

    The selected method defines the training loss and constructs its targets.
    Each call returns ``total_loss`` for backpropagation and detached values
    for logging. Your training loop controls optimization.

    Initialize and wrap your models before constructing the loss. Call it on
    each batch, backpropagate through ``losses["total_loss"]``, and step the
    optimizer for the selected phase. See the distillation guide for
    compilation options.

    Parameters
    ----------
    student : torch.nn.Module
        Model to train, with the :class:`~physicsnemo.diffusion.DiffusionModel`
        interface.
    noise_scheduler : LinearGaussianNoiseScheduler or DomainParallelNoiseScheduler
        Noise schedule for the student. Its ``sample_time`` method supplies
        training times unless you pass times to the loss call.
    method : {"sCM", "MeanFlow", "DMD2", "f-distill"}
        Training loss. MeanFlow students receive ``(t, DiffusionExtraTimes)``
        and predict average velocity over the interval from ``t`` to ``r``.
    teacher : torch.nn.Module, optional
        Model providing training targets. DMD2 and f-distill require a
        teacher. For sCM and MeanFlow, omit it to train from clean data
        and noise without a pretrained teacher.
    auxiliary_models : mapping, optional
        Models trained during the auxiliary phase. Supply ``fake_score`` for
        DMD2 and f-distill, and ``discriminator`` for adversarial training.
        Initialize these models before passing them.
    method_options : mapping, optional
        Method-specific loss settings. Unknown settings raise an error.
    model_options : mapping, optional
        Settings for ``student``, ``teacher``, or ``fake_score``. Set ``prediction_type`` to
        describe what each model predicts. ``model_kwargs`` forwards model-specific options,
        and ``noise_scheduler`` selects a different schedule for that model.
        Adversarial teachers also need ``feature_names`` and a
        ``feature_fn(prediction, extras)`` callback that returns named features
        as a ``TensorDict``. The teacher always returns its prediction; the
        callback selects model-specific extras for the discriminator.

    Examples
    --------
    A MeanFlow loss can train a student directly from clean data and sampled times.
    Backpropagation through ``total_loss`` reaches the student parameter.

    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
    >>> class Flow(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.gain = torch.nn.Parameter(torch.tensor(0.5))
    ...     def forward(self, x, time, condition=None):
    ...         return self.gain * x
    >>> student = Flow()
    >>> loss_fn = DistillationLoss(student, RectifiedFlowNoiseScheduler(), method="MeanFlow")
    >>> losses = loss_fn(torch.randn(2, 3), step=0)
    >>> losses["total_loss"].backward()
    >>> student.gain.grad is not None
    True
    """

    def __init__(
        self,
        student: torch.nn.Module,
        noise_scheduler: LinearGaussianNoiseScheduler | DomainParallelNoiseScheduler,
        *,
        method: str,
        teacher: torch.nn.Module | None = None,
        auxiliary_models: Mapping[str, torch.nn.Module] | None = None,
        method_options: Mapping[str, Any] | None = None,
        model_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {tuple(_METHODS)}."
            )
        module, name = _METHODS[method]
        implementation = getattr(
            importlib.import_module(f"{__package__}.methods.{module}"), name
        )
        auxiliary_options = {}
        if method in ("DMD2", "f-distill"):
            auxiliary_options["auxiliary_models"] = dict(auxiliary_models or {})
        elif auxiliary_models:
            raise ValueError(f"{method} does not use auxiliary models.")
        self._implementation = implementation(
            student,
            noise_scheduler,
            teacher=teacher,
            method_options=dict(method_options or {}),
            model_options=dict(model_options or {}),
            **auxiliary_options,
        )

    def _bind_execution_models(self, models: Mapping[str, torch.nn.Module]) -> None:
        r"""Rebind trainer execution wrappers while preserving loss configuration."""
        self._implementation._bind_execution_models(models)

    def __call__(
        self,
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"]
        | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]]
        | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        *,
        phase: str | None = None,
        step: int = 0,
        return_outputs: bool = False,
        update_state: bool = True,
        **call_options: Any,
    ) -> LossMap | tuple[LossMap, Mapping[str, Any]]:
        r"""Compute the loss and logging values for one batch.

        Pass explicit times to use a chosen diffusion interval, or omit
        them to sample training times. Backpropagate through ``total_loss``
        only. Set ``return_outputs=True`` to inspect detached predictions
        and times from the same calculation.

        Parameters
        ----------
        x0 : Tensor
            Clean states with shape :math:`(B, \ldots)`.
        t : Tensor or tuple[Tensor, DiffusionExtraTimes], optional
            Diffusion times to use instead of sampling them. For MeanFlow,
            pass ``(t, DiffusionExtraTimes(r=r, batch_size=[B]))`` with
            :math:`r \leq t`; the loss uses this pair as given.
        condition : Tensor or TensorDict, optional
            Conditioning forwarded to the supplied models.
        phase : str, optional
            Training phase. sCM and MeanFlow accept only ``"student"``
            or ``None``. For DMD2 and f-distill, choose ``"student"``
            to train the student or ``"auxiliary"`` to train the fake-score
            model and optional discriminator. Step the matching optimizers.
        step : int, optional, default=0
            Training step, used to adjust loss warmup.
        return_outputs : bool, optional, default=False
            Also return detached method diagnostics without extra sampling.
        update_state : bool, optional, default=True
            Set to ``False`` to compute the loss without updating its statistics.
        **call_options : Any
            Method-specific controls, such as negative conditioning. A
            ``sample_t_fn(N, *, device, dtype)`` callback can sample diffusion
            times instead of passing ``t``. For MeanFlow, return
            ``(t, DiffusionExtraTimes(r=r, batch_size=[B]))``.

        Returns
        -------
        LossMap or tuple
            Dictionary containing ``total_loss`` for backpropagation and
            detached entries for logging. With ``return_outputs=True``, also
            return a dictionary of detached predictions and diffusion times.
        """
        return self._implementation.evaluate(
            x0,
            t,
            condition,
            phase=phase,
            step=step,
            return_outputs=return_outputs,
            update_state=update_state,
            **call_options,
        )

    def state_dict(self) -> dict[str, Any]:
        r"""Return running loss statistics to save alongside model checkpoints."""
        return self._implementation.loss_state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Restore saved statistics after creating the loss with matching settings."""
        self._implementation.load_loss_state_dict(state)


class WeightedDistillationLoss(DistillationLoss):
    r"""Compute distillation objectives with masks or element weights.

    Unlike :class:`DistillationLoss`, this loss accepts ``weight`` at call
    time. Zero weights exclude entries from residuals and adaptive statistics.
    Positive weights multiply the final squared residual once, without changing
    the adaptive statistics. Mean reductions keep the full element count,
    matching ``WeightedMSEDSMLoss``. Samples with no positive weights contribute
    zero, including the sCM log-variance term.

    Use a binary mask to exclude missing targets or positive weights to adjust
    their contribution. Weights must be finite, nonnegative, and broadcast to
    the state shape. Replace invalid inputs before calling the loss: masking
    the loss cannot prevent them from affecting predictions elsewhere.

    Examples
    --------
    Use a channel mask to exclude the middle feature from a weighted loss.
    Expanding the mask across the batch gives one weight per sample and feature.

    >>> weight = torch.tensor([[1.0, 0.0, 1.0]])
    >>> weight.expand(2, 3).shape
    torch.Size([2, 3])

    See :class:`DistillationLoss` for model setup and a training example.
    """

    def __call__(
        self,
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"]
        | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]]
        | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        *,
        weight: Float[Tensor, " #B *#dims"] | None = None,
        **kwargs: Any,
    ) -> LossMap | tuple[LossMap, Mapping[str, Any]]:
        r"""Apply the same loss with optional element weights."""
        return super().__call__(x0, t, condition, weight=weight, **kwargs)
