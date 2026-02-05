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
import warnings
from typing import Any

import torch
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.models.diffusion_unets import (
    DhariwalUNet,
    SongUNet,
    SongUNetPosEmbd,
    SongUNetPosLtEmbd,
)


class ConcatConditionWrapper(Module):
    r"""
    Wrapper that handles channel-concatenated conditioning in addition to
    optional vector conditioning.

    This wrapper adapts backbones with different conditioning signatures to the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface by concatenating
    image-like conditions to ``x`` and routing vector conditions to the
    appropriate argument. It is intended for cases where image-like conditioning
    (i.e., conditioning with the same spatial dimensions as ``x``) should be
    concatenated along the channel dimension to the noised latent state.

    Externally, wrapping with this wrapper will allow a backbone with an
    incompatible conditioning signature to satisfy the :class:`~physicsnemo.diffusion.DiffusionModel`
    interface. Currently, the wrapper supports the following backbones:

     - :class:`~physicsnemo.models.diffusion_unets.SongUNet`,
     - :class:`~physicsnemo.models.diffusion_unets.SongUNetPosEmbd`,
     - :class:`~physicsnemo.models.diffusion_unets.SongUNetPosLtEmbd`,
     - :class:`~physicsnemo.models.diffusion_unets.DhariwalUNet`,
     - :class:`~physicsnemo.experimental.models.dit.DiT`.
     - Any backbone with an forward signatude matching that of the `DiT` model:
       ``model(x, t, condition=None, **model_kwargs)`` where ``condition`` is a
       tensor of shape :math:`(B, d)` and ``**model_kwargs`` are additional
       keyword arguments forwarded to the backbone.

    The wrapper supports conditioning passed as either a ``TensorDict`` or a
    ``torch.Tensor``. If a ``TensorDict`` is passed, it must use the keys
    ``cond_concat`` and ``cond_vec`` to identify the image and vector
    conditioning tensors, respectively, or the user may specify the expected
    keys using the ``image_cond_key`` and ``vector_cond_key`` arguments when
    instantiating the wrapper.

    If a ``torch.Tensor`` is passed, it is treated as the image conditioning
    input to be concatenated (i.e., treated like the value of ``cond_concat``
    when a ``TensorDict`` is passed).

    The wrapper will route arguments approporiately depending on the backbone
    type. The default behavior for unknown backbones is to concatenate the value
    of ``cond_concat`` (if provided) and pass the value of ``cond_vec`` as the
    ``condition`` keyword argument.

    Parameters
    ----------
    model : physicsnemo.Module
        Backbone model to wrap.
    image_cond_key : str, optional
        TensorDict key for the image conditioning tensor to concatenate to
        ``x`` along channels, by default ``"cond_concat"``.
    vector_cond_key : str, optional
        TensorDict key for the vector conditioning tensor to pass through to
        the backbone, by default ``"cond_vec"``.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Conditioning data. Use a ``TensorDict`` to explicitly supply
        concatenated conditioning and vector conditioning as keys
        ``cond_concat`` and ``cond_vec`` (or the custom keys
        specified by ``image_cond_key`` and ``vector_cond_key``).
        Alternately, supply a plain ``Tensor`` input to be concatenated.
    **model_kwargs : Any
        Additional keyword arguments forwarded to the backbone.

    Outputs
    -------
    torch.Tensor
        Model output with the same shape as ``x``.

    Examples
    --------
    Wrap :class:`~physicsnemo.models.diffusion_unets.SongUNet` with channel
    concatenation and vector conditioning:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from physicsnemo.models.diffusion_unets import SongUNet
    >>> net = SongUNet(in_channels=4, out_channels=3, label_dim=4, img_resolution=8)
    >>> wrapper = ConcatConditionWrapper(net)
    >>> x = torch.randn(2, 3, 8, 8)
    >>> t = torch.rand(2)
    >>> condition = TensorDict(
    ...     {
    ...         "cond_concat": torch.randn(2, 1, 8, 8),
    ...         "cond_vec": torch.randn(2, 4),
    ...     },
    ...     batch_size=[2],
    ... )
    >>> out = wrapper(x, t, condition)
    >>> out.shape
    torch.Size([2, 3, 8, 8])

    Wrap :class:`~physicsnemo.experimental.models.dit.DiT` similarly:

    >>> from physicsnemo.experimental.models.dit import DiT
    >>> dit = DiT(in_channels=4, out_channels=3, condition_dim=4, input_size=8, patch_size=4)
    >>> dit_wrapper = ConcatConditionWrapper(dit)
    >>> out = dit_wrapper(x, t, condition)
    >>> out.shape
    torch.Size([2, 3, 8, 8])
    """

    def __init__(
        self,
        model: Module,
        image_cond_key: str = "cond_concat",
        vector_cond_key: str = "cond_vec",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_cond_key = image_cond_key
        self.vector_cond_key = vector_cond_key

    def forward(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"],
        condition: Float[torch.Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[torch.Tensor, " B *dims"]:
        r"""
        Forward pass for the conditioned wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size.
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(B,)`.
        condition : torch.Tensor, TensorDict, or None, optional, default=None
            Conditioning data. Use a ``TensorDict`` to supply ``cond_concat`` and
            ``cond_vec`` tensors. A plain tensor is treated as the concatenated
            conditioning input (equivalent to ``cond_concat``).
        **model_kwargs : Any
            Additional keyword arguments forwarded to the backbone.

        Returns
        -------
        torch.Tensor
            Model output with the same shape as ``x``.
        """
        cond_concat: torch.Tensor | None = None
        cond_vec: torch.Tensor | None = None

        if isinstance(condition, TensorDict):
            if self.image_cond_key in condition:
                cond_concat = condition[self.image_cond_key]
            if self.vector_cond_key in condition:
                cond_vec = condition[self.vector_cond_key]
            if cond_concat is None or cond_vec is None:
                raise ValueError(
                    "Condition TensorDict must include at least one of "
                    f"'{self.image_cond_key}' and '{self.vector_cond_key}'."
                    f" If you are only supplying image-like conditioning for"
                    f" concatenation and don't need vector conditioning, "
                    f"supply a plain torch.Tensor instead of a TensorDict."
                )
        elif isinstance(condition, torch.Tensor):
            if (
                self.image_cond_key != "cond_concat"
                or self.vector_cond_key != "cond_vec"
            ):
                warnings.warn(
                    f"ConcatConditionWrapper was instantiated with custom image and vector "
                    f"conditioning keys '{self.image_cond_key}' and '{self.vector_cond_key}' "
                    f"but a plain torch.Tensor was passed as condition. The tensor will be "
                    f"treated as the image conditioning input to be concatenated."
                )
            cond_concat = condition
        elif condition is not None:
            raise TypeError("Condition must be a torch.Tensor, TensorDict, or None.")

        if not torch.compiler.is_compiling():
            batch_size = x.shape[0]
            if t.shape != (batch_size,):
                raise ValueError(
                    f"Expected t to have shape ({batch_size},) matching batch size of "
                    f"x, but got {t.shape}."
                )
            if cond_vec is not None:
                if cond_vec.shape[0] != batch_size:
                    raise ValueError(
                        f"Condition vector has batch size {cond_vec.shape[0]} "
                        f"but expected {batch_size} to match x."
                    )
                if cond_vec.ndim != 2:
                    raise ValueError(
                        f"Condition vector must have 2 dimensions (batch, vector dim), "
                        f"got {cond_vec.ndim}."
                    )
            if cond_concat is not None:
                if cond_concat.shape[0] != batch_size:
                    raise ValueError(
                        f"Condition concat tensor has batch size {cond_concat.shape[0]} "
                        f"but expected {batch_size} to match x."
                    )
                if cond_concat.ndim != x.ndim:
                    raise ValueError(
                        f"Condition concat tensor must have {x.ndim} dims to match x, "
                        f"got {cond_concat.ndim}."
                    )
                if cond_concat.shape[2:] != x.shape[2:]:
                    raise ValueError(
                        "Condition concat tensor must match x spatial dimensions, "
                        f"got {cond_concat.shape[2:]} vs {x.shape[2:]}."
                    )

        if cond_concat is not None:
            x = torch.cat([x, cond_concat], dim=1)

        if isinstance(
            self.model, (SongUNet, SongUNetPosEmbd, SongUNetPosLtEmbd, DhariwalUNet)
        ):
            augment_labels = model_kwargs.pop("augment_labels", None)
            return self.model(
                x,
                noise_labels=t,
                class_labels=cond_vec,
                augment_labels=augment_labels,
            )

        return self.model(x, t, condition=cond_vec, **model_kwargs)
