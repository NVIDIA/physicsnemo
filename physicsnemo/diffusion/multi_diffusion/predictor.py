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

"""Multi-diffusion predictor wrapper for patch-based diffusion sampling."""

from typing import Any

from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.base import Predictor
from physicsnemo.diffusion.multi_diffusion.models import MultiDiffusionModel2D
from physicsnemo.diffusion.multi_diffusion.patching import GridPatching2D
from physicsnemo.diffusion.utils.utils import _unwrap_module


class MultiDiffusionPredictor(Predictor):
    r"""Predictor for sampling from a trained
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D`.

    Satisfies the :class:`~physicsnemo.diffusion.Predictor` protocol, so it
    plugs into any sampling utility that accepts a ``Predictor``
    (:func:`~physicsnemo.diffusion.samplers.sample`,
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
    and all standard solvers) with no other changes.  All patch-based logic
    — patching the state, running the per-patch predictions and fusing them
    back to the global domain — is handled internally.

    The wrapped model must have grid patching configured via
    :meth:`~MultiDiffusionModel2D.set_grid_patching` before constructing the
    predictor.

    .. warning::

        :class:`MultiDiffusionPredictor` is intended for **test-time
        sampling**: it is not suitable for training. The wrapped
        multi-diffusion model should already be trained before being passed
        to the predictor.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        A trained multi-diffusion model with grid patching already configured.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        When provided, the shape should be :math:`(B, *cond_dims)`.
        Conditioning information at the global resolution, bound once at
        construction and reused at every diffusion step. Pass ``None`` for
        unconditional models.
    fuse : bool, default=True
        Whether to fuse per-patch outputs back to the global resolution
        before returning.
    **model_kwargs : Any
        Additional keyword arguments bound once at construction and
        forwarded to the wrapped model at every call.

    See Also
    --------
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D` :
        The multi-diffusion wrapper used for training; its grid patching
        must be configured before creating the predictor.
    :class:`~physicsnemo.diffusion.Predictor` :
        The protocol this class implements.
    :func:`~physicsnemo.diffusion.samplers.sample` :
        The main sampling entry point.

    Examples
    --------
    **Example 1:** Predictor in isolation. Input and output live at the
    global resolution; patching, per-patch prediction and fusing are all
    handled internally:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionPredictor,
    ... )
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> # Create a trained multi-diffusion model (training omitted here)
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_grid_patching(patch_shape=(8, 8))  # P = 4 patches per sample
    >>> _ = md.eval()
    >>>
    >>> predictor = MultiDiffusionPredictor(md)
    >>> x = torch.randn(2, 3, 16, 16)  # global-resolution state
    >>> t = 0.5 * torch.ones(2)
    >>> predictor(x, t).shape  # fused output at global resolution
    torch.Size([2, 3, 16, 16])
    >>>
    >>> # fuse=False returns raw per-patch predictions — (P*B, C, Hp, Wp)
    >>> predictor.fuse = False
    >>> predictor(x, t).shape
    torch.Size([8, 3, 8, 8])

    **Example 2:** Unconditional sampling. The predictor plugs straight into
    the standard diffusion sampling stack (noise scheduler, denoiser,
    solver):

    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> # The wrapped model must have grid patching + fuse=True for sampling
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_grid_patching(patch_shape=(8, 8), overlap_pix=2, fuse=True)
    >>> _ = md.eval()
    >>>
    >>> predictor = MultiDiffusionPredictor(md)
    >>> scheduler = EDMNoiseScheduler()
    >>> denoiser = scheduler.get_denoiser(x0_predictor=predictor)
    >>> xN = torch.randn(2, 3, 16, 16)  # initial noise at global resolution
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])

    **Example 3:** Conditional sampling with mixed conditioning — an image
    that shares the spatial resolution (patched like the state) and a vector
    (repeated across patches). Both kinds are bound once at construction
    and handled internally:

    >>> from tensordict import TensorDict
    >>> class MultiCondBackbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = torch.nn.Conv2d(6, 3, 1)
    ...         self.vec_proj = torch.nn.Linear(5, 3 * 8 * 8)
    ...     def forward(self, x, t, condition=None):
    ...         img = condition["image"]
    ...         vec = condition["vector"]
    ...         h = self.conv(torch.cat([x, img], dim=1))
    ...         return h + self.vec_proj(vec).view_as(h)
    >>>
    >>> md = MultiDiffusionModel2D(
    ...     MultiCondBackbone(),
    ...     global_spatial_shape=(16, 16),
    ...     condition_patch={"image": True},  # image is patched
    ...     # vector has no flag: default is repeat across patches
    ... )
    >>> md.set_grid_patching(patch_shape=(8, 8), fuse=True)
    >>> _ = md.eval()
    >>>
    >>> condition = TensorDict({
    ...     "image":  torch.randn(2, 3, 16, 16),
    ...     "vector": torch.randn(2, 5),
    ... }, batch_size=[2])
    >>> predictor = MultiDiffusionPredictor(md, condition=condition)
    >>> denoiser = scheduler.get_denoiser(x0_predictor=predictor)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        fuse: bool = True,
        **model_kwargs: Any,
    ) -> None:
        self._md_model: MultiDiffusionModel2D = _unwrap_module(
            model, MultiDiffusionModel2D
        )

        if not isinstance(self._md_model._patching, GridPatching2D):
            raise RuntimeError(
                "MultiDiffusionPredictor requires grid patching to be configured. "
                "Call model.set_grid_patching() before creating the predictor."
            )

        self._patching: GridPatching2D = self._md_model._patching

        self.model = model
        self._model_kwargs = model_kwargs
        self._P: int = self._patching.patch_num

        # Pre-patch condition once (without PE)
        self._cond_patched: Tensor | TensorDict | None = self._md_model.patch_condition(
            condition
        )

        # Pre-patch PE for B=1, expanded to (P*B) at call time
        if self._md_model.pos_embd is not None:
            self._pos_embd_patched: Tensor | None = self._md_model.patch_x(
                self._md_model.pos_embd.unsqueeze(0)
            )  # (P, C_PE, Hp, Wp)
        else:
            self._pos_embd_patched = None

        # PE will be injected by this class from the pre-patched cache;
        # suppress the wrapper's own per-step PE injection to avoid redundant work
        self._md_model._skip_positional_embedding_injection = True

        self.fuse = fuse

    @property
    def fuse(self) -> bool:
        """Whether the predictor fuses per-patch outputs back to the global
        resolution at each call."""
        return self._md_model._fuse

    @fuse.setter
    def fuse(self, value: bool) -> None:
        """Enable or disable fusing at each call."""
        self._md_model._fuse = value

    def __call__(
        self,
        x: Float[Tensor, "B C H W"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, "B C H W"] | Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Run the predictor on a noisy latent and diffusion time at the
        global resolution.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent at global resolution, shape :math:`(B, C, H, W)`.
        t : torch.Tensor
            Diffusion time, shape :math:`(B,)`.

        Returns
        -------
        torch.Tensor
            If ``self.fuse`` is ``True``: prediction at the global
            resolution, shape :math:`(B, C, H, W)`.
            Otherwise: per-patch predictions, shape
            :math:`(P \times B, C, H_p, W_p)`.
        """
        B = x.shape[0]
        x_patched = self._md_model.patch_x(x)  # (P*B, C, Hp, Wp)
        t_patched = self._md_model.patch_t(t)  # (P*B,)
        cond = self._cond_patched
        if self._pos_embd_patched is not None:
            # Expand cached PE from (P, ...) to (P*B, ...) and inject
            pe = self._pos_embd_patched.repeat_interleave(B, dim=0)
            cond = self._md_model._inject_patched_pos_embd(cond, pe, self._P * B)
        return self._md_model(
            x_patched,
            t_patched,
            condition=cond,
            x_is_patched=True,
            t_is_patched=True,
            condition_is_patched=True,
            **self._model_kwargs,
        )
