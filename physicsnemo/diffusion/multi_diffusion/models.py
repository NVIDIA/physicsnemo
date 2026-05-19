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

"""Multi-diffusion model wrapper for patch-based diffusion."""

import warnings
from collections import defaultdict
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.diffusion.multi_diffusion.patching import (
    GridPatching2D,
    RandomPatching2D,
)


class MultiDiffusionModel2D(Module):
    r"""Multi-diffusion model wrapper for 2D patch-based diffusion.

    Multi-diffusion is a method useful for scaling diffusion models to large
    domains. A multi-diffusion model splits a 2D latent state into smaller
    patches, processes each patch independently through the wrapped model, and
    optionally fuses the outputs back into a full-resolution image.

    The wrapper handles:

    - Patching the state :math:`\mathbf{x}` into :math:`P` smaller
      patches, expanding the batch dimension from :math:`B` to
      :math:`P \times B`.
    - For conditional diffusion models, pre-processing each conditioning tensor
      according to some specified strategies: patching, interpolating to patch
      resolution, or simply repeating along the batch dimension.
    - Extracting positional embeddings for each patch and injecting them into
      the condition under the key ``"positional_embedding"``. This optional
      feature is useful to encode the relative position of the patches within
      the global domain into the condition.
    - Calling the wrapped model on the patched inputs and the pre-processed
      conditioning tensors.
    - Optionally fusing the per-patch outputs back to the global spatial
      resolution (grid patching only).

    Before a forward pass, a patching strategy must be configured via
    :meth:`set_random_patching` (typically for training) or
    :meth:`set_grid_patching` (typically for sampling/inference).

    The wrapped ``model`` must be an instance of
    :class:`~physicsnemo.core.Module` that satisfies the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface:

    .. code-block:: python

        model(
            x: torch.Tensor,       # Noisy state, shape: (P*B, C, Hp, Wp)
            t: torch.Tensor,       # Diffusion time, shape: (P*B,)
            condition: TensorDict | None = None, # Pre-processed conditioning tensors, shape: (P*B, *cond_dims)
            **model_kwargs: Any,
        ) -> torch.Tensor          # Prediction, shape: (P*B, C, Hp, Wp)

    The ``MultiDiffusionModel2D`` wrapper itself also satisfies the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface.

    .. important::

        The wrapped model is responsible for consuming the patched inputs and
        the pre-processed conditioning tensors appropriately. For example, if
        the wrapped model concatenates conditioning tensors to the input, it
        should be designed to handle the pre-processed (patched, interpolated,
        expanded, etc.) conditioning tensors.

    **Condition pre-processing strategies.** Each conditioning tensor is
    pre-processed according to one of three mutually exclusive strategies
    controlled by ``condition_patch`` and ``condition_interp``:

    - **Patch** (``condition_patch=True``): the conditioning tensor is
      decomposed into the same spatial patches as the state
      :math:`\mathbf{x}`. Requires a 4D tensor :math:`(B, C, H, W)` whose
      spatial dimensions match ``global_spatial_shape``. Useful to provide
      local, patch-level information to the model.
    - **Interpolate** (``condition_interp=True``): the conditioning tensor
      is bilinearly interpolated to the patch spatial resolution
      :math:`(H_p, W_p)` and repeated for each of the :math:`P` patches.
      Requires a 4D tensor :math:`(B, C, H', W')` (spatial dimensions
      need not match the global shape). Useful to encode a coarse global
      view into each patch.
    - **Default** (both ``False``): the tensor is simply repeated
      :math:`P` times along the batch dimension without spatial
      processing.  Useful for vector-valued conditioning.

    A given conditioning key cannot have both ``condition_patch`` and
    ``condition_interp`` set to ``True`` simultaneously.

    **Converting inputs to patch-compatible format.** The wrapper exposes
    three public methods that convert each input to the patch-compatible
    format with shape :math:`(P \times B, ...)`:

    - :meth:`patch_x`: spatial patching of the global domain, or global state
        :math:`\mathbf{x}`. Can be used to patch any global spatial tensor with
        shape :math:`(B, C, H, W)` to the patch-compatible format with shape
        :math:`(P \times B, C, H_p, W_p)`.
    - :meth:`patch_t`: batch-dimension expansion of the diffusion time.
    - :meth:`patch_condition`: patching / interpolation / expansion of
      the condition, depending on the configured strategy.

    These methods are called internally by :meth:`forward`, but can also
    be called externally for finer control (e.g., from a loss function
    that needs to add per-patch noise).

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network to wrap, with the signature described
        above. Must be an instance of :class:`~physicsnemo.core.Module` that
        satisfies the :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
    global_spatial_shape : Tuple[int, int]
        Height and width :math:`(H, W)` of the global (un-patched) spatial
        domain.
    positional_embedding : Literal["learnable", "sinusoidal", "linear"] | None, default=None
        Type of positional embedding to generate. Controls how global spatial
        coordinates are encoded into the conditioning. ``"learnable"`` creates
        a trainable parameter grid. ``"sinusoidal"`` uses fixed sin/cos
        encodings. ``"linear"`` uses a rectilinear grid over
        :math:`[-1, 1]^2`. ``None`` disables positional embeddings. When
        enabled, patches of the embedding grid are extracted and injected into
        the condition under the key ``"positional_embedding"`` during
        :meth:`forward`. The wrapped model must accept a ``TensorDict``
        condition and consume this key.
    channels_positional_embedding : int, default=4
        Number of channels :math:`C_{PE}` in the positional embedding grid.
        For ``"sinusoidal"`` must be a multiple of 4. For ``"linear"`` must
        be 2. For ``"learnable"`` can be any positive integer. Ignored when
        ``positional_embedding`` is ``None``.
    condition_patch : bool | Dict[str, bool], default=False
        Controls whether conditioning tensors are patched. When a single
        ``bool``, the flag applies uniformly to every conditioning tensor
        (or to the single ``Tensor`` condition). When a ``Dict[str, bool]``,
        each key maps to a specific key in a ``TensorDict`` condition; keys
        not present default to ``False``.
    condition_interp : bool or Dict[str, bool], default=False
        Controls whether conditioning tensors are interpolated to patch
        resolution. Follows the same ``bool`` / ``Dict[str, bool]``
        convention as ``condition_patch``.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state. Shape :math:`(B, C, H, W)` at global resolution,
        or :math:`(P \times B, C, H_p, W_p)` if ``x_is_patched=True``.
    t : torch.Tensor
        Diffusion time. Shape :math:`(B,)`, or :math:`(P \times B,)` if
        ``t_is_patched=True``.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Conditioning information at **global** resolution (batch size
        :math:`B`), or already in patch-compatible format if
        ``condition_is_patched=True``. When positional embeddings are enabled
        and ``condition_is_patched=False``, must be a ``TensorDict`` or
        ``None``.
    x_is_patched : bool, default=False
        If ``True``, ``x`` is assumed to already be in patch-compatible
        shape :math:`(P \times B, C, H_p, W_p)` and :meth:`patch_x` is
        skipped.
    t_is_patched : bool, default=False
        If ``True``, ``t`` is assumed to already be in patch-compatible
        shape :math:`(P \times B,)` and :meth:`patch_t` is skipped.
    condition_is_patched : bool, default=False
        If ``True``, ``condition`` is assumed to already be in
        patch-compatible format and both :meth:`patch_condition` and
        positional-embedding injection are skipped.

    **model_kwargs : Any
        Additional keyword arguments forwarded to the wrapped model.

    Outputs
    -------
    torch.Tensor
        If fusing is enabled (grid patching with ``fuse=True``), the output
        has shape :math:`(B, C, H, W)`. Otherwise the output has shape
        :math:`(P \times B, C, H_p, W_p)`.

    Notes
    -----
    Reference: Bar-Tal, O., Yariv, L., Lipman, Y. and Dekel, T., 2023.
    `MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
    <https://arxiv.org/abs/2302.08113>`_

    See Also
    --------
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionMSEDSMLoss` :
        Patch-based denoising score matching loss for use with this wrapper.
    :class:`~physicsnemo.diffusion.multi_diffusion.RandomPatching2D` :
        Random patching strategy (training).
    :class:`~physicsnemo.diffusion.multi_diffusion.GridPatching2D` :
        Grid patching strategy (sampling).

    Examples
    --------
    **Example 1:** Unconditional model: training with random patches then
    sampling with grid patches:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = UnconditionalModel()
    >>> md_model = MultiDiffusionModel2D(model, global_spatial_shape=(16, 16))
    >>>
    >>> # Training: random patching, P=6 patches of 8x8 per batch element
    >>> md_model.set_random_patching(patch_shape=(8, 8), patch_num=6)
    >>> x0 = torch.randn(2, 3, 16, 16)  # clean global state
    >>> sigma = 0.5
    >>> xt = x0 + sigma * torch.randn_like(x0)  # noisy global state
    >>> t = sigma * torch.ones(2)
    >>> x0_hat = md_model(xt, t)  # patched denoised estimate, shape: (P*B, C, Hp, Wp)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>> # Patch x0 for loss computation
    >>> x0_patched = md_model.patch_x(x0)  # patched global state, shape: (P*B, C, Hp, Wp)
    >>> loss = ((x0_hat - x0_patched) ** 2).mean()
    >>>
    >>> # Re-draw random patch positions for the next training step
    >>> md_model.reset_patch_indices()
    >>> x0_hat = md_model(xt, t)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # One can also patch x and t externally
    >>> xt_patched = md_model.patch_x(xt)
    >>> t_patched = md_model.patch_t(t)
    >>> x0_hat = md_model(xt_patched, t_patched, x_is_patched=True, t_is_patched=True)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # -- Sampling: grid patching with overlap and fusion --
    >>> _ = md_model.eval()
    >>> md_model.set_grid_patching(patch_shape=(8, 8), overlap_pix=2, fuse=True)
    >>> xN = torch.randn(2, 3, 16, 16)  # noisy global state
    >>> t = 0.5 * torch.ones(2)
    >>> denoised = md_model(xN, t)  # denoised global state, shape: (B, C, H, W)
    >>> denoised.shape
    torch.Size([2, 3, 16, 16])

    **Example 2:** Conditional model with a single image condition that is
    patched alongside the state:

    >>> class ConditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # 5 input channels: 3 (state) + 2 (conditioning image)
    ...         self.net = torch.nn.Conv2d(5, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(torch.cat([x, condition], dim=1))
    >>>
    >>> cond_md_model = MultiDiffusionModel2D(
    ...     model=ConditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ...     condition_patch=True,
    ... )
    >>>
    >>> # Training: random patching
    >>> cond_md_model.set_random_patching(patch_shape=(8, 8), patch_num=6)
    >>> x0 = torch.randn(2, 3, 16, 16)  # clean global state
    >>> xt = x0 + 0.5 * torch.randn_like(x0)  # noisy global state
    >>> t = 0.5 * torch.ones(2)
    >>> cond_img = torch.randn(2, 2, 16, 16)  # conditioning image
    >>> x0_hat = cond_md_model(xt, t, condition=cond_img)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>> x0_patched = cond_md_model.patch_x(x0)
    >>> loss = ((x0_hat - x0_patched) ** 2).mean()
    >>>
    >>> # Re-draw random patch positions for the next training step
    >>> cond_md_model.reset_patch_indices()
    >>> x0_hat = cond_md_model(xt, t, condition=cond_img)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # One can also patch x and t externally
    >>> xt_patched = cond_md_model.patch_x(xt)
    >>> t_patched = cond_md_model.patch_t(t)
    >>> x0_hat = cond_md_model(xt_patched, t_patched,
    ...                        condition=cond_img,
    ...                        x_is_patched=True, t_is_patched=True)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # -- Sampling: grid patching with overlap and fusion --
    >>> _ = cond_md_model.eval()
    >>> cond_md_model.set_grid_patching(patch_shape=(8, 8), overlap_pix=2, fuse=True)
    >>> xN = torch.randn(2, 3, 16, 16)  # noisy global state
    >>> denoised = cond_md_model(xN, t, condition=cond_img)  # denoised global state, shape: (B, C, H, W)
    >>> denoised.shape
    torch.Size([2, 3, 16, 16])

    **Example 3:** Conditional model with positional embeddings and two
    conditioning tensors (an image interpolated to patch resolution and a
    vector repeated for each patch):

    >>> class MultiCondModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # 9 input channels: 3 (state) + 2 (interpolated image) + 4 (positional embedding)
    ...         self.net = torch.nn.Conv2d(9, 3, 1)
    ...         self.vec_proj = torch.nn.Linear(5, 3 * 8 * 8)
    ...     def forward(self, x, t, condition=None):
    ...         # Wrapped model is designed to consume the conditioning
    ...         # tensors and positional embeddings
    ...         img = condition["image"]
    ...         pe = condition["positional_embedding"]
    ...         vec = condition["vector"]
    ...         h = self.net(torch.cat([x, img, pe], dim=1))
    ...         return h + self.vec_proj(vec).view_as(h)
    >>>
    >>> from tensordict import TensorDict
    >>> mc_md_model = MultiDiffusionModel2D(
    ...     model=MultiCondModel(),
    ...     global_spatial_shape=(16, 16),
    ...     positional_embedding="sinusoidal",
    ...     channels_positional_embedding=4,
    ...     condition_interp={"image": True},
    ... )
    >>>
    >>> # Training: random patching
    >>> mc_md_model.set_random_patching(patch_shape=(8, 8), patch_num=6)
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> xt = x0 + 0.5 * torch.randn_like(x0)  # noisy global state
    >>> t = 0.5 * torch.ones(2)
    >>> cond = TensorDict({
    ...     "image": torch.randn(2, 2, 16, 16),
    ...     "vector": torch.randn(2, 5),
    ... }, batch_size=[2])
    >>> x0_hat = mc_md_model(xt, t, condition=cond)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>> x0_patched = mc_md_model.patch_x(x0)
    >>> loss = ((x0_hat - x0_patched) ** 2).mean()
    >>>
    >>> # Re-draw random patch positions for the next training step
    >>> mc_md_model.reset_patch_indices()
    >>> x0_hat = mc_md_model(xt, t, condition=cond)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # One can also patch x and t externally
    >>> xt_patched = mc_md_model.patch_x(xt)
    >>> t_patched = mc_md_model.patch_t(t)
    >>> x0_hat = mc_md_model(xt_patched, t_patched,
    ...                      condition=cond,
    ...                      x_is_patched=True, t_is_patched=True)
    >>> x0_hat.shape
    torch.Size([12, 3, 8, 8])
    >>>
    >>> # -- Sampling: grid patching with overlap and fusion --
    >>> _ = mc_md_model.eval()
    >>> mc_md_model.set_grid_patching(patch_shape=(8, 8), overlap_pix=2, fuse=True)
    >>> xN = torch.randn(2, 3, 16, 16)  # noisy global state
    >>> denoised = mc_md_model(xN, t, condition=cond)  # denoised global state, shape: (B, C, H, W)
    >>> denoised.shape
    torch.Size([2, 3, 16, 16])
    """

    # Class-level type annotation so static type checkers resolve the subscript
    # operations on _patch_shape (set in __init__ via register_buffer).
    _patch_shape: Tensor

    def __init__(
        self,
        model: Module,
        global_spatial_shape: Tuple[int, int],
        positional_embedding: Literal["learnable", "sinusoidal", "linear"]
        | None = None,
        channels_positional_embedding: int = 4,
        condition_patch: bool | Dict[str, bool] = False,
        condition_interp: bool | Dict[str, bool] = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.global_spatial_shape = tuple(global_spatial_shape)
        self._patching: RandomPatching2D | GridPatching2D | None = None
        self._fuse: bool = False
        self._skip_positional_embedding_injection: bool = False
        # Persistent buffer so that patch_shape survives checkpoint save/load.
        # Zeros sentinel means "not yet configured".
        self.register_buffer(
            "_patch_shape", torch.zeros(2, dtype=torch.long), persistent=True
        )
        # Normalise condition flags to defaultdict for uniform access
        if not isinstance(condition_patch, (bool, dict)):
            raise TypeError(
                f"condition_patch must be bool or Dict[str, bool], "
                f"got {type(condition_patch).__name__}."
            )
        if not isinstance(condition_interp, (bool, dict)):
            raise TypeError(
                f"condition_interp must be bool or Dict[str, bool], "
                f"got {type(condition_interp).__name__}."
            )
        self._condition_has_per_key_flags = isinstance(
            condition_patch, dict
        ) or isinstance(condition_interp, dict)
        self._condition_patch: defaultdict[str, bool] = (
            defaultdict(lambda: condition_patch)
            if isinstance(condition_patch, bool)
            else defaultdict(lambda: False, condition_patch)
        )
        self._condition_interp: defaultdict[str, bool] = (
            defaultdict(lambda: condition_interp)
            if isinstance(condition_interp, bool)
            else defaultdict(lambda: False, condition_interp)
        )

        # Positional embedding
        if positional_embedding is not None:
            H, W = self.global_spatial_shape
            C = channels_positional_embedding
            if positional_embedding == "learnable":
                self.pos_embd = torch.nn.Parameter(torch.randn(C, H, W))
            elif positional_embedding == "linear":
                if C != 2:
                    raise ValueError(
                        "channels_positional_embedding must be 2 for "
                        "'linear' positional embedding."
                    )
                gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
                grid = torch.from_numpy(np.stack([gy, gx], axis=0)).float()
                self.register_buffer("pos_embd", grid)
            elif positional_embedding == "sinusoidal":
                if C % 4 != 0:
                    raise ValueError(
                        "channels_positional_embedding must be a multiple "
                        "of 4 for 'sinusoidal' positional embedding."
                    )
                num_freq = C // 4
                freq_bands = 2.0 ** np.arange(num_freq)
                gx, gy = np.meshgrid(
                    np.linspace(0, 2 * np.pi, W),
                    np.linspace(0, 2 * np.pi, H),
                )
                grids = []
                for freq in freq_bands:
                    for fn in [np.sin, np.cos]:
                        grids.append(fn(gx * freq))
                        grids.append(fn(gy * freq))
                grid = torch.from_numpy(np.stack(grids, axis=0)).float()
                self.register_buffer("pos_embd", grid, persistent=False)
            else:
                raise ValueError(
                    f"positional_embedding must be 'learnable', "
                    f"'sinusoidal', 'linear', or None, "
                    f"got '{positional_embedding}'."
                )
        else:
            self.pos_embd = None

    # ------------------------------------------------------------------
    # Properties for condition pre-processing flags
    # ------------------------------------------------------------------
    @property
    def condition_patch(self) -> bool | Dict[str, bool]:
        """Whether conditioning tensors are patched."""
        return self._condition_patch

    @property
    def condition_interp(self) -> bool | Dict[str, bool]:
        """Whether conditioning tensors are interpolated to patch resolution."""
        return self._condition_interp

    @property
    def patch_shape(self) -> tuple[int, int] | None:
        r"""Spatial shape :math:`(H_p, W_p)` of each patch, or ``None`` if no
        patching strategy has been configured yet.

        The value is read from the live patching object when available, and
        falls back to the persistent checkpoint buffer when the model was
        loaded from a checkpoint but ``set_grid_patching`` / ``set_random_patching``
        have not been called yet.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
        >>> class M(Module):
        ...     def __init__(self): super().__init__(); self.net = torch.nn.Conv2d(3,3,1)
        ...     def forward(self, x, t, condition=None): return self.net(x)
        >>> md = MultiDiffusionModel2D(M(), global_spatial_shape=(16, 16))
        >>> md.patch_shape is None
        True
        >>> md.set_grid_patching(patch_shape=(8, 8))
        >>> md.patch_shape
        (8, 8)
        """
        patching = self._patching
        if patching is not None:
            return patching.patch_shape
        ps = self._patch_shape
        if int(ps[0]) > 0 or int(ps[1]) > 0:
            return (int(ps[0]), int(ps[1]))
        return None

    # ------------------------------------------------------------------
    # Patching strategy configuration
    # ------------------------------------------------------------------

    def set_random_patching(
        self,
        patch_shape: Tuple[int, int],
        patch_num: int,
    ) -> None:
        r"""Configure random patching.

        After calling this method, the forward pass decomposes each input
        sample into ``patch_num`` randomly placed patches of size
        ``patch_shape``, expanding the batch dimension from :math:`B` to
        :math:`P \times B`. This is typically used during training. Random
        patches cannot be fused back to the global resolution.

        To re-draw random patch positions between training
        steps, call :meth:`reset_patch_indices`.

        Parameters
        ----------
        patch_shape : Tuple[int, int]
            Height and width :math:`(H_p, W_p)` of each patch.
        patch_num : int
            Number of patches :math:`P` to extract per sample.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
        >>> class M(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = torch.nn.Conv2d(3, 3, 1)
        ...     def forward(self, x, t, condition=None):
        ...         return self.net(x)
        >>> md = MultiDiffusionModel2D(M(), global_spatial_shape=(16, 16))
        >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
        >>> md(torch.randn(2, 3, 16, 16), torch.rand(2)).shape
        torch.Size([8, 3, 8, 8])
        >>> md.reset_patch_indices()  # re-draw positions for next step
        >>> md(torch.randn(2, 3, 16, 16), torch.rand(2)).shape
        torch.Size([8, 3, 8, 8])
        """
        self._patching = RandomPatching2D(
            img_shape=self.global_spatial_shape,
            patch_shape=patch_shape,
            patch_num=patch_num,
        )
        self._fuse = False
        self._patch_shape[0] = patch_shape[0]
        self._patch_shape[1] = patch_shape[1]

    def reset_patch_indices(self) -> None:
        r"""Re-draw random patch positions for the current random patching
        strategy.

        Raises
        ------
        RuntimeError
            If the current patching strategy is not random patching.
        """
        if not isinstance(self._patching, RandomPatching2D):
            raise RuntimeError(
                "reset_patch_indices() is only available when random "
                "patching is active. Call set_random_patching() first."
            )
        self._patching.reset_patch_indices()

    def set_grid_patching(
        self,
        patch_shape: Tuple[int, int],
        overlap_pix: int = 0,
        boundary_pix: int = 0,
        fuse: bool = True,
    ) -> None:
        r"""Configure deterministic grid patching. Typically used for sampling.

        The global domain is tiled with a regular grid of patches. When
        ``fuse=True``, the per-patch outputs are stitched back into a
        full-resolution image at the end of each forward pass (overlapping
        regions are averaged).

        Parameters
        ----------
        patch_shape : Tuple[int, int]
            Height and width :math:`(H_p, W_p)` of each patch.
        overlap_pix : int, default=0
            Number of overlapping pixels between adjacent patches.
        boundary_pix : int, default=0
            Number of boundary pixels to pad on each side.
        fuse : bool, default=True
            If ``True``, per-patch outputs are fused back to global
            resolution. Set to ``False`` when you want to model's forward pass
            to return the per-patch outputs, i.e. (P*B, C, Hp, Wp).

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
        >>> class M(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = torch.nn.Conv2d(3, 3, 1)
        ...     def forward(self, x, t, condition=None):
        ...         return self.net(x)
        >>> md = MultiDiffusionModel2D(M(), global_spatial_shape=(16, 16))
        >>> md.set_grid_patching(patch_shape=(8, 8), overlap_pix=2, fuse=True)
        >>> md(torch.randn(2, 3, 16, 16), torch.rand(2)).shape
        torch.Size([2, 3, 16, 16])
        """
        self._patching = GridPatching2D(
            img_shape=self.global_spatial_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        self._fuse = fuse
        self._patch_shape[0] = patch_shape[0]
        self._patch_shape[1] = patch_shape[1]

    # ------------------------------------------------------------------
    # Public patching utilities
    # ------------------------------------------------------------------

    def patch_x(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Convert a global spatial 2D tensor to patch-compatible format.

        Decomposes ``x`` into :math:`P` tiles according to the active
        patching strategy. Can be used on any 4D tensor that shares the
        same global spatial layout as the global state (e.g., ground-truth data
        or weight masks).

        Parameters
        ----------
        x : Tensor
            Tensor of shape :math:`(B, C, H, W)`.

        Returns
        -------
        Tensor
            Patched tensor of shape :math:`(P \times B, C, H_p, W_p)`.

        Raises
        ------
        RuntimeError
            If no patching strategy has been configured.
        """
        patching = self._patching
        if patching is None:
            raise RuntimeError(
                "No patching strategy set. Call set_random_patching() "
                "or set_grid_patching() first."
            )
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"patch_x expects a 4D tensor (B, C, H, W), got {x.ndim}D."
                )
            if tuple(x.shape[2:]) != self.global_spatial_shape:
                raise ValueError(
                    f"Spatial dimensions {tuple(x.shape[2:])} do not match "
                    f"global_spatial_shape {self.global_spatial_shape}."
                )
        return patching.apply(x)

    def patch_t(self, t: Float[Tensor, " B"]) -> Float[Tensor, " P_times_B"]:
        r"""Convert a diffusion-time tensor to patch-compatible format.

        Repeats ``t`` :math:`P` times along the batch dimension so that
        each patch receives the same diffusion time as its parent sample.

        Parameters
        ----------
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Expanded tensor of shape :math:`(P \times B,)`.

        Raises
        ------
        RuntimeError
            If no patching strategy has been configured.
        """
        patching = self._patching
        if patching is None:
            raise RuntimeError(
                "No patching strategy set. Call set_random_patching() "
                "or set_grid_patching() first."
            )
        return t.repeat(patching.patch_num)

    def patch_condition(
        self,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None,
    ) -> Tensor | TensorDict | None:
        r"""Convert the condition to patch-compatible format.

        Each tensor in the condition is pre-processed according to the
        strategy set by the arguments ``condition_patch`` and
        ``condition_interp``: patched, interpolated, or simply repeated along
        the batch dimension (default).

        Positional embeddings are **not** injected by this method; they are
        handled internally by :meth:`forward`.

        Parameters
        ----------
        condition : Tensor, TensorDict, or None
            Conditioning information at global resolution (batch size
            :math:`B`).

        Returns
        -------
        Tensor, TensorDict, or None
            Condition in patch-compatible format (batch size
            :math:`P \times B`), or ``None`` if the input is ``None``.

        Raises
        ------
        RuntimeError
            If no patching strategy has been configured.

        Examples
        --------
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
        >>> class M(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = torch.nn.Conv2d(3, 3, 1)
        ...     def forward(self, x, t, condition=None):
        ...         return self.net(x)
        >>> md = MultiDiffusionModel2D(
        ...     M(), global_spatial_shape=(16, 16),
        ...     condition_patch={"img": True},
        ... )
        >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
        >>> cond = TensorDict({
        ...     "img": torch.randn(2, 3, 16, 16),
        ...     "vec": torch.randn(2, 5),
        ... }, batch_size=[2])
        >>> cp = md.patch_condition(cond)
        >>> cp["img"].shape  # patched: (P*B, C, Hp, Wp)
        torch.Size([8, 3, 8, 8])
        >>> cp["vec"].shape  # default: repeated P times
        torch.Size([8, 5])
        """
        patching = self._patching
        if patching is None:
            raise RuntimeError(
                "No patching strategy set. Call set_random_patching() "
                "or set_grid_patching() first."
            )
        if condition is None:
            return None

        P = patching.patch_num

        if isinstance(condition, Tensor):
            if self._condition_has_per_key_flags:
                raise ValueError(
                    "condition_patch and condition_interp must be bool (not "
                    "dict) when condition is a plain Tensor. Use a TensorDict "
                    "for per-key control."
                )
            do_patch = self._condition_patch[""]
            do_interp = self._condition_interp[""]
            if not torch.compiler.is_compiling():
                if do_patch and condition.ndim == 4:
                    if tuple(condition.shape[2:]) != self.global_spatial_shape:
                        raise ValueError(
                            f"condition_patch=True requires spatial dims "
                            f"{tuple(condition.shape[2:])} to match "
                            f"global_spatial_shape "
                            f"{self.global_spatial_shape}."
                        )
            return self._process_condition_tensor(
                condition, do_patch=do_patch, do_interp=do_interp, P=P
            )

        if isinstance(condition, TensorDict):
            B = condition.batch_size[0]
            if not torch.compiler.is_compiling():
                cond_keys = set(condition.keys())
                extra = set(self._condition_patch) - cond_keys
                if extra:
                    raise ValueError(
                        f"condition_patch has keys {extra} not present "
                        f"in condition. Available keys: {cond_keys}."
                    )
                extra = set(self._condition_interp) - cond_keys
                if extra:
                    raise ValueError(
                        f"condition_interp has keys {extra} not present "
                        f"in condition. Available keys: {cond_keys}."
                    )
            result = {}
            for key in condition.keys():
                do_patch = self._condition_patch[key]
                do_interp = self._condition_interp[key]
                tensor = condition[key]
                if not torch.compiler.is_compiling():
                    if do_patch and tensor.ndim == 4:
                        if tuple(tensor.shape[2:]) != self.global_spatial_shape:
                            raise ValueError(
                                f"condition_patch=True for key '{key}' "
                                f"requires spatial dims "
                                f"{tuple(tensor.shape[2:])} to match "
                                f"global_spatial_shape "
                                f"{self.global_spatial_shape}."
                            )
                result[key] = self._process_condition_tensor(
                    tensor, do_patch, do_interp, P
                )
            return TensorDict(result, batch_size=[P * B])

        raise TypeError(
            f"condition must be Tensor, TensorDict, or None, "
            f"got {type(condition).__name__}."
        )

    def fuse(
        self,
        input: Float[Tensor, "P_times_B C Hp Wp"],
        batch_size: int,
    ) -> Float[Tensor, "B C H W"]:
        r"""Fuse patches back into a full-resolution image.

        Only supported when :meth:`set_grid_patching` has been called.
        Random patches cannot be fused because their positions may overlap
        arbitrarily.

        Parameters
        ----------
        input : Tensor
            Patched tensor of shape :math:`(P \times B, C, H_p, W_p)`.
        batch_size : int
            Original batch size :math:`B` before patching.

        Returns
        -------
        Tensor
            Fused tensor of shape :math:`(B, C, H, W)`.

        Raises
        ------
        RuntimeError
            If no patching strategy is set, or if the current strategy is
            random patching.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
        >>> class M(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = torch.nn.Conv2d(3, 3, 1)
        ...     def forward(self, x, t, condition=None):
        ...         return self.net(x)
        >>> md = MultiDiffusionModel2D(M(), global_spatial_shape=(16, 16))
        >>> md.set_grid_patching(patch_shape=(8, 8))
        >>> x = torch.randn(2, 3, 16, 16)
        >>> x_patched = md.patch_x(x)
        >>> torch.allclose(md.fuse(x_patched, batch_size=2), x)
        True
        """
        if not isinstance(self._patching, GridPatching2D):
            raise RuntimeError(
                "Fusing is only supported with grid patching. "
                "Call set_grid_patching() first."
            )
        return self._patching.fuse(input, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Float[Tensor, "B C H W"] | Float[Tensor, "P_times_B C Hp Wp"],
        t: Float[Tensor, " B"] | Float[Tensor, " P_times_B"],
        condition: Float[Tensor, " B *cond_dims"]
        | Float[Tensor, " P_times_B *cond_dims"]
        | TensorDict
        | None = None,
        x_is_patched: bool = False,
        t_is_patched: bool = False,
        condition_is_patched: bool = False,
        **model_kwargs: Any,
    ) -> Float[Tensor, "P_times_B C Hp Wp"] | Float[Tensor, "B C H W"]:
        # No patching strategy: warn and pass through
        patching = self._patching
        if patching is None:
            if not torch.compiler.is_compiling():
                warnings.warn(
                    "No patching strategy set on MultiDiffusionModel2D. "
                    "The model will run without patching.",
                    stacklevel=2,
                )
            if (
                self.pos_embd is not None
                and not self._skip_positional_embedding_injection
            ):
                B = x.shape[0]
                # .expand creates a stride-0 view that can trip up downstream
                # torch ops (e.g. nn.ReflectionPad2d / F.unfold on torch 2.10).
                # Materialise a contiguous copy before handing it off.
                pos_embd = self.pos_embd.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
                condition = self._inject_patched_pos_embd(condition, pos_embd, B)
            return self.model(x, t, condition=condition, **model_kwargs)

        P = patching.patch_num

        # B is only consumed by PE injection and fusing. When neither runs
        # (e.g., MultiDiffusionPredictor calls into this method with both
        # disabled to stream partial chunks), B is unused — skip computing
        # and validating it so partial (K, C, Hp, Wp) tensors with K < P
        # can be passed through.
        _b_consumed = (
            self.pos_embd is not None and not self._skip_positional_embedding_injection
        ) or self._fuse
        if x_is_patched:
            if (
                _b_consumed
                and not torch.compiler.is_compiling()
                and x.shape[0] % P != 0
            ):
                raise ValueError(
                    f"x_is_patched=True but x batch dim ({x.shape[0]}) is "
                    f"not divisible by patch_num ({P})."
                )
            B = x.shape[0] // P if _b_consumed else 0
        else:
            B = x.shape[0]

        # Convert each input to patch-compatible format
        if not x_is_patched:
            x = self.patch_x(x)
        if not t_is_patched:
            t = self.patch_t(t)
        if not condition_is_patched:
            condition = self.patch_condition(condition)

        # Positional embeddings injected here unless _skip_positional_embedding_injection
        # is set (e.g. by MultiDiffusionPredictor which pre-patches PE at construction time)
        if self.pos_embd is not None and not self._skip_positional_embedding_injection:
            # .expand creates a stride-0 view that can trip up downstream
            # torch ops (e.g. nn.ReflectionPad2d / F.unfold on torch 2.10).
            # Materialise a contiguous copy before passing to patching.
            pos_embd_patched = patching.apply(
                self.pos_embd.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            )  # (P*B, C_PE, Hp, Wp)
            condition = self._inject_patched_pos_embd(
                condition, pos_embd_patched, P * B
            )

        output = self.model(x, t, condition=condition, **model_kwargs)

        if self._fuse:
            output = patching.fuse(output, batch_size=B)

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_condition_tensor(
        self,
        tensor: Tensor,
        do_patch: bool,
        do_interp: bool,
        P: int,
    ) -> Tensor:
        """Apply the appropriate pre-processing to a single condition tensor.

        Exactly one of ``do_patch`` / ``do_interp`` may be ``True``, or both
        ``False`` (default: repeat along batch dimension).
        """
        if do_patch and do_interp:
            raise ValueError(
                "condition_patch and condition_interp cannot both be True "
                "for the same condition key. Use one or the other."
            )

        if (do_patch or do_interp) and tensor.ndim != 4:
            raise ValueError(
                f"condition_patch={do_patch} or "
                f"condition_interp={do_interp} requires a 4D tensor "
                f"(B, C, H, W), got {tensor.ndim}D."
            )

        # Default case: no patching needed, just repeat along the batch dim.
        if not do_patch and not do_interp:
            return tensor.repeat(P, *([1] * (tensor.ndim - 1)))

        # Both patch and interp need an active patching strategy.
        patching = self._patching
        if patching is None:
            raise RuntimeError(
                "No patching strategy set. Call set_random_patching() "
                "or set_grid_patching() first."
            )

        if do_patch:
            return patching.apply(tensor)

        # do_interp case
        Hp, Wp = patching.patch_shape
        tensor = F.interpolate(tensor, size=(Hp, Wp), mode="bilinear")
        return tensor.repeat(P, 1, 1, 1)

    def _inject_patched_pos_embd(
        self,
        condition: Tensor | TensorDict | None,
        pos_embd_patched: Float[Tensor, "P_times_B C_PE Hp Wp"],
        PB: int,
    ) -> TensorDict:
        """Inject an already-patched positional embedding into the (possibly
        already-patched) condition under the ``"positional_embedding"`` key.

        Common logic factored out of :meth:`forward` so it can be reused by
        :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionPredictor`.
        When ``condition`` is a ``TensorDict`` it is mutated in place for
        efficiency; otherwise a new ``TensorDict`` is built.
        """
        if condition is None:
            return TensorDict(
                {"positional_embedding": pos_embd_patched},
                batch_size=[PB],
            )
        if isinstance(condition, TensorDict):
            condition["positional_embedding"] = pos_embd_patched
            return condition
        if isinstance(condition, Tensor):
            return TensorDict(
                {"condition": condition, "positional_embedding": pos_embd_patched},
                batch_size=[PB],
            )
        raise ValueError(
            "When positional embeddings are configured, condition must be a "
            f"Tensor, TensorDict, or None, got {type(condition).__name__}."
        )
