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

import warnings
from typing import Any, Callable, Iterator, cast

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from physicsnemo.diffusion.base import Predictor, PredictorType
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
    and all standard solvers) with no other changes. All patch-based logic
    (patching the state, running the per-patch predictions, fusing them back
    to the global domain) is handled internally.

    The patching strategy must be configured by calling :meth:`set_patching`
    before any other method. ``patch_shape`` and ``global_shape`` default
    to the values saved on the wrapped model (typically restored from the
    training-time checkpoint), while ``overlap_pix`` and ``boundary_pix``
    must be provided explicitly because they cannot be inferred from a
    model trained with random patching.

    On very large global domains the full :math:`P \times B` activation
    tensor may not fit in GPU memory, leading to OOM errors. Two
    independent strategies mitigate this:

    - ``chunk_size`` processes the :math:`P \times B` patches in
      consecutive chunks of at most ``chunk_size`` rows. Useful for
      both plain inference and gradient-based use cases. Trades batch
      parallelism for memory.
    - ``use_checkpointing`` recomputes activations on demand during
      backpropagation instead of storing them. Only meaningful when
      gradients flow through the predictor, the typical use case being
      DPS guidance (see
      :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionDPSScorePredictor`).
      Trades compute for memory.

    The two options can be combined; for use cases that need explicit
    control over chunk-level processing, the streaming iterator
    :meth:`chunks` exposes the per-chunk model outputs directly.

    .. warning::

        :class:`MultiDiffusionPredictor` is intended for **test-time
        sampling**: it is not suitable for training. The wrapped
        multi-diffusion model should already be trained before being passed
        to the predictor.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        A trained multi-diffusion model. The grid patching configuration
        must be supplied through :meth:`set_patching` after construction.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Conditioning at the global resolution, bound once at construction
        and reused at every diffusion step. Shape :math:`(B, *cond\_dims)`.
        Pass ``None`` for unconditional models.
    fuse : bool, default=True
        Whether to fuse per-patch outputs back to the global resolution
        before returning.
    chunk_size : int or None, default=None
        Number of patch rows along the :math:`P \times B` dimension
        processed per model call. ``None`` runs all patches in a single
        call. Set to a small integer to reduce peak GPU memory when the
        full :math:`P \times B` activation tensor does not fit at once.
    use_checkpointing : bool, default=False
        Trade compute for memory: activations are recomputed on demand
        during backpropagation instead of being stored from the forward
        pass. Useful when differentiating through the predictor on large
        domains. Works with or without ``chunk_size``.
    prediction_type : PredictorType, default="x0"
        Output type of the wrapped model. One of ``"x0"``, ``"score"``, or
        ``"epsilon"``. The predictor always exposes an x0-compatible
        output; pass the appropriate conversion function below when the
        model does not directly predict x0.
    score_to_x0_fn : callable, optional
        Conversion ``(score, x_t, t) -> x0`` applied to the model output.
        Required when ``prediction_type="score"``. Typically obtained from
        a noise scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.score_to_x0`.
    epsilon_to_x0_fn : callable, optional
        Conversion ``(epsilon, x_t, t) -> x0`` applied to the model output.
        Required when ``prediction_type="epsilon"``. Typically obtained from
        a noise scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.epsilon_to_x0`.
    **model_kwargs : Any
        Additional keyword arguments bound once at construction and
        forwarded to the wrapped model at every call.

    See Also
    --------
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D` :
        The multi-diffusion wrapper used for training.
    :class:`~physicsnemo.diffusion.Predictor` :
        The protocol this class implements.
    :func:`~physicsnemo.diffusion.samplers.sample` :
        The main sampling entry point.
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionDPSScorePredictor` :
        Patch-local DPS score predictor that consumes
        :meth:`chunks` for memory-efficient guided sampling.

    Examples
    --------
    **Example 1:** Predictor in isolation. Input and output live at the
    global resolution; patching, per-patch prediction, and fusing are all
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
    >>> # Train and save the model (training omitted here)
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)  # training config
    >>> _ = md.eval()
    >>>
    >>> predictor = MultiDiffusionPredictor(md)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)  # P = 4 patches per sample
    >>> x = torch.randn(2, 3, 16, 16)  # global-resolution state
    >>> t = 0.5 * torch.ones(2)
    >>> predictor(x, t).shape  # fused output at global resolution
    torch.Size([2, 3, 16, 16])
    >>>
    >>> predictor.fuse = False  # raw per-patch predictions instead
    >>> predictor(x, t).shape
    torch.Size([8, 3, 8, 8])

    **Example 2:** Unconditional sampling. The predictor plugs straight into
    the standard diffusion sampling stack (noise scheduler, denoiser,
    solver):

    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>>
    >>> predictor = MultiDiffusionPredictor(md)
    >>> predictor.set_patching(overlap_pix=2, boundary_pix=0)
    >>> scheduler = EDMNoiseScheduler()
    >>> denoiser = scheduler.get_denoiser(x0_predictor=predictor)
    >>> xN = torch.randn(2, 3, 16, 16)  # initial noise at global resolution
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])

    **Example 3:** Conditional sampling with mixed conditioning, an image
    sharing the spatial resolution (patched like the state) and a vector
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
    ... )
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>>
    >>> condition = TensorDict({
    ...     "image":  torch.randn(2, 3, 16, 16),
    ...     "vector": torch.randn(2, 5),
    ... }, batch_size=[2])
    >>> predictor = MultiDiffusionPredictor(md, condition=condition)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>> denoiser = scheduler.get_denoiser(x0_predictor=predictor)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])

    **Example 4:** Memory-efficient inference on large domains.
    ``chunk_size`` and ``use_checkpointing`` are independent and address
    different bottlenecks. ``chunk_size`` is useful for any kind of
    inference (with or without gradients) and reduces peak memory by
    sacrificing some batch parallelism. ``use_checkpointing`` is only
    meaningful when gradients flow through the predictor (see
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionDPSScorePredictor`
    for the DPS guidance use case) and trades compute for memory by
    replaying the forward during backpropagation.

    Set ``chunk_size`` to process patches in chunks instead of all at once
    (helpful for plain inference on a domain that would otherwise OOM):

    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>> x = torch.randn(3, 3, 16, 16)                       # B = 3
    >>> t = 0.5 * torch.ones(3)
    >>> predictor(x, t).shape                                # fused (B, C, H, W)
    torch.Size([3, 3, 16, 16])

    Set ``use_checkpointing=True`` (independent of chunking) when
    differentiating through the predictor on a large domain:

    >>> predictor = MultiDiffusionPredictor(md, use_checkpointing=True)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>> x = torch.randn(3, 3, 16, 16, requires_grad=True)
    >>> y = predictor(x, t)
    >>> grad = torch.autograd.grad(y.sum(), x)[0]
    >>> grad.shape
    torch.Size([3, 3, 16, 16])

    Combine both for differentiable inference on very large domains.
    A typical use case is computing the gradient of a DPS guidance
    likelihood through the predictor (see
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionDPSScorePredictor`):

    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2, use_checkpointing=True)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>> x = torch.randn(3, 3, 16, 16, requires_grad=True)
    >>> y = predictor(x, t)
    >>> grad = torch.autograd.grad(y.sum(), x)[0]
    >>> grad.shape
    torch.Size([3, 3, 16, 16])

    Use :meth:`chunks` for explicit per-chunk control. The iterator yields
    ``(slice_start, x0_chunk, x_chunk, t_chunk)`` tuples; the caller is
    responsible for fusing via :meth:`fuse_fn` after the loop. This is
    functionally equivalent to ``__call__`` with ``chunk_size`` set, but
    exposes the intermediate per-chunk values so callers can interleave
    their own per-chunk processing (e.g. accumulating patch-local guidance
    terms):

    >>> predictor = MultiDiffusionPredictor(md, chunk_size=4, use_checkpointing=True)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>> x = torch.randn(3, 3, 16, 16, requires_grad=True)   # B = 3, P*B = 12
    >>> t = 0.5 * torch.ones(3)
    >>> outs = []
    >>> for s, x0_c, x_c, t_c in predictor.chunks(x, t):
    ...     outs.append(x0_c)                               # (chunk_size=4, C, Hp, Wp)
    >>> x0_patched = torch.cat(outs, dim=0)                 # (P*B=12, C, Hp, Wp)
    >>> x0_global = predictor.fuse_fn(x0_patched)           # (B=3, C, H, W)
    >>> grad = torch.autograd.grad(x0_global.sum(), x)[0]
    >>> grad.shape
    torch.Size([3, 3, 16, 16])
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        fuse: bool = True,
        chunk_size: int | None = None,
        use_checkpointing: bool = False,
        prediction_type: PredictorType = "x0",
        score_to_x0_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B *dims"], Float[Tensor, " B"]],
            Float[Tensor, " B *dims"],
        ]
        | None = None,
        epsilon_to_x0_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B *dims"], Float[Tensor, " B"]],
            Float[Tensor, " B *dims"],
        ]
        | None = None,
        **model_kwargs: Any,
    ) -> None:
        self._md_model: MultiDiffusionModel2D = _unwrap_module(
            model, MultiDiffusionModel2D
        )
        self.model = model
        self._model_kwargs = model_kwargs
        self._chunk_size = chunk_size
        self._use_checkpointing = use_checkpointing
        self._fuse: bool = fuse
        self._cond_input = condition

        # Predictor-owned patching parameters; default to the values saved on
        # the wrapped model. set_patching() overrides them at call time and
        # warns when the override differs from the saved value.
        self._patch_shape: tuple[int, int] | None = self._md_model.patch_shape
        self._global_shape: tuple[int, int] = tuple(self._md_model.global_spatial_shape)

        # Caches populated by set_patching(); guarded against use before then.
        self._patching: GridPatching2D | None = None
        self._P: int | None = None
        self._cond_patched: Tensor | TensorDict | None = None
        self._pos_embd_patched: Tensor | None = None

        # PE injection is handled by this class from a pre-patched cache;
        # suppress the wrapper's per-step PE injection.
        self._md_model._skip_positional_embedding_injection = True
        # Internal model fusing is handled externally by this predictor (via
        # fuse_fn); keep it disabled so gradient checkpointing replays observe
        # a stable model state across forward and backward passes.
        self._md_model._fuse = False

        # Prediction-type conversion (same pattern as MultiDiffusionMSEDSMLoss).
        match prediction_type:
            case "x0":
                self._to_x0: Callable[[Tensor, Tensor, Tensor], Tensor] = (
                    lambda pred, _x, _t: pred
                )
            case "score":
                if score_to_x0_fn is None:
                    raise ValueError(
                        "score_to_x0_fn must be provided when prediction_type='score'."
                    )
                self._to_x0 = score_to_x0_fn
            case "epsilon":
                if epsilon_to_x0_fn is None:
                    raise ValueError(
                        "epsilon_to_x0_fn must be provided when prediction_type='epsilon'."
                    )
                self._to_x0 = epsilon_to_x0_fn
            case _:
                raise ValueError(
                    f"prediction_type must be 'x0', 'score', or 'epsilon', "
                    f"got '{prediction_type}'."
                )

        # Bind the model-call helper once so the use_checkpointing branch is
        # resolved at construction rather than on every forward call.
        self._call_model: Callable[[Tensor, Tensor, Any], Tensor] = (
            self._call_model_with_checkpoint
            if self._use_checkpointing
            else self._call_model_direct
        )

    @property
    def fuse(self) -> bool:
        """Whether per-patch outputs are fused back to the global resolution
        before being returned."""
        return self._fuse

    @fuse.setter
    def fuse(self, value: bool) -> None:
        """Set whether per-patch outputs are fused before being returned."""
        self._fuse = value

    @property
    def patch_shape(self) -> tuple[int, int] | None:
        r"""Spatial shape :math:`(H_p, W_p)` of each patch."""
        return self._md_model.patch_shape

    def patch_fn(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Patch a global-resolution spatial tensor.

        Forwards to
        :meth:`physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D.patch_x`
        on the wrapped model. Useful for pre-patching auxiliary tensors
        (observations, masks) outside the predictor — for example in the
        constructor of a DPS guidance — so that all patching uses the same
        grid configuration as the predictor.

        Parameters
        ----------
        x : Tensor
            Global-resolution tensor of shape :math:`(B, C, H, W)`.

        Returns
        -------
        Tensor
            Patched tensor of shape :math:`(P \times B, C, H_p, W_p)`.
        """
        self._check_patching_set()
        return self._md_model.patch_x(x)

    def fuse_fn(
        self,
        patched: Float[Tensor, "P_times_B C Hp Wp"],
    ) -> Float[Tensor, "B C H W"]:
        r"""Fuse a complete patched tensor back to the global resolution.

        General-purpose fusing utility: takes a patched tensor of shape
        :math:`(P \times B, C, H_p, W_p)` and returns the corresponding
        global-resolution tensor :math:`(B, C, H, W)`. The original batch
        size :math:`B` is inferred as ``patched.shape[0] // patch_num``;
        the input must therefore contain the full :math:`P \times B` rows.

        Parameters
        ----------
        patched : Tensor
            Full patched tensor of shape :math:`(P \times B, C, H_p, W_p)`.

        Returns
        -------
        Tensor
            Fused tensor of shape :math:`(B, C, H, W)`.

        Raises
        ------
        ValueError
            If ``patched.shape[0]`` is not divisible by ``patch_num``.
        """
        self._check_patching_set()
        P = cast(int, self._P)
        if not torch.compiler.is_compiling() and patched.shape[0] % P != 0:
            raise ValueError(
                f"patched.shape[0] ({patched.shape[0]}) is not divisible by "
                f"patch_num ({P}); fuse_fn requires the full "
                f"(P*B, …) tensor."
            )
        B = patched.shape[0] // P
        return self._md_model.fuse(patched, batch_size=B)

    def set_patching(
        self,
        overlap_pix: int,
        boundary_pix: int,
        *,
        patch_shape: tuple[int, int] | None = None,
        global_shape: tuple[int, int] | None = None,
    ) -> None:
        r"""Set the grid patching configuration.

        Must be called once after construction and before any other method
        (:meth:`__call__`, :meth:`chunks`, :meth:`patch_fn`, :meth:`fuse_fn`,
        ...). Calling it again reconfigures the patching and rebuilds the
        internal pre-patched caches.

        ``overlap_pix`` and ``boundary_pix`` are required: they cannot be
        inferred from the wrapped model (which is typically trained with a
        random patching strategy that has neither concept).
        ``patch_shape`` and ``global_shape`` default to the values saved in
        the wrapped model. Overriding either emits a warning, since
        mismatching the training-time geometry can produce unexpected
        results (in particular, positional embeddings baked at model
        construction will no longer match a different ``global_shape``).

        Parameters
        ----------
        overlap_pix : int
            Overlapping pixels between adjacent patches.
        boundary_pix : int
            Boundary pixels padded on each side.
        patch_shape : tuple[int, int], optional
            Override for the patch spatial shape :math:`(H_p, W_p)`. Default
            to the value saved on the wrapped model.
        global_shape : tuple[int, int], optional
            Override for the global spatial shape :math:`(H, W)`. Default to
            the value saved on the wrapped model.
        """
        if patch_shape is not None:
            new_ps = tuple(patch_shape)
            if self._patch_shape is not None and new_ps != self._patch_shape:
                warnings.warn(
                    f"Overriding saved patch_shape {self._patch_shape} with "
                    f"{new_ps}. Inference-time patching that differs from "
                    f"training may produce unexpected results.",
                    stacklevel=2,
                )
            self._patch_shape = new_ps

        if global_shape is not None:
            new_gs = tuple(global_shape)
            if new_gs != self._global_shape:
                warnings.warn(
                    f"Overriding saved global_spatial_shape "
                    f"{self._global_shape} with {new_gs}. Positional "
                    f"embeddings baked at model construction will no "
                    f"longer match the new shape.",
                    stacklevel=2,
                )
            self._global_shape = new_gs

        if self._patch_shape is None:
            raise RuntimeError(
                "patch_shape is not available on the wrapped model and was "
                "not provided. Pass patch_shape=(Hp, Wp) explicitly."
            )

        # Sync the underlying model's global_spatial_shape so its patching
        # logic uses the same geometry as the predictor.
        self._md_model.global_spatial_shape = self._global_shape
        self._md_model.set_grid_patching(
            patch_shape=self._patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
            fuse=False,
        )
        self._patching = cast(GridPatching2D, self._md_model._patching)
        self._P = self._patching.patch_num

        # Rebuild pre-patched condition / PE caches now that patching is set.
        self._cond_patched = self._md_model.patch_condition(self._cond_input)
        if self._md_model.pos_embd is not None:
            self._pos_embd_patched = self._md_model.patch_x(
                self._md_model.pos_embd.unsqueeze(0)
            )  # (P, C_PE, Hp, Wp)
        else:
            self._pos_embd_patched = None

    def _build_cond(self, B: int) -> Tensor | TensorDict | None:
        # Expand the pre-patched condition to batch size B and inject PE.
        cond = self._cond_patched
        if self._pos_embd_patched is not None:
            P = cast(int, self._P)
            pe = self._pos_embd_patched.repeat_interleave(B, dim=0)
            cond = self._md_model._inject_patched_pos_embd(cond, pe, P * B)
        return cond

    def chunks(
        self,
        x: Float[Tensor, "B C H W"],
        t: Float[Tensor, " B"],
    ) -> Iterator[
        tuple[
            int,  # slice_start: row index of x0_chunk along (P*B)
            Float[Tensor, "K C Hp Wp"],  # x0_chunk: model output (converted to x0)
            Float[Tensor, "K C Hp Wp"],  # x_chunk:  noisy input slice
            Float[Tensor, " K"],  # t_chunk:  time slice
        ]
    ]:
        r"""Stream the per-chunk model outputs alongside their inputs.

        Always returns patched outputs and does not fuse them. This makes
        the iterator particularly useful for use cases that need to combine
        the per-chunk model output with auxiliary patched data before
        fusing, such as patch-local DPS guidance (see
        :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionDPSScorePredictor`).
        On large global domains, the equivalent non-chunked call
        :meth:`__call__` may run out of GPU memory; iterating the patches
        in chunks of size ``chunk_size`` keeps the peak activation
        footprint bounded. Use :meth:`fuse_fn` after the loop to fuse the
        concatenated chunks back to the global resolution.

        Functionally equivalent to ``__call__`` with ``chunk_size`` set
        (which streams internally and fuses at the end). Use this iterator
        when explicit per-chunk control is needed; otherwise prefer the
        higher-level ``__call__``.

        Patching is performed once at the start of the iteration; subsequent
        iterations only slice the pre-patched tensors. Requires
        ``chunk_size`` to be set at construction.

        Parameters
        ----------
        x : Tensor
            Noisy latent at global resolution, shape :math:`(B, C, H, W)`.
        t : Tensor
            Diffusion time, shape :math:`(B,)`.

        Returns
        -------
        Iterator yielding tuples ``(slice_start, x0_chunk, x_chunk, t_chunk)``:

        - ``slice_start`` (``int``): row index of the chunk along the
          :math:`(P \times B)` dimension. Allows downstream consumers (e.g.
          patch-local DPS guidances) to align their own pre-patched data
          with the current chunk.
        - ``x0_chunk`` (``Tensor``): model output for this chunk (already
          converted to x0 when a conversion was configured), shape
          :math:`(K, C, H_p, W_p)` with :math:`K \leq chunk\_size`. All
          chunks have ``chunk_size`` rows except possibly the last, which
          may be smaller when :math:`P \times B` is not divisible by
          ``chunk_size``.
        - ``x_chunk`` (``Tensor``): noisy input slice corresponding to
          ``x0_chunk``, same shape.
        - ``t_chunk`` (``Tensor``): time slice corresponding to
          ``x0_chunk``, shape :math:`(K,)`.

        Raises
        ------
        RuntimeError
            If ``chunk_size`` was not set at construction, or if
            :meth:`set_patching` has not been called.

        Examples
        --------
        Streaming inference with a per-chunk computation in the loop body
        (here just collecting a per-chunk statistic, but in practice this
        is where DPS guidance terms or other patch-local processing would
        plug in):

        >>> import torch
        >>> from physicsnemo.core import Module
        >>> from physicsnemo.diffusion.multi_diffusion import (
        ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
        ... )
        >>> class Backbone(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = torch.nn.Conv2d(3, 3, 1)
        ...     def forward(self, x, t, condition=None):
        ...         return self.net(x)
        >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
        >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
        >>> _ = md.eval()
        >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
        >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
        >>> x = torch.randn(2, 3, 16, 16)
        >>> t = 0.5 * torch.ones(2)
        >>> chunks_list, chunk_norms = [], []
        >>> for s, x0_c, x_c, t_c in predictor.chunks(x, t):
        ...     chunk_norms.append(x0_c.pow(2).sum())  # per-chunk computation
        ...     chunks_list.append(x0_c)
        >>> x0_global = predictor.fuse_fn(torch.cat(chunks_list, dim=0))
        >>> x0_global.shape
        torch.Size([2, 3, 16, 16])
        """
        self._check_patching_set()
        if self._chunk_size is None:
            raise RuntimeError(
                "chunk_size must be set at construction to use chunks(). "
                "Pass chunk_size=<int> to MultiDiffusionPredictor.__init__."
            )

        B = x.shape[0]
        x_patched = self._md_model.patch_x(x)  # (P*B, C, Hp, Wp)
        t_patched = self._md_model.patch_t(t)  # (P*B,)
        cond = self._build_cond(B)

        K = self._chunk_size
        PB = x_patched.shape[0]
        for s in range(0, PB, K):
            e = min(s + K, PB)
            x_c = x_patched[s:e]
            t_c = t_patched[s:e]
            c_c = cond[s:e] if cond is not None else None
            out = self._call_model(x_c, t_c, c_c)
            x0_c = self._to_x0(out, x_c, t_c)
            yield s, x0_c, x_c, t_c

    def _check_patching_set(self) -> None:
        """Raise if :meth:`set_patching` has not been called yet.

        Guarded by ``torch.compiler.is_compiling`` so it is a no-op under
        ``torch.compile``.
        """
        if not torch.compiler.is_compiling() and self._patching is None:
            raise RuntimeError(
                "Grid patching is not configured. Call set_patching("
                "overlap_pix, boundary_pix, ...) before any other method."
            )

    def _call_model_direct(
        self,
        x_p: Tensor,
        t_p: Tensor,
        cond: Any,
    ) -> Tensor:
        """Call the wrapped model on the patched inputs, no checkpointing."""
        return self._md_model(
            x_p,
            t_p,
            condition=cond,
            x_is_patched=True,
            t_is_patched=True,
            condition_is_patched=True,
            **self._model_kwargs,
        )

    def _call_model_with_checkpoint(
        self,
        x_p: Tensor,
        t_p: Tensor,
        cond: Any,
    ) -> Tensor:
        """Call the wrapped model under :func:`torch.utils.checkpoint.checkpoint`.

        ``cond`` is captured as a default argument so each invocation binds
        its own value, which is required for correct backward replay when
        called inside a loop.
        """

        def _inner(xc: Tensor, tc: Tensor, _cond=cond) -> Tensor:  # noqa: ANN001
            return self._md_model(
                xc,
                tc,
                condition=_cond,
                x_is_patched=True,
                t_is_patched=True,
                condition_is_patched=True,
                **self._model_kwargs,
            )

        return checkpoint(_inner, x_p, t_p, use_reentrant=False)

    def __call__(
        self,
        x: Float[Tensor, "B C H W"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, "B C H W"] | Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Run the predictor on a noisy latent and diffusion time.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent at global resolution, shape :math:`(B, C, H, W)`.
        t : torch.Tensor
            Diffusion time, shape :math:`(B,)`.

        Returns
        -------
        torch.Tensor
            If ``self.fuse=True``: prediction at the global resolution,
            shape :math:`(B, C, H, W)`.
            Otherwise: per-patch predictions, shape
            :math:`(P \times B, C, H_p, W_p)`.
        """
        self._check_patching_set()
        if self._chunk_size is not None:
            x0_chunks = [x0_c for _, x0_c, _, _ in self.chunks(x, t)]
            output = torch.cat(x0_chunks, dim=0)  # (P*B, C, Hp, Wp)
            return self.fuse_fn(output) if self._fuse else output

        B = x.shape[0]
        x_patched = self._md_model.patch_x(x)  # (P*B, C, Hp, Wp)
        t_patched = self._md_model.patch_t(t)  # (P*B,)
        cond = self._build_cond(B)
        result = self._call_model(x_patched, t_patched, cond)
        result = self._to_x0(result, x_patched, t_patched)
        return self.fuse_fn(result) if self._fuse else result
