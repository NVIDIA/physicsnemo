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

"""Patch-local DPS guidance for multi-diffusion sampling."""

from typing import Callable, Protocol, Sequence, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from physicsnemo.diffusion.base import Predictor
from physicsnemo.diffusion.multi_diffusion.predictor import MultiDiffusionPredictor
from physicsnemo.diffusion.utils.utils import _as_broadcastable


def _lp_loss_fn(
    p: int,
) -> Callable[
    [Float[Tensor, " *shape"], Float[Tensor, " *shape"]], Float[Tensor, " *shape"]
]:
    """
    Return a pointwise (not reduced) Lp residual function with exponent ``p``.
    """

    def _loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return (y_pred - y_true).abs().pow(p)

    return _loss


def _prepatch_param(
    value: float | Float[Tensor, " *#shape"],
    predictor: MultiDiffusionPredictor,
    y: Float[Tensor, "B C H W"],
) -> Float[Tensor, "P_times_B C Hp Wp"]:
    """Pre-patch a ``value`` tensor to the patch layout.

    ``value`` is validated to be a scalar or a tensor broadcastable to the
    global observation ``y``, then broadcast to the global resolution and
    patched. The result lines up with the :math:`(P \\times B, \\dots)` layout
    of the pre-patched observations, so a chunk slice ``[s : s + K]`` selects
    the matching patches.
    """
    value = _as_broadcastable(value, y)
    return predictor.patch_fn(torch.broadcast_to(value, y.shape).contiguous())


@runtime_checkable
class MultiDiffusionDPSGuidance(Protocol):
    r"""Protocol for **patch-local** DPS guidance compatible with
    :class:`MultiDiffusionDPSScorePredictor`.

    A guidance is **patch-local** when its computation decomposes along
    the multi-diffusion patch grid: the guidance value at each patch
    depends only on the data of that patch. This protocol is **not
    applicable** to globally-coupled guidances (e.g. ones that mix
    information across patches), use
    :class:`~physicsnemo.diffusion.guidance.DPSGuidance` for those.

    Identical to the standard
    :class:`~physicsnemo.diffusion.guidance.DPSGuidance` protocol, plus an
    optional ``slice_start`` argument that enables chunked evaluation:

    - **Full batch mode** (``slice_start=None``, the default): the call
      processes the full :math:`P \times B` batch of patches at once.
      Inputs match the size of the pre-patched data stored on the
      guidance. The implementation may optionally fuse the result back to
      the global resolution.
    - **Chunked batch mode** (``slice_start=s``): the call processes a
      single chunk of :math:`K \leq \text{chunk\_size}` patches starting
      at row ``s``. The implementation slices its pre-patched data with
      ``[s : s + K]`` and returns a chunk-sized guidance term (no fusing).

    Chunked batch mode is the key memory-efficiency knob, the per-chunk
    activations are released between iterations, so peak GPU memory stays
    proportional to ``chunk_size`` rather than to the full
    :math:`P \times B`. Use it for large global domains where the
    full-batch counterpart from :class:`~physicsnemo.diffusion.guidance.DPSGuidance`
    would OOM.

    A guidance satisfying this protocol also satisfies
    :class:`~physicsnemo.diffusion.guidance.DPSGuidance` because the extra
    argument is optional.

    Examples
    --------
    Implementing a simple patch-local guidance from scratch. The mask and
    observations are pre-patched once at construction time and sliced per
    chunk based on ``slice_start``:

    >>> import torch
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDPSGuidance,
    ... )
    >>>
    >>> class InpaintGuidance:
    ...     def __init__(self, mask_patched, y_patched, gamma=1.0):
    ...         self.mask = mask_patched
    ...         self.y = y_patched
    ...         self.gamma = gamma
    ...
    ...     def __call__(self, x, t, x_0, slice_start=None):
    ...         if slice_start is None:
    ...             mask, y = self.mask, self.y
    ...         else:
    ...             K = x.shape[0]
    ...             mask = self.mask[slice_start : slice_start + K]
    ...             y = self.y[slice_start : slice_start + K]
    ...         return -self.gamma * mask * (x_0 - y)
    ...
    >>> mask = torch.ones(8, 3, 8, 8)  # (P*B, C, Hp, Wp)
    >>> y = torch.randn(8, 3, 8, 8)
    >>> guidance = InpaintGuidance(mask, y)
    >>> isinstance(guidance, MultiDiffusionDPSGuidance)
    True
    >>>
    >>> # Full batch mode: process all P*B = 8 patches at once
    >>> x = torch.randn(8, 3, 8, 8)
    >>> t = torch.full((8,), 1.0)
    >>> x_0 = x * 0.9
    >>> guidance(x, t, x_0).shape
    torch.Size([8, 3, 8, 8])
    >>>
    >>> # Chunked batch mode: process a chunk of 2 patches starting at row 0
    >>> guidance(x[:2], t[:2], x_0[:2], slice_start=0).shape
    torch.Size([2, 3, 8, 8])
    """

    def __call__(
        self,
        x: Float[Tensor, "K C Hp Wp"],
        t: Float[Tensor, " K"],
        x_0: Float[Tensor, "K C Hp Wp"],
        slice_start: int | None = None,
    ) -> Float[Tensor, "K C Hp Wp"]: ...


class MultiDiffusionDPSScorePredictor(Predictor):
    r"""Score predictor that combines a
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionPredictor`
    with one or more patch-local DPS guidances for guided sampling on
    large multi-diffusion domains.

    Implements the same :class:`~physicsnemo.diffusion.Predictor`
    interface as :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor`
    and slots into the standard sampling stack: pass it to
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
    to obtain a :class:`~physicsnemo.diffusion.Denoiser` that can be used
    with :func:`~physicsnemo.diffusion.samplers.sample` or any sampling
    utility that consumes a denoiser.

    Use this class instead of
    :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` when every
    guidance is patch-local (see :class:`MultiDiffusionDPSGuidance`) and
    the global domain is too large for the full
    :math:`(P \times B, \dots)` activation tensor to fit in memory. The
    predictor streams score and guidance contributions chunk by chunk in
    patch space and fuses once at the end:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{x})
        + \sum_i g_i(\mathbf{x}, t, \hat{\mathbf{x}}_0)
        \;=\;
        \mathrm{Fuse}\!\left[\, s^k + \sum_i g_i^k\, \right]_{k=1..P}

    where the superscript :math:`k` denotes the :math:`k`-th patch chunk
    and :math:`\mathrm{Fuse}` is the multi-diffusion fusing operator. The
    full :math:`(P \times B, \dots)` activation tensor is never
    materialized.

    .. important::

        Use :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` for
        guidances that do **not** decompose patch-locally. Passing a
        globally-coupled guidance to this class produces incorrect results.

    Each guidance must implement the :class:`MultiDiffusionDPSGuidance`
    protocol:

    .. code-block:: python

        def guidance(
            x: Tensor,                 # shape: (K, C, Hp, Wp)
            t: Tensor,                 # shape: (K,)
            x_0: Tensor,               # shape: (K, C, Hp, Wp)
            slice_start: int | None,   # row index of the chunk in (P*B);
                                       # None means full-batch mode
        ) -> Tensor: ...               # shape: (K, C, Hp, Wp)

    where :math:`K` is the number of patches in the current chunk
    (:math:`K = P \times B` in full batch mode, :math:`K \leq
    \text{chunk\_size}` in chunked batch mode). The predictor forwards
    each chunk's ``slice_start`` from
    :meth:`MultiDiffusionPredictor.chunks` directly to every guidance, so
    each guidance reads the corresponding slice of its own pre-patched
    observations without any internal state.

    The ``x0_to_score_fn`` callback must be an elementwise conversion
    with the signature:

    .. code-block:: python

        def x0_to_score_fn(
            x_0: Tensor,    # shape: (K, C, Hp, Wp)
            x_t: Tensor,    # shape: (K, C, Hp, Wp)
            t: Tensor,      # shape: (K,)
        ) -> Tensor: ...    # shape: (K, C, Hp, Wp)

    Parameters
    ----------
    x0_predictor : MultiDiffusionPredictor
        A trained predictor with ``chunk_size`` set, returning x0
        estimates.
    x0_to_score_fn : callable
        Elementwise conversion ``(x_0, x_t, t) -> score`` (see the
        signature above). Typically obtained from a noise scheduler,
        e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_score`.
    guidances : MultiDiffusionDPSGuidance or sequence of MultiDiffusionDPSGuidance
        One or more patch-local guidance objects implementing the
        :class:`MultiDiffusionDPSGuidance` protocol.

    See Also
    --------
    :class:`MultiDiffusionDPSGuidance` : Protocol that guidances must satisfy.
    :class:`MultiDiffusionDataConsistencyDPSGuidance` : Patch-local
        guidance for masked observations.
    :class:`MultiDiffusionModelConsistencyDPSGuidance` : Patch-local
        guidance for generic patch-local observation operators.
    :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` : Use for
        non-patch-local guidances.

    Notes
    -----
    **Recommended call pattern:** wrap inference-only sample loops in
    ``with torch.no_grad():``. The predictor re-enables autograd locally
    for the guidance's ``autograd.grad`` call, so functionality is
    preserved, and the returned score does not carry a graph back to the
    per-chunk state. Without ``no_grad``, the score's autograd graph
    accumulates across solver steps and can dominate memory at large
    global domains. Do NOT use ``torch.inference_mode()`` — it disables
    autograd entirely and breaks the guidance's internal gradient
    computation.

    Examples
    --------
    **Example 1:** Basic usage with a single inpainting guidance:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> # x0-to-score for EDM: score = (x_0 - x) / t^2
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape((-1,) + (1,) * (x.ndim - 1))
    ...     return (x_0 - x) / (t_bc ** 2)
    >>>
    >>> # Inline inpainting guidance; mask and observations are pre-patched
    >>> # by the user via predictor.patch_fn so all patching uses the same
    >>> # grid as the predictor.
    >>> class InpaintGuidance:
    ...     def __init__(self, mask_patched, y_patched, gamma=0.1):
    ...         self.mask = mask_patched
    ...         self.y = y_patched
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0, slice_start=None):
    ...         if slice_start is None:
    ...             mask, y = self.mask, self.y
    ...         else:
    ...             K = x.shape[0]
    ...             mask = self.mask[slice_start : slice_start + K]
    ...             y = self.y[slice_start : slice_start + K]
    ...         return -self.gamma * mask * (x_0 - y)
    >>>
    >>> mask_patched = predictor.patch_fn(torch.ones(2, 3, 16, 16))
    >>> y_patched = predictor.patch_fn(torch.randn(2, 3, 16, 16))
    >>> guidance = InpaintGuidance(mask_patched, y_patched)
    >>>
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>> x = torch.randn(2, 3, 16, 16)
    >>> t = torch.tensor([1.0, 1.0])
    >>> dps(x, t).shape
    torch.Size([2, 3, 16, 16])

    **Example 2:** Multiple guidances for multi-constraint problems. The
    predictor returned by this class is a drop-in score predictor that
    plugs into any sampling utility (here
    :func:`~physicsnemo.diffusion.samplers.sample`):

    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDataConsistencyDPSGuidance,
    ...     MultiDiffusionModelConsistencyDPSGuidance,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # First guidance: masked observations (inpainting)
    >>> mask = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask[:, :, 4:, :] = True
    >>> y_obs1 = torch.randn(2, 3, 16, 16)
    >>> g1 = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor, mask=mask, y=y_obs1, std_y=0.1,
    ...     retain_graph=True,  # required: not the last autograd guidance
    ... )
    >>>
    >>> # Second guidance: nonlinear patch-local channel response
    >>> A = lambda x_0: torch.sigmoid(x_0[:, :1])
    >>> y_obs2 = torch.rand(2, 1, 16, 16)
    >>> g2 = MultiDiffusionModelConsistencyDPSGuidance(
    ...     predictor=predictor, observation_operator=A,
    ...     y=y_obs2, std_y=0.1,
    ... )
    >>>
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=[g1, g2],
    ... )
    >>> denoiser = scheduler.get_denoiser(score_predictor=dps)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        x0_predictor: MultiDiffusionPredictor,
        x0_to_score_fn: Callable[
            [
                Float[Tensor, "K C Hp Wp"],
                Float[Tensor, "K C Hp Wp"],
                Float[Tensor, " K"],
            ],
            Float[Tensor, "K C Hp Wp"],
        ],
        guidances: MultiDiffusionDPSGuidance | Sequence[MultiDiffusionDPSGuidance],
    ) -> None:
        if not isinstance(x0_predictor, MultiDiffusionPredictor):
            raise TypeError(
                f"x0_predictor must be a MultiDiffusionPredictor, "
                f"got {type(x0_predictor).__name__}."
            )
        if x0_predictor._chunk_size is None:
            raise ValueError(
                "x0_predictor must have chunk_size set. "
                "Pass chunk_size=<int> to MultiDiffusionPredictor.__init__."
            )
        self.x0_predictor = x0_predictor
        self.x0_to_score_fn = x0_to_score_fn
        if isinstance(guidances, Sequence) and not isinstance(guidances, str):
            self.guidances: list[MultiDiffusionDPSGuidance] = list(guidances)
        else:
            self.guidances = [guidances]  # type: ignore[list-item]

    def __call__(
        self,
        x: Float[Tensor, "B C H W"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, "B C H W"]:
        r"""Compute the guided score at the global resolution.

        Parameters
        ----------
        x : Tensor
            Noisy latent at global resolution, shape :math:`(B, C, H, W)`.
        t : Tensor
            Diffusion time, shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Guided score at global resolution, shape :math:`(B, C, H, W)`.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionDPSScorePredictor requires autograd but torch "
                "inference mode is enabled. Wrap the calling code with "
                "'with torch.inference_mode(False):' or 'with torch.no_grad():' "
                "instead."
            )

        # Capture the caller's grad mode so the score-conversion and chunk-add
        # ops below honor it. Under ``torch.no_grad()`` this makes the appended
        # tensors leaves, so the cat/fuse return value does not carry a
        # graph-tail through ``x_patched`` and friends. The guidance call still
        # runs under ``enable_grad`` since it needs ``autograd.grad`` internally.
        outer_grad_enabled = torch.is_grad_enabled()

        x = x.detach().requires_grad_(True)
        combined_list: list[Tensor] = []

        with torch.enable_grad():
            for s, x0_chunk, x_chunk, t_chunk in self.x0_predictor.chunks(x, t):
                g_chunk = torch.zeros_like(x0_chunk)
                for g in self.guidances:
                    g_chunk = g_chunk + g(x_chunk, t_chunk, x0_chunk, slice_start=s)
                with torch.set_grad_enabled(outer_grad_enabled):
                    score_chunk = self.x0_to_score_fn(x0_chunk, x_chunk, t_chunk)
                    combined_list.append(score_chunk + g_chunk)

        combined_patched = torch.cat(combined_list, dim=0)  # (P*B, C, Hp, Wp)
        return self.x0_predictor.fuse_fn(combined_patched)


class MultiDiffusionModelConsistencyDPSGuidance(MultiDiffusionDPSGuidance):
    r"""Patch-local DPS guidance for generic observation operators with
    Gaussian noise.

    Multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance`,
    intended for cases where the observation operator :math:`A`
    decomposes along the multi-diffusion patch grid. Implements the
    :class:`MultiDiffusionDPSGuidance` protocol, see it for the two-mode
    (``slice_start``) semantics and the :math:`K` chunk-size convention.

    Computes the likelihood score assuming Gaussian measurement noise
    with standard deviation :math:`\sigma_y`. For the current patch chunk
    :math:`k`:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{y}^k | \mathbf{x}_t^k)
        = -\nabla_{\mathbf{x}^k} \sum_i
          \frac{\big( A(\hat{\mathbf{x}}_0^k) - \mathbf{y}^k \big)_i^2}
               {2 \left( \sigma_{y,i}^2 + \Gamma_i\, \sigma(t)^2 / \alpha(t)^2
               \right)}

    where :math:`i` indexes the (patch-local) observation components. See the
    global :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance`
    for :math:`\sigma_{y,i}` (``std_y``), :math:`\Gamma_i` (``gamma``), and the
    SDA scaling.

    Observations ``y`` are pre-patched once at construction; calling the
    guidance many times during sampling never re-patches them.

    .. important::

        ``y`` must be **patcheable** in the same way as the latent state
        :math:`\mathbf{x}`, so its spatial dimensions must equal the
        global resolution :math:`(H, W)`. This is a stronger requirement
        than the global counterpart
        :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance`,
        which allows arbitrary observation shapes. The operator
        :math:`A` must therefore produce observations matching the input
        spatial resolution (e.g. channel-selection, pointwise
        nonlinearities, local convolutions within an overlap region).
        Tensor-valued ``std_y`` / ``gamma`` follow the same rule: a scalar, or
        a 4D tensor broadcastable to the global observation
        :math:`(B, C_{obs}, H, W)`, pre-patched at construction like ``y``.

    The ``observation_operator`` must be a differentiable callable with
    the following signature:

    .. code-block:: python

        def observation_operator(
            x_0: Tensor,    # shape: (K, C, Hp, Wp)
        ) -> Tensor: ...    # shape: (K, C_obs, Hp, Wp)

    When ``norm`` is a callable, it must be an elementwise loss with the
    signature:

    .. code-block:: python

        def norm(
            y_pred: Tensor,    # shape: (K, C_obs, Hp, Wp)
            y_true: Tensor,    # shape: (K, C_obs, Hp, Wp)
        ) -> Tensor: ...       # elementwise loss, shape: (K, C_obs, Hp, Wp)

    Parameters
    ----------
    predictor : MultiDiffusionPredictor
        Predictor used to pre-patch ``y`` and (optionally) fuse the
        guidance. Stored on ``self.predictor`` for later access.
    observation_operator : callable
        Differentiable patch-local observation operator :math:`A`. See
        the signature above.
    y : Tensor
        Global observations of shape :math:`(B, C_{obs}, H, W)` matching
        the latent's global spatial shape.
    std_y : float or Tensor
        Standard deviation of the measurement noise
        :math:`\boldsymbol{\sigma}_y`. A ``float`` applies a single standard
        deviation to every observation component. A ``Tensor`` must be 4D and
        broadcastable to the global observation :math:`(B, C_{obs}, H, W)`; it
        is pre-patched like ``y``.
    norm : int or callable, default=2
        Residual loss. An ``int`` selects the corresponding :math:`L^p` norm;
        a callable must return an elementwise loss with the same shape as its
        inputs (see the signature above).
    gamma : float or Tensor, default=0.0
        SDA covariance scaling factor :math:`\boldsymbol{\Gamma}`. Set to
        ``0`` for classical DPS without SDA scaling. Like ``std_y``, may be a
        ``float`` or a ``Tensor`` broadcastable to the global observation.
    sigma_fn : callable or None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``gamma > 0``. Typically obtained from a noise
        scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
    alpha_fn : callable or None, default=None
        Function mapping diffusion time to signal coefficient
        :math:`\alpha(t)`. Defaults to :math:`\alpha(t) = 1`. Typically
        obtained from a noise scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha`.
    fuse : bool, default=False
        Whether :meth:`__call__` fuses the guidance term to the global
        resolution in full batch mode (``slice_start=None``). Ignored in
        chunked batch mode.
    retain_graph : bool, default=False
        Retain the computation graph after the gradient call. Required
        on all but the last guidance when combining multiple
        autograd-based guidances in a single
        :class:`MultiDiffusionDPSScorePredictor`.
    create_graph : bool, default=False
        Allow higher-order derivatives.

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance` :
        Global counterpart for non-patch-local operators.
    :class:`MultiDiffusionDPSScorePredictor` :
        Score predictor that consumes this guidance.

    Examples
    --------
    **Example 1:** Patch-local channel selection. The operator selects
    the first channel of each patch, clearly patch-local. Inputs are
    chunk-sized patched tensors:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionModelConsistencyDPSGuidance,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> A = lambda x: x[:, :1]
    >>> y_obs = torch.randn(2, 1, 16, 16)
    >>>
    >>> guidance = MultiDiffusionModelConsistencyDPSGuidance(
    ...     predictor=predictor, observation_operator=A, y=y_obs, std_y=0.1,
    ... )
    >>> x_chunk = torch.randn(2, 3, 8, 8, requires_grad=True)
    >>> t_chunk = torch.tensor([1.0, 1.0])
    >>> x0_chunk = x_chunk * 0.9
    >>> guidance(x_chunk, t_chunk, x0_chunk, slice_start=0).shape
    torch.Size([2, 3, 8, 8])

    **Example 2:** SDA-scaled guidance with a nonlinear patch-local operator
    (here a sigmoid response on the first two channels) and **tensor-valued**
    ``std_y`` / ``gamma``, plugged into the full sampling stack; ``std_y`` is
    per-channel (spatially constant :math:`(B, C_{obs}, 1, 1)`)
    while ``gamma`` is pointwise at the global resolution
    :math:`(B, C_{obs}, H, W)`; both are pre-patched like ``y``:

    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> A_nl = lambda x_0: torch.sigmoid(x_0[:, :2])  # 2 observed channels
    >>> y_obs_nl = torch.rand(2, 2, 16, 16)
    >>>
    >>> std_y = torch.tensor([0.05, 0.1]).reshape(1, 2, 1, 1)  # per-channel
    >>> gamma = 0.05 * torch.rand(2, 2, 16, 16)                # pointwise (B, C, H, W)
    >>> guidance_sda = MultiDiffusionModelConsistencyDPSGuidance(
    ...     predictor=predictor,
    ...     observation_operator=A_nl,
    ...     y=y_obs_nl,
    ...     std_y=std_y,
    ...     gamma=gamma,         # enable SDA scaling
    ...     sigma_fn=scheduler.sigma,
    ...     alpha_fn=scheduler.alpha,
    ... )
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance_sda,
    ... )
    >>> denoiser = scheduler.get_denoiser(score_predictor=dps)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        predictor: MultiDiffusionPredictor,
        observation_operator: Callable[
            [Float[Tensor, "K C Hp Wp"]], Float[Tensor, "K C_obs Hp Wp"]
        ],
        y: Float[Tensor, "B C_obs H W"],
        std_y: float | Float[Tensor, "#B #C_obs #H #W"],
        norm: int
        | Callable[
            [Float[Tensor, "K C_obs Hp Wp"], Float[Tensor, "K C_obs Hp Wp"]],
            Float[Tensor, "K C_obs Hp Wp"],
        ] = 2,
        gamma: float | Float[Tensor, "#B #C_obs #H #W"] = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        fuse: bool = False,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        self.predictor = predictor
        # Pre-patch observations once via the predictor's patch_fn.
        self._y_patched: Tensor = predictor.patch_fn(y)
        self.observation_operator = observation_operator
        # Pre-patch std_y / gamma once like ``y`` (a scalar broadcasts to
        # everything), so they slice per chunk alongside the observations.
        self._std_y_patched = _prepatch_param(std_y, predictor, y)
        self._gamma_patched = _prepatch_param(gamma, predictor, y)
        if sigma_fn is None and (self._gamma_patched > 0).any():
            raise ValueError("sigma_fn must be provided when gamma > 0")
        # Resolve the loss callable at construction so __call__ has no branch.
        if isinstance(norm, int):
            self._loss_fn: Callable[[Tensor, Tensor], Tensor] = _lp_loss_fn(norm)
        else:
            self._loss_fn = norm
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.fuse = fuse
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, "K C Hp Wp"],
        t: Float[Tensor, " K"],
        x_0: Float[Tensor, "K C Hp Wp"],
        slice_start: int | None = None,
    ) -> Float[Tensor, "K C Hp Wp"] | Float[Tensor, "B C H W"]:
        r"""Compute the patch-local likelihood score guidance term.

        See :class:`MultiDiffusionDPSGuidance` for the meaning of
        ``slice_start`` (full vs chunked batch mode) and the :math:`K`
        chunk-size convention.

        Parameters
        ----------
        x : Tensor
            Noisy patched latent slice :math:`\mathbf{x}_t^k`, of shape
            :math:`(K, C, H_p, W_p)`. Must have ``requires_grad=True``
            and be part of a computational graph connecting to ``x_0``.
            Its ``dtype`` and ``device`` determine those of all internal
            computations.
        t : Tensor
            Patched diffusion time slice, shape :math:`(K,)`.
        x_0 : Tensor
            Estimate of the patched clean state
            :math:`\hat{\mathbf{x}}_0^k(\mathbf{x}_t^k, t)`, of shape
            :math:`(K, C, H_p, W_p)`. Must be computed from ``x`` so
            gradients can backpropagate.
        slice_start : int or None, default=None
            Chunk offset along the :math:`(P \times B)` dimension. See
            class docstring.

        Returns
        -------
        Tensor
            Patch-local guidance term of shape :math:`(K, C, H_p, W_p)`.
            Fused to the global resolution :math:`(B, C, H, W)` when
            ``slice_start=None`` and ``fuse=True`` was passed at
            construction.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionModelConsistencyDPSGuidance requires autograd "
                "but torch inference mode is enabled."
            )

        # Slice the pre-patched observation / strengths to the current chunk.
        K = x.shape[0]
        sl = slice(None) if slice_start is None else slice(slice_start, slice_start + K)
        y_chunk = self._y_patched[sl].to(dtype=x.dtype, device=x.device)
        std_y_chunk = self._std_y_patched[sl].to(dtype=x.dtype, device=x.device)
        gamma_chunk = self._gamma_patched[sl].to(dtype=x.dtype, device=x.device)

        with torch.enable_grad():
            y_pred = self.observation_operator(x_0)
            loss = self._loss_fn(y_pred, y_chunk)
            # Guidance strength rho(t), broadcast over the observation.
            bc_shape = (-1,) + (1,) * (loss.ndim - 1)
            t_bc = t.reshape(bc_shape)
            sigma_t = self.sigma_fn(t_bc)
            alpha_t = self.alpha_fn(t_bc)
            rho = 1.0 / (
                2.0 * (std_y_chunk**2 + gamma_chunk * (sigma_t**2) / (alpha_t**2))
            )
            grad_x = torch.autograd.grad(
                outputs=(rho * loss).sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )[0]

        g = -grad_x
        if slice_start is None and self.fuse:
            return self.predictor.fuse_fn(g)
        return g


class MultiDiffusionDataConsistencyDPSGuidance(MultiDiffusionDPSGuidance):
    r"""Patch-local DPS guidance for masked observations with Gaussian
    noise.

    Multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance`,
    intended for masked observations whose mask decomposes along the
    multi-diffusion patch grid. Use cases: inpainting, sparse pointwise
    data assimilation on large domains. Implements the
    :class:`MultiDiffusionDPSGuidance` protocol, see it for the two-mode
    (``slice_start``) semantics and the :math:`K` chunk-size convention.

    Computes the likelihood score assuming Gaussian measurement noise
    with standard deviation :math:`\sigma_y`. For the current patch chunk
    :math:`k`:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{y}^k | \mathbf{x}_t^k)
        = -\nabla_{\mathbf{x}^k} \sum_i
          \frac{\big( \mathbf{M}^k \odot (\hat{\mathbf{x}}_0^k - \mathbf{y}^k)
               \big)_i^2}
               {2 \left( \sigma_{y,i}^2 + \Gamma_i\, \sigma(t)^2 / \alpha(t)^2
               \right)}

    where :math:`\mathbf{M}` is a binary mask (1 = observed, 0 = missing) and
    :math:`\odot` element-wise multiplication. See the global
    :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance` for
    :math:`\sigma_{y,i}` (``std_y``), :math:`\Gamma_i` (``gamma``), and the SDA
    scaling.

    Both ``mask`` and ``y`` are pre-patched once at construction;
    calling the guidance many times during sampling never re-patches
    them.

    .. important::

        ``mask`` and ``y`` must be **patcheable** in the same way as the
        latent state :math:`\mathbf{x}`, so their spatial dimensions
        must equal the global resolution :math:`(H, W)`. The mask
        defines per-pixel observability within the global spatial
        domain. Tensor-valued ``std_y`` / ``gamma`` follow the same rule: a
        scalar, or a 4D tensor broadcastable to the global resolution
        :math:`(B, C, H, W)`, pre-patched at construction.

    When ``norm`` is a callable, it must be an elementwise loss with the
    signature:

    .. code-block:: python

        def norm(
            y_pred: Tensor,    # shape: (K, C, Hp, Wp)
            y_true: Tensor,    # shape: (K, C, Hp, Wp)
        ) -> Tensor: ...       # elementwise loss, shape: (K, C, Hp, Wp)

    Parameters
    ----------
    predictor : MultiDiffusionPredictor
        Predictor used to pre-patch ``mask`` and ``y`` and (optionally)
        fuse the guidance. Stored on ``self.predictor`` for later access.
    mask : Tensor
        Boolean mask of shape :math:`(B, C, H, W)`. ``True`` marks
        observed locations, ``False`` marks missing.
    y : Tensor
        Observed values of shape :math:`(B, C, H, W)`. Values at
        unobserved locations are ignored.
    std_y : float or Tensor
        Standard deviation of the measurement noise
        :math:`\boldsymbol{\sigma}_y`. A ``float`` applies a single standard
        deviation everywhere. A ``Tensor`` must be 4D and broadcastable to the
        global resolution :math:`(B, C, H, W)`; it is pre-patched like ``mask``
        and ``y``.
    norm : int or callable, default=2
        Residual loss. An ``int`` selects the corresponding :math:`L^p` norm; a
        callable receives ``(x_0, y)`` and must return an elementwise loss with
        the same shape as its inputs (the mask is applied to the result); see
        the signature above.
    gamma : float or Tensor, default=0.0
        SDA covariance scaling factor :math:`\boldsymbol{\Gamma}`. Set to
        ``0`` for classical DPS without SDA scaling. Like ``std_y``, may be a
        ``float`` or a ``Tensor`` broadcastable to the global observation.
    sigma_fn : callable or None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``gamma > 0``. Typically obtained from a noise
        scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
    alpha_fn : callable or None, default=None
        Function mapping diffusion time to signal coefficient
        :math:`\alpha(t)`. Defaults to :math:`\alpha(t) = 1`. Typically
        obtained from a noise scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha`.
    fuse : bool, default=False
        Whether :meth:`__call__` fuses the guidance term to the global
        resolution in full batch mode (``slice_start=None``). Ignored in
        chunked batch mode.
    retain_graph : bool, default=False
        Retain the computation graph after the gradient call. Required
        on all but the last guidance when combining multiple
        autograd-based guidances in a single
        :class:`MultiDiffusionDPSScorePredictor`.
    create_graph : bool, default=False
        Allow higher-order derivatives.

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance` :
        Global counterpart for non-patch-local masks.
    :class:`MultiDiffusionDPSScorePredictor` :
        Score predictor that consumes this guidance.

    Examples
    --------
    **Example 1:** Inpainting on a large domain. The mask is a spatial
    pattern, so it decomposes along the patch grid:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionDataConsistencyDPSGuidance,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> mask = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask[:, :, 4:, :] = True
    >>> y_obs = torch.randn(2, 3, 16, 16)
    >>>
    >>> guidance = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor, mask=mask, y=y_obs, std_y=0.1,
    ... )
    >>> x_chunk = torch.randn(2, 3, 8, 8, requires_grad=True)
    >>> t_chunk = torch.tensor([1.0, 1.0])
    >>> x0_chunk = x_chunk * 0.9
    >>> guidance(x_chunk, t_chunk, x0_chunk, slice_start=0).shape
    torch.Size([2, 3, 8, 8])

    **Example 2:** Sparse probe observations with a **per-probe** measurement
    noise and a custom elementwise (Huber) loss, plugged into the full sampling
    stack; ``std_y`` is a pointwise tensor at the global resolution that sets an
    independent noise standard deviation at each probe (only its entries at
    observed locations matter), while ``gamma`` stays a scalar:

    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # Elementwise Huber loss (no reduction)
    >>> def huber_loss(y_pred, y_true):
    ...     return F.huber_loss(y_pred, y_true, reduction="none")
    ...
    >>> # Sparse probe locations on the global domain
    >>> mask = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask[:, :, 2, 3] = True
    >>> mask[:, :, 5, 6] = True
    >>> mask[:, :, 11, 4] = True
    >>> y_obs = torch.randn(2, 3, 16, 16)
    >>>
    >>> # Per-probe measurement noise (pointwise tensor at the global resolution);
    >>> # entries away from the probes are unused but must stay positive
    >>> std_y = torch.ones(2, 3, 16, 16)
    >>> std_y[:, :, 2, 3] = 0.05
    >>> std_y[:, :, 5, 6] = 0.1
    >>> std_y[:, :, 11, 4] = 0.3
    >>> guidance = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor,
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=std_y,         # per-probe measurement noise
    ...     norm=huber_loss,     # custom elementwise loss
    ...     gamma=1.0,           # scalar SDA scaling
    ...     sigma_fn=scheduler.sigma,
    ...     alpha_fn=scheduler.alpha,
    ... )
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance,
    ... )
    >>> denoiser = scheduler.get_denoiser(score_predictor=dps)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        predictor: MultiDiffusionPredictor,
        mask: Bool[Tensor, "B C H W"],
        y: Float[Tensor, "B C H W"],
        std_y: float | Float[Tensor, "#B #C #H #W"],
        norm: int
        | Callable[
            [Float[Tensor, "K C Hp Wp"], Float[Tensor, "K C Hp Wp"]],
            Float[Tensor, "K C Hp Wp"],
        ] = 2,
        gamma: float | Float[Tensor, "#B #C #H #W"] = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        fuse: bool = False,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        self.predictor = predictor
        # Pre-patch mask and observations once via the predictor's patch_fn.
        patch = predictor.patch_fn
        self._mask_patched: Tensor = patch(mask.float())
        self._y_patched: Tensor = patch(y)
        # Pre-patch std_y / gamma once like ``mask`` / ``y`` (a scalar
        # broadcasts to everything), so they slice per chunk alongside them.
        self._std_y_patched = _prepatch_param(std_y, predictor, y)
        self._gamma_patched = _prepatch_param(gamma, predictor, y)
        if sigma_fn is None and (self._gamma_patched > 0).any():
            raise ValueError("sigma_fn must be provided when gamma > 0")
        # Resolve the loss callable at construction so __call__ has no branch.
        if isinstance(norm, int):
            self._loss_fn: Callable[[Tensor, Tensor], Tensor] = _lp_loss_fn(norm)
        else:
            self._loss_fn = norm
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.fuse = fuse
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, "K C Hp Wp"],
        t: Float[Tensor, " K"],
        x_0: Float[Tensor, "K C Hp Wp"],
        slice_start: int | None = None,
    ) -> Float[Tensor, "K C Hp Wp"] | Float[Tensor, "B C H W"]:
        r"""Compute the patch-local likelihood score guidance term.

        See :class:`MultiDiffusionDPSGuidance` for the meaning of
        ``slice_start`` (full vs chunked batch mode) and the :math:`K`
        chunk-size convention.

        Parameters
        ----------
        x : Tensor
            Noisy patched latent slice :math:`\mathbf{x}_t^k`, of shape
            :math:`(K, C, H_p, W_p)`. Must have ``requires_grad=True``
            and be part of a computational graph connecting to ``x_0``.
            Its ``dtype`` and ``device`` determine those of all internal
            computations.
        t : Tensor
            Patched diffusion time slice, shape :math:`(K,)`.
        x_0 : Tensor
            Estimate of the patched clean state
            :math:`\hat{\mathbf{x}}_0^k(\mathbf{x}_t^k, t)`, of shape
            :math:`(K, C, H_p, W_p)`. Must be computed from ``x`` so
            gradients can backpropagate.
        slice_start : int or None, default=None
            Chunk offset along the :math:`(P \times B)` dimension. See
            class docstring.

        Returns
        -------
        Tensor
            Patch-local guidance term of shape :math:`(K, C, H_p, W_p)`.
            Fused to the global resolution :math:`(B, C, H, W)` when
            ``slice_start=None`` and ``fuse=True`` was passed at
            construction.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionDataConsistencyDPSGuidance requires autograd "
                "but torch inference mode is enabled."
            )

        # Slice the pre-patched mask / observation / strengths to the chunk.
        K = x.shape[0]
        sl = slice(None) if slice_start is None else slice(slice_start, slice_start + K)
        mask_chunk = self._mask_patched[sl].to(dtype=x.dtype, device=x.device)
        y_chunk = self._y_patched[sl].to(dtype=x.dtype, device=x.device)
        std_y_chunk = self._std_y_patched[sl].to(dtype=x.dtype, device=x.device)
        gamma_chunk = self._gamma_patched[sl].to(dtype=x.dtype, device=x.device)

        with torch.enable_grad():
            # Elementwise loss on the full state, then keep observed locations.
            loss = mask_chunk * self._loss_fn(x_0, y_chunk)
            # Guidance strength rho(t), broadcast over the state.
            bc_shape = (-1,) + (1,) * (loss.ndim - 1)
            t_bc = t.reshape(bc_shape)
            sigma_t = self.sigma_fn(t_bc)
            alpha_t = self.alpha_fn(t_bc)
            rho = 1.0 / (
                2.0 * (std_y_chunk**2 + gamma_chunk * (sigma_t**2) / (alpha_t**2))
            )
            grad_x = torch.autograd.grad(
                outputs=(rho * loss).sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )[0]

        g = -grad_x
        if slice_start is None and self.fuse:
            return self.predictor.fuse_fn(g)
        return g
