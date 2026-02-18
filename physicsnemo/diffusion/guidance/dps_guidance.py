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

"""DPS (Diffusion Posterior Sampling) guidance for diffusion models."""

from typing import Callable, Protocol, Sequence, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser, Predictor


@runtime_checkable
class DPSGuidance(Protocol):
    r"""
    Protocol defining the interface for Diffusion Posterior Sampling (DPS)
    guidance.

    A DPS guidance is a callable that computes a guidance term to steer the
    diffusion sampling process toward satisfying some observation constraint.
    It returns a quantity analogous to a likelihood score, which is typically
    added to the unconditional score during sampling.

    The typical form is:

    .. math::
        \gamma(t) \nabla_{\mathbf{x}}
        \ell(A(\hat{\mathbf{x}}_0) - \mathbf{y})

    where :math:`\gamma(t)` is a time-dependent guidance strength,
    :math:`A` is a (potentially nonlinear) observation operator,
    :math:`\mathbf{y}` is the observed data, and :math:`\ell` is a scalar loss
    function. However, variants are possible as long as the guidance produces
    a quantity similar to a score (e.g., a likelihood score).

    This is the minimal interface for guidance, and any object that implements
    this interface can be used with :class:`DPSDenoiser` to build a guided
    score-predictor, which implements the
    :class:`~physicsnemo.diffusion.Predictor` interface.

    See Also
    --------
    :class:`DPSDenoiser` : Combines an x0-predictor with one or more guidances.

    Examples
    --------
    **Example 1:** Minimal guidance for inpainting. Given a binary mask and
    observed pixels, guide the diffusion to match observations:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSGuidance
    >>>
    >>> class InpaintingGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask  # Binary mask: 1 = observed, 0 = missing
    ...         self.y_obs = y_obs  # Observed pixel values
    ...         self.gamma = gamma
    ...
    ...     def __call__(self, x, t, x_0):
    ...         # Compute residual at observed locations
    ...         residual = self.mask * (x_0 - self.y_obs)
    ...         # Gradient of L2 loss w.r.t. x_0 is just the residual
    ...         # (simplified: assumes identity observation operator)
    ...         return -self.gamma * residual
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = InpaintingGuidance(mask, y_obs)
    >>> isinstance(guidance, DPSGuidance)
    True

    **Example 2:** Building a guided denoiser from scratch. A common pattern
    is to combine an x0-predictor with a guidance to create a score predictor
    that can be used for sampling. This shows the complete workflow:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSGuidance
    >>>
    >>> # Define a guidance that pushes toward observed values
    >>> class MyGuidance:
    ...     def __init__(self, y_obs, gamma=0.1):
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * (x_0 - self.y_obs)
    ...
    >>> # Toy x0-predictor (in practice, a trained neural network)
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = MyGuidance(y_obs, gamma=0.5)
    >>>
    >>> # Build a guided denoiser that combines x0-predictor + guidance
    >>> def guided_denoiser(x, t):
    ...     # Step 1: Get x0 estimate
    ...     x_0 = x0_predictor(x, t)
    ...     # Step 2: Compute guidance term
    ...     guidance_term = guidance(x, t, x_0)
    ...     # Step 3: Convert x0 to score (for EDM: score = (x_0 - x) / t^2)
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     score = (x_0 - x) / (t_bc ** 2)
    ...     # Step 4: Sum and return
    ...     return score + guidance_term
    ...
    >>> # guided_denoiser is now a Predictor (score predictor); pass it to
    >>> # scheduler.get_denoiser(score_predictor=...) to obtain a Denoiser
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> output = guided_denoiser(x, t)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    Note: :class:`DPSDenoiser` provides a convenient way to apply one or more
    guidances to an x0-predictor without manually implementing the above pattern.
    """

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
            Typically used to compute gradients when the guidance requires
            backpropagation through the diffusion process, in which case it
            needs to have ``requires_grad=True``.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state, of shape :math:`(B, *)`.
            Typically produced by an x0-predictor or clean data predictor.

        Returns
        -------
        Tensor
            Guidance term of the same shape as ``x``. This is analogous to a
            likelihood score and is typically added to the unconditional score
            to guide the sampling process.
        """
        ...


class DPSDenoiser(Denoiser):
    r"""
    Score predictor that combines an x0-predictor with DPS-style guidance.

    This class transforms a :class:`~physicsnemo.diffusion.Predictor`
    (specifically, an **x0-predictor**) into a score-predictor
    (with the :class:`~physicsnemo.diffusion.Denoiser` interface) by applying one or more DPS
    guidances. The resulting score-predictor can be passed to
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
    to obtain a :class:`~physicsnemo.diffusion.Denoiser` for sampling.

    The output is the sum of the unconditional score (derived from the
    x0-prediction) and all guidance terms:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{x})
        + \sum_i g_i(\mathbf{x}, t, \hat{\mathbf{x}}_0)

    where :math:`g_i` are the guidance terms implementing the
    :class:`DPSGuidance` interface.

    Each guidance must implement the :class:`DPSGuidance` protocol, which is a
    callable with the following signature:

    .. code-block:: python

        def guidance(
            x: Tensor,    # shape: (B, *dims)
            t: Tensor,    # shape: (B,)
            x_0: Tensor,  # shape: (B, *dims)
        ) -> Tensor: ...  # guidance term, shape: (B, *dims)

    .. important::

        When using **multiple guidances** that internally call
        ``torch.autograd.grad`` (e.g., :class:`ModelConsistencyDPSGuidance`
        or :class:`DataConsistencyDPSGuidance`), each guidance except the last
        must be constructed with ``retain_graph=True``. Otherwise the
        computational graph is destroyed after the first guidance computes its
        gradient and subsequent guidances will fail. With a **single guidance**
        this is not needed.

    Parameters
    ----------
    x0_predictor : Predictor
        A :class:`~physicsnemo.diffusion.Predictor` that takes ``(x, t)``
        and returns an estimate of the clean data
        :math:`\hat{\mathbf{x}}_0`. Typically obtained from a trained
        :class:`~physicsnemo.diffusion.DiffusionModel` via
        ``functools.partial``.
    x0_to_score_fn : Callable[[Tensor, Tensor, Tensor], Tensor]
        Callback to convert x0-prediction to score. Signature:
        ``x0_to_score_fn(x_0, x, t) -> score``. Typically obtained from a noise
        scheduler, e.g.,
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_score`.
    guidances : DPSGuidance | Sequence[DPSGuidance]
        One or more guidance objects implementing the :class:`DPSGuidance`
        interface.

    See Also
    --------
    :class:`DPSGuidance` : Protocol for guidance implementations.
    :class:`~physicsnemo.diffusion.Predictor` : Protocol satisfied by this class.
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser` :
        Converts the score-predictor to a denoiser for sampling.

    Examples
    --------
    **Example 1:** Basic usage with a single guidance for inpainting:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSDenoiser, DPSGuidance
    >>>
    >>> # Toy x0-predictor (in practice, this is a trained neural network)
    >>> x0_predictor = lambda x, t: x * 0.9
    >>>
    >>> # Simple x0_to_score function (for EDM: score = (x_0 - x) / t^2)
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     return (x_0 - x) / (t_bc ** 2)
    ...
    >>> # Simple inpainting guidance
    >>> class InpaintGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * self.mask * (x_0 - self.y_obs)
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = InpaintGuidance(mask, y_obs)
    >>>
    >>> # Create DPS score predictor
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>>
    >>> # Use in sampling
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> output = dps_denoiser(x, t)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** Multiple guidances for multi-constraint problems:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSDenoiser
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> # Use scheduler to get x0_to_score_fn
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_predictor = lambda x, t: x * 0.9
    >>>
    >>> # Guidance 1: match observed values at specific locations
    >>> class ObservationGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * self.mask * (x_0 - self.y_obs)
    ...
    >>> # Guidance 2: regularization toward zero mean
    >>> class ZeroMeanGuidance:
    ...     def __init__(self, gamma=0.1):
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * x_0.mean() * torch.ones_like(x_0)
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance1 = ObservationGuidance(mask, y_obs)
    >>> guidance2 = ZeroMeanGuidance()
    >>>
    >>> # Combine multiple guidances
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=[guidance1, guidance2],
    ... )
    >>>
    >>> x = torch.randn(2, 3, 8, 8)
    >>> t = torch.tensor([1.0, 1.0])
    >>> output = dps_denoiser(x, t)
    >>> output.shape
    torch.Size([2, 3, 8, 8])

    **Example 3:** Multiple autograd-based guidances require
    ``retain_graph=True`` on all but the last:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import (
    ...     DPSDenoiser,
    ...     DataConsistencyDPSGuidance,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_predictor = lambda x, t: x * 0.9
    >>>
    >>> mask1 = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    >>> mask1[:, :, 2, 3] = True
    >>> mask2 = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    >>> mask2[:, :, 5, 6] = True
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>>
    >>> # First guidance retains the graph for the second one
    >>> g1 = DataConsistencyDPSGuidance(
    ...     mask=mask1, y=y_obs, std_y=0.1, retain_graph=True,
    ... )
    >>> # Last guidance does not need retain_graph
    >>> g2 = DataConsistencyDPSGuidance(
    ...     mask=mask2, y=y_obs, std_y=0.1,
    ... )
    >>>
    >>> dps = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=[g1, g2],
    ... )
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> dps(x, t).shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        x0_predictor: Predictor,
        x0_to_score_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B *dims"], Float[Tensor, " B"]],
            Float[Tensor, " B *dims"],
        ],
        guidances: DPSGuidance | Sequence[DPSGuidance],
    ) -> None:
        self.x0_predictor = x0_predictor
        self.x0_to_score_fn = x0_to_score_fn
        # Normalize guidances to a list
        if isinstance(guidances, Sequence) and not isinstance(guidances, str):
            self.guidances = list(guidances)
        else:
            self.guidances = [guidances]

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the guided score for sampling.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Guided score of same shape :math:`(B, *)` as ``x``. Computed as the
            sum of the unconditional score and all guidance terms.
        """
        x = x.detach().requires_grad_(True)

        with torch.enable_grad():
            x_0 = self.x0_predictor(x, t)
            guidance_sum = torch.zeros_like(x)
            for guidance in self.guidances:
                guidance_sum += guidance(x, t, x_0)

        score = self.x0_to_score_fn(x_0, x, t)
        return score + guidance_sum


class ModelConsistencyDPSGuidance(DPSGuidance):
    r"""
    DPS guidance for generic observation models with Gaussian noise.

    Implements the :class:`DPSGuidance` interface for generic (possibly
    nonlinear) observation operators.

    Computes the likelihood score assuming Gaussian measurement noise with
    standard deviation ``std_y``. The guidance term is:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{y} | \mathbf{x}_t)
        = -\frac{1}{2 \left( \sigma_y^2 + \gamma \frac{\sigma(t)^2}{\alpha(t)^2}
        \right)} \nabla_{\mathbf{x}}
        \| A\left(\hat{\mathbf{x}}_0\right) - \mathbf{y} \|^2

    where :math:`A` is the observation operator and the scaling incorporates
    a Score-Based Data Assimilation (SDA) correction through the parameter
    :math:`\gamma` that accounts for the variance of the
    :math:`\hat{\mathbf{x}}_0(\mathbf{x}_t, t)` estimate at different diffusion
    times. The L2 norm can be replaced by other Lp norms or custom loss
    functions via the ``norm`` parameter.

    The ``observation_operator`` must be a differentiable callable with the
    following signature:

    .. code-block:: python

        def observation_operator(
            x_0: Tensor,  # shape: (B, *dims)
        ) -> Tensor: ...  # predicted observations, shape: (B, *obs_dims)

    When ``norm`` is a callable, it must have the following signature:

    .. code-block:: python

        def norm(
            y_pred: Tensor,  # shape: (B, *obs_dims)
            y_true: Tensor,  # shape: (B, *obs_dims)
        ) -> Tensor: ...    # scalar loss per batch element, shape: (B,)

    Parameters
    ----------
    observation_operator : Callable[[Tensor], Tensor]
        Observation operator mapping clean state to observations.
        Must be differentiable (supports ``torch.autograd``).
    y : Tensor
        Observed data of shape :math:`(B, *obs\_dims)` matching the output
        of ``A``.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm : int | Callable[[Tensor, Tensor], Tensor], default=2
        Loss function used to compute the residual. An ``int`` value (default
        ``2``) uses the corresponding Lp norm. A callable receives
        ``(y_pred, y_true)`` and returns a scalar loss per batch element of
        shape :math:`(B,)`.
    gamma : float, default=0.0
        SDA scaling parameter. When ``gamma > 0``, applies SDA correction
        that accounts for the variance of the x0 estimate. Set to ``0`` for
        classical DPS without SDA scaling.
    sigma_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``gamma > 0``. Typically obtained from a noise
        scheduler, e.g.,
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`
        for a linear-Gaussian noise schedule.
    alpha_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to signal coefficient :math:`\alpha(t)`.
        Optional; defaults to :math:`\alpha(t) = 1` if not provided. Typically
        obtained from a noise scheduler, e.g.,
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha`
        for a linear-Gaussian noise schedule.
    retain_graph : bool, default=False
        If ``True``, the computational graph is retained after computing
        gradients. Required when combining multiple autograd-based guidances
        in a single :class:`DPSDenoiser` — all guidances except the last
        must set this to ``True``.
    create_graph : bool, default=False
        If ``True``, a graph of the derivative is constructed, allowing
        higher-order derivatives (e.g., differentiating through the entire
        sampling process).

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`DataConsistencyDPSGuidance` : Simplified guidance for masked
        observations.
    :class:`DPSDenoiser` : Combines an x0-predictor with one or more guidances.

    Examples
    --------
    **Example 1:** Super-resolution with a nonlinear blur + downsampling
    operator:

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion.guidance import (
    ...     ModelConsistencyDPSGuidance,
    ...     DPSDenoiser,
    ... )
    >>>
    >>> # Observation operator: Gaussian blur + 2x downsampling
    >>> def blur_downsample(x):
    ...     # Apply 3x3 Gaussian-like blur
    ...     kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9
    ...     kernel = kernel.expand(x.shape[1], 1, 3, 3)
    ...     blurred = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
    ...     # Downsample 2x
    ...     return F.avg_pool2d(blurred, kernel_size=2, stride=2)
    ...
    >>> # Low-resolution observations (4x4 from 8x8 high-res)
    >>> y_obs = torch.randn(1, 3, 4, 4)
    >>>
    >>> guidance = ModelConsistencyDPSGuidance(
    ...     observation_operator=blur_downsample,
    ...     y=y_obs,
    ...     std_y=0.1,
    ... )
    >>>
    >>> # Use in DPS sampling
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9  # Toy x0 estimate
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    >>>
    >>> # Combine with DPSDenoiser for complete sampling workflow
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     return (x_0 - x) / (t_bc ** 2)
    ...
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>> score = dps_denoiser(x, t)
    >>> score.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** With SDA scaling using noise scheduler methods:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import (
    ...     ModelConsistencyDPSGuidance,
    ...     DPSDenoiser,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # Linear observation operator (select first channel)
    >>> A = lambda x: x[:, :1]
    >>> y_obs = torch.randn(1, 1, 8, 8)
    >>>
    >>> # Enable SDA scaling with gamma > 0, providing sigma and alpha functions
    >>> guidance = ModelConsistencyDPSGuidance(
    ...     observation_operator=A,
    ...     y=y_obs,
    ...     std_y=0.075,
    ...     gamma=0.05,  # Enable SDA scaling
    ...     sigma_fn=scheduler.sigma,
    ...     alpha_fn=scheduler.alpha,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    >>>
    >>> # Use with DPSDenoiser and scheduler's x0_to_score
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance,
    ... )
    >>> score = dps_denoiser(x, t)
    >>> score.shape
    torch.Size([1, 3, 8, 8])

    **Example 3:** With a custom loss function (Huber loss):

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion.guidance import ModelConsistencyDPSGuidance
    >>>
    >>> # Wrap torch's Huber loss to return per-batch scalars
    >>> def huber_loss(y_pred, y_true):
    ...     per_elem = F.huber_loss(y_pred, y_true, reduction="none")
    ...     return per_elem.reshape(y_pred.shape[0], -1).sum(dim=1)
    ...
    >>> A = lambda x: x[:, :1]  # Select first channel
    >>> y_obs = torch.randn(1, 1, 8, 8)
    >>>
    >>> guidance = ModelConsistencyDPSGuidance(
    ...     observation_operator=A,
    ...     y=y_obs,
    ...     std_y=0.1,
    ...     norm=huber_loss,  # Custom loss function
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        observation_operator: Callable[
            [Float[Tensor, " B *dims"]], Float[Tensor, " B *obs_dims"]
        ],
        y: Float[Tensor, " B *obs_dims"],
        std_y: float,
        norm: int
        | Callable[
            [Float[Tensor, " B *obs_dims"], Float[Tensor, " B *obs_dims"]],
            Float[Tensor, " B"],
        ] = 2,
        gamma: float = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        if gamma > 0 and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when gamma > 0")
        self.observation_operator = observation_operator
        self.y = y
        self.std_y = std_y
        self.norm = norm
        self.gamma = gamma
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state :math:`\mathbf{x}_t`, of shape :math:`(B, *)`.
            Must have ``requires_grad=True`` and be part of a computational
            graph connecting to ``x_0``. Its ``dtype`` and ``device``
            determine those of all internal computations.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state :math:`\hat{\mathbf{x}}_0
            (\mathbf{x}_t, t)`, with same shape as ``x``. Must be computed
            from ``x`` via an x0-predictor to allow gradient backpropagation.

        Returns
        -------
        Tensor
            Likelihood score guidance term of same shape as ``x``.
        """
        # Ensure stored tensors match x's dtype and device
        y = self.y.to(dtype=x.dtype, device=x.device)

        with torch.enable_grad():
            # Compute predicted observations
            y_pred = self.observation_operator(x_0)

            # Compute loss
            if callable(self.norm):
                loss = self.norm(y_pred, y)
            else:
                residual = (y_pred - y).reshape(y_pred.shape[0], -1)
                loss = residual.abs().pow(self.norm).sum(dim=1)

            # Compute gradient of loss w.r.t. x (backprop through x_0)
            grad_x = torch.autograd.grad(
                outputs=loss.sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )[0]

        # Compute scaling factor
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
        sigma_t = self.sigma_fn(t_bc)
        alpha_t = self.alpha_fn(t_bc)
        variance = self.std_y**2 + self.gamma * (sigma_t**2) / (alpha_t**2)

        return -grad_x / (2 * variance)


class DataConsistencyDPSGuidance(DPSGuidance):
    r"""
    DPS guidance for masked observations with Gaussian noise.

    Implements the :class:`DPSGuidance` interface for masked observation
    operators, a simplified version of :class:`ModelConsistencyDPSGuidance`.
    This is typical for data assimilation tasks like inpainting, outpainting,
    or sparse observations, where measurements are available at specific
    locations.

    Computes the likelihood score assuming Gaussian measurement noise with
    standard deviation ``std_y``. The guidance term is:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{y} | \mathbf{x}_t)
        = -\frac{1}{2 \left( \sigma_y^2 + \gamma \frac{\sigma(t)^2}{\alpha(t)^2}
        \right)} \nabla_{\mathbf{x}}
        \| \mathbf{M} \odot (\hat{\mathbf{x}}_0 - \mathbf{y}) \|^2

    where :math:`\mathbf{M}` is a binary mask (1 = observed, 0 = missing),
    :math:`\odot` denotes element-wise multiplication, and the scaling
    incorporates an SDA correction through the parameter :math:`\gamma`. The
    L2 norm can be replaced by other Lp norms or custom loss functions via the
    ``norm`` parameter.

    When ``norm`` is a callable, it must have the following signature:

    .. code-block:: python

        def norm(
            y_pred: Tensor,  # shape: (B, *obs_dims)
            y_true: Tensor,  # shape: (B, *obs_dims)
        ) -> Tensor: ...    # scalar loss per batch element, shape: (B,)

    Parameters
    ----------
    mask : Tensor
        Boolean mask of shape :math:`(B, *)` matching the state shape.
        ``True`` for observed locations, ``False`` for missing.
    y : Tensor
        Observed data of shape :math:`(B, *)` matching the state shape.
        Values at unobserved locations (where ``mask=0``) are ignored.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm : int | Callable[[Tensor, Tensor], Tensor], default=2
        Loss function used to compute the residual. An ``int`` value (default
        ``2``) uses the corresponding Lp norm. A callable receives
        ``(mask * x_0, mask * y)`` and returns a scalar loss per batch element
        of shape :math:`(B,)`.
    gamma : float, default=0.0
        SDA scaling parameter. When ``gamma > 0``, applies SDA correction
        that accounts for the variance of the x0 estimate. Set to ``0`` for
        classical DPS without SDA scaling.
    sigma_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``gamma > 0``. Typically obtained from a noise
        scheduler. For example, use
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`
        for a linear-Gaussian noise schedule.
    alpha_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to signal coefficient :math:`\alpha(t)`.
        Optional; defaults to :math:`\alpha(t) = 1` if not provided. For example, use
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha`
        for a linear-Gaussian noise schedule.
    retain_graph : bool, default=False
        If ``True``, the computational graph is retained after computing
        gradients. Required when combining multiple autograd-based guidances
        in a single :class:`DPSDenoiser` — all guidances except the last
        must set this to ``True``.
    create_graph : bool, default=False
        If ``True``, a graph of the derivative is constructed, allowing
        higher-order derivatives (e.g., differentiating through the entire
        sampling process).

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`ModelConsistencyDPSGuidance` : Guidance for general observation
        operators.
    :class:`DPSDenoiser` : Combines an x0-predictor with one or more guidances.

    Examples
    --------
    **Example 1:** Sparse observations at probe locations:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import (
    ...     DataConsistencyDPSGuidance,
    ...     DPSDenoiser,
    ... )
    >>>
    >>> # Boolean mask: only observe a few probe locations
    >>> mask = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    >>> mask[:, :, 2, 3] = True  # Probe at (2, 3)
    >>> mask[:, :, 5, 6] = True  # Probe at (5, 6)
    >>> mask[:, :, 1, 7] = True  # Probe at (1, 7)
    >>> y_obs = torch.randn(1, 3, 8, 8)  # Observed values
    >>>
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.1,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9  # Toy x0 estimate (must be computed from x)
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    >>>
    >>> # Use with DPSDenoiser for complete sampling workflow
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     return (x_0 - x) / (t_bc ** 2)
    ...
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>> score = dps_denoiser(x, t)
    >>> score.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** With SDA scaling and L1 norm using noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import (
    ...     DataConsistencyDPSGuidance,
    ...     DPSDenoiser,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # Same sparse probe locations as Example 1
    >>> mask = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    >>> mask[:, :, 2, 3] = True
    >>> mask[:, :, 5, 6] = True
    >>> mask[:, :, 1, 7] = True
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>>
    >>> # Enable SDA scaling and use L1 norm for robustness
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.075,
    ...     norm=1,  # L1 norm
    ...     gamma=1.0,  # Enable SDA scaling
    ...     sigma_fn=scheduler.sigma,
    ...     alpha_fn=scheduler.alpha,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9  # Must be computed from x
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    >>>
    >>> # Use with DPSDenoiser and scheduler's x0_to_score
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> dps_denoiser = DPSDenoiser(
    ...     x0_predictor=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance,
    ... )
    >>> score = dps_denoiser(x, t)
    >>> score.shape
    torch.Size([1, 3, 8, 8])

    **Example 3:** With a custom loss function (Huber loss):

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion.guidance import DataConsistencyDPSGuidance
    >>>
    >>> # Wrap torch's Huber loss to return per-batch scalars
    >>> def huber_loss(y_pred, y_true):
    ...     per_elem = F.huber_loss(y_pred, y_true, reduction="none")
    ...     return per_elem.reshape(y_pred.shape[0], -1).sum(dim=1)
    ...
    >>> mask = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    >>> mask[:, :, 2, 3] = True
    >>> mask[:, :, 5, 6] = True
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>>
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.1,
    ...     norm=huber_loss,  # Custom loss function
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        mask: Bool[Tensor, " B *dims"],
        y: Float[Tensor, " B *dims"],
        std_y: float,
        norm: int
        | Callable[
            [Float[Tensor, "B *dims"], Float[Tensor, "B *dims"]],  # noqa: F821
            Float[Tensor, " B"],
        ] = 2,
        gamma: float = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        if gamma > 0 and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when gamma > 0")
        self.mask = mask.float()
        self.y = y
        self.std_y = std_y
        self.norm = norm
        self.gamma = gamma
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state :math:`\mathbf{x}_t`, of shape :math:`(B, *)`.
            Must have ``requires_grad=True`` and be part of a computational
            graph connecting to ``x_0``. Its ``dtype`` and ``device``
            determine those of all internal computations.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state :math:`\hat{\mathbf{x}}_0
            (\mathbf{x}_t, t)`, with same shape as ``x``. Must be computed
            from ``x`` via an x0-predictor to allow gradient backpropagation.

        Returns
        -------
        Tensor
            Likelihood score guidance term of same shape as ``x``.
        """
        # Ensure stored tensors match x's dtype and device
        mask = self.mask.to(dtype=x.dtype, device=x.device)
        y = self.y.to(dtype=x.dtype, device=x.device)

        with torch.enable_grad():
            # Compute masked predicted and observed values
            y_pred = mask * x_0
            y_true = mask * y

            # Compute loss
            if callable(self.norm):
                loss = self.norm(y_pred, y_true)
            else:
                residual = (y_pred - y_true).reshape(x_0.shape[0], -1)
                loss = residual.abs().pow(self.norm).sum(dim=1)

            # Compute gradient of loss w.r.t. x (backprop through x_0)
            grad_x = torch.autograd.grad(
                outputs=loss.sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )[0]

        # Compute scaling factor
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
        sigma_t = self.sigma_fn(t_bc)
        alpha_t = self.alpha_fn(t_bc)
        variance = self.std_y**2 + self.gamma * (sigma_t**2) / (alpha_t**2)

        return -grad_x / (2 * variance)
