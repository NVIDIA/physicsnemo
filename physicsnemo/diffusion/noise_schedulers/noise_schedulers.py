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

"""Noise schedulers for diffusion models."""

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser, Predictor


@runtime_checkable
class NoiseScheduler(Protocol):
    r"""
    Protocol defining the minimal interface for noise schedulers.

    A noise scheduler defines methods for training (adding noise, sampling
    diffusion time) and for sampling (generating diffusion time-steps,
    initializing latent state, obtaining a denoiser). This interface is generic
    and does not assume any specific form of noise schedule.

    Any object that implements this interface can be used with the diffusion
    training and sampling utilities.

    **Training methods:**

    - :meth:`sample_time`: Sample diffusion time values for training
    - :meth:`add_noise`: Add noise to clean data at given diffusion time
    - :meth:`loss_weight`: Compute per-sample loss weight for training

    **Sampling methods:**

    - :meth:`timesteps`: Generate discrete time-steps for sampling
    - :meth:`init_latents`: Initialize noisy latent state :math:`\mathbf{x}_N`
    - :meth:`get_denoiser`: Convert a predictor (e.g. model that predicts
         clean, data, score, etc.) to a sampling-compatible denoiser

    See Also
    --------
    :class:`LinearGaussianNoiseScheduler` : base abstract class for
        linear-Gaussian schedules. Implements the NoiseScheduler protocol.
    :func:`~physicsnemo.diffusion.samplers.sample` : sampling function for
        generating data samples from a diffusion model.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import NoiseScheduler
    >>>
    >>> class MyScheduler:
    ...     def sample_time(self, N, device=None, dtype=None):
    ...         return torch.rand(N, device=device, dtype=dtype)
    ...     def add_noise(self, x0, time):
    ...         return x0 + time.view(-1, 1) * torch.randn_like(x0)
    ...     def timesteps(self, num_steps, device=None, dtype=None):
    ...         return torch.linspace(1, 0, num_steps + 1, device=device)
    ...     def init_latents(self, spatial_shape, tN, device=None, dtype=None):
    ...         return torch.randn(tN.shape[0], *spatial_shape, device=device)
    ...     def get_denoiser(self, x0_predictor=None, score_predictor=None, **kwargs):
    ...         def denoiser(x, t):
    ...             if x0_predictor is not None:
    ...                 return (x - x0_predictor(x, t)) / (t.view(-1, 1))
    ...             elif score_predictor is not None:
    ...                 return -score_predictor(x, t) * t.view(-1, 1)
    ...         return denoiser
    ...     def loss_weight(self, t):
    ...         return 1 / t**2
    ...
    >>> scheduler = MyScheduler()
    >>> isinstance(scheduler, NoiseScheduler)
    True
    """

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion time values for training.

        Used in training to sample random diffusion times, typically in the
        denoising score matching loss.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        ...

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Add noise to clean data at the given diffusion times.

        Used in training to create noisy samples from clean data.

        Parameters
        ----------
        x0 : Tensor
            Clean latent state of shape :math:`(B, *)`.
        time : Tensor
            Diffusion time values of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Noisy latent state of shape :math:`(B, *)`.
        """
        ...

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate discrete time-steps for sampling.

        Used in sampling to produce the sequence of diffusion times.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Time-steps tensor of shape :math:`(N + 1,)` in decreasing order,
            with the last element being 0.
        """
        ...

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""
        Initialize the noisy latent state :math:`\mathbf{x}_N` for sampling.

        Used in sampling to generate the initial condition at diffusion time
        ``tN``.

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g., ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`. Determines the noise
            level for the initial latent state.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Initial noisy latent state of shape :math:`(B, *spatial\_shape)`.
        """
        ...

    def get_denoiser(
        self,
        **kwargs: Any,
    ) -> Denoiser:
        r"""
        Factory that converts a predictor into a denoiser for sampling.

        Used in sampling to transform a :class:`Predictor` (e.g., x0-predictor,
        score-predictor) into a :class:`Denoiser` that returns the
        update term compatible with the solver. The exact transformation
        depends on the noise scheduler implementation.

        Parameters
        ----------
        **kwargs : Any
            Implementation-specific keyword arguments. Concrete
            implementations typically accept keyword-only predictor arguments
            (e.g., ``score_predictor``, ``x0_predictor``). See concrete classes
            docstrings for details (e.g.
            :meth:`LinearGaussianNoiseScheduler.get_denoiser`).

        Returns
        -------
        Denoiser
            A callable that implements the
            :class:`~physicsnemo.diffusion.Denoiser` interface, for use
            with solvers and the
            :func:`~physicsnemo.diffusion.samplers.sample` function.
        """
        ...

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute loss weight for denoising score matching training.

        Used in training to weight the per-sample loss in
        :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`, same device and dtype
            as ``t``.
        """
        ...


class LinearGaussianNoiseScheduler(ABC, NoiseScheduler):
    r"""
    Abstract base class for linear-Gaussian noise schedules.

    It implements the :class:`NoiseScheduler` interface and it can be
    subclassed to define custom linear-Gaussian noise schedules of the form:

    .. math::
        \mathbf{x}(t) = \alpha(t) \mathbf{x}_0
        + \sigma(t) \boldsymbol{\epsilon}

    where :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})` is
    standard Gaussian noise, :math:`\alpha(t)` is the signal coefficient, and
    :math:`\sigma(t)` is the noise level.

    **Training:**

    The :meth:`add_noise` method implements the forward diffusion process using
    the formula above. The :meth:`sample_time` method samples diffusion times.

    **Sampling:**

    For ODE-based sampling, the reverse process follows the probability flow
    ODE:

    .. math::
        \frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)
        - \frac{1}{2} g^2(\mathbf{x}, t) \nabla_{\mathbf{x}} \log p(\mathbf{x})

    For SDE-based sampling:

    .. math::
        d\mathbf{x} = \left[ f(\mathbf{x}, t)
        - g^2(\mathbf{x}, t) \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right] dt
        + g(\mathbf{x}, t) d\mathbf{W}

    The :meth:`get_denoiser` factory converts a predictor (either a
    score-predictor or an x0-predictor) into the appropriate ODE/SDE
    right-hand side.

    **Abstract methods (must be implemented by subclasses):**

    - :meth:`sigma`: Map time to noise level :math:`\sigma(t)`
    - :meth:`sigma_inv`: Map noise level back to time
    - :meth:`sigma_dot`: Time derivative :math:`\dot{\sigma}(t)`
    - :meth:`alpha`: Compute the signal coefficient :math:`\alpha(t)`
    - :meth:`alpha_dot`: Time derivative :math:`\dot{\alpha}(t)`
    - :meth:`timesteps`: Generate discrete time-steps for sampling
    - :meth:`sample_time`: Sample diffusion times for training
    - :meth:`loss_weight`: Compute loss weight for training

    **Concrete methods (have default implementations, but can be overridden for
    custom behavior):**

    - :meth:`drift`: Drift term :math:`f(\mathbf{x}, t)` for ODE/SDE
    - :meth:`diffusion`: Squared diffusion term :math:`g^2(\mathbf{x}, t)`
    - :meth:`x0_to_score`: Convert x0-prediction to score
    - :meth:`score_to_x0`: Convert score to x0-prediction
    - :meth:`add_noise`: Add noise to clean data (training)
    - :meth:`init_latents`: Initialize latent state (sampling)
    - :meth:`get_denoiser`: Get ODE/SDE RHS (sampling)

    Examples
    --------
    **Example 1:** A minimal EDM-like noise schedule. Only the abstract methods
    need to be implemented since defaults work for EDM:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     LinearGaussianNoiseScheduler,
    ... )
    >>>
    >>> class SimpleEDMScheduler(LinearGaussianNoiseScheduler):
    ...     def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ...         self.sigma_min = sigma_min
    ...         self.sigma_max = sigma_max
    ...         self.rho = rho
    ...
    ...     def sigma(self, t): return t
    ...     def sigma_inv(self, sigma): return sigma
    ...     def sigma_dot(self, t): return torch.ones_like(t)
    ...     def alpha(self, t): return torch.ones_like(t)
    ...     def alpha_dot(self, t): return torch.zeros_like(t)
    ...
    ...     def timesteps(self, num_steps, *, device=None, dtype=None):
    ...         i = torch.arange(num_steps, device=device, dtype=dtype)
    ...         smax_rho = self.sigma_max**(1/self.rho)
    ...         smin_rho = self.sigma_min**(1/self.rho)
    ...         frac = i/(num_steps-1)
    ...         t = (smax_rho + frac * (smin_rho - smax_rho))**self.rho
    ...         return torch.cat([t, torch.zeros(1, device=device)])
    ...
    ...     def sample_time(self, N, *, device=None, dtype=None):
    ...         u = torch.rand(N, device=device, dtype=dtype)
    ...         return self.sigma_min * (self.sigma_max/self.sigma_min)**u
    ...     def loss_weight(self, t):
    ...         return 1 / t**2
    ...
    >>> scheduler = SimpleEDMScheduler()
    >>> t_steps = scheduler.timesteps(10)
    >>> t_steps.shape
    torch.Size([11])

    **Example 2:** Customizing behavior by overriding concrete methods. This
    shows how to override the drift term for a custom diffusion process:

    >>> class CustomDriftScheduler(SimpleEDMScheduler):
    ...     def drift(self, x, t):
    ...         # Custom drift: f(x, t) = -0.5 * x (Ornstein-Uhlenbeck style)
    ...         return -0.5 * x
    ...
    >>> custom = CustomDriftScheduler()
    >>>
    >>> # The custom drift is used internally by get_denoiser
    >>> score_pred = lambda x, t: -x / (1 + t.view(-1, 1)**2)  # Toy score predictor
    >>> denoiser = custom.get_denoiser(score_predictor=score_pred)
    >>> x = torch.randn(2, 4)
    >>> t = torch.tensor([1.0, 1.0])
    >>> out = denoiser(x, t)  # Uses custom drift in ODE RHS computation
    >>> out.shape
    torch.Size([2, 4])

    """

    @abstractmethod
    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Map diffusion time to noise level :math:`\sigma(t)`.

        Used in both training and sampling.

        Parameters
        ----------
        t : Tensor
            Diffusion time tensor of any shape.

        Returns
        -------
        Tensor
            Noise coefficient :math:`\sigma(t)` with same shape as ``t``.
        """
        ...

    @abstractmethod
    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Map noise level back to diffusion time.

        Used in both training and sampling.

        Parameters
        ----------
        sigma : Tensor
            Noise level tensor of any shape.

        Returns
        -------
        Tensor
            Diffusion time with same shape as ``sigma``.
        """
        ...

    @abstractmethod
    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Compute time derivative of noise level :math:`\dot{\sigma}(t)`.

        Used in sampling.

        Parameters
        ----------
        t : Tensor
            Diffusion time tensor of any shape.

        Returns
        -------
        Tensor
            Time derivative :math:`\dot{\sigma}(t)` with same shape as ``t``.
        """
        ...

    @abstractmethod
    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Compute the signal coefficient :math:`\alpha(t)`.

        Used in both training and sampling.

        Parameters
        ----------
        t : Tensor
            Diffusion time tensor of any shape.

        Returns
        -------
        Tensor
            Signal coefficient :math:`\alpha(t)` with same shape as ``t``.
        """
        ...

    @abstractmethod
    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Compute time derivative of signal coefficient :math:`\dot{\alpha}(t)`.

        Used in sampling.

        Parameters
        ----------
        t : Tensor
            Diffusion time tensor of any shape.

        Returns
        -------
        Tensor
            Time derivative :math:`\dot{\alpha}(t)` with same shape as ``t``.
        """
        ...

    @abstractmethod
    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate discrete time-steps for sampling.

        Used in sampling to produce the sequence of diffusion times. Returns
        a tensor of shape :math:`(N + 1,)` in decreasing order, with the last
        element being 0.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        ...

    @abstractmethod
    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion time values for training.

        Used in training to sample random diffusion times for the denoising
        score matching loss.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        ...

    @abstractmethod
    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute loss weight for denoising score matching training.

        Used in training to weight the per-sample loss in
        :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`. The loss
        weight is designed for training an x0-predictor (clean data
        predictor).
        For training a score-predictor, additionally provide a
        ``score_to_x0_fn`` callback to
        :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`, same device and dtype
            as ``t``.
        """
        ...

    def drift(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute drift term :math:`f(\mathbf{x}, t)` for ODE/SDE sampling.

        Used by :meth:`get_denoiser` to build the ODE/SDE right-hand side.

        By default: :math:`f(\mathbf{x}, t) = \frac{\dot{\alpha}(t)}{\alpha(t)}
        \mathbf{x}`.

        This method can be overridden to implement different drift terms.

        Parameters
        ----------
        x : Tensor
            Latent state of shape :math:`(B, *)`.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Drift term with same shape as ``x``.
        """
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        return (alpha_dot_t_bc / alpha_t_bc) * x

    def diffusion(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *_"]:
        r"""
        Compute squared diffusion term :math:`g^2(\mathbf{x}, t)`.

        Used by :meth:`get_denoiser` to build the ODE/SDE right-hand side.

        By default: :math:`g^2 = 2 \dot{\sigma} \sigma - 2 \frac{\dot{\alpha}}
        {\alpha} \sigma^2`.
        This method can be overridden to implement different diffusion terms.

        Parameters
        ----------
        x : Tensor
            Latent state of shape :math:`(B, *)`.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Squared diffusion term, broadcastable to shape of ``x``.
        """
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
        sigma_t_bc = self.sigma(t_bc)
        sigma_dot_t_bc = self.sigma_dot(t_bc)
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        g_sq_bc = (
            2 * sigma_dot_t_bc * sigma_t_bc
            - 2 * (alpha_dot_t_bc / alpha_t_bc) * sigma_t_bc**2
        )
        return g_sq_bc

    def x0_to_score(
        self,
        x0: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert x0-predictor output to score.

        This conversion is done automatically by :meth:`get_denoiser` when
        ``x0_predictor`` is provided, but can also be called manually.

        The score is: :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
        = \frac{\alpha(t) \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma^2(t)}`.

        This is a helper method that usually does not need to be overridden in
        subclasses.

        Parameters
        ----------
        x0 : Tensor
            Predicted clean data :math:`\hat{\mathbf{x}}_0` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` of shape :math:`(B, *)`.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Score with same shape as ``x0``.

        Examples
        --------
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an x0-predictor, wrap it for manual conversion
        >>> # (done automatically by get_denoiser):
        >>> def x0_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x / (1 + t_bc**2)
        >>> def score_predictor(x, t):
        ...     x0_pred = x0_predictor(x, t)
        ...     return scheduler.x0_to_score(x0_pred, x, t)
        >>> # Or simply: scheduler.get_denoiser(x0_predictor=x0_predictor)
        """
        t_bc = t.reshape(-1, *([1] * (x0.ndim - 1)))
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        return (alpha_t_bc * x0 - x_t) / (sigma_t_bc**2)

    def score_to_x0(
        self,
        score: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert score to x0-prediction.

        This is the inverse of :meth:`x0_to_score`. Given a score
        prediction :math:`s(\mathbf{x}_t, t)` and
        the noisy state :math:`\mathbf{x}_t`, recover the corresponding
        :math:`\hat{\mathbf{x}}_0` estimate:

        .. math::
            \hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t
            + \sigma^2(t) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)}
            {\alpha(t)}

        A common use case is with
        :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss` to train a
        score-predictor instead of an x0-predictor: pass this method as the
        ``score_to_x0_fn`` argument with ``prediction_type="score"``.

        This is a helper method that usually does not need to be overridden
        in subclasses.

        Parameters
        ----------
        score : Tensor
            Predicted score :math:`s(\mathbf{x}_t, t)` of shape :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``score``.
        t : Tensor
            Diffusion time with shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Estimated clean data :math:`\hat{\mathbf{x}}_0` with same shape
            as ``score``.

        Examples
        --------
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a score-predictor, convert to x0 for DSM loss:
        >>> def score_predictor(x, t):
        ...     return -x / (1 + t.view(-1, *([1] * (x.ndim - 1)))**2)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> score = score_predictor(x_t, t)
        >>> x0_est = scheduler.score_to_x0(score, x_t, t)
        >>> x0_est.shape
        torch.Size([2, 4])
        """
        t_bc = t.reshape(-1, *([1] * (score.ndim - 1)))
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        return (x_t + sigma_t_bc**2 * score) / alpha_t_bc

    def get_denoiser(
        self,
        *,
        score_predictor: Predictor | None = None,
        x0_predictor: Predictor | None = None,
        denoising_type: Literal["ode", "sde"] = "ode",
        **kwargs: Any,
    ) -> Denoiser:
        r"""
        Factory that converts a predictor to a denoiser for sampling.

        Accepts either a **score-predictor** or an **x0-predictor** (exactly
        one must be provided). The returned denoiser computes the right-hand
        side of the reverse ODE or SDE.

        For ODE (``denoising_type="ode"``):

        .. math::
            \frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2} g^2(t)
            s(\mathbf{x}, t)

        For SDE (``denoising_type="sde"``):

        .. math::
            d\mathbf{x} = \left[ f(\mathbf{x}, t) - g^2(t) s(\mathbf{x}, t)
            \right] dt + g(t) d\mathbf{W}

        where :math:`s(\mathbf{x}, t)` is the score. When an x0-predictor is
        provided, the score is computed internally via :meth:`x0_to_score`.
        When a score-predictor is provided, it is used directly.
        *Note:* As usually done in SDE integration, the stochastic term
        :math:`g(t) d\mathbf{W}` is handled by the solver, not returned by the
        denoiser itself.

        Parameters
        ----------
        score_predictor : Predictor, optional
            A score-predictor that takes ``(x_t, t)`` and returns a score
            (e.g. :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)`). Can be
            unconditional, conditional, guidance-augmented, etc. Mutually
            exclusive with ``x0_predictor``.
        x0_predictor : Predictor, optional
            An x0-predictor that takes ``(x_t, t)`` and returns an estimate
            of clean data :math:`\hat{\mathbf{x}}_0`. The score is computed
            internally via :meth:`x0_to_score`. Mutually exclusive with
            ``score_predictor``.
        denoising_type : {"ode", "sde"}, default="ode"
            Type of reverse process. Use ``"ode"`` for deterministic sampling,
            ``"sde"`` for stochastic sampling.
        **kwargs : Any
            Ignored.

        Returns
        -------
        Denoiser
            A denoiser computing the RHS of the reverse ODE/SDE. Implements
            the :class:`~physicsnemo.diffusion.Denoiser` interface.

        Raises
        ------
        ValueError
            If both or neither ``score_predictor`` and ``x0_predictor`` are
            provided.

        Examples
        --------
        Generate ODE RHS from a score-predictor:

        >>> import torch
        >>> scheduler = EDMNoiseScheduler()
        >>> score_pred = lambda x, t: -x / t.view(-1, 1, 1, 1)**2  # Toy score-predictor
        >>> denoiser = scheduler.get_denoiser(
        ...     score_predictor=score_pred, denoising_type="ode")
        >>> x = torch.randn(2, 3, 8, 8)
        >>> t = torch.tensor([1.0, 1.0])
        >>> dx_dt = denoiser(x, t)  # Returns ODE RHS for sampling
        >>> dx_dt.shape
        torch.Size([2, 3, 8, 8])

        Generate ODE RHS from an x0-predictor (score conversion is done internally):

        >>> x0_pred = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy x0-predictor
        >>> denoiser = scheduler.get_denoiser(
        ...     x0_predictor=x0_pred, denoising_type="ode")
        >>> dx_dt = denoiser(x, t)  # Returns ODE RHS for sampling
        >>> dx_dt.shape
        torch.Size([2, 3, 8, 8])
        """
        # Validate: exactly one of score_predictor or x0_predictor
        if (score_predictor is None) == (x0_predictor is None):
            raise ValueError(
                "Exactly one of 'score_predictor' or 'x0_predictor' "
                "must be provided, not both or neither."
            )

        # Capture methods as local variables to avoid referencing self
        drift = self.drift
        diffusion = self.diffusion
        # Build the score function
        if x0_predictor is not None:
            x0_to_score = self.x0_to_score

            def _score(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                x0 = x0_predictor(x, t)
                return x0_to_score(x0, x, t)

            score_fn = _score
        else:
            score_fn = score_predictor

        if denoising_type == "ode":

            def ode_denoiser(
                x: Float[Tensor, "B *dims"],  # noqa: F821
                t: Float[Tensor, "B"],  # noqa: F821
            ) -> Float[Tensor, " B *dims"]:
                score = score_fn(x, t)
                f = drift(x, t)
                g_sq_bc = diffusion(x, t)
                dx_dt = f - 0.5 * g_sq_bc * score
                return dx_dt

            return ode_denoiser

        elif denoising_type == "sde":

            def sde_denoiser(
                x: Float[Tensor, "B *dims"],  # noqa: F821
                t: Float[Tensor, "B"],  # noqa: F821
            ) -> Float[Tensor, " B *dims"]:
                score = score_fn(x, t)
                f = drift(x, t)
                g_sq_bc = diffusion(x, t)
                # Deterministic part of the SDE drift
                # Note: stochastic term g(t)*dW is handled by the solver
                dx_dt = f - g_sq_bc * score
                return dx_dt

            return sde_denoiser

        else:
            raise ValueError(
                f"denoising_type must be 'ode' or 'sde', got '{denoising_type}'"
            )

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Add noise to clean data at the given diffusion times.

        Used in training to create noisy samples from clean data. Implements:

        .. math::
            \mathbf{x}(t) = \alpha(t) \mathbf{x}_0
            + \sigma(t) \boldsymbol{\epsilon}

        Usually does not need to be overridden in subclasses: overriding the
        :meth:`alpha` and :meth:`sigma` methods is sufficient for most use
        cases.


        Parameters
        ----------
        x0 : Tensor
            Clean latent state of shape :math:`(B, *)`.
        time : Tensor
            Diffusion time values of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Noisy latent state of shape :math:`(B, *)`.
        """
        t_bc = time.reshape(-1, *([1] * (x0.ndim - 1)))
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        noise = torch.randn_like(x0)
        return alpha_t_bc * x0 + sigma_t_bc * noise

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""
        Initialize the noisy latent state :math:`\mathbf{x}_N` for sampling.

        Generates:

        .. math::
            \mathbf{x}_N = \sigma(t_N) \cdot \boldsymbol{\epsilon}

        where :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`.

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g., ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Initial noisy latent of shape :math:`(B, *spatial\_shape)`.
        """
        B = tN.shape[0]
        noise = torch.randn(B, *spatial_shape, device=device, dtype=dtype)
        tN_bc = tN.reshape(-1, *([1] * len(spatial_shape)))
        sigma_tN_bc = self.sigma(tN_bc)
        return sigma_tN_bc * noise


# =============================================================================
# Concrete noise schedule implementations
# =============================================================================


class EDMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    EDM noise scheduler.

    The EDM formulation uses :math:`\alpha(t) = 1` (no signal attenuation)
    and :math:`\sigma(t) = t` (identity mapping between time and noise level).

    **Sampling time-steps** are computed with polynomial spacing:

    .. math::
        t_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}
        \left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)
        \right)^{\rho}

    **Training times** are sampled from a log-normal distribution with
    mean :math:`P_{\text{mean}}` and standard deviation :math:`P_{\text{std}}`.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for sampling time-steps, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for sampling time-steps, by default 80.
    rho : float, optional
        Exponent controlling time-step spacing. Larger values concentrate more
        steps at lower noise levels (better for fine details). By default 7.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
        Used by :meth:`loss_weight` to compute the per-sample loss weight.
    P_mean : float, optional
        Mean of the log-normal distribution used to sample training times,
        by default -1.2.
    P_std : float, optional
        Standard deviation of the log-normal distribution used to sample
        training times, by default 1.2.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    Basic training and sampling workflow using the EDM noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0, rho=7)
    >>>
    >>> # Training: sample times and add noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Create noisy samples
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Initial noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert x0-predictor to denoiser for sampling
    >>> x0_predictor = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`\sigma(t) = t`."""
        return t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`t = \sigma`."""
        return sigma

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant derivative: :math:`\dot{\sigma}(t) = 1`."""
        return torch.ones_like(t)

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant signal coefficient: :math:`\alpha(t) = 1`."""
        return torch.ones_like(t)

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Zero derivative: :math:`\dot{\alpha}(t) = 0`."""
        return torch.zeros_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate EDM time-steps with polynomial spacing.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)` where :math:`N` is
            ``num_steps``.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        smax_inv_rho = self.sigma_max ** (1 / self.rho)
        smin_inv_rho = self.sigma_min ** (1 / self.rho)
        frac = step_indices / (num_steps - 1)
        interp = smax_inv_rho + frac * (smin_inv_rho - smax_inv_rho)
        t_steps = interp**self.rho
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times from a log-normal distribution:
        :math:`\ln(\sigma) \sim \mathcal{N}(P_{\text{mean}},
        P_{\text{std}}^2)`.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        rnd_normal = torch.randn(N, device=device, dtype=dtype)
        return (rnd_normal * self.P_std + self.P_mean).exp()

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute EDM loss weight.

        .. math::
            w(t) = \frac{\sigma(t)^2 + \sigma_{\text{data}}^2}
            {\left(\sigma(t) \cdot \sigma_{\text{data}}\right)^2}

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        sd = self.sigma_data
        sigma_t = self.sigma(t)
        return (sigma_t**2 + sd**2) / (sigma_t * sd) ** 2


class VENoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Variance Exploding (VE) noise scheduler.

    Implements the VE formulation with :math:`\sigma(t) = \sqrt{t}` and
    :math:`\alpha(t) = 1` (no signal attenuation).

    **Sampling time-steps** use geometric spacing in :math:`\sigma^2` space:

    .. math::
        \sigma_i^2 = \sigma_{\max}^2 \cdot
        \left(\frac{\sigma_{\min}^2}{\sigma_{\max}^2}\right)^{i/(N-1)}

    **Training times** are sampled log-uniformly between ``sigma_min`` and
    ``sigma_max``, then mapped to time via :math:`t = \sigma^2`.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.02.
    sigma_max : float, optional
        Maximum noise level, by default 100.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    Basic training and sampling workflow using the VE noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import VENoiseScheduler
    >>>
    >>> scheduler = VENoiseScheduler(sigma_min=0.02, sigma_max=100.0)
    >>>
    >>> # Training: sample times and add noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Create noisy samples
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Initial noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert x0-predictor to denoiser for sampling
    >>> x0_predictor = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""VE noise coefficient: :math:`\sigma(t) = \sqrt{t}`."""
        return t.sqrt()

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Inverse VE mapping: :math:`t = \sigma^2`."""
        return sigma**2

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Time derivative: :math:`\dot{\sigma}(t) = 1/(2\sqrt{t})`."""
        return 0.5 / t.sqrt()

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant signal coefficient: :math:`\alpha(t) = 1`."""
        return torch.ones_like(t)

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Zero derivative: :math:`\dot{\alpha}(t) = 0`."""
        return torch.zeros_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate VE time-steps with geometric spacing in :math:`\sigma^2`.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        ratio = self.sigma_min**2 / self.sigma_max**2
        exponent = step_indices / (num_steps - 1)
        t_steps = (self.sigma_max**2) * (ratio**exponent)
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times log-uniformly in sigma space, mapped to time.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        sigma = self.sigma_min * torch.exp(u * log_ratio)
        return self.sigma_inv(sigma)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute VE loss weight: :math:`w(t) = 1 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.VEPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        return 1 / self.sigma(t) ** 2


class IDDPMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Improved DDPM (iDDPM) noise scheduler.

    Uses identity mappings :math:`\sigma(t) = t` and :math:`\alpha(t) = 1`.
    The key feature is a precomputed noise level schedule derived from a
    cosine schedule, providing improved sample quality in comparison to
    original DDPM.

    **Sampling time-steps** are selected from a precomputed schedule of
    :math:`M` discrete noise levels, subsampled to ``num_steps``.

    **Training times** are sampled uniformly from the precomputed schedule.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for filtering, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for filtering, by default 81.
    C_1 : float, optional
        Clipping threshold for alpha ratio, by default 0.001.
    C_2 : float, optional
        Cosine schedule parameter, by default 0.008.
    M : int, optional
        Number of precomputed discretization steps, by default 1000.

    Note
    ----
    Reference: `Improved Denoising Diffusion Probabilistic Models
    <https://arxiv.org/abs/2102.09672>`_

    Examples
    --------
    Basic training and sampling workflow using the iDDPM noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import IDDPMNoiseScheduler
    >>>
    >>> scheduler = IDDPMNoiseScheduler(C_1=0.001, C_2=0.008, M=1000)
    >>>
    >>> # Training: sample times and add noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Create noisy samples
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Initial noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert x0-predictor to denoiser for sampling
    >>> x0_predictor = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 81.0,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M

        # Precompute the noise level schedule u_j, j = 0, ..., M
        self._u = self._compute_u_schedule()

    def _compute_u_schedule(self) -> Tensor:
        """Precompute the iDDPM noise level schedule."""
        u = torch.zeros(self.M + 1)
        for j in range(self.M, 0, -1):
            angle_j = 0.5 * math.pi * j / self.M / (self.C_2 + 1)
            angle_jm1 = 0.5 * math.pi * (j - 1) / self.M / (self.C_2 + 1)
            alpha_bar_j = math.sin(angle_j) ** 2
            alpha_bar_jm1 = math.sin(angle_jm1) ** 2
            alpha_ratio = alpha_bar_jm1 / alpha_bar_j
            val = (u[j] ** 2 + 1) / max(alpha_ratio, self.C_1) - 1
            u[j - 1] = val.sqrt()
        return u

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""For iDDPM, :math:`\sigma(t) = t` (identity mapping)."""
        return t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""For iDDPM, :math:`t = \sigma` (identity mapping)."""
        return sigma

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant derivative: :math:`\dot{\sigma}(t) = 1`."""
        return torch.ones_like(t)

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant signal coefficient: :math:`\alpha(t) = 1`."""
        return torch.ones_like(t)

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Zero derivative: :math:`\dot{\alpha}(t) = 0`."""
        return torch.zeros_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate iDDPM time-steps from precomputed schedule.

        Subsamples ``num_steps`` values from the precomputed schedule of
        :math:`M` noise levels.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        u = self._u.to(device=device, dtype=dtype)
        # Filter to valid sigma range
        in_range = torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)
        u_filtered = u[in_range]

        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        scale = (len(u_filtered) - 1) / (num_steps - 1)
        indices = (scale * step_indices).round().to(torch.int64)
        sigma_steps = u_filtered[indices]

        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([sigma_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times uniformly from precomputed schedule.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = self._u.to(device=device, dtype=dtype)
        in_range = torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)
        u_filtered = u[in_range]
        # Sample random indices
        indices = torch.randint(0, len(u_filtered), (N,), device=device)
        return u_filtered[indices]

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute iDDPM loss weight: :math:`w(t) = 1 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.IDDPMPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        return 1 / self.sigma(t) ** 2


class VPNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Variance Preserving (VP) noise scheduler.

    Implements the VP formulation where the total variance is preserved:
    :math:`\alpha(t)^2 + \sigma(t)^2 = 1`. This is based on a linear beta
    schedule: :math:`\beta(t) = \beta_{\min} + t \cdot \beta_d`.

    The noise and signal coefficients are:

    .. math::
        \alpha(t) = \exp\left(-\frac{1}{2}
        \left(\frac{\beta_d}{2} t^2 + \beta_{\min} t\right)\right)

    .. math::
        \sigma(t) = \sqrt{1 - \alpha(t)^2}
        = \sqrt{1 - \exp\left(-\frac{\beta_d}{2} t^2
        - \beta_{\min} t\right)}

    **Sampling time-steps** are linearly spaced from ``t_max`` (usually 1) to
    ``epsilon_s`` (small positive value to avoid singularities).

    **Training times** are sampled uniformly between ``epsilon_s`` and
    ``t_max``.

    Parameters
    ----------
    beta_min : float, optional
        Minimum beta value for the linear schedule, by default 0.1.
    beta_d : float, optional
        Beta slope (delta) for the linear schedule, by default 19.1.
    epsilon_s : float, optional
        Small positive value for minimum time, by default 1e-3.
    t_max : float, optional
        Maximum diffusion time, by default 1.0.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    Basic training and sampling workflow using the VP noise scheduler:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>>
    >>> scheduler = VPNoiseScheduler(beta_min=0.1, beta_d=19.1)
    >>>
    >>> # Training: sample times and add noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Create noisy samples
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)  # Initial time for batch of 4
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Initial noise
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Convert x0-predictor to denoiser for sampling
    >>> x0_predictor = lambda x, t: x * 0.9  # Toy x0-predictor
    >>> denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    >>> denoiser(xN, tN).shape  # ODE RHS for sampling
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_d: float = 19.1,
        epsilon_s: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.epsilon_s = epsilon_s
        self.t_max = t_max

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Signal coefficient: :math:`\alpha(t) = \exp(-a(t)/2)`."""
        return torch.exp(-0.5 * (0.5 * self.beta_d * t**2 + self.beta_min * t))

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Derivative: :math:`\dot{\alpha}(t) = -\frac{\beta(t)}{2} \alpha(t)`."""
        beta_t = self.beta_min + self.beta_d * t
        return -0.5 * beta_t * self.alpha(t)

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Noise level: :math:`\sigma(t) = \sqrt{1 - \alpha(t)^2}`."""
        alpha_sq = self.alpha(t) ** 2
        return torch.sqrt(1 - alpha_sq)

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Derivative: :math:`\dot{\sigma}(t) = -\alpha(t) \dot{\alpha}(t) / \sigma(t)`."""  # noqa: E501
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        alpha_dot_t = self.alpha_dot(t)
        # d/dt sqrt(1 - alpha^2) = -alpha * alpha_dot / sqrt(1 - alpha^2)
        return -alpha_t * alpha_dot_t / sigma_t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""
        Inverse mapping from sigma to time.

        Solves: :math:`\sigma^2 = 1 - \exp(-a(t))` for :math:`t`.
        """
        # sigma^2 = 1 - exp(-a) => a = -log(1 - sigma^2)
        # a = beta_d/2 * t^2 + beta_min * t
        # Quadratic: beta_d * t^2 + 2*beta_min * t + 2*log(1-sigma^2) = 0
        log_term = torch.log(1 - sigma**2 + 1e-8)  # small eps for stability
        discriminant = self.beta_min**2 - 2 * self.beta_d * log_term
        return (-self.beta_min + torch.sqrt(discriminant.clamp(min=0))) / self.beta_d

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate VP time-steps with linear spacing.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        # Linear spacing from t_max to epsilon_s
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        frac = step_indices / (num_steps - 1)
        t_steps = self.t_max + frac * (self.epsilon_s - self.t_max)
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times uniformly in :math:`[\epsilon_s, t_{max}]`.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        u = torch.rand(N, device=device, dtype=dtype)
        return self.epsilon_s + u * (self.t_max - self.epsilon_s)

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute VP loss weight: :math:`w(t) = \alpha(t)^2 / \sigma(t)^2`.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.VPPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        return (self.alpha(t) / self.sigma(t)) ** 2


class StudentTEDMNoiseScheduler(LinearGaussianNoiseScheduler):
    r"""
    Student-t EDM noise scheduler for heavy-tailed diffusion models.

    This scheduler is a variant of :class:`EDMNoiseScheduler` that uses
    Student-t noise instead of Gaussian noise. It is useful for modeling
    heavy-tailed distributions and can improve sample quality for certain
    data types.

    .. important::

        Despite inheriting from :class:`LinearGaussianNoiseScheduler`, this
        scheduler is **not truly Gaussian**. It uses the same linear structure
        (identity mappings :math:`\sigma(t) = t` and :math:`\alpha(t) = 1`) but
        replaces Gaussian noise with Student-t noise. The "Linear" part of
        :class:`LinearGaussianNoiseScheduler` still applies, but the "Gaussian"
        part does not.

    This scheduler uses a non-gaussian forward process:

    .. math::
        \mathbf{x}(t) = \mathbf{x}_0 + \sigma(t) \mathbf{n}, \quad
        \mathbf{n} \sim \text{Student-}t(\nu)

    The marginal distribution :math:`p(\mathbf{x}_t | \mathbf{x}_0)` is
    therefore a scaled Student-t distribution, not Gaussian.

    **Comparison with EDMNoiseScheduler:**

    This scheduler shares the same time-to-noise mappings as
    :class:`EDMNoiseScheduler`.
    The only differences are in :meth:`add_noise` and :meth:`init_latents`,
    which use Student-t noise instead of Gaussian noise.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level for sampling time-steps, by default 0.002.
    sigma_max : float, optional
        Maximum noise level for sampling time-steps, by default 80.
    rho : float, optional
        Exponent controlling time-step spacing. Larger values concentrate more
        steps at lower noise levels (better for fine details). By default 7.
    nu : int, optional
        Degrees of freedom for Student-t distribution. Must be > 2.
        As ``nu`` increases, the distribution approaches Gaussian. Lower values
        produce heavier tails. By default 10.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
        Used by :meth:`loss_weight` to compute the per-sample loss weight.
    P_mean : float, optional
        Mean of the log-normal distribution used to sample training times,
        by default -1.2.
    P_std : float, optional
        Standard deviation of the log-normal distribution used to sample
        training times, by default 1.2.

    Note
    ----
    Reference: `Heavy-Tailed Diffusion Models
    <https://arxiv.org/abs/2410.14171>`_

    Examples
    --------
    Basic training and sampling workflow with Student-t noise:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     StudentTEDMNoiseScheduler,
    ... )
    >>>
    >>> scheduler = StudentTEDMNoiseScheduler(nu=10)
    >>>
    >>> # Training: sample times and add Student-t noise
    >>> x0 = torch.randn(4, 3, 8, 8)  # Clean data
    >>> t = scheduler.sample_time(4)    # Sample diffusion times
    >>> x_t = scheduler.add_noise(x0, t)  # Adds Student-t noise
    >>> x_t.shape
    torch.Size([4, 3, 8, 8])
    >>>
    >>> # Sampling: generate timesteps and Student-t initial latents
    >>> t_steps = scheduler.timesteps(10)
    >>> tN = t_steps[0].expand(4)
    >>> xN = scheduler.init_latents((3, 8, 8), tN)  # Student-t latents
    >>> xN.shape
    torch.Size([4, 3, 8, 8])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        nu: int = 10,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ) -> None:
        if nu <= 2:
            raise ValueError(f"nu must be > 2, got {nu}")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.nu = nu
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std

    def sigma(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`\sigma(t) = t`."""
        return t

    def sigma_inv(
        self,
        sigma: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Identity mapping: :math:`t = \sigma`."""
        return sigma

    def sigma_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant derivative: :math:`\dot{\sigma}(t) = 1`."""
        return torch.ones_like(t)

    def alpha(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Constant signal coefficient: :math:`\alpha(t) = 1`."""
        return torch.ones_like(t)

    def alpha_dot(
        self,
        t: Float[Tensor, " *shape"],
    ) -> Float[Tensor, " *shape"]:
        r"""Zero derivative: :math:`\dot{\alpha}(t) = 0`."""
        return torch.zeros_like(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N+1"]:
        r"""
        Generate EDM time-steps with polynomial spacing.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)` where :math:`N` is
            ``num_steps``.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        smax_inv_rho = self.sigma_max ** (1 / self.rho)
        smin_inv_rho = self.sigma_min ** (1 / self.rho)
        frac = step_indices / (num_steps - 1)
        interp = smax_inv_rho + frac * (smin_inv_rho - smax_inv_rho)
        t_steps = interp**self.rho
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""
        Sample N diffusion times from a log-normal distribution:
        :math:`\ln(\sigma) \sim \mathcal{N}(P_{\text{mean}},
        P_{\text{std}}^2)`.

        Parameters
        ----------
        N : int
            Number of time values to sample.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sampled diffusion times of shape :math:`(N,)`.
        """
        rnd_normal = torch.randn(N, device=device, dtype=dtype)
        return (rnd_normal * self.P_std + self.P_mean).exp()

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        r"""
        Compute Student-t EDM loss weight: :math:`w(t) = \frac{\tilde{\sigma}(t)^2 + \sigma_{\text{data}}^2}
        {\left(\tilde{\sigma}(t) \cdot \sigma_{\text{data}}\right)^2}`

        where :math:`\tilde{\sigma}(t) = \sigma(t) \cdot \sqrt{\frac{\nu}{\nu -
        2}}` is the scaled noise level.

        .. important::

            This loss weight is designed for training an x0-predictor
            (clean data predictor) wrapped with
            :class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner`.
            It is not suitable for training a score-predictor, or a model
            without a pre-conditioner.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight of shape :math:`(N,)`.
        """
        sd = self.sigma_data
        sigma_t_scaled = self.sigma(t) * np.sqrt(self.nu / (self.nu - 2))
        return (sigma_t_scaled**2 + sd**2) / (sigma_t_scaled * sd) ** 2

    def _sample_student_t(
        self,
        *shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        r"""
        Sample from standard Student-t distribution.

        Student-t samples are generated as: :math:`X / \sqrt{V / \nu}` where
        :math:`X \sim \mathcal{N}(0, 1)` and :math:`V \sim \chi^2(\nu)`.

        Parameters
        ----------
        *shape : int
            Shape of the output tensor.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Student-t samples of the specified shape.
        """
        normal = torch.randn(*shape, device=device, dtype=dtype)

        nu = torch.tensor(self.nu, device=device, dtype=dtype)
        chi2_dist = torch.distributions.Chi2(df=nu)
        chi2_samples = chi2_dist.sample((shape[0], *([1] * (len(shape) - 1))))
        kappa = chi2_samples / nu

        return normal / torch.sqrt(kappa)

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Add Student-t noise to clean data at the given diffusion times.

        Unlike the Gaussian case in :class:`LinearGaussianNoiseScheduler`,
        this method uses Student-t noise:

        .. math::
            \mathbf{x}(t) = \mathbf{x}_0 + \sigma(t) \mathbf{n}, \quad
            \mathbf{n} \sim \text{Student-}t(\nu)

        Parameters
        ----------
        x0 : Tensor
            Clean latent state of shape :math:`(B, *)`.
        time : Tensor
            Diffusion time values of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Noisy latent state of shape :math:`(B, *)`.
        """
        t_bc = time.reshape(-1, *([1] * (x0.ndim - 1)))
        sigma_t_bc = self.sigma(t_bc)
        noise = self._sample_student_t(*x0.shape, device=x0.device, dtype=x0.dtype)
        return x0 + sigma_t_bc * noise

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""
        Initialize noisy latent state with Student-t noise.

        Unlike the Gaussian case in :class:`LinearGaussianNoiseScheduler`,
        this method uses Student-t noise:

        .. math::
            \mathbf{x}_N = \sigma(t_N) \cdot \mathbf{n}, \quad
            \mathbf{n} \sim \text{Student-}t(\nu)

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g., ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Initial noisy latent of shape :math:`(B, *spatial\_shape)`.
        """
        B = tN.shape[0]
        noise = self._sample_student_t(B, *spatial_shape, device=device, dtype=dtype)
        tN_bc = tN.reshape(-1, *([1] * len(spatial_shape)))
        sigma_tN_bc = self.sigma(tN_bc)
        return sigma_tN_bc * noise
