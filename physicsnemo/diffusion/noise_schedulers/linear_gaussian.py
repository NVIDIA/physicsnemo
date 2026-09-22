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

"""Abstract base class for linear-Gaussian noise schedules."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser, Predictor

from .base import NoiseScheduler


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

    The :meth:`get_denoiser` factory converts a predictor (a score-, x0-,
    epsilon-, or flow-predictor) into the appropriate ODE/SDE right-hand
    side.

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
    - :meth:`epsilon_to_score`: Convert epsilon-prediction to score
    - :meth:`score_to_epsilon`: Convert score to epsilon-prediction
    - :meth:`epsilon_to_x0`: Convert epsilon-prediction to x0-prediction
    - :meth:`x0_to_epsilon`: Convert x0-prediction to epsilon-prediction
    - :meth:`x0_to_flow`: Convert x0-prediction to flow (velocity) prediction
    - :meth:`flow_to_x0`: Convert flow prediction to x0-prediction
    - :meth:`score_to_flow`: Convert score prediction to flow prediction
    - :meth:`flow_to_score`: Convert flow prediction to score prediction
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
    ) -> Float[Tensor, " N"] | Float[Tensor, " N C"]:
        r"""
        Compute loss weight for denoising score matching training.

        Used in training to weight the per-sample loss in
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`. The loss
        weight is designed for training an x0-predictor (clean data
        predictor).
        For training a score-predictor, additionally provide a
        ``score_to_x0_fn`` callback to
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight with leading dimension :math:`N`.  Shape is
            :math:`(N,)` for scalar ``sigma_data``, or :math:`(N, C)`
            when the scheduler uses per-channel ``sigma_data`` (see
            :class:`EDMNoiseScheduler`).
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
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_bc = t.reshape(expected_shape)
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
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_bc = t.reshape(expected_shape)
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

        The :meth:`get_denoiser` method performs this conversion automatically
        when you pass ``x0_predictor``, but you can also call it manually.

        The score is: :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
        = \frac{\alpha(t) \hat{\mathbf{x}}_0 - \mathbf{x}_t}{\sigma^2(t)}`.

        This is a helper method that subclasses, in most cases, do not need to
        override.

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
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an x0-predictor, wrap it for manual conversion
        >>> # (done automatically by get_denoiser):
        >>> def x0_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x / (1 + t_bc**2)
        >>> def score_predictor(x, t):
        ...     x0_pred = x0_predictor(x, t)
        ...     return scheduler.x0_to_score(x0_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> # Or simply: scheduler.get_denoiser(x0_predictor=x0_predictor)
        """
        expected_shape = (-1,) + (1,) * (x0.ndim - 1)
        t_bc = t.reshape(expected_shape)
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
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` to train a
        score-predictor instead of an x0-predictor: pass this method as the
        ``score_to_x0_fn`` argument with ``prediction_type="score"``.

        This is a helper method that subclasses, in most cases, do not need to
        override.

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
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a score-predictor, wrap it for manual conversion
        >>> # (done automatically by MSEDSMLoss with prediction_type="score"):
        >>> def score_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return -x / (1 + t_bc**2)
        >>> def x0_predictor(x, t):
        ...     score_pred = score_predictor(x, t)
        ...     return scheduler.score_to_x0(score_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        >>> # Or simply: MSEDSMLoss(..., prediction_type="score", score_to_x0_fn=scheduler.score_to_x0)
        """
        expected_shape = (-1,) + (1,) * (score.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        return (x_t + sigma_t_bc**2 * score) / alpha_t_bc

    def epsilon_to_score(
        self,
        epsilon: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert epsilon-predictor (noise) output to score.

        The :meth:`get_denoiser` method performs this conversion automatically
        when you pass ``epsilon_predictor``, but you can also call it manually.

        The score is: :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
        = -\frac{\hat{\boldsymbol{\epsilon}}}{\sigma(t)}`.

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        epsilon : Tensor
            Predicted noise :math:`\hat{\boldsymbol{\epsilon}}` of shape
            :math:`(B, *)`.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Score with same shape as ``epsilon``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an epsilon-predictor, wrap it for manual conversion
        >>> # (done automatically by get_denoiser):
        >>> def epsilon_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x * t_bc / (1 + t_bc**2)
        >>> def score_predictor(x, t):
        ...     eps_pred = epsilon_predictor(x, t)
        ...     return scheduler.epsilon_to_score(eps_pred, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> eps_pred = epsilon_predictor(x_t, t)
        >>> eps_pred.shape
        torch.Size([2, 4])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> # Or simply: scheduler.get_denoiser(epsilon_predictor=epsilon_predictor)
        """
        expected_shape = (-1,) + (1,) * (epsilon.ndim - 1)
        t_bc = t.reshape(expected_shape)
        sigma_t_bc = self.sigma(t_bc)
        return -epsilon / sigma_t_bc

    def score_to_epsilon(
        self,
        score: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert score to epsilon (noise) prediction.

        This is the inverse of :meth:`epsilon_to_score`:

        .. math::
            \hat{\boldsymbol{\epsilon}}
            = -\sigma(t) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        score : Tensor
            Score :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)` of
            shape :math:`(B, *)`.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Epsilon with same shape as ``score``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a score-predictor, wrap it for manual conversion:
        >>> def score_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return -x / (1 + t_bc**2)
        >>> def epsilon_predictor(x, t):
        ...     score_pred = score_predictor(x, t)
        ...     return scheduler.score_to_epsilon(score_pred, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> eps_pred = epsilon_predictor(x_t, t)
        >>> eps_pred.shape
        torch.Size([2, 4])
        """
        expected_shape = (-1,) + (1,) * (score.ndim - 1)
        t_bc = t.reshape(expected_shape)
        sigma_t_bc = self.sigma(t_bc)
        return -sigma_t_bc * score

    def epsilon_to_x0(
        self,
        epsilon: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert epsilon-predictor (noise) output to x0-prediction.

        A common use case is with
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` to train an
        epsilon-predictor instead of an x0-predictor: pass this method as the
        ``epsilon_to_x0_fn`` argument with ``prediction_type="epsilon"``.

        .. math::
            \hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t
            - \sigma(t)\hat{\boldsymbol{\epsilon}}}{\alpha(t)}

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        epsilon : Tensor
            Predicted noise :math:`\hat{\boldsymbol{\epsilon}}` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``epsilon``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Estimated clean data :math:`\hat{\mathbf{x}}_0` with same shape
            as ``epsilon``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an epsilon-predictor, wrap it for manual conversion
        >>> # (done automatically by MSEDSMLoss with prediction_type="epsilon"):
        >>> def epsilon_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x * t_bc / (1 + t_bc**2)
        >>> def x0_predictor(x, t):
        ...     eps_pred = epsilon_predictor(x, t)
        ...     return scheduler.epsilon_to_x0(eps_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> eps_pred = epsilon_predictor(x_t, t)
        >>> eps_pred.shape
        torch.Size([2, 4])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        >>> # Or simply: MSEDSMLoss(..., prediction_type="epsilon", epsilon_to_x0_fn=scheduler.epsilon_to_x0)
        """
        expected_shape = (-1,) + (1,) * (epsilon.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        return (x_t - sigma_t_bc * epsilon) / alpha_t_bc

    def x0_to_epsilon(
        self,
        x0: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert x0-prediction to epsilon (noise) prediction.

        This is the inverse of :meth:`epsilon_to_x0`:

        .. math::
            \hat{\boldsymbol{\epsilon}} = \frac{\mathbf{x}_t
            - \alpha(t)\hat{\mathbf{x}}_0}{\sigma(t)}

        This is a helper method that subclasses, in most cases, do not need to override.

        Parameters
        ----------
        x0 : Tensor
            Predicted clean data :math:`\hat{\mathbf{x}}_0` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``x0``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Epsilon with same shape as ``x0``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an x0-predictor, wrap it for manual conversion:
        >>> def x0_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x / (1 + t_bc**2)
        >>> def epsilon_predictor(x, t):
        ...     x0_pred = x0_predictor(x, t)
        ...     return scheduler.x0_to_epsilon(x0_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        >>> eps_pred = epsilon_predictor(x_t, t)
        >>> eps_pred.shape
        torch.Size([2, 4])
        """
        expected_shape = (-1,) + (1,) * (x0.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        return (x_t - alpha_t_bc * x0) / sigma_t_bc

    def x0_to_flow(
        self,
        x0: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert x0-predictor (clean data) output to flow (velocity) prediction.

        A common use case is with
        :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss` to
        train an x0-predictor instead of a flow-predictor: pass
        ``prediction_type="x0"`` and the loss applies this conversion
        internally.

        .. math::
            \hat{\mathbf{v}} = \frac{\dot{\sigma}(t)}{\sigma(t)} \mathbf{x}_t
            + \left(\dot{\alpha}(t)
            - \frac{\dot{\sigma}(t)}{\sigma(t)}\alpha(t)\right)
            \hat{\mathbf{x}}_0

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        x0 : Tensor
            Predicted clean data :math:`\hat{\mathbf{x}}_0` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``x0``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Flow (velocity) with same shape as ``x0``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have an x0-predictor, wrap it for manual conversion
        >>> # (done automatically by FlowMatchingLoss with prediction_type="x0"):
        >>> def x0_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x / (1 + t_bc**2)
        >>> def flow_predictor(x, t):
        ...     x0_pred = x0_predictor(x, t)
        ...     return scheduler.x0_to_flow(x0_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        >>> flow_pred = flow_predictor(x_t, t)
        >>> flow_pred.shape
        torch.Size([2, 4])
        >>> # Or: FlowMatchingLoss(..., prediction_type="x0", x0_to_flow_fn=scheduler.x0_to_flow)
        """
        expected_shape = (-1,) + (1,) * (x0.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        sigma_dot_t_bc = self.sigma_dot(t_bc)
        ratio = sigma_dot_t_bc / sigma_t_bc
        return ratio * x_t + (alpha_dot_t_bc - ratio * alpha_t_bc) * x0

    def flow_to_x0(
        self,
        flow: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert flow prediction to x0-prediction.

        This is the inverse of :meth:`x0_to_flow`:

        .. math::
            \hat{\mathbf{x}}_0 = \frac{\dot{\sigma}(t)\mathbf{x}_t
            - \sigma(t)\hat{\mathbf{v}}}
            {\dot{\sigma}(t)\alpha(t) - \dot{\alpha}(t)\sigma(t)}

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        flow : Tensor
            Predicted flow (velocity) :math:`\hat{\mathbf{v}}` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``flow``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Estimated clean data :math:`\hat{\mathbf{x}}_0` with same shape
            as ``flow``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a flow-predictor, wrap it for manual conversion:
        >>> def flow_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x * t_bc / (1 + t_bc**2)
        >>> def x0_predictor(x, t):
        ...     flow_pred = flow_predictor(x, t)
        ...     return scheduler.flow_to_x0(flow_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> flow_pred = flow_predictor(x_t, t)
        >>> flow_pred.shape
        torch.Size([2, 4])
        >>> x0_pred = x0_predictor(x_t, t)
        >>> x0_pred.shape
        torch.Size([2, 4])
        """
        expected_shape = (-1,) + (1,) * (flow.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        sigma_dot_t_bc = self.sigma_dot(t_bc)
        denom = sigma_dot_t_bc * alpha_t_bc - alpha_dot_t_bc * sigma_t_bc
        return (sigma_dot_t_bc * x_t - sigma_t_bc * flow) / denom

    def score_to_flow(
        self,
        score: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert score-predictor output to flow (velocity) prediction.

        A common use case is with
        :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss` to
        train a score-predictor instead of a flow-predictor: pass
        ``prediction_type="score"`` and the loss applies this conversion
        internally.

        .. math::
            \hat{\mathbf{v}} = \frac{\dot{\alpha}(t) \mathbf{x}_t
            + \left(\dot{\alpha}(t) \sigma^2(t)
            - \dot{\sigma}(t)\sigma(t)\alpha(t)\right) s(\mathbf{x}_t, t)}
            {\alpha(t)}

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        score : Tensor
            Predicted score :math:`s(\mathbf{x}_t, t)` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``score``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Flow (velocity) with same shape as ``score``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a score-predictor, wrap it for manual conversion
        >>> # (done automatically by FlowMatchingLoss with prediction_type="score"):
        >>> def score_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return -x / (1 + t_bc**2)
        >>> def flow_predictor(x, t):
        ...     score_pred = score_predictor(x, t)
        ...     return scheduler.score_to_flow(score_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> flow_pred = flow_predictor(x_t, t)
        >>> flow_pred.shape
        torch.Size([2, 4])
        >>> # Or: FlowMatchingLoss(..., prediction_type="score", score_to_flow_fn=scheduler.score_to_flow)
        """
        expected_shape = (-1,) + (1,) * (score.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        sigma_dot_t_bc = self.sigma_dot(t_bc)
        coeff = (
            alpha_dot_t_bc * sigma_t_bc**2 - sigma_dot_t_bc * sigma_t_bc * alpha_t_bc
        )
        return (alpha_dot_t_bc * x_t + coeff * score) / alpha_t_bc

    def flow_to_score(
        self,
        flow: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Convert flow-predictor output to score.

        :meth:`get_denoiser` performs this conversion automatically when you
        pass ``flow_predictor``, but you can also call it manually.

        .. math::
            s(\mathbf{x}_t, t) = \frac{\alpha(t)\hat{\mathbf{v}}
            - \dot{\alpha}(t) \mathbf{x}_t}
            {\dot{\alpha}(t) \sigma^2(t)
            - \dot{\sigma}(t)\sigma(t)\alpha(t)}

        This is a helper method that subclasses, in most cases, do not need to
        override.

        Parameters
        ----------
        flow : Tensor
            Predicted flow (velocity) :math:`\hat{\mathbf{v}}` of shape
            :math:`(B, *)`.
        x_t : Tensor
            Current noisy state :math:`\mathbf{x}_t` with same shape as
            ``flow``.
        t : Tensor
            Diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Score with same shape as ``flow``.

        Examples
        --------
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        >>> scheduler = EDMNoiseScheduler()
        >>> # If you have a flow-predictor, wrap it for manual conversion
        >>> # (done automatically by get_denoiser):
        >>> def flow_predictor(x, t):
        ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        ...     return x * t_bc / (1 + t_bc**2)
        >>> def score_predictor(x, t):
        ...     flow_pred = flow_predictor(x, t)
        ...     return scheduler.flow_to_score(flow_pred, x, t)
        >>> x_t = torch.randn(2, 4)
        >>> t = torch.tensor([1.0, 1.0])
        >>> flow_pred = flow_predictor(x_t, t)
        >>> flow_pred.shape
        torch.Size([2, 4])
        >>> score_pred = score_predictor(x_t, t)
        >>> score_pred.shape
        torch.Size([2, 4])
        >>> # Or simply: scheduler.get_denoiser(flow_predictor=flow_predictor)
        """
        expected_shape = (-1,) + (1,) * (flow.ndim - 1)
        t_bc = t.reshape(expected_shape)
        alpha_t_bc = self.alpha(t_bc)
        alpha_dot_t_bc = self.alpha_dot(t_bc)
        sigma_t_bc = self.sigma(t_bc)
        sigma_dot_t_bc = self.sigma_dot(t_bc)
        denom = (
            alpha_dot_t_bc * sigma_t_bc**2 - sigma_dot_t_bc * sigma_t_bc * alpha_t_bc
        )
        return (alpha_t_bc * flow - alpha_dot_t_bc * x_t) / denom

    def get_denoiser(
        self,
        *,
        score_predictor: Predictor | None = None,
        x0_predictor: Predictor | None = None,
        epsilon_predictor: Predictor | None = None,
        flow_predictor: Predictor | None = None,
        denoising_type: Literal["ode", "sde"] = "ode",
        **kwargs: Any,
    ) -> Denoiser:
        r"""
        Factory that converts a predictor to a denoiser for sampling.

        Accepts exactly one of **score-predictor**, **x0-predictor**,
        **epsilon-predictor**, or **flow-predictor**. The returned
        denoiser computes the right-hand side of the reverse ODE or SDE.

        For ODE (``denoising_type="ode"``):

        .. math::
            \frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2} g^2(t)
            s(\mathbf{x}, t)

        For SDE (``denoising_type="sde"``):

        .. math::
            d\mathbf{x} = \left[ f(\mathbf{x}, t) - g^2(t) s(\mathbf{x}, t)
            \right] dt + g(t) d\mathbf{W}

        where :math:`s(\mathbf{x}, t)` is the score. The denoiser uses a
        score-predictor directly; for the other predictors, it derives the
        score via :meth:`x0_to_score`, :meth:`epsilon_to_score`, or
        :meth:`flow_to_score`. *Note:* following SDE integration
        convention, the solver handles the stochastic term
        :math:`g(t) d\mathbf{W}`, which the denoiser itself does not return.

        Parameters
        ----------
        score_predictor : Predictor, optional
            A score-predictor that takes ``(x_t, t)`` and returns a score
            (e.g. :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)`). Can be
            unconditional, conditional, guidance-augmented, etc. Mutually
            exclusive with the other predictors.
        x0_predictor : Predictor, optional
            An x0-predictor that takes ``(x_t, t)`` and returns an estimate
            of clean data :math:`\hat{\mathbf{x}}_0`. The denoiser computes
            the score internally via :meth:`x0_to_score`. Mutually exclusive
            with other predictors.
        epsilon_predictor : Predictor, optional
            An epsilon-predictor that takes ``(x_t, t)`` and returns an
            estimate of the noise
            :math:`\hat{\boldsymbol{\epsilon}}`. The denoiser computes the
            score internally via :meth:`epsilon_to_score`. Mutually exclusive
            with other predictors.
        flow_predictor : Predictor, optional
            A flow-predictor that takes ``(x_t, t)`` and returns an
            estimate of the flow (velocity) :math:`\hat{\mathbf{v}}`. The
            denoiser computes the score internally via
            :meth:`flow_to_score`. Mutually exclusive with the other
            predictors.
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
            If the call does not supply exactly one of ``score_predictor``,
            ``x0_predictor``, ``epsilon_predictor``, or ``flow_predictor``.

        Examples
        --------
        Generate ODE RHS from a score-predictor:

        >>> import torch
        >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
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

        Generate ODE RHS from an epsilon-predictor:

        >>> eps_pred = lambda x, t: x * 0.1  # Toy epsilon-predictor
        >>> denoiser = scheduler.get_denoiser(
        ...     epsilon_predictor=eps_pred, denoising_type="ode")
        >>> dx_dt = denoiser(x, t)  # Returns ODE RHS for sampling
        >>> dx_dt.shape
        torch.Size([2, 3, 8, 8])
        """
        # Validate: exactly one predictor must be provided
        provided = sum(
            p is not None
            for p in (
                score_predictor,
                x0_predictor,
                epsilon_predictor,
                flow_predictor,
            )
        )
        if provided != 1:
            raise ValueError(
                "Exactly one of 'score_predictor', 'x0_predictor', "
                "'epsilon_predictor', or 'flow_predictor' must be provided."
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
        elif flow_predictor is not None:
            flow_to_score = self.flow_to_score

            def _score_from_flow(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                return flow_to_score(flow_predictor(x, t), x, t)

            score_fn = _score_from_flow
        elif epsilon_predictor is not None:
            eps_to_score = self.epsilon_to_score

            def _score_from_eps(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                eps = epsilon_predictor(x, t)
                return eps_to_score(eps, t)

            score_fn = _score_from_eps
        else:
            score_fn = score_predictor

        if denoising_type == "ode":

            def ode_denoiser(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
            ) -> Float[Tensor, " B *dims"]:
                score = score_fn(x, t)
                f = drift(x, t)
                g_sq_bc = diffusion(x, t)
                dx_dt = f - 0.5 * g_sq_bc * score
                return dx_dt

            return ode_denoiser

        elif denoising_type == "sde":

            def sde_denoiser(
                x: Float[Tensor, " B *dims"],
                t: Float[Tensor, " B"],
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
        expected_shape = (-1,) + (1,) * (x0.ndim - 1)
        t_bc = time.reshape(expected_shape)
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
        expected_shape = (-1,) + (1,) * len(spatial_shape)
        tN_bc = tN.reshape(expected_shape)
        sigma_tN_bc = self.sigma(tN_bc)
        return sigma_tN_bc * noise
