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

"""ODE/SDE solvers for diffusion model sampling."""

import math
from typing import Callable, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser


@runtime_checkable
class Solver(Protocol):
    r"""
    Protocol defining the interface for diffusion solvers.

    A solver implements a numerical method to integrate the diffusion process
    from a noisy state to a less noisy (or clean) state. Each call to
    :meth:`step` advances the state from time ``t_cur`` (:math:`t_n`) to
    ``t_next`` (:math:`t_{n-1}`).

    This is the minimal interface required for sampling from a diffusion model,
    and any object that implements this interface can be used as a solver in
    sampling utilities.

    The update rule applied by the sampler is roughly:

    .. math::
        \mathbf{x}_{n-1} = \text{Step}(F(\mathbf{x}_n, t_n); \mathbf{x}_n, t_n, t_{n-1})

    where :math:`F` is the denoiser (e.g. the right hand side in the case of
    ODE/SDE-based sampling, the denoised latent state in the case of discrete
    Markov chain-based sampling, etc.) and :math:`\text{Step}` is
    the update rule of the solver, implemented by the :meth:`step` method.

    See Also
    --------
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function that
        uses solvers to generate samples.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import Solver
    >>>
    >>> class SimpleEuler:
    ...     def __init__(self, denoiser):
    ...         self.denoiser = denoiser
    ...     def step(self, x, t_cur, t_next):
    ...         d = (x - self.denoiser(x, t_cur)) / t_cur
    ...         return x + (t_next - t_cur) * d
    ...
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1)**2)  # Toy denoiser
    >>> solver = SimpleEuler(denoiser)
    >>> isinstance(solver, Solver)
    True
    """

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one integration step from ``t_cur`` to ``t_next``.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_{n}` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        ...


class EulerSolver(Solver):
    r"""
    First-order Euler solver for diffusion ODEs.

    This is a fast solver with one denoiser evaluation per step, but typically
    produces lower quality samples compared to higher-order methods.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it is
        expected to return the right hand side of the ODE. Typically obtained
        via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature can be used.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import EulerSolver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EulerSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    >>> isinstance(solver, Solver)
    True
    """

    def __init__(self, denoiser: Denoiser) -> None:
        self.denoiser = denoiser

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one Euler integration step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_{n}` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        # RHS evaluation and step update
        d_cur = self.denoiser(x, t_cur)
        x_next = x + (t_next_bc - t_cur_bc) * d_cur

        return x_next


class HeunSolver(Solver):
    r"""
    Second-order Heun solver for diffusion ODEs.

    This method requires two denoiser evaluations per step but usually produces
    higher quality samples than :class:`EulerSolver`.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it is
        expected to return the right hand side of the ODE. Typically obtained
        via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature can be used.
    alpha : float, optional
        Interpolation parameter for the corrector step, must be in (0, 1].
        ``alpha=1`` gives the standard Heun method (trapezoidal rule),
        ``alpha=0.5`` gives the midpoint method. By default 1.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import HeunSolver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = HeunSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        alpha: float = 1.0,
    ) -> None:
        self.denoiser = denoiser
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one Heun integration step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_n` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        h_bc = t_next_bc - t_cur_bc

        # First RHS evaluation
        d_cur = self.denoiser(x, t_cur)

        # Predictor step to intermediate point
        t_prime_bc = t_cur_bc + self.alpha * h_bc
        x_prime = x + self.alpha * h_bc * d_cur

        # Mask for elements where t_next != 0 (need 2nd order correction)
        # Shape: (B, 1, ..., 1) for broadcasting
        mask_bc = (t_next_bc != 0).float()

        # Second RHS evaluation (compute everywhere, masked later)
        # Avoid division by zero in denoiser by using t_cur where t_prime is 0
        t_prime = t_prime_bc.reshape(x.shape[0])
        t_prime_safe = torch.where(t_prime == 0, t_cur, t_prime)
        d_prime = self.denoiser(x_prime, t_prime_safe)

        # Apply 2nd order correction only where t_next != 0
        # Where t_next == 0, use first-order Euler step
        w_cur = 1 - 1 / (2 * self.alpha)
        w_prime = 1 / (2 * self.alpha)
        x_euler = x + h_bc * d_cur
        x_heun = x + h_bc * (w_cur * d_cur + w_prime * d_prime)
        x_next = mask_bc * x_heun + (1 - mask_bc) * x_euler

        return x_next


class EDMStochasticEulerSolver(Solver):
    r"""
    First-order stochastic Euler sampler from the EDM paper.

    Implements stochastic sampling with configurable noise injection
    controlled by the "churn" parameters.

    .. important::

        This is **not** a true SDE solver. It performs ad-hoc noise injection
        ("churn") at each step to improve sample diversity, but the underlying
        integration is still an ODE step. Therefore, the denoiser should return
        the right-hand side of the **ODE**, not the SDE.

    By default, noise injection is performed directly in time-step space.
    For linear-Gaussian noise schedules where diffusion time and noise level
    are not equal (e.g., VP schedule), provide ``sigma_fn`` and
    ``sigma_inv_fn`` to apply churn in noise-level space rather than
    time-step space. Optionally provide ``diffusion_fn`` to control the
    time-dependent magnitude of the injected noise.

    .. code-block:: python

        def sigma_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # noise level, same shape as t

        def sigma_inv_fn(
            sigma: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # diffusion time, same shape as sigma

        def diffusion_fn(
            x: Tensor,  # shape: (B, *dims)
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # g^2(x, t), broadcastable to shape of x

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Should
        return the right-hand side of the **ODE** (not the SDE, since the
        stochastic noise injection is handled internally by this solver).
        Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
        with ``denoising_type="ode"``.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (deterministic), in which case this
        solver is equivalent to the deterministic :class:`EulerSolver`.
    S_min : float, optional
        Minimum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default 0.
    S_max : float, optional
        Maximum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default
        ``float("inf")``.
    S_noise : float, optional
        Noise scaling factor. Large values add more noise to the latent state.
        By default 1.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.
    sigma_fn : Callable[[Tensor], Tensor] | None, optional
        Maps time to noise level :math:`\sigma(t)`. Useful for linear-Gaussian
        schedules where :math:`\sigma(t) \neq t`. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
        If provided, ``sigma_inv_fn`` must also be provided.
        By default ``None`` (identity mapping).
    sigma_inv_fn : Callable[[Tensor], Tensor] | None, optional
        Maps noise level back to time. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma_inv`.
        If provided, ``sigma_fn`` must also be provided.
        By default ``None`` (identity mapping).
    diffusion_fn : Callable[[Tensor, Tensor], Tensor] | None, optional
        Controls the time-dependent magnitude of the injected
        noise, in addition of the ``S_noise`` scaling factor. Typically the
        squared diffusion coefficient :math:`g^2(\mathbf{x}, t)` from the
        reverse SDE, obtained from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.diffusion`.
        By default ``None`` (:math:`g^2 = 2t`), which corresponds to an
        EDM-like noise schedule.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    Basic usage with default parameters (noise injection in time-step space):

    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import (
    ...     EDMStochasticEulerSolver,
    ... )
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EDMStochasticEulerSolver(denoiser, S_churn=40, num_steps=18)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Using noise scheduler methods for linear-Gaussian schedules where
    :math:`\sigma(t) \neq t` (e.g., VP schedule). The callbacks map between
    time and noise level, allowing the churn to be applied in noise-level
    space before converting back to time-step space:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> num_steps = 10
    >>> solver = EDMStochasticEulerSolver(
    ...     denoiser,
    ...     S_churn=40,
    ...     num_steps=num_steps,
    ...     sigma_fn=scheduler.sigma,
    ...     sigma_inv_fn=scheduler.sigma_inv,
    ...     diffusion_fn=scheduler.diffusion,
    ... )
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        num_steps: int = 18,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        sigma_inv_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        diffusion_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B"]], Float[Tensor, " B *_"]
        ]
        | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.num_steps = num_steps
        # Validate sigma_fn and sigma_inv_fn
        if (sigma_fn is None) != (sigma_inv_fn is None):
            raise ValueError(
                "sigma_fn and sigma_inv_fn must both be provided or both None."
            )
        if sigma_fn is None and sigma_inv_fn is None:
            self.sigma_fn = lambda t: t
            self.sigma_inv_fn = lambda sigma: sigma
            self._use_noise_level_space = False
        else:
            self.sigma_fn = sigma_fn
            self.sigma_inv_fn = sigma_inv_fn
            self._use_noise_level_space = True
        if diffusion_fn is None:
            self.diffusion_fn = lambda x, t: 2 * t.reshape(-1, *([1] * (x.ndim - 1)))
        else:
            self.diffusion_fn = diffusion_fn

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one stochastic Euler sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_n` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        gamma_base = min(self.S_churn / self.num_steps, math.sqrt(2) - 1)

        # Compute perturbed time t_hat with increased noise
        # NOTE: sigma_fn and sigma_inv_fn are identity if not provided (stays
        # in time-step space). diffusion_fn defaults to g^2 = 2t (EDM-like
        # noise schedule).
        sigma_cur_bc = self.sigma_fn(t_cur_bc)
        # Mask: apply churn only where S_min <= sigma <= S_max
        churn_mask = (sigma_cur_bc >= self.S_min) & (sigma_cur_bc <= self.S_max)
        gamma_bc = torch.where(churn_mask, gamma_base, 0.0)
        sigma_hat_bc = sigma_cur_bc + gamma_bc * sigma_cur_bc
        t_hat_bc = self.sigma_inv_fn(sigma_hat_bc)
        # Noise scale: sqrt(sigma_hat^2 - sigma_cur^2) * S_noise * g(x,t) / sqrt(2*t)
        g_sq_bc = self.diffusion_fn(x, t_cur)
        safe_t_cur_bc = torch.where(t_cur_bc == 0, torch.ones_like(t_cur_bc), t_cur_bc)
        noise_scale_bc = (
            (t_hat_bc**2 - t_cur_bc**2).clamp(min=0).sqrt()
            * self.S_noise
            * (g_sq_bc / (2 * safe_t_cur_bc)).sqrt()
        )
        noise_scale_bc = torch.where(
            t_cur_bc == 0, torch.zeros_like(noise_scale_bc), noise_scale_bc
        )

        # Perturb latent with noise
        x_hat = x + noise_scale_bc * torch.randn_like(x)

        # Euler step from t_hat to t_next
        t_hat = t_hat_bc.reshape(x.shape[0])
        d_cur = self.denoiser(x_hat, t_hat)
        x_next = x_hat + (t_next_bc - t_hat_bc) * d_cur

        return x_next


class EDMStochasticHeunSolver(Solver):
    r"""
    Second-order stochastic Heun sampler from the EDM paper.

    Implements stochastic sampling with configurable noise injection
    controlled by the "churn" parameters, using a second-order Heun
    correction step.

    .. important::

        This is **not** a true SDE solver. It performs ad-hoc noise injection
        ("churn") at each step to improve sample diversity, but the underlying
        integration is still an ODE step. Therefore, the denoiser should return
        the right-hand side of the **ODE**, not the SDE.

    By default, noise injection is performed directly in time-step space.
    For linear-Gaussian noise schedules where diffusion time and noise level
    are not equal (e.g., VP schedule), provide ``sigma_fn`` and
    ``sigma_inv_fn`` to apply churn in noise-level space rather than
    time-step space. Optionally provide ``diffusion_fn`` to control the
    time-dependent magnitude of the injected noise.

    .. code-block:: python

        def sigma_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # noise level, same shape as t

        def sigma_inv_fn(
            sigma: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # diffusion time, same shape as sigma

        def diffusion_fn(
            x: Tensor,  # shape: (B, *dims)
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # g^2(x, t), broadcastable to shape of x

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Should
        return the right-hand side of the **ODE** (not the SDE, since the
        stochastic noise injection is handled internally by this solver).
        Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
        with ``denoising_type="ode"``.
    alpha : float, optional
        Interpolation parameter for the corrector step, must be in (0, 1].
        ``alpha=1`` gives the standard Heun method (trapezoidal rule),
        ``alpha=0.5`` gives the midpoint method. By default 1.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (deterministic), in which case this
        solver is equivalent to the deterministic :class:`HeunSolver`.
    S_min : float, optional
        Minimum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default 0.
    S_max : float, optional
        Maximum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default
        ``float("inf")``.
    S_noise : float, optional
        Noise scaling factor. Large values add more noise to the latent state.
        By default 1.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.
    sigma_fn : Callable[[Tensor], Tensor] | None, optional
        Maps time to noise level :math:`\sigma(t)`. Useful for linear-Gaussian
        schedules where :math:`\sigma(t) \neq t`. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
        If provided, ``sigma_inv_fn`` must also be provided.
        By default ``None`` (identity mapping).
    sigma_inv_fn : Callable[[Tensor], Tensor] | None, optional
        Maps noise level back to time. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma_inv`.
        If provided, ``sigma_fn`` must also be provided.
        By default ``None`` (identity mapping).
    diffusion_fn : Callable[[Tensor, Tensor], Tensor] | None, optional
        Controls the time-dependent magnitude of the injected
        noise, in addition of the ``S_noise`` scaling factor. Typically the
        squared diffusion coefficient :math:`g^2(\mathbf{x}, t)` from the
        reverse SDE, obtained from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.diffusion`.
        By default ``None`` (:math:`g^2 = 2t`), which corresponds to an
        EDM-like noise schedule.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    Basic usage with default parameters (noise injection in time-step space):

    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import (
    ...     EDMStochasticHeunSolver,
    ... )
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EDMStochasticHeunSolver(denoiser, S_churn=40, num_steps=18)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Using noise scheduler methods for linear-Gaussian schedules where
    :math:`\sigma(t) \neq t` (e.g., VP schedule). The callbacks map between
    time and noise level, allowing the churn to be applied in noise-level
    space before converting back to time-step space:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> num_steps = 10
    >>> solver = EDMStochasticHeunSolver(
    ...     denoiser,
    ...     S_churn=40,
    ...     num_steps=num_steps,
    ...     sigma_fn=scheduler.sigma,
    ...     sigma_inv_fn=scheduler.sigma_inv,
    ...     diffusion_fn=scheduler.diffusion,
    ... )
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        alpha: float = 1.0,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        num_steps: int = 18,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        sigma_inv_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        diffusion_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B"]], Float[Tensor, " B *_"]
        ]
        | None = None,
    ) -> None:
        self.denoiser = denoiser
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.num_steps = num_steps
        # Validate sigma_fn and sigma_inv_fn
        if (sigma_fn is None) != (sigma_inv_fn is None):
            raise ValueError(
                "sigma_fn and sigma_inv_fn must both be provided or both None."
            )
        if sigma_fn is None and sigma_inv_fn is None:
            self.sigma_fn = lambda t: t
            self.sigma_inv_fn = lambda sigma: sigma
            self._use_noise_level_space = False
        else:
            self.sigma_fn = sigma_fn
            self.sigma_inv_fn = sigma_inv_fn
            self._use_noise_level_space = True
        if diffusion_fn is None:
            self.diffusion_fn = lambda x, t: 2 * t.reshape(-1, *([1] * (x.ndim - 1)))
        else:
            self.diffusion_fn = diffusion_fn

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one stochastic Heun sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_n` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        gamma_base = min(self.S_churn / self.num_steps, math.sqrt(2) - 1)

        # Compute perturbed time t_hat with increased noise
        # NOTE: sigma_fn and sigma_inv_fn are identity if not provided (stays
        # in time-step space). diffusion_fn defaults to g^2 = 2t (EDM-like
        # noise schedule).
        sigma_cur_bc = self.sigma_fn(t_cur_bc)
        # Mask: apply churn only where S_min <= sigma <= S_max
        churn_mask = (sigma_cur_bc >= self.S_min) & (sigma_cur_bc <= self.S_max)
        gamma_bc = torch.where(churn_mask, gamma_base, 0.0)
        sigma_hat_bc = sigma_cur_bc + gamma_bc * sigma_cur_bc
        t_hat_bc = self.sigma_inv_fn(sigma_hat_bc)
        # Noise scale: sqrt(sigma_hat^2 - sigma_cur^2) * S_noise * g(x,t) / sqrt(2*t)
        g_sq_bc = self.diffusion_fn(x, t_cur)
        safe_t_cur_bc = torch.where(t_cur_bc == 0, torch.ones_like(t_cur_bc), t_cur_bc)
        noise_scale_bc = (
            (sigma_hat_bc**2 - sigma_cur_bc**2).clamp(min=0).sqrt()
            * self.S_noise
            * (g_sq_bc / (2 * safe_t_cur_bc)).sqrt()
        )
        noise_scale_bc = torch.where(
            t_cur_bc == 0, torch.zeros_like(noise_scale_bc), noise_scale_bc
        )

        # Perturb latent with noise
        x_hat = x + noise_scale_bc * torch.randn_like(x)

        # Euler step from t_hat to intermediate point (predictor)
        t_hat = t_hat_bc.reshape(x.shape[0])
        h_bc = t_next_bc - t_hat_bc
        d_cur = self.denoiser(x_hat, t_hat)
        t_prime_bc = t_hat_bc + self.alpha * h_bc
        x_prime = x_hat + self.alpha * h_bc * d_cur

        # Mask for elements where t_next != 0 (need 2nd order correction)
        mask_bc = (t_next_bc != 0).float()

        # Second RHS evaluation (compute everywhere, masked later)
        t_prime = t_prime_bc.reshape(x.shape[0])
        # Avoid issues by using t_hat where t_prime would be 0
        t_prime_safe = torch.where(t_prime == 0, t_hat, t_prime)
        d_prime = self.denoiser(x_prime, t_prime_safe)

        # Apply 2nd order correction only where t_next != 0
        w_cur = 1 - 1 / (2 * self.alpha)
        w_prime = 1 / (2 * self.alpha)
        x_euler = x_hat + h_bc * d_cur
        x_heun = x_hat + h_bc * (w_cur * d_cur + w_prime * d_prime)
        x_next = mask_bc * x_heun + (1 - mask_bc) * x_euler

        return x_next
