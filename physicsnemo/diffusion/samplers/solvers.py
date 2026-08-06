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
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

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
        # Ensure contiguous strides so that both denoiser calls (at t_cur
        # and at the intermediate t_prime) present the same stride layout
        # to torch.compile, avoiding spurious recompilations.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

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
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

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
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

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


class DPMSolverPlusPlus2M(Solver):
    r"""
    DPM-Solver++(2M): second-order multistep solver for diffusion ODEs.

    Unlike :class:`HeunSolver`, which attains second order with *two* denoiser
    evaluations per step, this solver attains second order with a *single*
    evaluation per step by reusing the previous step's data prediction: an
    exponential-integrator multistep update of the probability-flow ODE. For a
    fixed budget of denoiser evaluations it can therefore take twice as many
    steps as a second-order single-step method.

    The scheme applies to linear-Gaussian diffusion schedules
    :math:`\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}`
    whose schedule functions return finite, strictly positive :math:`\alpha` and
    :math:`\sigma` at every step before the zero-noise endpoint, and for which
    :math:`\mathrm{d}\lambda / \mathrm{d}t \neq 0` there, where
    :math:`\lambda = \ln(\alpha / \sigma)`. Positivity is a requirement on the
    returned values, not on the schedule in exact arithmetic: a schedule whose
    :math:`\alpha` underflows to zero in the dtype of the time-steps is outside
    this range. The recovery below divides by
    :math:`\alpha \dot{\sigma} - \sigma \dot{\alpha}`, which equals
    :math:`-\alpha \sigma \, \mathrm{d}\lambda / \mathrm{d}t`, so a
    stationary :math:`\lambda` is not admissible even if it is monotonic.
    With :math:`\lambda` (half the log-SNR) and step size
    :math:`h = \lambda_{n-1} - \lambda_n`, so that

    .. math::
        e^{-h} = \frac{\alpha_n \sigma_{n-1}}{\alpha_{n-1} \sigma_n} ,

    the update is

    .. math::
        \mathbf{x}_{n-1} = \frac{\sigma_{n-1}}{\sigma_n}\, \mathbf{x}_n
                           + \alpha_{n-1} \left(1 - e^{-h}\right)
                             \bar{\mathbf{D}}_n ,

    where the extrapolated data prediction is

    .. math::
        \bar{\mathbf{D}}_n = \left(1 + \frac{1}{2r}\right) \mathbf{D}_n
                             - \frac{1}{2r} \mathbf{D}_{n+1} ,
        \qquad r = \frac{h_{n+1}}{h_n} ,

    and :math:`\mathbf{D}_n` is the denoised (data-space) prediction at step
    :math:`n`. On the first step -- and whenever no usable history is available,
    including after a zero-length step in :math:`\lambda` -- the scheme falls
    back to the first-order update
    :math:`\bar{\mathbf{D}}_n = \mathbf{D}_n`, which for :math:`\alpha_t = 1` is
    algebraically identical to an explicit Euler step in :math:`\sigma`.

    The final step, where :math:`\sigma_{n-1} = 0`, likewise uses
    :math:`\mathbf{D}_n` rather than the extrapolation
    :math:`\bar{\mathbf{D}}_n`. It returns
    :math:`\alpha_{n-1} \mathbf{D}_n`, which is :math:`\mathbf{D}_n` only
    when :math:`\alpha = 1`. This lowers the order of that one step relative
    to a literal reading of Algorithm 2, and matches k-diffusion's terminal
    handling and diffusers' ``lower_order_final`` behavior.

    The multistep extrapolation amplifies rounding error when the previous step
    is much shorter than the current one in :math:`\lambda`, since the
    coefficient :math:`h_n / (2 h_{n+1})` is then large and scales the difference
    of successive data predictions. Very fine or strongly non-uniform schedules
    may become rounding-limited with ``float16`` or ``bfloat16`` latents; prefer
    ``float32`` latent arithmetic in that regime.

    .. note::

        When selected by name,
        :func:`~physicsnemo.diffusion.samplers.sample` configures the solver from
        its noise scheduler. For direct construction, including passing an
        instance to :func:`~physicsnemo.diffusion.samplers.sample`, omitting the
        four schedule functions selects the EDM defaults
        (:math:`\alpha_t = 1`, :math:`\sigma_t = t`); provide all four for
        another schedule. Successive manual :meth:`step` calls must
        belong to one uninterrupted trajectory.

    The four schedule functions are supplied together or not at all; a partial
    set is rejected, since it would combine a custom schedule with the EDM
    defaults (:math:`\alpha_t = 1`, :math:`\sigma_t = t`) for the rest. The
    signatures of the ``denoiser`` and of the four schedule functions are:

    .. code-block:: python

        def denoiser(
            x: Tensor,  # shape: (B, *dims)
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # ODE right-hand side, same shape as x

        def alpha_fn(
            t: Tensor,  # shape: (B, 1, ..., 1)
        ) -> Tensor: ...  # alpha_t, broadcastable to the shape of t

        def sigma_fn(
            t: Tensor,  # shape: (B, 1, ..., 1)
        ) -> Tensor: ...  # sigma_t, broadcastable to the shape of t

        def alpha_dot_fn(
            t: Tensor,  # shape: (B, 1, ..., 1)
        ) -> Tensor: ...  # d(alpha_t)/dt, broadcastable to the shape of t

        def sigma_dot_fn(
            t: Tensor,  # shape: (B, 1, ..., 1)
        ) -> Tensor: ...  # d(sigma_t)/dt, broadcastable to the shape of t

    .. note::

        This solver is **stateful**: it caches the previous data prediction and
        step size across calls to :meth:`step`. Call :meth:`reset` before
        starting a new sampling trajectory when reusing an instance;
        :func:`~physicsnemo.diffusion.samplers.sample` does this automatically.
        A single instance cannot be shared by two interleaved trajectories.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it is
        expected to return the right-hand side of the probability-flow ODE.
        The data prediction is recovered internally as

        .. math::
            \mathbf{D} = \frac{\dot{\sigma} \mathbf{x} - \sigma\,
            \text{RHS}}{\alpha \dot{\sigma} - \sigma \dot{\alpha}} ,

        which for the EDM schedule reduces to
        :math:`\mathbf{x} - t \cdot \text{RHS}`.
        Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature can be used.
    alpha_fn : Callable[[Tensor], Tensor] | None, default=None
        The schedule coefficient :math:`\alpha_t`.
    sigma_fn : Callable[[Tensor], Tensor] | None, default=None
        The noise level :math:`\sigma_t`.
    alpha_dot_fn : Callable[[Tensor], Tensor] | None, default=None
        The derivative :math:`\dot{\alpha}_t`.
    sigma_dot_fn : Callable[[Tensor], Tensor] | None, default=None
        The derivative :math:`\dot{\sigma}_t`.

    Note
    ----
    Reference: `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models <https://arxiv.org/abs/2211.01095>`_, Algorithm 2.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import DPMSolverPlusPlus2M
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = DPMSolverPlusPlus2M(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_t = solver.step(x_t, torch.tensor([2.0]), torch.tensor([1.0]))
    >>> x_t = solver.step(x_t, torch.tensor([1.0]), torch.tensor([0.5]))
    >>> x_t.shape
    torch.Size([1, 3, 8, 8])
    >>> isinstance(solver, Solver)
    True
    """

    # Opts this solver into the ``reset`` lifecycle call in ``sample``. Gated on
    # a marker rather than on ``hasattr(solver, "reset")`` so that a pre-existing
    # user-defined solver with an unrelated ``reset`` method is left untouched.
    _requires_state_reset = True

    # Tells ``sample`` to inject the schedule functions of its noise scheduler.
    # Kept off the ``Solver`` protocol, which is runtime checkable: a new member
    # would make ``isinstance`` fail for solvers that implement only ``step``.
    _requires_schedule_fns = True

    # Multistep history, cleared by ``reset``.
    _D_prev: Tensor | None
    _h_prev: Tensor | None

    def __init__(
        self,
        denoiser: Denoiser,
        *,
        alpha_fn: Callable[[Tensor], Tensor] | None = None,
        sigma_fn: Callable[[Tensor], Tensor] | None = None,
        alpha_dot_fn: Callable[[Tensor], Tensor] | None = None,
        sigma_dot_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.denoiser = denoiser
        provided = [alpha_fn, sigma_fn, alpha_dot_fn, sigma_dot_fn]
        if any(f is not None for f in provided) and not all(
            f is not None for f in provided
        ):
            missing = [
                name
                for name, f in zip(
                    ("alpha_fn", "sigma_fn", "alpha_dot_fn", "sigma_dot_fn"), provided
                )
                if f is None
            ]
            raise ValueError(
                "The schedule functions must be given together or not at all, "
                f"but {', '.join(missing)} "
                + ("is" if len(missing) == 1 else "are")
                + " missing. Supplying only some of them would silently combine a "
                "custom schedule with the EDM defaults for the rest."
            )

        # Default to the EDM schedule (alpha = 1, sigma = t), which makes the
        # update reduce to the variance-exploding form.
        self.alpha_fn = alpha_fn if alpha_fn is not None else torch.ones_like
        self.sigma_fn = sigma_fn if sigma_fn is not None else (lambda t: t)
        self.alpha_dot_fn = (
            alpha_dot_fn if alpha_dot_fn is not None else torch.zeros_like
        )
        self.sigma_dot_fn = (
            sigma_dot_fn if sigma_dot_fn is not None else torch.ones_like
        )
        self.reset()

    def reset(self) -> None:
        r"""
        Clear the multistep history, starting a fresh trajectory.

        Drops the cached data prediction and step size from the previous
        :meth:`step` call, so that the next call uses the first-order fallback,
        and releases the cached latent-sized tensor.
        :func:`~physicsnemo.diffusion.samplers.sample` calls this automatically
        at the start of every trajectory. Call it explicitly when driving
        :meth:`step` in a hand-written loop and reusing the instance for more
        than one trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._D_prev = None
        self._h_prev = None

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one DPM-Solver++(2M) integration step.

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
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)
        t_next_bc = t_next.reshape(expected_shape)

        a_cur, s_cur = self.alpha_fn(t_cur_bc), self.sigma_fn(t_cur_bc)
        a_next, s_next = self.alpha_fn(t_next_bc), self.sigma_fn(t_next_bc)
        a_dot, s_dot = self.alpha_dot_fn(t_cur_bc), self.sigma_dot_fn(t_cur_bc)

        # At sigma == 0 the state is noise-free, and the right-hand side may be
        # singular there. Evaluate at a surrogate time to keep the call finite;
        # the result is dropped below, as such a step is the identity.
        is_degenerate = s_cur == 0
        t_cur_safe = torch.where(is_degenerate, torch.ones_like(t_cur_bc), t_cur_bc)
        a_cur = torch.where(is_degenerate, torch.ones_like(a_cur), a_cur)
        s_cur = torch.where(is_degenerate, torch.ones_like(s_cur), s_cur)
        a_dot = torch.where(is_degenerate, torch.zeros_like(a_dot), a_dot)
        s_dot = torch.where(is_degenerate, torch.ones_like(s_dot), s_dot)

        # Single evaluation; recover the data prediction from the right-hand
        # side. Written to avoid an explicit division by sigma near the
        # endpoint. For the EDM schedule this is x - t * RHS.
        rhs = self.denoiser(x, t_cur_safe.reshape(t_cur.shape))
        D = (s_dot * x - s_cur * rhs) / (a_cur * s_dot - s_cur * a_dot)

        # Step in lambda = log(alpha / sigma), half the log-SNR. lambda diverges
        # at the zero-noise endpoint, so the general branch is evaluated with a
        # strictly positive surrogate and the final mask selects alpha_next * D.
        is_final = s_next == 0
        s_next_safe = torch.where(is_final, 0.5 * s_cur, s_next)

        compute_dtype = torch.promote_types(t_cur_bc.dtype, torch.float32)
        emh_hp = (s_next_safe.to(compute_dtype) * a_cur.to(compute_dtype)) / (
            a_next.to(compute_dtype) * s_cur.to(compute_dtype)
        )
        h = -torch.log(emh_hp)
        # expm1 rather than 1 - exp(-h): the latter cancels catastrophically for
        # the small h of a fine ladder, losing most of the value in bfloat16.
        one_minus_emh = (-torch.expm1(-h)).to(x.dtype)
        # Computed in the same promoted dtype. On non-terminal EDM steps it
        # equals exp(-h); matching precision avoids rounding differences.
        sigma_ratio = (s_next.to(compute_dtype) / s_cur.to(compute_dtype)).to(x.dtype)

        if self._D_prev is None or self._h_prev is None:
            D_bar = D  # first-order DPM-Solver++ update
        else:
            # Extrapolate the data prediction in lambda to lambda_cur + h/2:
            # D + (1 / 2r) * (D - D_prev), r = h_prev / h, written h / (2 * h_prev).
            # A repeated timestep gives h_prev == 0. Use a finite dummy
            # denominator because torch.where evaluates both branches; the zero
            # coefficient then falls back to first order.
            has_history = self._h_prev != 0
            h_prev_safe = torch.where(
                has_history, self._h_prev, torch.ones_like(self._h_prev)
            )
            coeff = torch.where(
                has_history, h / (2.0 * h_prev_safe), torch.zeros_like(h)
            ).to(x.dtype)
            D_bar = (1.0 + coeff) * D - coeff * self._D_prev

        general = sigma_ratio * x + a_next * one_minus_emh * D_bar
        # At sigma_next == 0, use the lower-order final update alpha_next * D
        # instead of the extrapolated D_bar; see the class docstring.
        x_next = torch.where(
            is_degenerate, x, torch.where(is_final, a_next * D, general)
        )

        # Cache unconditionally to avoid a data-dependent branch (device sync,
        # breaks fullgraph); `reset` clears it. Terminal and degenerate steps use
        # a surrogate step size, which is not a real one, so store zero; a
        # repeated timestep naturally yields h == 0 already.
        self._D_prev = D
        self._h_prev = torch.where(is_final | is_degenerate, torch.zeros_like(h), h)

        return x_next
