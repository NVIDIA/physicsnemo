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

"""Domain-parallel noise scheduler for diffusion training and sampling."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.diffusion.base import Denoiser
from physicsnemo.diffusion.noise_schedulers.noise_schedulers import (
    LinearGaussianNoiseScheduler,
    NoiseScheduler,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import scatter_tensor


class DomainParallelNoiseScheduler(NoiseScheduler):
    r"""Domain-parallel noise scheduler for distributed diffusion training and sampling.

    This class implements the
    :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler` protocol by
    wrapping a :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
    and distributing its operations across a domain-parallel device mesh.

    In domain-parallel diffusion the spatial domain is split across multiple
    ranks.  This scheduler ensures that tensors produced by the underlying
    noise scheduler are distributed correctly across the domain mesh:

    * :meth:`sample_time` — broadcasts sampled times so every shard sees the
      same noise level per batch element (training).
    * :meth:`add_noise` — promotes scalar :math:`\alpha(t)` and
      :math:`\sigma(t)` coefficients to replicated ``DTensor``\s so that
      element-wise operations are type-compatible with sharded data.
    * :meth:`loss_weight` — delegates to the inner scheduler (loss weights
      are per-sample scalars, independent of spatial sharding).
    * :meth:`timesteps` — returns a *replicated* tensor on the domain mesh so
      that solver arithmetic with sharded latents is type-compatible
      (sampling).
    * :meth:`init_latents` — returns a *sharded* tensor on the domain mesh,
      split along the chosen spatial dimension (sampling).
    * :meth:`get_denoiser` — delegates to the inner scheduler's denoiser
      factory.

    .. note::

        The inner scheduler must be a
        :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
        because domain-parallel ``add_noise`` requires access to
        :math:`\alpha(t)` and :math:`\sigma(t)` to construct DTensor-compatible
        coefficients.

    Parameters
    ----------
    scheduler : LinearGaussianNoiseScheduler
        The inner noise scheduler to wrap.
    device_mesh : DeviceMesh
        The device mesh defining the domain-parallel group.
    shard_dim : int
        The tensor dimension along which :meth:`init_latents` shards the
        initial latent state.  For example, for ``(B, C, H, W)`` data sharded
        along the height axis, use ``shard_dim=2``.

    Examples
    --------
    .. code-block:: python

        import torch
        from torch.distributed.device_mesh import init_device_mesh
        from physicsnemo.diffusion.noise_schedulers import (
            DomainParallelNoiseScheduler,
            EDMNoiseScheduler,
        )
        from physicsnemo.diffusion.samplers import sample

        # Create an inner noise scheduler and wrap it for domain parallelism
        inner = EDMNoiseScheduler()
        mesh = init_device_mesh("cuda", (world_size,))
        scheduler = DomainParallelNoiseScheduler(inner, mesh, shard_dim=2)

        # --- Training ---
        t = scheduler.sample_time(batch_size, device="cuda")  # broadcast
        x_noisy = scheduler.add_noise(x0_sharded, t)          # DTensor-aware
        w = scheduler.loss_weight(t)

        # --- Sampling ---
        t_steps = scheduler.timesteps(num_steps, device="cuda")    # replicated
        xN = scheduler.init_latents((C, H, W), t_steps[0:1])      # sharded
        denoiser = scheduler.get_denoiser(x0_predictor=predictor)
        samples = sample(denoiser, xN, scheduler, num_steps=num_steps)

    See Also
    --------
    :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler` :
        The protocol this class implements.
    :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler` :
        The base class required for the inner scheduler.
    """

    def __init__(
        self,
        scheduler: LinearGaussianNoiseScheduler,
        device_mesh: DeviceMesh,
        shard_dim: int,
    ) -> None:
        self._inner = scheduler
        if not isinstance(scheduler, LinearGaussianNoiseScheduler):
            raise ValueError(
                f"DomainParallelNoiseScheduler only supports wrapping LinearGaussianNoiseScheduler, got {type(scheduler).__name__}."
            )
        self._mesh = device_mesh
        self._shard_dim = shard_dim
        dm = DistributedManager()
        self._group = dm.get_mesh_group(device_mesh)
        self._source_rank = dist.get_global_rank(self._group, 0)

    @property
    def inner_scheduler(self) -> LinearGaussianNoiseScheduler:
        """The wrapped noise scheduler."""
        return self._inner

    @property
    def device_mesh(self) -> DeviceMesh:
        """The device mesh used for broadcasting."""
        return self._mesh

    def sample_time(
        self,
        N: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N"]:
        r"""Sample diffusion times and broadcast across the domain-parallel group.

        Rank 0 of the mesh group draws ``N`` random times from the inner
        scheduler; all other ranks receive the same values via broadcast.

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
            Sampled diffusion times of shape :math:`(N,)`, identical on all
            ranks within the domain-parallel group.
        """
        if dist.get_rank(self._group) == 0:
            t = self._inner.sample_time(N, device=device, dtype=dtype)
        else:
            t = torch.empty(N, device=device, dtype=dtype)
        return self._broadcast_time(t)

    def timesteps(
        self,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " N_plus_1"]:
        r"""Generate time-steps replicated across the domain-parallel group.

        The inner scheduler produces a plain 1-D tensor of time-steps.  This
        method wraps it as a *replicated* :class:`ShardTensor` on the domain
        mesh so that solver arithmetic with sharded latents is type-compatible.

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
            Replicated time-steps tensor of shape :math:`(N + 1,)`.
        """
        t = self._inner.timesteps(num_steps, device=device, dtype=dtype)
        return self._scatter(t, placements=(Replicate(),))

    def init_latents(
        self,
        spatial_shape: Tuple[int, ...],
        tN: Float[Tensor, " B"],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, " B *spatial_shape"]:
        r"""Initialize latent state sharded across the domain-parallel group.

        Rank 0 generates the full initial noise via the inner scheduler's
        :meth:`init_latents`, then scatters it across the domain mesh with
        ``Shard(shard_dim)`` placement.

        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Spatial shape of the latent state, e.g. ``(C, H, W)``.
        tN : Tensor
            Initial diffusion time of shape :math:`(B,)`.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor.

        Returns
        -------
        Tensor
            Sharded initial noisy latent of shape :math:`(B, *spatial\_shape)`.
        """
        # Unwrap tN to a plain tensor if it is a DTensor/ShardTensor
        # (e.g. from timesteps() which returns Replicate placement),
        # because the inner scheduler operates on plain tensors.
        if hasattr(tN, "to_local"):
            tN = tN.to_local()
        xN = self._inner.init_latents(spatial_shape, tN, device=device, dtype=dtype)
        return self._scatter(xN, placements=(Shard(self._shard_dim),))

    def add_noise(
        self,
        x0: Float[Tensor, " B *dims"],
        time: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Add noise, promoting scalar coefficients for ``ShardTensor`` data.

        When ``x0`` is a ``ShardTensor``, the scalar :math:`\alpha(t)` and
        :math:`\sigma(t)` coefficients from the inner scheduler are promoted
        to replicated ``DTensor``\s on ``x0``'s device mesh so that
        element-wise operations are type-compatible.

        Falls back to the inner scheduler's ``add_noise`` when ``x0`` is a
        plain tensor.

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
        mesh = getattr(x0, "device_mesh", None)
        if mesh is None:
            return self._inner.add_noise(x0, time)

        t_bc = time.reshape(-1, *([1] * (x0.ndim - 1)))
        alpha_t = self._inner.alpha(t_bc)
        sigma_t = self._inner.sigma(t_bc)

        if not isinstance(alpha_t, DTensor):
            alpha_t = DTensor.from_local(
                alpha_t, device_mesh=mesh, placements=[Replicate()]
            )
        if not isinstance(sigma_t, DTensor):
            sigma_t = DTensor.from_local(
                sigma_t, device_mesh=mesh, placements=[Replicate()]
            )

        noise = torch.randn_like(x0)
        return alpha_t * x0 + sigma_t * noise

    def loss_weight(
        self,
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, "N *channels"]:  # noqa: F821
        r"""Compute loss weight for denoising score matching training.

        Delegates to the inner scheduler. Loss weights are per-sample scalars
        (or per-sample-per-channel), independent of spatial sharding.

        Parameters
        ----------
        t : Tensor
            Diffusion time values of shape :math:`(N,)`.

        Returns
        -------
        Tensor
            Loss weight with leading dimension :math:`N`.
        """
        return self._inner.loss_weight(t)

    def get_denoiser(
        self,
        **kwargs: Any,
    ) -> Denoiser:
        r"""Factory that converts a predictor into a denoiser for sampling.

        Delegates to the inner scheduler's
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_denoiser`.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments forwarded to the inner scheduler's
            ``get_denoiser``. See
            :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_denoiser`
            for accepted arguments (e.g. ``score_predictor``,
            ``x0_predictor``, ``denoising_type``).

        Returns
        -------
        Denoiser
            A callable implementing the
            :class:`~physicsnemo.diffusion.Denoiser` interface.
        """
        return self._inner.get_denoiser(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast_time(self, t: torch.Tensor) -> torch.Tensor:
        """Broadcast *t* from rank 0 of the domain group to all other ranks."""
        dist.broadcast(t, src=self._source_rank, group=self._group)
        return t

    def _scatter(
        self,
        tensor: torch.Tensor,
        placements: tuple,
    ) -> torch.Tensor:
        """Scatter *tensor* from rank 0 across the domain mesh."""

        return scatter_tensor(
            tensor,
            self._source_rank,
            self._mesh,
            placements=placements,
            global_shape=tensor.shape,
            dtype=tensor.dtype,
        )
