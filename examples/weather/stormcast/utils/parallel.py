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

"""Domain parallelization utilities."""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.tensor import DTensor, distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import ShardTensor, scatter_tensor
from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

from datasets.dataset import worker_init
from utils.nn import nested_to


class ParallelHelper:
    """Manage model and data distribution and sharding in domain parallel training.

    Parameters
    ----------
    domain_parallel_size : int
        Number of ranks in the domain-parallel dimension.
    use_shard_tensor : bool, optional
        Whether to shard batches across the domain mesh.
    shard_dim : int, optional
        Spatial dimension along which tensors are partitioned for domain
        parallelism.  For ``(B, C, H, W)`` data sharded along the height
        axis, set ``shard_dim=2``.
    """

    def __init__(
        self,
        domain_parallel_size: int,
        use_shard_tensor: bool = False,
        shard_dim: int = 2,
    ):
        if not DistributedManager.is_initialized:
            DistributedManager.initialize()
        self.dist = DistributedManager()
        self.domain_parallel_size = domain_parallel_size
        self.shard_dim = shard_dim

        if self.dist.world_size % domain_parallel_size != 0:
            raise ValueError(
                "domain_parallel_size must evenly divide the number of processes"
            )
        self.data_parallel_size = self.dist.world_size // domain_parallel_size
        self.mesh = self.dist.initialize_mesh(
            mesh_shape=(self.data_parallel_size, domain_parallel_size),
            mesh_dim_names=["ddp", "domain"],
        )
        self.domain_rank = self.mesh["domain"].get_local_rank()
        self.use_shard_tensor = use_shard_tensor

    def get_domain_group_zero_rank(self) -> int:
        """Return the global rank of domain-group rank 0.

        Returns
        -------
        int
            Global rank for local domain rank 0.
        """
        return torch.distributed.get_global_rank(self.mesh["domain"].get_group(), 0)

    def local_batch_size(self, global_batch_size: int) -> int:
        """Compute per-rank batch size for data parallelism.

        Parameters
        ----------
        global_batch_size : int
            Global batch size across data-parallel ranks.

        Returns
        -------
        int
            Per-rank batch size.
        """
        return global_batch_size // self.data_parallel_size

    def sharded_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        seed: int | None = None,
        num_workers: int = 2,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create a rank-sharded DataLoader.

        Each rank accesses the dataset at indices [i_start : i_end] where
        i_start = int(rank / world_size * len(dataset))
        i_end = int((rank+1) / world_size * len(dataset))

        Therefore each rank gets a contiguous slice of samples, in contrast to torch
        DistributedSampler which gives a strided slice. This helps with caching as
        forecasting models frequently access subsequent time steps.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from.
        batch_size : int, optional
            Batch size per rank.
        seed : int or None, optional
            RNG seed base for shuffling.
        num_workers : int, optional
            Number of worker processes.
        shuffle : bool, optional
            Whether to shuffle local indices.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader that yields data from the local shard only.
        """

        # determine samples used by the current rank
        global_samples = np.arange(len(dataset))
        num_samples_global = len(global_samples)
        source_rank = (
            global_samples / num_samples_global * self.dist.world_size
        ).astype(int)
        local_samples = global_samples[source_rank == self.dist.rank]

        def sampler():
            """Iterate sample indices accessed by the current rank."""
            local_seed = None if seed is None else seed + self.dist.rank
            rng = np.random.default_rng(seed=local_seed)
            while True:
                if shuffle:
                    rng.shuffle(local_samples)
                yield from local_samples

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler(),
            num_workers=num_workers,
            worker_init_fn=worker_init,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def sharded_data_iter(
        self, dataloader: torch.utils.data.DataLoader, num_samples: int | None = None
    ) -> Iterator[torch.Tensor | dict | list]:
        """Iterate over sharded batches.

        If domain parallelism is used, each rank within a domain group receives the same
        sample from one rank within the group used as the source. The source rank rotates
        within the domain group so that each rank contributes equally to data loading.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader that yields local batches.
        num_samples : int or None, optional
            Optional number of batches to yield.

        Returns
        -------
        Iterator[torch.Tensor | dict | list]
            Iterator over (sharded if the shard attribute if True) batches.
        """
        data_iter = iter(dataloader)

        i = 0
        batch = None
        domain_group = self.mesh["domain"].get_group()
        while True:
            # the source rank within the domain group (always 0 when domain_parallel_size == 1)
            source_rank_in_mesh = i % self.domain_parallel_size
            # the global rank of the source
            source_rank = torch.distributed.get_global_rank(
                domain_group, source_rank_in_mesh
            )
            if source_rank == self.dist.rank or i == 0:
                # The source rank is the current rank: fetch a batch of data
                # We use prefetching in the dataloader so this should be fast
                batch = nested_to(
                    next(data_iter), device=self.dist.device, non_blocking=True
                )

            # scatter sample within the domain group (if using domain parallelism)
            yield (
                self.nested_scatter(batch, source_rank)
                if self.use_shard_tensor
                else batch
            )

            i += 1
            if i == num_samples:
                break

    def distribute_tensor(self, x: torch.Tensor) -> ShardTensor:
        """Scatter a tensor from domain rank 0.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to distribute.

        Returns
        -------
        ShardTensor
            Sharded or replicated tensor on domain mesh.
        """
        if self.use_shard_tensor:
            source_rank = self.get_domain_group_zero_rank()
            return self.nested_scatter(x, source_rank)
        else:
            return x

    def distribute_model(self, model: torch.nn.Module) -> FSDP:
        """Shard model parameters across the domain mesh and wrap with FSDP.

        Parameters
        ----------
        model : torch.nn.Module
            Model to distribute.

        Returns
        -------
        torch.distributed.fsdp.FullyShardedDataParallel
            Distributed model wrapper.
        """
        if self.use_shard_tensor:
            model = distribute_module(
                model,
                device_mesh=self.mesh["domain"],
                partition_fn=partition_model_selective,
            )
        return FSDP(
            model,
            device_mesh=self.mesh["ddp"],
            use_orig_params=False,  # Required for use with ShardTensor
            sharding_strategy=ShardingStrategy.NO_SHARD,
            sync_module_states=True,  # Ensure initialized weights match across ranks
            forward_prefetch=True,  # Optimization for faster training
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Backward prefetching for overlap
        )

    def make_domain_parallel_scheduler(self, scheduler: object) -> object:
        """Wrap a noise scheduler for domain-parallel diffusion.

        When ``use_shard_tensor`` is *False* the scheduler is returned unchanged.
        Otherwise it is wrapped with
        :class:`~physicsnemo.diffusion.DomainParallelNoiseScheduler` so that
        sampled times are broadcast and initial latents are sharded on
        ``self.shard_dim``.

        Parameters
        ----------
        scheduler : NoiseScheduler
            A noise scheduler implementing the
            :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
            protocol.

        Returns
        -------
        NoiseScheduler or DomainParallelNoiseScheduler
            The (possibly wrapped) scheduler.
        """
        if not self.use_shard_tensor:
            return scheduler

        return DomainParallelNoiseScheduler(
            scheduler,
            self.mesh["domain"],
            shard_dim=self.shard_dim,
        )

    def replicate_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Promote a plain tensor to a replicated DTensor on the domain mesh.

        When ``use_shard_tensor`` is False or *t* is already a DTensor,
        returns *t* unchanged.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to replicate.

        Returns
        -------
        torch.Tensor or DTensor
            Replicated DTensor on the domain mesh, or *t* unchanged.
        """
        if not self.use_shard_tensor or isinstance(t, DTensor):
            return t
        return DTensor.from_local(
            t, device_mesh=self.mesh["domain"], placements=[Replicate()]
        )

    def nested_scatter(
        self,
        x: torch.Tensor | Mapping | list | tuple | Any,
        global_rank_of_source: int,
        shard_dim: int | None = None,
    ) -> ShardTensor | dict | list | Any:
        """Scatter tensors within nested structures.

        Parameters
        ----------
        x : torch.Tensor or Mapping or list or tuple
            Input data to scatter.
        global_rank_of_source : int
            Global rank providing the source data.
        shard_dim : int or None, optional
            Dimension to shard for tensors with >= 3 dims.  Defaults to
            ``self.shard_dim``.

        Returns
        -------
        ShardTensor or dict or list
            Scattered structure with tensors sharded or replicated.
        """
        if shard_dim is None:
            shard_dim = self.shard_dim
        if isinstance(x, Mapping):
            return {
                k: self.nested_scatter(v, global_rank_of_source, shard_dim=shard_dim)
                for (k, v) in x.items()
            }
        elif isinstance(x, (list, tuple)):
            return [
                self.nested_scatter(v, global_rank_of_source, shard_dim=shard_dim)
                for v in x
            ]
        else:
            x_type = type(x)
            is_scalar = not isinstance(x, torch.Tensor)
            if is_scalar:
                x = torch.as_tensor(x, device=self.dist.device)

            placement = (
                Shard(shard_dim)
                if (x.ndim >= 3 and x.shape[shard_dim] > 1)
                else Replicate()
            )
            x = scatter_tensor(
                x,
                global_rank_of_source,
                self.mesh["domain"],
                placements=(placement,),  # Shard along height (H dimension)
                global_shape=x.shape,
                dtype=x.dtype,
            )

            if is_scalar:
                x = x_type(x.cpu())

            return x


def shard_dim_selector(param_name: str) -> int | None:
    """
    Return the dimension along which a model parameter should be sharded, if any.

    Parameters
    ----------
    param_name: str
        The name of the parameter.

    Returns
    -------
    int or None
        Shard dimension for param_name, or None if the tensor corresponding to
        param_name should not be sharded.
    """
    # this should find the spatial parameters for SongUNet and DiT
    sharded_params = ["pos_embed", "pos_embd", "spatial_emb"]
    if any(sharded_param in param_name for sharded_param in sharded_params):
        return 1
    else:
        return None


def partition_model_selective(
    name: str,  # pylint:disable=W0613
    submodule: torch.nn.Module,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
):
    """Shard positional embeddings across the domain mesh.

    Parameters
    ----------
    name : str
        Module name (unused by this selector).
    submodule : torch.nn.Module
        Submodule to inspect for sharding.
    device_mesh : torch.distributed.device_mesh.DeviceMesh
        Domain mesh used for distribution.
    """
    for key, param in submodule._parameters.items():
        if param is None:
            continue
        # Explicitly handle every parameter so that distribute_module's
        # internal replicate_module_params_buffers (which drops
        # requires_grad in PyTorch <= 2.10) never sees a plain tensor.
        # This prevents a bug where `distribute_module` silently flips
        # `requires_grad` on frozen params.
        if (shard_dim := shard_dim_selector(key)) is not None:
            dt = distribute_tensor(
                param, device_mesh=device_mesh, placements=[Shard(shard_dim)]
            )
        else:
            dt = distribute_tensor(
                param, device_mesh=device_mesh, placements=[Replicate()]
            )
        submodule.register_parameter(
            key, torch.nn.Parameter(dt, requires_grad=param.requires_grad)
        )
