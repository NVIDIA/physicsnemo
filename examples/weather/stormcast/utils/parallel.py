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
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    OptimStateKeyType,
)
from torch.distributed.tensor import distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import ShardTensor, scatter_tensor

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
    """

    def __init__(self, domain_parallel_size: int, use_shard_tensor: bool = False):
        if not DistributedManager.is_initialized:
            DistributedManager.initialize()
        self.dist = DistributedManager()
        self.domain_parallel_size = domain_parallel_size

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
        source_rank = self.get_domain_group_zero_rank()
        return self.nested_scatter(x, source_rank)

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
            use_orig_params=False,  # Set to True if you want to see individual params
            sharding_strategy=ShardingStrategy.NO_SHARD,
            sync_module_states=False,  # load after sharding
            forward_prefetch=True,  # Optimization for faster training
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Backward prefetching for overlap
        )

    def scatter_object(self, x: Any | None) -> Any:
        """Scatter a Python object from rank 0 to all ranks.

        Parameters
        ----------
        x : Any or None
            Object to scatter from rank 0.

        Returns
        -------
        Any
            Object received by the local rank.
        """
        states_to_sync = [x] * self.dist.world_size if self.dist.rank == 0 else None
        output_list = [None]
        torch.distributed.barrier()
        torch.distributed.scatter_object_list(output_list, states_to_sync, src=0)
        return output_list[0]

    def shard_state_dict(self, state_dict: dict[str, Any] | None) -> dict[str, Any]:
        """Shard a state dict across the domain mesh and scatter.

        Parameters
        ----------
        state_dict : dict[str, Any] or None
            Full state dict provided on rank 0.

        Returns
        -------
        dict[str, Any]
            Sharded state dict for the local rank.
        """
        if self.dist.rank == 0:
            # shard of state dict for each domain rank
            shards = [
                self.get_state_dict_shard(state_dict, domain_rank=i)
                for i in range(self.domain_parallel_size)
            ]
            # shard of state dict for each global rank
            shards = [
                shards[i % self.domain_parallel_size]
                for i in range(self.dist.world_size)
            ]

        states_to_sync = shards if self.dist.rank == 0 else None
        output_list = [None]
        torch.distributed.barrier()
        torch.distributed.scatter_object_list(output_list, states_to_sync, src=0)
        return output_list[0]

    def scatter_optimizer_state(
        self,
        model_full: torch.nn.Module | None,
        optimizer_full: torch.optim.Optimizer | None,
        scheduler_full: torch.optim.lr_scheduler.LRScheduler | None,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ):
        """Scatter and load optimizer and scheduler state.

        Parameters
        ----------
        model_full : torch.nn.Module or None
            Full model on rank 0 (used for rekeying).
        optimizer_full : torch.optim.Optimizer or None
            Full optimizer on rank 0.
        scheduler_full : torch.optim.lr_scheduler.LRScheduler or None
            Full scheduler on rank 0.
        model : torch.nn.Module
            Local model instance.
        optimizer : torch.optim.Optimizer
            Local optimizer instance.
        scheduler : torch.optim.lr_scheduler.LRScheduler or None
            Local scheduler instance.
        """
        if self.dist.rank == 0:
            optim_state_dict = optimizer_full.state_dict()
            if isinstance(model, FSDP):
                optim_state_dict = FSDP.rekey_optim_state_dict(
                    optim_state_dict, OptimStateKeyType.PARAM_NAME, model_full
                )

        if self.use_shard_tensor:
            # shard positional embeddings
            optim_state_dict = self.shard_state_dict(
                optim_state_dict if self.dist.rank == 0 else None
            )
        else:
            optim_state_dict = self.scatter_object(
                optim_state_dict if self.dist.rank == 0 else None
            )

        options = StateDictOptions(full_state_dict=True)
        set_optimizer_state_dict(model, optimizer, optim_state_dict, options=options)

        if scheduler is not None:
            sched_state_dict_full = (
                None if scheduler_full is None else scheduler_full.state_dict()
            )
            sched_state_dict_full = self.scatter_object(sched_state_dict_full)
            scheduler.load_state_dict(sched_state_dict_full)

    def gather_training_state(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        model_full: torch.nn.Module | None,
        optimizer_full: torch.optim.Optimizer | None,
        scheduler_full: torch.optim.lr_scheduler.LRScheduler | None,
    ):
        """Gather model and optimizer state onto rank 0.

        Parameters
        ----------
        model : torch.distributed.fsdp.FullyShardedDataParallel
            Distributed model wrapper.
        optimizer : torch.optim.Optimizer
            Local optimizer.
        scheduler : torch.optim.lr_scheduler.LRScheduler or None
            Local scheduler.
        model_full : torch.nn.Module or None
            Full model to populate on rank 0, or None if rank != 0.
        optimizer_full : torch.optim.Optimizer or None
            Full optimizer to populate on rank 0, or None if rank != 0.
        scheduler_full : torch.optim.lr_scheduler.LRScheduler or None
            Full scheduler to populate on rank 0, or None if rank != 0.
        """
        # TODO: we should be using the cpu_offload=True option but it seems to cause this to hang
        options = StateDictOptions(full_state_dict=True)
        (state_dict, optim_state_dict) = get_state_dict(
            model, optimizer, options=options
        )
        if self.dist.rank == 0:
            model_full.load_state_dict(state_dict)
            optimizer_full.load_state_dict(optim_state_dict)
            if scheduler is not None:
                scheduler_full.load_state_dict(scheduler.state_dict())

    def nested_scatter(
        self,
        x: torch.Tensor | Mapping | list | tuple | Any,
        global_rank_of_source: int,
        shard_dim: int | None = 2,
    ) -> ShardTensor | dict | list | Any:
        """Scatter tensors within nested structures.

        Parameters
        ----------
        x : torch.Tensor or Mapping or list or tuple
            Input data to scatter.
        global_rank_of_source : int
            Global rank providing the source data.
        shard_dim : int or None, optional
            Dimension to shard for tensors with >= 3 dims.

        Returns
        -------
        ShardTensor or dict or list
            Scattered structure with tensors sharded or replicated.
        """
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
                if (x.ndim >= 3 and shard_dim is not None)
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

    def get_state_dict_shard(
        self,
        x: Any,
        domain_rank: int | None = None,
        _key: str = "",
    ) -> Any:
        """Extract shard of a nested state dict for one domain rank.

        Parameters
        ----------
        x : Any
            State dict or nested structure.
        domain_rank : int or None, optional
            Domain rank to shard for.

        Returns
        -------
        Any
            Sharded structure for the target domain rank.
        """
        if domain_rank is None:
            domain_rank = self.domain_rank

        kwargs = {"domain_rank": domain_rank}
        if isinstance(x, Mapping):
            return {
                k: self.get_state_dict_shard(v, _key=(_key + "." + k), **kwargs)
                for (k, v) in x.items()
            }
        elif isinstance(x, (list, tuple)):
            return [
                self.get_state_dict_shard(v, _key=(_key + "." + str(i)), **kwargs)
                for (i, v) in enumerate(x)
            ]
        else:
            shard_dim = shard_dim_selector(_key)
            if (
                isinstance(x, torch.Tensor)
                and (shard_dim is not None)
                and (shard_dim < x.ndim)
            ):
                shard_size = x.shape[shard_dim] // self.domain_parallel_size
                i0 = domain_rank * shard_size
                i1 = i0 + shard_size
                shard_slice = tuple(
                    slice(i0, i1) if i == shard_dim else slice(None)
                    for i in range(x.ndim)
                )
                return x[shard_slice]
            else:
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
        if (shard_dim := shard_dim_selector(key)) is not None:
            sharded = distribute_tensor(
                param,
                device_mesh=device_mesh,
                placements=[Shard(shard_dim)],
            )
            submodule.register_parameter(key, torch.nn.Parameter(sharded))
