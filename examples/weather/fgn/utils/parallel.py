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

"""Data- and domain-parallel helpers for FGN training.

Slim adaptation of ``examples/weather/stormcast/utils/parallel.py`` tailored
to the FGN recipe: same FSDP-plus-ShardTensor strategy, same sharded-
dataloader conventions, but without the diffusion noise-scheduler plumbing
StormCast needs.

See StormCast's ``utils/parallel.py`` for the reference implementation and
docstrings; this file mirrors its public API so the two recipes can stay in
lockstep.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from datasets.dataset import worker_init
from torch.distributed.fsdp import (
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.tensor import DTensor, distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from utils.nn import nested_to

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_tensor import scatter_tensor


class ParallelHelper:
    """Manage data + domain parallelism for the FGN recipe.

    Mirrors StormCast's ``ParallelHelper`` so FGN inherits the same tested
    pattern: a 2D device mesh with a ``ddp`` axis and a ``domain`` axis, FSDP
    on the ddp axis, optional ShardTensor spatial sharding on the domain
    axis.

    Parameters
    ----------
    domain_parallel_size : int
        Number of ranks in the domain-parallel dimension. Use 1 for pure DDP
        or single-process runs.
    use_shard_tensor : bool
        Whether to shard batches and selected module parameters across the
        domain mesh. Typically ``domain_parallel_size > 1`` OR
        ``force_sharding`` is true.
    shard_dim : int, default 2
        Spatial dimension along which tensors are partitioned for domain
        parallelism. For ``(B, C, H, W)`` sharded along height, use ``2``.
    """

    def __init__(
        self,
        domain_parallel_size: int,
        use_shard_tensor: bool = False,
        shard_dim: int = 2,
    ):
        if not DistributedManager.is_initialized():
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
        return torch.distributed.get_global_rank(self.mesh["domain"].get_group(), 0)

    def local_batch_size(self, global_batch_size: int) -> int:
        return global_batch_size // self.data_parallel_size

    def sharded_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        seed: int | None = None,
        num_workers: int = 2,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Build a rank-sharded DataLoader.

        Each rank sees a contiguous slice of ``range(len(dataset))`` (rather
        than a strided slice as in ``DistributedSampler``), which plays
        nicely with caches that key on neighbouring time indices.
        """
        global_samples = np.arange(len(dataset))
        num_samples_global = len(global_samples)
        source_rank = (
            global_samples / num_samples_global * self.dist.world_size
        ).astype(int)
        local_samples = global_samples[source_rank == self.dist.rank]

        def sampler() -> Iterator[int]:
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
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int | None = None,
    ) -> Iterator[Any]:
        data_iter = iter(dataloader)
        i = 0
        batch: Any = None
        domain_group = self.mesh["domain"].get_group()
        while True:
            source_rank_in_mesh = i % self.domain_parallel_size
            source_rank = torch.distributed.get_global_rank(
                domain_group, source_rank_in_mesh
            )
            if source_rank == self.dist.rank or i == 0:
                batch = nested_to(
                    next(data_iter),
                    device=self.dist.device,
                    non_blocking=True,
                )

            yield (
                self.nested_scatter(batch, source_rank)
                if self.use_shard_tensor
                else batch
            )

            i += 1
            if i == num_samples:
                break

    def distribute_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_shard_tensor:
            return self.nested_scatter(x, self.get_domain_group_zero_rank())
        return x

    def distribute_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap a model with FSDP, with optional ShardTensor domain sharding."""
        if self.use_shard_tensor:
            model = distribute_module(
                model,
                device_mesh=self.mesh["domain"],
                partition_fn=partition_model_selective,
            )
        return FSDP(
            model,
            device_mesh=self.mesh["ddp"],
            use_orig_params=False,  # required for ShardTensor compatibility
            sharding_strategy=ShardingStrategy.NO_SHARD,
            sync_module_states=True,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

    def replicate_tensor(self, t: torch.Tensor) -> torch.Tensor:
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
    ) -> Any:
        if shard_dim is None:
            shard_dim = self.shard_dim
        if isinstance(x, Mapping):
            return {
                k: self.nested_scatter(v, global_rank_of_source, shard_dim=shard_dim)
                for (k, v) in x.items()
            }
        if isinstance(x, (list, tuple)):
            return [
                self.nested_scatter(v, global_rank_of_source, shard_dim=shard_dim)
                for v in x
            ]

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
            placements=(placement,),
            global_shape=x.shape,
            dtype=x.dtype,
        )
        if is_scalar:
            x = x_type(x.cpu())
        return x


def shard_dim_selector(param_name: str) -> int | None:
    """Return the spatial axis along which a parameter should be sharded, if any.

    Matches the FGN backbone's spatial-parameter naming. Currently returns
    ``None`` since the U-Net has no spatial positional embeddings; add names
    here when the CLN / graph-transformer backbone lands (e.g.
    ``"pos_embed"``, ``"mesh_pos_embed"``).
    """
    sharded_params: tuple[str, ...] = ()  # e.g. ("pos_embed",) in the future
    return 1 if any(p in param_name for p in sharded_params) else None


def partition_model_selective(
    name: str,  # noqa: ARG001 — signature required by distribute_module
    submodule: torch.nn.Module,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
) -> None:
    """Parameter-by-parameter domain-mesh placement selector.

    Mirrors StormCast's ``partition_model_selective``: every parameter is
    wrapped in a ``DTensor`` (Shard or Replicate) so that
    ``distribute_module``'s internal ``replicate_module_params_buffers``
    never sees a plain tensor and cannot silently flip ``requires_grad`` on
    frozen params.
    """
    for key, param in submodule._parameters.items():
        if param is None:
            continue
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
