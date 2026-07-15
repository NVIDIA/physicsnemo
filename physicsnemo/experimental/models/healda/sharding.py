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
"""Separable context-parallel resharding for ``(b, t, x, c)`` activations.

A factorized video transformer alternates between spatial attention (which is
independent per frame) and temporal attention (which is independent per spatial
location). Each prefers the opposite activation layout under context
parallelism:

- **t-sharded** (the block input): each rank holds all spatial tokens for a
  slice of the time axis -- ``(b, t / N, x, c)`` -- so spatial / observation
  attention is local.
- **x-sharded**: each rank holds all time steps for a slice of the spatial axis
  -- ``(b, t, x / N, c)`` -- so temporal attention is local.

:func:`shard_x` and :func:`shard_t` swap between the two layouts with a single
``all_to_all`` over the context-parallel group. They are autograd-aware via
``torch.distributed.nn.functional.all_to_all_single``.
"""

import einops
import torch
import torch.distributed as dist
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_to_all_single

DATA_DIM = 0
MODEL_DIM = 1


def shard_x(
    tensor: Float[torch.Tensor, "batch time space hidden_size"],
    group: dist.ProcessGroup,
) -> Float[torch.Tensor, "batch time space hidden_size"]:
    r"""Reshard from t-sharded to x-sharded (gather time, scatter space).

    Parameters
    ----------
    tensor : torch.Tensor
        Activation of shape :math:`(b, t, N \cdot x, c)`, currently t-sharded
        (this rank holds the full spatial axis for its time slice).
    group : torch.distributed.ProcessGroup
        Context-parallel process group of size :math:`N`.

    Returns
    -------
    torch.Tensor
        Activation of shape :math:`(b, N \cdot t, x, c)`, now x-sharded (this
        rank holds the full time axis for its spatial slice).
    """
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b t (n x) c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b (n t) x c")
    return output


def shard_t(
    tensor: Float[torch.Tensor, "batch time space hidden_size"],
    group: dist.ProcessGroup,
) -> Float[torch.Tensor, "batch time space hidden_size"]:
    r"""Reshard from x-sharded to t-sharded (gather space, scatter time).

    Parameters
    ----------
    tensor : torch.Tensor
        Activation of shape :math:`(b, N \cdot t, x, c)`, currently x-sharded.
    group : torch.distributed.ProcessGroup
        Context-parallel process group of size :math:`N`.

    Returns
    -------
    torch.Tensor
        Activation of shape :math:`(b, t, N \cdot x, c)`, now t-sharded.
    """
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b (n t) x c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b t (n x) c")
    return output


# ---------------------------------------------------------------------------
# ShardTensor variant: express the same t<->x reshard as a placement change on a
# 1D device mesh (Shard(time) <-> Shard(space)). The redistribute is an
# all_to_all under the hood but is autograd-aware and composes with the rest of
# physicsnemo.domain_parallel; unlike the manual shard_x/shard_t above (which
# require evenly divisible shards), it also handles uneven shards. ``(b, t, x, c)``
# so the time axis is dim 1 and the spatial axis is dim 2.
# ---------------------------------------------------------------------------
_TIME_DIM = 1
_SPACE_DIM = 2


@torch._dynamo.disable
def _reshard_via_shardtensor(
    tensor: torch.Tensor, mesh: DeviceMesh, from_dim: int, to_dim: int
) -> torch.Tensor:
    # The redistribute is a DTensor collective; dynamo cannot usefully trace it
    # (and tracing the resumed frame mis-resolves names), so run it eager.
    from torch.distributed.tensor.placement_types import Shard

    from physicsnemo.domain_parallel import ShardTensor

    st = ShardTensor.from_local(tensor, mesh, [Shard(from_dim)])
    st = st.redistribute(placements=[Shard(to_dim)])
    return st.to_local()


def shard_x_shardtensor(
    tensor: Float[torch.Tensor, "batch time space hidden_size"], mesh: DeviceMesh
) -> Float[torch.Tensor, "batch time space hidden_size"]:
    r"""ShardTensor equivalent of :func:`shard_x` (Shard(time) -> Shard(space))."""
    return _reshard_via_shardtensor(tensor, mesh, _TIME_DIM, _SPACE_DIM)


def shard_t_shardtensor(
    tensor: Float[torch.Tensor, "batch time space hidden_size"], mesh: DeviceMesh
) -> Float[torch.Tensor, "batch time space hidden_size"]:
    r"""ShardTensor equivalent of :func:`shard_t` (Shard(space) -> Shard(time))."""
    return _reshard_via_shardtensor(tensor, mesh, _SPACE_DIM, _TIME_DIM)
