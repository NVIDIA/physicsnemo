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

r"""Shared plumbing for the distributed mesh test suite.

The modules in this directory rerun tests imported from ``test/mesh/`` on
ShardTensor-backed inputs: each import module rebinds one seam in the base
module (an entry-point function or a mesh factory) to a sharding wrapper,
then re-exports the base test functions so every oracle assertion runs
verbatim against the distributed result.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor


@pytest.fixture
def device(distributed_mesh):
    """Each rank's own GPU, as a device-type string.

    Replaces the base suite's cpu/cuda ``device`` parametrization for tests
    imported into this directory: under torchrun every rank must target its
    local device, and the cpu variant is covered by the base suite already.
    The bare type string resolves per rank because DistributedManager sets
    the current CUDA device, and it satisfies the base suite's
    ``tensor.device.type == device`` assertions.
    """
    return DistributedManager().device.type


def shard_queries(
    query_points: torch.Tensor, mesh, placements=(Shard(0),)
) -> torch.Tensor:
    """Distribute a replicated plain tensor as a ShardTensor over ``mesh``."""
    return scatter_tensor(query_points, 0, mesh, placements)


def gather_full(result):
    """Gather every ShardTensor field of a (named) tuple back to plain tensors."""
    return type(result)(
        *(x.full_tensor() if hasattr(x, "full_tensor") else x for x in result)
    )


def _gather(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if hasattr(t, "full_tensor") else t


@pytest.fixture(autouse=True)
def _patch_torch_comparisons(monkeypatch):
    """Gather ShardTensors before any comparison that DTensor can't propagate.

    ``torch.allclose`` and ``torch.equal`` are data-dependent reductions
    (they return a scalar whose value depends on every shard) so DTensor's
    sharding propagator raises DataDependentOutputException.  Gathering to a
    plain replicated tensor on each rank before comparing is semantically
    identical and avoids the DTensor dispatch path entirely.
    """
    _orig_allclose = torch.allclose
    _orig_equal = torch.equal

    def _allclose(input, other, *args, **kwargs):
        return _orig_allclose(_gather(input), _gather(other), *args, **kwargs)

    def _equal(input, other):
        return _orig_equal(_gather(input), _gather(other))

    monkeypatch.setattr(torch, "allclose", _allclose)
    monkeypatch.setattr(torch, "equal", _equal)
