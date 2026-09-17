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

"""Domain-parallel tests for Transolver.

Transolver's parameters are all replicated, so it is distributed with DDP while
the point/functional inputs are sharded along the sequence axis; ShardTensor
auto-promotion handles the plain weights meeting sharded activations. The shared
harness runs the distributed forward/backward against a single-GPU reference.
"""

import numpy
import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.transolver import Transolver
from test.domain_parallel.models.harness import (
    DomainParallelModelCase,
    run_domain_parallel_model_check,
)


def _build_transolver(structured_shape):
    """Construct a small Transolver for distributed checks."""

    def build(device):
        return Transolver(
            structured_shape=structured_shape,
            n_layers=2,
            n_hidden=32,
            dropout=0,
            n_head=4,
            time_input=False,
            act="gelu",
            mlp_ratio=1,
            functional_dim=3,
            embedding_dim=5,
            out_dim=2,
            slice_num=32,
            ref=1,
            unified_pos=False,
            use_te=False,
        ).to(device)

    return build


# Per-axis side length by dimensionality. The harness builds a *full* (gathered)
# single-GPU reference and runs forward+backward on it, so the total token count
# N = prod(spatial_dims) drives activation memory. Keep N in the same ballpark
# across cases (2D: 128**2 = 16384) rather than a fixed 128 per axis; 128**3 is
# ~2.1M tokens and OOMs the reference. Sides stay divisible by common GPU counts
# (2/4/8) since inputs are sharded along the first spatial axis.
_SIDE_BY_NDIMS = {2: 128, 3: 24}


def _nd_case(n_dims):
    """A structured n-D case; inputs sharded along the sequence axis (dim 1)."""
    spatial_dims = (_SIDE_BY_NDIMS[n_dims],) * n_dims

    def build_inputs(device):
        image_embedding = torch.randn(1, *spatial_dims, 5, device=device)
        functional_input = torch.randn(1, *spatial_dims, 3, device=device)
        return (image_embedding, functional_input), {}

    def shard_inputs(args, kwargs, mesh):
        image_embedding, functional_input = args
        sharded_image = scatter_tensor(
            image_embedding, 0, mesh, (Shard(1),), requires_grad=False
        )
        sharded_functional = scatter_tensor(
            functional_input, 0, mesh, (Shard(1),), requires_grad=False
        )
        # Collapse the spatial grid into a single (sharded) sequence axis.
        sharded_image = sharded_image.reshape(1, -1, 5)
        sharded_functional = sharded_functional.reshape(1, -1, 3)
        return (sharded_image, sharded_functional), kwargs

    def check_output(d_out):
        assert d_out.shape == (1, numpy.prod(spatial_dims), 2)
        assert d_out._spec.placements == (Shard(1),)

    return DomainParallelModelCase(
        name=f"structured_{n_dims}d",
        build_model=_build_transolver(spatial_dims),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        output_check_fn=check_output,
        # Structured attention uses convs; the sharded path runs them on a
        # local shard + halo while the reference runs the full grid, and cuDNN
        # picks different algorithms per input size. The resulting fp32
        # discrepancy (~1e-5 per conv) accumulates across layers, so relax the
        # tolerance relative to the linear-only irregular case.
        atol=1e-3,
        rtol=1e-3,
    )


def _irregular_case():
    """An unstructured (point-cloud) case; sequence axis sharded (dim 1)."""
    spatial_dims = (16384,)

    def build_inputs(device):
        image_embedding = torch.randn(1, *spatial_dims, 5, device=device)
        functional_input = torch.randn(1, *spatial_dims, 3, device=device)
        return (image_embedding, functional_input), {}

    def shard_inputs(args, kwargs, mesh):
        image_embedding, functional_input = args
        sharded_image = scatter_tensor(
            image_embedding, 0, mesh, (Shard(1),), requires_grad=False
        )
        sharded_functional = scatter_tensor(
            functional_input, 0, mesh, (Shard(1),), requires_grad=False
        )
        return (sharded_image, sharded_functional), kwargs

    def check_output(d_out):
        assert d_out.shape == (1, *spatial_dims, 2)
        assert d_out._spec.placements == (Shard(1),)

    return DomainParallelModelCase(
        name="irregular",
        build_model=_build_transolver(None),
        build_inputs=build_inputs,
        shard_inputs=shard_inputs,
        output_check_fn=check_output,
    )


_CASES = [_nd_case(2), _nd_case(3), _irregular_case()]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_transolver_distributed(distributed_mesh, case):
    """Distributed forward/backward matches a single-GPU reference."""
    run_domain_parallel_model_check(case, mesh=distributed_mesh)
