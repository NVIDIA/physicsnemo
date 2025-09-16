# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
Test unary operations on ShardTensor.  This file tests torch.unsqueeze with
both regular dimensions as well as negative dimensions.

The input tensors are not significant in any way, we just need them to be
ShardTensors.
"""

import sys

sys.path.append("../")

import pytest

from physicsnemo.utils.version_check import check_module_requirements

try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")

except ImportError:
    pytest.skip(
        "Skipping test because physicsnemo.distributed.shard_tensor is not available",
        allow_module_level=True,
    )

import torch

from ..test_redistribute import shard_tensor_factory


@pytest.mark.multigpu_static
def test_shard_tensor_unsqueeze(distributed_mesh, verbose=True):
    run_shard_tensor_unsqueeze(distributed_mesh, verbose=verbose)


@pytest.mark.multigpu_static
def test_shard_tensor_unsqueeze_2d(distributed_mesh_2d, verbose=True):
    run_shard_tensor_unsqueeze(distributed_mesh_2d, verbose=verbose)


def run_shard_tensor_unsqueeze(mesh, verbose=False):
    shard_tensor = shard_tensor_factory(mesh)

    # For this test, we're testing that the unsqueeze of the tensor works correctly
    if verbose:
        print(
            f"Shard tensor shape is {shard_tensor.shape} and local tensor shape is {shard_tensor._local_tensor.shape}"
        )
    full_original_tensor = shard_tensor.full_tensor()

    indexes = list(range(len(full_original_tensor.shape)))

    for i in indexes:
        i_sharded_unsqueeze = shard_tensor.unsqueeze(i)
        i_unsharded_unsqueeze = full_original_tensor.unsqueeze(i)

        assert i_sharded_unsqueeze.shape == i_sharded_unsqueeze.shape
        assert torch.allclose(i_sharded_unsqueeze.full_tensor(), i_unsharded_unsqueeze)

        ni_sharded_unsqueeze = shard_tensor.unsqueeze(-i)
        ni_unsharded_unsqueeze = full_original_tensor.unsqueeze(-i)

        assert ni_sharded_unsqueeze.shape == ni_sharded_unsqueeze.shape
        assert torch.allclose(
            ni_sharded_unsqueeze.full_tensor(), ni_unsharded_unsqueeze
        )
