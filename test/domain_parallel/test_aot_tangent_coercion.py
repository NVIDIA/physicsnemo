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

r"""Tests for the AOTAutograd plain->ShardTensor runtime-tangent shim.

The shim rebuilds a ShardTensor from a plain runtime tangent + traced
``SubclassCreationMeta`` when AOT materializes a subclass cotangent as a plain tensor.
These tests guard the shim install and its lossless reconstruction.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_tensor import (
    install_aot_plain_tangent_coercion,
)


def test_aot_plain_tangent_shim_installed_and_idempotent() -> None:
    """Importing ``shard_tensor`` installs the shim; re-installing is a no-op."""
    from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd

    fn = AOTDispatchAutograd.process_runtime_tangent
    assert getattr(fn, "_shardtensor_plain_tangent_shim", False) is True

    install_aot_plain_tangent_coercion()  # idempotent
    assert AOTDispatchAutograd.process_runtime_tangent is fn


def run_flatten_unflatten_roundtrip_is_lossless(mesh):
    # Rebuilding a replicated ShardTensor via __tensor_unflatten__ (as the shim does).
    local = torch.randn(5, 4, dtype=torch.float64, device="cuda")
    st = ShardTensor.from_local(local, mesh, [Replicate()] * mesh.ndim)

    inner_names, flatten_spec = st.__tensor_flatten__()
    inner = {name: getattr(st, name) for name in inner_names}
    rebuilt = type(st).__tensor_unflatten__(inner, flatten_spec, st.shape, st.stride())

    assert isinstance(rebuilt, ShardTensor)
    assert rebuilt.placements == st.placements
    assert rebuilt.shape == st.shape
    torch.testing.assert_close(rebuilt._local_tensor, st._local_tensor)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_flatten_unflatten_roundtrip_is_lossless_1d(distributed_mesh):
    run_flatten_unflatten_roundtrip_is_lossless(distributed_mesh)


def run_shim_reconstructs_shardtensor_from_plain_tangent(mesh):
    # Shim hot path: meta + a plain tangent comes back as a ShardTensor whose local equals the plain tensor.
    from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd
    from torch._subclasses.fake_tensor import FakeTensorMode

    try:  # AOT internal API -- skip (don't fail) if it drifts across torch versions
        from torch._functorch._aot_autograd.subclass_utils import create_subclass_meta

        # Meta must be built over a fake subclass; ``sharding_shapes="chunk"`` stays collective-free.
        with FakeTensorMode():
            local = torch.randn(6, 3, dtype=torch.float64, device="cuda")
            st = ShardTensor.from_local(
                local,
                mesh,
                [Replicate()] * mesh.ndim,
                sharding_shapes="chunk",
                global_shape=(6, 3),
            )
            meta = create_subclass_meta([st], with_memory_format=True)[0]
        # AOT fills ``original_subclass_type`` during tracing; the shim keys off it.
        meta.original_subclass_type = ShardTensor
    except Exception as exc:  # pragma: no cover - version-drift guard
        pytest.skip(f"AOT subclass-meta construction API changed: {exc!r}")

    plain = torch.randn(6, 3, dtype=torch.float64, device="cuda")
    result = AOTDispatchAutograd.process_runtime_tangent(plain, meta)
    out = result[0] if isinstance(result, tuple) else result

    assert isinstance(out, ShardTensor)
    torch.testing.assert_close(out._local_tensor, plain)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_shim_reconstructs_shardtensor_from_plain_tangent_1d(distributed_mesh):
    run_shim_reconstructs_shardtensor_from_plain_tangent(distributed_mesh)
