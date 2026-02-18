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

"""
Test view and reshape operations on ShardTensor.

Tests cover tensor.view, tensor.reshape, and torch.reshape with sharding
on various dimensions. The shard dimension is never the one being merged
or split — it is preserved 1:1 through the view, or the view operates
exclusively on non-sharded dimensions.

Backward (gradient) correctness is tested for every configuration.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.domain_parallel.shard_tensor import ShardTensor
from physicsnemo.domain_parallel.shard_utils.view_ops import _match_view_dim_groups

from .utils import numerical_shard_tensor_check


@pytest.mark.parametrize(
    "old_shape,new_shape,expected",
    [
        ((6,), (2, 3), [([0], [0, 1])]),
        ((4, 8), (32,), [([0, 1], [0])]),
        ((2, 0), (1, 0), [([0, 1], [0, 1])]),
        ((2, 0, 3), (1, 0, 3), [([0, 1], [0, 1]), ([2], [2])]),
        ((4, 5), (4, 5), [([0], [0]), ([1], [1])]),
    ],
)
def test_match_view_dim_groups_compatible(old_shape, new_shape, expected):
    """Unit test: compatible view shapes produce the expected dimension groups."""
    assert _match_view_dim_groups(old_shape, new_shape) == expected


@pytest.mark.parametrize(
    "old_shape,new_shape",
    [
        ((2, 3), (5,)),
        ((2, 0), (1, 10)),
    ],
)
def test_match_view_dim_groups_incompatible(old_shape, new_shape):
    """Unit test: incompatible view shapes raise ValueError."""
    with pytest.raises(ValueError, match="are not compatible"):
        _match_view_dim_groups(old_shape, new_shape)


class ViewWrapper(torch.nn.Module):
    """Wrapper class for testing tensor.view operation."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.view(self.target_shape)


class ViewVariadicWrapper(torch.nn.Module):
    """Wrapper for testing tensor.view(*shape) with variadic int arguments."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.view(*self.target_shape)


class ReshapeWrapper(torch.nn.Module):
    """Wrapper class for testing tensor.reshape(shape) with shape as a single tuple."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.reshape(self.target_shape)


class ReshapeVariadicWrapper(torch.nn.Module):
    """Wrapper for testing tensor.reshape(*shape) with variadic int arguments."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return tensor.reshape(*self.target_shape)


class TorchReshapeWrapper(torch.nn.Module):
    """Wrapper class for testing torch.reshape(tensor, shape) with shape as tuple."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return torch.reshape(tensor, self.target_shape)


class TorchReshapeListWrapper(torch.nn.Module):
    """Wrapper for testing torch.reshape(tensor, shape) with shape as list."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return torch.reshape(tensor, list(self.target_shape))


class TorchReshapeKwargWrapper(torch.nn.Module):
    """Wrapper for testing torch.reshape(tensor, shape=...) with shape as kwarg."""

    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor: torch.Tensor):
        return torch.reshape(tensor, shape=self.target_shape)


class ViewRoundTrip(torch.nn.Module):
    """View to merge last two dims, then view back to the original shape.

    Exercises view in a differentiable pipeline so gradients must flow
    back through two consecutive view backward passes.
    """

    def __init__(self, original_shape: tuple[int, ...]):
        super().__init__()
        self.original_shape = original_shape

    def forward(self, tensor: torch.Tensor):
        b, t = tensor.shape[:2]
        merged = tensor.reshape(b, t, -1)
        return merged.view(self.original_shape)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_merge_last_two_dims(
    distributed_mesh,
    backward,
):
    """Test tensor.view merging the last two dims (einops-like 'b t h d -> b t (h d)')."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_split_last_dim(
    distributed_mesh,
    backward,
):
    """Test tensor.view splitting the last dim (einops-like 'b t (h d) -> b t h d')."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 32)
    target_shape = (4, 128, 8, 4)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_flatten_to_2d(
    distributed_mesh,
    backward,
):
    """Test tensor.view flattening spatial dims into one."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8)
    target_shape = (4, -1)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_neg1_infer(
    distributed_mesh,
    backward,
):
    """Test tensor.view with -1 dimension inference."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, -1, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_reshape_merge_last_two_dims(
    distributed_mesh,
    backward,
):
    """Test tensor.reshape merging the last two dims."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ReshapeWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_torch_reshape_operation(
    distributed_mesh,
    backward,
):
    """Test torch.reshape(tensor, shape) on a ShardTensor."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = TorchReshapeWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "wrapper_cls,arg_style",
    [
        (ViewWrapper, "tuple"),
        (ViewVariadicWrapper, "variadic"),
        (ReshapeWrapper, "tuple"),
        (ReshapeVariadicWrapper, "variadic"),
        (TorchReshapeWrapper, "tuple"),
        (TorchReshapeListWrapper, "list"),
        (TorchReshapeKwargWrapper, "kwarg"),
    ],
    ids=[
        "view_tuple",
        "view_variadic",
        "reshape_tuple",
        "reshape_variadic",
        "torch_reshape_tuple",
        "torch_reshape_list",
        "torch_reshape_kwarg",
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_view_reshape_argument_permutations(
    distributed_mesh,
    wrapper_cls,
    arg_style,
    backward,
):
    """Test all argument permutations: view/reshape with shape as tuple, variadic, list, or kwarg.

    Covers tensor.view(shape), tensor.view(*shape), tensor.reshape(shape),
    tensor.reshape(*shape), torch.reshape(tensor, shape),
    torch.reshape(tensor, list(shape)), and torch.reshape(tensor, shape=...).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = wrapper_cls(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_shard_on_non_viewed_dim(
    distributed_mesh,
    backward,
):
    """Test view when shard dim is not involved in the reshape at all."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 128, 8, 4)
    target_shape = (4, 128, 32)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    # Shard on dim 0 (batch) — the view only touches dims 2+3.
    placements = (Shard(0),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewWrapper(target_shape=target_shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_round_trip(
    distributed_mesh,
    backward,
):
    """Test that gradients flow through two consecutive views (merge then split)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (2, 64, 8, 4)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(1),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    module = ViewRoundTrip(original_shape=shape)

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_tensor],
        {},
        check_grads=backward,
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_view_trailing_dims_1d_to_3d(
    distributed_mesh,
    backward,
):
    """Test view (6,) -> (2, 3, 1) with Shard(0): trailing dim must stay in group.

    With the shard on dim 0, each rank has a contiguous chunk of the 1D tensor.
    The target shape has a trailing singleton (2, 3, 1). The trailing dimension
    must be included in the same dimension group so that the local element
    count is correct (product of local shape equals chunk_size). Without that,
    the old code produced wrong local shapes (e.g. product 4 instead of 2 or 3).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (6,)
    target_shape = (2, 3, 1)

    original_tensor = torch.rand(shape, device=dm.device, requires_grad=backward)

    placements = (Shard(0),)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=placements,
        requires_grad=backward,
    )

    expected_local_numel = sharded_tensor._local_tensor.numel()

    viewed = sharded_tensor.view(target_shape)

    assert viewed.shape == target_shape, (
        f"expected global shape {target_shape}, got {viewed.shape}"
    )
    assert viewed._local_tensor.numel() == expected_local_numel, (
        f"local numel mismatch: viewed has {viewed._local_tensor.numel()}, "
        f"expected {expected_local_numel} (original local had {expected_local_numel} elements)"
    )

    if backward:
        module = ViewWrapper(target_shape=target_shape)
        numerical_shard_tensor_check(
            distributed_mesh,
            module,
            [sharded_tensor],
            {},
            check_grads=True,
        )


@pytest.mark.multigpu_static
@pytest.mark.parametrize(
    "input_dtype,output_dtype,shape",
    [
        (torch.int32, torch.float32, (8,)),  # same element count
        (
            torch.int64,
            torch.float32,
            (4,),
        ),  # 4 int64 -> 4 float32 (shape same, bytes 32)
        (torch.float32, torch.int32, (8,)),  # same element count
    ],
    ids=["int32_to_float32", "int64_to_float32", "float32_to_int32"],
)
def test_view_dtype(distributed_mesh, input_dtype, output_dtype, shape):
    """Test view(dtype) on ShardTensor returns 1D ShardTensor with expected dtype.

    .view(torch.dtype) reinterprets storage; PyTorch returns 1D. We assert
    the call succeeds and the result has the correct dtype and 1D shape
    consistent with the same op on a plain tensor.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    if input_dtype in (torch.int32, torch.int64):
        original_tensor = torch.randint(
            0, 100, shape, device=dm.device, dtype=input_dtype
        )
    else:
        original_tensor = torch.randn(shape, device=dm.device, dtype=input_dtype)

    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    result = sharded_tensor.view(output_dtype)

    assert isinstance(result, ShardTensor), "view(dtype) should return a ShardTensor"
    assert result.dtype == output_dtype, f"expected {output_dtype}, got {result.dtype}"
    assert result.ndim == 1, f"view(dtype) returns 1D; got ndim={result.ndim}"
    expected_size = (
        original_tensor.numel() * input_dtype.itemsize // output_dtype.itemsize
    )
    assert result.shape[0] == expected_size, (
        f"expected size {expected_size}, got {result.shape[0]}"
    )
    plain_viewed = original_tensor.view(output_dtype)
    assert result.shape == plain_viewed.shape, (
        f"ShardTensor view(dtype) shape {result.shape} should match plain {plain_viewed.shape}"
    )


@pytest.mark.multigpu_static
def test_view_dtype_invalid_byte_size(distributed_mesh):
    """Test view(dtype) raises RuntimeError when byte size not divisible by target dtype."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    # 3 float32 = 12 bytes; float64.itemsize = 8, 12 % 8 != 0
    shape = (3,)
    original_tensor = torch.randn(shape, device=dm.device, dtype=torch.float32)
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    with pytest.raises(RuntimeError, match="byte size.*divisible"):
        sharded_tensor.view(torch.float64)


@pytest.mark.multigpu_static
def test_view_dtype_nd_input(distributed_mesh):
    """Test view(dtype) on a multi-dimensional ShardTensor preserves shape when itemsizes match."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dm = DistributedManager()
    shape = (4, 4)
    original_tensor = torch.randint(0, 100, shape, device=dm.device, dtype=torch.int32)
    sharded_tensor = scatter_tensor(
        original_tensor,
        global_src=0,
        mesh=distributed_mesh,
        placements=(Shard(0),),
        requires_grad=False,
    )

    result = sharded_tensor.view(torch.float32)

    assert isinstance(result, ShardTensor), "view(dtype) should return a ShardTensor"
    assert result.dtype == torch.float32
    assert result.shape == shape, (
        f"int32 and float32 have same itemsize; shape should be preserved {shape}, got {result.shape}"
    )
    plain_viewed = original_tensor.view(torch.float32)
    assert result.shape == plain_viewed.shape
