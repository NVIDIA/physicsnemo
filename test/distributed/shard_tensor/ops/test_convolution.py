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
The tests here validate the correctness of the convolution operator over
sharded inputs.  We check conv1d, conv2d, conv3d, and the transposed
operations as well.  For 2d and 3d, we test on both 1D and 2D sharding.

Sharding must always be over spatial dimensions (H, W, D) so the sharded
axes are always Shard(2), Shard(3), or Shard(4).

The channels dimension is largely irrelevant, we have a couple parameters
for it but it just has to be non-zero.

Some sharded convolutions aren't well supported: these ones are xfail'd.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager, scatter_tensor

from .utils import generate_image_like_data, numerical_shard_tensor_check


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv1d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    if padding != 0 and stride > 1:
        pytest.xfail(
            "Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding"
        )
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv1d with stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H,), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.Conv1d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv_transpose_1d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    # For transpose convolutions, odd kernels aren't supported at all:
    if kernel % 2 != 0:
        pytest.xfail("Odd Kernels not yet supported for transposed convolutions")

    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    # if padding != 0 and stride > 1:
    #     pytest.xfail("Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding")
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv1d with stride > 1 and kernel size != stride is expected to fail"
        )

    # Mark test as expected to fail if stride > 1 and kernel size != stride
    if stride > 1 and kernel != stride:
        pytest.xfail(
            "Conv1d Transposed with stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H,), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ConvTranspose1d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv2d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    if padding != 0 and stride > 1:
        pytest.xfail(
            "Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding"
        )
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv2d with stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H)).to(dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.Conv2d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv_transpose_2d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    # For transpose convolutions, odd kernels aren't supported at all:
    if kernel % 2 != 0:
        pytest.xfail("Odd Kernels not yet supported for transposed convolutions")

    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    # if padding != 0 and stride > 1:
    #     pytest.xfail("Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding")
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv2d with stride > 1 and kernel size != stride is expected to fail"
        )

    # Mark test as expected to fail if stride > 1 and kernel size != stride
    if stride > 1 and kernel != stride:
        pytest.xfail(
            "Conv2d Transposedwith stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(
        2,
        C_in,
        (
            H,
            H,
        ),
        device=dm.device,
    )

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ConvTranspose2d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv2d_2dmesh(
    distributed_mesh_2d, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    if padding != 0 and stride > 1:
        pytest.xfail(
            "Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding"
        )
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv2d with stride > 1 and kernel size != stride is expected to fail"
        )

    if backward:
        # See: https://github.com/pytorch/pytorch/issues/153615
        pytest.xfail("DTensor with 2D mesh and backwards fails currently upstream")

    dm = DistributedManager()

    if dm.world_size < 4:
        pytest.skip("Conv3d with 2D mesh requires at least 4 GPUs")

    image = generate_image_like_data(2, C_in, (H, H), device=dm.device)

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.Conv2d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 256])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv_transpose_2d_2dmesh(
    distributed_mesh_2d, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    # For transpose convolutions, odd kernels aren't supported at all:
    if kernel % 2 != 0:
        pytest.xfail("Odd Kernels not yet supported for transposed convolutions")

    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    # if padding != 0 and stride > 1:
    #     pytest.xfail("Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding")
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv2d with stride > 1 and kernel size != stride is expected to fail"
        )

    # Mark test as expected to fail if stride > 1 and kernel size != stride
    if stride > 1 and kernel != stride:
        pytest.xfail(
            "Conv2d Transposedwith stride > 1 and kernel size != stride is expected to fail"
        )

    if backward:
        # See: https://github.com/pytorch/pytorch/issues/153615
        pytest.xfail("DTensor with 2D mesh and backwards fails currently upstream")

    dm = DistributedManager()

    if dm.world_size < 4:
        pytest.skip("Conv3d with 2D mesh requires at least 4 GPUs")

    image = generate_image_like_data(
        2,
        C_in,
        (
            H,
            H,
        ),
        device=dm.device,
    )

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.ConvTranspose2d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv3d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    if padding != 0 and stride > 1:
        pytest.xfail(
            "Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding"
        )
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv3d with stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.Conv3d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 64])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv_transpose_3d_1dmesh(
    distributed_mesh, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    # For transpose convolutions, odd kernels aren't supported at all:
    if kernel % 2 != 0:
        pytest.xfail("Odd Kernels not yet supported for transposed convolutions")

    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    # if padding != 0 and stride > 1:
    #     pytest.xfail("Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding")
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv3d with stride > 1 and kernel size != stride is expected to fail"
        )

    # Mark test as expected to fail if stride > 1 and kernel size != stride
    if stride > 1 and kernel != stride:
        pytest.xfail(
            "Conv3d Transposedwith stride > 1 and kernel size != stride is expected to fail"
        )

    dm = DistributedManager()

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (Shard(2),)

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh, placements, requires_grad=backward
    )

    module = torch.nn.ConvTranspose3d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv3d_2dmesh(
    distributed_mesh_2d, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    if padding != 0 and stride > 1:
        pytest.xfail(
            "Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding"
        )
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv3d with stride > 1 and kernel size != stride is expected to fail"
        )

    if backward:
        # See: https://github.com/pytorch/pytorch/issues/153615
        pytest.xfail("DTensor with 2D mesh and backwards fails currently upstream")

    dm = DistributedManager()

    if dm.world_size < 4:
        pytest.skip("Conv3d with 2D mesh requires at least 4 GPUs")

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.Conv3d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("H", [32, 64])
@pytest.mark.parametrize(
    "C_in",
    [
        16,
    ],
)
@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("backward", [False, True])
def test_conv_transpose_3d_2dmesh(
    distributed_mesh_2d, H, C_in, kernel, stride, padding, dilation, groups, backward
):
    # For transpose convolutions, odd kernels aren't supported at all:
    if kernel % 2 != 0:
        pytest.xfail("Odd Kernels not yet supported for transposed convolutions")

    if kernel % 2 == 0 and stride != kernel:
        pytest.xfail(
            "Even Kernels only supported for stride = kernel size and padding = 0"
        )
    # if padding != 0 and stride > 1:
    #     pytest.xfail("Padding != 0 is not supported for sharded convolutions with stride > 1 and non-zero padding")
    if stride > 1 and stride != kernel:
        pytest.xfail(
            "Conv3d with stride > 1 and kernel size != stride is expected to fail"
        )

    # Mark test as expected to fail if stride > 1 and kernel size != stride
    if stride > 1 and kernel != stride:
        pytest.xfail(
            "Conv3d Transposedwith stride > 1 and kernel size != stride is expected to fail"
        )

    if backward:
        # See: https://github.com/pytorch/pytorch/issues/153615
        pytest.xfail("DTensor with 2D mesh and backwards fails currently upstream")

    dm = DistributedManager()

    if dm.world_size < 4:
        pytest.skip("Conv3d with 2D mesh requires at least 4 GPUs")

    image = generate_image_like_data(2, C_in, (H, H, H), device=dm.device)

    placements = (Shard(2), Shard(3))

    sharded_image = scatter_tensor(
        image, 0, distributed_mesh_2d, placements, requires_grad=backward
    )

    module = torch.nn.ConvTranspose3d(
        in_channels=C_in,
        out_channels=8,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    numerical_shard_tensor_check(
        distributed_mesh_2d, module, [sharded_image], {}, check_grads=backward
    )
