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


from typing import List, Optional

import torch
import torch.distributed as dist

from .utils import (
    all_gather_v_bwd_wrapper,
    all_gather_v_wrapper,
    gather_v_wrapper,
    indexed_all_to_all_v_wrapper,
    indexed_all_to_all_v_wrapper_bwd,
    scatter_v_wrapper,
)


class AllGatherVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed AllGatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank. Its indended to be used in
    tensor-parallel settings on tensors which require gradients
    to be passed through.
    The backward pass performs an AllReduceV operation where
    each rank gathers its corresponding chunk of a global tensor
    from each other rank and sums up these individual gradients.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        use_fp32: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the Distributed AllGatherV primitive"""

        gathered_tensor = all_gather_v_wrapper(tensor, sizes, dim=dim, group=group)
        ctx.sizes = sizes
        ctx.group = group
        ctx.dim = dim
        ctx.use_fp32 = use_fp32
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pragma: no cover
        """backward pass of the of the Distributed AllGatherV primitive"""

        grad_tensor = all_gather_v_bwd_wrapper(
            grad_output,
            ctx.sizes,
            dim=ctx.dim,
            use_fp32=ctx.use_fp32,
            group=ctx.group,
        )

        if not ctx.needs_input_grad[0]:
            grad_tensor = None

        return grad_tensor, None, None, None, None


class GatherVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed GatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to a straightforward
    ScatterV primitive distributing the global gradient from the
    specified destination rank to all the other ranks.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the distributed GatherV primitive"""

        gathered_tensor = gather_v_wrapper(tensor, sizes, dim=dim, dst=dst, group=group)

        ctx.sizes = sizes
        ctx.dim = dim
        ctx.dst = dst
        ctx.group = group
        return gathered_tensor

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        """backward pass of the Distributed GatherV primitive"""

        grad_tensor = scatter_v_wrapper(
            grad_output, ctx.sizes, dim=ctx.dim, src=ctx.dst, group=ctx.group
        )

        if not ctx.needs_input_grad[0]:
            grad_tensor = None

        return grad_tensor, None, None, None, None


class ScatterVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for Distributed ScatterV. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to an GatherV primitive
    gathering local gradients from all the other ranks into a single
    global gradient on the specified source rank.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        src: int = 0,
        group=Optional[dist.ProcessGroup],
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the Distributed ScatterV primitive"""

        scattered_tensor = scatter_v_wrapper(
            tensor, sizes, dim=dim, src=src, group=group
        )

        ctx.tensor = tensor
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.src = src
        ctx.group = group
        return scattered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """backward pass of the Distributed ScatterV primitive"""

        grad_tensor = gather_v_wrapper(
            grad_output, ctx.sizes, dim=ctx.dim, dst=ctx.src, group=ctx.group
        )

        if not ctx.needs_input_grad[0]:
            grad_tensor = None

        return grad_tensor, None, None, None, None


class IndexedAllToAllVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for an Indexed AllToAllV primitive. It is based on the
    idea of a single global tensor which is distributed along a
    specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive. It is intended to be used in tensor-parallel settings
    on tensors which require gradients to be passed through.
    The backward pass more or less corresponds to the same operation as in the forward
    pass but with reversed roles and does an additional reduction of gathered gradients
    so that each rank finally will compute the overall gradient on its local tensor partition.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        indices: List[torch.Tensor],
        sizes: List[List[int]],
        use_fp32: bool = True,
        dim: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the Distributed IndexedAlltoAllV primitive"""

        tensor_to_recv = indexed_all_to_all_v_wrapper(
            tensor,
            indices,
            sizes,
            dim=dim,
            group=group,
        )

        ctx.sizes = sizes
        ctx.use_fp32 = use_fp32
        ctx.group = group
        ctx.tensor_size_along_dim = tensor.size(dim)
        ctx.indices = indices
        ctx.dim = dim

        return tensor_to_recv

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        """backward pass of the Distributed IndexedAlltoAllV primitive"""

        grad_tensor = indexed_all_to_all_v_wrapper_bwd(
            grad_output,
            ctx.indices,
            ctx.sizes,
            tensor_size_along_dim=ctx.tensor_size_along_dim,
            use_fp32=ctx.use_fp32,
            dim=ctx.dim,
            group=ctx.group,
        )

        if not ctx.needs_input_grad[0]:
            grad_tensor = None

        return grad_tensor, None, None, None, None, None, None


def all_gather_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Autograd Wrapper for a distributed AllGatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank. Its indended to be used in
    tensor-parallel settings on tensors which require gradients
    to be passed through.
    The backward pass performs an AllReduceV operation where
    each rank gathers its corresponding chunk of a global tensor
    from each other rank and sums up these individual gradients.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        boolean flag to indicate whether to use FP32 precision for the
        reduction in the backward pass, by default True
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    return AllGatherVAutograd.apply(tensor, sizes, dim, use_fp32, group)


def gather_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Autograd Wrapper for a distributed GatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to a straightforward
    ScatterV primitive distributing the global gradient from the
    specified destination rank to all the other ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """

    return GatherVAutograd.apply(tensor, sizes, dim, dst, group)


def scatter_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Autograd Wrapper for Distributed ScatterV. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to an GatherV primitive
    gathering local gradients from all the other ranks into a single
    global gradient on the specified source rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """

    return ScatterVAutograd.apply(tensor, sizes, dim, src, group)


def indexed_all_to_all_v(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    use_fp32: bool = True,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Autograd Wrapper for an Indexed AllToAllV primitive. It is based on the
    idea of a single global tensor which is distributed along a
    specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive. It is intended to be used in tensor-parallel settings
    on tensors which require gradients to be passed through.
    The backward pass more or less corresponds to the same operation as in the forward
    pass but with reversed roles and does an additional reduction of gathered gradients
    so that each rank finally will compute the overall gradient on its local tensor partition.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    use_fp32 : bool, optional
        flag to specify whether to use FP32 precision in the reduction
        in the backward pass, by default True
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    return IndexedAllToAllVAutograd.apply(
        tensor,
        indices,
        sizes,
        use_fp32,
        dim,
        group,
    )
