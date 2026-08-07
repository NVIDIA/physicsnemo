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

r"""Autograd boundary guards for shard patches.

Identity-forward ``autograd.Function``\ s that act only on the gradient in
backward. Shard patches place them on tensors entering a local computation
so that the gradients crossing back out satisfy a distributed or layout
invariant the surrounding graph relies on:

- :class:`GradReducer` / :class:`ConvGradReducer` -- all-reduce gradients
  that are rank-local partial sums (each over a different placement
  condition; see their docstrings).
- :class:`ContiguousGrad` -- normalize kernel-layout gradients (e.g. the
  BSHD layout attention kernels emit) to contiguous.

All collectives here use funcol rather than ``dist.*`` so an AOT-captured
backward graph holds a ``DeviceMesh`` instead of a ProcessGroup
ScriptObject, which cannot be deepcopied when AOTAutograd caches the
backward GraphModule.
"""

import torch
import torch.distributed._functional_collectives as funcol

from physicsnemo.domain_parallel._shard_tensor_spec import ShardTensorSpec

__all__ = ["ContiguousGrad", "ConvGradReducer", "GradReducer"]


class ContiguousGrad(torch.autograd.Function):
    r"""Identity forward; makes the incoming gradient contiguous in backward.

    Attention kernels emit gradients in their BSHD layout, while the graphs
    upstream (e.g. the K/V projection linears, recorded through the DTensor
    fallback) folded contiguous forward tensors and reject BSHD grads in
    their internal ``view`` calls. Placed on the local tensors entering an
    SDPA kernel so every gradient crossing back is contiguous.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        r"""Return ``x`` unchanged (as a fresh alias for autograd metadata)."""
        return x.view_as(x)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Nothing to save: backward only touches the incoming gradient."""

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        r"""Return the incoming gradient, made contiguous."""
        return grad_output.contiguous()


class GradReducer(torch.autograd.Function):
    r"""Custom autograd function that performs an allreduce on gradients if they are replicated."""

    @staticmethod
    def forward(
        input: torch.Tensor,
        spec: ShardTensorSpec,
    ) -> torch.Tensor:
        r"""Forward pass: return the input tensor unchanged.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to pass through.
        spec : ShardTensorSpec
            Shard specification for determining reduction behavior.

        Returns
        -------
        torch.Tensor
            The input tensor unchanged.
        """
        return input

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the input ShardTensorSpec for the backward all-reduce."""
        _input, spec = inputs
        ctx.spec = spec

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        r"""Backward pass that performs allreduce on gradients if replicated.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved variables from forward.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Tuple of (reduced gradient, ``None`` for spec).
        """
        spec = ctx.spec
        placement = spec.placements[0]
        if placement.is_replicate():
            grad_output = funcol.all_reduce(grad_output, "sum", (spec.mesh, 0))
            # Prevent an asynchronous wrapper from escaping into gradient hooks
            # or leaf ``.grad`` storage without first completing the reduction.
            if isinstance(grad_output, funcol.AsyncCollectiveTensor):
                grad_output = grad_output.wait()
        return grad_output, None


class ConvGradReducer(torch.autograd.Function):
    r"""Custom autograd function that performs an allreduce on gradients in backward pass.

    This makes defining a forward-only shard patch easier. If you need to allreduce
    weight grads in the backward pass, call this on the weight in the forward pass.
    """

    @staticmethod
    def forward(
        weight_or_bias: torch.Tensor,
        spec: ShardTensorSpec,
    ) -> torch.Tensor:
        r"""Forward pass: return the weight/bias tensor unchanged.

        Parameters
        ----------
        weight_or_bias : torch.Tensor
            The weight or bias tensor to pass through.
        spec : ShardTensorSpec
            Shard spec of the convolutional input (not the weight_or_bias).

        Returns
        -------
        torch.Tensor
            The input tensor unchanged.
        """
        return weight_or_bias

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the input ShardTensorSpec for the backward all-reduce."""
        _weight_or_bias, spec = inputs
        ctx.spec = spec

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_weight_or_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        r"""Backward pass: all-reduce gradients over each sharded mesh dim.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved variables from forward.
        grad_weight_or_bias : torch.Tensor
            Gradient of the loss with respect to weight or bias.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Tuple of (reduced gradient, ``None`` for spec).
        """
        for mesh_dim in range(ctx.spec.mesh.ndim):
            if ctx.spec.placements[mesh_dim].is_shard():
                # funcol.all_reduce returns a new tensor (AsyncCollectiveTensor)
                # that auto-waits when used; assigning back into the loop var
                # serializes the iterations correctly.
                grad_weight_or_bias = funcol.all_reduce(
                    grad_weight_or_bias, "sum", (ctx.spec.mesh, mesh_dim)
                )

        # Do not let the final asynchronous result escape into parameter hooks
        # or ``param.grad``, where storage may be accessed without dispatch.
        if isinstance(grad_weight_or_bias, funcol.AsyncCollectiveTensor):
            grad_weight_or_bias = grad_weight_or_bias.wait()

        return grad_weight_or_bias, None
