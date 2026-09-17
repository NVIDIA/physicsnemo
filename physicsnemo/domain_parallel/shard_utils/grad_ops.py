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

- :class:`GradReducer` -- all-reduce gradients that are rank-local partial
  sums over another tensor's sharded mesh dims.
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

__all__ = ["ContiguousGrad", "GradReducer"]


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
    r"""Identity forward; all-reduces the gradient in backward.

    Apply to a tensor entering a local computation whose gradient is a
    rank-local partial sum: pass the spec of the *counterpart* tensor whose
    sharding splits the contraction (e.g. the sharded input for a
    conv/linear weight, sharded q for replicated k/v in attention). The
    gradient is summed over each mesh dim where that spec is sharded.
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
        ref_spec: ShardTensorSpec,
    ) -> torch.Tensor:
        r"""Forward pass: return the input tensor unchanged.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to pass through.
        ref_spec : ShardTensorSpec
            Spec of the counterpart tensor whose sharded mesh dims make
            ``input``'s gradient a partial sum (not ``input``'s own spec).

        Returns
        -------
        torch.Tensor
            The input tensor unchanged.
        """
        return input

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the reference ShardTensorSpec for the backward all-reduce."""
        _input, ref_spec = inputs
        ctx.ref_spec = ref_spec

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        r"""Backward pass: all-reduce over each sharded mesh dim of the ref spec.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved variables from forward.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Tuple of (reduced gradient, ``None`` for ref_spec).
        """
        for mesh_dim in range(ctx.ref_spec.mesh.ndim):
            if ctx.ref_spec.placements[mesh_dim].is_shard():
                # funcol.all_reduce returns a new tensor (AsyncCollectiveTensor)
                # that auto-waits when used; assigning back into the loop var
                # serializes the iterations correctly.
                grad_output = funcol.all_reduce(
                    grad_output, "sum", (ctx.ref_spec.mesh, mesh_dim)
                )

        # Do not let the final asynchronous result escape into parameter hooks
        # or ``param.grad``, where storage may be accessed without dispatch.
        if isinstance(grad_output, funcol.AsyncCollectiveTensor):
            grad_output = grad_output.wait()

        return grad_output, None
