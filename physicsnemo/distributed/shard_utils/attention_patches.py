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

from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


from torch.distributed import DeviceMesh  # noqa: E402

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
)
from physicsnemo.distributed.shard_utils.ring import (  # noqa: E402
    RingPassingConfig,
    perform_ring_iteration,
)

aten = torch.ops.aten


def add_log_sumexp(
    log_a: Optional[torch.Tensor], log_b: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Add two log_sumexp values together.

    Args:
        log_a: First log-space value, can be None
        log_b: Second log-space value, can be None

    Returns:
        torch.Tensor: Result of log(exp(log_a) + exp(log_b)) computed in a numerically stable way

    Think of this function as taking two values, A and B,
    passed in via log form: log(A) and log(B).  This function
    will return log(A+B) in a numerically stable way.

    """
    if log_a is None or log_b is None:
        return log_a if log_a is not None else log_b

    diff = torch.abs(log_a - log_b)
    return torch.max(log_a, log_b) + torch.log(torch.exp(-diff) + 1.0)


def stable_signed_accumulate(
    log_abs_global_O: Optional[torch.Tensor],
    sign_global_O: Optional[torch.Tensor],
    log_O: torch.Tensor,
    sign_O: torch.Tensor,
    log_A: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate two functions together, keeping track of the sign and log_abs.

    Args:
        log_abs_global_O: Log of absolute value of accumulated output so far, can be None
        sign_global_O: Sign of accumulated output so far, can be None
        log_O: Log of absolute value of current output
        sign_O: Sign of current output
        log_A: Log of normalization factor for current output

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated (log_abs, sign) pair for accumulated output

    The block attention algorithm needs to continuously accumulate the output of each block,
    however, the normalization is done in log space.  This function accomodates that by
    accumulating the output in log space using log space normalizations.  Note that because
    the output of an attention block can be negative, we must use both log(|O|) and sign(O)
    for each term.
    """
    if log_abs_global_O is None and sign_global_O is None:
        return log_O + log_A, sign_O

    log_abs_T = log_O + log_A
    sign_T = sign_O

    # Find larger magnitude term
    max_log = torch.maximum(log_abs_global_O, log_abs_T)
    min_log = torch.minimum(log_abs_global_O, log_abs_T)

    # If signs are the same, use log-sum-exp
    same_sign = sign_global_O == sign_T
    log_abs_new = torch.where(
        same_sign,
        max_log + torch.log1p(torch.exp(min_log - max_log)),  # log-sum-exp
        max_log + torch.log1p(-torch.exp(min_log - max_log)),  # log-subtraction
    )

    # Determine new sign
    sign_new = torch.where(
        same_sign,
        sign_global_O,
        torch.where(log_abs_global_O >= log_abs_T, sign_global_O, sign_T),
    )

    return log_abs_new, sign_new


class RingSDPA(torch.autograd.Function):
    """
    Performs scaled dot product attention on sharded Q, K, V.

    The ring allreduce happens concurrently and overlapping with the computation,
    for performance improvements.

    For details about the ring attention, see: https://arxiv.org/abs/2310.01889
    Note that the original implementation is a combination of JAX + flash attention + ring attention.
    Here, instead, we leverage the underlying and built-in pytorch efficienct attention.

    A key difference with this algorithm is how we track the per-block normalizations.  The pytorch
    function returns log_sumexp, which we use for a running normalization.  But it has to be kept in log
    space to prevent underflow/overflow as well as precision issues.  See the helper functions
    `add_log_sumexp` and `stable_signed_accumulate` for more details.
    """

    # comm_stream = torch.cuda.Stream()

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        mesh: DeviceMesh,
        ring_config: RingPassingConfig,
        attn_args: dict,
    ) -> torch.Tensor:
        """
        Forward pass for the ring attention implementation.  This implementation
        will overlap the communication with the computation.  Note that there is an
        explicit sync in each iteration to prevent the communication stream from getting
        ahead of the computation stream, by waiting on the all_to_all operation to complete.
        """

        ctx.attn_args = attn_args
        ctx.mesh = mesh
        ctx.ring_config = ring_config

        # Create buffers to store outputs
        log_global_output = None
        sign_global_output = None
        global_log_sumexp = None

        # For the first iteration, use local tensors
        current_k, current_v = k, v

        # Pre-allocate a single combined buffer for k,v
        # Find the dimension to concatenate on - should be a dim that preserves tensor structure
        cat_dim = 0  # Usually batch or sequence dimension

        # Set up communication
        local_group = mesh.get_group(ring_config.mesh_dim)
        local_rank = mesh.get_local_rank(ring_config.mesh_dim)
        local_size = dist.get_world_size(group=local_group)

        id_of_right = (local_rank + 1) % ring_config.mesh_size
        id_of_left = (local_rank - 1) % ring_config.mesh_size

        # Create streams that persist for the duration of the ring computation.
        compute_stream = torch.cuda.default_stream()
        comm_stream = torch.cuda.Stream()

        for i in range(ring_config.mesh_size):
            # Launch communication for the next iteration early
            with record_function(f"sdpa_send_data_{i}_{dist.get_rank()}"):
                if i < ring_config.mesh_size - 1:

                    # Use a dedicated stream for communication
                    with torch.cuda.stream(comm_stream):

                        send_tensors = [
                            torch.empty((), device=q.device, dtype=q.dtype)
                            for _ in range(local_size)
                        ]
                        recv_tensors = [
                            torch.empty((), device=q.device, dtype=q.dtype)
                            for _ in range(local_size)
                        ]

                        # Combine k and v for communication
                        send_tensors[id_of_right] = torch.cat(
                            [current_k, current_v], dim=cat_dim
                        ).contiguous()
                        recv_tensors[id_of_left] = torch.empty_like(
                            send_tensors[id_of_right]
                        )

                        # Use async_op=True to enable overlapping
                        a2a_op = dist.all_to_all(
                            recv_tensors, send_tensors, group=local_group, async_op=True
                        )

                    # Mark these as used by the communication stream:
                    current_k.record_stream(comm_stream)
                    current_v.record_stream(comm_stream)

            # Perform computation on current k,v while communication happens
            with record_function(f"sdpa_forward_{i}_{dist.get_rank()}"):
                with torch.cuda.stream(compute_stream):
                    (
                        output,
                        log_sumexp,
                        philox_seed,
                        philox_offset,
                    ) = aten._scaled_dot_product_efficient_attention(
                        q,
                        current_k,
                        current_v,
                        attn_mask,
                        compute_log_sumexp=True,
                        **attn_args,
                    )

                    # Add an extra dimension to the log_sumexp:
                    log_sumexp = log_sumexp.unsqueeze(-1)
                    log_output = torch.log(torch.abs(output))
                    sign_output = torch.sign(output)

                    log_global_output, sign_global_output = stable_signed_accumulate(
                        log_global_output,
                        sign_global_output,
                        log_output,
                        sign_output,
                        log_sumexp,
                    )

                    global_log_sumexp = add_log_sumexp(global_log_sumexp, log_sumexp)

            if i < ring_config.mesh_size - 1:
                # Wait for communication operations to complete before allowing more work
                a2a_op.wait()

                # compute_stream.wait_stream(comm_stream)

                # Explicit synchronization to ensure communication is complete
                # Also makes sure that the attention computation is complete before changing current_k, current_v
                # compute_stream.synchronize()

                current_k, current_v = torch.chunk(
                    recv_tensors[id_of_left], 2, dim=cat_dim
                )

        # Compute the final output
        stable_output = sign_global_output * torch.exp(
            log_global_output - global_log_sumexp
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            attn_mask,
            stable_output,
            global_log_sumexp,
            philox_seed,
            philox_offset,
        )
        ctx.grad_input_mask = (True, True, True, attn_mask is not None)

        return stable_output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        None,
    ]:
        """
        Backward pass for the ring SDPA.

        Currently, this is not overlapping communication with the computation.
        Note that the backward pass has 2x communication: send k, v but also grad_k, grad_v.

        """
        (
            q,
            k,
            v,
            attn_mask,
            output,
            log_sumexp,
            philox_seed,
            philox_offset,
        ) = ctx.saved_tensors
        attn_args = ctx.attn_args

        grad_q = torch.zeros_like(
            q, device=q.device, memory_format=torch.contiguous_format
        )
        grad_k = torch.zeros_like(
            k, device=k.device, memory_format=torch.contiguous_format
        )
        grad_v = torch.zeros_like(
            v, device=v.device, memory_format=torch.contiguous_format
        )
        grad_attn_mask = None

        # TODO: overlap communication with computation.
        # This needs to be done in two stages.  First, we can send k, v along the ring before computing
        # the gradients.  We also need to send grad_k, grad_v along the ring and accumulate them.

        # Since the next iteration's grad_k, grad_v do not depend on the current iteration's gradient
        # outputs, we can still overlap.  But we need two sync spots instead of one.
        # Algorithm therefore looks like this:
        # 1. If iteration != N-1, send k, v to the next GPU asycn after combining them into one tensor.
        # 2. If iteration != 0, wait for grad_k, grad_v to be received from the previous GPU and split them.
        # 2. Compute the gradients on the local block (grad_q, grad_k, grad_v)
        # 3. Accumulate the gradients on the local block.
        # 5. If iteration != N-1, wait for k, v to be received from the previous GPU (and split them) before the next iteration
        # 4. If iteration != 0, send grad_k, grad_v to the next GPU after combining them into one tensor.

        for i in range(ctx.ring_config.mesh_size):

            (
                block_grad_q,
                block_grad_k,
                block_grad_v,
                block_grad_attn_mask,
            ) = aten._scaled_dot_product_efficient_attention_backward(
                grad_output,
                q,
                k,
                v,
                attn_mask,
                output,
                log_sumexp,
                philox_seed,
                philox_offset,
                grad_input_mask=ctx.grad_input_mask,
                **attn_args,
            )

            grad_q += block_grad_q
            grad_k += block_grad_k
            grad_v += block_grad_v

            # Send k, v, grad_k, grad_v to the next rank:
            k = perform_ring_iteration(k, ctx.mesh, ctx.ring_config)
            v = perform_ring_iteration(v, ctx.mesh, ctx.ring_config)
            grad_k = perform_ring_iteration(grad_k, ctx.mesh, ctx.ring_config)
            grad_v = perform_ring_iteration(grad_v, ctx.mesh, ctx.ring_config)

        return grad_q, grad_k, grad_v, grad_attn_mask, None, None, None


class RingSDPABlocking(torch.autograd.Function):
    """
    Performs scaled dot product attention on sharded Q, K, V.

    The ring allreduce happens in a blocking manner.  This isn't more efficient, but
    it is useful for understanding the algorithm and debugging.

    For details about the ring attention, see: https://arxiv.org/abs/2310.01889
    Note that the original implementation is a combination of JAX + flash attention + ring attention.
    Here, instead, we leverage the underlying and built-in pytorch efficienct attention.

    A key difference with this algorithm is how we track the per-block normalizations.  The pytorch
    function returns log_sumexp, which we use for a running normalization.  But it has to be kept in log
    space to prevent underflow/overflow as well as precision issues.  See the helper functions
    `add_log_sumexp` and `stable_signed_accumulate` for more details.
    """

    # comm_stream = torch.cuda.Stream()

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        mesh: DeviceMesh,
        ring_config: RingPassingConfig,
        attn_args: dict,
    ) -> torch.Tensor:
        """
        Forward pass for the ring attention implementation.  This implementation
        will overlap the communication with the computation.  Note that there is an
        explicit sync in each iteration to prevent the communication stream from getting
        ahead of the computation stream, by waiting on the all_to_all operation to complete.
        """

        ctx.attn_args = attn_args
        ctx.mesh = mesh
        ctx.ring_config = ring_config

        # Create buffers to store outputs
        log_global_output = None
        sign_global_output = None
        global_log_sumexp = None

        # For the first iteration, use local tensors
        current_k, current_v = k, v

        for i in range(ring_config.mesh_size):

            # Perform computation on current k,v while communication happens
            (
                output,
                log_sumexp,
                philox_seed,
                philox_offset,
            ) = aten._scaled_dot_product_efficient_attention(
                q,
                current_k,
                current_v,
                attn_mask,
                compute_log_sumexp=True,
                **attn_args,
            )

            # Add an extra dimension to the log_sumexp:
            log_sumexp = log_sumexp.unsqueeze(-1)
            log_output = torch.log(torch.abs(output))
            sign_output = torch.sign(output)

            log_global_output, sign_global_output = stable_signed_accumulate(
                log_global_output,
                sign_global_output,
                log_output,
                sign_output,
                log_sumexp,
            )

            global_log_sumexp = add_log_sumexp(global_log_sumexp, log_sumexp)

            # send k and v to the next rank:
            current_k = perform_ring_iteration(current_k, ctx.mesh, ctx.ring_config)
            current_v = perform_ring_iteration(current_v, ctx.mesh, ctx.ring_config)

        # Compute the final output
        stable_output = sign_global_output * torch.exp(
            log_global_output - global_log_sumexp
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            attn_mask,
            stable_output,
            global_log_sumexp,
            philox_seed,
            philox_offset,
        )
        ctx.grad_input_mask = (True, True, True, attn_mask is not None)

        return stable_output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        None,
    ]:
        """
        Backward pass for the ring SDPA.

        Currently, this is not overlapping communication with the computation.
        Note that the backward pass has 2x communication: send k, v but also grad_k, grad_v.

        """
        (
            q,
            k,
            v,
            attn_mask,
            output,
            log_sumexp,
            philox_seed,
            philox_offset,
        ) = ctx.saved_tensors
        attn_args = ctx.attn_args

        grad_q = torch.zeros_like(
            q, device=q.device, memory_format=torch.contiguous_format
        )
        grad_k = torch.zeros_like(
            k, device=k.device, memory_format=torch.contiguous_format
        )
        grad_v = torch.zeros_like(
            v, device=v.device, memory_format=torch.contiguous_format
        )
        grad_attn_mask = None

        # TODO: overlap communication with computation.
        # This needs to be done in two stages.  First, we can send k, v along the ring before computing
        # the gradients.  We also need to send grad_k, grad_v along the ring and accumulate them.

        # Since the next iteration's grad_k, grad_v do not depend on the current iteration's gradient
        # outputs, we can still overlap.  But we need two sync spots instead of one.
        # Algorithm therefore looks like this:
        # 1. If iteration != N-1, send k, v to the next GPU asycn after combining them into one tensor.
        # 2. If iteration != 0, wait for grad_k, grad_v to be received from the previous GPU and split them.
        # 2. Compute the gradients on the local block (grad_q, grad_k, grad_v)
        # 3. Accumulate the gradients on the local block.
        # 5. If iteration != N-1, wait for k, v to be received from the previous GPU (and split them) before the next iteration
        # 4. If iteration != 0, send grad_k, grad_v to the next GPU after combining them into one tensor.

        for i in range(ctx.ring_config.mesh_size):

            (
                block_grad_q,
                block_grad_k,
                block_grad_v,
                block_grad_attn_mask,
            ) = aten._scaled_dot_product_efficient_attention_backward(
                grad_output,
                q,
                k,
                v,
                attn_mask,
                output,
                log_sumexp,
                philox_seed,
                philox_offset,
                grad_input_mask=ctx.grad_input_mask,
                **attn_args,
            )

            grad_q += block_grad_q
            grad_k += block_grad_k
            grad_v += block_grad_v

            # Send k, v, grad_k, grad_v to the next rank:
            k = perform_ring_iteration(k, ctx.mesh, ctx.ring_config)
            v = perform_ring_iteration(v, ctx.mesh, ctx.ring_config)
            grad_k = perform_ring_iteration(grad_k, ctx.mesh, ctx.ring_config)
            grad_v = perform_ring_iteration(grad_v, ctx.mesh, ctx.ring_config)

        return grad_q, grad_k, grad_v, grad_attn_mask, None, None, None


def ring_sdpa(
    q: ShardTensor,
    k: ShardTensor,
    v: ShardTensor,
    attn_mask: Optional[ShardTensor] = None,
    **kwargs: dict,
) -> ShardTensor:
    """
    High Level, differentiable function to compute neighborhood attention on a sharded tensor.

    Operation works like so:
    - Figure out the size of halos needed.
    - Apply the halo padding (differentiable)
    - Perform the neighborhood attention on the padded tensor. (differentiable)
    - "UnHalo" the output tensor (different from, say, convolutions)
    - Return the updated tensor as a ShardTensor.

    """

    mesh = q._spec.mesh

    # We can be confident of this because 1D meshes are enforced
    mesh_dim = 0

    local_group = mesh.get_group(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Create a config object to simplify function args for message passing:
    ring_config = RingPassingConfig(
        mesh_dim=mesh_dim,
        mesh_size=local_size,
        communication_method="p2p",
    )

    # First, get the tensors locally and perform halos:
    lq, lk, lv = q.to_local(), k.to_local(), v.to_local()

    if attn_mask is not None:
        latn_mask = attn_mask.to_local()
    else:
        latn_mask = None

    x = RingSDPABlocking.apply(lq, lk, lv, latn_mask, q._spec.mesh, ring_config, kwargs)

    # Convert back to ShardTensor
    x = ShardTensor.from_local(
        x, q._spec.mesh, q._spec.placements, q._spec.sharding_shapes()
    )
    return x


def sdpa_wrapper(func: Callable, types: Any, args: tuple, kwargs: dict) -> ShardTensor:
    """Wrapper for natten.functional.na2d to support sharded tensors.

    Handles both regular torch.Tensor inputs and distributed ShardTensor inputs.
    For regular tensors, passes through to the wrapped na2d function.
    For ShardTensor inputs, handles adding halos and applying distributed na2d.

    Args:
        wrapped: Original na2d function being wrapped
        instance: Instance the wrapped function is bound to
        args: Positional arguments containing query, key, value tensors
        kwargs: Keyword arguments including kernel_size and dilation

    Returns:
        Result tensor as either torch.Tensor or ShardTensor depending on input types

    Raises:
        UndeterminedShardingError: If input tensor types are mismatched
    """

    q, k, v, attn_mask, kwargs = repackage_sdpa_args(*args, **kwargs)

    # Make sure all tensors are on the same mesh
    if not (q._spec.mesh == k._spec.mesh == v._spec.mesh):
        raise MissingShardPatch("q, k, and v must all be on the same mesh")

    # Make sure the mesh is 1D
    if q._spec.mesh.ndim != 1:
        raise MissingShardPatch("q must be on a 1D mesh")

    return ring_sdpa(q, k, v, attn_mask, **kwargs)


def repackage_sdpa_args(
    query: Union[torch.Tensor, ShardTensor],
    key: Union[torch.Tensor, ShardTensor],
    value: Union[torch.Tensor, ShardTensor],
    attn_mask: Optional[Union[torch.Tensor, ShardTensor]] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
    *args,
    **kwargs,
) -> Tuple[
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, ShardTensor],
    dict,
]:
    """
    Repackages scaled dot product attention arguments into standard format.

    """

    if enable_gqa:
        raise NotImplementedError("GQA is not implemented for sharded tensors")

    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        # "enable_gqa": enable_gqa,
    }

    return query, key, value, attn_mask, return_kwargs


ShardTensor.register_function_handler(
    torch.nn.functional.scaled_dot_product_attention, sdpa_wrapper
)
