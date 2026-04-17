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

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from torch.distributed import DeviceMesh

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
)
from physicsnemo.domain_parallel.shard_utils.ring import (
    RingPassingConfig,
    get_comm_stream,
    perform_ring_iteration,
    perform_ring_iteration_async,
)

aten = torch.ops.aten


def add_log_sumexp(
    log_a: torch.Tensor | None, log_b: torch.Tensor | None
) -> torch.Tensor:
    r"""Add two log_sumexp values together.

    Think of this function as taking two values, A and B,
    passed in via log form: :math:`\log(A)` and :math:`\log(B)`. This function
    will return :math:`\log(A+B)` in a numerically stable way.

    Parameters
    ----------
    log_a : Optional[torch.Tensor]
        First log-space value, can be ``None``.
    log_b : Optional[torch.Tensor]
        Second log-space value, can be ``None``.

    Returns
    -------
    torch.Tensor
        Result of :math:`\log(\exp(\text{log\_a}) + \exp(\text{log\_b}))` computed
        in a numerically stable way.
    """
    if log_a is None or log_b is None:
        return log_a if log_a is not None else log_b

    diff = torch.abs(log_a - log_b)
    return torch.max(log_a, log_b) + torch.log(torch.exp(-diff) + 1.0)


def stable_signed_accumulate(
    log_abs_global_O: torch.Tensor | None,
    sign_global_O: torch.Tensor | None,
    log_O: torch.Tensor,
    sign_O: torch.Tensor,
    log_A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Accumulate two functions together, keeping track of the sign and log_abs.

    The block attention algorithm needs to continuously accumulate the output of each block,
    however, the normalization is done in log space. This function accommodates that by
    accumulating the output in log space using log space normalizations. Note that because
    the output of an attention block can be negative, we must use both :math:`\log(|O|)` and
    :math:`\text{sign}(O)` for each term.

    Parameters
    ----------
    log_abs_global_O : Optional[torch.Tensor]
        Log of absolute value of accumulated output so far, can be ``None``.
    sign_global_O : Optional[torch.Tensor]
        Sign of accumulated output so far, can be ``None``.
    log_O : torch.Tensor
        Log of absolute value of current output.
    sign_O : torch.Tensor
        Sign of current output.
    log_A : torch.Tensor
        Log of normalization factor for current output.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Updated (log_abs, sign) pair for accumulated output.
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
    r"""Performs scaled dot product attention on sharded Q, K, V.

    The ring allreduce happens concurrently and overlapping with the computation,
    for performance improvements.

    For details about the ring attention, see:
    `Ring Attention <https://arxiv.org/abs/2310.01889>`_.
    Note that the original implementation is a combination of JAX + flash attention + ring attention.
    Here, instead, we leverage the underlying and built-in PyTorch efficient attention.

    A key difference with this algorithm is how we track the per-block normalizations. The PyTorch
    function returns log_sumexp, which we use for a running normalization. But it has to be kept in log
    space to prevent underflow/overflow as well as precision issues. See the helper functions
    ``add_log_sumexp`` and ``stable_signed_accumulate`` for more details.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        mesh: DeviceMesh,
        ring_config: RingPassingConfig,
        attn_args: dict,
    ) -> torch.Tensor:
        r"""Forward pass for the ring attention implementation.

        Overlaps communication with computation using a dedicated comm stream
        and double-buffered K/V tensors. The p2p ring shift for the next
        iteration's K/V runs concurrently with the current iteration's
        attention kernel.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        q : torch.Tensor
            Query tensor of shape :math:`(B, H, S, D)`.
        k : torch.Tensor
            Key tensor of shape :math:`(B, H, S, D)`.
        v : torch.Tensor
            Value tensor of shape :math:`(B, H, S, D)`.
        attn_mask : Optional[torch.Tensor]
            Optional attention mask tensor.
        mesh : DeviceMesh
            Device mesh for distributed computation.
        ring_config : RingPassingConfig
            Configuration for ring passing communication.
        attn_args : dict
            Additional arguments to pass to the attention function.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, H, S, D)`.
        """

        ctx.attn_args = attn_args
        ctx.mesh = mesh
        ctx.ring_config = ring_config

        # Accumulation state (log-space for numerical stability)
        log_global_output = None
        sign_global_output = None
        global_log_sumexp = None

        compute_stream = torch.cuda.current_stream()
        comm_stream = get_comm_stream(q.device)

        # Pre-allocate double buffers for K and V on the default stream to
        # avoid caching-allocator cross-stream synchronization inside the loop.
        k_buffers = [torch.empty_like(k), torch.empty_like(k)]
        v_buffers = [torch.empty_like(v), torch.empty_like(v)]

        # Iteration 0 reads from the original k, v (no copy needed).
        current_k, current_v = k, v

        # CUDA event used to signal that comm_stream has finished receiving
        # the next iteration's K/V into the recv buffer.
        comm_done = torch.cuda.Event()

        for i in range(ring_config.mesh_size):
            # --- Async communication: send current K/V, recv next K/V ---
            with record_function(f"sdpa_send_data_{i}_{dist.get_rank()}"):
                if i < ring_config.mesh_size - 1:
                    recv_idx = (i + 1) % 2
                    next_k_buf = k_buffers[recv_idx]
                    next_v_buf = v_buffers[recv_idx]

                    # comm_stream must wait for compute_stream to finish
                    # producing current_k / current_v before reading them.
                    comm_stream.wait_stream(compute_stream)

                    with torch.cuda.stream(comm_stream):
                        _, k_work = perform_ring_iteration_async(
                            current_k,
                            mesh,
                            ring_config,
                            recv_tensor=next_k_buf,
                        )
                        _, v_work = perform_ring_iteration_async(
                            current_v,
                            mesh,
                            ring_config,
                            recv_tensor=next_v_buf,
                        )

                    # Prevent the allocator from recycling current_k/v while
                    # comm_stream is still reading them for the send.
                    current_k.record_stream(comm_stream)
                    current_v.record_stream(comm_stream)

            # --- Compute: attention on current K/V ---
            with record_function(f"sdpa_forward_{i}_{dist.get_rank()}"):
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

            # --- Synchronize: wait for next K/V to arrive ---
            if i < ring_config.mesh_size - 1:
                for w in k_work + v_work:
                    w.wait()

                # Record completion on comm_stream and make compute_stream
                # wait so the next iteration reads valid recv buffer data.
                comm_stream.record_event(comm_done)
                compute_stream.wait_event(comm_done)

                current_k = next_k_buf
                current_v = next_v_buf

        # Final normalization
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
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        r"""Backward pass for the ring SDPA with overlapped communication.

        Overlaps k/v communication with the backward attention kernel.
        Each iteration:
        1. Wait for k, v from the previous iteration's async send.
        2. Wait for grad_k, grad_v from the previous iteration's async send.
        3. Async-send k, v for the next iteration (overlaps with compute).
        4. Compute block gradients and accumulate.
        5. Async-send accumulated grad_k, grad_v (overlaps with next
           iteration's waits and k/v send).
        The final grad_k/grad_v shift uses blocking communication since
        there is no further compute to overlap with.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors from forward.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]
            Gradients for (q, k, v, attn_mask, mesh, ring_config, attn_args).
            ``None`` values indicate non-differentiable parameters.
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
        mesh = ctx.mesh
        ring_config = ctx.ring_config
        mesh_size = ring_config.mesh_size

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

        compute_stream = torch.cuda.current_stream()
        comm_stream = get_comm_stream(q.device)

        # Pre-allocate double buffers on the default stream.
        k_bufs = [torch.empty_like(k), torch.empty_like(k)]
        v_bufs = [torch.empty_like(v), torch.empty_like(v)]
        grad_k_bufs = [torch.empty_like(k), torch.empty_like(k)]
        grad_v_bufs = [torch.empty_like(v), torch.empty_like(v)]

        kv_done = torch.cuda.Event()
        grad_done = torch.cuda.Event()

        kv_work = None
        grad_work = None
        next_k_buf = None
        next_v_buf = None
        next_grad_k_buf = None
        next_grad_v_buf = None

        for i in range(mesh_size):
            # --- Wait for k,v from previous async send ---
            if kv_work is not None:
                for w in kv_work:
                    w.wait()
                comm_stream.record_event(kv_done)
                compute_stream.wait_event(kv_done)
                k = next_k_buf
                v = next_v_buf
                kv_work = None

            # --- Wait for grad_k,v from previous async send ---
            if grad_work is not None:
                for w in grad_work:
                    w.wait()
                comm_stream.record_event(grad_done)
                compute_stream.wait_event(grad_done)
                grad_k = next_grad_k_buf
                grad_v = next_grad_v_buf
                grad_work = None

            # --- Async send k,v for next iteration (overlaps with compute) ---
            if i < mesh_size - 1:
                recv_idx = (i + 1) % 2
                next_k_buf = k_bufs[recv_idx]
                next_v_buf = v_bufs[recv_idx]

                comm_stream.wait_stream(compute_stream)
                with torch.cuda.stream(comm_stream):
                    _, kv_work_k = perform_ring_iteration_async(
                        k,
                        mesh,
                        ring_config,
                        recv_tensor=next_k_buf,
                    )
                    _, kv_work_v = perform_ring_iteration_async(
                        v,
                        mesh,
                        ring_config,
                        recv_tensor=next_v_buf,
                    )
                kv_work = kv_work_k + kv_work_v
                k.record_stream(comm_stream)
                v.record_stream(comm_stream)

            # --- Compute block gradients ---
            with record_function(f"sdpa_backward_{i}_{dist.get_rank()}"):
                (
                    block_grad_q,
                    block_grad_k,
                    block_grad_v,
                    _,
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

            # --- Send grad_k,v: async for non-last, blocking for last ---
            if i < mesh_size - 1:
                recv_idx = (i + 1) % 2
                next_grad_k_buf = grad_k_bufs[recv_idx]
                next_grad_v_buf = grad_v_bufs[recv_idx]

                comm_stream.wait_stream(compute_stream)
                with torch.cuda.stream(comm_stream):
                    _, grad_work_k = perform_ring_iteration_async(
                        grad_k,
                        mesh,
                        ring_config,
                        recv_tensor=next_grad_k_buf,
                    )
                    _, grad_work_v = perform_ring_iteration_async(
                        grad_v,
                        mesh,
                        ring_config,
                        recv_tensor=next_grad_v_buf,
                    )
                grad_work = grad_work_k + grad_work_v
                grad_k.record_stream(comm_stream)
                grad_v.record_stream(comm_stream)
            else:
                # Last iteration: blocking shift to place grads at the right rank
                grad_k = perform_ring_iteration(grad_k, mesh, ring_config)
                grad_v = perform_ring_iteration(grad_v, mesh, ring_config)

        return grad_q, grad_k, grad_v, grad_attn_mask, None, None, None


class RingSDPABlocking(torch.autograd.Function):
    r"""Performs scaled dot product attention on sharded Q, K, V.

    The ring allreduce happens in a blocking manner. This isn't more efficient, but
    it is useful for understanding the algorithm and debugging.

    For details about the ring attention, see:
    `Ring Attention <https://arxiv.org/abs/2310.01889>`_.
    Note that the original implementation is a combination of JAX + flash attention + ring attention.
    Here, instead, we leverage the underlying and built-in PyTorch efficient attention.

    A key difference with this algorithm is how we track the per-block normalizations. The PyTorch
    function returns log_sumexp, which we use for a running normalization. But it has to be kept in log
    space to prevent underflow/overflow as well as precision issues. See the helper functions
    ``add_log_sumexp`` and ``stable_signed_accumulate`` for more details.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        mesh: DeviceMesh,
        ring_config: RingPassingConfig,
        attn_args: dict,
    ) -> torch.Tensor:
        r"""Forward pass for the ring attention implementation.

        This implementation will NOT overlap the communication with the computation.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        q : torch.Tensor
            Query tensor of shape :math:`(B, H, S, D)`.
        k : torch.Tensor
            Key tensor of shape :math:`(B, H, S, D)`.
        v : torch.Tensor
            Value tensor of shape :math:`(B, H, S, D)`.
        attn_mask : Optional[torch.Tensor]
            Optional attention mask tensor.
        mesh : DeviceMesh
            Device mesh for distributed computation.
        ring_config : RingPassingConfig
            Configuration for ring passing communication.
        attn_args : dict
            Additional arguments to pass to the attention function.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, H, S, D)`.
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
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        r"""Backward pass for the ring SDPA.

        Currently, this is not overlapping communication with the computation.
        Note that the backward pass has 2x communication: send k, v but also grad_k, grad_v.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors from forward.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]
            Gradients for (q, k, v, attn_mask, mesh, ring_config, attn_args).
            ``None`` values indicate non-differentiable parameters.
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
    attn_mask: ShardTensor | None = None,
    **kwargs: dict,
) -> ShardTensor:
    r"""High-level, differentiable function to compute global attention on a sharded tensor.

    The implementation is a ring communication pattern. Each rank computes attention
    locally on its tensors, and then kv is passed to the next rank while receiving from
    the previous rank.

    Parameters
    ----------
    q : ShardTensor
        The attention queries.
    k : ShardTensor
        The attention keys.
    v : ShardTensor
        The attention values.
    attn_mask : Optional[ShardTensor], optional
        The attention mask.
    **kwargs : dict
        Keyword arguments to pass to the attention call.

    Returns
    -------
    ShardTensor
        A distributed tensor representing the attention computed on the global context.
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
    lq, lk, lv = (
        q.to_local().contiguous(),
        k.to_local().contiguous(),
        v.to_local().contiguous(),
    )

    if attn_mask is not None:
        latn_mask = attn_mask.to_local().contiguous()
    else:
        latn_mask = None

    x = RingSDPA.apply(lq, lk, lv, latn_mask, q._spec.mesh, ring_config, kwargs)

    # Convert back to ShardTensor
    x = ShardTensor.from_local(
        x, q._spec.mesh, q._spec.placements, q._spec.sharding_shapes()
    )
    return x


def sdpa_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:  # noqa: C901
    r"""Wrapper for ``torch.nn.functional.scaled_dot_product_attention`` to support sharded tensors.

    Parameters
    ----------
    func : Callable
        Will be ``torch.nn.functional.scaled_dot_product_attention``.
    types : Any
        The object types of the inputs.
    args : tuple
        Positional arguments containing query, key, value tensors.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    ShardTensor
        ShardTensor with global attention computed.

    Raises
    ------
    MissingShardPatch
        If sharding of inputs is not on the same mesh, or is not on a 1D mesh.
    """

    q, k, v, attn_mask, kwargs = repackage_sdpa_args(*args, **kwargs)

    # Make sure all tensors are on the same mesh
    if not (q._spec.mesh == k._spec.mesh == v._spec.mesh):
        raise MissingShardPatch("q, k, and v must all be on the same mesh")

    # Make sure the mesh is 1D
    if q._spec.mesh.ndim != 1:
        raise MissingShardPatch("q must be on a 1D mesh")

    # This is to implement sequence-parallel attention.
    # Make sure the shardings are all the same:
    if not (q._spec.placements[0] == k._spec.placements[0] == v._spec.placements[0]):
        raise MissingShardPatch("q, k, and v must all be on the same placement")

    # Make sure the attention mask, if provided, has the same placement as q, k, and v
    if attn_mask is not None and hasattr(attn_mask, "_spec"):
        if attn_mask._spec.placements[0] != q._spec.placements[0]:
            raise MissingShardPatch(
                "attn_mask must have the same placement as q, k, and v"
            )

    # if the placements are replicated (which is what we expect in transolver's
    # Physics Attention)
    # then just run locally and convert the output back to a replicated tensor:

    if v._spec.placements[0].is_replicate():
        local_q = q.to_local()
        local_k = k.to_local()
        local_v = v.to_local()
        if attn_mask is not None:
            local_attn_mask = attn_mask.to_local()
        else:
            local_attn_mask = None
        local_output = torch.nn.functional.scaled_dot_product_attention(
            local_q, local_k, local_v, attn_mask=local_attn_mask, **kwargs
        )

        output = ShardTensor.from_local(
            local_output,
            q._spec.mesh,
            q._spec.placements,
            # We don't have to worry about sharding shapes here since it's not sharded ...
        )
        return output
    else:
        return ring_sdpa(q, k, v, attn_mask, **kwargs)


def repackage_sdpa_args(
    query: torch.Tensor | ShardTensor,
    key: torch.Tensor | ShardTensor,
    value: torch.Tensor | ShardTensor,
    attn_mask: torch.Tensor | ShardTensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
    *args,
    **kwargs,
) -> tuple[
    torch.Tensor | ShardTensor,
    torch.Tensor | ShardTensor,
    torch.Tensor | ShardTensor,
    torch.Tensor | ShardTensor | None,
    dict,
]:
    r"""Repackage scaled dot product attention arguments into standard format.

    Parameters
    ----------
    query : Union[torch.Tensor, ShardTensor]
        Query tensor.
    key : Union[torch.Tensor, ShardTensor]
        Key tensor.
    value : Union[torch.Tensor, ShardTensor]
        Value tensor.
    attn_mask : Optional[Union[torch.Tensor, ShardTensor]], optional
        Attention mask tensor.
    dropout_p : float, default=0.0
        Dropout probability.
    is_causal : bool, default=False
        Whether to apply causal masking.
    scale : float, optional
        Scale factor for attention scores.
    enable_gqa : bool, default=False
        Whether to enable grouped query attention.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[Union[torch.Tensor, ShardTensor], Union[torch.Tensor, ShardTensor], Union[torch.Tensor, ShardTensor], Union[torch.Tensor, ShardTensor], dict]
        Tuple of (query, key, value, attn_mask, kwargs_dict).
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
