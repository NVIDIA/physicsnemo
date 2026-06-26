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

"""Warp-accelerated CSR segmented softmax."""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .kernels import segment_softmax_backward_kernel, segment_softmax_forward_kernel
from .utils import flatten_logits, validate_inputs

wp.config.log_level = wp.LOG_WARNING
wp.init()


def _require_cuda(tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(
            "The Warp segment_softmax backend requires CUDA tensors; "
            "use implementation='torch' on CPU."
        )


@torch.library.custom_op("physicsnemo::segment_softmax_warp", mutates_args=())
def segment_softmax(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Warp segmented softmax custom op.

    The Warp kernels compute in float32 and the public result is cast back to
    the input dtype. The operation is grouped by CSR ``offsets`` along axis 0.
    """
    validate_inputs(logits, offsets)
    _require_cuda(logits)
    if int(logits.shape[0]) == 0 or int(logits.numel()) == 0:
        return logits.clone()

    input_dtype = logits.dtype
    logits_f = logits.to(torch.float32).contiguous()
    offsets_i64 = offsets.to(dtype=torch.int64).contiguous()
    flat, original_shape = flatten_logits(logits_f)
    out = torch.empty_like(flat)
    num_segments = int(offsets_i64.shape[0]) - 1
    num_channels = int(flat.shape[1])

    wp_device, wp_stream = FunctionSpec.warp_launch_context(flat)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            segment_softmax_forward_kernel,
            dim=(num_segments, num_channels),
            inputs=[
                wp.from_torch(flat, return_ctype=True),
                wp.from_torch(offsets_i64, return_ctype=True),
                wp.from_torch(out, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )

    return out.reshape(original_shape).to(input_dtype)


@segment_softmax.register_fake
def _(
    logits: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(logits)


@torch.library.custom_op(
    "physicsnemo::segment_softmax_warp_backward", mutates_args=()
)
def segment_softmax_backward(
    grad_out: torch.Tensor,
    softmax_out: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Backward custom op for Warp segmented softmax."""
    validate_inputs(softmax_out, offsets)
    if grad_out.shape != softmax_out.shape:
        raise ValueError(
            "grad_out and softmax_out must have the same shape, got "
            f"{tuple(grad_out.shape)} and {tuple(softmax_out.shape)}"
        )
    if grad_out.device != softmax_out.device:
        raise ValueError(
            "grad_out and softmax_out must be on the same device, got "
            f"{grad_out.device} and {softmax_out.device}"
        )
    _require_cuda(softmax_out)
    if int(softmax_out.shape[0]) == 0 or int(softmax_out.numel()) == 0:
        return grad_out.clone()

    grad_dtype = grad_out.dtype
    grad_f = grad_out.to(torch.float32).contiguous()
    out_f = softmax_out.to(torch.float32).contiguous()
    offsets_i64 = offsets.to(dtype=torch.int64).contiguous()
    grad_flat, original_shape = flatten_logits(grad_f)
    out_flat, _ = flatten_logits(out_f)
    grad_logits = torch.empty_like(grad_flat)
    num_segments = int(offsets_i64.shape[0]) - 1
    num_channels = int(grad_flat.shape[1])

    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_flat)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            segment_softmax_backward_kernel,
            dim=(num_segments, num_channels),
            inputs=[
                wp.from_torch(grad_flat, return_ctype=True),
                wp.from_torch(out_flat, return_ctype=True),
                wp.from_torch(offsets_i64, return_ctype=True),
                wp.from_torch(grad_logits, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )

    return grad_logits.reshape(original_shape).to(grad_dtype)


@segment_softmax_backward.register_fake
def _(
    grad_out: torch.Tensor,
    softmax_out: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(grad_out)


def setup_segment_softmax_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    _, offsets = inputs
    ctx.save_for_backward(output, offsets)


def backward_segment_softmax(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    softmax_out, offsets = ctx.saved_tensors
    grad_logits = segment_softmax_backward(grad_out, softmax_out, offsets)
    return grad_logits, None


segment_softmax.register_autograd(
    backward_segment_softmax, setup_context=setup_segment_softmax_context
)
