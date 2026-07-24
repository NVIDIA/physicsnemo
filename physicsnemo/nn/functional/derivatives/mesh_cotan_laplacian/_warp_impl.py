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

import math
from typing import cast

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

wp.init()
wp.config.log_level = wp.LOG_WARNING


def _safe_eps(dtype: torch.dtype) -> float:
    """Return a dtype-aware floor for dual-volume normalization."""
    info = torch.finfo(dtype)
    return min(info.tiny**0.25, info.eps)


def _validate_inputs(
    *,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Validate cotangent Laplacian inputs for the Warp implementation."""
    function_name = "mesh_cotan_laplacian"
    if values.ndim < 1:
        raise ValueError(
            f"{function_name}: values must have shape (n_points, ...), "
            f"got {values.shape=}"
        )
    if not torch.is_floating_point(values):
        raise TypeError(f"{function_name}: values must be floating-point")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"{function_name}: edges must have shape (n_edges, 2), got {edges.shape=}"
        )
    if edges.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{function_name}: edges must be int32 or int64")
    if cotan_weights.ndim != 1:
        raise ValueError(
            f"{function_name}: cotan_weights must have shape (n_edges,), "
            f"got {cotan_weights.shape=}"
        )
    if cotan_weights.shape[0] != edges.shape[0]:
        raise ValueError(
            f"{function_name}: cotan_weights length must match edges: "
            f"{cotan_weights.shape[0]} != {edges.shape[0]}"
        )
    if not torch.is_floating_point(cotan_weights):
        raise TypeError(f"{function_name}: cotan_weights must be floating-point")
    if dual_volumes.ndim != 1:
        raise ValueError(
            f"{function_name}: dual_volumes must have shape (n_points,), "
            f"got {dual_volumes.shape=}"
        )
    if dual_volumes.shape[0] != values.shape[0]:
        raise ValueError(
            f"{function_name}: dual_volumes length must match n_points: "
            f"{dual_volumes.shape[0]} != {values.shape[0]}"
        )
    if not torch.is_floating_point(dual_volumes):
        raise TypeError(f"{function_name}: dual_volumes must be floating-point")
    if (
        values.device != edges.device
        or edges.device != cotan_weights.device
        or edges.device != dual_volumes.device
    ):
        raise ValueError(f"{function_name}: values and geometry must be on same device")
    if edges.numel() > 0:
        # Keep validation to one device-to-host transfer rather than separately
        # synchronizing for the lower and upper bounds.
        idx_min, idx_max = torch.stack(torch.aminmax(edges)).tolist()
        if idx_min < 0 or idx_max >= values.shape[0]:
            raise ValueError(
                f"{function_name}: edges must satisfy "
                f"0 <= index < n_points ({values.shape[0]})"
            )


@wp.kernel
def _cotan_laplacian_forward_i32_kernel(
    edges: wp.array2d(dtype=wp.int32),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    eps: float,
    n_components: int,
    output: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    edge_id = tid // n_components
    comp = tid - edge_id * n_components

    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps

    delta = values[v1, comp] - values[v0, comp]
    contrib = cotan_weights[edge_id] * delta
    wp.atomic_add(output, v0, comp, contrib / volume0)
    wp.atomic_add(output, v1, comp, -contrib / volume1)


@wp.kernel
def _cotan_laplacian_forward_i64_kernel(
    edges: wp.array2d(dtype=wp.int64),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    eps: float,
    n_components: int,
    output: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    edge_id = tid // n_components
    comp = tid - edge_id * n_components

    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps

    delta = values[v1, comp] - values[v0, comp]
    contrib = cotan_weights[edge_id] * delta
    wp.atomic_add(output, v0, comp, contrib / volume0)
    wp.atomic_add(output, v1, comp, -contrib / volume1)


@wp.kernel
def _cotan_laplacian_backward_edges_i32_kernel(
    edges: wp.array2d(dtype=wp.int32),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    grad_output: wp.array2d(dtype=wp.float32),
    eps: float,
    n_components: int,
    needs_weights: int,
    needs_values: int,
    grad_weights: wp.array(dtype=wp.float32),
    grad_values: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    edge_id = tid // n_components
    comp = tid - edge_id * n_components

    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps

    qdiff = grad_output[v0, comp] / volume0 - grad_output[v1, comp] / volume1
    if needs_weights != 0:
        delta = values[v1, comp] - values[v0, comp]
        wp.atomic_add(grad_weights, edge_id, qdiff * delta)
    if needs_values != 0:
        contrib = cotan_weights[edge_id] * qdiff
        wp.atomic_add(grad_values, v0, comp, -contrib)
        wp.atomic_add(grad_values, v1, comp, contrib)


@wp.kernel
def _cotan_laplacian_backward_edges_i64_kernel(
    edges: wp.array2d(dtype=wp.int64),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    grad_output: wp.array2d(dtype=wp.float32),
    eps: float,
    n_components: int,
    needs_weights: int,
    needs_values: int,
    grad_weights: wp.array(dtype=wp.float32),
    grad_values: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    edge_id = tid // n_components
    comp = tid - edge_id * n_components

    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps

    qdiff = grad_output[v0, comp] / volume0 - grad_output[v1, comp] / volume1
    if needs_weights != 0:
        delta = values[v1, comp] - values[v0, comp]
        wp.atomic_add(grad_weights, edge_id, qdiff * delta)
    if needs_values != 0:
        contrib = cotan_weights[edge_id] * qdiff
        wp.atomic_add(grad_values, v0, comp, -contrib)
        wp.atomic_add(grad_values, v1, comp, contrib)


@wp.kernel
def _cotan_laplacian_backward_volumes_kernel(
    dual_volumes: wp.array(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
    grad_output: wp.array2d(dtype=wp.float32),
    eps: float,
    n_components: int,
    grad_volumes: wp.array(dtype=wp.float32),
):
    point_id = wp.tid()
    volume = dual_volumes[point_id]
    if volume >= eps:
        grad_volume = float(0.0)
        for comp in range(n_components):
            grad_volume = grad_volume - (
                grad_output[point_id, comp] * output[point_id, comp] / volume
            )
        grad_volumes[point_id] = grad_volume


def _laplacian_backward_torch(
    *,
    edges: torch.Tensor | None,
    cotan_weights: torch.Tensor | None,
    dual_volumes: torch.Tensor,
    values_flat: torch.Tensor | None,
    output_flat: torch.Tensor | None,
    grad_output_flat: torch.Tensor,
    needs_weights: bool,
    needs_volumes: bool,
    needs_values: bool,
    eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Differentiable fallback used when a higher-order graph is requested."""
    volumes_fp32 = dual_volumes.to(dtype=torch.float32)
    grad_output_fp32 = grad_output_flat.to(dtype=torch.float32)
    safe_volumes = volumes_fp32.clamp(min=eps)

    grad_weights = None
    grad_values = None
    if needs_weights or needs_values:
        edge_indices = cast(torch.Tensor, edges).to(
            dtype=torch.int64, device=grad_output_flat.device
        )
        v0 = edge_indices[:, 0]
        v1 = edge_indices[:, 1]
        q = grad_output_fp32 / safe_volumes.view(-1, 1)
        qdiff = q[v0] - q[v1]

        if needs_weights:
            values_fp32 = cast(torch.Tensor, values_flat).to(dtype=torch.float32)
            value_delta = values_fp32[v1] - values_fp32[v0]
            grad_weights = (qdiff * value_delta).sum(dim=-1)

        if needs_values:
            weights_fp32 = cast(torch.Tensor, cotan_weights).to(dtype=torch.float32)
            weighted_qdiff = weights_fp32.view(-1, 1) * qdiff
            grad_values = torch.zeros_like(grad_output_fp32)
            grad_values.index_add_(0, v0, -weighted_qdiff)
            grad_values.index_add_(0, v1, weighted_qdiff)

    grad_volumes = None
    if needs_volumes:
        output_fp32 = cast(torch.Tensor, output_flat).to(dtype=torch.float32)
        grad_volumes = -(grad_output_fp32 * output_fp32 / safe_volumes.view(-1, 1)).sum(
            dim=-1
        )
        grad_volumes = grad_volumes * (volumes_fp32 >= eps)

    return grad_weights, grad_volumes, grad_values


@torch.library.custom_op("physicsnemo::mesh_cotan_laplacian_warp_impl", mutates_args=())
def mesh_cotan_laplacian_impl(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the normalized cotangent Laplacian with a fused Warp kernel."""
    _validate_inputs(
        edges=edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes,
        values=values,
    )

    n_points = values.shape[0]
    value_shape = values.shape[1:]
    n_components = math.prod(value_shape) if value_shape else 1
    output_flat = torch.zeros(
        (n_points, n_components), dtype=torch.float32, device=values.device
    )

    # Warp cannot wrap every zero-sized Torch tensor. Return the mathematically
    # correct zeros before creating any descriptors or launch context.
    if n_points == 0 or edges.shape[0] == 0 or n_components == 0:
        output = output_flat.reshape(n_points, *value_shape)
        return output.to(dtype=values.dtype)

    edges_contiguous = edges.contiguous()
    weights_fp32 = cotan_weights.to(dtype=torch.float32).contiguous()
    volumes_fp32 = dual_volumes.to(dtype=torch.float32).contiguous()
    values_fp32 = (
        values.to(dtype=torch.float32).reshape(n_points, n_components).contiguous()
    )

    wp_edges = wp.from_torch(edges_contiguous, return_ctype=True)
    wp_weights = wp.from_torch(weights_fp32, return_ctype=True)
    wp_volumes = wp.from_torch(volumes_fp32, return_ctype=True)
    wp_values = wp.from_torch(values_fp32, return_ctype=True)
    wp_output = wp.from_torch(output_flat, return_ctype=True)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(values_fp32)
    kernel = (
        _cotan_laplacian_forward_i32_kernel
        if edges.dtype == torch.int32
        else _cotan_laplacian_forward_i64_kernel
    )
    with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
        wp.launch(
            kernel=kernel,
            dim=edges.shape[0] * n_components,
            inputs=[
                wp_edges,
                wp_weights,
                wp_volumes,
                wp_values,
                float(eps),
                int(n_components),
                wp_output,
            ],
            device=wp_device,
            stream=wp_stream,
        )

    output = output_flat.reshape(n_points, *value_shape)
    if output.dtype != values.dtype:
        output = output.to(dtype=values.dtype)
    return output


@mesh_cotan_laplacian_impl.register_fake
def _mesh_cotan_laplacian_impl_fake(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fake tensor propagation for the forward custom op."""
    _ = edges, cotan_weights, dual_volumes, eps
    return torch.empty_like(values)


@torch.library.custom_op(
    "physicsnemo::mesh_cotan_laplacian_warp_backward_edges_impl", mutates_args=()
)
def _mesh_cotan_laplacian_backward_edges_impl(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values_flat: torch.Tensor,
    grad_output_flat: torch.Tensor,
    needs_weights: bool,
    needs_values: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute first-order weight/value gradients with a Warp kernel."""
    n_edges = edges.shape[0]
    n_points, n_components = grad_output_flat.shape
    grad_weights = (
        torch.zeros((n_edges,), dtype=torch.float32, device=edges.device)
        if needs_weights
        else torch.empty((0,), dtype=torch.float32, device=edges.device)
    )
    grad_values = (
        torch.zeros(
            (n_points, n_components),
            dtype=torch.float32,
            device=grad_output_flat.device,
        )
        if needs_values
        else torch.empty((0, 0), dtype=torch.float32, device=grad_output_flat.device)
    )

    # In particular, do not pass empty meshes/components to wp.from_torch.
    if n_edges == 0 or n_points == 0 or n_components == 0:
        return grad_weights, grad_values

    edges_contiguous = edges.contiguous()
    weights_fp32 = cotan_weights.to(dtype=torch.float32).contiguous()
    volumes_fp32 = dual_volumes.to(dtype=torch.float32).contiguous()
    values_fp32 = values_flat.to(dtype=torch.float32).contiguous()
    grad_output_fp32 = grad_output_flat.to(dtype=torch.float32).contiguous()

    # Unrequested outputs use compatible, nonempty input tensors as inert
    # placeholders. Runtime flags ensure the kernel never writes through them.
    grad_weights_arg = grad_weights if needs_weights else weights_fp32
    grad_values_arg = grad_values if needs_values else grad_output_fp32

    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    kernel = (
        _cotan_laplacian_backward_edges_i32_kernel
        if edges.dtype == torch.int32
        else _cotan_laplacian_backward_edges_i64_kernel
    )
    with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
        wp.launch(
            kernel=kernel,
            dim=n_edges * n_components,
            inputs=[
                wp.from_torch(edges_contiguous, return_ctype=True),
                wp.from_torch(weights_fp32, return_ctype=True),
                wp.from_torch(volumes_fp32, return_ctype=True),
                wp.from_torch(values_fp32, return_ctype=True),
                wp.from_torch(grad_output_fp32, return_ctype=True),
                float(eps),
                int(n_components),
                int(needs_weights),
                int(needs_values),
                wp.from_torch(grad_weights_arg, return_ctype=True),
                wp.from_torch(grad_values_arg, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return grad_weights, grad_values


@_mesh_cotan_laplacian_backward_edges_impl.register_fake
def _mesh_cotan_laplacian_backward_edges_impl_fake(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values_flat: torch.Tensor,
    grad_output_flat: torch.Tensor,
    needs_weights: bool,
    needs_values: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake tensor propagation for the edge-gradient custom op."""
    _ = cotan_weights, dual_volumes, values_flat, eps
    grad_weights = (
        torch.empty((edges.shape[0],), dtype=torch.float32, device=edges.device)
        if needs_weights
        else torch.empty((0,), dtype=torch.float32, device=edges.device)
    )
    grad_values = (
        torch.empty_like(grad_output_flat, dtype=torch.float32)
        if needs_values
        else torch.empty((0, 0), dtype=torch.float32, device=grad_output_flat.device)
    )
    return grad_weights, grad_values


@torch.library.custom_op(
    "physicsnemo::mesh_cotan_laplacian_warp_backward_volumes_impl", mutates_args=()
)
def _mesh_cotan_laplacian_backward_volumes_impl(
    dual_volumes: torch.Tensor,
    output_flat: torch.Tensor,
    grad_output_flat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute first-order dual-volume gradients with a Warp kernel."""
    grad_volumes = torch.zeros_like(dual_volumes, dtype=torch.float32)
    n_points, n_components = grad_output_flat.shape
    if n_points == 0 or n_components == 0:
        return grad_volumes

    volumes_fp32 = dual_volumes.to(dtype=torch.float32).contiguous()
    output_fp32 = output_flat.to(dtype=torch.float32).contiguous()
    grad_output_fp32 = grad_output_flat.to(dtype=torch.float32).contiguous()
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
        wp.launch(
            kernel=_cotan_laplacian_backward_volumes_kernel,
            dim=n_points,
            inputs=[
                wp.from_torch(volumes_fp32, return_ctype=True),
                wp.from_torch(output_fp32, return_ctype=True),
                wp.from_torch(grad_output_fp32, return_ctype=True),
                float(eps),
                int(n_components),
                wp.from_torch(grad_volumes, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return grad_volumes


@_mesh_cotan_laplacian_backward_volumes_impl.register_fake
def _mesh_cotan_laplacian_backward_volumes_impl_fake(
    dual_volumes: torch.Tensor,
    output_flat: torch.Tensor,
    grad_output_flat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fake tensor propagation for the volume-gradient custom op."""
    _ = output_flat, grad_output_flat, eps
    return torch.empty_like(dual_volumes, dtype=torch.float32)


def _save_for_backward(
    ctx: torch.autograd.function.FunctionCtx,
    tensors: list[torch.Tensor],
    tensor: torch.Tensor,
) -> int:
    """Append a tensor and return its position in ``ctx.saved_tensors``."""
    # Fixed geometry may originate in inference preprocessing. PyTorch cannot
    # save inference tensors for backward, so materialize an ordinary tensor at
    # this boundary. Inference tensors cannot themselves require gradients.
    if tensor.is_inference():
        tensor = tensor.clone()
    slot = len(tensors)
    tensors.append(tensor)
    return slot


def setup_mesh_cotan_laplacian_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store only tensors required by the requested input gradients."""
    edges, cotan_weights, dual_volumes, values, eps = inputs
    needs_weights = ctx.needs_input_grad[1]
    needs_volumes = ctx.needs_input_grad[2]
    needs_values = ctx.needs_input_grad[3]

    tensors: list[torch.Tensor] = []
    ctx.edges_slot = -1
    ctx.weights_slot = -1
    ctx.volumes_slot = -1
    ctx.values_slot = -1
    ctx.output_slot = -1
    if needs_weights or needs_values:
        ctx.edges_slot = _save_for_backward(ctx, tensors, edges)
    if needs_values:
        ctx.weights_slot = _save_for_backward(ctx, tensors, cotan_weights)
    if needs_weights or needs_volumes or needs_values:
        ctx.volumes_slot = _save_for_backward(ctx, tensors, dual_volumes)
    if needs_weights:
        ctx.values_slot = _save_for_backward(ctx, tensors, values)
    if needs_volumes:
        ctx.output_slot = _save_for_backward(ctx, tensors, output)
    ctx.save_for_backward(*tensors)

    ctx.eps = float(eps)
    ctx.values_shape = values.shape
    ctx.weights_dtype = cotan_weights.dtype
    ctx.volumes_dtype = dual_volumes.dtype
    ctx.values_dtype = values.dtype


def backward_mesh_cotan_laplacian(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
    """Backward pass for the cotangent Laplacian custom op."""
    needs_weights = ctx.needs_input_grad[1]
    needs_volumes = ctx.needs_input_grad[2]
    needs_values = ctx.needs_input_grad[3]
    if grad_output is None or not (needs_weights or needs_volumes or needs_values):
        return None, None, None, None, None

    saved = ctx.saved_tensors
    edges = saved[ctx.edges_slot] if ctx.edges_slot >= 0 else None
    cotan_weights = saved[ctx.weights_slot] if ctx.weights_slot >= 0 else None
    dual_volumes = saved[ctx.volumes_slot]
    values = saved[ctx.values_slot] if ctx.values_slot >= 0 else None
    output = saved[ctx.output_slot] if ctx.output_slot >= 0 else None

    n_points = grad_output.shape[0]
    value_shape = ctx.values_shape[1:]
    n_components = math.prod(value_shape) if value_shape else 1
    grad_output_flat = grad_output.reshape(n_points, n_components)
    values_flat = values.reshape(n_points, n_components) if values is not None else None
    output_flat = output.reshape(n_points, n_components) if output is not None else None

    if torch.is_grad_enabled():
        grad_weights, grad_volumes, grad_values_flat = _laplacian_backward_torch(
            edges=edges,
            cotan_weights=cotan_weights,
            dual_volumes=dual_volumes,
            values_flat=values_flat,
            output_flat=output_flat,
            grad_output_flat=grad_output_flat,
            needs_weights=needs_weights,
            needs_volumes=needs_volumes,
            needs_values=needs_values,
            eps=ctx.eps,
        )
    else:
        grad_weights = None
        grad_values_flat = None
        if needs_weights or needs_values:
            # The edge kernel does not read these placeholders when their
            # corresponding gradient flag is false.
            weights_arg = cotan_weights if cotan_weights is not None else dual_volumes
            values_arg = values_flat if values_flat is not None else grad_output_flat
            grad_weights_out, grad_values_out = (
                _mesh_cotan_laplacian_backward_edges_impl(
                    cast(torch.Tensor, edges),
                    weights_arg,
                    dual_volumes,
                    values_arg,
                    grad_output_flat,
                    needs_weights,
                    needs_values,
                    ctx.eps,
                )
            )
            if needs_weights:
                grad_weights = grad_weights_out
            if needs_values:
                grad_values_flat = grad_values_out

        grad_volumes = None
        if needs_volumes:
            grad_volumes = _mesh_cotan_laplacian_backward_volumes_impl(
                dual_volumes,
                cast(torch.Tensor, output_flat),
                grad_output_flat,
                ctx.eps,
            )

    if grad_weights is not None and grad_weights.dtype != ctx.weights_dtype:
        grad_weights = grad_weights.to(dtype=ctx.weights_dtype)
    if grad_volumes is not None and grad_volumes.dtype != ctx.volumes_dtype:
        grad_volumes = grad_volumes.to(dtype=ctx.volumes_dtype)
    grad_values = None
    if grad_values_flat is not None:
        grad_values = grad_values_flat.reshape(ctx.values_shape)
        if grad_values.dtype != ctx.values_dtype:
            grad_values = grad_values.to(dtype=ctx.values_dtype)

    return None, grad_weights, grad_volumes, grad_values, None


mesh_cotan_laplacian_impl.register_autograd(
    backward_mesh_cotan_laplacian,
    setup_context=setup_mesh_cotan_laplacian_context,
)


def mesh_cotan_laplacian_warp(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply the normalized cotangent Laplacian with Warp kernels."""
    eps = _safe_eps(torch.float32)
    return mesh_cotan_laplacian_impl(
        edges,
        cotan_weights,
        dual_volumes,
        values,
        float(eps),
    )
