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
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
) -> None:
    """Validate cotangent divergence inputs for the Warp implementation."""
    function_name = "mesh_cotan_divergence"
    if points.ndim != 2:
        raise ValueError(
            f"{function_name}: points must have shape (n_points, dims), "
            f"got {points.shape=}"
        )
    if not torch.is_floating_point(points):
        raise TypeError(f"{function_name}: points must be floating-point")
    if vector_field.shape != points.shape:
        raise ValueError(
            f"{function_name}: vector_field shape must match points shape, "
            f"got {vector_field.shape} and {points.shape}"
        )
    if not torch.is_floating_point(vector_field):
        raise TypeError(f"{function_name}: vector_field must be floating-point")
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
    if dual_volumes.shape[0] != points.shape[0]:
        raise ValueError(
            f"{function_name}: dual_volumes length must match n_points: "
            f"{dual_volumes.shape[0]} != {points.shape[0]}"
        )
    if not torch.is_floating_point(dual_volumes):
        raise TypeError(f"{function_name}: dual_volumes must be floating-point")
    if (
        vector_field.device != points.device
        or edges.device != points.device
        or cotan_weights.device != points.device
        or dual_volumes.device != points.device
    ):
        raise ValueError(
            f"{function_name}: points, vector_field, and geometry must be on same device"
        )
    if edges.numel() > 0:
        # aminmax computes both bounds in one reduction; tolist performs one
        # device-to-host transfer instead of synchronizing once per bound.
        idx_min, idx_max = torch.stack(torch.aminmax(edges)).tolist()
        if idx_min < 0 or idx_max >= points.shape[0]:
            raise ValueError(
                f"{function_name}: edges must satisfy "
                f"0 <= index < n_points ({points.shape[0]})"
            )


@wp.kernel
def _cotan_divergence_forward_i32_kernel(
    points: wp.array2d(dtype=wp.float32),
    edges: wp.array2d(dtype=wp.int32),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    vector_field: wp.array2d(dtype=wp.float32),
    n_dims: int,
    eps: float,
    output: wp.array(dtype=wp.float32),
):
    edge_id = wp.tid()
    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]

    flat_flux = float(0.0)
    for dim in range(n_dims):
        edge_vector = points[v1, dim] - points[v0, dim]
        edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
        flat_flux = flat_flux + edge_average * edge_vector

    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps
    weighted_flux = cotan_weights[edge_id] * flat_flux
    wp.atomic_add(output, v0, weighted_flux / volume0)
    wp.atomic_add(output, v1, -weighted_flux / volume1)


@wp.kernel
def _cotan_divergence_forward_i64_kernel(
    points: wp.array2d(dtype=wp.float32),
    edges: wp.array2d(dtype=wp.int64),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    vector_field: wp.array2d(dtype=wp.float32),
    n_dims: int,
    eps: float,
    output: wp.array(dtype=wp.float32),
):
    edge_id = wp.tid()
    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]

    flat_flux = float(0.0)
    for dim in range(n_dims):
        edge_vector = points[v1, dim] - points[v0, dim]
        edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
        flat_flux = flat_flux + edge_average * edge_vector

    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps
    weighted_flux = cotan_weights[edge_id] * flat_flux
    wp.atomic_add(output, v0, weighted_flux / volume0)
    wp.atomic_add(output, v1, -weighted_flux / volume1)


@wp.kernel
def _cotan_divergence_backward_i32_kernel(
    points: wp.array2d(dtype=wp.float32),
    edges: wp.array2d(dtype=wp.int32),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    vector_field: wp.array2d(dtype=wp.float32),
    grad_output: wp.array(dtype=wp.float32),
    n_dims: int,
    eps: float,
    needs_points: int,
    needs_weights: int,
    needs_vector: int,
    grad_points: wp.array2d(dtype=wp.float32),
    grad_weights: wp.array(dtype=wp.float32),
    grad_vector: wp.array2d(dtype=wp.float32),
):
    edge_id = wp.tid()
    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps
    edge_q = grad_output[v0] / volume0 - grad_output[v1] / volume1

    if needs_weights != 0:
        flat_flux = float(0.0)
        for dim in range(n_dims):
            edge_vector = points[v1, dim] - points[v0, dim]
            edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
            flat_flux = flat_flux + edge_average * edge_vector
        grad_weights[edge_id] = edge_q * flat_flux

    if needs_points != 0:
        edge_scale = edge_q * cotan_weights[edge_id]
        for dim in range(n_dims):
            edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
            point_contrib = edge_scale * edge_average
            wp.atomic_add(grad_points, v0, dim, -point_contrib)
            wp.atomic_add(grad_points, v1, dim, point_contrib)

    if needs_vector != 0:
        edge_scale = edge_q * cotan_weights[edge_id]
        for dim in range(n_dims):
            edge_vector = points[v1, dim] - points[v0, dim]
            vector_contrib = 0.5 * edge_scale * edge_vector
            wp.atomic_add(grad_vector, v0, dim, vector_contrib)
            wp.atomic_add(grad_vector, v1, dim, vector_contrib)


@wp.kernel
def _cotan_divergence_backward_i64_kernel(
    points: wp.array2d(dtype=wp.float32),
    edges: wp.array2d(dtype=wp.int64),
    cotan_weights: wp.array(dtype=wp.float32),
    dual_volumes: wp.array(dtype=wp.float32),
    vector_field: wp.array2d(dtype=wp.float32),
    grad_output: wp.array(dtype=wp.float32),
    n_dims: int,
    eps: float,
    needs_points: int,
    needs_weights: int,
    needs_vector: int,
    grad_points: wp.array2d(dtype=wp.float32),
    grad_weights: wp.array(dtype=wp.float32),
    grad_vector: wp.array2d(dtype=wp.float32),
):
    edge_id = wp.tid()
    v0 = edges[edge_id, 0]
    v1 = edges[edge_id, 1]
    volume0 = dual_volumes[v0]
    volume1 = dual_volumes[v1]
    if volume0 < eps:
        volume0 = eps
    if volume1 < eps:
        volume1 = eps
    edge_q = grad_output[v0] / volume0 - grad_output[v1] / volume1

    if needs_weights != 0:
        flat_flux = float(0.0)
        for dim in range(n_dims):
            edge_vector = points[v1, dim] - points[v0, dim]
            edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
            flat_flux = flat_flux + edge_average * edge_vector
        grad_weights[edge_id] = edge_q * flat_flux

    if needs_points != 0:
        edge_scale = edge_q * cotan_weights[edge_id]
        for dim in range(n_dims):
            edge_average = 0.5 * (vector_field[v0, dim] + vector_field[v1, dim])
            point_contrib = edge_scale * edge_average
            wp.atomic_add(grad_points, v0, dim, -point_contrib)
            wp.atomic_add(grad_points, v1, dim, point_contrib)

    if needs_vector != 0:
        edge_scale = edge_q * cotan_weights[edge_id]
        for dim in range(n_dims):
            edge_vector = points[v1, dim] - points[v0, dim]
            vector_contrib = 0.5 * edge_scale * edge_vector
            wp.atomic_add(grad_vector, v0, dim, vector_contrib)
            wp.atomic_add(grad_vector, v1, dim, vector_contrib)


@wp.kernel
def _cotan_divergence_volume_backward_kernel(
    dual_volumes: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
    grad_output: wp.array(dtype=wp.float32),
    eps: float,
    grad_volumes: wp.array(dtype=wp.float32),
):
    point_id = wp.tid()
    volume = dual_volumes[point_id]
    if volume < eps:
        grad_volumes[point_id] = 0.0
    else:
        grad_volumes[point_id] = -grad_output[point_id] * output[point_id] / volume


def _empty_grad(reference: torch.Tensor) -> torch.Tensor:
    """Return the placeholder used for a gradient that was not requested."""
    return torch.empty((0,), dtype=torch.float32, device=reference.device)


def _divergence_backward_torch(
    *,
    points: torch.Tensor | None,
    edges: torch.Tensor | None,
    cotan_weights: torch.Tensor | None,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor | None,
    output: torch.Tensor | None,
    grad_output: torch.Tensor,
    needs_points: bool,
    needs_weights: bool,
    needs_volumes: bool,
    needs_vector: bool,
    eps: float,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute a differentiable fallback for higher-order gradients."""
    grad_points = None
    grad_weights = None
    grad_vector = None
    edge_needs = needs_points or needs_weights or needs_vector
    if edge_needs:
        edges = cast(torch.Tensor, edges)
        safe_volumes = dual_volumes.to(torch.float32).clamp(min=eps)
        q = grad_output.to(torch.float32) / safe_volumes
        v0 = edges[:, 0].to(torch.int64)
        v1 = edges[:, 1].to(torch.int64)
        edge_q = q[v0] - q[v1]

        edge_scale = None
        if needs_points or needs_vector:
            cotan_weights = cast(torch.Tensor, cotan_weights)
            edge_scale = edge_q * cotan_weights.to(torch.float32)

        if needs_points:
            vector_field = cast(torch.Tensor, vector_field)
            edge_scale = cast(torch.Tensor, edge_scale)
            vector_fp32 = vector_field.to(torch.float32)
            edge_average = 0.5 * (vector_fp32[v0] + vector_fp32[v1])
            point_contrib = edge_scale.view(-1, 1) * edge_average
            grad_points = torch.zeros_like(vector_fp32)
            grad_points.index_add_(0, v0, -point_contrib)
            grad_points.index_add_(0, v1, point_contrib)

        if needs_weights:
            points = cast(torch.Tensor, points)
            vector_field = cast(torch.Tensor, vector_field)
            points_fp32 = points.to(torch.float32)
            vector_fp32 = vector_field.to(torch.float32)
            edge_vectors = points_fp32[v1] - points_fp32[v0]
            edge_average = 0.5 * (vector_fp32[v0] + vector_fp32[v1])
            grad_weights = edge_q * (edge_average * edge_vectors).sum(dim=-1)

        if needs_vector:
            points = cast(torch.Tensor, points)
            edge_scale = cast(torch.Tensor, edge_scale)
            points_fp32 = points.to(torch.float32)
            edge_vectors = points_fp32[v1] - points_fp32[v0]
            vector_contrib = 0.5 * edge_scale.view(-1, 1) * edge_vectors
            grad_vector = torch.zeros_like(points_fp32)
            grad_vector.index_add_(0, v0, vector_contrib)
            grad_vector.index_add_(0, v1, vector_contrib)

    grad_volumes = None
    if needs_volumes:
        output = cast(torch.Tensor, output)
        volumes_fp32 = dual_volumes.to(torch.float32)
        safe_volumes = volumes_fp32.clamp(min=eps)
        grad_volumes = -(
            grad_output.to(torch.float32) * output.to(torch.float32) / safe_volumes
        )
        grad_volumes = grad_volumes * (volumes_fp32 >= eps)

    return grad_points, grad_weights, grad_volumes, grad_vector


@torch.library.custom_op(
    "physicsnemo::mesh_cotan_divergence_warp_backward_impl", mutates_args=()
)
def mesh_cotan_divergence_backward_impl(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
    needs_points: bool,
    needs_weights: bool,
    needs_vector: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the selective first-order edge backward as an opaque op."""
    points_fp32 = points.detach().to(dtype=torch.float32).contiguous()
    edges_index = edges.contiguous()
    weights_fp32 = cotan_weights.detach().to(dtype=torch.float32).contiguous()
    volumes_fp32 = dual_volumes.detach().to(dtype=torch.float32).contiguous()
    vector_fp32 = vector_field.detach().to(dtype=torch.float32).contiguous()
    grad_output_fp32 = grad_output.detach().to(dtype=torch.float32).contiguous()

    grad_points = (
        torch.zeros_like(points_fp32) if needs_points else _empty_grad(points_fp32)
    )
    grad_weights = (
        torch.empty((edges.shape[0],), dtype=torch.float32, device=edges.device)
        if needs_weights
        else _empty_grad(points_fp32)
    )
    grad_vector = (
        torch.zeros_like(vector_fp32) if needs_vector else _empty_grad(points_fp32)
    )

    if edges.shape[0] == 0:
        return grad_points, grad_weights, grad_vector

    # Warp cannot wrap every zero-sized Torch tensor. Keep the public custom-op
    # placeholders empty, but reuse compatible read-only inputs as inert launch
    # buffers for output branches that are statically disabled.
    grad_points_launch = grad_points if needs_points else points_fp32
    grad_weights_launch = grad_weights if needs_weights else grad_output_fp32
    grad_vector_launch = grad_vector if needs_vector else vector_fp32

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
    kernel = (
        _cotan_divergence_backward_i32_kernel
        if edges_index.dtype == torch.int32
        else _cotan_divergence_backward_i64_kernel
    )
    edge_dtype = wp.int32 if edges_index.dtype == torch.int32 else wp.int64
    with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
        wp.launch(
            kernel=kernel,
            dim=edges_index.shape[0],
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(edges_index, dtype=edge_dtype, return_ctype=True),
                wp.from_torch(weights_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(volumes_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(vector_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(grad_output_fp32, dtype=wp.float32, return_ctype=True),
                int(points_fp32.shape[1]),
                float(eps),
                int(needs_points),
                int(needs_weights),
                int(needs_vector),
                wp.from_torch(grad_points_launch, dtype=wp.float32, return_ctype=True),
                wp.from_torch(grad_weights_launch, dtype=wp.float32, return_ctype=True),
                wp.from_torch(grad_vector_launch, dtype=wp.float32, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return grad_points, grad_weights, grad_vector


@mesh_cotan_divergence_backward_impl.register_fake
def _mesh_cotan_divergence_backward_impl_fake(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
    needs_points: bool,
    needs_weights: bool,
    needs_vector: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate fake tensors for the selective edge backward."""
    _ = cotan_weights, dual_volumes, grad_output, eps
    empty = torch.empty((0,), dtype=torch.float32, device=points.device)
    grad_points = (
        torch.empty_like(points, dtype=torch.float32) if needs_points else empty
    )
    # Derive this shape from topology, not the possibly-placeholder weights
    # argument. In selective weight-only backward that placeholder may have
    # n_points elements while the true gradient has n_edges elements.
    grad_weights = (
        torch.empty((edges.shape[0],), dtype=torch.float32, device=edges.device)
        if needs_weights
        else empty
    )
    grad_vector = (
        torch.empty_like(vector_field, dtype=torch.float32) if needs_vector else empty
    )
    return grad_points, grad_weights, grad_vector


@torch.library.custom_op(
    "physicsnemo::mesh_cotan_divergence_warp_volume_backward_impl", mutates_args=()
)
def mesh_cotan_divergence_volume_backward_impl(
    dual_volumes: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Launch the pointwise dual-volume gradient without an edge kernel."""
    volumes_fp32 = dual_volumes.detach().to(dtype=torch.float32).contiguous()
    output_fp32 = output.detach().to(dtype=torch.float32).contiguous()
    grad_output_fp32 = grad_output.detach().to(dtype=torch.float32).contiguous()
    grad_volumes = torch.empty_like(volumes_fp32)
    if volumes_fp32.shape[0] == 0:
        return grad_volumes

    wp_device, wp_stream = FunctionSpec.warp_launch_context(volumes_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
        wp.launch(
            kernel=_cotan_divergence_volume_backward_kernel,
            dim=volumes_fp32.shape[0],
            inputs=[
                wp.from_torch(volumes_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(output_fp32, dtype=wp.float32, return_ctype=True),
                wp.from_torch(grad_output_fp32, dtype=wp.float32, return_ctype=True),
                float(eps),
                wp.from_torch(grad_volumes, dtype=wp.float32, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return grad_volumes


@mesh_cotan_divergence_volume_backward_impl.register_fake
def _mesh_cotan_divergence_volume_backward_impl_fake(
    dual_volumes: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Propagate fake tensors for the dual-volume backward."""
    _ = output, grad_output, eps
    return torch.empty_like(dual_volumes, dtype=torch.float32)


@torch.library.custom_op(
    "physicsnemo::mesh_cotan_divergence_warp_impl", mutates_args=()
)
def mesh_cotan_divergence_impl(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute cotangent/DEC mesh divergence with one fused Warp kernel."""
    _validate_inputs(
        points=points,
        edges=edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes,
        vector_field=vector_field,
    )

    points_fp32 = points.detach().to(dtype=torch.float32).contiguous()
    vector_fp32 = vector_field.detach().to(dtype=torch.float32).contiguous()
    edges_index = edges.contiguous()
    weights_fp32 = cotan_weights.detach().to(dtype=torch.float32).contiguous()
    volumes_fp32 = dual_volumes.detach().to(dtype=torch.float32).contiguous()
    output = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
    if edges_index.shape[0] > 0:
        wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
        kernel = (
            _cotan_divergence_forward_i32_kernel
            if edges_index.dtype == torch.int32
            else _cotan_divergence_forward_i64_kernel
        )
        edge_dtype = wp.int32 if edges_index.dtype == torch.int32 else wp.int64
        with FunctionSpec.warp_stream_scope(wp_stream, requires_cleanup_guard=False):
            wp.launch(
                kernel=kernel,
                dim=edges_index.shape[0],
                inputs=[
                    wp.from_torch(points_fp32, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(edges_index, dtype=edge_dtype, return_ctype=True),
                    wp.from_torch(weights_fp32, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(volumes_fp32, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(vector_fp32, dtype=wp.float32, return_ctype=True),
                    int(points_fp32.shape[1]),
                    float(eps),
                    wp.from_torch(output, dtype=wp.float32, return_ctype=True),
                ],
                device=wp_device,
                stream=wp_stream,
            )

    if output.dtype != vector_field.dtype:
        output = output.to(dtype=vector_field.dtype)
    return output


@mesh_cotan_divergence_impl.register_fake
def _mesh_cotan_divergence_impl_fake(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fake tensor propagation for cotangent divergence custom op."""
    _ = edges, cotan_weights, dual_volumes, eps
    return torch.empty(
        (points.shape[0],), device=points.device, dtype=vector_field.dtype
    )


def setup_mesh_cotan_divergence_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Save only the tensors required by the selected input gradients."""
    points, edges, cotan_weights, dual_volumes, vector_field, eps = inputs
    needs_points = ctx.needs_input_grad[0]
    needs_weights = ctx.needs_input_grad[2]
    needs_volumes = ctx.needs_input_grad[3]
    needs_vector = ctx.needs_input_grad[4]
    edge_needs = needs_points or needs_weights or needs_vector

    saved: list[torch.Tensor] = []
    saved_indices: dict[str, int] = {}

    def save(name: str, tensor: torch.Tensor) -> None:
        # A fixed mesh may be created during inference preprocessing and later
        # reused with a trainable vector field. PyTorch forbids saving inference
        # tensors for backward, so turn only those saved constants into ordinary
        # tensors at the autograd boundary. Inference tensors cannot themselves
        # require gradients, so this does not detach a differentiable input.
        if tensor.is_inference():
            tensor = tensor.clone()
        saved_indices[name] = len(saved)
        saved.append(tensor)

    if edge_needs:
        save("edges", edges)
    if needs_weights or needs_vector:
        save("points", points)
    if needs_points or needs_vector:
        save("cotan_weights", cotan_weights)
    if needs_points or needs_weights:
        save("vector_field", vector_field)
    if edge_needs or needs_volumes:
        save("dual_volumes", dual_volumes)
    if needs_volumes:
        save("output", output)

    ctx.save_for_backward(*saved)
    ctx.saved_indices = saved_indices
    ctx.eps = float(eps)
    ctx.points_dtype = points.dtype
    ctx.weights_dtype = cotan_weights.dtype
    ctx.volumes_dtype = dual_volumes.dtype
    ctx.vector_dtype = vector_field.dtype


def backward_mesh_cotan_divergence(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
]:
    """Backward pass for the cotangent divergence custom op."""
    needs_points = ctx.needs_input_grad[0]
    needs_weights = ctx.needs_input_grad[2]
    needs_volumes = ctx.needs_input_grad[3]
    needs_vector = ctx.needs_input_grad[4]
    if grad_output is None or not (
        needs_points or needs_weights or needs_volumes or needs_vector
    ):
        return None, None, None, None, None, None

    tensors = ctx.saved_tensors

    def get(name: str) -> torch.Tensor | None:
        index = ctx.saved_indices.get(name)
        return None if index is None else tensors[index]

    points = get("points")
    edges = get("edges")
    weights = get("cotan_weights")
    volumes = get("dual_volumes")
    vector_field = get("vector_field")
    output = get("output")
    volumes = cast(torch.Tensor, volumes)

    if torch.is_grad_enabled():
        grad_points, grad_weights, grad_volumes, grad_vector = (
            _divergence_backward_torch(
                points=points,
                edges=edges,
                cotan_weights=weights,
                dual_volumes=volumes,
                vector_field=vector_field,
                output=output,
                grad_output=grad_output,
                needs_points=needs_points,
                needs_weights=needs_weights,
                needs_volumes=needs_volumes,
                needs_vector=needs_vector,
                eps=ctx.eps,
            )
        )
    else:
        grad_points = None
        grad_weights = None
        grad_vector = None
        edge_needs = needs_points or needs_weights or needs_vector
        if edge_needs:
            edges = cast(torch.Tensor, edges)
            # Selective saves mean an unused custom-op argument may be backed by
            # a shape-compatible tensor (or grad_output for the unused weight).
            # Kernel branches never read an argument whose gradient is not needed.
            points_arg = points if points is not None else vector_field
            vector_arg = vector_field if vector_field is not None else points
            weights_arg = weights if weights is not None else grad_output
            points_arg = cast(torch.Tensor, points_arg)
            vector_arg = cast(torch.Tensor, vector_arg)
            grad_points_raw, grad_weights_raw, grad_vector_raw = (
                mesh_cotan_divergence_backward_impl(
                    points_arg,
                    edges,
                    weights_arg,
                    volumes,
                    vector_arg,
                    grad_output,
                    ctx.eps,
                    needs_points,
                    needs_weights,
                    needs_vector,
                )
            )
            if needs_points:
                grad_points = grad_points_raw
            if needs_weights:
                grad_weights = grad_weights_raw
            if needs_vector:
                grad_vector = grad_vector_raw

        grad_volumes = None
        if needs_volumes:
            output = cast(torch.Tensor, output)
            grad_volumes = mesh_cotan_divergence_volume_backward_impl(
                volumes, output, grad_output, ctx.eps
            )

    if grad_points is not None and grad_points.dtype != ctx.points_dtype:
        grad_points = grad_points.to(dtype=ctx.points_dtype)
    if grad_weights is not None and grad_weights.dtype != ctx.weights_dtype:
        grad_weights = grad_weights.to(dtype=ctx.weights_dtype)
    if grad_volumes is not None and grad_volumes.dtype != ctx.volumes_dtype:
        grad_volumes = grad_volumes.to(dtype=ctx.volumes_dtype)
    if grad_vector is not None and grad_vector.dtype != ctx.vector_dtype:
        grad_vector = grad_vector.to(dtype=ctx.vector_dtype)

    return grad_points, None, grad_weights, grad_volumes, grad_vector, None


mesh_cotan_divergence_impl.register_autograd(
    backward_mesh_cotan_divergence,
    setup_context=setup_mesh_cotan_divergence_context,
)


def mesh_cotan_divergence_warp(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute cotangent/DEC mesh divergence with Warp kernels."""
    eps = _safe_eps(torch.float32)
    return mesh_cotan_divergence_impl(
        points,
        edges,
        cotan_weights,
        dual_volumes,
        vector_field,
        float(eps),
    )
