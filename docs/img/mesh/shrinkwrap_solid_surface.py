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

"""Render selected cap repair on a broad optimized lid dome.

The closed triangular enclosure represents a candidate produced by shape
optimization. Its inset lid forms one broad dome whose upper portion crosses a
horizontal packaging clearance plane. The example derives a boolean
``point_weights`` mask from vertices actually above the plane. The real
:meth:`physicsnemo.mesh.Mesh.shrinkwrap` API then projects only that cap onto
the plane. Every admissible optimized vertex remains exactly unchanged.

Run from the repository root:

.. code-block:: console

    uv run --no-sync python \
        docs/img/mesh/shrinkwrap_solid_surface.py

The Torch backend is always validated on CPU. When CUDA is available, the Warp
backend is also run on CUDA and compared with the Torch result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyvista as pv
import torch
import torch.nn.functional as F

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).with_suffix(".png")
LID_LENGTH = 3.80
LID_WIDTH = 1.85
BODY_LENGTH = 4.25
BODY_WIDTH = 2.25
BOTTOM_Z = -0.32
SHOULDER_Z = 0.14
BASE_TOP_Z = 0.36
CLEARANCE_Z = 0.74
TOP_X_POINTS = 51
TOP_Y_POINTS = 37
PLANE_X_POINTS = 27
PLANE_Y_POINTS = 19

DOME_HEIGHT = 0.82

PLANE_COLOR = "#3182A8"
VIOLATION_COLOR = "#F06B32"
REPAIRED_COLOR = "#2B9A66"
OPTIMIZED_COLOR = "#D59B2D"
SHELL_COLOR = "#A8B4C0"
EDGE_COLOR = "#34495E"
TEXT_COLOR = "#1E293B"


@dataclass(frozen=True)
class SurfaceCase:
    """Constraint plane, optimized solid, and point selections."""

    target: Mesh
    source: Mesh
    optimized: torch.Tensor
    movable: torch.Tensor


@dataclass(frozen=True)
class SurfaceMetrics:
    """Geometry, feasibility, and preservation metrics."""

    source_volume: float
    result_volume: float
    fixed_error: float
    retained_error: float
    surface_residual: float
    plane_error: float
    retained_deformation: float
    max_correction: float


@dataclass(frozen=True)
class AdjointProbe:
    """Source and target gradients from one deterministic scalar probe."""

    source: torch.Tensor
    target: torch.Tensor


def _triangle_grid_faces(
    point_ids: torch.Tensor,
    *,
    upward: bool,
) -> torch.Tensor:
    """Triangulate a structured point grid with one consistent orientation."""

    lower_left = point_ids[:-1, :-1].reshape(-1)
    upper_left = point_ids[1:, :-1].reshape(-1)
    lower_right = point_ids[:-1, 1:].reshape(-1)
    upper_right = point_ids[1:, 1:].reshape(-1)
    if upward:
        return torch.cat(
            (
                torch.stack((lower_left, upper_left, upper_right), dim=1),
                torch.stack((lower_left, upper_right, lower_right), dim=1),
            ),
            dim=0,
        )
    return torch.cat(
        (
            torch.stack((lower_left, upper_right, upper_left), dim=1),
            torch.stack((lower_left, lower_right, upper_right), dim=1),
        ),
        dim=0,
    )


def _grid_perimeter(point_ids: torch.Tensor) -> torch.Tensor:
    """Return one counterclockwise rectangular boundary loop."""

    return torch.cat(
        (
            point_ids[:, 0],
            point_ids[-1, 1:],
            point_ids[:-1, -1].flip(0),
            point_ids[0, 1:-1].flip(0),
        )
    )


def _rectangular_ring_points(
    *,
    length: float,
    width: float,
    height: float,
    device: torch.device,
) -> torch.Tensor:
    """Build one rectangular perimeter ring with the lid sampling."""

    x = torch.linspace(-0.5 * length, 0.5 * length, TOP_X_POINTS, device=device)
    y = torch.linspace(-0.5 * width, 0.5 * width, TOP_Y_POINTS, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack(
        (
            x_grid,
            y_grid,
            torch.full_like(x_grid, height),
        ),
        dim=-1,
    ).reshape(-1, 3)
    ids = torch.arange(len(grid), device=device).reshape(
        TOP_X_POINTS,
        TOP_Y_POINTS,
    )
    return grid[_grid_perimeter(ids)]


def _connect_rings(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Triangulate a strip between corresponding counterclockwise rings."""

    first_next = first.roll(-1)
    second_next = second.roll(-1)
    return torch.cat(
        (
            torch.stack((first, first_next, second_next), dim=1),
            torch.stack((first, second_next, second), dim=1),
        ),
        dim=0,
    )


def _signed_enclosed_volume(mesh: Mesh) -> torch.Tensor:
    """Return the signed volume enclosed by an oriented triangle surface."""

    triangles = mesh.points[mesh.cells]
    return (
        triangles[:, 0]
        * torch.linalg.cross(
            triangles[:, 1],
            triangles[:, 2],
            dim=1,
        )
    ).sum() / 6.0


def _validate_triangle_geometry(mesh: Mesh, name: str) -> None:
    """Check triangular connectivity and nondegenerate geometry."""

    if mesh.cells.ndim != 2 or mesh.cells.shape[1] != 3:
        raise RuntimeError(f"{name} must contain only triangles")
    triangles = mesh.points[mesh.cells]
    double_areas = torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        ),
        dim=1,
    )
    if bool((double_areas <= 1.0e-8).any()):
        raise RuntimeError(f"{name} contains a degenerate triangle")


def _validate_closed_oriented(mesh: Mesh, name: str) -> float:
    """Check watertight edges, consistent orientation, and positive volume."""

    _validate_triangle_geometry(mesh, name)
    directed_edges = torch.cat(
        (
            mesh.cells[:, (0, 1)],
            mesh.cells[:, (1, 2)],
            mesh.cells[:, (2, 0)],
        ),
        dim=0,
    )
    canonical_edges = directed_edges.sort(dim=1).values
    _, inverse, counts = torch.unique(
        canonical_edges,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    if not bool((counts == 2).all()):
        raise RuntimeError(f"{name} is not a closed two-manifold")

    direction = torch.where(
        directed_edges[:, 0] < directed_edges[:, 1],
        torch.ones_like(directed_edges[:, 0]),
        -torch.ones_like(directed_edges[:, 0]),
    )
    orientation_balance = torch.zeros(
        len(counts),
        dtype=direction.dtype,
        device=direction.device,
    )
    orientation_balance.index_add_(0, inverse, direction)
    if not bool((orientation_balance == 0).all()):
        raise RuntimeError(f"{name} has inconsistent face orientation")

    volume = _signed_enclosed_volume(mesh)
    if not bool(volume > 0.0):
        raise RuntimeError(f"{name} must have positive enclosed volume")
    return float(volume)


def _build_constraint_plane(device: torch.device) -> Mesh:
    """Build a triangulated horizontal packaging clearance plane."""

    x = torch.linspace(
        -0.5 * BODY_LENGTH - 0.15,
        0.5 * BODY_LENGTH + 0.15,
        PLANE_X_POINTS,
        device=device,
    )
    y = torch.linspace(
        -0.5 * BODY_WIDTH - 0.15,
        0.5 * BODY_WIDTH + 0.15,
        PLANE_Y_POINTS,
        device=device,
    )
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack(
        (
            x_grid,
            y_grid,
            torch.full_like(x_grid, CLEARANCE_Z),
        ),
        dim=-1,
    ).reshape(-1, 3)
    point_ids = torch.arange(len(points), device=device).reshape(
        PLANE_X_POINTS,
        PLANE_Y_POINTS,
    )
    target = Mesh(points=points, cells=_triangle_grid_faces(point_ids, upward=True))
    _validate_triangle_geometry(target, "target plane")
    return target


def build_case(device: torch.device | str = "cpu") -> SurfaceCase:
    """Build a closed housing with safe and violating lid deformations."""

    device = torch.device(device)
    x = torch.linspace(
        -0.5 * LID_LENGTH,
        0.5 * LID_LENGTH,
        TOP_X_POINTS,
        device=device,
    )
    y = torch.linspace(
        -0.5 * LID_WIDTH,
        0.5 * LID_WIDTH,
        TOP_Y_POINTS,
        device=device,
    )
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    u = 2.0 * x_grid / LID_LENGTH
    v = 2.0 * y_grid / LID_WIDTH
    dome_envelope = (
        torch.cos(0.5 * torch.pi * u).square() * torch.cos(0.5 * torch.pi * v).square()
    )

    top_z = BASE_TOP_Z + DOME_HEIGHT * dome_envelope
    optimized_x = x_grid + 0.022 * dome_envelope
    optimized_y = y_grid - 0.016 * dome_envelope
    top_points = torch.stack((optimized_x, optimized_y, top_z), dim=-1).reshape(-1, 3)
    bottom_x = torch.linspace(
        -0.5 * BODY_LENGTH,
        0.5 * BODY_LENGTH,
        TOP_X_POINTS,
        device=device,
    )
    bottom_y = torch.linspace(
        -0.5 * BODY_WIDTH,
        0.5 * BODY_WIDTH,
        TOP_Y_POINTS,
        device=device,
    )
    bottom_x_grid, bottom_y_grid = torch.meshgrid(
        bottom_x,
        bottom_y,
        indexing="ij",
    )
    bottom_points = torch.stack(
        (
            bottom_x_grid,
            bottom_y_grid,
            torch.full_like(bottom_x_grid, BOTTOM_Z),
        ),
        dim=-1,
    ).reshape(-1, 3)

    shoulder_points = _rectangular_ring_points(
        length=BODY_LENGTH,
        width=BODY_WIDTH,
        height=SHOULDER_Z,
        device=device,
    )
    points = torch.cat((top_points, bottom_points, shoulder_points), dim=0)

    top_ids = torch.arange(len(top_points), device=device).reshape(
        TOP_X_POINTS,
        TOP_Y_POINTS,
    )
    bottom_ids = top_ids + len(top_points)
    point_offset = len(top_points) + len(bottom_points)
    shoulder_ring = torch.arange(
        point_offset,
        point_offset + len(shoulder_points),
        device=device,
    )

    top_ring = _grid_perimeter(top_ids)
    bottom_ring = _grid_perimeter(bottom_ids)
    face_blocks = [
        _triangle_grid_faces(top_ids, upward=True),
        _triangle_grid_faces(bottom_ids, upward=False),
        _connect_rings(shoulder_ring, top_ring),
        _connect_rings(bottom_ring, shoulder_ring),
    ]
    source = Mesh(points=points, cells=torch.cat(face_blocks, dim=0))
    _validate_closed_oriented(source, "optimized enclosure")

    optimized_top = DOME_HEIGHT * dome_envelope > 1e-6
    optimized = torch.zeros(len(source.points), dtype=torch.bool, device=device)
    optimized[: len(top_points)] = optimized_top.reshape(-1)
    movable = source.points[:, 2] > CLEARANCE_Z
    if not bool(movable.any()):
        raise RuntimeError("Candidate construction did not create a violation")
    if not bool((optimized & ~movable).any()):
        raise RuntimeError("Candidate construction did not retain a valid deformation")
    if bool((source.points[optimized & ~movable, 2] > CLEARANCE_Z).any()):
        raise RuntimeError("Selection missed a violating optimized vertex")

    return SurfaceCase(
        target=_build_constraint_plane(device),
        source=source,
        optimized=optimized,
        movable=movable,
    )


def shrinkwrap_case(
    case: SurfaceCase,
    *,
    implementation: Literal["torch", "warp"],
) -> Mesh:
    """Project only vertices that penetrate the packaging clearance plane."""

    return case.source.shrinkwrap(
        case.target,
        point_weights=case.movable,
        implementation=implementation,
    )


def validate_result(
    case: SurfaceCase,
    result: Mesh,
    *,
    implementation: Literal["torch", "warp"],
) -> SurfaceMetrics:
    """Validate the repair and exact preservation of admissible geometry."""

    source_volume = _validate_closed_oriented(case.source, "source")
    result_volume = _validate_closed_oriented(result, "result")

    fixed_delta = result.points[~case.movable] - case.source.points[~case.movable]
    fixed_error = float(torch.linalg.vector_norm(fixed_delta, dim=1).max())
    if fixed_error != 0.0:
        raise RuntimeError(f"Unselected vertices moved by {fixed_error:.3e}")

    retained = case.optimized & ~case.movable
    retained_delta = result.points[retained] - case.source.points[retained]
    retained_error = float(torch.linalg.vector_norm(retained_delta, dim=1).max())
    if retained_error != 0.0:
        raise RuntimeError(
            f"Admissible optimized geometry moved by {retained_error:.3e}"
        )

    projected_again = result.shrinkwrap(
        case.target,
        point_weights=case.movable,
        implementation=implementation,
    )
    surface_residual = float(
        torch.linalg.vector_norm(
            projected_again.points[case.movable] - result.points[case.movable],
            dim=1,
        ).max()
    )
    if surface_residual > 2.0e-5:
        raise RuntimeError(f"Repaired surface residual is {surface_residual:.3e}")

    plane_error = float((result.points[case.movable, 2] - CLEARANCE_Z).abs().max())
    if plane_error > 2.0e-5:
        raise RuntimeError(f"Repaired cap misses the plane by {plane_error:.3e}")

    correction = torch.linalg.vector_norm(
        result.points[case.movable] - case.source.points[case.movable],
        dim=1,
    )
    max_correction = float(correction.max())
    if max_correction < 0.25:
        raise RuntimeError("Clearance violation is not visually significant")

    retained_deformation = float((case.source.points[retained, 2] - BASE_TOP_Z).max())
    if retained_deformation < 0.25 * DOME_HEIGHT:
        raise RuntimeError("Admissible optimization deformation is not visible")

    return SurfaceMetrics(
        source_volume=source_volume,
        result_volume=result_volume,
        fixed_error=fixed_error,
        retained_error=retained_error,
        surface_residual=surface_residual,
        plane_error=plane_error,
        retained_deformation=retained_deformation,
        max_correction=max_correction,
    )


def _move_case(case: SurfaceCase, device: torch.device | str) -> SurfaceCase:
    """Move both meshes and point selections to one device."""

    return SurfaceCase(
        target=case.target.to(device),
        source=case.source.to(device),
        optimized=case.optimized.to(device),
        movable=case.movable.to(device),
    )


def _adjoint_probe(
    case: SurfaceCase,
    *,
    implementation: Literal["torch", "warp"],
) -> AdjointProbe:
    """Differentiate one scalar probe through the selected repair."""

    source_points = case.source.points.detach().clone().requires_grad_()
    target_points = case.target.points.detach().clone().requires_grad_()
    source = Mesh(points=source_points, cells=case.source.cells)
    target = Mesh(points=target_points, cells=case.target.cells)
    result = source.shrinkwrap(
        target,
        point_weights=case.movable,
        implementation=implementation,
    )

    selected = result.points[case.movable]
    coefficients = selected.new_tensor((0.37, -0.23, 0.61))
    row_scale = torch.linspace(
        0.8,
        1.2,
        len(selected),
        device=selected.device,
        dtype=selected.dtype,
    )
    loss = (selected * coefficients * row_scale[:, None]).sum()
    source_gradient, target_gradient = torch.autograd.grad(
        loss,
        (source_points, target_points),
    )
    for name, gradient in (
        ("source", source_gradient),
        ("target", target_gradient),
    ):
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"{name} adjoint contains a non-finite value")
        if not bool(gradient.abs().sum() > 0.0):
            raise RuntimeError(f"{name} adjoint is identically zero")

    fixed_gradient = source_gradient[~case.movable].abs().max()
    if float(fixed_gradient) != 0.0:
        raise RuntimeError(
            f"Unselected source adjoint is nonzero: {float(fixed_gradient):.3e}"
        )
    return AdjointProbe(
        source=source_gradient.detach().cpu(),
        target=target_gradient.detach().cpu(),
    )


def run_backends(case: SurfaceCase) -> tuple[Mesh, SurfaceMetrics]:
    """Validate geometry and adjoints for Torch and Warp."""

    torch_result = shrinkwrap_case(case, implementation="torch")
    torch_metrics = validate_result(
        case,
        torch_result,
        implementation="torch",
    )
    torch_probe = _adjoint_probe(case, implementation="torch")
    print(
        "Torch CPU: "
        f"fixed={torch_metrics.fixed_error:.1e}, "
        f"retained={torch_metrics.retained_error:.1e}, "
        f"plane={torch_metrics.plane_error:.1e}, "
        f"residual={torch_metrics.surface_residual:.1e}, "
        f"valid deformation={torch_metrics.retained_deformation:.3f}, "
        f"max correction={torch_metrics.max_correction:.3f}, "
        f"adjoint L1={torch_probe.source.abs().sum():.3f}"
    )

    if torch.cuda.is_available():
        cuda_case = _move_case(case, "cuda")
        warp_result = shrinkwrap_case(cuda_case, implementation="warp")
        warp_metrics = validate_result(
            cuda_case,
            warp_result,
            implementation="warp",
        )
        backend_delta = float(
            torch.linalg.vector_norm(
                warp_result.points.cpu() - torch_result.points,
                dim=1,
            ).max()
        )
        if backend_delta > 2.0e-3:
            raise RuntimeError(f"Warp and Torch differ by up to {backend_delta:.3e}")

        warp_probe = _adjoint_probe(cuda_case, implementation="warp")
        source_gradient_delta = float(
            (warp_probe.source - torch_probe.source).abs().max()
        )
        target_gradient_delta = float(
            (warp_probe.target - torch_probe.target).abs().max()
        )
        adjoint_delta = max(source_gradient_delta, target_gradient_delta)
        if adjoint_delta > 2.0e-3:
            raise RuntimeError(f"Warp and Torch adjoints differ by {adjoint_delta:.3e}")
        print(
            "Warp CUDA: "
            f"fixed={warp_metrics.fixed_error:.1e}, "
            f"retained={warp_metrics.retained_error:.1e}, "
            f"plane={warp_metrics.plane_error:.1e}, "
            f"residual={warp_metrics.surface_residual:.1e}, "
            f"point delta={backend_delta:.1e}, "
            f"adjoint delta={adjoint_delta:.1e}"
        )
    else:
        print("Warp CUDA: skipped because CUDA is unavailable")

    return torch_result, torch_metrics


def _split_visual_regions(
    mesh: Mesh,
    optimized: torch.Tensor,
    movable: torch.Tensor,
) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid, pv.UnstructuredGrid]:
    """Split surface cells into shell, retained design, and repaired cap."""

    pyvista_mesh = to_pyvista(mesh)
    active_cells = movable[mesh.cells].any(dim=1).cpu().numpy()
    optimized_cells = optimized[mesh.cells].any(dim=1).cpu().numpy() & ~active_cells
    shell_cells = ~(active_cells | optimized_cells)
    return (
        pyvista_mesh.extract_cells(shell_cells),
        pyvista_mesh.extract_cells(optimized_cells),
        pyvista_mesh.extract_cells(active_cells),
    )


def _camera(
    source: Mesh,
    target: Mesh,
) -> tuple[list[list[float] | tuple[float, float, float]], float]:
    """Fit a low three-quarter camera that exposes plane penetration."""

    points = torch.cat((source.points, target.points), dim=0)
    center = 0.5 * (points.amin(dim=0) + points.amax(dim=0))
    eye_offset = points.new_tensor((5.5, -7.0, 3.5))
    view_direction = F.normalize(-eye_offset, dim=0)
    world_up = points.new_tensor((0.0, 0.0, 1.0))
    screen_right = F.normalize(
        torch.linalg.cross(view_direction, world_up),
        dim=0,
    )
    screen_up = F.normalize(
        torch.linalg.cross(screen_right, view_direction),
        dim=0,
    )
    centered = points - center
    half_width = float((centered @ screen_right).abs().max())
    half_height = float((centered @ screen_up).abs().max())
    viewport_aspect = 800.0 / 640.0
    parallel_scale = 1.07 * max(
        half_height,
        half_width / viewport_aspect,
    )
    return (
        [
            (center + eye_offset).tolist(),
            center.tolist(),
            (0.0, 0.0, 1.0),
        ],
        parallel_scale,
    )


def _add_surface(
    plotter: pv.Plotter,
    mesh: pv.UnstructuredGrid,
    color: str,
    *,
    edges: bool,
) -> None:
    """Add one shaded triangular surface region."""

    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        split_sharp_edges=True,
        feature_angle=28.0,
        show_edges=edges,
        edge_color=EDGE_COLOR,
        line_width=0.42,
        ambient=0.28,
        diffuse=0.72,
    )


def _add_constraint_plane(plotter: pv.Plotter, plane: pv.UnstructuredGrid) -> None:
    """Draw the packaging plane as a translucent blue triangle grid."""

    plotter.add_mesh(
        plane,
        color=PLANE_COLOR,
        opacity=0.10,
        smooth_shading=False,
        show_edges=False,
    )
    plotter.add_mesh(
        plane,
        color=PLANE_COLOR,
        opacity=0.70,
        style="wireframe",
        line_width=0.80,
        render_lines_as_tubes=True,
    )


def render_case(
    case: SurfaceCase,
    result: Mesh,
    metrics: SurfaceMetrics,
    output: Path,
) -> None:
    """Render the broad lid dome before and after selected cap repair."""

    target = case.target.to("cpu")
    source = case.source.to("cpu")
    result = result.to("cpu")
    optimized = case.optimized.cpu()
    movable = case.movable.cpu()

    source_shell, source_valid, source_cap = _split_visual_regions(
        source,
        optimized,
        movable,
    )
    result_shell, result_valid, result_cap = _split_visual_regions(
        result,
        optimized,
        movable,
    )
    plane = to_pyvista(target)

    plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 640))
    plotter.enable_anti_aliasing("ssaa")

    plotter.subplot(0, 0)
    _add_surface(plotter, source_shell, SHELL_COLOR, edges=False)
    _add_surface(plotter, source_valid, OPTIMIZED_COLOR, edges=True)
    _add_surface(plotter, source_cap, VIOLATION_COLOR, edges=True)
    _add_constraint_plane(plotter, plane)
    plotter.add_text(
        "BEFORE: OPTIMIZED ENCLOSURE\n"
        "gold and orange form one broad lid dome\n"
        f"orange upper dome crosses plane by {metrics.max_correction:.2f}",
        position="upper_left",
        font_size=11,
        color=TEXT_COLOR,
    )

    plotter.subplot(0, 1)
    _add_surface(plotter, result_shell, SHELL_COLOR, edges=False)
    _add_surface(plotter, result_valid, OPTIMIZED_COLOR, edges=True)
    _add_surface(plotter, result_cap, REPAIRED_COLOR, edges=True)
    _add_constraint_plane(plotter, plane)
    plotter.add_mesh(
        source_cap,
        color=VIOLATION_COLOR,
        opacity=0.46,
        style="wireframe",
        line_width=0.85,
        render_lines_as_tubes=True,
    )
    plotter.add_text(
        "AFTER: SELECTED CAP REPAIR\n"
        "gold lower dome is retained exactly\n"
        "green cap meets plane, orange wireframe shows prior dome",
        position="upper_left",
        font_size=11,
        color=TEXT_COLOR,
    )

    plotter.subplot(0, 2)
    _add_surface(
        plotter,
        to_pyvista(result),
        SHELL_COLOR,
        edges=True,
    )
    plotter.add_text(
        "FINAL TRIANGULATED MESH\nfeasible geometry with original connectivity",
        position="upper_left",
        font_size=11,
        color=TEXT_COLOR,
    )

    camera, parallel_scale = _camera(source, target)
    for column in range(3):
        plotter.subplot(0, column)
        plotter.camera_position = camera
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        plotter.hide_axes()
    plotter.set_background("#F8FAFC", all_renderers=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.render()
    plotter.screenshot(output, transparent_background=False)
    plotter.close()


def main(*, output: Path) -> None:
    """Run backend validation and save the comparison figure."""

    case = build_case()
    result, metrics = run_backends(case)
    render_case(case, result, metrics, output)
    valid_count = int((case.optimized & ~case.movable).sum())
    print(
        f"Solid: {case.source.n_points:,} vertices, "
        f"{case.source.n_cells:,} triangles, "
        f"{valid_count:,} retained optimized vertices, "
        f"{int(case.movable.sum()):,} repaired vertices, "
        f"volume={metrics.result_volume:.4f}"
    )
    print(f"Saved {output}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="PNG path (default: beside this script)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(output=args.output)
