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

"""Render shrinkwrap conformance for a curved engineering panel.

This documentation figure is self-contained. It builds both meshes directly
with PyTorch, calls the real :meth:`physicsnemo.mesh.Mesh.shrinkwrap` API, and
renders all three panels with the shared figure style.

The Torch backend is validated on CPU. When CUDA is available, the Warp result
is also checked against Torch.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torch.nn.functional as F
from _style import (
    ACCENT_LIGHTING,
    FIXED,
    FOURTH,
    SEQUENTIAL,
    SURFACE,
    SURFACE_EDGE,
    mesh_label,
    save,
    spheres,
)
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).with_suffix(".png")
MAX_DISTANCE = 0.34

TARGET_COLOR = SURFACE
SOURCE_COLOR = FOURTH
EDGE_COLOR = SURFACE_EDGE
FIXED_COLOR = FIXED


def structured_cells(
    n_span: int,
    n_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Triangulate a span-major parameter grid with upward cells."""

    lower_left = (
        torch.arange(n_span - 1, device=device)[:, None] * n_width
        + torch.arange(n_width - 1, device=device)[None, :]
    ).reshape(-1)
    lower_right = lower_left + 1
    upper_left = lower_left + n_width
    upper_right = upper_left + 1
    return torch.cat(
        (
            torch.stack((lower_left, lower_right, upper_left), dim=1),
            torch.stack((lower_right, upper_right, upper_left), dim=1),
        ),
        dim=0,
    )


def panel_surface(
    width_fraction: torch.Tensor,
    span_fraction: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a swept, tapered, compound-curved panel in meters."""

    half_width = 0.78 - 0.20 * span_fraction
    center_sweep = 0.26 * span_fraction + 0.08 * span_fraction.square()
    x = center_sweep + half_width * width_fraction
    y = 2.70 * span_fraction

    crown = 0.205 * (1.0 - width_fraction.square()) * (1.0 - 0.14 * span_fraction)
    longitudinal_bow = 0.17 * span_fraction + 0.038 * torch.sin(
        torch.pi * span_fraction
    )
    twist = -0.050 * span_fraction * width_fraction
    z = crown + longitudinal_bow + twist
    return torch.stack((x, y, z), dim=-1)


def panel_normals(
    width_fraction: torch.Tensor,
    span_fraction: torch.Tensor,
) -> torch.Tensor:
    """Evaluate smooth upward normals with centered differences."""

    epsilon = 1.0e-3
    width_tangent = (
        panel_surface(width_fraction + epsilon, span_fraction)
        - panel_surface(width_fraction - epsilon, span_fraction)
    ) / (2.0 * epsilon)
    span_tangent = (
        panel_surface(width_fraction, span_fraction + epsilon)
        - panel_surface(width_fraction, span_fraction - epsilon)
    ) / (2.0 * epsilon)
    return F.normalize(
        torch.linalg.cross(width_tangent, span_tangent, dim=-1),
        dim=-1,
    )


def parameter_grid(
    n_span: int,
    n_width: int,
    *,
    width_limit: float,
    span_limit: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return span-major parameter coordinates for one panel sheet."""

    span = torch.linspace(0.0, span_limit, n_span, device=device)
    width = torch.linspace(-width_limit, width_limit, n_width, device=device)
    return torch.meshgrid(span, width, indexing="ij")


def build_case(
    device: torch.device | str = "cpu",
) -> tuple[Mesh, Mesh, torch.Tensor]:
    """Build a dense target and an incommensurate lifted source sheet."""

    device = torch.device(device)
    target_span, target_width = parameter_grid(
        44,
        31,
        width_limit=1.0,
        span_limit=1.0,
        device=device,
    )
    target = Mesh(
        points=panel_surface(target_width, target_span).reshape(-1, 3),
        cells=structured_cells(44, 31, device=device),
    )

    source_span, source_width = parameter_grid(
        19,
        25,
        width_limit=0.96,
        span_limit=0.985,
        device=device,
    )
    nominal = panel_surface(source_width, source_span)
    normals = panel_normals(source_width, source_span)

    movable = source_span > 0.10
    transition = ((source_span - 0.10) / 0.18).clamp(0.0, 1.0)
    transition = transition.square() * (3.0 - 2.0 * transition)
    springback = 0.105 + 0.040 * torch.sin(
        3.2 * torch.pi * source_width + 1.55 * torch.pi * source_span
    )
    local_bulge = 0.085 * torch.exp(
        -((source_width - 0.18) / 0.34).square()
        - ((source_span - 0.58) / 0.20).square()
    )
    lift = transition * (springback + local_bulge)

    source = Mesh(
        points=(nominal + lift.unsqueeze(-1) * normals).reshape(-1, 3),
        cells=structured_cells(19, 25, device=device),
    )
    return target, source, movable.reshape(-1)


def shrinkwrap_case(
    target: Mesh,
    source: Mesh,
    movable: torch.Tensor,
    *,
    implementation: str,
    max_distance: float | None = MAX_DISTANCE,
) -> Mesh:
    """Project the sheet while preserving its root attachment strip."""

    return source.shrinkwrap(
        target,
        point_weights=movable,
        max_distance=max_distance,
        implementation=implementation,
    )


def surface_residual(
    target: Mesh,
    result: Mesh,
    movable: torch.Tensor,
    *,
    implementation: str,
) -> float:
    """Return the largest unbounded re-projection motion."""

    projected_again = shrinkwrap_case(
        target,
        result,
        movable,
        implementation=implementation,
        max_distance=None,
    )
    residual = torch.linalg.vector_norm(
        projected_again.points - result.points,
        dim=1,
    )
    return float(residual[movable].max())


def run_backends(
    target: Mesh,
    source: Mesh,
    movable: torch.Tensor,
) -> Mesh:
    """Validate Torch on CPU and Warp on CUDA when CUDA is available."""

    result = shrinkwrap_case(
        target,
        source,
        movable,
        implementation="torch",
    )
    correction = torch.linalg.vector_norm(result.points - source.points, dim=1)
    torch.testing.assert_close(
        correction[~movable],
        torch.zeros_like(correction[~movable]),
        atol=0.0,
        rtol=0.0,
    )
    if not torch.isfinite(result.points).all():
        raise RuntimeError("Torch shrinkwrap produced non-finite coordinates")
    if float(correction[movable].mean()) < 0.05:
        raise RuntimeError("The springback correction is not visually meaningful")

    torch_residual = surface_residual(
        target,
        result,
        movable,
        implementation="torch",
    )
    if torch_residual > 1.0e-5:
        raise RuntimeError(
            f"Torch vertices miss the target by up to {torch_residual:.3e}"
        )

    print(
        "Torch CPU: "
        f"mean correction={1000.0 * float(correction[movable].mean()):.1f} mm, "
        f"max correction={1000.0 * float(correction.max()):.1f} mm, "
        f"surface residual={torch_residual:.1e} m"
    )

    if torch.cuda.is_available():
        target_cuda = target.to("cuda")
        source_cuda = source.to("cuda")
        movable_cuda = movable.to("cuda")
        warp_result = shrinkwrap_case(
            target_cuda,
            source_cuda,
            movable_cuda,
            implementation="warp",
        )
        warp_residual = surface_residual(
            target_cuda,
            warp_result,
            movable_cuda,
            implementation="warp",
        )
        if warp_residual > 1.0e-5:
            raise RuntimeError(
                f"Warp vertices miss the target by up to {warp_residual:.3e}"
            )

        warp_result = warp_result.to("cpu")
        backend_delta = torch.linalg.vector_norm(
            warp_result.points - result.points,
            dim=1,
        )
        max_backend_delta = float(backend_delta[movable].max())
        if max_backend_delta > 2.0e-4:
            raise RuntimeError(
                f"Warp and Torch panel results differ by {max_backend_delta:.3e} m"
            )
        print(
            "Warp CUDA: "
            f"surface residual={warp_residual:.1e} m, "
            f"backend delta={max_backend_delta:.1e} m"
        )

    return result


def camera_position(mesh: Mesh) -> list[list[float]]:
    """Return one rolled three-quarter camera shared by all panels."""

    points = mesh.points.detach().cpu()
    center = 0.5 * (points.amin(dim=0) + points.amax(dim=0))
    diagonal = float((points.amax(dim=0) - points.amin(dim=0)).norm())
    eye = center + points.new_tensor(
        (0.90 * diagonal, -0.95 * diagonal, 0.60 * diagonal)
    )
    view = F.normalize(center - eye, dim=0)
    world_up = points.new_tensor((0.0, 0.0, 1.0))
    screen_right = F.normalize(torch.linalg.cross(view, world_up), dim=0)
    screen_up = F.normalize(
        world_up - torch.dot(world_up, view) * view,
        dim=0,
    )
    roll = math.radians(22.0)
    rolled_up = math.cos(roll) * screen_up + math.sin(roll) * screen_right
    rolled_right = math.cos(roll) * screen_right - math.sin(roll) * screen_up
    pan = -0.035 * diagonal * rolled_right
    return [
        (eye + pan).tolist(),
        (center + pan).tolist(),
        rolled_up.tolist(),
    ]


def render_panel(
    mesh: Mesh,
    camera: list[list[float]],
    movable: torch.Tensor | None = None,
    **kwargs,
):
    """Render one panel from the shared camera and return the image array."""

    plotter = pv.Plotter(off_screen=True, window_size=(1000, 900))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    plotter.add_mesh(
        to_pyvista(mesh) if "scalars" not in kwargs else kwargs.pop("dataset"),
        smooth_shading=True,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=0.6,
        ambient=0.35,
        diffuse=0.7,
        show_scalar_bar=False,
        **kwargs,
    )
    if movable is not None:
        # Mark the root vertices held by the point-weight mask.
        root = mesh.points[~movable].detach().cpu().numpy()
        plotter.add_mesh(spheres(root, 0.012), color=FIXED_COLOR, **ACCENT_LIGHTING)
    plotter.camera_position = camera
    plotter.camera.zoom(0.95)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def render(
    target: Mesh,
    source: Mesh,
    result: Mesh,
    movable: torch.Tensor,
) -> None:
    """Render target, lifted source, and fitted result panels."""

    correction_mm = (
        1000.0 * torch.linalg.vector_norm(result.points - source.points, dim=1)
    ).numpy()
    result_pv = to_pyvista(result)
    result_pv.point_data["correction"] = correction_mm
    upper = max(1.0, float(correction_mm.max()))
    camera = camera_position(target)

    panels = (
        (
            f"target panel\n{mesh_label(target)}",
            render_panel(target, camera, color=TARGET_COLOR),
        ),
        (
            f"lifted source sheet\n{mesh_label(source)}, root fixed",
            render_panel(source, camera, movable, color=SOURCE_COLOR),
        ),
        (
            "shrinkwrap(point_weights=movable)\nmovable vertices on the target",
            render_panel(
                result,
                camera,
                movable,
                dataset=result_pv,
                scalars="correction",
                cmap=SEQUENTIAL,
                clim=(0.0, upper),
            ),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
    for axis, (title, image) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.set_axis_off()
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.0, upper), cmap=SEQUENTIAL),
        cax=axes[2].inset_axes((0.15, -0.06, 0.7, 0.045)),
        orientation="horizontal",
    )
    colorbar.set_label("vertex correction (mm)")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=FIXED_COLOR,
                markeredgecolor="white",
                markersize=12,
                label="fixed root vertices",
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.36, 0.0),
        frameon=False,
    )
    save(figure, OUTPUT, tight=True, full_color=True)


def main() -> None:
    """Generate the self-contained documentation figure."""

    target, source, movable = build_case()
    result = run_backends(target, source, movable)
    render(target, source, result, movable)


if __name__ == "__main__":
    main()
