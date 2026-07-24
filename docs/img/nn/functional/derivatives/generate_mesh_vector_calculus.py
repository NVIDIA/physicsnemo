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

"""Generate deterministic documentation figures for mesh calculus functionals.

Run this script from the repository root with PhysicsNeMo and Pillow available::

    python docs/img/nn/functional/derivatives/generate_mesh_vector_calculus.py

Each figure evaluates the public functional with the PyTorch implementation on
the same seeded, irregular triangular mesh.  Pillow is used instead of a GUI
plotting backend so the script also works in headless documentation builders.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import (
    compute_cotan_weights_fem,
    compute_dual_volumes_0,
)
from physicsnemo.nn.functional import (
    mesh_cotan_divergence,
    mesh_cotan_laplacian,
    mesh_lsq_curl,
    mesh_lsq_divergence,
    mesh_lsq_laplacian,
)

OUTPUT_DIR = Path(__file__).parent
IMAGE_SIZE = (1980, 774)

# The panel geometry matches the other derivative-functional figures.
PANEL_TOP = 76
PANEL_SIZE = 620
LEFT_PANEL = (98, PANEL_TOP, 98 + PANEL_SIZE, PANEL_TOP + PANEL_SIZE)
RIGHT_PANEL = (1108, PANEL_TOP, 1108 + PANEL_SIZE, PANEL_TOP + PANEL_SIZE)
LEFT_COLORBAR = (752, PANEL_TOP, 782, PANEL_TOP + PANEL_SIZE)
RIGHT_COLORBAR = (1762, PANEL_TOP, 1792, PANEL_TOP + PANEL_SIZE)

# Anchor colors sampled from Matplotlib's coolwarm and viridis maps.  Linear
# interpolation between the anchors is sufficient for these documentation
# plots and keeps the generator free of a Matplotlib runtime dependency.
COOLWARM = (
    (0.00, (59, 76, 192)),
    (0.25, (141, 176, 254)),
    (0.50, (221, 221, 221)),
    (0.75, (244, 152, 122)),
    (1.00, (180, 4, 38)),
)
VIRIDIS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a common sans-serif font on macOS/Linux, with a Pillow fallback."""
    names = (
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


TITLE_FONT = _font(35)
AXIS_FONT = _font(24)
TICK_FONT = _font(19)
COLORBAR_FONT = _font(18)


def _make_mesh(n: int = 29) -> Mesh:
    """Create a seeded irregular triangular mesh of the unit square."""
    axis = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1735)
    jitter = (torch.rand(points.shape, generator=generator) - 0.5) * (0.32 / (n - 1))
    boundary = (
        (points[:, 0] == 0.0)
        | (points[:, 0] == 1.0)
        | (points[:, 1] == 0.0)
        | (points[:, 1] == 1.0)
    )
    jitter[boundary] = 0.0
    points = points + jitter

    cells: list[tuple[int, int, int]] = []
    for row in range(n - 1):
        for column in range(n - 1):
            lower_left = row * n + column
            lower_right = lower_left + 1
            upper_left = lower_left + n
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                cells.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )
            else:
                cells.extend(
                    (
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    )
                )

    return Mesh(points=points, cells=torch.tensor(cells, dtype=torch.int64))


def _sample_fields(points: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return smooth fields chosen for clean open-mesh operator responses."""
    x = points[:, 0]
    y = points[:, 1]

    # Zero normal derivative at the square boundary keeps the open-mesh
    # Laplacians focused on the operator response instead of boundary flux.
    scalar = torch.cos(math.pi * x) * torch.cos(math.pi * y)
    divergence_field = torch.stack(
        (
            torch.sin(2.0 * math.pi * x) * torch.sin(math.pi * y),
            torch.sin(math.pi * x) * torch.sin(2.0 * math.pi * y),
        ),
        dim=-1,
    )
    radius_squared = (x - 0.5).square() + (y - 0.5).square()
    vortex_envelope = torch.exp(-12.0 * radius_squared)
    curl_field = torch.stack(
        (-(y - 0.5) * vortex_envelope, (x - 0.5) * vortex_envelope), dim=-1
    )
    return scalar, divergence_field, curl_field


def _interpolate_color(
    value: float, anchors: tuple[tuple[float, tuple[int, int, int]], ...]
) -> tuple[int, int, int]:
    value = min(max(value, 0.0), 1.0)
    for (left_x, left_color), (right_x, right_color) in zip(
        anchors[:-1], anchors[1:], strict=True
    ):
        if value <= right_x:
            fraction = (value - left_x) / (right_x - left_x)
            return tuple(
                round(left + fraction * (right - left))
                for left, right in zip(left_color, right_color, strict=True)
            )
    return anchors[-1][1]


def _color(
    value: float,
    value_min: float,
    value_max: float,
    anchors: tuple[tuple[float, tuple[int, int, int]], ...],
) -> tuple[int, int, int]:
    if value_max <= value_min:
        return anchors[len(anchors) // 2][1]
    return _interpolate_color((value - value_min) / (value_max - value_min), anchors)


def _panel_point(
    point: torch.Tensor, panel: tuple[int, int, int, int]
) -> tuple[int, int]:
    left, top, right, bottom = panel
    margin = 14
    x = left + margin + float(point[0]) * (right - left - 2 * margin)
    y = bottom - margin - float(point[1]) * (bottom - top - 2 * margin)
    return round(x), round(y)


def _draw_title(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
) -> None:
    left, _, right, _ = panel
    bounds = draw.textbbox((0, 0), title, font=TITLE_FONT)
    width = bounds[2] - bounds[0]
    draw.text(((left + right - width) / 2, 20), title, fill="black", font=TITLE_FONT)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = panel
    draw.rectangle(panel, outline=(25, 25, 25), width=2)

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * (right - left)
        y = bottom - tick * (bottom - top)
        label = f"{tick:.2g}"

        draw.line((x, bottom, x, bottom + 8), fill="black", width=2)
        label_box = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, bottom + 11),
            label,
            fill="black",
            font=TICK_FONT,
        )

        draw.line((left - 8, y, left, y), fill="black", width=2)
        label_box = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text(
            (left - 14 - (label_box[2] - label_box[0]), y - 10),
            label,
            fill="black",
            font=TICK_FONT,
        )

    x_label_box = draw.textbbox((0, 0), "x", font=AXIS_FONT)
    draw.text(
        ((left + right - (x_label_box[2] - x_label_box[0])) / 2, bottom + 42),
        "x",
        fill="black",
        font=AXIS_FONT,
    )
    draw.text((left - 64, (top + bottom) / 2 - 15), "y", fill="black", font=AXIS_FONT)


def _draw_colorbar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value_min: float,
    value_max: float,
    anchors: tuple[tuple[float, tuple[int, int, int]], ...],
) -> None:
    left, top, right, bottom = box
    for y in range(top, bottom):
        fraction = 1.0 - (y - top) / max(bottom - top - 1, 1)
        draw.line((left, y, right, y), fill=_interpolate_color(fraction, anchors))
    draw.rectangle(box, outline="black", width=2)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - fraction * (bottom - top)
        value = value_min + fraction * (value_max - value_min)
        draw.line((right, y, right + 8, y), fill="black", width=2)
        draw.text(
            (right + 13, y - 10),
            f"{value:.2g}",
            fill="black",
            font=COLORBAR_FONT,
        )


def _robust_range(values: torch.Tensor, *, symmetric: bool) -> tuple[float, float]:
    finite = values[torch.isfinite(values)].detach().cpu()
    if symmetric:
        limit = float(torch.quantile(finite.abs(), 0.98))
        limit = max(limit, 1.0e-12)
        return -limit, limit
    lower = float(torch.quantile(finite, 0.02))
    upper = float(torch.quantile(finite, 0.98))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _draw_scalar_mesh(
    draw: ImageDraw.ImageDraw,
    mesh: Mesh,
    values: torch.Tensor,
    panel: tuple[int, int, int, int],
    value_range: tuple[float, float],
) -> None:
    value_min, value_max = value_range
    points = [_panel_point(point, panel) for point in mesh.points]
    for cell in mesh.cells.tolist():
        polygon = [points[index] for index in cell]
        mean_value = float(values[cell].mean())
        draw.polygon(
            polygon,
            fill=_color(mean_value, value_min, value_max, COOLWARM),
        )
    for cell in mesh.cells.tolist():
        polygon = [points[index] for index in cell]
        draw.line((*polygon, polygon[0]), fill=(255, 255, 255), width=1)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    vector: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    end = (start[0] + vector[0], start[1] + vector[1])
    draw.line((*start, *end), fill=color, width=3)

    length = math.hypot(*vector)
    if length < 1.0e-12:
        return
    unit_x = vector[0] / length
    unit_y = vector[1] / length
    normal_x = -unit_y
    normal_y = unit_x
    head_length = 8.0
    head_width = 4.5
    base = (end[0] - head_length * unit_x, end[1] - head_length * unit_y)
    draw.polygon(
        (
            end,
            (base[0] + head_width * normal_x, base[1] + head_width * normal_y),
            (base[0] - head_width * normal_x, base[1] - head_width * normal_y),
        ),
        fill=color,
    )


def _draw_vector_mesh(
    draw: ImageDraw.ImageDraw,
    mesh: Mesh,
    vector_field: torch.Tensor,
    panel: tuple[int, int, int, int],
    value_range: tuple[float, float],
) -> None:
    points = [_panel_point(point, panel) for point in mesh.points]
    unique_edges = torch.unique(
        torch.sort(
            torch.cat(
                (
                    mesh.cells[:, (0, 1)],
                    mesh.cells[:, (1, 2)],
                    mesh.cells[:, (2, 0)],
                ),
                dim=0,
            ),
            dim=1,
        ).values,
        dim=0,
    )
    for start_index, end_index in unique_edges.tolist():
        draw.line(
            (*points[start_index], *points[end_index]),
            fill=(222, 225, 230),
            width=1,
        )

    magnitudes = vector_field.norm(dim=-1)
    max_magnitude = max(float(magnitudes.max()), 1.0e-12)
    scale = 23.0 / max_magnitude
    n = round(math.sqrt(mesh.n_points))
    for row in range(1, n - 1, 2):
        for column in range(1, n - 1, 2):
            index = row * n + column
            vector = vector_field[index]
            # Pixel y increases downward, so invert the physical y component.
            display_vector = (float(vector[0]) * scale, -float(vector[1]) * scale)
            magnitude = float(magnitudes[index])
            color = _color(magnitude, value_range[0], value_range[1], VIRIDIS)
            _draw_arrow(draw, points[index], display_vector, color)


def _render_figure(
    *,
    mesh: Mesh,
    input_values: torch.Tensor,
    output_values: torch.Tensor,
    input_kind: str,
    input_title: str,
    output_title: str,
    filename: str,
) -> None:
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)

    if input_kind == "vector":
        input_scalar = input_values.norm(dim=-1)
        input_range = (0.0, max(float(input_scalar.max()), 1.0e-12))
        _draw_vector_mesh(draw, mesh, input_values, LEFT_PANEL, input_range)
        input_palette = VIRIDIS
    else:
        input_range = _robust_range(input_values, symmetric=True)
        _draw_scalar_mesh(draw, mesh, input_values, LEFT_PANEL, input_range)
        input_palette = COOLWARM

    # Open-mesh boundary values obey a one-sided operator and can be much
    # larger than the interior response.  Set the displayed range from the
    # interior, while still drawing (and clipping) every computed value.
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    interior = (x > 0.1) & (x < 0.9) & (y > 0.1) & (y < 0.9)
    output_range = _robust_range(output_values[interior], symmetric=True)
    _draw_scalar_mesh(draw, mesh, output_values, RIGHT_PANEL, output_range)

    _draw_axes(draw, LEFT_PANEL)
    _draw_axes(draw, RIGHT_PANEL)
    _draw_title(draw, LEFT_PANEL, input_title)
    _draw_title(draw, RIGHT_PANEL, output_title)
    _draw_colorbar(draw, LEFT_COLORBAR, *input_range, input_palette)
    _draw_colorbar(draw, RIGHT_COLORBAR, *output_range, COOLWARM)

    output_path = OUTPUT_DIR / filename
    image.save(output_path, format="PNG", optimize=True, dpi=(100, 100))
    print(f"Saved {output_path}")


def main() -> None:
    mesh = _make_mesh()
    points = mesh.points
    scalar, divergence_field, curl_field = _sample_fields(points)
    adjacency = mesh.get_point_to_points_adjacency()
    cotan_weights, edges = compute_cotan_weights_fem(mesh)
    dual_volumes = compute_dual_volumes_0(mesh)

    lsq_arguments = {
        "points": points,
        "neighbor_offsets": adjacency.offsets,
        "neighbor_indices": adjacency.indices,
        "weight_power": 2.0,
        "implementation": "torch",
    }

    figures: tuple[
        tuple[str, str, torch.Tensor, Callable[[], torch.Tensor], str], ...
    ] = (
        (
            "mesh_lsq_divergence.png",
            "LSQ Divergence",
            divergence_field,
            lambda: mesh_lsq_divergence(vector_field=divergence_field, **lsq_arguments),
            "vector",
        ),
        (
            "mesh_cotan_divergence.png",
            "Cotangent Divergence (DEC)",
            divergence_field,
            lambda: mesh_cotan_divergence(
                points=points,
                edges=edges,
                cotan_weights=cotan_weights,
                dual_volumes=dual_volumes,
                vector_field=divergence_field,
                implementation="torch",
            ),
            "vector",
        ),
        (
            "mesh_lsq_curl.png",
            "LSQ Curl (2D)",
            curl_field,
            lambda: mesh_lsq_curl(vector_field=curl_field, **lsq_arguments),
            "vector",
        ),
        (
            "mesh_lsq_laplacian.png",
            "Double-LSQ Laplacian",
            scalar,
            lambda: mesh_lsq_laplacian(values=scalar, **lsq_arguments),
            "scalar",
        ),
        (
            "mesh_cotan_laplacian.png",
            "Cotangent Laplacian",
            scalar,
            lambda: mesh_cotan_laplacian(
                edges=edges,
                cotan_weights=cotan_weights,
                dual_volumes=dual_volumes,
                values=scalar,
                implementation="torch",
            ),
            "scalar",
        ),
    )

    with torch.no_grad():
        for filename, title, input_values, compute, input_kind in figures:
            _render_figure(
                mesh=mesh,
                input_values=input_values,
                output_values=compute(),
                input_kind=input_kind,
                input_title=(
                    "Input Vector Field"
                    if input_kind == "vector"
                    else "Input Scalar Field"
                ),
                output_title=title,
                filename=filename,
            )


if __name__ == "__main__":
    main()
