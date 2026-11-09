# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render volume- and bending-regularized RBF deformation of a sphere."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import (
    ACCENT_LIGHTING,
    FIXED,
    RESULT,
    SECOND,
    SURFACE,
    SURFACE_EDGE,
    mesh_label,
    save,
    spheres,
)
from matplotlib.lines import Line2D

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.deformation import (
    closed_surface_volume_energy,
    surface_bending_energy,
)
from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.nn.functional import radial_basis_function_deform_points

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_deformation_energies.png"

CAMERA = [(4.3, -5.6, 2.4), (0.05, 0.0, 0.2), (0.0, 0.0, 1.0)]


def _backend() -> tuple[torch.device, str]:
    """Select the accelerated backend when a CUDA device is available."""

    if torch.cuda.is_available():
        return torch.device("cuda"), "warp"
    return torch.device("cpu"), "torch"


def _control_indices(points: torch.Tensor) -> torch.Tensor:
    """Select distinct sphere vertices nearest to prescribed directions."""

    directions = points.new_tensor(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.65],
            [-1.0, 0.0, 0.65],
            [0.0, 1.0, 0.65],
            [0.0, -1.0, 0.65],
            [1.0, 0.0, -0.55],
            [-1.0, 0.0, -0.55],
            [0.0, 1.0, -0.55],
            [0.0, -1.0, -0.55],
            [0.0, 0.0, 1.0],
        ]
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    indices = (directions @ points.T).argmax(dim=1)
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError("Sphere subdivision did not provide distinct controls")
    return indices


def _displacements(
    controls: torch.Tensor,
    auxiliary: torch.Tensor,
    target_displacement: torch.Tensor,
) -> torch.Tensor:
    """Assemble fixed, auxiliary, and target control displacements."""

    return torch.cat(
        (
            torch.zeros_like(controls[:2]),
            auxiliary,
            target_displacement.unsqueeze(0),
        )
    )


def _deform(
    reference_points: torch.Tensor,
    controls: torch.Tensor,
    auxiliary: torch.Tensor,
    target_displacement: torch.Tensor,
    implementation: str,
) -> torch.Tensor:
    """Evaluate the radial-basis deformation for one auxiliary-control state."""

    return radial_basis_function_deform_points(
        reference_points,
        controls,
        _displacements(controls, auxiliary, target_displacement),
        kernel="thin_plate_spline",
        polynomial=True,
        smoothing=0.0,
        implementation=implementation,
    )


def _signed_volume(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute signed volume independently for figure annotations."""

    vertices = points[triangles]
    return (
        torch.sum(
            vertices[:, 0] * torch.linalg.cross(vertices[:, 1], vertices[:, 2], dim=1)
        )
        / 6.0
    )


def _render(actors, camera):
    """Render PyVista actors from a fixed camera, so that panels share one scale."""

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.enable_anti_aliasing("ssaa")
    # The surface is translucent, so that controls on the far side stay visible.
    plotter.enable_depth_peeling()
    plotter.set_background("white")
    for dataset, kwargs in actors:
        plotter.add_mesh(dataset, **{"ambient": 0.35, **kwargs})
    plotter.camera_position = camera
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _trim(images, margin=40):
    """Crop the white border shared by same-size renders, keeping their common scale."""

    ink = np.any([(image[..., :3] < 250).any(axis=2) for image in images], axis=0)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    r0, r1 = max(rows[0] - margin, 0), rows[-1] + margin
    c0, c1 = max(cols[0] - margin, 0), cols[-1] + margin
    return [image[r0:r1, c0:c1] for image in images]


def _surface(mesh):
    """Return the shaded surface actor of a mesh."""

    return (
        to_pyvista(mesh),
        {
            "color": SURFACE,
            "show_edges": True,
            "edge_color": SURFACE_EDGE,
            "line_width": 0.6,
            "diffuse": 0.7,
            "opacity": 0.55,
        },
    )


def _arrows(starts, vectors, color):
    """Draw displacement arrows as solid geometry."""

    arrows = [
        pv.Arrow(
            start=start,
            direction=vector,
            scale=float(np.linalg.norm(vector)),
            tip_radius=0.18,
            shaft_radius=0.06,
            tip_length=0.35,
        )
        for start, vector in zip(starts, vectors, strict=True)
    ]
    return (pv.merge(arrows), {"color": color, **ACCENT_LIGHTING})


def _legend(figure, entries):
    """Add a marker legend at the bottom center."""

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=12,
            label=label,
        )
        for label, color in entries
    ]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)


def _panel(mesh, points, controls, displacements, *, original):
    """Render one deformed surface with its anchors, controls, and handle."""

    shown = Mesh(points=points.detach().cpu().float(), cells=mesh.cells)
    controls_np = controls.detach().cpu().numpy()
    displacements_np = displacements.detach().cpu().numpy()
    placed = controls_np if original else controls_np + displacements_np
    source = controls_np[-1]
    destination = source + displacements_np[-1]
    actors = [
        _surface(shown),
        (spheres(placed[:2], 0.06), {"color": FIXED, **ACCENT_LIGHTING}),
        (spheres(placed[2:13], 0.055), {"color": SECOND, **ACCENT_LIGHTING}),
        (
            spheres((source if original else destination)[None], 0.075),
            {"color": RESULT, **ACCENT_LIGHTING},
        ),
    ]
    if not original:
        actors.append(_arrows(source[None], (destination - source)[None], RESULT))
    return _render(actors, CAMERA)


def main() -> None:
    """Generate the documentation image."""

    device, implementation = _backend()
    reference = sphere_icosahedral.load(radius=1.0, subdivisions=3).to(
        device=device, dtype=torch.float64
    )
    controls = reference.points[_control_indices(reference.points)]
    target_displacement = controls.new_tensor([0.12, -0.06, 0.58])
    unregularized_auxiliary = torch.zeros_like(controls[2:13])
    unregularized = _deform(
        reference.points,
        controls,
        unregularized_auxiliary,
        target_displacement,
        implementation,
    ).detach()

    auxiliary = torch.nn.Parameter(torch.zeros_like(unregularized_auxiliary))
    optimizer = torch.optim.Adam([auxiliary], lr=2.0e-2)
    for _ in range(400):
        optimizer.zero_grad()
        points = _deform(
            reference.points,
            controls,
            auxiliary,
            target_displacement,
            implementation,
        )
        loss = 250.0 * closed_surface_volume_energy(
            reference, points, implementation=implementation
        ) + 0.04 * surface_bending_energy(
            reference, points, implementation=implementation
        )
        loss.backward()
        if not torch.isfinite(auxiliary.grad).all():
            raise RuntimeError("Non-finite deformation-energy gradient")
        optimizer.step()

    regularized = _deform(
        reference.points,
        controls,
        auxiliary,
        target_displacement,
        implementation,
    ).detach()
    reference_volume = _signed_volume(reference.points, reference.cells)

    def metrics(points: torch.Tensor) -> tuple[float, float]:
        """Return enclosed-volume and hinge-bending residuals."""

        volume_error = (
            _signed_volume(points, reference.cells) / reference_volume - 1.0
        ).abs()
        bending = surface_bending_energy(
            reference, points, implementation=implementation
        )
        return volume_error.item(), bending.item()

    original_metrics = metrics(reference.points)
    unregularized_metrics = metrics(unregularized)
    regularized_metrics = metrics(regularized)
    if not regularized_metrics[0] < 0.2 * unregularized_metrics[0]:
        raise RuntimeError("Volume penalty did not materially reduce the residual")
    if not regularized_metrics[1] < unregularized_metrics[1]:
        raise RuntimeError("Bending penalty did not reduce the bending residual")

    panels = (
        (
            "reference sphere, fixed RBF controls",
            reference.points,
            torch.zeros_like(controls),
            original_metrics,
            True,
        ),
        (
            "RBF target, no penalties",
            unregularized,
            _displacements(controls, unregularized_auxiliary, target_displacement),
            unregularized_metrics,
            False,
        ),
        (
            "volume + bending penalties",
            regularized,
            _displacements(controls, auxiliary.detach(), target_displacement),
            regularized_metrics,
            False,
        ),
    )

    cpu_reference = reference.to("cpu")
    images = _trim(
        [
            _panel(cpu_reference, points, controls, displacements, original=original)
            for _, points, displacements, _, original in panels
        ]
    )
    height, width = images[0].shape[:2]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5 * height / width + 1.15))
    for axis, image, (title, _, _, residuals, _) in zip(
        axes, images, panels, strict=True
    ):
        volume_error, bending = residuals
        axis.imshow(image)
        axis.set_title(
            f"{title}\n{mesh_label(reference)}\n"
            f"volume error {volume_error:.1e}, bending {bending:.2f}"
        )
        axis.set_axis_off()
    _legend(
        figure,
        (
            ("fixed anchors", FIXED),
            ("optimized controls", SECOND),
            ("target handle", RESULT),
        ),
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.07, top=0.98, wspace=0.02)
    save(figure, OUTPUT, tight=True)
    print(f"Saved {OUTPUT} with {implementation} on {device}")


if __name__ == "__main__":
    main()
