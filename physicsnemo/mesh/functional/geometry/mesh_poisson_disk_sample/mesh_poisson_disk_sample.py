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

from collections.abc import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._warp_impl import (
    _DART_THROWING_MODE,
    mesh_poisson_disk_sample_warp,
)


class MeshPoissonDiskSample(FunctionSpec):
    r"""Generate Poisson-disk samples on a triangle mesh surface with Warp.

    This functional supports two sampling modes on triangle meshes:

    1. ``dart_throwing``:
       iterative parallel dart throwing where each iteration draws area-weighted
       candidates, rejects points near accepted samples, resolves candidate-candidate
       conflicts with random-priority MIS, and commits survivors.
    2. ``weighted_sample_elimination``:
       builds an oversampled Poisson-quality pool, then downsamples to
       ``target_num_points`` using a radius-aware elimination pass.

    Both modes produce blue-noise-like sample sets. ``dart_throwing`` emphasizes
    throughput and minimum-distance control; ``weighted_sample_elimination``
    emphasizes distribution quality at a fixed output count.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Mesh vertex positions with shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity in shape ``(n_faces, 3)`` or flattened
        shape ``(3 * n_faces,)``.
    min_distance : float, optional
        Minimum Poisson distance for constant-radius mode. Default is ``0.02``.
        In ``weighted_sample_elimination`` mode this is treated as a lower-bound
        hint while the algorithm primarily targets ``target_num_points`` quality.
    per_vertex_radius : torch.Tensor | None, optional
        Optional adaptive radius with shape ``(n_vertices,)``.
        If provided, candidate radius is barycentrically interpolated.
    mode : str, optional
        Sampling mode. ``"dart_throwing"`` uses iterative parallel dart throwing.
        ``"weighted_sample_elimination"`` builds an oversampled Poisson pool and
        then downsamples to ``target_num_points`` with radius-aware elimination.
    batch_size : int, optional
        Number of generated candidates per iteration. Default is ``131072``.
    max_points : int, optional
        Maximum number of accepted samples. Default is ``2_000_000``.
        For ``mode="weighted_sample_elimination"``, this is also the default
        ``target_num_points`` when that argument is omitted.
    target_num_points : int | None, optional
        Number of output points for ``mode="weighted_sample_elimination"``.
        If ``None``, the mode uses ``max_points``.
    max_iterations : int, optional
        Iteration cap for the sampler. Default is ``64``.
    random_seed : int, optional
        Base random seed for deterministic candidate generation.
    hash_grid_resolution : int | Sequence[int], optional
        Hash-grid resolution, either scalar or ``(nx, ny, nz)``.
        Default is ``128``.
    implementation : str | None, optional
        Explicit implementation name. Defaults to dispatch behavior.

    Returns
    -------
    torch.Tensor
        Accepted sample positions with shape ``(n_samples, 3)`` and dtype
        ``torch.float32``.

    Notes
    -----
    - ``mode="weighted_sample_elimination"`` uses Warp kernels and follows
      Open3D's Yuksel-style weighting equations.
    - ``per_vertex_radius`` is ignored in weighted elimination mode.
    - The output order is implementation-specific and not semantically meaningful.
    """

    _BENCHMARK_CASES = (
        ("small-subdiv2-cst", 2, False, 4096, 0.07),
        ("medium-subdiv3-cst", 3, False, 8192, 0.05),
        ("large-subdiv3-adapt", 3, True, 8192, 0.05),
    )

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=0.6.0",),
        rank=0,
        baseline=True,
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        min_distance: float = 0.02,
        per_vertex_radius: torch.Tensor | None = None,
        batch_size: int = 131072,
        max_points: int = 2_000_000,
        max_iterations: int = 64,
        random_seed: int = 42,
        hash_grid_resolution: int | Sequence[int] | torch.Tensor = 128,
        mode: str = _DART_THROWING_MODE,
        target_num_points: int | None = None,
    ) -> torch.Tensor:
        """Run the Warp backend Poisson-disk sampler on triangle meshes."""
        return mesh_poisson_disk_sample_warp(
            mesh_vertices=mesh_vertices,
            mesh_indices=mesh_indices,
            min_distance=min_distance,
            per_vertex_radius=per_vertex_radius,
            batch_size=batch_size,
            max_points=max_points,
            max_iterations=max_iterations,
            random_seed=random_seed,
            hash_grid_resolution=hash_grid_resolution,
            mode=mode,
            target_num_points=target_num_points,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Build benchmark inputs spanning mesh size and adaptive-radius usage."""
        device = torch.device(device)

        # Build benchmark cases in increasing workload order.
        for seed, (
            label,
            subdivisions,
            adaptive,
            batch_size,
            min_distance,
        ) in enumerate(cls._BENCHMARK_CASES):
            n_rings = 4 * (2**subdivisions)
            n_segments = 8 * (2**subdivisions)

            phi = torch.linspace(0.0, torch.pi, n_rings + 2, device=device)[1:-1]
            theta = torch.linspace(0.0, 2.0 * torch.pi, n_segments + 1, device=device)[
                :-1
            ]
            phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

            sin_phi = phi_g.sin()
            ring_points = torch.stack(
                [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
                dim=-1,
            ).reshape(-1, 3)

            mesh_vertices = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    ring_points,
                    torch.tensor([[0.0, 0.0, -1.0]], device=device),
                ]
            ).to(torch.float32)

            south_idx = n_rings * n_segments + 1
            j = torch.arange(n_segments, device=device)
            j_next = (j + 1) % n_segments

            north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

            r = torch.arange(n_rings - 1, device=device).unsqueeze(1)
            base = 1 + r * n_segments
            p00 = base + j
            p01 = base + j_next
            p10 = base + n_segments + j
            p11 = base + n_segments + j_next
            body_tris = torch.stack(
                [
                    torch.stack([p00, p10, p11], dim=-1),
                    torch.stack([p00, p11, p01], dim=-1),
                ],
                dim=2,
            ).reshape(-1, 3)

            last = south_idx - n_segments
            south_fan = torch.stack(
                [last + j, torch.full_like(j, south_idx), last + j_next], dim=1
            )

            mesh_indices = (
                torch.cat([north_fan, body_tris, south_fan])
                .to(torch.int32)
                .contiguous()
            )
            mesh_vertices = mesh_vertices.contiguous()

            per_vertex_radius = None
            if adaptive:
                # Smoothly varying positive radii derived from normalized z-coordinate.
                z = mesh_vertices[:, 2]
                z_min = z.min()
                z_max = z.max()
                denom = (z_max - z_min).clamp_min(1.0e-6)
                z_norm = (z - z_min) / denom
                per_vertex_radius = (min_distance * (0.75 + 0.5 * z_norm)).to(
                    torch.float32
                )

            yield (
                label,
                (mesh_vertices, mesh_indices),
                {
                    "min_distance": min_distance,
                    "per_vertex_radius": per_vertex_radius,
                    "batch_size": batch_size,
                    "max_points": 32768,
                    "max_iterations": 12,
                    "random_seed": 2026 + seed,
                    "hash_grid_resolution": 128,
                },
            )


mesh_poisson_disk_sample = MeshPoissonDiskSample.make_function(
    "mesh_poisson_disk_sample"
)


__all__ = ["MeshPoissonDiskSample", "mesh_poisson_disk_sample"]
