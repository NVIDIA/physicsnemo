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

r"""Recipe-specific contact encoders for the crash example."""

from __future__ import annotations

import torch
import torch.nn as nn

from physicsnemo.experimental.models.meshtransolver import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ContactGraph,
    contact_edge_features,
    smooth_contact_cutoff,
)


class BumperCylinderContactEncoder(nn.Module):
    r"""Encode contact with the OpenRadioss bumper rigid cylinder.

    The canonical deck uses a cylinder whose axis is parallel to ``z``. Its
    ``y`` center varies per simulation and is supplied at forward time. Each active
    structural node receives one obstacle edge pointing to its closest point on the
    cylinder surface.
    """

    def __init__(
        self,
        center_x: float = -170.0,
        center_z: float = 0.0,
        radius: float = 127.0,
        search_distance: float = 200.0,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        include_velocity: bool = False,
        velocity_scale: float = 1000.0,
        smooth_cutoff: bool = False,
    ) -> None:
        super().__init__()
        if radius <= 0.0 or search_distance <= 0.0:
            raise ValueError("Cylinder radius and search_distance must be positive")
        axis_tensor = torch.tensor(axis, dtype=torch.float32)
        axis_norm = torch.linalg.vector_norm(axis_tensor)
        if axis_norm <= 0.0:
            raise ValueError("Cylinder axis must be non-zero")
        self.center_x = float(center_x)
        self.center_z = float(center_z)
        self.radius = float(radius)
        self.search_distance = float(search_distance)
        if velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        self.include_velocity = include_velocity
        self.velocity_scale = float(velocity_scale)
        self.smooth_cutoff = smooth_cutoff
        self.feature_dim = (
            KINEMATIC_CONTACT_FEATURE_DIM if include_velocity else CONTACT_FEATURE_DIM
        )
        self.register_buffer("axis", axis_tensor / axis_norm, persistent=True)

    def forward(
        self,
        positions: torch.Tensor,
        center_y: torch.Tensor,
        batch: torch.Tensor | None = None,
        shell_thickness: torch.Tensor | None = None,
        velocities: torch.Tensor | None = None,
    ) -> ContactGraph:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        num_nodes = positions.shape[0]
        if self.include_velocity and (
            velocities is None or velocities.shape != positions.shape
        ):
            raise ValueError("velocities must have shape [N, 3] for kinematic contact")
        device = positions.device
        if num_nodes == 0:
            return ContactGraph.empty(device, positions.dtype, self.feature_dim)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            batch = batch.to(device=device, dtype=torch.long)
            if batch.shape != (num_nodes,):
                raise ValueError("batch must have shape [N]")
        if shell_thickness is None:
            shell_thickness = positions.new_zeros(num_nodes)
        else:
            shell_thickness = torch.as_tensor(
                shell_thickness, dtype=positions.dtype, device=device
            ).flatten()
            if shell_thickness.shape != (num_nodes,):
                raise ValueError("shell_thickness must have shape [N] or [N, 1]")
            if torch.any(shell_thickness < 0.0):
                raise ValueError("shell_thickness cannot be negative")
        num_graphs = int(batch.max().item()) + 1 if num_nodes else 0

        center_y = torch.as_tensor(
            center_y, dtype=positions.dtype, device=device
        ).flatten()
        if center_y.numel() == 1 and num_graphs > 1:
            center_y = center_y.expand(num_graphs)
        if center_y.shape != (num_graphs,):
            raise ValueError("center_y must provide one value per graph")

        centers = positions.new_zeros((num_graphs, 3))
        centers[:, 0] = self.center_x
        centers[:, 1] = center_y
        centers[:, 2] = self.center_z
        relative_center = positions - centers.index_select(0, batch)
        axis = self.axis.to(device=device, dtype=positions.dtype)
        axial = (relative_center * axis).sum(dim=-1, keepdim=True) * axis
        radial = relative_center - axial
        radial_distance = torch.linalg.vector_norm(radial, dim=-1)
        outward_normal = radial / radial_distance.clamp_min(1.0e-8).unsqueeze(-1)
        geometric_gap = radial_distance - self.radius
        signed_gap = geometric_gap - 0.5 * shell_thickness
        active = signed_gap <= self.search_distance
        node_index = torch.nonzero(active, as_tuple=False).flatten()
        if node_index.numel() == 0:
            return ContactGraph.empty(device, positions.dtype, self.feature_dim)

        active_gap = signed_gap.index_select(0, node_index)
        active_geometric_gap = geometric_gap.index_select(0, node_index)
        active_normal = outward_normal.index_select(0, node_index)
        # Vector from receiving node to its closest point on the cylinder.
        displacement = -active_normal * active_geometric_gap.unsqueeze(-1)
        toward_surface = torch.where(
            (active_geometric_gap >= 0.0).unsqueeze(-1),
            -active_normal,
            active_normal,
        )
        return ContactGraph(
            edge_index=torch.stack((node_index, node_index), dim=0),
            edge_features=contact_edge_features(
                displacement,
                active_gap,
                self.search_distance,
                contact_normal=toward_surface,
                relative_velocity=-velocities[node_index]
                if self.include_velocity
                else None,
                velocity_scale=self.velocity_scale,
            ),
            obstacle_mask=torch.ones(
                node_index.shape[0], dtype=torch.bool, device=device
            ),
            edge_weights=smooth_contact_cutoff(active_gap, self.search_distance)
            if self.smooth_cutoff
            else None,
        )
