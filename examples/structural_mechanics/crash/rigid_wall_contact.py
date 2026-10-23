# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Analytic rigid-wall contact for the full-vehicle crash recipes.

The Yaris runs drive the car along +x into a rigid barrier: in every store the
kept nodes never exceed x = 100 mm in any frame. The wall is a plane normal to
one axis; each node whose shell clearance to it lies inside the activation band
receives one obstacle contact edge (the same encoding as the bumper cylinder:
displacement to the closest point, signed gap, inward normal), and the model
may also project its prediction back to the admissible side of the wall.
"""

from __future__ import annotations

import torch
from torch import nn

from physicsnemo.experimental.models.meshtransolver.contact import (
    CONTACT_FEATURE_DIM,
    ContactGraph,
    contact_edge_features,
    smooth_contact_cutoff,
)


class RigidWallContactEncoder(nn.Module):
    """Obstacle contact edges against the plane ``axis = position``.

    ``sign=+1`` means admissible points satisfy ``x_axis <= position`` (the
    wall lies at larger coordinates); ``sign=-1`` the opposite. The signed gap
    is the distance to the wall surface minus half the nodal thickness; nodes
    with ``gap < activation_distance`` are active, and the smooth cutoff
    tapers their weight to zero at the band edge.
    """

    def __init__(
        self,
        axis: int = 0,
        sign: float = 1.0,
        activation_distance: float = 5.0,
        feature_scale: float = 10.0,
        normal_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if sign not in (1.0, -1.0, 1, -1):
            raise ValueError("sign must be +1 or -1")
        if activation_distance <= 0 or feature_scale <= 0 or normal_epsilon <= 0:
            raise ValueError("activation_distance, feature_scale, normal_epsilon > 0")
        self.axis = int(axis)
        self.sign = float(sign)
        self.activation_distance = float(activation_distance)
        self.feature_scale = float(feature_scale)
        self.normal_epsilon = float(normal_epsilon)
        self.feature_dim = CONTACT_FEATURE_DIM

    def signed_gap(self, positions, shell_thickness, wall_position):
        """Shell clearance to the wall surface, ``[N]`` (negative = penetrating)."""
        coordinate = positions[:, self.axis]
        distance = self.sign * (float(wall_position) - coordinate)
        return distance - 0.5 * shell_thickness

    def forward(self, positions, shell_thickness, wall_position) -> ContactGraph:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        n = positions.shape[0]
        device, dtype = positions.device, positions.dtype
        if shell_thickness is None:
            shell_thickness = positions.new_zeros(n)
        shell_thickness = torch.as_tensor(shell_thickness, dtype=dtype, device=device)
        if shell_thickness.shape != (n,):
            raise ValueError("shell_thickness must have shape [N]")
        gap = self.signed_gap(positions, shell_thickness, wall_position)
        active = torch.nonzero(gap < self.activation_distance, as_tuple=False).flatten()
        if active.numel() == 0:
            return ContactGraph.empty(device, dtype, self.feature_dim)
        active_gap = gap.index_select(0, active)
        # Vector from the node to its closest point on the wall surface; the
        # inward normal points from the node toward the wall.
        geometric = active_gap + 0.5 * shell_thickness.index_select(0, active)
        normal = positions.new_zeros((active.numel(), 3))
        normal[:, self.axis] = self.sign
        displacement = normal * geometric.unsqueeze(-1)
        toward_surface = torch.where((geometric >= 0).unsqueeze(-1), normal, -normal)
        return ContactGraph(
            edge_index=torch.stack((active, active), dim=0),
            edge_features=contact_edge_features(
                displacement,
                active_gap,
                self.feature_scale,
                contact_normal=toward_surface,
                normal_epsilon=self.normal_epsilon,
            ),
            obstacle_mask=torch.ones(active.numel(), dtype=torch.bool, device=device),
            edge_weights=smooth_contact_cutoff(active_gap, self.activation_distance),
        )
