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
"""Evaluation-time geometric corrections of a predicted shape.

Two corrections, both pure geometry and both off unless a recipe turns them on:

* the **envelope projection** moves nodes that violate the shell thickness
  envelope back onto it (and nodes that passed through a shell back through
  it), using the crossing-aware pair search of
  :mod:`physicsnemo.experimental.nn.shell_contact`;
* the **rigid-wall clamp** caps the coordinate of every node at an analytic
  rigid wall plane (the Yaris barrier at ``x = 100`` mm), so no node is
  predicted inside the wall.

The settings are small frozen dataclasses that the models and recipes hold;
:func:`correct_shape` applies them to one predicted frame in the datapipe's
normalized coordinates. Nothing here is trained or differentiated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from physicsnemo.experimental.nn.shell_contact import (
    ShellContactSearch,
    clamp_to_wall,
    project_shell_penetrations,
)


@dataclass(frozen=True)
class PenetrationProjection:
    """Settings of the envelope projection (``iterations=0`` disables it).

    Parameters
    ----------
    iterations : int
        Search-and-move passes per frame.
    tolerance_mm : float
        Envelope violation left in place.
    damping : float
        Fraction of the move applied per pass, in ``(0, 1]``; ``0.5`` lets
        both sides of a mutual crossing meet at the envelope.
    band_mm : float
        Activation band of the projection's own pair search.
    clearance_mm : float or None
        Reference clearance of that search: pairs assembled closer than this
        in the reference configuration are structure, not contact.
    crossed_only : bool
        Move only nodes whose deepest pair has crossed a shell.
    side_symmetric : bool
        Symmetric side test of the search (a node that slid under a facet's
        edge is not a crossing).
    implementation : {"warp", "torch"}
        Broad-phase backend of the search.
    max_pairs : int
        Active-pair budget of the search.
    """

    iterations: int = 0
    tolerance_mm: float = 0.0
    damping: float = 1.0
    band_mm: float = 5.0
    clearance_mm: float | None = 10.0
    crossed_only: bool = False
    side_symmetric: bool = False
    implementation: str = "warp"
    max_pairs: int = 8_000_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.iterations, bool)
            or int(self.iterations) != self.iterations
            or self.iterations < 0
        ):
            raise ValueError(
                "penetration_projection_iterations must be an integer >= 0"
            )
        if not math.isfinite(self.tolerance_mm) or self.tolerance_mm < 0:
            raise ValueError("penetration_projection_tolerance_mm must be >= 0")
        if not 0.0 < float(self.damping) <= 1.0:
            raise ValueError("penetration_projection_damping must be in (0, 1]")
        if not self.band_mm > 0:
            raise ValueError("penetration_projection_band_mm must be positive")
        if self.clearance_mm is not None and (
            not math.isfinite(self.clearance_mm) or self.clearance_mm < 0
        ):
            raise ValueError("penetration_projection_clearance_mm must be >= 0 or null")
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "tolerance_mm", float(self.tolerance_mm))
        object.__setattr__(self, "damping", float(self.damping))
        object.__setattr__(self, "band_mm", float(self.band_mm))
        object.__setattr__(self, "crossed_only", bool(self.crossed_only))
        object.__setattr__(self, "side_symmetric", bool(self.side_symmetric))
        if self.clearance_mm is not None:
            object.__setattr__(self, "clearance_mm", float(self.clearance_mm))
        # The search settings validate the implementation and the budget.
        self.search()

    @property
    def enabled(self) -> bool:
        """Whether any projection iterations are configured."""
        return self.iterations > 0

    def search(self) -> ShellContactSearch:
        """The crossing-aware pair search of the projection (material fan on)."""
        return ShellContactSearch(
            activation_distance=self.band_mm,
            implementation=self.implementation,
            max_pairs=self.max_pairs,
            material_fan=True,
            side_from_reference=True,
            reference_clearance=self.clearance_mm,
            side_symmetric=self.side_symmetric,
        )


@dataclass(frozen=True)
class RigidWallClamp:
    """Settings of the rigid-wall clamp.

    Parameters
    ----------
    enabled : bool
        Apply the clamp (the wall position comes from ``graph.rigid_wall``,
        set by ``datapipe.rigid_wall_position``).
    axis : int
        Coordinate axis of the wall normal (0, 1 or 2).
    sign : float
        ``+1`` when the admissible half space is ``x[axis] <= wall``.
    """

    enabled: bool = False
    axis: int = 0
    sign: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2) or float(self.sign) not in (1.0, -1.0):
            raise ValueError("wall_axis must be 0, 1 or 2 and wall_sign +1 or -1")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "axis", int(self.axis))
        object.__setattr__(self, "sign", float(self.sign))


def wall_position(sample) -> float:
    """The rigid wall plane coordinate the datapipe attached to the graph."""
    wall = getattr(sample.graph, "rigid_wall", None)
    if wall is None:
        raise ValueError(
            "the rigid-wall clamp needs graph.rigid_wall (datapipe.rigid_wall_position)"
        )
    return float(wall.reshape(-1)[0])


def physical_geometry(normalized, sample, data_stats):
    """Positions, reference positions and nodal thickness in float32 millimetres.

    ``normalized`` are positions in the datapipe's standardized coordinates;
    the reference is the sample's initial mesh; the thickness comes from
    ``graph.shell_thickness`` (``datapipe.contact_surface`` or
    ``datapipe.rigid_wall_position``).
    """
    thickness = getattr(sample.graph, "shell_thickness", None)
    if thickness is None:
        raise ValueError("physical geometry needs graph.shell_thickness")
    scale = data_stats["node"]["pos_std"].to(normalized.device).float()
    mean = data_stats["node"]["pos_mean"].to(normalized.device).float()
    positions = normalized.float() * scale + mean
    reference = sample.node_features["coords"].detach().float() * scale + mean
    thickness = thickness.to(device=normalized.device, dtype=torch.float32)
    return positions, reference, thickness.reshape(-1)


@torch.no_grad()
def correct_shape(
    prediction: torch.Tensor,
    sample,
    data_stats,
    *,
    projection: PenetrationProjection | None = None,
    clamp: RigidWallClamp | None = None,
) -> torch.Tensor:
    """Apply the enabled corrections to one predicted frame ``[N, 3]``.

    The projection runs in physical millimetres on the shell facets and
    one-ring exclusions the datapipe attached to the graph
    (``datapipe.contact_surface=true``) and returns to normalized
    coordinates; the clamp then caps the wall coordinate in normalized
    coordinates. Returns ``prediction`` itself when nothing is enabled.
    """
    if (projection is None or not projection.enabled) and (
        clamp is None or not clamp.enabled
    ):
        return prediction
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        if projection is not None and projection.enabled:
            graph = sample.graph
            faces = getattr(graph, "contact_faces", None)
            exclusions = getattr(graph, "contact_surface_exclusions", None)
            if faces is None or exclusions is None:
                raise ValueError(
                    "the envelope projection needs graph.contact_faces and "
                    "graph.contact_surface_exclusions (datapipe.contact_surface=true)"
                )
            positions, reference, thickness = physical_geometry(
                prediction, sample, data_stats
            )
            positions = project_shell_penetrations(
                positions,
                faces,
                thickness,
                reference,
                search=projection.search(),
                iterations=projection.iterations,
                tolerance=projection.tolerance_mm,
                damping=projection.damping,
                crossed_only=projection.crossed_only,
                excluded_pairs=exclusions,
                batch=getattr(graph, "batch", None),
            )
            scale = data_stats["node"]["pos_std"].to(prediction.device).float()
            mean = data_stats["node"]["pos_mean"].to(prediction.device).float()
            prediction = ((positions - mean) / scale).to(prediction.dtype)
        if clamp is not None and clamp.enabled:
            std = data_stats["node"]["pos_std"].reshape(-1)[clamp.axis].to(prediction)
            mean = data_stats["node"]["pos_mean"].reshape(-1)[clamp.axis].to(prediction)
            limit = (wall_position(sample) - mean) / (std + 1e-12)
            prediction = clamp_to_wall(
                prediction, limit, axis=clamp.axis, sign=clamp.sign
            )
    return prediction
