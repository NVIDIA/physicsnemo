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
"""Evaluation-time rigid-wall clamp of a predicted shape.

The clamp caps the coordinate of every node at an analytic rigid wall plane
(the Yaris barrier at ``x = 100`` mm), so no node is predicted inside the wall.
Its settings are a small frozen dataclass that the models and recipes hold;
:func:`correct_shape` applies it to one predicted frame in the datapipe's
normalized coordinates. Nothing here is trained or differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physicsnemo.experimental.nn.shell_contact import clamp_to_wall


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


@torch.no_grad()
def correct_shape(
    prediction: torch.Tensor,
    sample,
    data_stats,
    *,
    clamp: RigidWallClamp | None = None,
) -> torch.Tensor:
    """Apply the rigid-wall clamp to one predicted frame ``[N, 3]`` when enabled.

    The clamp caps the wall coordinate in the datapipe's normalized coordinates;
    the wall position comes from ``graph.rigid_wall``. Returns ``prediction``
    itself when the clamp is off.
    """
    if clamp is None or not clamp.enabled:
        return prediction
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        std = data_stats["node"]["pos_std"].reshape(-1)[clamp.axis].to(prediction)
        mean = data_stats["node"]["pos_mean"].reshape(-1)[clamp.axis].to(prediction)
        limit = (wall_position(sample) - mean) / (std + 1e-12)
        return clamp_to_wall(prediction, limit, axis=clamp.axis, sign=clamp.sign)
