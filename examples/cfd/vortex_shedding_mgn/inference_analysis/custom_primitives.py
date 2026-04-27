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

"""Simple 2D point primitive with signed distance and boundary sampling."""

import numpy as np


class Point2D:
    """A 2D point with signed distance field and boundary sampling.

    Parameters
    ----------
    point : tuple[float, float]
        (x, y) coordinates of the point.
    """

    def __init__(self, point):
        """Initialize with (x, y) coordinates."""
        self.point = point

    def sdf(self, points, params=None):
        """Signed distance from query points to this point.

        Sign is determined by ``sign(x - point_x)``.
        """
        dx = points[:, 0] - self.point[0]
        dy = points[:, 1] - self.point[1]
        dist = np.sqrt(dx**2 + dy**2)
        sign = np.where(dx >= 0, 1.0, -1.0)
        return {"sdf": dist * sign}

    def sample_boundary(self, num_points):
        """Return the point coordinates repeated *num_points* times."""
        return {
            "x": np.full((num_points, 1), self.point[0]),
            "y": np.full((num_points, 1), self.point[1]),
            "normal_x": np.ones((num_points, 1)),
            "normal_y": np.zeros((num_points, 1)),
            "area": np.ones((num_points, 1)),
        }
