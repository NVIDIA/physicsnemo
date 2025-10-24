# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import pytest
from pytest_utils import import_or_fail

from physicsnemo.metrics.cae.integral import line_integral, surface_integral

pv = pytest.importorskip("pyvista")


@pytest.fixture
def generate_circle(num_points=1000):
    """
    Generates discretized points and edges along a 2D unit circle in the x-y plane.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    x = np.cos(angles)
    y = np.sin(angles)

    points = np.column_stack((x, y, np.zeros_like(x)))

    edges = [[i, (i + 1) % num_points] for i in range(num_points)]
    edges = np.stack(edges, axis=0)
    return points, edges


@pytest.fixture
def generate_sphere(theta_res=300, phi_res=300):
    """
    Generates discretized sphere mesh in 3D of unit radius
    """
    sphere = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
    return sphere


def test_line_integral(generate_circle):
    points, edges = generate_circle

    # test scalar field
    # circumference of circle
    integral = line_integral(edges, points, np.ones((points.shape[0],)))
    assert np.allclose(integral, 2 * np.pi)

    # test vector field
    # integrate [x**2 - y**2, 2*x*y, 0] along circle
    integral = line_integral(
        edges,
        points,
        np.concatenate(
            [
                points[:, 0:1] ** 2 - points[:, 1:2] ** 2,
                2 * points[:, 0:1] * points[:, 1:2],
                points[:, 2:3],
            ],
            axis=1,
        ),
    )
    assert np.allclose(integral, 0)


@import_or_fail(["pyvista"])
def test_surface_integral(generate_sphere, pytestconfig):
    sphere = generate_sphere

    # test scalar field
    # surface area of sphere
    sphere.point_data["field"] = np.ones_like(sphere.point_data["Normals"][:, 0])
    integral = surface_integral(sphere, data_type="point_data", array_name="field")
    assert np.allclose(integral["integral_field"], np.array(4 * np.pi), rtol=1e-3)

    # test vector field
    # integrate [x**2, y**2, z**2] on sphere
    sphere.point_data["field_vector"] = np.stack(
        [sphere.points[:, 0] ** 2, sphere.points[:, 1] ** 2, sphere.points[:, 2] ** 2],
        axis=1,
    )
    integral = surface_integral(
        sphere, data_type="point_data", array_name="field_vector"
    )
    assert np.allclose(integral["integral_field_vector"], np.array(0.0), rtol=1e-3)
