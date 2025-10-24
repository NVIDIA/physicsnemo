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
import torch
from pytest_utils import import_or_fail

from physicsnemo.metrics.cae.cfd import (
    compute_force_coefficients,
    compute_p_q_r,
    compute_tke_spectrum,
    dominant_freq_calc,
)

pv = pytest.importorskip("pyvista")


@pytest.fixture
def generate_sphere(theta_res=100, phi_res=100):
    """
    Generates discretized sphere mesh in 3D of unit radius
    """
    sphere = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
    return sphere


@pytest.fixture
def generate_box(level=500):
    """
    Generates discretized cube mesh in 3D of unit side
    """
    box = pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), level=level, quads=True)
    return box


@import_or_fail(["pyvista", "shapely"])
def test_frontal_area(generate_sphere, pytestconfig):
    from physicsnemo.metrics.cae.cfd import compute_frontal_area

    sphere = generate_sphere

    # area of circle
    area = compute_frontal_area(sphere, direction="x")
    assert np.allclose(area, np.pi, rtol=1e-3)


@import_or_fail(["pyvista"])
def test_force_coeffs(generate_box, pytestconfig):
    box = generate_box
    box = box.compute_normals()
    box = box.point_data_to_cell_data()
    box = box.compute_cell_sizes()
    cell_centers = box.cell_centers().points

    # forces on a unit cube
    c_total, c_p, c_f = compute_force_coefficients(
        box.cell_data["Normals"],
        box.cell_data["Area"],
        1.0,
        cell_centers[:, 0],
        np.stack([cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]], axis=1),
        force_direction=np.array([1, 0, 0]),
    )

    assert np.allclose(c_total, 8, rtol=1e-3, atol=1e-2)
    assert np.allclose(c_p, 8, rtol=1e-3, atol=1e-2)
    assert np.allclose(c_f, 0, rtol=1e-3, atol=1e-2)


def test_dominant_freq_calc():
    x = np.linspace(0, 10 * np.pi, 1000)

    signal = np.sin(x) + 10 * np.sin(20 * x)

    dominant_freq = dominant_freq_calc(signal, sample_spacing=10 * np.pi / 1000)
    assert np.allclose(dominant_freq, 20 / (2 * np.pi))


def test_p_q_r():
    n = 32
    nx, ny, nz = n, n, n

    vel_gradient = torch.rand((2, 3, 3, nx, ny, nz))
    p, q, r = compute_p_q_r(vel_gradient)

    # TODO: Add more rigorous mathematical tests
    assert p.shape == (2, nx * ny * nz)
    assert q.shape == (2, nx * ny * nz)
    assert r.shape == (2, nx * ny * nz)


def test_compute_tke_spectrum():
    n = 32
    nx, ny, nz = n, n, n
    field = np.random.rand(3, nx, ny, nz)

    # TODO: Add more rigorous mathematical tests
    mk, E, wv, tke = compute_tke_spectrum(field)
    assert mk.shape == (nx * ny * nz,)
    assert E.shape == (nx * ny * nz,)
    assert wv.shape == (n + 1,)
    assert tke.shape == (n + 1,)
