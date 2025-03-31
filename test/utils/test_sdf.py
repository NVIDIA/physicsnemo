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
# ruff: noqa: E402


import numpy as np
from pytest_utils import import_or_fail


def tet_verts(flip_x=1):
    tet = np.array(
        [
            flip_x * 0,
            0,
            0,  # bottom
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            0,  # front
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            0,
            0,  # left
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,  # "top"
            flip_x * 0,
            1,
            0,
            flip_x * 0,
            0,
            1,
        ],
        dtype=np.float64,
    )

    return tet


@import_or_fail("warp")
def test_sdf(pytestconfig):

    from physicsnemo.utils.sdf import signed_distance_field

    tet = tet_verts()

    sdf_tet = signed_distance_field(
        tet,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64),
    )
    np.testing.assert_allclose(sdf_tet.numpy(), [1.15470052, -0.1], atol=1e-7)

    sdf_tet, sdf_hit_point, sdf_hit_point_id = signed_distance_field(
        tet,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32),
        np.array([1, 1, 1, 0.12, 0.11, 0.1], dtype=np.float64),
        include_hit_points=True,
        include_hit_points_id=True,
    )
    np.testing.assert_allclose(
        sdf_hit_point.numpy(),
        [[0.33333322, 0.33333334, 0.3333334], [0.12000002, 0.11, 0.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(sdf_hit_point_id.numpy(), [3, 0], atol=1e-7)
