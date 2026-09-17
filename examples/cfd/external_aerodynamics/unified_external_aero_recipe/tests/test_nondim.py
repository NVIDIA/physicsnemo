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

"""``NonDimensionalizeByMetadata`` geometry scaling is applied once per chain."""

import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh

from nondim import NonDimensionalizeByMetadata

L_REF, U_INF = 0.4, 10.0


def _mesh() -> Mesh:
    return Mesh(
        points=torch.randn(6, 3),
        cells=torch.arange(6).view(2, 3),
        point_data=TensorDict({"velocity": torch.randn(6, 3)}, batch_size=[6]),
        cell_data=TensorDict(
            {"prescribed": {"velocity": torch.randn(2, 3)}}, batch_size=[2]
        ),
        global_data=TensorDict(
            {
                "U_inf": torch.tensor([U_INF, 0.0, 0.0]),
                "rho_inf": torch.tensor(1.225),
                "p_inf": torch.tensor(0.0),
                "L_ref": torch.tensor(L_REF),
            },
            batch_size=[],
        ),
    )


def test_geometry_is_scaled_by_default():
    """A single instance divides coordinates by L_ref and velocity by |U_inf|."""
    mesh = _mesh()
    out = NonDimensionalizeByMetadata({"velocity": "velocity"})(mesh)
    assert torch.allclose(out.points, mesh.points / L_REF)
    assert torch.allclose(
        out.point_data["velocity"], mesh.point_data["velocity"] / U_INF
    )


def test_second_instance_leaves_geometry_alone():
    """With scale_geometry=False the second instance scales fields only."""
    mesh = _mesh()
    once = NonDimensionalizeByMetadata({"velocity": "velocity"})(mesh)
    twice = NonDimensionalizeByMetadata(
        {"prescribed.velocity": "velocity"},
        association="cell_data",
        scale_geometry=False,
    )(once)
    assert torch.allclose(twice.points, mesh.points / L_REF)  # not / L_REF**2
    assert torch.allclose(
        twice.cell_data["prescribed", "velocity"],
        mesh.cell_data["prescribed", "velocity"] / U_INF,
    )


def test_inverse_respects_scale_geometry():
    """Inverting both instances restores geometry and fields exactly once."""
    mesh = _mesh()
    first = NonDimensionalizeByMetadata({"velocity": "velocity"})
    second = NonDimensionalizeByMetadata(
        {"prescribed.velocity": "velocity"},
        association="cell_data",
        scale_geometry=False,
    )
    back = first.inverse(second.inverse(second(first(mesh))))
    assert torch.allclose(back.points, mesh.points, atol=1e-6)
    assert torch.allclose(
        back.point_data["velocity"], mesh.point_data["velocity"], atol=1e-6
    )
    assert torch.allclose(
        back.cell_data["prescribed", "velocity"],
        mesh.cell_data["prescribed", "velocity"],
        atol=1e-6,
    )
