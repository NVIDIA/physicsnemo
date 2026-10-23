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
"""Optional face-area quadrature for predictive surface contact weights."""

import math

import pytest
import torch

from physicsnemo.experimental.models.meshtransolver import (
    ContactGraph,
    SurfaceContactGraphBuilder,
)


def _scene():
    """A unit quad, a half-area padded triangle, and one query above each."""
    x = torch.tensor(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # quad, area 1
            [3.0, 0, 0],
            [4, 0, 0],
            [3, 1, 0],  # triangle, area 0.5
            [0.5, 0.5, 0.1],  # query above the quad
            [3.3, 0.3, 0.1],  # query above the triangle
        ]
    )
    faces = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 6]])
    return x, faces


@pytest.mark.parametrize("material_fan", [False, True])
def test_area_quadrature_scales_taper_weights_by_face_area(material_fan):
    x, faces = _scene()
    common = dict(
        implementation="torch",
        include_velocity=False,
        activation_distance=0.5,
        material_fan=material_fan,
    )
    extra = {"reference_positions": x} if material_fan else {}
    plain = SurfaceContactGraphBuilder(**common)(
        x, faces=faces, shell_thickness=x.new_zeros(9), **extra
    )
    assert plain.edge_index[1].tolist() == [7, 8]
    assert torch.all((plain.edge_weights > 0) & (plain.edge_weights <= 1))
    unit = SurfaceContactGraphBuilder(
        **common, area_weighting=True, reference_area=1.0
    )(x, faces=faces, shell_thickness=x.new_zeros(9), **extra)
    torch.testing.assert_close(
        unit.edge_weights, plain.edge_weights * x.new_tensor([1.0, 0.5])
    )
    # Everything except the weights is untouched by the quadrature factor.
    torch.testing.assert_close(unit.edge_features, plain.edge_features)
    torch.testing.assert_close(unit.source_weights, plain.source_weights)
    banded = SurfaceContactGraphBuilder(**common, area_weighting=True)(
        x, faces=faces, shell_thickness=x.new_zeros(9), **extra
    )
    disk = math.pi * 0.5**2
    torch.testing.assert_close(
        banded.edge_weights, plain.edge_weights * x.new_tensor([1.0, 0.5]) / disk
    )
    assert banded.edge_weights[0] > 1  # large face relative to the band disk
    banded.validate(9, 8)


def test_area_quadrature_has_no_gradient_and_rejects_bad_reference():
    x, faces = _scene()
    x = x.requires_grad_()
    builder = SurfaceContactGraphBuilder(
        implementation="torch",
        include_velocity=False,
        activation_distance=0.5,
        area_weighting=True,
    )
    graph = builder(x, faces=faces, shell_thickness=x.new_zeros(9))
    (graph.edge_weights * graph.edge_features[:, 1]).sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="reference_area"):
        SurfaceContactGraphBuilder(implementation="torch", reference_area=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        ContactGraph(
            torch.tensor([[0], [1]]),
            torch.zeros(1, 8),
            torch.zeros(1, dtype=torch.bool),
            edge_weights=torch.tensor([-0.1]),
        ).validate(2, 8)
    ContactGraph(
        torch.tensor([[0], [1]]),
        torch.zeros(1, 8),
        torch.zeros(1, dtype=torch.bool),
        edge_weights=torch.tensor([3.5]),
    ).validate(2, 8)
