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

"""Tests for Mesh methods backed directly by their functional APIs."""

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.boundaries import is_manifold, is_watertight
from physicsnemo.mesh.calculus import (
    compute_cell_derivatives,
    compute_point_derivatives,
    integrate,
    integrate_flux,
    integrate_moment,
)
from physicsnemo.mesh.transformations.deform import displace, free_form_deform, morph
from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)
from physicsnemo.mesh.validation import (
    compute_mesh_statistics,
    compute_quality_metrics,
)
from physicsnemo.mesh.visualization.draw_mesh import draw_mesh


def test_mesh_methods_reuse_function_objects():
    assert Mesh.is_manifold is is_manifold
    assert Mesh.is_watertight is is_watertight

    assert Mesh.displace is displace
    assert Mesh.free_form_deform is free_form_deform
    assert Mesh.morph is morph
    assert Mesh.rotate is rotate
    assert Mesh.scale is scale
    assert Mesh.transform is transform
    assert Mesh.translate is translate

    assert Mesh.draw is draw_mesh

    assert Mesh.compute_cell_derivatives is compute_cell_derivatives
    assert Mesh.compute_point_derivatives is compute_point_derivatives
    assert Mesh.integrate is integrate
    assert Mesh.integrate_flux is integrate_flux
    assert Mesh.integrate_moment is integrate_moment


def test_mesh_properties_reuse_function_objects():
    assert Mesh.quality_metrics.fget is compute_quality_metrics
    assert Mesh.statistics.fget is compute_mesh_statistics
