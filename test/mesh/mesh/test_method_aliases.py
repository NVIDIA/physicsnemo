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

"""Public-contract tests for Mesh methods backed by functional APIs."""

import inspect
import pickle
from typing import get_type_hints

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.boundaries import is_manifold, is_watertight
from physicsnemo.mesh.calculus import (
    compute_cell_derivatives,
    compute_point_derivatives,
    integrate,
    integrate_flux,
    integrate_moment,
)
from physicsnemo.mesh.transformations.deform import (
    displace,
    free_form_deform,
    morph,
    radial_basis_function_deform,
)
from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)
from physicsnemo.mesh.validation import validate
from physicsnemo.mesh.visualization import draw

_METHOD_FUNCTION_PAIRS = (
    ("is_manifold", is_manifold),
    ("is_watertight", is_watertight),
    ("displace", displace),
    ("free_form_deform", free_form_deform),
    ("morph", morph),
    ("radial_basis_function_deform", radial_basis_function_deform),
    ("rotate", rotate),
    ("scale", scale),
    ("transform", transform),
    ("translate", translate),
    ("draw", draw),
    ("compute_cell_derivatives", compute_cell_derivatives),
    ("compute_point_derivatives", compute_point_derivatives),
    ("integrate", integrate),
    ("integrate_flux", integrate_flux),
    ("integrate_moment", integrate_moment),
    ("validate", validate),
)


@pytest.fixture
def triangle_mesh() -> Mesh:
    """Return a small picklable mesh for bound-method contract tests."""
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )


@pytest.mark.parametrize(("method_name", "function"), _METHOD_FUNCTION_PAIRS)
def test_mesh_method_public_contract(method_name, function, triangle_mesh):
    """Bound methods preserve the functional API without identity coupling."""
    method = getattr(Mesh, method_name)
    bound_method = getattr(triangle_mesh, method_name)

    assert method is not function
    assert method.__name__ == method_name
    assert method.__qualname__ == f"Mesh.{method_name}"
    assert method.__module__ == "physicsnemo.mesh.mesh"
    assert method.__doc__ == function.__doc__

    function_signature = inspect.signature(function)
    expected_bound_signature = function_signature.replace(
        parameters=list(function_signature.parameters.values())[1:]
    )
    assert inspect.signature(bound_method) == expected_bound_signature

    hints = get_type_hints(method)
    assert hints["mesh"] is Mesh

    restored = pickle.loads(pickle.dumps(bound_method))  # noqa: S301
    assert restored.__func__ is method
    assert isinstance(restored.__self__, Mesh)
    torch.testing.assert_close(restored.__self__.points, triangle_mesh.points)
    assert torch.equal(restored.__self__.cells, triangle_mesh.cells)


def test_mesh_method_wrapper_delegates_to_function(triangle_mesh):
    """The shared method adapter forwards arguments and return values."""
    offset = torch.tensor([0.25, -0.5])
    expected = translate(triangle_mesh, offset)
    actual = triangle_mesh.translate(offset)

    torch.testing.assert_close(actual.points, expected.points)
    assert torch.equal(actual.cells, expected.cells)


@pytest.mark.parametrize("property_name", ("quality_metrics", "statistics"))
def test_mesh_validation_property_contract(property_name):
    """Property help describes property access rather than functional arguments."""
    descriptor = inspect.getattr_static(Mesh, property_name)

    assert isinstance(descriptor, property)
    assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
    assert "mesh : Mesh" not in descriptor.__doc__
    assert "tolerance :" not in descriptor.__doc__
    assert f"mesh.{property_name}" in descriptor.__doc__
