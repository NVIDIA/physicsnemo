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

"""Regression tests for Python 3.14 tensorclass annotation evaluation."""

import builtins
import inspect

import pytest

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.spatial import BVH, ClusterTree, DualInteractionPlan
from physicsnemo.mesh.spatial.cluster_tree import SourceAggregates


def _physicsnemo_members(tensorclass):
    """Yield functions defined directly on a PhysicsNeMo tensorclass."""
    qualname_prefix = f"{tensorclass.__qualname__}."
    for member in vars(tensorclass).values():
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        if inspect.isfunction(member) and member.__qualname__.startswith(
            qualname_prefix
        ):
            yield member


@pytest.mark.parametrize(
    "tensorclass",
    (
        Mesh,
        DomainMesh,
        Adjacency,
        BVH,
        ClusterTree,
        DualInteractionPlan,
        SourceAggregates,
    ),
)
def test_tensorclass_annotations_are_introspectable(tensorclass):
    """Unqualified builtin annotations must resolve to the builtins.

    Python 3.14 evaluates annotations lazily and looks names up in the class
    namespace first. ``@tensorclass`` installed conversion methods such as
    ``int`` and ``float`` there; ``TensorClass`` subclasses inherit them instead.
    """
    shadowing = sorted(
        name
        for name, member in vars(tensorclass).items()
        if name in vars(builtins)
        and callable(member)
        and member is not getattr(builtins, name)
    )
    assert not shadowing, (
        f"{tensorclass.__name__} shadows builtins in its class namespace: {shadowing}"
    )

    for member in _physicsnemo_members(tensorclass):
        try:
            inspect.signature(member)
        except Exception as error:
            pytest.fail(f"Could not inspect {member.__qualname__}: {error}")
