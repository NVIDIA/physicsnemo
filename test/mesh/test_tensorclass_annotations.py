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
from typing import get_args

import pytest

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.spatial import BVH, ClusterTree, DualInteractionPlan
from physicsnemo.mesh.spatial.cluster_tree import SourceAggregates


def _annotation_parts(annotation):
    """Yield an annotation and each recursively nested type argument."""
    yield annotation
    for argument in get_args(annotation):
        yield from _annotation_parts(argument)


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
    """Tensor conversion methods must not shadow builtin type annotations."""
    conversion_methods = {
        member
        for name, member in vars(tensorclass).items()
        if name in vars(builtins)
        and callable(member)
        and member is not getattr(builtins, name)
    }

    for member in _physicsnemo_members(tensorclass):
        try:
            signature = inspect.signature(member)
        except Exception as error:
            pytest.fail(f"Could not inspect {member.__qualname__}: {error}")

        annotations = [
            parameter.annotation for parameter in signature.parameters.values()
        ]
        annotations.append(signature.return_annotation)
        for annotation in annotations:
            for part in _annotation_parts(annotation):
                assert all(part is not method for method in conversion_methods), (
                    f"{member.__qualname__} resolved a builtin type annotation to a "
                    "tensorclass conversion method"
                )
