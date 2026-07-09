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

"""Device-independent tests for the public Mesh remeshing API."""

import importlib
import inspect

import pytest

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import WarpRemeshOptions, remesh


def test_remesh_public_signatures():
    assert tuple(inspect.signature(remesh).parameters) == (
        "mesh",
        "n_clusters",
        "max_iterations",
        "warp_options",
        "implementation",
    )
    assert "warp_options" in inspect.signature(Mesh.remesh).parameters


def test_remesh_rejects_wrong_options_type_without_cuda():
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(TypeError, match="WarpRemeshOptions instance.*dict"):
        remesh(
            source,
            48,
            warp_options={"hash_grid_resolution": 64},
            implementation="warp",
        )


def test_warp_options_are_rejected_by_pyacvd_before_importing_it():
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(
        ValueError,
        match="warp_options can only be used with implementation='warp'",
    ):
        remesh(
            source,
            48,
            warp_options=WarpRemeshOptions(),
            implementation="pyacvd",
        )


def test_missing_pyacvd_error_has_direct_install_instructions(monkeypatch):
    module = importlib.import_module("physicsnemo.mesh.remeshing._remeshing")
    missing = OptionalImport(
        "physicsnemo_test_missing_pyacvd",
        package_hint='pip install "pyacvd>=0.3.2" "pyvista>=0.47.0"',
    )
    monkeypatch.setattr(module, "pyacvd", missing)
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(ImportError, match=r'pip install "pyacvd>=0\.3\.2"'):
        remesh(source, 48, implementation="pyacvd")
