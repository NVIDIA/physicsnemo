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

"""Tests for dtype preservation when exporting mesh data to PyVista."""

import numpy as np
import pytest
import torch

from physicsnemo.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io import to_pyvista  # noqa: E402


def _line_mesh(**kwargs) -> Mesh:
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        cells=torch.tensor([[0, 1]]),
        **kwargs,
    )


def test_preserves_supported_point_cell_and_global_data_dtypes():
    large_ids = torch.tensor([2**40, 2**40 + 1], dtype=torch.int64)
    mesh = _line_mesh(
        point_data={
            "large_ids": large_ids,
            "flags": torch.tensor([True, False]),
        },
        cell_data={"weight": torch.tensor([1.25], dtype=torch.float64)},
        global_data={
            "phase": torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        },
    )

    result = to_pyvista(mesh)

    assert np.asarray(result.point_data["large_ids"]).dtype == np.int64
    assert np.array_equal(result.point_data["large_ids"], large_ids.numpy())
    assert np.asarray(result.point_data["flags"]).dtype == np.bool_
    assert np.asarray(result.cell_data["weight"]).dtype == np.float64
    assert np.asarray(result.field_data["phase"]).dtype == np.complex64
    assert np.array_equal(
        np.asarray(result.field_data["phase"]).reshape(-1),
        mesh.global_data["phase"].numpy(),
    )


def test_resolves_lazy_conjugate_before_export():
    values = torch.tensor([1.0 + 2.0j], dtype=torch.complex64).conj()

    result = to_pyvista(_line_mesh(global_data={"phase": values}))

    exported = np.asarray(result.field_data["phase"]).reshape(-1)
    assert np.array_equal(exported, values.resolve_conj().numpy())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_promotes_reduced_precision_real_data_to_float32(dtype):
    values = torch.tensor([1.0, 2.0], dtype=dtype)

    result = to_pyvista(_line_mesh(point_data={"value": values}))

    exported = np.asarray(result.point_data["value"])
    assert exported.dtype == np.float32
    assert np.array_equal(exported, values.float().numpy())


@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_promotes_reduced_precision_complex_data_to_complex64():
    complex32 = getattr(torch, "complex32", None)
    if complex32 is None:
        pytest.skip("torch.complex32 is unavailable")
    values = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=complex32)

    result = to_pyvista(_line_mesh(point_data={"value": values}))

    exported = np.asarray(result.point_data["value"])
    assert exported.dtype == np.complex64
    assert np.array_equal(exported, values.to(dtype=torch.complex64).numpy())
