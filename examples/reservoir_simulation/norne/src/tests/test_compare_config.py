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

"""Smoke tests for the compare workflow's configuration dataclasses.

These dataclasses bundle the ~68 parameters of ``compare_and_analyze_results``
into 9 grouped arguments. The tests verify each dataclass:
- can be constructed with no arguments (every field has a default), and
- accepts a full set of keyword arguments without ``TypeError``.
"""

import os
import sys

import pytest

# Allow imports from src/ when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compare.sequential.utils.compare_config import (
    CompareFields,
    CompareFlow,
    CompareGrid,
    CompareNorms,
    CompareRuntime,
    CompareSurrogate,
    CompareTiming,
    CompareWellResults,
    CompareWells,
)


def test_compare_timing_defaults_and_full():
    """Test compare timing defaults and full."""
    t = CompareTiming()
    assert t.physicsnemo_time == 0.0 and t.flow_time == 0.0
    t = CompareTiming(physicsnemo_time=12.5, flow_time=300.0)
    assert t.physicsnemo_time == pytest.approx(12.5)


def test_compare_grid_defaults_and_full():
    """Test compare grid defaults and full."""
    g = CompareGrid()
    assert g.nx == 0 and g.Ne == 0
    g = CompareGrid(nx=46, ny=112, nz=22, steppi=10, steppi_indices=[0, 1], Ne=200)
    assert g.nx == 46 and g.steppi == 10 and g.Ne == 200


def test_compare_wells_defaults_independent_lists():
    """Test compare wells defaults independent lists."""
    w1 = CompareWells()
    w2 = CompareWells()
    w1.producers.append("X")
    assert w2.producers == [], "default lists must not be shared between instances"


def test_compare_wells_full():
    """Test compare wells full."""
    w = CompareWells(
        N_pr=22,
        N_injw=9,
        N_injg=4,
        lenwels=3,
        injectors=[("i1",)],
        producers=[("p1",)],
        gas_injectors=[("g1",)],
        well_names=["B-2H"],
        columns=["WOPR"],
        compdat_data={"B-2H": []},
    )
    assert w.N_pr == 22 and w.lenwels == 3
    assert w.injectors == [("i1",)]


def test_compare_fields_full():
    """Test compare fields full."""
    f = CompareFields(
        pressure_pred=[1.0],
        pressure_true=[2.0],
        water_pred=[0.3],
        water_true=[0.4],
        oil_pred=[0.6],
        oil_true=[0.5],
        gas_pred=[0.1],
        gas_true=[0.1],
    )
    assert f.pressure_pred == [1.0]
    assert f.gas_true == [0.1]


def test_compare_well_results_full():
    """Test compare well results full."""
    r = CompareWellResults(ouut_peacemann=[1.0], out_fcn_true=[2.0])
    assert r.ouut_peacemann == [1.0] and r.out_fcn_true == [2.0]


def test_compare_runtime_full():
    """Test compare runtime full."""
    r = CompareRuntime(
        cfg=object(),
        device="cuda",
        num_cores=8,
        oldfolder="/tmp",
        folderr="/out",
        output_variables=["PRESSURE"],
        well_measurements=[1.0],
    )
    assert r.num_cores == 8
    assert r.output_variables == ["PRESSURE"]


def test_compare_norms_defaults_and_full():
    """Test compare norms defaults and full."""
    n = CompareNorms()
    assert n.target_min == pytest.approx(0.01)
    assert n.minK is None and n.maxQg is None
    n = CompareNorms(
        minK=1.0,
        maxK=2000.0,
        minT=0.0,
        maxT=1e4,
        minP=14.0,
        maxP=5000.0,
        minQ=0.0,
        maxQ=1e5,
        minQw=0.0,
        maxQw=1e4,
        minQg=0.0,
        maxQg=1e6,
        target_min=0.01,
        target_max=1.0,
    )
    assert n.maxQg == pytest.approx(1e6)


def test_compare_surrogate_defaults_and_full():
    """Test compare surrogate defaults and full."""
    s = CompareSurrogate()
    assert s.degg == 3 and s.experts == 5
    s = CompareSurrogate(models={"p": object()}, degg=4, experts=8, inn=[1.0])
    assert s.degg == 4 and s.experts == 8


def test_compare_flow_full():
    """Test compare flow full."""
    f = CompareFlow(
        active_cells_ensemble=[True],
        active_mask_3d=[True],
        awater=[1.0],
        agas=[2.0],
        aoil=[3.0],
        aqq=[6.0],
        Time=[0.0, 1.0],
    )
    assert f.aqq == [6.0]
    assert f.Time == [0.0, 1.0]
