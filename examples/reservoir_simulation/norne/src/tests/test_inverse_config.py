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

"""Smoke tests for the inverse-problem configuration dataclasses.

These tests verify that every ``inverse.utils.inverse_config`` dataclass:
- can be constructed with no arguments (all fields have defaults), and
- can be constructed with full field values without ``TypeError``.

The dataclass module has no third-party runtime dependencies, so these tests
run in any environment with pytest installed.
"""

import os
import sys

import pytest

# Allow imports from src/ when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inverse.utils.inverse_config import (
    EnsembleRuntime,
    EnsembleSetup,
    FlowArrays,
    GridConfig,
    InversionParams,
    NormBounds,
    PermBounds,
    PriorEnsembles,
    SurrogateConfig,
    TimeArrays,
    WellConfig,
)


# ---------------------------------------------------------------------------
# GridConfig
# ---------------------------------------------------------------------------


def test_grid_config_defaults():
    """Test grid config defaults."""
    g = GridConfig()
    assert g.nx == 0 and g.ny == 0 and g.nz == 0
    assert g.steppi == 0
    assert g.steppi_indices is None


def test_grid_config_full():
    """Test grid config full."""
    g = GridConfig(nx=46, ny=112, nz=22, steppi=246, steppi_indices=[0, 1, 2])
    assert g.nx == 46 and g.ny == 112 and g.nz == 22
    assert g.steppi == 246
    assert g.steppi_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# NormBounds — covers all 22 normalisation fields
# ---------------------------------------------------------------------------


def test_norm_bounds_defaults():
    """Test norm bounds defaults."""
    n = NormBounds()
    assert n.target_min == pytest.approx(0.01)
    assert n.target_max == pytest.approx(1.0)
    assert n.minK is None and n.maxK is None
    assert n.minQ is None and n.maxQ is None


def test_norm_bounds_full():
    """Test norm bounds full."""
    n = NormBounds(
        minK=1.0,
        maxK=2000.0,
        minT=0.0,
        maxT=10000.0,
        minP=14.0,
        maxP=5000.0,
        minQ=0.0,
        maxQ=1e5,
        minQw=0.0,
        maxQw=1e4,
        minQg=0.0,
        maxQg=1e6,
        min_inn_fcn=[0.0],
        max_inn_fcn=[1.0],
        min_out_fcn=[0.0],
        max_out_fcn=[1.0],
        min_inn_fcn2=[0.0],
        max_inn_fcn2=[1.0],
        min_out_fcn2=[0.0],
        max_out_fcn2=[1.0],
        target_min=0.05,
        target_max=0.95,
    )
    assert n.maxK == pytest.approx(2000.0)
    assert n.maxQg == pytest.approx(1e6)
    assert n.target_min == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# PermBounds
# ---------------------------------------------------------------------------


def test_perm_bounds_defaults():
    """Test perm bounds defaults."""
    p = PermBounds()
    assert all(
        getattr(p, attr) == 0.0
        for attr in ("High_K", "Low_K", "High_K1", "Low_K1", "High_P", "Low_P")
    )


def test_perm_bounds_full():
    """Test perm bounds full."""
    p = PermBounds(
        High_K=8.5,
        Low_K=-5.0,
        High_K1=10000.0,
        Low_K1=0.1,
        High_P=0.4,
        Low_P=0.05,
    )
    assert p.High_K == pytest.approx(8.5)
    assert p.Low_P == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# EnsembleSetup
# ---------------------------------------------------------------------------


def test_ensemble_setup_defaults():
    """Test ensemble setup defaults."""
    e = EnsembleSetup()
    assert e.Ne == 0 and e.N_ens == 0
    assert e.effective is None and e.effec is None


def test_ensemble_setup_full():
    """Test ensemble setup full."""
    e = EnsembleSetup(
        Ne=200,
        N_ens=200,
        effective=[1.0],
        effec=[1.0],
        active_cells_ensemble=[1.0],
        active_mask_3d=[True],
        rows_to_remove=[],
        indii=[],
    )
    assert e.Ne == 200 and e.N_ens == 200


# ---------------------------------------------------------------------------
# WellConfig
# ---------------------------------------------------------------------------


def test_well_config_defaults():
    """Test well config defaults."""
    w = WellConfig()
    # ``field(default_factory=list)`` must produce independent lists per instance.
    w1 = WellConfig()
    w.producers.append("X")
    assert w1.producers == []
    assert w.N_pr == 0 and w.lenwels == 0


def test_well_config_full():
    """Test well config full."""
    w = WellConfig(
        producers=[("p1",)],
        injectors=[("i1",)],
        gas_injectors=[("g1",)],
        well_names=["B-2H"],
        N_pr=22,
        N_injw=9,
        N_injg=4,
        lenwels=3,
        compdat_data={"B-2H": []},
    )
    assert w.N_pr == 22 and w.N_injw == 9 and w.N_injg == 4
    assert w.lenwels == 3
    assert w.producers == [("p1",)]


# ---------------------------------------------------------------------------
# SurrogateConfig
# ---------------------------------------------------------------------------


def test_surrogate_config_defaults():
    """Test surrogate config defaults."""
    s = SurrogateConfig()
    assert s.models is None
    assert s.Trainmoe == "FNO"
    assert s.degg == 3 and s.experts == 5 and s.num_cores == 1


def test_surrogate_config_full():
    """Test surrogate config full."""
    s = SurrogateConfig(
        models={"pressure": object()},
        Trainmoe="MoE",
        pred_type="hard",
        degg=4,
        experts=8,
        num_cores=8,
    )
    assert s.Trainmoe == "MoE"
    assert s.degg == 4 and s.experts == 8


# ---------------------------------------------------------------------------
# InversionParams
# ---------------------------------------------------------------------------


def test_inversion_params_defaults():
    """Test inversion params defaults."""
    i = InversionParams()
    assert i.Termm == 20
    assert i.Do_parametrisation == "No"
    assert i.do_localisation == "Yes"
    assert i.noise_level == pytest.approx(0.05)


def test_inversion_params_full():
    """Test inversion params full."""
    i = InversionParams(
        Termm=10,
        Do_parametrisation="Yes",
        Do_param_method="VAE",
        do_localisation="No",
        size1=8,
        size2=8,
        noise_level=0.10,
        Deccor="Yes",
    )
    assert i.Do_parametrisation == "Yes"
    assert i.Do_param_method == "VAE"
    assert i.size1 == 8 and i.size2 == 8


# ---------------------------------------------------------------------------
# TimeArrays / FlowArrays
# ---------------------------------------------------------------------------


def test_time_arrays_full():
    """Test time arrays full."""
    t = TimeArrays(Time=[0.0, 1.0], Time_unie1=[0.0, 1.0], timestep=1.0)
    assert t.timestep == pytest.approx(1.0)


def test_flow_arrays_full():
    """Test flow arrays full."""
    f = FlowArrays(awater=[1.0], agas=[2.0], aoil=[3.0], aqq=[6.0])
    assert f.aqq == [6.0]


# ---------------------------------------------------------------------------
# EnsembleRuntime / PriorEnsembles (added by the setup_models_and_data
# refactor)
# ---------------------------------------------------------------------------


def test_ensemble_runtime_defaults():
    """Test ensemble runtime defaults."""
    r = EnsembleRuntime()
    assert r.cfg is None and r.dist is None and r.device is None
    assert r.oldfolder == ""
    assert r.DEFAULT == "Yes"
    assert r.excel == 0
    assert r.TEMPLATEFILE is None


def test_ensemble_runtime_full():
    """Test ensemble runtime full."""
    r = EnsembleRuntime(
        cfg=object(),
        dist=object(),
        device="cuda",
        oldfolder="/tmp/run",
        DEFAULT="No",
        excel=1,
        TEMPLATEFILE={"key": "val"},
    )
    assert r.DEFAULT == "No"
    assert r.excel == 1
    assert r.TEMPLATEFILE == {"key": "val"}


def test_prior_ensembles_defaults_and_full():
    """Test prior ensembles defaults and full."""
    p = PriorEnsembles()
    assert p.perm is None and p.poro is None and p.fault is None
    p = PriorEnsembles(perm=[1.0], poro=[2.0], fault=[3.0])
    assert p.perm == [1.0] and p.poro == [2.0] and p.fault == [3.0]
