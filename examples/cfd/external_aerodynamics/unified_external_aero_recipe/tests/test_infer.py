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

"""Unit tests for `src/infer.py` helpers and the shared utils/metrics helpers.

`infer.py` is tensorboard-free (it imports nothing from `train.py`), so it
imports directly with no skip guard. These pin the pure helpers -- field-type
resolution, re-dimensionalization, point-wise squeezing, sample-id derivation,
checkpoint-path resolution, and the DomainMesh write/round-trip -- none of
which need a model, checkpoint, or on-disk dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from conftest import make_surface_domain_mesh, make_volume_domain_mesh
from omegaconf import OmegaConf
from tensordict import TensorDict

import infer
from metrics import DEFAULT_METRICS, resolve_metrics
from nondim import NonDimensionalizeByMetadata, freestream_scales
from physicsnemo.mesh import DomainMesh
from utils import make_jsonl_logger, resolve_dict

_RECIPE = Path(__file__).resolve().parent.parent
_DATASETS = _RECIPE / "datasets"


### ---------------------------------------------------------------------------
### build_redim_field_types
### ---------------------------------------------------------------------------


def test_build_redim_field_types_surface():
    ds_yaml = infer.load_dataset_config(_DATASETS / "drivaer_ml_surface.yaml")
    assert infer.build_redim_field_types(ds_yaml) == {
        "pressure": "pressure",
        "wss": "stress",
    }


def test_build_redim_field_types_volume():
    ds_yaml = infer.load_dataset_config(_DATASETS / "drivaer_ml_volume.yaml")
    assert infer.build_redim_field_types(ds_yaml) == {
        "velocity": "velocity",
        "pressure": "pressure",
        "nut": "identity",
    }


### ---------------------------------------------------------------------------
### redimensionalize
### ---------------------------------------------------------------------------


def _freestream_td() -> TensorDict:
    return TensorDict(
        {
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(1.225),
            "p_inf": torch.tensor(101325.0),
        },
        batch_size=[],
    )


def test_redimensionalize_noop_without_transforms():
    td = TensorDict({"pressure": torch.randn(8)}, batch_size=[8])
    out = infer.redimensionalize(
        td, normalizer=None, nondim=None, field_types={}, global_data=_freestream_td()
    )
    assert torch.allclose(out["pressure"], td["pressure"].float())


def test_redimensionalize_inverts_nondim():
    gd = _freestream_td()
    q_inf, p_inf, _u, _rho, _t = freestream_scales(gd)
    cp = torch.tensor([0.0, 1.0, -2.0])
    td = TensorDict({"pressure": cp.clone()}, batch_size=[3])
    nondim = NonDimensionalizeByMetadata(fields={"pressure": "pressure"})
    out = infer.redimensionalize(
        td,
        normalizer=None,
        nondim=nondim,
        field_types={"pressure": "pressure"},
        global_data=gd,
    )
    # Cp -> p = Cp * q_inf + p_inf
    expected = cp * q_inf + p_inf
    assert torch.allclose(out["pressure"], expected, atol=1e-3)


### ---------------------------------------------------------------------------
### _to_pointwise
### ---------------------------------------------------------------------------


def test_to_pointwise_tensors_squeezes_batch():
    td = TensorDict(
        {"pressure": torch.randn(1, 5), "wss": torch.randn(1, 5, 3)}, batch_size=[1, 5]
    )
    out = infer._to_pointwise(td, "tensors")
    assert list(out.batch_size) == [5]
    assert out["wss"].shape == (5, 3)


def test_to_pointwise_mesh_passthrough():
    td = TensorDict({"pressure": torch.randn(5)}, batch_size=[5])
    out = infer._to_pointwise(td, "mesh")
    assert list(out.batch_size) == [5]


### ---------------------------------------------------------------------------
### _sample_id
### ---------------------------------------------------------------------------


def test_sample_id_from_pdmsh_path():
    md = {"source_path": "/data/case/geo_LHC001_AoA_4/domain_0.pdmsh"}
    assert infer._sample_id(md, 7) == "00007_geo_LHC001_AoA_4_domain_0"


def test_sample_id_surface_boundary_path():
    md = {"source_path": "/d/geo_X/run.pdmsh/_tensordict/boundaries/vehicle"}
    assert infer._sample_id(md, 0) == "00000_geo_X_run"


def test_sample_id_sanitizes_and_falls_back():
    # non-mesh path -> stem; spaces / specials sanitized to underscores
    assert infer._sample_id({"source_path": "/d/a b*c.txt"}, 3) == "00003_a_b_c"
    # no source_path -> index-only fallback
    assert infer._sample_id({}, 12) == "sample_00012"


### ---------------------------------------------------------------------------
### resolve_checkpoint_path
### ---------------------------------------------------------------------------


def test_resolve_checkpoint_path_explicit_wins():
    cfg = OmegaConf.create(
        {"checkpoint_path": "/abs/ckpts", "run_id": "r", "output_dir": "inference"}
    )
    assert infer.resolve_checkpoint_path(cfg) == "/abs/ckpts"


def test_resolve_checkpoint_path_from_run_id():
    cfg = OmegaConf.create(
        {
            "checkpoint_path": None,
            "run_id": "myrun",
            "checkpoint_dir": "runs",
            "output_dir": "inference",
        }
    )
    assert infer.resolve_checkpoint_path(cfg) == os.path.join(
        "runs", "myrun", "checkpoints"
    )


def test_resolve_checkpoint_path_falls_back_to_output_dir():
    cfg = OmegaConf.create(
        {
            "checkpoint_path": None,
            "run_id": "myrun",
            "checkpoint_dir": None,
            "output_dir": "out",
        }
    )
    assert infer.resolve_checkpoint_path(cfg) == os.path.join(
        "out", "myrun", "checkpoints"
    )


def test_resolve_checkpoint_path_requires_run_id_or_path():
    cfg = OmegaConf.create({"checkpoint_path": None, "run_id": None, "output_dir": "x"})
    with pytest.raises(ValueError):
        infer.resolve_checkpoint_path(cfg)


### ---------------------------------------------------------------------------
### attach_and_save (round-trip)
### ---------------------------------------------------------------------------


def test_attach_and_save_surface_roundtrip(tmp_path):
    targets = {"pressure": "scalar", "wss": "vector"}
    domain = make_surface_domain_mesh(targets, n_cells=16)
    phys = domain.interior.point_data.select("pressure", "wss")
    out_path = tmp_path / "s.pdmsh"
    infer.attach_and_save(domain, phys, phys, targets, out_path, rescale_geometry=False)

    keys = set(DomainMesh.load(str(out_path)).interior.point_data.keys())
    assert {"pred_pressure", "pred_wss", "true_pressure", "true_wss"} <= keys
    # training-space targets dropped (replaced by physical true_<name>)
    assert "pressure" not in keys and "wss" not in keys


def test_attach_and_save_keeps_non_target_inputs(tmp_path):
    targets = {"velocity": "vector", "pressure": "scalar", "nut": "scalar"}
    domain = make_volume_domain_mesh(targets, n_pts=64)
    phys = domain.interior.point_data.select(*targets)
    out_path = tmp_path / "v.pdmsh"
    infer.attach_and_save(domain, phys, phys, targets, out_path, rescale_geometry=False)

    keys = set(DomainMesh.load(str(out_path)).interior.point_data.keys())
    # non-target geometry inputs are preserved
    assert {"sdf", "sdf_normals"} <= keys
    assert {"pred_velocity", "true_nut"} <= keys


def test_attach_and_save_rescale_geometry_scales_points(tmp_path):
    targets = {"pressure": "scalar", "wss": "vector"}
    domain = make_surface_domain_mesh(targets, n_cells=16)  # global_data L_ref = 5.0
    l_ref = float(domain.global_data["L_ref"])
    orig_points = domain.interior.points.clone()
    phys = domain.interior.point_data.select("pressure", "wss")
    out_path = tmp_path / "r.pdmsh"
    infer.attach_and_save(domain, phys, phys, targets, out_path, rescale_geometry=True)

    reloaded = DomainMesh.load(str(out_path))
    assert torch.allclose(reloaded.interior.points, orig_points * l_ref, atol=1e-4)


### ---------------------------------------------------------------------------
### Shared utils / metrics helpers
### ---------------------------------------------------------------------------


def test_make_jsonl_logger_writes_timestamped_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    log = make_jsonl_logger(path)
    log({"phase": "summary", "value": 1.5})
    log({"phase": "sample", "step": 0})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["phase"] == "summary" and first["value"] == 1.5 and "ts" in first


def test_resolve_metrics_default_and_override():
    assert resolve_metrics(OmegaConf.create({"metrics": ["l2"]})) == ["l2"]
    assert resolve_metrics(OmegaConf.create({})) == list(DEFAULT_METRICS)


def test_resolve_dict():
    cfg = OmegaConf.create({"a": {"b": 1}, "empty": {}})
    assert resolve_dict(cfg, "a") == {"b": 1}
    assert resolve_dict(cfg, "empty") is None  # empty -> None
    assert resolve_dict(cfg, "missing") is None
