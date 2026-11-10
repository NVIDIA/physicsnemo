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

"""SMART integration regressions: full geometry, independent RNG and recipe I/O."""

from pathlib import Path

import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import build_dataset, load_dataset_config
from hydra import compose, initialize_config_dir
from loss import LossCalculator
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict
from utils import field_dim

from physicsnemo import Module
from physicsnemo.mesh import DomainMesh, Mesh

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_cases(tmp_path):
    """Small CFD domains with complete surface topology and physical fields."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
            [4, 6, 5],
            [4, 5, 7],
            [4, 7, 6],
            [5, 6, 7],
        ]
    )
    for i in range(3):
        body = Mesh(
            points=points + i * 0.1,
            cells=faces,
            cell_data={
                "pMeanTrim": torch.arange(8.0).unsqueeze(-1),
                "wallShearStressMeanTrim": torch.arange(8.0).unsqueeze(-1).expand(-1, 3)
                * 0.01,
            },
        )
        gen = torch.Generator().manual_seed(i)
        volume = Mesh(
            points=torch.rand(48, 3, generator=gen) * 3 + 0.01,
            point_data={
                "pMeanTrim": torch.arange(48.0).unsqueeze(-1),
                "UMeanTrim": torch.rand(48, 3, generator=gen),
                "nutMeanTrim": torch.ones(48, 1) * 0.001,
            },
        )
        domain = DomainMesh(
            interior=volume,
            boundaries={"vehicle": body},
            global_data={
                "U_inf": torch.tensor([30.0, 0.0, 0.0]),
                "p_inf": torch.tensor(0.0),
                "rho_inf": torch.tensor(1.2),
                "L_ref": torch.tensor(5.0),
                "nu": torch.tensor(1e-5),
            },
        )
        domain.save(tmp_path / f"run_{i}/domain_{i}.pdmsh")
    return tmp_path


def dataset_cfg(path, field, n=4):
    """Resolve the real SMART pipeline against a temporary CFD dataset."""
    cfg = load_dataset_config(ROOT / f"datasets/drivaer_ml_smart_{field}.yaml")
    return OmegaConf.merge(
        cfg, {"dataset_paths": {"drivaer_ml": str(path)}, "sampling_resolution": n}
    )


def model_cfg(field):
    """Compose the public model template at tiny test dimensions."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"model=smart_{field}", f"dataset=drivaer_ml_smart_{field}"],
        )
    cfg.model.hidden_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 2
    cfg.model.num_latents = 3
    cfg.model.geometry_subsample = 4
    cfg.model.eval_chunk_size = 2
    OmegaConf.update(cfg, "out_dim", 4 if field == "surface" else 5, force_add=True)
    return cfg


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_pipeline_geometry_alignment(native_cases, field):
    """Keep full geometry and aligned targets through query sampling."""
    cfg = dataset_cfg(native_cases, field)
    ds = build_dataset(cfg, device="cpu", num_workers=1)
    ds.set_generator(torch.Generator().manual_seed(42))
    ds.set_epoch(3)
    sample, _ = ds[0]
    assert sample.interior.n_points == 4
    assert sample.global_data["geometry_points"].shape[0] == 8
    # Sampling selects coordinates and target rows together.
    if field == "surface":
        torch.testing.assert_close(
            sample.interior.points, sample.boundaries["vehicle"].cell_centroids
        )
        from datasets import find_normalizer

        unscaled = find_normalizer([ds]).inverse_td(sample.interior.point_data)
        pressure, shear = unscaled["pressure"], unscaled["wss"]
        torch.testing.assert_close(
            shear[:, 0], pressure.flatten() * 0.01, atol=1e-8, rtol=1e-5
        )
        assert "pressure" not in sample.boundaries["vehicle"].cell_data
    else:
        assert sample.boundaries["vehicle"].n_cells == 8
        assert "stl_geometry" not in sample.boundaries
        assert sample.interior.point_data["sdf"].shape == (4, 1)
    raw = DomainMesh.load(native_cases / "run_0/domain_0.pdmsh")
    torch.testing.assert_close(
        sample.global_data["geometry_points"], raw.boundaries["vehicle"].cell_centroids
    )
    torch.testing.assert_close(
        sample.global_data["parameters"], torch.tensor([1.205, 30.0])
    )
    torch.testing.assert_close(sample.global_data["U_inf"], raw.global_data["U_inf"])
    ds.close()


@pytest.mark.parametrize("field", ["surface", "volume"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_pipeline_forward_backward_checkpoint_and_chunking(
    native_cases, field, tmp_path, device
):
    """Exercise both heads through the native pipeline and checkpoint contract."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ds = build_dataset(dataset_cfg(native_cases, field), device="cpu", num_workers=1)
    ds.set_generator(torch.Generator().manual_seed(42))
    sample = ds[0]
    cfg = model_cfg(field)
    targets = OmegaConf.to_container(dataset_cfg(native_cases, field).targets)
    collate = build_collate_fn(
        "tensors", OmegaConf.to_container(cfg.forward_kwargs), targets
    )
    from utils import recursive_to_device

    batch = recursive_to_device(collate([sample]), device)
    inputs = batch["forward_kwargs"]
    model = hydra.utils.instantiate(cfg.model).to(device)
    output = model(**inputs)
    assert output.shape == (1, 4, sum(field_dim(t) for t in targets.values()))
    loss = LossCalculator(target_config=targets, loss_type="mse")
    result = loss(
        normalize_output_to_tensordict(output, targets, "tensors"), batch["targets"]
    )
    from datasets import find_normalizer
    from metrics import MetricCalculator

    normalizer = find_normalizer([ds]).to(device)
    prediction = normalize_output_to_tensordict(output, targets, "tensors")
    before = batch["targets"].clone()
    measured = MetricCalculator(targets, metrics=["l2"], normalizer=normalizer)(
        prediction, batch["targets"]
    )
    expected = MetricCalculator(targets, metrics=["l2"])(
        normalizer.inverse_td(prediction), normalizer.inverse_td(batch["targets"])
    )
    for name in expected.keys():
        torch.testing.assert_close(measured[name], expected[name])
    for name in before.keys():
        torch.testing.assert_close(batch["targets"][name], before[name], rtol=0, atol=0)
    total = result[0]
    total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    model.eval()
    chunked = model(**inputs)
    model.eval_chunk_size = 100
    torch.testing.assert_close(model(**inputs), chunked, rtol=2e-5, atol=2e-6)
    path = tmp_path / f"{field}.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).to(device).eval()
    torch.testing.assert_close(restored(**inputs), chunked, rtol=2e-5, atol=2e-6)
    ds.close()


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_full_query_mode_and_sampling_budgets(native_cases, field):
    """Changing the query budget preserves full geometry."""
    small = build_dataset(
        dataset_cfg(native_cases, field, 3), device="cpu", num_workers=1
    )
    full = build_dataset(
        dataset_cfg(native_cases, field, 1000), device="cpu", num_workers=1
    )
    for ds in [small, full]:
        ds.set_generator(torch.Generator().manual_seed(9))
    a, _ = small[0]
    b, _ = full[0]
    assert b.interior.n_points == (8 if field == "surface" else 48)
    assert (
        a.global_data["geometry_points"].shape[0]
        == b.global_data["geometry_points"].shape[0]
        == 8
    )
    torch.testing.assert_close(
        a.global_data["geometry_points"],
        b.global_data["geometry_points"],
        rtol=0,
        atol=0,
    )
    for ds in [small, full]:
        ds.close()


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_loader_validation_is_independent_of_training(native_cases, field, monkeypatch):
    """Manifest readers must not share mutable sampling epochs across splits."""
    import json
    from types import SimpleNamespace

    import datasets as ds_module

    (native_cases / "manifest.json").write_text(
        json.dumps({"train": ["run_0", "run_1"], "val": ["run_2"]})
    )
    original = ds_module.load_dataset_config
    monkeypatch.setattr(
        ds_module,
        "load_dataset_config",
        lambda path: OmegaConf.merge(
            original(path), {"dataset_paths": {"drivaer_ml": str(native_cases)}}
        ),
    )
    cfg = model_cfg(field)
    cfg.sampling_resolution = 4
    cfg.dataloader.num_workers = 1
    cfg.dataloader.prefetch_factor = 0
    cfg.dataloader.use_streams = False
    cfg.dataloader.pin_memory = False
    monkeypatch.setattr(
        ds_module,
        "DistributedManager",
        lambda: SimpleNamespace(world_size=1, rank=0, distributed=False),
    )
    train, val, _, _ = ds_module.build_dataloaders(cfg)
    assert train.dataset is not val.dataset
    first = next(iter(val))
    train.set_epoch(8)
    for _ in train:
        pass
    assert train.dataset.reader._epoch == 8
    assert val.dataset.reader._epoch == 0
    second = next(iter(val))
    for key in ("geometry", "parameters", "frame_center"):
        torch.testing.assert_close(
            first["forward_kwargs"][key], second["forward_kwargs"][key], rtol=0, atol=0
        )
    train.dataset.close()
    val.dataset.close()


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_augmentation_keeps_geometry_and_features_aligned(native_cases, field):
    """Rotate normals and vector targets in the same frame as the geometry."""
    ds = build_dataset(
        dataset_cfg(native_cases, field), device="cpu", num_workers=1, augment=True
    )
    ds.set_generator(torch.Generator().manual_seed(9))
    domain, _ = ds[0]
    assert domain.global_data["geometry_points"].shape[0] == 8
    if field == "surface":
        torch.testing.assert_close(
            domain.interior.points, domain.boundaries["vehicle"].cell_centroids
        )
        torch.testing.assert_close(
            domain.boundaries["vehicle"].cell_data["normals"],
            domain.boundaries["vehicle"].cell_normals,
        )
    else:
        assert (
            domain.global_data["geometry_points"].shape
            == domain.boundaries["vehicle"].cell_centroids.shape
        )
        torch.testing.assert_close(
            domain.interior.point_data["sdf_normals"].norm(dim=-1), torch.ones(4)
        )
    # Retained raw coordinates must receive the same rotation as targets.
    if field == "surface":
        torch.testing.assert_close(
            domain.boundaries["vehicle"].cell_data["query_normals"],
            domain.boundaries["vehicle"].cell_normals,
        )
    ds.close()


def test_full_surface_force_and_export_contract(native_cases, tmp_path):
    """Use the unchanged force integrator and native prediction writer."""
    from forces import ForceContext
    from infer import attach_and_save

    # Assert the actual full-resolution mesh/point relation used by inference.
    ds = build_dataset(
        dataset_cfg(native_cases, "surface", 1000), device="cpu", num_workers=1
    )
    ds.set_generator(torch.Generator().manual_seed(0))
    domain, _ = ds[0]
    assert domain.interior.n_points == domain.boundaries["vehicle"].n_cells == 8
    torch.testing.assert_close(
        domain.interior.points, domain.boundaries["vehicle"].cell_centroids
    )
    from datasets import find_normalizer

    fields = {"pressure": "scalar", "wss": "vector"}
    normalizer = find_normalizer([ds])
    pred = domain.interior.point_data.select("pressure", "wss").clone()
    ctx = ForceContext.from_config(
        OmegaConf.create({"enabled": True, "reference_area": 2.17}),
        {"pressure": "pressure", "wss": "stress"},
        "cpu",
    )
    pair = ctx.coefficients(domain, pred, pred, normalizer)
    assert pair is not None and pair[0] == pair[1]
    assert all(torch.isfinite(torch.tensor(v)) for v in pair[0].values())
    physical = normalizer.inverse_td(pred)
    from nondim import NonDimensionalizeByMetadata

    physical = NonDimensionalizeByMetadata(
        fields={"pressure": "pressure", "wss": "stress"}
    ).inverse_td(
        physical,
        {"pressure": "pressure", "wss": "stress"},
        torch.tensor(540.0),
        torch.tensor(0.0),
        torch.tensor(30.0),
    )
    path = tmp_path / "prediction.pdmsh"
    attach_and_save(domain, physical, physical, fields, path, rescale_geometry=False)
    restored = DomainMesh.load(path)
    assert restored.interior.n_points == 8
    for field in fields:
        torch.testing.assert_close(
            restored.interior.point_data["pred_" + field], physical[field]
        )
    ds.close()
