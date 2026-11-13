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

"""Native-domain sampling, validation isolation, normalization, and physical export."""

import json
from pathlib import Path
from types import SimpleNamespace

import datasets as datasets_module
import pytest
import torch
from collate import build_collate_fn
from datasets import (
    apply_model_config,
    build_dataloaders,
    build_dataset,
)
from forces import ForceContext
from forward_kwargs import walk_path
from hydra import compose, initialize_config_dir
from infer import (
    _to_pointwise,
    attach_and_save,
    build_redim_field_types,
    redimensionalize,
)
from omegaconf import OmegaConf

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.datapipes.readers import DomainMeshReader
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures

ROOT = Path(__file__).resolve().parents[1]
COUNTS = dict(
    geometry_points=17,
    num_supernodes=5,
    num_surface_anchors=7,
    num_volume_anchors=9,
    surface_queries=11,
    volume_queries=13,
)


COUNT_KEYS = {
    "geometry": "geometry_points",
    "supernodes": "num_supernodes",
    "surface_anchors": "num_surface_anchors",
    "volume_anchors": "num_volume_anchors",
    "surface_queries": "surface_queries",
    "interior": "volume_queries",
}


def make_reader(root, mode="joint", **counts):
    """Use the actual dataset selection schema with small test counts."""
    cfg = OmegaConf.load(ROOT / "datasets" / f"drivaer_ml_{mode}_point_sets.yaml")
    cfg.train_datadir = str(root)
    cfg.sampling_resolution = 13
    model_cfg = OmegaConf.load(ROOT / "conf" / "model" / f"ab_upt_{mode}.yaml")
    model_cfg.sampling_resolution = 13
    cfg = apply_model_config(cfg, model_cfg)
    specs = OmegaConf.to_container(cfg.pipeline.reader.point_sets, resolve=True)
    for name, spec in specs.items():
        spec["count"] = counts[
            "surface_queries"
            if name == "interior" and mode == "surface"
            else COUNT_KEYS[name]
        ]
    return DomainMeshReader(root, pattern="run_*/domain_*.pdmsh", point_sets=specs)


def anchor_queries(specs, enabled):
    """Select shared training indices or independent validation queries."""
    for name, anchors in (
        ("surface_queries", "surface_anchors"),
        (
            "interior",
            "volume_anchors" if "volume_anchors" in specs else "surface_anchors",
        ),
    ):
        if name in specs:
            if OmegaConf.is_config(specs):
                OmegaConf.update(
                    specs,
                    f"{name}.indices_from",
                    anchors if enabled else None,
                    force_add=True,
                )
            else:
                specs[name]["indices_from"] = anchors if enabled else None


@pytest.fixture
def native_case(tmp_path):
    """Write a real memory-mapped domain with distinct surface/volume labels."""
    torch.manual_seed(41)
    # Separate, consistently wound nondegenerate triangles.
    origins = torch.rand(40, 3)
    corners = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    surface = Mesh(
        points=(origins[:, None] + corners).reshape(-1, 3),
        cells=torch.arange(120).reshape(-1, 3),
        cell_data={
            "pMeanTrim": torch.arange(40).float() + 10,
            "wallShearStressMeanTrim": torch.randn(40, 3) * 0.001,
        },
    )
    interior = Mesh(
        points=torch.rand(61, 3),
        point_data={
            "pMeanTrim": torch.arange(61).float() + 100,
            "UMeanTrim": torch.randn(61, 3) + 30,
            "nutMeanTrim": torch.rand(61) * 0.001,
        },
    )
    domain = DomainMesh(
        interior=interior,
        boundaries={"vehicle": surface},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(1.2),
            "p_inf": torch.tensor(10.0),
            "L_ref": torch.tensor(4.0),
        },
    )
    path = tmp_path / "run_1" / "domain_1.pdmsh"
    path.parent.mkdir()
    domain.save(str(path))
    return tmp_path, domain


def pipeline(root, mode):
    """Compose actual model/dataset YAMLs with small point-set counts."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=[f"model=ab_upt_{mode}"])
    ds = OmegaConf.load(ROOT / "datasets" / f"drivaer_ml_{mode}_point_sets.yaml")
    ds.train_datadir = str(root)
    ds.sampling_resolution = 13
    for name, spec in cfg.dataset_reader_overrides.point_sets.items():
        spec.count = COUNTS[
            "surface_queries"
            if name == "interior" and mode == "surface"
            else COUNT_KEYS[name]
        ]
    ds = apply_model_config(ds, cfg)
    dataset = build_dataset(ds, device="cpu", num_workers=1)
    dataset.reader.set_generator(torch.Generator().manual_seed(123))
    return cfg, ds, dataset


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_native_pipeline_normalization_and_export(native_case, mode, tmp_path):
    """All modes preserve physical fields through normalization and export."""
    root, original = native_case
    if mode == "surface":
        # Exercise the actual recipe with no volume coordinates or labels on disk.
        root = tmp_path / "surface_only"
        path = root / "run_1" / "domain_1.pdmsh"
        path.parent.mkdir(parents=True)
        DomainMesh(
            interior=Mesh(points=torch.empty(0, 3)),
            boundaries={"vehicle": original.boundaries["vehicle"]},
            global_data=original.global_data,
        ).save(str(path))
    cfg, ds, dataset = pipeline(root, mode)
    domain, meta = dataset[0]
    assert ("surface_anchors" in domain.boundary_names) == (mode != "volume")
    assert ("volume_anchors" in domain.boundary_names) == (mode != "surface")
    if mode == "surface":
        assert domain.interior.n_points == COUNTS["surface_queries"]
        assert "volume_anchors" not in cfg.forward_kwargs
    if mode == "volume":
        assert "surface_anchors" not in cfg.forward_kwargs
    targets = OmegaConf.to_container(ds.targets)
    sources = (
        OmegaConf.to_container(ds.target_sources) if "target_sources" in ds else None
    )
    paths = sources or {name: f"interior.point_data.{name}" for name in targets}
    collate = build_collate_fn(
        "tensors", OmegaConf.to_container(cfg.forward_kwargs), targets, sources
    )
    batch = collate([(domain, meta)])
    normalizer = next(
        t
        for t in dataset.transforms
        if hasattr(t, "inverse_td") and hasattr(t, "_stats")
    )
    nondim = next(t for t in dataset.transforms if hasattr(t, "_fields"))
    field_types = build_redim_field_types(ds)
    true_pts = _to_pointwise(batch["targets"], "tensors")
    physical = redimensionalize(
        true_pts,
        normalizer=normalizer,
        nondim=nondim,
        field_types=field_types,
        global_data=domain.global_data,
        target_sources=sources,
    )
    # Check inversion against the selected original values, independent of transforms.
    raw, _ = dataset.reader[0]
    for name, source in paths.items():
        torch.testing.assert_close(
            physical[as_nested_key(name)], walk_path(raw, source), rtol=1e-5, atol=2e-5
        )
    output = tmp_path / f"export_{mode}.pdmsh"
    attach_and_save(
        domain,
        physical,
        physical,
        targets,
        output,
        rescale_geometry=True,
        target_sources=sources,
    )
    saved = DomainMesh.load(str(output))
    for name, source in paths.items():
        mesh_path, leaf = source.split(".point_data.")
        data = walk_path(saved, mesh_path).point_data
        parts = leaf.split(".")
        parts[-1] = "pred_" + parts[-1]
        torch.testing.assert_close(data[tuple(parts)], physical[as_nested_key(name)])
        assert as_nested_key(leaf) not in data
    torch.testing.assert_close(saved.interior.points, raw.interior.points)
    if mode != "volume":
        context = ForceContext.from_config(
            OmegaConf.create({"enabled": True}),
            field_types,
            "cpu",
            dataset_cfg=ds,
            target_sources=sources,
        )
        forces = context.coefficients(domain, true_pts, true_pts, normalizer, sources)
        assert forces is not None and forces[0] == forces[1]
        # Check the absolute signs against the canonical DrivAerML conventions;
        # matching prediction/reference alone misses a shared sign inversion.
        reference = ForceContext.from_config(
            OmegaConf.create({"enabled": True}),
            field_types,
            "cpu",
            dataset_cfg=OmegaConf.load(ROOT / "datasets" / "drivaer_ml_surface.yaml"),
            target_sources=sources,
        )
        assert forces == reference.coefficients(
            domain, true_pts, true_pts, normalizer, sources
        )
        assert any(value != 0 for value in forces[0].values())
    dataset.close()


def test_sampling_stream_independence(native_case):
    """Epoch changes train samples; query counts and order cannot change anchors."""
    root, _ = native_case
    reader = make_reader(root, **COUNTS)
    reader.set_generator(torch.Generator().manual_seed(37))
    first, _ = reader[0]
    again, _ = reader[0]
    reader.point_sets["surface_queries"]["count"] = 19
    reader.point_sets["interior"]["count"] = 23
    changed_queries, _ = reader[0]
    for name in (
        "geometry",
        "supernodes",
        "surface_anchors",
        "volume_anchors",
    ):
        torch.testing.assert_close(
            first.boundaries[name].points, again.boundaries[name].points
        )
        torch.testing.assert_close(
            first.boundaries[name].points, changed_queries.boundaries[name].points
        )
    reader.set_epoch(1)
    changed_epoch, _ = reader[0]
    assert not torch.equal(
        first.boundaries["geometry"].points,
        changed_epoch.boundaries["geometry"].points,
    )


@pytest.mark.parametrize(
    "mode,split",
    [
        ("surface", "manifest"),
        ("volume", "manifest"),
        ("joint", "manifest"),
        ("joint", "directory"),
        ("joint", "fallback"),
    ],
)
def test_anchor_training_loaders_keep_independent_validation_queries(
    native_case, mode, split, monkeypatch
):
    """Production train/inference loaders honor query counts during evaluation."""
    root, original = native_case
    cfg, ds, initial_dataset = pipeline(root, mode)
    initial_dataset.close()
    cfg.dataset = f"drivaer_ml_{mode}_point_sets"
    anchor_queries(cfg.dataset_reader_overrides.point_sets, True)
    cfg.dataloader.prefetch_factor = 0
    cfg.dataloader.num_workers = 1
    cfg.dataloader.use_streams = False
    cfg.dataloader.pin_memory = False
    if split == "manifest":
        path = root / "run_2" / "domain_2.pdmsh"
        path.parent.mkdir()
        original.save(str(path))
        (root / "manifest.json").write_text(
            json.dumps({"train": ["run_1"], "val": ["run_2"]})
        )
    else:
        cfg.train_split = cfg.val_split = None
        if split == "directory":
            val_root = root / "validation"
            path = val_root / "run_2" / "domain_2.pdmsh"
            path.parent.mkdir(parents=True)
            original.save(str(path))
            ds.val_datadir = str(val_root)
    monkeypatch.setattr(datasets_module, "load_dataset_config", lambda _: ds)
    monkeypatch.setattr(
        datasets_module,
        "DistributedManager",
        lambda: SimpleNamespace(world_size=1, rank=0),
    )
    train_loader, val_loader, _, _ = build_dataloaders(cfg)
    try:
        train = next(iter(train_loader))["forward_kwargs"]
        val = next(iter(val_loader))["forward_kwargs"]
        branches = ["surface", "volume"] if mode == "joint" else [mode]
        for branch in branches:
            anchors, queries = f"{branch}_anchors", f"{branch}_queries"
            torch.testing.assert_close(train[queries], train[anchors])
            assert val[queries].shape[1] == COUNTS[queries]
            assert val[anchors].shape[1] == COUNTS[f"num_{branch}_anchors"]
            assert val[queries].shape[1] != val[anchors].shape[1]
        train_loader.set_epoch(4)
        later_val = next(iter(val_loader))["forward_kwargs"]
        for key in val:
            torch.testing.assert_close(later_val[key], val[key])
    finally:
        train_loader.dataset.close()
        val_loader.dataset.close()


def test_surface_integration_measures(native_case):
    """Equal-area sampling recovers the complete surface integration measure."""
    root, original = native_case
    reader = make_reader(root, **COUNTS)
    sampled, _ = reader[0]
    torch.testing.assert_close(
        cell_measures(sampled.boundaries["vehicle"]).sum(),
        cell_measures(original.boundaries["vehicle"]).sum(),
    )


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_reader_keeps_labels_aligned_with_original_coordinates(native_case, mode):
    """Match sampled coordinates to the original mesh, independently of the reader."""
    root, original = native_case
    reader = make_reader(root, mode=mode, **COUNTS)
    reader.set_generator(torch.Generator().manual_seed(654))
    sampled, _ = reader[0]
    surface = (
        sampled.interior
        if mode == "surface"
        else sampled.boundaries.get("surface_queries")
    )
    if mode != "volume":
        torch.testing.assert_close(
            surface.points, sampled.boundaries["vehicle"].cell_centroids
        )
    else:
        assert surface is None
    for enabled, query_mesh, source_positions, source_data, fields in (
        (
            mode != "volume",
            surface,
            original.boundaries["vehicle"].cell_centroids,
            original.boundaries["vehicle"].cell_data,
            (
                ("surface.pressure", "pMeanTrim"),
                ("surface.wss", "wallShearStressMeanTrim"),
            ),
        ),
        (
            mode != "surface",
            sampled.interior if mode != "surface" else None,
            original.interior.points,
            original.interior.point_data,
            (
                ("volume.velocity", "UMeanTrim"),
                ("volume.pressure", "pMeanTrim"),
                ("volume.nut", "nutMeanTrim"),
            ),
        ),
    ):
        if not enabled:
            assert query_mesh is None or query_mesh.n_points == 0
            continue
        distances = torch.cdist(query_mesh.points.double(), source_positions.double())
        nearest = distances.argmin(dim=1)
        torch.testing.assert_close(query_mesh.points, source_positions[nearest])
        assert nearest.unique().numel() == query_mesh.n_points
        for target, source in fields:
            torch.testing.assert_close(
                query_mesh.point_data[as_nested_key(target.split(".")[-1])],
                source_data[source][nearest],
            )


@pytest.mark.parametrize("mode", ["surface", "volume", "joint"])
def test_anchor_supervision_keeps_labels_aligned(native_case, mode):
    """Anchor supervision selects the same coordinates and labels."""
    root, original = native_case
    _, _, dataset = pipeline(root, mode)
    anchor_queries(dataset.reader.point_sets, True)
    raw, _ = dataset.reader[0]
    for field, enabled, query_mesh, original_mesh in (
        (
            "surface",
            mode != "volume",
            raw.interior
            if mode == "surface"
            else raw.boundaries.get("surface_queries"),
            original.boundaries["vehicle"],
        ),
        ("volume", mode != "surface", raw.interior, original.interior),
    ):
        if not enabled:
            continue
        torch.testing.assert_close(
            query_mesh.points, raw.boundaries[f"{field}_anchors"].points
        )
        source_points = (
            original_mesh.cell_centroids if field == "surface" else original_mesh.points
        )
        source_data = (
            original_mesh.cell_data if field == "surface" else original_mesh.point_data
        )
        nearest = torch.cdist(query_mesh.points, source_points).argmin(dim=1)
        torch.testing.assert_close(query_mesh.points, source_points[nearest])
        torch.testing.assert_close(
            query_mesh.point_data["pressure"],
            source_data["pMeanTrim"][nearest],
        )
    dataset.close()
