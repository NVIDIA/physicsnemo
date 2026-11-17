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

"""Exercise the documented cache workflow with real, small CPU datasets."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from physicsnemo.datapipes.cae.domino_datapipe import (
    CachedDoMINODataset,
    create_domino_dataset,
)

SRC = Path(__file__).parents[1] / "src"


@pytest.fixture
def cache_config(tmp_path, monkeypatch):
    """Create a tetrahedral CFD sample and its structured scaling factors."""
    monkeypatch.syspath_prepend(str(SRC))
    # CPU-only PyTorch builds omit NVTX; only profiling annotations are stubbed.
    if torch.version.cuda is None:
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *args: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    with initialize_config_dir(version_base="1.3", config_dir=str(SRC / "conf")):
        config = compose(config_name="config")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    vertices = np.array(
        [[-0.4, -0.4, -0.4], [0.4, -0.4, -0.4], [0.0, 0.4, -0.4], [0.0, 0.0, 0.4]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(cross, axis=1)
    data = {
        "stl_coordinates": vertices,
        "stl_faces": faces.flatten(),
        "stl_centers": triangles.mean(axis=1),
        "stl_areas": areas / 2,
        "surface_mesh_centers": triangles.mean(axis=1),
        "surface_normals": cross / areas[:, None],
        "surface_areas": areas / 2,
        "surface_fields": np.arange(16, dtype=np.float32).reshape(4, 4) / 10,
        "volume_mesh_centers": np.array(
            [[0, 0, 0], [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.1, -0.1, 0]],
            dtype=np.float32,
        ),
        "volume_fields": np.arange(20, dtype=np.float32).reshape(4, 5) / 10,
    }
    np.save(raw_dir / "tetra.npy", data)
    config.project_dir = str(tmp_path)
    config.data_processor.output_dir = str(raw_dir)
    config.data_processor.cached_dir = str(tmp_path / "cache")
    config.data_processor.use_cache = True
    config.data.input_dir = str(raw_dir)
    config.data.input_dir_val = str(raw_dir)
    config.data.gpu_preprocessing = False
    config.data.gpu_output = False
    config.data.volume_sample_from_disk = False
    config.data.bounding_box.min = [-1.0, -1.0, -1.0]
    config.data.bounding_box.max = [1.0, 1.0, 1.0]
    config.data.bounding_box_surface.min = [-1.0, -1.0, -1.0]
    config.data.bounding_box_surface.max = [1.0, 1.0, 1.0]
    config.model.interp_res = [3, 3, 3]
    config.model.volume_points_sample = 3
    config.model.surface_points_sample = 3
    config.model.geom_points_sample = 3
    config.model.num_neighbors_surface = 3

    ScalingFactors = importlib.import_module("utils").ScalingFactors
    fields = {key: data[key] for key in ["volume_fields", "surface_fields"]}
    factors = ScalingFactors(
        mean={key: value.mean(axis=0) for key, value in fields.items()},
        std={key: value.std(axis=0) for key, value in fields.items()},
        min_val={key: value.min(axis=0) for key, value in fields.items()},
        max_val={key: value.max(axis=0) for key, value in fields.items()},
        field_keys=list(fields),
    )
    factors.save(config.data.scaling_factors)
    return config


@pytest.mark.parametrize("model_type", ["volume", "surface", "combined"])
def test_cache_round_trip(cache_config, model_type):
    """Cache and validate actual preprocessing, then sample through the factory."""
    cache_config.model.model_type = model_type
    utils = importlib.import_module("utils")
    factors = utils.ScalingFactors.load(cache_config.data.scaling_factors)
    if model_type != "combined":
        unused_key = "surface_fields" if model_type == "volume" else "volume_fields"
        for attribute in ["mean", "std", "min_val", "max_val"]:
            getattr(factors, attribute).pop(unused_key)
        factors.field_keys.remove(unused_key)
        factors.save(cache_config.data.scaling_factors)
    importlib.import_module("cache_data").main.__wrapped__(cache_config)
    cached_files = list(Path(cache_config.data_processor.cached_dir).glob("*.npy"))
    assert len(cached_files) == 1
    importlib.import_module("validate_cache").main.__wrapped__(cache_config)

    cache_config.data.input_dir = cache_config.data_processor.cached_dir
    volume_factors, surface_factors = utils.load_scaling_factors(cache_config)
    dataset = create_domino_dataset(
        cache_config,
        phase="train",
        keys_to_read=[],
        keys_to_read_if_available={},
        vol_factors=volume_factors,
        surf_factors=surface_factors,
    )
    sample = dataset[0]
    assert sample["geometry_coordinates"].shape == (1, 3, 3)
    assert all(isinstance(value, torch.Tensor) for value in sample.values())
    assert all(value.device == torch.device("cpu") for value in sample.values())
    # The training script changes the backing dataset's indices every epoch.
    dataset.dataset.set_indices([])
    assert list(dataset) == []
    assert len(dataset) == 1
    dataset.dataset.set_indices([0])
    assert len(list(dataset)) == 1
    volume, surface = dataset.unscale_model_outputs(
        sample.get("volume_fields"), sample.get("surface_fields")
    )
    original = np.load(
        Path(cache_config.data_processor.output_dir, "tetra.npy"), allow_pickle=True
    ).item()
    if model_type in ["volume", "combined"]:
        assert sample["volume_fields"].shape == (1, 3, 5)
        assert torch.all(volume >= original["volume_fields"].min())
        assert torch.all(volume <= original["volume_fields"].max() + 1e-6)
    if model_type in ["surface", "combined"]:
        assert sample["surface_fields"].shape == (1, 3, 4)
        assert sample["surface_mesh_neighbors"].shape == (1, 3, 2, 3)
        assert "neighbor_indices" not in sample
        assert torch.all(surface >= original["surface_fields"].min())
        assert torch.all(surface <= original["surface_fields"].max() + 1e-6)


@pytest.mark.parametrize("corruption", ["missing", "shape", "values"])
def test_validation_rejects_corrupt_cache(cache_config, corruption):
    """Validation must fail on missing fields, incorrect shapes, or values."""
    importlib.import_module("cache_data").main.__wrapped__(cache_config)
    cache_file = next(Path(cache_config.data_processor.cached_dir).glob("*.npy"))
    data = np.load(cache_file, allow_pickle=True).item()
    if corruption == "missing":
        del data["volume_fields"]
    elif corruption == "shape":
        data["volume_fields"] = data["volume_fields"][:1]
    else:
        data["volume_fields"] += 1
    np.save(cache_file, data)
    with pytest.raises(ValueError, match="volume_fields"):
        importlib.import_module("validate_cache").main.__wrapped__(cache_config)


def test_validation_matches_source_filenames(cache_config, monkeypatch):
    """Different raw/cache iteration orders must not mix up CFD samples."""
    raw_dir = Path(cache_config.data_processor.output_dir)
    data = np.load(raw_dir / "tetra.npy", allow_pickle=True).item()
    data["volume_fields"] *= 2
    np.save(raw_dir / "second.npy", data)
    monkeypatch.setattr(
        np.random,
        "shuffle",
        lambda values: values.__setitem__(slice(None), values[::-1]),
    )
    importlib.import_module("cache_data").main.__wrapped__(cache_config)
    importlib.import_module("validate_cache").main.__wrapped__(cache_config)
    Path(cache_config.data_processor.cached_dir, "second.npy_cached.npy").unlink()
    with pytest.raises(FileNotFoundError, match="second.npy"):
        importlib.import_module("validate_cache").main.__wrapped__(cache_config)


def test_standalone_cached_dataset_keeps_unbatched_output(tmp_path, monkeypatch):
    """Existing standalone callers retain their unbatched cached sample shape."""
    if torch.version.cuda is None:
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *args: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    np.save(tmp_path / "legacy.npy", {"geometry_coordinates": np.ones((4, 3))})
    assert CachedDoMINODataset(tmp_path)[0]["geometry_coordinates"].shape == (4, 3)


def test_legacy_cached_neighbors(cache_config):
    """Legacy neighbor indices remain supported when resampling surface points."""
    cache_dir = Path(cache_config.data_processor.cached_dir)
    cache_dir.mkdir()
    centers = np.arange(18, dtype=np.float32).reshape(6, 3)
    np.save(
        cache_dir / "legacy.npy",
        {
            "surface_mesh_centers": centers,
            "surface_normals": centers + 1,
            "surface_areas": np.ones(6, dtype=np.float32),
            "surface_fields": centers[:, :2],
            "neighbor_indices": np.tile(np.array([0, 1]), (6, 1)),
        },
    )
    sample = CachedDoMINODataset(cache_dir, sampling=True, surface_points_sample=3)[0]
    assert sample["surface_mesh_neighbors"].shape == (3, 2, 3)
    assert torch.equal(
        sample["surface_mesh_neighbors"],
        torch.from_numpy(np.tile(centers[:2], (3, 1, 1))),
    )
    assert "neighbor_indices" not in sample


def test_cached_factory_has_stable_distributed_order(cache_config, monkeypatch):
    """Complementary sampler positions must select different files across ranks."""
    cache_dir = Path(cache_config.data_processor.cached_dir)
    cache_dir.mkdir()
    for filename, value in [("b.npy", 2.0), ("a.npy", 1.0)]:
        np.save(
            cache_dir / filename,
            {"volume_fields": np.full((4, 5), value, dtype=np.float32)},
        )
    cache_config.data.input_dir = str(cache_dir)
    kwargs = dict(
        phase="train",
        keys_to_read=[],
        keys_to_read_if_available={},
        vol_factors=None,
        surf_factors=None,
        sampling=False,
    )
    monkeypatch.setattr(np.random, "shuffle", lambda values: None)
    rank_zero = create_domino_dataset(cache_config, **kwargs)
    monkeypatch.setattr(
        np.random,
        "shuffle",
        lambda values: values.__setitem__(slice(None), values[::-1]),
    )
    rank_one = create_domino_dataset(cache_config, **kwargs)
    assert torch.equal(rank_zero[0]["volume_fields"], rank_one[0]["volume_fields"])
    rank_zero.dataset.set_indices([0])
    rank_one.dataset.set_indices([1])
    assert not torch.equal(
        next(iter(rank_zero))["volume_fields"], next(iter(rank_one))["volume_fields"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output requires a GPU")
def test_cached_factory_cuda_output(cache_config):
    """CPU-generated caches must transfer and unscale correctly on a CUDA model."""
    importlib.import_module("cache_data").main.__wrapped__(cache_config)
    cache_config.data.input_dir = cache_config.data_processor.cached_dir
    cache_config.data.gpu_output = True
    # Cached data has already been preprocessed on CPU; this flag should not
    # force normalization factors away from the CUDA output device.
    cache_config.data.gpu_preprocessing = False
    volume_factors, surface_factors = importlib.import_module(
        "utils"
    ).load_scaling_factors(cache_config)
    dataset = create_domino_dataset(
        cache_config,
        phase="train",
        keys_to_read=[],
        keys_to_read_if_available={},
        vol_factors=volume_factors,
        surf_factors=surface_factors,
    )
    sample = dataset[0]
    assert all(value.device.type == "cuda" for value in sample.values())
    volume, surface = dataset.unscale_model_outputs(
        sample["volume_fields"], sample["surface_fields"]
    )
    assert volume.device.type == surface.device.type == "cuda"
    assert torch.isfinite(volume).all() and torch.isfinite(surface).all()


@pytest.mark.parametrize("normalization", ["min_max_scaling", "mean_std_scaling"])
@pytest.mark.parametrize("model_type", ["volume", "surface", "combined"])
def test_nim_data_setup(cache_config, monkeypatch, model_type, normalization):
    """Build NIM training/validation data using its real example configuration."""
    nim_src = SRC.parents[1] / "domino_nim_finetuning" / "src"
    monkeypatch.syspath_prepend(str(nim_src))
    with initialize_config_dir(version_base="1.3", config_dir=str(nim_src / "conf")):
        config = compose(config_name="config")
    config.project_dir = cache_config.project_dir
    config.data.input_dir = cache_config.data.input_dir
    config.data.input_dir_val = cache_config.data.input_dir_val
    config.data.gpu_preprocessing = False
    config.data.gpu_output = False
    config.data.bounding_box = cache_config.data.bounding_box
    config.data.bounding_box_surface = cache_config.data.bounding_box_surface
    config.model.model_type = model_type
    config.model.normalization = normalization
    for key in [
        "interp_res",
        "volume_points_sample",
        "surface_points_sample",
        "geom_points_sample",
        "num_neighbors_surface",
    ]:
        config.model[key] = cache_config.model[key]
    module = importlib.import_module("domino_data")
    train, validation, volume_factors, surface_factors = module.prepare_datasets(config)
    assert len(train) == len(validation) == 1
    loader_options = {
        key: value
        for key, value in config.train.dataloader.items()
        if key != "preload_depth"
    }
    loader_options["batch_size"] = 2
    loader = torch.utils.data.DataLoader(
        train,
        sampler=[0, 0],
        collate_fn=module.collate_domino_samples,
        **loader_options,
    )
    batch = next(iter(loader))
    assert batch["geometry_coordinates"].shape == (2, 3, 3)
    if model_type in ["volume", "combined"]:
        assert volume_factors.shape == (2, 5)
        assert batch["volume_fields"].shape == (2, 3, 5)
        assert Path(config.project_dir, "volume_scaling_factors.npy").exists()
    else:
        assert volume_factors is None
    if model_type in ["surface", "combined"]:
        assert surface_factors.shape == (2, 4)
        assert batch["surface_fields"].shape == (2, 3, 4)
        assert Path(config.project_dir, "surface_scaling_factors.npy").exists()
    else:
        assert surface_factors is None
