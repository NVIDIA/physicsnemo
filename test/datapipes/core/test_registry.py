# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for the component registry and Hydra integration."""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp
from physicsnemo.datapipes.registry import (
    COMPONENT_REGISTRY,
    ComponentRegistry,
    _resolve_component,
    register,
)

# ============================================================================
# ComponentRegistry Basic Tests
# ============================================================================


def test_registry_init():
    """Test registry initialization."""
    registry = ComponentRegistry("test")
    assert registry.name == "test"
    assert len(registry) == 0
    assert registry.list() == []


def test_register_class():
    """Test registering a class."""
    registry = ComponentRegistry("test")

    @registry.register()
    class MyClass:
        pass

    assert "MyClass" in registry
    assert len(registry) == 1
    assert registry.get("MyClass") is MyClass


def test_register_with_custom_name():
    """Test registering with a custom name."""
    registry = ComponentRegistry("test")

    @registry.register("custom_name")
    class MyClass:
        pass

    assert "custom_name" in registry
    assert "MyClass" not in registry
    assert registry.get("custom_name") is MyClass


def test_register_duplicate_raises():
    """Test that registering duplicate names raises."""
    registry = ComponentRegistry("test")

    @registry.register()
    class MyClass:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register("MyClass")
        class AnotherClass:
            pass


def test_get_unregistered_raises():
    """Test that getting an unregistered name raises."""
    registry = ComponentRegistry("test")

    with pytest.raises(KeyError, match="not found"):
        registry.get("NonExistent")


def test_list_returns_sorted():
    """Test that list returns sorted names."""
    registry = ComponentRegistry("test")

    @registry.register("zebra")
    class A:
        pass

    @registry.register("alpha")
    class B:
        pass

    @registry.register("middle")
    class C:
        pass

    assert registry.list() == ["alpha", "middle", "zebra"]


def test_contains():
    """Test __contains__ method."""
    registry = ComponentRegistry("test")

    @registry.register()
    class MyClass:
        pass

    assert "MyClass" in registry
    assert "NonExistent" not in registry


def test_repr():
    """Test __repr__ method."""
    registry = ComponentRegistry("test")
    assert repr(registry) == "ComponentRegistry('test', count=0)"

    @registry.register()
    class MyClass:
        pass

    assert repr(registry) == "ComponentRegistry('test', count=1)"


# ============================================================================
# Global Registry Tests
# ============================================================================


def test_transforms_registered():
    """Test that all transforms are registered."""
    expected_transforms = [
        "BoundingBoxFilter",
        "BroadcastGlobalFeatures",
        "CenterOfMass",
        "Compose",
        "ComputeNormals",
        "ComputeSDF",
        "ConcatFields",
        "ConstantField",
        "CreateGrid",
        "FieldSlice",
        "KNearestNeighbors",
        "Normalize",
        "NormalizeVectors",
        "Purge",
        "Rename",
        "Scale",
        "SubsamplePoints",
        "Translate",
    ]

    for name in expected_transforms:
        assert name in COMPONENT_REGISTRY, f"Transform {name} not registered"


def test_readers_registered():
    """Test that all readers are registered."""
    expected_readers = [
        "HDF5Reader",
        "NumpyReader",
        "ZarrReader",
        "VTKReader",
        "TensorStoreZarrReader",
    ]

    for name in expected_readers:
        assert name in COMPONENT_REGISTRY, f"Reader {name} not registered"


def test_registry_count():
    """Test that the expected number of components are registered."""
    # At minimum, we expect all transforms + readers
    # This may grow as more components are added
    assert len(COMPONENT_REGISTRY) >= 24  # 19 transforms + 5 readers


# ============================================================================
# OmegaConf Resolver Tests
# ============================================================================


def test_resolve_transform():
    """Test resolving a transform name to full path."""
    result = _resolve_component("Normalize")
    assert result == "physicsnemo.datapipes.transforms.normalize.Normalize"


def test_resolve_reader():
    """Test resolving a reader name to full path."""
    result = _resolve_component("HDF5Reader")
    assert result == "physicsnemo.datapipes.readers.hdf5.HDF5Reader"


def test_resolve_unknown_raises():
    """Test that resolving unknown name raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        _resolve_component("NonExistentComponent")


def test_omegaconf_resolver_registered():
    """Test that the 'dp' resolver is registered with OmegaConf."""
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    # Create a config with the resolver
    cfg = OmegaConf.create({"_target_": "${dp:Normalize}"})

    # Resolve should work
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert (
        resolved["_target_"] == "physicsnemo.datapipes.transforms.normalize.Normalize"
    )


def test_omegaconf_resolver_with_reader():
    """Test OmegaConf resolver with a reader."""
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create({"_target_": "${dp:ZarrReader}"})
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["_target_"] == "physicsnemo.datapipes.readers.zarr.ZarrReader"


# ============================================================================
# Hydra Instantiation Tests
# ============================================================================


def test_instantiate_normalize():
    """Test instantiating Normalize transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Normalize}",
            "_convert_": "all",
            "input_keys": ["pressure"],
            "method": "mean_std",
            "means": {"pressure": 0.0},
            "stds": {"pressure": 1.0},
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.Normalize)
    assert transform.input_keys == ["pressure"]
    assert transform.method == "mean_std"


def test_instantiate_subsample_points():
    """Test instantiating SubsamplePoints transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:SubsamplePoints}",
            "_convert_": "all",
            "input_keys": ["points", "features"],
            "n_points": 1000,
            "algorithm": "uniform",
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.SubsamplePoints)
    assert transform.input_keys == ["points", "features"]
    assert transform.n_points == 1000
    assert transform.algorithm == "uniform"


def test_instantiate_compose():
    """Test instantiating Compose transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Compose}",
            "_convert_": "all",
            "transforms": [
                {
                    "_target_": "${dp:Normalize}",
                    "_convert_": "all",
                    "input_keys": ["x"],
                    "method": "mean_std",
                    "means": {"x": 0.0},
                    "stds": {"x": 1.0},
                },
                {
                    "_target_": "${dp:SubsamplePoints}",
                    "_convert_": "all",
                    "input_keys": ["x"],
                    "n_points": 100,
                },
            ],
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.Compose)
    assert len(transform) == 2
    assert isinstance(transform[0], dp.Normalize)
    assert isinstance(transform[1], dp.SubsamplePoints)


def test_instantiate_center_of_mass():
    """Test instantiating CenterOfMass transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:CenterOfMass}",
            "_convert_": "all",
            "coords_key": "positions",
            "areas_key": "areas",
            "output_key": "center",
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.CenterOfMass)


def test_instantiate_rename():
    """Test instantiating Rename transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Rename}",
            "_convert_": "all",
            "mapping": {"old_key": "new_key"},
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.Rename)


def test_instantiate_purge():
    """Test instantiating Purge transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Purge}",
            "_convert_": "all",
            "keep_only": ["field_a", "field_b"],
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.Purge)


def test_instantiate_concat_fields():
    """Test instantiating ConcatFields transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:ConcatFields}",
            "_convert_": "all",
            "input_keys": ["field_a", "field_b"],
            "output_key": "combined",
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.ConcatFields)


def test_instantiate_field_slice():
    """Test instantiating FieldSlice transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:FieldSlice}",
            "_convert_": "all",
            "slicing": {
                "features": {
                    "-1": [0, 1, 2],
                },
            },
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.FieldSlice)


def test_instantiate_constant_field():
    """Test instantiating ConstantField transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:ConstantField}",
            "_convert_": "all",
            "output_key": "zeros",
            "reference_key": "positions",
            "fill_value": 0.0,
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.ConstantField)


def test_instantiate_bounding_box_filter():
    """Test instantiating BoundingBoxFilter transform via Hydra."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:BoundingBoxFilter}",
            "_convert_": "all",
            "input_keys": ["positions"],
            "dependent_keys": ["features"],
            "bbox_min": [-1.0, -1.0, -1.0],
            "bbox_max": [1.0, 1.0, 1.0],
        }
    )

    transform = hydra.utils.instantiate(cfg)

    assert isinstance(transform, dp.BoundingBoxFilter)


# ============================================================================
# End-to-End Hydra Tests
# ============================================================================


def test_normalize_from_hydra_applies_correctly():
    """Test that a Hydra-instantiated Normalize works correctly."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Normalize}",
            "_convert_": "all",
            "input_keys": ["values"],
            "method": "mean_std",
            "means": {"values": 10.0},
            "stds": {"values": 5.0},
        }
    )

    transform = hydra.utils.instantiate(cfg)

    # Apply to data
    data = TensorDict({"values": torch.tensor([5.0, 10.0, 15.0, 20.0])})
    result = transform(data)

    expected = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    torch.testing.assert_close(result["values"], expected, atol=1e-6, rtol=1e-6)


def test_subsample_from_hydra_applies_correctly():
    """Test that a Hydra-instantiated SubsamplePoints works correctly."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:SubsamplePoints}",
            "_convert_": "all",
            "input_keys": ["points"],
            "n_points": 50,
            "algorithm": "uniform",
        }
    )

    transform = hydra.utils.instantiate(cfg)

    # Apply to data
    data = TensorDict({"points": torch.randn(1000, 3)})
    result = transform(data)

    assert result["points"].shape == (50, 3)


def test_compose_pipeline_from_hydra():
    """Test a complete pipeline instantiated from Hydra config."""
    hydra = pytest.importorskip("hydra")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "${dp:Compose}",
            "_convert_": "all",
            "transforms": [
                {
                    "_target_": "${dp:Normalize}",
                    "_convert_": "all",
                    "input_keys": ["features"],
                    "method": "mean_std",
                    "means": {"features": 0.0},
                    "stds": {"features": 1.0},
                },
                {
                    "_target_": "${dp:SubsamplePoints}",
                    "_convert_": "all",
                    "input_keys": ["points", "features"],
                    "n_points": 100,
                },
                {
                    "_target_": "${dp:Rename}",
                    "_convert_": "all",
                    "mapping": {"points": "positions"},
                },
            ],
        }
    )

    pipeline = hydra.utils.instantiate(cfg)

    # Apply to data
    data = TensorDict(
        {
            "points": torch.randn(500, 3),
            "features": torch.randn(500, 8),
        }
    )

    result = pipeline(data)

    # Check results
    assert "positions" in result.keys()
    assert "points" not in result.keys()
    assert result["positions"].shape == (100, 3)
    assert result["features"].shape == (100, 8)


# ============================================================================
# Register Decorator Tests
# ============================================================================


def test_register_adds_to_global_registry():
    """Test that @register() adds to COMPONENT_REGISTRY."""
    # Note: We can't easily test this without polluting the global registry
    # So we just verify the decorator returns the class unchanged
    initial_count = len(COMPONENT_REGISTRY)

    # The decorator should return the class unchanged
    # We'll test with a unique name to avoid conflicts

    @register("_TestUniqueClass_12345")
    class TestClass:
        pass

    assert "_TestUniqueClass_12345" in COMPONENT_REGISTRY
    assert len(COMPONENT_REGISTRY) == initial_count + 1
    assert COMPONENT_REGISTRY.get("_TestUniqueClass_12345") is TestClass
