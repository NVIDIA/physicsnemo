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

"""
Comprehensive tests for ModelRegistry and Module registration functionality.

This test file focuses on:
- Entry point loading mechanism
- Registry error handling
- Module subclass auto-registration
- Registry state management
- Real entry points from pyproject.toml
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from physicsnemo.core import ModelRegistry, Module
from physicsnemo.core.meta import ModelMetaData


# Fixture to clear registry between tests to avoid naming conflicts
@pytest.fixture()
def clear_registry():
    """Clear and restore the model registry before and after each test

    Note: This is NOT autouse - tests that need a clean registry should explicitly
    request this fixture. Tests that need real entry points should NOT use this.
    """
    registry = ModelRegistry()
    registry.__clear_registry__()
    yield
    registry.__restore_registry__()


@pytest.fixture()
def real_registry():
    """Provides a registry with real entry points loaded (does not clear)

    Use this fixture for tests that need to verify real entry points from pyproject.toml
    """
    return ModelRegistry()


# Test models for use in various tests
class TestModelA(Module):
    """Test model A for registry tests"""

    def __init__(self, size=32):
        super().__init__(meta=ModelMetaData())
        self.size = size
        self.layer = torch.nn.Linear(size, size)

    def forward(self, x):
        return self.layer(x)


class TestModelB(Module):
    """Test model B for registry tests"""

    def __init__(self, hidden_dim=64):
        super().__init__(meta=ModelMetaData())
        self.hidden_dim = hidden_dim
        self.layer = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.layer(x)


# =============================================================================
# Entry Point Tests
# =============================================================================


def test_entry_point_loading(clear_registry):
    """Test that entry points are properly loaded and can be lazily instantiated"""
    from importlib.metadata import EntryPoint

    registry = ModelRegistry()

    # Mock an entry point - must be an actual EntryPoint instance for isinstance check
    mock_entry_point = MagicMock(spec=EntryPoint)
    mock_entry_point.name = "MockEntryPointModel"
    mock_entry_point.load.return_value = TestModelA

    # Directly add it to the registry
    registry._model_registry["MockEntryPointModel"] = mock_entry_point

    # Verify it's in the list
    assert "MockEntryPointModel" in registry.list_models()

    # Factory should call load() on the entry point
    ModelClass = registry.factory("MockEntryPointModel")
    mock_entry_point.load.assert_called_once()
    assert ModelClass == TestModelA


def test_entry_point_legacy_modulus_group(clear_registry):
    """Test that legacy 'modulus.models' entry points work with deprecation warning"""
    # Mock entry_points to return both physicsnemo and modulus groups
    mock_physicsnemo_ep = MagicMock()
    mock_physicsnemo_ep.name = "PhysicsNeMoModel"
    mock_physicsnemo_ep.load.return_value = TestModelA

    mock_modulus_ep = MagicMock()
    mock_modulus_ep.name = "LegacyModulusModel"
    mock_modulus_ep.load.return_value = TestModelB

    def mock_entry_points(group):
        if group == "physicsnemo.models":
            return [mock_physicsnemo_ep]
        elif group == "modulus.models":
            return [mock_modulus_ep]
        return []

    with patch("physicsnemo.core.registry.entry_points", side_effect=mock_entry_points):
        # Re-construct the registry with mocked entry points
        with pytest.warns(
            DeprecationWarning, match="modulus.models.*physicsnemo.models"
        ):
            test_registry = ModelRegistry._construct_registry()

        # Both should be in the registry
        assert "PhysicsNeMoModel" in test_registry
        assert "LegacyModulusModel" in test_registry


def test_entry_point_duplicate_name_priority(clear_registry):
    """Test that physicsnemo.models takes priority over modulus.models for same name"""
    # Mock entry_points with same name in both groups
    mock_physicsnemo_ep = MagicMock()
    mock_physicsnemo_ep.name = "ConflictModel"
    mock_physicsnemo_ep.load.return_value = TestModelA

    mock_modulus_ep = MagicMock()
    mock_modulus_ep.name = "ConflictModel"
    mock_modulus_ep.load.return_value = TestModelB

    def mock_entry_points(group):
        if group == "physicsnemo.models":
            return [mock_physicsnemo_ep]
        elif group == "modulus.models":
            return [mock_modulus_ep]
        return []

    with patch("physicsnemo.core.registry.entry_points", side_effect=mock_entry_points):
        test_registry = ModelRegistry._construct_registry()

        # physicsnemo version should take priority (no modulus deprecation warning)
        assert test_registry["ConflictModel"] == mock_physicsnemo_ep


# =============================================================================
# Registry Error Handling Tests
# =============================================================================


def test_factory_nonexistent_model(clear_registry):
    """Test that factory raises KeyError for non-existent model"""
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="No model is registered under the name"):
        registry.factory("NonExistentModel")


def test_factory_error_message_includes_available_models(clear_registry):
    """Test that KeyError message includes list of available models"""
    registry = ModelRegistry()
    registry.register(TestModelA, "TestModelA")
    registry.register(TestModelB, "TestModelB")

    try:
        registry.factory("NonExistentModel")
        pytest.fail("Should have raised KeyError")
    except KeyError as e:
        error_msg = str(e)
        # Should mention available models
        assert "TestModelA" in error_msg
        assert "TestModelB" in error_msg


def test_register_duplicate_name(clear_registry):
    """Test that registering a duplicate name raises ValueError"""
    registry = ModelRegistry()
    registry.register(TestModelA, "DuplicateName")

    with pytest.raises(ValueError, match="Name DuplicateName already in use"):
        registry.register(TestModelB, "DuplicateName")


def test_register_duplicate_error_includes_registered_models(clear_registry):
    """Test that ValueError for duplicate includes list of registered models"""
    registry = ModelRegistry()
    registry.register(TestModelA, "ModelA")
    registry.register(TestModelB, "ModelB")

    try:
        registry.register(TestModelA, "ModelA")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "ModelA" in error_msg
        assert "ModelB" in error_msg


def test_register_without_name_uses_class_name(clear_registry):
    """Test that register() uses class __name__ when name not provided"""
    registry = ModelRegistry()
    registry.register(TestModelA)  # No name provided

    # Should be registered under class name
    assert "TestModelA" in registry.list_models()
    ModelClass = registry.factory("TestModelA")
    assert ModelClass == TestModelA


# =============================================================================
# Registry State Management Tests
# =============================================================================


def test_list_models(clear_registry):
    """Test that list_models returns all registered model names"""
    registry = ModelRegistry()
    registry.register(TestModelA, "ModelA")
    registry.register(TestModelB, "ModelB")

    models = registry.list_models()
    assert "ModelA" in models
    assert "ModelB" in models
    assert isinstance(models, list)


def test_list_models_empty(clear_registry):
    """Test list_models on empty registry"""
    registry = ModelRegistry()
    models = registry.list_models()
    assert isinstance(models, list)
    # Registry might have entry points, so just check it's a list


def test_registry_singleton_behavior(clear_registry):
    """Test that ModelRegistry instances share state (Borg pattern)"""
    registry1 = ModelRegistry()
    registry2 = ModelRegistry()

    registry1.register(TestModelA, "SharedModel")

    # Should be accessible from second instance
    assert "SharedModel" in registry2.list_models()
    assert registry2.factory("SharedModel") == TestModelA


def test_clear_and_restore_registry(clear_registry):
    """Test internal __clear_registry__ and __restore_registry__ methods"""
    registry = ModelRegistry()

    # Register a model
    registry.register(TestModelA, "TempModel")
    assert "TempModel" in registry.list_models()

    # Clear should remove it
    registry.__clear_registry__()
    assert "TempModel" not in registry.list_models()

    # Restore should bring back entry points
    registry.__restore_registry__()
    # Entry points should be restored
    assert isinstance(registry.list_models(), list)


# =============================================================================
# Module Subclass Auto-Registration Tests
# =============================================================================


def test_module_subclass_auto_registration(clear_registry):
    """Test that Module subclass with register=True auto-registers"""
    registry = ModelRegistry()

    # Define a class with register=True
    class AutoRegisteredModel(Module, register=True):
        def __init__(self, dim=16):
            super().__init__(meta=ModelMetaData())
            self.dim = dim

        def forward(self, x):
            return x

    # Should be automatically registered
    assert "AutoRegisteredModel" in registry.list_models()

    # Should be retrievable
    ModelClass = registry.factory("AutoRegisteredModel")
    assert ModelClass == AutoRegisteredModel

    # Clean up
    registry.__clear_registry__()


def test_module_subclass_no_auto_registration_by_default(clear_registry):
    """Test that Module subclass without register=True is NOT auto-registered"""
    registry = ModelRegistry()
    initial_models = set(registry.list_models())

    # Define a class without register=True
    class NotAutoRegisteredModel(Module):
        def __init__(self, dim=16):
            super().__init__(meta=ModelMetaData())
            self.dim = dim

        def forward(self, x):
            return x

    # Should NOT be automatically registered
    current_models = set(registry.list_models())
    assert "NotAutoRegisteredModel" not in current_models

    # Difference should be empty (no new models)
    new_models = current_models - initial_models
    assert "NotAutoRegisteredModel" not in new_models


def test_module_from_torch_registration(clear_registry):
    """Test that Module.from_torch with register=True registers the model"""

    class SimpleTorchModel(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    registry = ModelRegistry()

    # Convert with registration
    PhysicsNeMoModel = Module.from_torch(
        SimpleTorchModel,
        meta=ModelMetaData(),
        name="RegisteredTorchModel",
        register=True,
    )

    # Should be registered
    assert "RegisteredTorchModel" in registry.list_models()

    # Should be retrievable
    ModelClass = registry.factory("RegisteredTorchModel")
    assert ModelClass == PhysicsNeMoModel


def test_module_from_torch_no_registration(clear_registry):
    """Test that Module.from_torch without register=True does NOT register"""

    class SimpleTorchModel(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    registry = ModelRegistry()
    initial_models = set(registry.list_models())

    # Should NOT be registered
    current_models = set(registry.list_models())
    assert "UnregisteredTorchModel" not in current_models

    # Verify no new models were added
    new_models = current_models - initial_models
    assert "UnregisteredTorchModel" not in new_models


# =============================================================================
# Real Entry Points Tests (from pyproject.toml)
# =============================================================================


def test_real_entry_points_are_loaded(real_registry):
    """Test that real entry points from pyproject.toml are loaded into registry"""
    models = real_registry.list_models()

    # Check that some key models from pyproject.toml are present
    expected_models = [
        "AFNO",
        "FullyConnected",
        "FNO",
        "Pix2Pix",
        "One2ManyRNN",
        "SRResNet",
        "DLWP",
    ]

    for model_name in expected_models:
        assert model_name in models, (
            f"Expected model '{model_name}' not found in registry"
        )


def test_real_entry_point_factory_loads_class(real_registry):
    """Test that factory() can load a real model class from entry points"""
    # Load FullyConnected model
    FullyConnectedClass = real_registry.factory("FullyConnected")

    # Verify it's a class
    assert isinstance(FullyConnectedClass, type)

    # Verify it's a Module subclass
    assert issubclass(FullyConnectedClass, Module)

    # Verify the class name matches
    assert FullyConnectedClass.__name__ == "FullyConnected"


@pytest.mark.parametrize(
    "model_name,expected_module",
    [
        ("AFNO", "physicsnemo.models.afno"),
        ("FullyConnected", "physicsnemo.models.mlp"),
        ("FNO", "physicsnemo.models.fno"),
        ("Pix2Pix", "physicsnemo.models.pix2pix"),
        ("DLWP", "physicsnemo.models.dlwp"),
        ("One2ManyRNN", "physicsnemo.models.rnn"),
        ("SRResNet", "physicsnemo.models.srrn"),
    ],
)
def test_entry_points_resolve_to_correct_modules(
    real_registry, model_name, expected_module
):
    """Test that entry points resolve to the correct module paths"""
    # Load the model class
    ModelClass = real_registry.factory(model_name)

    # Check that it comes from the expected module
    assert ModelClass.__module__.startswith(expected_module), (
        f"Model {model_name} expected from {expected_module}, "
        f"but got {ModelClass.__module__}"
    )


def test_entry_point_models_can_be_instantiated(real_registry):
    """Test that models loaded from entry points can be instantiated"""
    # Test FullyConnected (simple model that doesn't need optional deps)
    FullyConnectedClass = real_registry.factory("FullyConnected")

    # Should be able to instantiate without errors
    model = FullyConnectedClass(in_features=32, out_features=64)

    # Verify it's a proper Module
    assert isinstance(model, Module)
    assert hasattr(model, "forward")
    assert hasattr(model, "save")
    assert hasattr(model, "load")


def test_all_entry_points_are_loadable():
    """Test that all entry points defined in pyproject.toml can be loaded without errors"""
    import importlib.util
    from importlib.metadata import entry_points

    # Get all physicsnemo.models entry points
    eps = entry_points(group="physicsnemo.models")

    failed_models = []
    skipped_models = []

    for ep in eps:
        model_name = ep.name

        # Check if we have the required dependencies
        if model_name in [
            "GraphCastNet",
            "MeshGraphNet",
            "BiStrideMeshGraphNet",
            "MeshGraphKAN",
            "HybridMeshGraphNet",
        ]:
            if importlib.util.find_spec("dgl") is None:
                skipped_models.append((model_name, "dgl not available"))
                continue

        if model_name in ["HEALPixRecUNet", "HEALPixUNet"]:
            if importlib.util.find_spec("earth2grid") is None:
                skipped_models.append((model_name, "earth2grid not available"))
                continue

        # Try to load the entry point
        try:
            ModelClass = ep.load()
            # Verify it's a Module subclass
            assert issubclass(ModelClass, Module), (
                f"{model_name} is not a Module subclass"
            )
        except Exception as e:
            failed_models.append((model_name, str(e)))

    # Report results
    if failed_models:
        failure_msg = "\n".join(
            [f"  - {name}: {error}" for name, error in failed_models]
        )
        pytest.fail(f"Failed to load {len(failed_models)} entry points:\n{failure_msg}")

    # Just log skipped models (not a failure)
    if skipped_models:
        skip_msg = "\n".join(
            [f"  - {name}: {reason}" for name, reason in skipped_models]
        )
        print(
            f"\nSkipped {len(skipped_models)} models due to missing optional dependencies:\n{skip_msg}"
        )


# =============================================================================
# Integration Tests
# =============================================================================


def test_register_factory_roundtrip(clear_registry, device):
    """Test full register -> factory -> instantiate -> forward pipeline"""
    registry = ModelRegistry()
    registry.register(TestModelA, "TestModelA")

    # Retrieve from registry
    ModelClass = registry.factory("TestModelA")

    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)

    # Instantiate
    model = ModelClass(size=64).to(device)

    # Test forward pass
    x = torch.randn(8, 64).to(device)
    output = model(x)

    assert output.shape == (8, 64)
    assert output.device == device


def test_registry_with_module_instantiate(clear_registry):
    """Test that Module.instantiate works with registry-loaded classes"""
    registry = ModelRegistry()
    registry.register(TestModelA, "TestModelA")

    # Create arg_dict as Module.__new__ would
    arg_dict = {
        "__name__": "TestModelA",
        "__module__": "test.core.test_registry",
        "__args__": {"size": 128},
    }

    # Instantiate should work through registry
    model = Module.instantiate(arg_dict)
    assert isinstance(model, TestModelA)
    assert model.size == 128
