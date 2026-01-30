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

import pytest
from torch import nn

from physicsnemo.active_learning._registry import ActiveLearningRegistry


@pytest.fixture(scope="function")
def simple_class():
    class SimpleClass:
        def __init__(self, param1: int, param2: str):
            self.param1 = param1
            self.param2 = param2

    return SimpleClass


def test_initialization():
    """Test that the registry can be initialized."""
    registry = ActiveLearningRegistry()
    assert registry._registry == {}


def test_registration(simple_class):
    """Test that the register method registers a class"""
    registry = ActiveLearningRegistry()

    registry.register("my_strategy")(simple_class)

    assert registry.is_registered("my_strategy")
    assert "my_strategy" in registry.registered_names
    assert registry._registry["my_strategy"] == simple_class


def test_missing_registration():
    """Test accessing a missing class raises a KeyError"""
    registry = ActiveLearningRegistry()
    with pytest.raises(KeyError):
        registry["missing_strategy"]

    with pytest.raises(NameError):
        registry.construct("missing_strategy")


def test_construction(simple_class):
    """Test that the construct method returns an instance of the registered class."""
    registry = ActiveLearningRegistry()

    registry.register("my_strategy")(simple_class)

    strategy = registry.construct("my_strategy", param1=42, param2="test")
    assert strategy.param1 == 42
    assert strategy.param2 == "test"


def test_torch_module():
    """Test that registry can construct different types of objects"""
    registry = ActiveLearningRegistry()

    @registry.register("simple_model")
    class SimpleModel(nn.Module):
        def __init__(self, input_size: int, output_size: int):
            super().__init__()
            self.linear = nn.Linear(input_size, output_size)

    input_size = 10
    output_size = 10
    model = registry.construct(
        "simple_model", input_size=input_size, output_size=output_size
    )
    assert isinstance(model, nn.Module)
    assert model.linear.weight.shape == (output_size, input_size)


def test_bad_construction():
    """Test that the construct method raises an error with bad arguments"""
    registry = ActiveLearningRegistry()

    @registry.register("simple_model")
    class SimpleModel(nn.Module):
        def __init__(self, input_size: int, output_size: int):
            super().__init__()
            self.linear = nn.Linear(input_size, output_size)

    with pytest.raises(TypeError):
        registry.construct("simple_model", input_size=10, bad_arg=215)


def test_get_class_no_module_path():
    """Test that the get_class method returns a class from the registry or module path."""
    from time import monotonic

    registry = ActiveLearningRegistry()
    cls = registry.get_class("monotonic", "time")
    assert cls == monotonic


def test_get_class_with_module_path():
    """Test that the get_class method returns a class from a module path."""
    registry = ActiveLearningRegistry()
    cls = registry.get_class("Linear", "torch.nn")
    model = cls(8, 16)

    # add the import now and make sure they are equivalent
    from torch import nn

    assert isinstance(model, nn.Linear)


def test_get_class_missing():
    """Test that the get_class method raises an error in three scenarios."""
    registry = ActiveLearningRegistry()
    # when the module is completely missing
    with pytest.raises(ModuleNotFoundError):
        registry.get_class("missing_class", "missing_module")
    # when we are missing the class in a module
    with pytest.raises(NameError):
        registry.get_class("missing_class", "torch.nn")
    # when we are missing the class in the registry and no module path is provided
    with pytest.raises(NameError):
        registry.get_class("missing_class")
