# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from physicsnemo.utils.profiling import Profiler, _register_profilers, profile
from physicsnemo.utils.profiling.core import (
    PhysicsNeMoProfilerWrapper,
    ProfileRegistry,
    _Profiler_Singleton,
)

# Mock config class for testing


@dataclass
class MockProfilerConfig:
    """
    Specific configuration for the pytorch profiler.
    """

    name: str = "mock"
    option1: bool = True
    option2: int = 42


# Mock profiler class for testing
class MockProfiler(PhysicsNeMoProfilerWrapper, metaclass=_Profiler_Singleton):

    _is_context = True
    _is_decorator = True

    def __init__(self, config: Optional[MockProfilerConfig] = None, **config_overrides):

        default_config = MockProfilerConfig()

        # Replace any overrides right into the config:
        if config is None:
            self._config = replace(default_config, **config_overrides)
        else:
            self._config = replace(config, **config_overrides)

    def enable(self):
        # Progress the state to enabled
        self.enabled = True

    def _standup(self):
        # ... to initialized
        self.initialized = True

    def finalize(self, output_path: Path):
        # ... to finalized
        self.finalized = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def step(self):
        pass


@pytest.fixture(autouse=True)
def reset_profiler():
    Profiler._clear_instance()
    ProfileRegistry._clear()
    _register_profilers()
    # Reset and re-register the mock profiler
    MockProfiler._clear_instance()

    ProfileRegistry.register_profiler("mock", MockProfiler)
    # Reset the singleton instance


def test_profiler_initialization():
    profiler = Profiler()
    assert not profiler.initialized
    assert not profiler.enabled

    # Enable a mock profiler
    profiler.enable("mock")
    assert profiler.enabled

    # Test initialization
    profiler.initialize()
    assert profiler.initialized

    # It's a singleton, so we should get the same instance
    mock_profiler = MockProfiler()
    assert mock_profiler.initialized


def test_profiler_context_manager():
    profiler = Profiler()

    profiler.enable("mock")

    # Not initialized yet
    assert not profiler.initialized

    with profiler as p:
        # Now it is initialized
        assert p.initialized
        assert p.enabled


def test_profiler_state_progression():
    profiler = Profiler()
    mock_profiler = MockProfiler()

    # Everything should be disabled by default
    assert not mock_profiler.enabled
    assert not mock_profiler.initialized
    assert not mock_profiler.finalized

    profiler.enable(mock_profiler)

    assert mock_profiler.enabled
    assert not mock_profiler.initialized
    assert not mock_profiler.finalized

    profiler.initialize()

    assert mock_profiler.enabled
    assert mock_profiler.initialized
    assert not mock_profiler.finalized

    profiler.finalize()

    assert mock_profiler.enabled
    assert mock_profiler.initialized
    assert mock_profiler.finalized


def test_profiler_decoration():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)

    @profile
    def test_function():
        return "test"

    # Function should be registered for decoration before initialization
    assert test_function in profiler._decoration_registry

    # After initialization, function should be decorated
    profiler.initialize()
    assert test_function not in profiler._decoration_registry


def test_profiler_config_update():
    profiler = Profiler()

    mock_profiler = profiler.get("mock")

    mock_profiler.reconfigure(option1=False)
    assert not mock_profiler._config.option1

    mock_profiler.reconfigure(option2=100)
    assert mock_profiler._config.option2 == 100

    profiler.enable(mock_profiler)


def test_output_path():
    profiler = Profiler()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_output"
        profiler.output_path = test_path
        assert profiler.output_path == test_path

        # Test string conversion
        test_path2 = str(Path(tmpdir) / "test_output2")
        profiler.output_path = test_path2
        assert isinstance(profiler.output_path, Path)


def test_profiler_finalization():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)

    # Mock the finalize method
    mock_profiler.finalize = MagicMock()

    profiler.finalize()
    mock_profiler.finalize.assert_called_once()


def test_profiler_step():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)

    # Mock the step method
    mock_profiler.step = MagicMock()

    profiler.step()
    mock_profiler.step.assert_called_once()


def test_function_replacement():
    def original_function():
        return "original"

    def wrapped_function():
        return "wrapped"

    profiler = Profiler()
    with patch("sys.modules") as mock_modules:
        mock_module = MagicMock()
        mock_modules.__getitem__.return_value = mock_module
        original_function.__module__ = "test_module"
        original_function.__qualname__ = "original_function"

        profiler.replace_function(original_function, wrapped_function)

        # Check that the function was replaced in the module
        mock_module.original_function = wrapped_function


def test_state_enum_ge_operator():
    # Access the private State enum
    state_enum = PhysicsNeMoProfilerWrapper.State

    # Test the __ge__ operator
    assert state_enum.ENABLED >= state_enum.DISABLED
    assert state_enum.INITIALIZED >= state_enum.ENABLED
    assert state_enum.FINALIZED >= state_enum.INITIALIZED

    # Test the reverse, which should be False
    assert not (state_enum.DISABLED >= state_enum.ENABLED)
    assert not (state_enum.ENABLED >= state_enum.INITIALIZED)
    assert not (state_enum.INITIALIZED >= state_enum.FINALIZED)

    # Test equality
    assert state_enum.ENABLED >= state_enum.ENABLED
    assert state_enum.INITIALIZED >= state_enum.INITIALIZED
    assert state_enum.FINALIZED >= state_enum.FINALIZED
