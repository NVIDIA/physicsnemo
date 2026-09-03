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

import io
import tarfile
from pathlib import Path

import pytest
import torch

import physicsnemo.core
from physicsnemo.core import ModelRegistry


# Fixture to clear registry between tests to avoid naming conflicts
@pytest.fixture(autouse=True)
def clear_registry():
    """Clear and restore the model registry before and after each test"""
    registry = ModelRegistry()
    registry.__clear_registry__()
    yield
    registry.__restore_registry__()


class MockModel(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class NewMockModel(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class MockModelNoOverride(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, value1, value2, x):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))
        self.x = x


class MockModelWithOverride(physicsnemo.core.Module):
    """Fake model"""

    _overridable_args = {"value2", "x"}

    def __init__(self, value1, value2, x):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))
        self.x = x


@pytest.mark.parametrize("LoadModel", [MockModel, NewMockModel])
def test_from_checkpoint_custom(device, LoadModel):
    """Test checkpointing custom physicsnemo module"""
    torch.manual_seed(0)

    # Construct Mock Model and save it
    mock_model = MockModel().to(device)
    mock_model.save("checkpoint.mdlus")

    # Load from checkpoint using class
    LoadModel.from_checkpoint("checkpoint.mdlus")
    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)


def test_from_checkpoint_override(device):
    """Test checkpointing custom physicsnemo module with override"""
    torch.manual_seed(0)

    # Model with no overrides, loading without overrides
    mock_model = MockModelNoOverride(1, 2, 3).to(device)
    mock_model.save("checkpoint.mdlus")
    mock_model = MockModelWithOverride.from_checkpoint("checkpoint.mdlus")

    # Model with no overrides, loading with overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"value2": 20}
        )

    Path("checkpoint.mdlus").unlink(missing_ok=False)

    # Model with overrides, loading without overrides
    mock_model = MockModelWithOverride(1, 2, 3).to(device)
    mock_model.save("checkpoint.mdlus")
    mock_model = MockModelWithOverride.from_checkpoint("checkpoint.mdlus")

    # Model with overrides, loading with allowed overrides (``value2`` value
    # should be erased by the state-dict, ``x`` should be overriden and kept)
    mock_model = MockModelWithOverride.from_checkpoint(
        "checkpoint.mdlus", override_args={"value2": 20, "x": 30}
    )
    assert torch.equal(mock_model.w2, torch.tensor(2, dtype=torch.float32))
    assert mock_model.x == 30

    # Model with overrides, loading with disallowed overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"value1": 10, "value2": 20}
        )

    # Model with overrides, loading with unexpected overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"value3": 4}
        )

    Path("checkpoint.mdlus").unlink(missing_ok=False)


def test_checkpoint_archive_members_stay_within_destination(tmp_path):
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for name in ("model.pt", "../outside.pt", "/absolute.pt"):
            content = b"checkpoint"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

        link = tarfile.TarInfo("linked-model.pt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside.pt"
        archive.addfile(link)

    archive_buffer.seek(0)
    with tarfile.open(fileobj=archive_buffer, mode="r") as archive:
        safe_names = [
            member.name
            for member in physicsnemo.core.Module._safe_members(archive, tmp_path)
        ]

    assert safe_names == ["model.pt"]


def test_from_checkpoint_state_dict_mapper(device, tmp_path):
    """Version-aware state-dict key remapping loads refactored checkpoints."""

    class LegacyLinearModel(physicsnemo.core.Module):
        __model_checkpoint_version__ = "0.1.0"

        def __init__(self, features=4):
            super().__init__()
            self.features = features
            self.layer = torch.nn.Linear(features, features)

    class CurrentLinearModel(physicsnemo.core.Module):
        __model_checkpoint_version__ = "0.2.0"
        __supported_model_checkpoint_version__ = {
            "0.1.0": "Loading legacy checkpoint with renamed layer keys."
        }

        def __init__(self, features=4):
            super().__init__()
            self.features = features
            self.block = torch.nn.Linear(features, features)

        @classmethod
        def _backward_compat_state_dict_mapper(cls, version, state_dict):
            if version == "0.1.0":
                return {k.replace("layer.", "block."): v for k, v in state_dict.items()}
            return state_dict

    torch.manual_seed(0)
    ckpt_path = tmp_path / "refactored.mdlus"
    model = LegacyLinearModel().to(device)
    layer_weight = model.layer.weight.detach().clone()
    layer_bias = model.layer.bias.detach().clone()
    model.save(ckpt_path)

    loaded = CurrentLinearModel.from_checkpoint(ckpt_path).to(device)
    assert torch.allclose(loaded.block.weight, layer_weight)
    assert torch.allclose(loaded.block.bias, layer_bias)
