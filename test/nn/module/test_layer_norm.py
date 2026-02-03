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

import builtins
import importlib
import importlib.util
import sys
import tempfile
from dataclasses import dataclass

import pytest
import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import get_installed_version
from physicsnemo.utils import load_checkpoint, save_checkpoint
from test.conftest import requires_module

LAYER_NORM_PATH = "physicsnemo.nn.module.layer_norm"


def reload_layer_norm():
    """Reload the layer_norm module to re-evaluate TE availability and env vars."""
    if LAYER_NORM_PATH in sys.modules:
        del sys.modules[LAYER_NORM_PATH]
    return importlib.import_module(LAYER_NORM_PATH)


@pytest.fixture(autouse=True)
def clear_version_check_cache():
    get_installed_version.cache_clear()
    yield
    get_installed_version.cache_clear()


def test_torch_fallback(monkeypatch):
    """
    This test pretends that transformer_engine.pytorch is not available,
    and checks that the LayerNorm class falls back to torch.nn.LayerNorm
    """
    # Remove from sys.modules if present
    monkeypatch.delenv("PHYSICSNEMO_FORCE_TE", raising=False)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", None)
    monkeypatch.setitem(sys.modules, "transformer_engine", None)

    # Patch check_version_spec to return False for transformer_engine
    from physicsnemo.core import version_check

    original_check = version_check.check_version_spec

    def fake_check_version_spec(module_name, *args, **kwargs):
        if module_name == "transformer_engine":
            return False
        # For other modules, use the original function
        return original_check(module_name, *args, **kwargs)

    monkeypatch.setattr(version_check, "check_version_spec", fake_check_version_spec)

    # Patch importlib to simulate ImportError
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformer_engine.pytorch" or name.startswith(
            "transformer_engine.pytorch"
        ):
            raise ImportError("Simulated missing transformer_engine.pytorch")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    layer_norm = reload_layer_norm()
    ln = layer_norm.LayerNorm(8)
    assert isinstance(ln, torch.nn.LayerNorm)


@requires_module(["transformer_engine"])
@pytest.mark.parametrize(
    "force_val,expected_type",
    [
        ("true", "te"),
        ("1", "te"),
        ("false", "torch"),
        ("0", "torch"),
    ],
)
def test_force_env(force_val, expected_type, device, pytestconfig, monkeypatch):
    if device == "cpu":
        force_val = False

    monkeypatch.setenv("PHYSICSNEMO_FORCE_TE", force_val)

    layer_norm = reload_layer_norm()
    ln = layer_norm.LayerNorm(8).to(device)
    if expected_type == "te":
        if device == "cuda:0":
            import transformer_engine.pytorch as te  # noqa: F401

            assert isinstance(ln, te.LayerNorm)
        else:
            assert isinstance(ln, torch.nn.LayerNorm)
    else:
        assert isinstance(ln, torch.nn.LayerNorm)


@requires_module("transformer_engine")
@pytest.mark.parametrize(
    "order",
    [
        0,
    ],
)
def test_serialization(device, order, monkeypatch, pytestconfig):
    """
    This test checks that the LayerNorm class can be serialized and deserialized
    while switching between TE and torch layer norm.  Uses physicsnemo checkpoint
    utils.
    """

    @dataclass
    class FakeModelMetaData(ModelMetaData):
        name = "FakeModel"

    class FakeModel(Module):
        def __init__(self):
            super().__init__(meta=FakeModelMetaData())
            self.ln = layer_norm.LayerNorm(8)

        def forward(self, x):
            return self.ln(x)

    # Control the order of swapping
    if order == 1:
        first = "false"  # force pytorch
        second = "true"  # force te
    else:
        first = "true"  # force te
        second = "false"  # force pytorch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Force to use pytorch
        monkeypatch.setenv("PHYSICSNEMO_FORCE_TE", first)
        layer_norm = reload_layer_norm()
        ln = FakeModel().cuda()
        print(ln.state_dict().keys())

        x = torch.randn(2, 8).cuda()
        y = ln(x)
        print(f"Y shape: {y.shape}")
        ckpt_args = {
            "path": tmpdir + "/checkpoints",
            "models": ln,
        }
        save_checkpoint(**ckpt_args)

        del ln

        # Now, reload
        monkeypatch.setenv("PHYSICSNEMO_FORCE_TE", second)
        layer_norm = reload_layer_norm()

        ln = FakeModel().cuda()

        print(f"new state dict keys: {ln.state_dict().keys()}")
        ckpt_args = {
            "path": tmpdir + "/checkpoints",
            "models": ln,
        }
        load_checkpoint(**ckpt_args)
        y_te = ln(x)

        assert torch.allclose(y, y_te)
