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

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.registry import ModelRegistry

registry = ModelRegistry()


@dataclass
class MMetaData(ModelMetaData):
    """Custom User Metadata for Model"""

    # Optimization    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    onnx: bool = True
    onnx_runtime: bool = True
    func_torch: bool = True
    auto_grad: bool = True


class M(physicsnemo.core.Module):
    """Fake model"""

    _overridable_args = {"a"}

    def __init__(self, a, m1, m2):
        super().__init__(meta=MMetaData())
        self.a = torch.nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self._a = a
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return -self.a * (self.m1(x) + self.m2(x))


@dataclass
class M1MetaData(ModelMetaData):
    """Custom User Metadata for Model"""

    # Optimization    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    onnx: bool = True
    onnx_runtime: bool = True
    func_torch: bool = True
    auto_grad: bool = True


class M1(physicsnemo.core.Module):
    """Fake model"""

    _overridable_args = {"b"}

    def __init__(self, b):
        super().__init__(meta=M1MetaData())
        self.b = torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self._b = b

    def forward(self, x):
        return self.b * x


@dataclass
class TorchModelMetaData(ModelMetaData):
    """Custom User Metadata for Model"""

    # Optimization    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    onnx: bool = True
    onnx_runtime: bool = True
    func_torch: bool = True
    auto_grad: bool = True


class TorchModel(torch.nn.Module):
    """Fake model"""

    def __init__(self, c):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(c, dtype=torch.float32))

    def forward(self, x):
        return self.c * x


def make_model():
    Mt = physicsnemo.core.Module.from_torch(
        TorchModel, meta=TorchModelMetaData(), register=True
    )
    m21 = Mt(21.0)
    m22 = M1(22.0)
    m11 = M1(11.0)
    m12 = M(12.0, m21, m22)
    m = M(1.0, m11, m12)
    return m, Mt


@pytest.mark.parametrize("override", [True, False], ids=["override", "no_override"])
def test_save_load(device, override):
    m_orig, Mt = make_model()
    m_orig = m_orig.to(device)
    m_orig.save("checkpoint.mdlus")
    if not override:
        m_loaded = physicsnemo.core.Module.from_checkpoint("checkpoint.mdlus")
    else:
        m_loaded = physicsnemo.core.Module.from_checkpoint(
            "checkpoint.mdlus", override_args={"a": -0.1, "m2.a": -0.2, "m2.m2.b": -0.3}
        )
    assert isinstance(m_loaded, M)
    assert isinstance(m_loaded.m1, M1)
    assert isinstance(m_loaded.m2, M)
    assert isinstance(m_loaded.m2.m1, Mt)
    assert isinstance(m_loaded.m2.m2, M1)
    assert m_loaded.a == m_orig.a
    assert m_loaded._a == (m_orig._a if not override else -0.1)
    assert m_loaded.m1.b == m_orig.m1.b
    assert m_loaded.m2.a == m_orig.m2.a
    assert m_loaded.m2._a == (m_orig.m2._a if not override else -0.2)
    assert m_loaded.m2.m1.inner_model.c == m_orig.m2.m1.inner_model.c
    assert m_loaded.m2.m2.b == m_orig.m2.m2.b
    assert m_loaded.m2.m2._b == (m_orig.m2.m2._b if not override else -0.3)

    if override:
        with pytest.raises(ValueError):
            physicsnemo.core.Module.from_checkpoint(
                "checkpoint.mdlus", override_args={"m2.m1.c": -0.4}
            )

    Path("checkpoint.mdlus").unlink(missing_ok=False)
    registry.__clear_registry__()
    registry.__restore_registry__()


@pytest.mark.parametrize("override", [True, False], ids=["override", "no_override"])
def test_load_from_checkpoint(device, override):
    # CJA - Had to add this, the model was still registered here ...
    # I think its becauses the tests aren't completing, yet, so the clear
    # never happens at the bottom.  Delete eventually.
    registry.__clear_registry__()
    file_name: str = str(
        Path(__file__).parents[0].resolve()
        / Path("data")
        / Path("checkpoint_nested_modules.mdlus")
    )

    m_orig, Mt = make_model()
    m_orig = m_orig.to(device)

    if not override:
        m_loaded = physicsnemo.core.Module.from_checkpoint(file_name).to(device)
    else:
        m_loaded = physicsnemo.core.Module.from_checkpoint(
            file_name, override_args={"a": -0.1, "m2.a": -0.2, "m2.m2.b": -0.3}
        ).to(device)
    assert isinstance(m_loaded, M)
    assert isinstance(m_loaded.m1, M1)
    assert isinstance(m_loaded.m2, M)
    assert isinstance(m_loaded.m2.m1, Mt)
    assert isinstance(m_loaded.m2.m2, M1)
    assert m_loaded.a == m_orig.a
    assert m_loaded._a == (m_orig._a if not override else -0.1)
    assert m_loaded.m1.b == m_orig.m1.b
    assert m_loaded.m2.a == m_orig.m2.a
    assert m_loaded.m2._a == (m_orig.m2._a if not override else -0.2)
    assert m_loaded.m2.m1.inner_model.c == m_orig.m2.m1.inner_model.c
    assert m_loaded.m2.m2.b == m_orig.m2.m2.b
    assert m_loaded.m2.m2._b == (m_orig.m2.m2._b if not override else -0.3)

    if override:
        with pytest.raises(ValueError):
            physicsnemo.core.Module.from_checkpoint(
                file_name, override_args={"m2.m1.c": -0.4}
            )
    registry.__clear_registry__()
    registry.__restore_registry__()
