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


import random
import re
import sys

import pytest
import torch
import torch.nn as nn

from physicsnemo.core.module import Module
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.models.flare import FLARE, FLAREPlusPlus
from physicsnemo.nn import FLARE as FLAREAttention
from physicsnemo.nn import FLAREPlusPlus as FLAREPlusPlusAttention
from test.common import (
    check_ort_version,
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_cuda_graphs,
    validate_forward_accuracy,
    validate_jit,
    validate_onnx_export,
    validate_onnx_runtime,
)

_SMALL = dict(
    functional_dim=2,
    embedding_dim=3,
    out_dim=2,
    n_layers=2,
    n_hidden=16,
    n_head=4,
    mlp_ratio=2,
    slice_num=4,
)

_MODELS = {
    "flare": (FLARE, {}),
    "flare_qk_norm": (FLARE, {"qk_norm": True}),
    "flarepp": (FLAREPlusPlus, {}),
}


def _inputs(device, batch=2, n=24):
    return (
        torch.randn(batch, n, 2, device=device),
        torch.randn(batch, n, 3, device=device),
    )


def _assert_parameter_gradients_close(actual, expected, *, atol, rtol):
    """Compare gradients for every parameter in two equivalent models."""
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_parameter in expected_parameters.items():
        torch.testing.assert_close(
            actual_parameters[name].grad,
            expected_parameter.grad,
            atol=atol,
            rtol=rtol,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )


def test_flare_legacy_checkpoint_class_path():
    """Test resolving the model class path stored by experimental checkpoints."""
    # Drop the cached legacy modules so the shim warning fires again.
    for module_name in list(sys.modules):
        if module_name.startswith("physicsnemo.experimental.models.flare"):
            del sys.modules[module_name]

    with pytest.warns(
        LegacyFeatureWarning, match=re.escape("physicsnemo.models.flare")
    ):
        from physicsnemo.experimental.models.flare import FLARE as LegacyPackageFLARE
        from physicsnemo.experimental.models.flare.flare import (
            FLARE as LegacyModuleFLARE,
        )

    assert LegacyPackageFLARE is FLARE
    assert LegacyModuleFLARE is FLARE


def test_flare_constructor_defaults():
    """Default configuration and model structure."""
    model = FLARE(functional_dim=3, out_dim=1)
    assert isinstance(model, Module)
    assert model.n_hidden == 256
    assert len(model.blocks) == 4
    assert model.embedding_dim is None
    block = model.blocks[0]
    assert isinstance(block.mixer, FLAREAttention)
    assert block.mixer.heads == 8 and block.mixer.n_global_queries == 32
    assert isinstance(block.mixer.q_norm, nn.Identity)
    # PN input projection C_in -> 2C -> C, tanh-GELU, block MLP ratio 4.
    linears = [m for m in model.preprocess.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (3, 512),
        (512, 256),
    ]
    assert model.preprocess.layers[1].approximate == "tanh"
    assert block.mlp.layers[0].out_features == 1024
    # Norm + Linear head.
    assert isinstance(model.head[0], nn.RMSNorm)
    assert model.head[1].out_features == 1


@pytest.mark.parametrize("name", list(_MODELS))
@pytest.mark.parametrize("use_rmsnorm", [True, False])
def test_norm_type(name, use_rmsnorm):
    """use_rmsnorm switches every norm, including the mixers' head norms."""
    cls, kwargs = _MODELS[name]
    model = cls(**_SMALL, use_rmsnorm=use_rmsnorm, **kwargs)
    norm_cls = nn.RMSNorm if use_rmsnorm else nn.LayerNorm
    norms = [m for m in model.modules() if isinstance(m, (nn.RMSNorm, nn.LayerNorm))]
    assert all(type(m) is norm_cls for m in norms)
    # 2 per block + head, plus the mixer norms when present.
    assert len(norms) >= 2 * _SMALL["n_layers"] + 1


def test_flare_plus_plus_constructor():
    """FLARE++ passes its mixer options to every block."""
    model = FLAREPlusPlus(
        **_SMALL, k_norm=False, share_k0_v0=False, gate_logit_init=1.0
    )
    for block in model.blocks:
        assert isinstance(block.mixer, FLAREPlusPlusAttention)
        assert isinstance(block.mixer.k_norm, nn.Identity)
        assert block.mixer.v0_proj is not None
        torch.testing.assert_close(block.mixer.gate_logit, torch.ones(4))


@pytest.mark.parametrize("name", list(_MODELS))
def test_init(name):
    """Backbone init keeps the mixers' identity-plus-noise projections and latents."""
    cls, kwargs = _MODELS[name]
    torch.manual_seed(0)
    model = cls(**{**_SMALL, "n_hidden": 64, "slice_num": 16}, **kwargs)
    mixer = model.blocks[0].mixer
    eye = torch.eye(64)
    assert (mixer.k_proj.weight - eye).abs().max() < 0.15
    assert (mixer.v_proj.weight - eye).abs().max() < 0.15
    assert 0.015 < mixer.out_proj.weight.std() < 0.025
    latent = mixer.latent_q if cls is FLARE else mixer.latent_q_fixed
    assert 0.01 < latent.std() < 0.03
    assert model.preprocess.layers[0].weight.std() < 0.03


@pytest.mark.parametrize("name", list(_MODELS))
def test_no_weight_decay(name):
    """The model collects the mixers' no-weight-decay parameter names."""
    cls, kwargs = _MODELS[name]
    model = cls(**_SMALL, **kwargs)
    names = model.no_weight_decay()
    if cls is FLARE:
        per_block = {"latent_q"}
    else:
        per_block = {"latent_q0", "latent_q_fixed", "gate_logit"}
    assert names == {f"blocks.{i}.mixer.{n}" for i in range(2) for n in per_block}
    assert names <= dict(model.named_parameters()).keys()


@pytest.mark.parametrize("name", list(_MODELS))
def test_forward_backward(device, name):
    """End-to-end forward and gradients to inputs and all parameters."""
    cls, kwargs = _MODELS[name]
    torch.manual_seed(0)
    model = cls(**_SMALL, **kwargs).to(device)
    fx, embedding = (t.requires_grad_(True) for t in _inputs(device))
    out = model(fx, embedding=embedding)
    assert out.shape == (2, 24, 2)
    out.square().mean().backward()
    assert torch.isfinite(fx.grad).all() and torch.isfinite(embedding.grad).all()
    for pname, param in model.named_parameters():
        assert param.grad is not None, pname
        assert torch.isfinite(param.grad).all(), pname


def test_forward_without_embedding(device):
    model = FLARE(functional_dim=5, out_dim=3, n_layers=1, n_hidden=16, n_head=4)
    model = model.to(device)
    assert model(torch.randn(1, 7, 5, device=device)).shape == (1, 7, 3)


def test_embedding_validation():
    model = FLARE(**_SMALL)
    fx, embedding = _inputs("cpu")
    with pytest.raises(ValueError, match="3 channels.*got none"):
        model(fx)
    with pytest.raises(ValueError, match="3 channels.*got 2"):
        model(fx, embedding=fx)
    no_emb = FLARE(**{**_SMALL, "embedding_dim": None, "functional_dim": 5})
    with pytest.raises(ValueError, match="0 channels.*got 3"):
        no_emb(torch.randn(1, 7, 5), embedding=torch.randn(1, 7, 3))
    # A zero-width embedding is accepted.
    assert no_emb(torch.randn(1, 7, 5), embedding=torch.randn(1, 7, 0)).shape == (
        1,
        7,
        2,
    )
    with pytest.raises(ValueError, match="divisible"):
        FLARE(functional_dim=2, out_dim=1, n_hidden=30, n_head=4)


@pytest.mark.parametrize("name", list(_MODELS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast(device, name, dtype):
    """Mixed-precision forward/backward stays finite."""
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 autocast is CUDA-only")
    cls, kwargs = _MODELS[name]
    torch.manual_seed(0)
    model = cls(**_SMALL, **kwargs).to(device)
    fx, embedding = _inputs(device)
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        out = model(fx, embedding=embedding)
    assert torch.isfinite(out).all()
    out.float().square().mean().backward()


@pytest.mark.parametrize(
    "name, file_name",
    [
        ("flare", "models/flare/data/flare_output.pth"),
        ("flare_qk_norm", "models/flare/data/flare_qk_norm_output.pth"),
        ("flarepp", "models/flare/data/flare_plus_plus_output.pth"),
    ],
)
def test_forward_accuracy(device, name, file_name):
    """Forward passes match committed golden outputs."""
    cls, kwargs = _MODELS[name]
    torch.manual_seed(1234)
    model = cls(**_SMALL, **kwargs).to(device)
    torch.manual_seed(1)
    fx, embedding = _inputs(device, n=17)
    assert validate_forward_accuracy(model, (fx, embedding), file_name=file_name)


@pytest.mark.parametrize("name", list(_MODELS))
def test_checkpoint(device, name):
    """Save/load roundtrip through physicsnemo checkpoints."""
    cls, kwargs = _MODELS[name]
    model_1 = cls(**_SMALL, **kwargs).to(device)
    model_2 = cls(**_SMALL, **kwargs).to(device)
    fx, embedding = _inputs(device, batch=random.randint(1, 2))
    assert validate_checkpoint(model_1, model_2, (fx, embedding))


@pytest.mark.parametrize("name", list(_MODELS))
def test_from_checkpoint(tmp_path, name):
    """Module.from_checkpoint restores the class and its constructor options."""
    cls, kwargs = _MODELS[name]
    model = cls(
        **_SMALL,
        use_rmsnorm=False,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
        **kwargs,
    )
    path = tmp_path / f"{name}.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path)
    assert type(restored) is cls
    assert restored._activation_checkpointing_ratio == 0.5
    assert isinstance(restored.head[0], nn.LayerNorm)
    fx, embedding = _inputs("cpu")
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(fx, embedding), model(fx, embedding))


@pytest.mark.parametrize("name", list(_MODELS))
def test_activation_checkpointing_matches(device, name):
    """Checkpointed blocks preserve outputs and gradients."""
    cls, kwargs = _MODELS[name]
    torch.manual_seed(0)
    plain = cls(**_SMALL, **kwargs).to(device)
    checkpointed = cls(**_SMALL, activation_checkpointing=True, **kwargs).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    fx, embedding = _inputs(device)
    out_plain = plain(fx, embedding)
    out_checkpointed = checkpointed(fx, embedding)
    torch.testing.assert_close(out_checkpointed, out_plain, atol=1e-6, rtol=1e-5)
    out_plain.square().mean().backward()
    out_checkpointed.square().mean().backward()
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", list(_MODELS))
def test_torch_compile_fullgraph(device, name):
    """The models compile with fullgraph=True and match eager."""
    cls, kwargs = _MODELS[name]
    model = cls(**_SMALL, **kwargs).to(device).eval()
    fx, embedding = _inputs(device)
    expected = model(fx, embedding)
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(fx, embedding), expected, atol=1e-5, rtol=1e-4)


def test_flare_optims(device):
    """CUDA graphs, JIT, AMP and combined optimizations."""

    def setup_model():
        model = FLARE(**_SMALL).to(device)
        return model, *_inputs(device)

    for validate in (
        validate_cuda_graphs,
        validate_jit,
        validate_amp,
        validate_combo_optims,
    ):
        model, fx, embedding = setup_model()
        assert validate(model, (fx, embedding))


@check_ort_version()
def test_flare_deploy(device):
    """ONNX export and runtime."""
    model = FLARE(**_SMALL).to(device)
    fx, embedding = _inputs(device)
    assert validate_onnx_export(model, (fx, embedding))
    assert validate_onnx_runtime(model, (fx, embedding), 1e-2, 1e-2)
