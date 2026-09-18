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
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.models.flare import FLARE, FLAREPlusPlus
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
from test.conftest import requires_module


def _assert_parameter_gradients_close(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    atol: float,
    rtol: float,
):
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


def _make_checkpointing_model_pair(
    model_kwargs: dict[str, object], device: str | torch.device = "cpu"
) -> tuple[FLARE, FLARE]:
    """Build equivalent plain and fully checkpointed FLARE models."""
    plain = FLARE(**model_kwargs, activation_checkpointing=False).to(device)
    checkpointed = FLARE(**model_kwargs, activation_checkpointing=True).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    return plain, checkpointed


def _load_or_create_output_reference(
    file_name: str, output: torch.Tensor
) -> torch.Tensor:
    """Load a local golden output, or create it and require a second run."""
    reference_path = Path(__file__).parent / "data" / file_name
    if not reference_path.exists():
        torch.save({"output": output.detach().cpu()}, reference_path)
        raise IOError(
            f"Golden output {reference_path} was missing and has been created; "
            "commit it and re-run the test."
        )
    reference = torch.load(reference_path, weights_only=True)
    return next(iter(reference.values())).to(output.device)


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


@pytest.mark.parametrize(
    "config",
    ["default_structured", "custom_irregular"],
    ids=["with_defaults_structured", "with_custom_irregular"],
)
def test_flare_constructor(config):
    """Test FLARE model constructor and attributes."""
    if config == "default_structured":
        model = FLARE(
            functional_dim=3,
            out_dim=1,
            structured_shape=(64, 64),
            unified_pos=True,
        )
        assert model.n_hidden == 256, "Default n_hidden should be 256"
        assert model.time_input is False, "Default time_input should be False"
        assert model.unified_pos is True
        assert model.structured_shape == (64, 64)
        assert model.embedding_dim == 64  # ref * ref = 8 * 8 = 64
        assert len(model.blocks) == 4, "Default n_layers should be 4"
    else:
        model = FLARE(
            functional_dim=2,
            out_dim=4,
            embedding_dim=3,
            n_layers=8,
            n_hidden=64,
            dropout=0.1,
            n_head=4,
            act="gelu",
            mlp_ratio=2,
            slice_num=16,
            unified_pos=False,
            structured_shape=None,
            time_input=True,
        )
        assert model.n_hidden == 64
        assert model.time_input is True
        assert model.unified_pos is False
        assert model.structured_shape is None
        assert model.embedding_dim == 3
        assert len(model.blocks) == 8

    assert isinstance(model, Module), "FLARE should inherit from physicsnemo.Module"
    assert hasattr(model, "preprocess"), "Model should have preprocess MLP"
    assert hasattr(model, "blocks"), "Model should have transformer blocks"
    assert hasattr(model, "meta"), "Model should have metadata"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {
                "functional_dim": 3,
                "out_dim": 1,
                "structured_shape": (8, 8),
                "unified_pos": True,
            },
            (4, 256, 8, 32),
        ),
        (
            {
                "functional_dim": 2,
                "out_dim": 3,
                "embedding_dim": 3,
                "n_layers": 3,
                "n_hidden": 24,
                "n_head": 3,
                "slice_num": 5,
                "structured_shape": None,
            },
            (3, 24, 3, 5),
        ),
    ],
    ids=("defaults", "custom"),
)
def test_flare_plus_plus_constructor(kwargs, expected):
    """Standalone FLARE++ exposes default and custom model configuration."""
    model = FLAREPlusPlus(**kwargs)
    n_layers, n_hidden, n_head, n_queries = expected

    assert isinstance(model, Module)
    assert model.n_hidden == n_hidden
    assert len(model.blocks) == n_layers
    assert all(isinstance(block.Attn, FLAREPlusPlusAttention) for block in model.blocks)
    assert all(block.Attn.heads == n_head for block in model.blocks)
    assert all(block.Attn.n_global_queries == n_queries for block in model.blocks)


@pytest.mark.parametrize(
    "n_hidden,n_head,slice_num,file_name",
    [
        (16, 4, 4, "flare_plus_plus_small_output.pth"),
        (24, 3, 5, "flare_plus_plus_custom_output.pth"),
    ],
    ids=("small", "custom_heads"),
)
def test_flare_plus_plus_forward_accuracy(
    device, n_hidden, n_head, slice_num, file_name
):
    """Standalone FLARE++ forward passes match committed golden outputs."""
    if torch.__version__ < "2.12":
        pytest.skip(
            "FLARE++ golden outputs require torch >= 2.12 because "
            "trunc_normal_ initialization changed in PyTorch 2.12."
        )
    torch.manual_seed(1234)
    model = FLAREPlusPlus(
        functional_dim=2,
        out_dim=2,
        embedding_dim=3,
        n_layers=2,
        n_hidden=n_hidden,
        n_head=n_head,
        mlp_ratio=2,
        slice_num=slice_num,
        structured_shape=None,
    ).to(device)
    functional_input = torch.randn(2, 17, 2).to(device)
    embedding = torch.randn(2, 17, 3).to(device)

    with torch.no_grad():
        output = model(functional_input, embedding)
    reference = _load_or_create_output_reference(file_name, output)
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def test_flare_plus_plus_forward_backward(device):
    """Standalone FLARE++ supports end-to-end backpropagation."""
    model = FLAREPlusPlus(
        functional_dim=2,
        out_dim=1,
        embedding_dim=3,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        mlp_ratio=1,
        slice_num=4,
    ).to(device)
    functional_input = torch.randn(2, 19, 2, device=device, requires_grad=True)
    embedding = torch.randn(2, 19, 3, device=device, requires_grad=True)

    output = model(functional_input, embedding)
    output.square().mean().backward()

    assert output.shape == (2, 19, 1)
    assert functional_input.grad is not None
    assert embedding.grad is not None
    assert torch.isfinite(functional_input.grad).all()
    assert torch.isfinite(embedding.grad).all()


def test_flare_plus_plus_eval_is_rng_free(device):
    """The standalone FLARE++ model is deterministic during evaluation."""
    torch.manual_seed(44)
    model = FLAREPlusPlus(
        functional_dim=2,
        out_dim=1,
        embedding_dim=3,
        n_layers=2,
        n_hidden=32,
        dropout=0.4,
        n_head=4,
        mlp_ratio=1,
        slice_num=8,
    ).to(device)
    functional_input = torch.randn(2, 19, 2, device=device)
    embedding = torch.randn(2, 19, 3, device=device)

    model.eval()
    model_device = next(model.parameters()).device
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(model_device).clone()
        if model_device.type == "cuda"
        else None
    )
    with torch.no_grad():
        output_1 = model(functional_input, embedding)
        output_2 = model(functional_input, embedding)

    assert torch.equal(output_1, output_2)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(model_device), cuda_rng_before)


def test_flare_plus_plus_checkpoint_roundtrip(device):
    """Standalone FLARE++ saves and restores through PhysicsNeMo checkpoints."""

    def make_model():
        return FLAREPlusPlus(
            functional_dim=2,
            out_dim=1,
            embedding_dim=3,
            n_layers=2,
            n_hidden=16,
            n_head=4,
            mlp_ratio=1,
            slice_num=4,
        ).to(device)

    functional_input = torch.randn(1, 13, 2, device=device)
    embedding = torch.randn(1, 13, 3, device=device)
    assert validate_checkpoint(
        make_model(),
        make_model(),
        (functional_input, embedding),
    )


def test_flare_plus_plus_reference_checkpoint(device):
    """The committed v1 FLARE++ checkpoint remains loadable."""
    checkpoint = Path(__file__).parent / "data/flare_plus_plus_v1.mdlus"
    if not checkpoint.exists():
        torch.manual_seed(0)
        FLAREPlusPlus(
            functional_dim=2,
            out_dim=1,
            embedding_dim=3,
            n_layers=2,
            n_hidden=16,
            n_head=4,
            mlp_ratio=1,
            slice_num=4,
        ).save(checkpoint)
        raise IOError(
            f"Reference checkpoint {checkpoint} was missing and has been created; "
            "commit it and re-run the test."
        )
    model = Module.from_checkpoint(checkpoint).to(device)
    torch.manual_seed(4321)
    functional_input = torch.randn(1, 13, 2).to(device)
    embedding = torch.randn(1, 13, 3).to(device)
    with torch.no_grad():
        output = model(functional_input, embedding)
    reference = _load_or_create_output_reference(
        "flare_plus_plus_checkpoint_output.pth", output
    )

    assert isinstance(model, FLAREPlusPlus)
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def test_flare_plus_plus_torch_compile_fullgraph(device):
    """The standalone FLARE++ model supports full-graph compilation."""
    torch._dynamo.config.error_on_recompile = True
    model = FLAREPlusPlus(
        functional_dim=2,
        out_dim=1,
        embedding_dim=3,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        mlp_ratio=1,
        slice_num=4,
    ).to(device)
    functional_input = torch.randn(1, 13, 2, device=device)
    embedding = torch.randn(1, 13, 3, device=device)
    expected = model(functional_input, embedding)
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled = torch.compile(model, backend=backend, fullgraph=True)

    actual = compiled(functional_input, embedding)
    repeated = compiled(functional_input, embedding)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)


def test_flare_activation_checkpointing_matches_outputs_and_gradients(device):
    """Checkpointed FLARE blocks preserve outputs and gradients."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=3,
        n_hidden=16,
        dropout=0.0,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        structured_shape=None,
        use_te=False,
    )
    torch.manual_seed(0)
    plain, checkpointed = _make_checkpointing_model_pair(kwargs, device)

    fx_plain = torch.randn(2, 24, 2, device=device, requires_grad=True)
    embedding_plain = torch.randn(2, 24, 3, device=device, requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    embedding_checkpointed = embedding_plain.detach().clone().requires_grad_(True)

    output_plain = plain(fx_plain, embedding=embedding_plain)
    output_checkpointed = checkpointed(
        fx_checkpointed, embedding=embedding_checkpointed
    )
    torch.testing.assert_close(output_checkpointed, output_plain, atol=1e-6, rtol=1e-5)

    output_plain.square().mean().backward()
    output_checkpointed.square().mean().backward()
    torch.testing.assert_close(fx_checkpointed.grad, fx_plain.grad)
    torch.testing.assert_close(embedding_checkpointed.grad, embedding_plain.grad)
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


def test_flare_activation_checkpointing_torch_compile(device):
    """torch.compile preserves checkpointed FLARE outputs and gradients."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=2,
        n_hidden=16,
        dropout=0.0,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        structured_shape=None,
        use_te=False,
    )
    plain, checkpointed = _make_checkpointing_model_pair(kwargs, device)
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled_plain = torch.compile(plain, backend=backend, fullgraph=True)
    compiled_checkpointed = torch.compile(checkpointed, backend=backend, fullgraph=True)

    fx_plain = torch.randn(2, 24, 2, device=device, requires_grad=True)
    embedding_plain = torch.randn(2, 24, 3, device=device, requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    embedding_checkpointed = embedding_plain.detach().clone().requires_grad_(True)

    output_plain = compiled_plain(fx_plain, embedding=embedding_plain)
    output_plain.square().mean().backward()
    output_checkpointed = compiled_checkpointed(
        fx_checkpointed, embedding=embedding_checkpointed
    )
    output_checkpointed.square().mean().backward()
    torch.testing.assert_close(output_checkpointed, output_plain, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fx_checkpointed.grad, fx_plain.grad)
    torch.testing.assert_close(embedding_checkpointed.grad, embedding_plain.grad)
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


@requires_module("transformer_engine>=2.14.0")
def test_flare_te_basic(device):
    """Test the full FLARE model with Transformer Engine enabled."""
    if device == "cpu":
        pytest.skip("Transformer Engine requires CUDA")

    torch.manual_seed(42)
    model = FLARE(
        functional_dim=2,
        out_dim=1,
        embedding_dim=3,
        n_layers=2,
        n_hidden=64,
        n_head=4,
        mlp_ratio=2,
        slice_num=7,
        use_te=True,
    ).to(device)
    functional_input = torch.randn(2, 19, 2, device=device)
    embedding = torch.randn(2, 19, 3, device=device)

    output = model(functional_input, embedding=embedding)
    assert output.shape == (2, 19, 1)
    assert not torch.isnan(output).any()
    assert model.use_te is True
    assert model.blocks[0].Attn.use_te is True


@requires_module("transformer_engine>=2.14.0")
def test_flare_te_activation_checkpointing(monkeypatch):
    """Checkpointed Transformer Engine FLARE preserves outputs and gradients."""
    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")

    monkeypatch.setenv("NVTE_BIAS_GELU_NVFUSION", "0")
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        mlp_ratio=2,
        slice_num=8,
        structured_shape=None,
        use_te=True,
    )
    torch.manual_seed(0)
    plain, checkpointed = _make_checkpointing_model_pair(kwargs, "cuda")
    fx_plain = torch.randn(2, 128, 2, device="cuda", requires_grad=True)
    embedding_plain = torch.randn(2, 128, 3, device="cuda", requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    embedding_checkpointed = embedding_plain.detach().clone().requires_grad_(True)

    output_plain = plain(fx_plain, embedding=embedding_plain)
    output_plain.square().mean().backward()
    output_checkpointed = checkpointed(
        fx_checkpointed, embedding=embedding_checkpointed
    )
    output_checkpointed.square().mean().backward()

    torch.testing.assert_close(output_checkpointed, output_plain, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        fx_checkpointed.grad, fx_plain.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        embedding_checkpointed.grad,
        embedding_plain.grad,
        atol=1e-5,
        rtol=1e-4,
    )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-5, rtol=1e-4)


def test_flare_2d_forward(device):
    """Test FLARE 2D forward pass"""
    torch.manual_seed(0)
    model = FLARE(
        structured_shape=(85, 85),
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=1,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=True,
    ).to(device)

    bsize = 4

    fx = torch.randn(bsize, 85 * 85, 1).to(device)
    embedding = torch.randn(bsize, 85, 85).to(device)

    assert validate_forward_accuracy(
        model,
        (
            fx,
            embedding,
        ),
        file_name="models/flare/data/flare_2d_output.pth",
        atol=2e-3,
    )


def test_flare_irregular_forward(device):
    """Test FLARE irregular forward pass"""
    torch.manual_seed(0)
    model = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    bsize = 4

    embedding = torch.randn(bsize, 12345, 3).to(device)
    functional_input = torch.randn(bsize, 12345, 2).to(device)

    assert validate_forward_accuracy(
        model,
        (
            embedding,
            functional_input,
        ),
        file_name="models/flare/data/flare_irregular_output.pth",
        atol=1e-3,
    )


def test_flare_optims(device):
    """Test FLARE optimizations"""

    def setup_model():
        """Set up fresh FLARE model and inputs for each optim test"""

        model = FLARE(
            structured_shape=None,
            n_layers=8,
            n_hidden=64,
            dropout=0,
            n_head=4,
            time_input=False,
            act="gelu",
            mlp_ratio=1,
            functional_dim=2,
            embedding_dim=3,
            out_dim=1,
            slice_num=32,
            ref=1,
            unified_pos=False,
        ).to(device)

        if device == "cuda:0":
            bsize = 4
            n_points = 12345
        else:
            bsize = 1
            n_points = 123

        embedding = torch.randn(bsize, n_points, 3).to(device)
        functional_input = torch.randn(bsize, n_points, 2).to(device)

        return model, embedding, functional_input

    # Ideally always check graphs first
    model, pos, invar = setup_model()
    assert validate_cuda_graphs(
        model,
        (
            pos,
            invar,
        ),
    )

    # Check JIT
    model, pos, invar = setup_model()
    assert validate_jit(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check AMP
    model, pos, invar = setup_model()
    assert validate_amp(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check Combo
    model, pos, invar = setup_model()
    assert validate_combo_optims(
        model,
        (
            pos,
            invar,
        ),
    )


def test_flare_checkpoint(device):
    """Test FLARE checkpoint save/load"""
    model_1 = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    model_2 = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    bsize = random.randint(1, 2)

    embedding = torch.randn(bsize, 12345, 3).to(device)
    functional_input = torch.randn(bsize, 12345, 2).to(device)

    assert validate_checkpoint(
        model_1,
        model_2,
        (
            functional_input,
            embedding,
        ),
    )


def test_flare_activation_checkpointing_serialization(tmp_path):
    """FLARE checkpoint metadata preserves its checkpointing policy."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        n_layers=4,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
    )
    default_model = FLARE(**kwargs)
    checkpointed_model = FLARE(
        **kwargs, activation_checkpointing=True, checkpointing_ratio=0.5
    )
    checkpointed_model.load_state_dict(default_model.state_dict())
    assert checkpointed_model.state_dict().keys() == default_model.state_dict().keys()

    checkpoint_path = tmp_path / "flare_checkpointed.mdlus"
    checkpointed_model.save(checkpoint_path)
    restored = Module.from_checkpoint(checkpoint_path)
    assert isinstance(restored, FLARE)
    assert restored._activation_checkpointing_ratio == 0.5
    assert restored.state_dict().keys() == default_model.state_dict().keys()


@check_ort_version()
def test_flare_deploy(device):
    """Test FLARE deployment support"""
    model = FLARE(
        structured_shape=(85, 85),
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=1,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=True,
    ).to(device)

    bsize = 4

    pos = torch.randn(bsize, 85 * 85, 1).to(device)
    invar = torch.randn(bsize, 85, 85).to(device)

    assert validate_onnx_export(
        model,
        (
            pos,
            invar,
        ),
    )
    assert validate_onnx_runtime(
        model,
        (
            pos,
            invar,
        ),
        1e-2,
        1e-2,
    )
