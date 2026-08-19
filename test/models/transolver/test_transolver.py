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

import gc
import pickle
import random
from types import SimpleNamespace

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.models.transolver import Transolver
from physicsnemo.models.transolver import transolver as transolver_module
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
) -> None:
    """Compare every named parameter gradient without relying on iteration order."""
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


@pytest.mark.parametrize(
    "config",
    ["default_structured", "custom_irregular"],
    ids=["with_defaults_structured", "with_custom_irregular"],
)
def test_transolver_constructor(config):
    """Test Transolver model constructor and attributes per MOD-008a."""
    if config == "default_structured":
        # Test with structured 2D data and default parameters
        model = Transolver(
            functional_dim=3,
            out_dim=1,
            structured_shape=(64, 64),
            unified_pos=True,
            use_te=False,
        )
        # Verify default attribute values
        assert model.n_hidden == 256, "Default n_hidden should be 256"
        assert model.time_input is False, "Default time_input should be False"
        assert model.unified_pos is True
        assert model.structured_shape == (64, 64)
        assert model.embedding_dim == 64  # ref * ref = 8 * 8 = 64
        assert len(model.blocks) == 4, "Default n_layers should be 4"
    else:
        # Test with irregular mesh data and custom parameters
        model = Transolver(
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
            use_te=False,
            time_input=True,
            plus=True,
        )
        # Verify custom attribute values
        assert model.n_hidden == 64
        assert model.time_input is True
        assert model.unified_pos is False
        assert model.structured_shape is None
        assert model.embedding_dim == 3
        assert len(model.blocks) == 8

    # Common assertions for all configurations
    assert isinstance(model, Module), (
        "Transolver should inherit from physicsnemo.Module"
    )
    assert hasattr(model, "preprocess"), "Model should have preprocess MLP"
    assert hasattr(model, "blocks"), "Model should have transformer blocks"
    assert hasattr(model, "meta"), "Model should have metadata"


@pytest.mark.parametrize(
    "enabled,ratio,expected",
    [
        (False, 1.0, 0.0),
        (False, 0.5, 0.0),
        (True, 0.0, 0.0),
        (True, 0.5, 0.5),
        (True, 1.0, 1.0),
    ],
)
def test_transolver_activation_checkpointing_configuration(enabled, ratio, expected):
    """Checkpointing uses a boolean switch and an interleaved block ratio."""
    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        n_layers=4,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
        activation_checkpointing=enabled,
        checkpointing_ratio=ratio,
    )

    assert model._activation_checkpointing_ratio == expected
    model.train()
    expected_masks = {
        0.0: [False, False, False, False],
        0.5: [True, False, True, False],
        1.0: [True, True, True, True],
    }
    assert [model._should_checkpoint_block(i) for i in range(len(model.blocks))] == (
        expected_masks[expected]
    )

    # Activation checkpointing is a training-time memory optimization.
    model.eval()
    assert not any(model._should_checkpoint_block(i) for i in range(len(model.blocks)))


@pytest.mark.parametrize("value", [0, 0.5, 1, "all", None])
def test_transolver_activation_checkpointing_rejects_non_boolean_switch(value):
    """The checkpointing switch accepts booleans only."""
    with pytest.raises(TypeError, match="activation_checkpointing"):
        Transolver(
            functional_dim=2,
            embedding_dim=3,
            out_dim=1,
            n_hidden=16,
            n_head=4,
            structured_shape=None,
            use_te=False,
            activation_checkpointing=value,
        )


@pytest.mark.parametrize(
    "value,error",
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (True, TypeError),
        ("all", TypeError),
        (None, TypeError),
    ],
)
def test_transolver_activation_checkpointing_rejects_invalid_ratio(value, error):
    """Invalid checkpointing ratios fail during model construction."""
    with pytest.raises(error, match="checkpointing_ratio"):
        Transolver(
            functional_dim=2,
            embedding_dim=3,
            out_dim=1,
            n_hidden=16,
            n_head=4,
            structured_shape=None,
            use_te=False,
            activation_checkpointing=True,
            checkpointing_ratio=value,
        )


@pytest.mark.parametrize(
    "structured_shape,plus,time_input",
    [
        (None, False, False),
        (None, True, True),
        ((4, 5), False, False),
        ((3, 3, 2), True, False),
    ],
    ids=["irregular", "irregular_plus_time", "structured_2d", "structured_3d_plus"],
)
def test_transolver_activation_checkpointing_matches_outputs_and_gradients(
    device, structured_shape, plus, time_input
):
    """Checkpointed blocks reproduce outputs and gradients, including RNG use."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=3,
        n_hidden=16,
        dropout=0.2,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        structured_shape=structured_shape,
        use_te=False,
        time_input=time_input,
        plus=plus,
    )
    torch.manual_seed(1)
    plain = Transolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = Transolver(**kwargs, activation_checkpointing=True).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    spatial = structured_shape if structured_shape is not None else (24,)
    fx_plain = torch.randn(2, *spatial, 2, device=device, requires_grad=True)
    emb_plain = torch.randn(2, *spatial, 3, device=device, requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    emb_checkpointed = emb_plain.detach().clone().requires_grad_(True)
    time_plain = (
        torch.rand(2, device=device, requires_grad=True) if time_input else None
    )
    time_checkpointed = (
        time_plain.detach().clone().requires_grad_(True) if time_input else None
    )

    # Reset the RNG because dropout and Transolver++ slice routing are stochastic.
    torch.manual_seed(7)
    out_plain = plain(fx_plain, embedding=emb_plain, time=time_plain)
    torch.manual_seed(7)
    out_checkpointed = checkpointed(
        fx_checkpointed, embedding=emb_checkpointed, time=time_checkpointed
    )
    torch.testing.assert_close(out_checkpointed, out_plain, atol=1e-6, rtol=1e-5)

    out_plain.square().mean().backward()
    out_checkpointed.square().mean().backward()

    torch.testing.assert_close(
        fx_checkpointed.grad, fx_plain.grad, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        emb_checkpointed.grad, emb_plain.grad, atol=1e-6, rtol=1e-5
    )
    if time_input:
        torch.testing.assert_close(
            time_checkpointed.grad, time_plain.grad, atol=1e-6, rtol=1e-5
        )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("plus", [False, True], ids=["standard", "plus"])
def test_transolver_activation_checkpointing_reduces_saved_activations(plus):
    """Checkpointing saves fewer forward activations for backward."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=4,
        n_hidden=32,
        n_head=4,
        mlp_ratio=2,
        slice_num=8,
        structured_shape=None,
        use_te=False,
        plus=plus,
    )
    plain = Transolver(**kwargs, activation_checkpointing=False)
    checkpointed = Transolver(**kwargs, activation_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    fx = torch.randn(2, 256, 2)
    embedding = torch.randn(2, 256, 3)

    def saved_activation_bytes(model):
        total = 0

        def pack(tensor):
            nonlocal total
            total += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model(fx, embedding=embedding).square().mean()
        return total

    plain_bytes = saved_activation_bytes(plain)
    checkpointed_bytes = saved_activation_bytes(checkpointed)
    assert checkpointed_bytes < 0.5 * plain_bytes, (
        "checkpointing did not cut saved activation bytes by at least 50%: "
        f"{checkpointed_bytes} bytes versus {plain_bytes} bytes"
    )


def test_transolver_activation_checkpointing_recomputes_selected_blocks(monkeypatch):
    """Only the selected interleaved blocks are recomputed during backward."""
    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=4,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
    )
    model.train()
    call_counts = [0] * len(model.blocks)

    for block_idx, block in enumerate(model.blocks):
        original_forward = block.forward

        def counting_forward(fx, idx=block_idx, forward=original_forward):
            call_counts[idx] += 1
            return forward(fx)

        monkeypatch.setattr(block, "forward", counting_forward)

    fx = torch.randn(2, 16, 2)
    embedding = torch.randn(2, 16, 3)
    model(fx, embedding=embedding).square().mean().backward()
    assert call_counts == [2, 1, 2, 1]


def test_transolver_activation_checkpointing_uses_te_wrapper(monkeypatch):
    """TE blocks route through TE's state-aware non-reentrant checkpoint."""
    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
        activation_checkpointing=True,
    )
    model.train()
    calls = []

    def fake_te_checkpoint(function, *args, **kwargs):
        calls.append((function, kwargs))
        return function(*args)

    monkeypatch.setattr(
        transolver_module,
        "te",
        SimpleNamespace(checkpoint=fake_te_checkpoint),
    )
    # Construction with use_te=True requires the optional CUDA package. Flip
    # only the routing flag so this backend-selection invariant is testable on
    # every CI platform; the CUDA integration test below exercises real TE.
    model.use_te = True

    fx = torch.randn(2, 16, 2)
    embedding = torch.randn(2, 16, 3)
    model(fx, embedding=embedding)

    assert [function for function, _ in calls] == list(model.blocks)
    assert all(kwargs == {"use_reentrant": False} for _, kwargs in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_transolver_activation_checkpointing_reduces_peak_cuda_memory():
    """Checkpointing lowers peak allocated CUDA memory for a training step."""

    def peak_step_bytes(activation_checkpointing):
        torch.cuda.empty_cache()
        model = Transolver(
            functional_dim=2,
            embedding_dim=3,
            out_dim=4,
            n_layers=6,
            n_hidden=128,
            n_head=8,
            mlp_ratio=4,
            slice_num=32,
            structured_shape=None,
            use_te=False,
            activation_checkpointing=activation_checkpointing,
        ).to("cuda")
        model.train()
        fx = torch.randn(1, 4096, 2, device="cuda")
        embedding = torch.randn(1, 4096, 3, device="cuda")

        # Warm up each policy before resetting the allocator peak so one-time
        # kernel initialization is excluded symmetrically.
        model(fx, embedding=embedding).square().mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        model(fx, embedding=embedding).square().mean().backward()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline

        del model, fx, embedding
        gc.collect()
        torch.cuda.empty_cache()
        return peak_delta

    plain_peak = peak_step_bytes(False)
    checkpointed_peak = peak_step_bytes(True)
    assert checkpointed_peak < plain_peak, (
        f"checkpointing peaked at {checkpointed_peak} bytes versus {plain_peak} bytes"
    )


@pytest.mark.parametrize(
    "structured_shape,plus",
    [(None, False), ((4, 4), True)],
    ids=["irregular", "structured_2d_plus"],
)
def test_transolver_activation_checkpointing_torch_compile(
    device, structured_shape, plus
):
    """torch.compile preserves checkpointed output and gradient numerics."""
    kwargs = dict(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=2,
        n_hidden=16,
        dropout=0.2,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        structured_shape=structured_shape,
        use_te=False,
        plus=plus,
    )
    plain = Transolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = Transolver(**kwargs, activation_checkpointing=True).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    # AOTAutograd exercises the checkpointed backward path quickly on CPU;
    # CUDA uses the recipe's real inductor backend.
    compile_backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled_plain = torch.compile(plain, backend=compile_backend, fullgraph=True)
    compiled_checkpointed = torch.compile(
        checkpointed, backend=compile_backend, fullgraph=True
    )

    spatial = structured_shape if structured_shape is not None else (16,)
    fx_plain = torch.randn(2, *spatial, 2, device=device, requires_grad=True)
    emb_plain = torch.randn(2, *spatial, 3, device=device, requires_grad=True)
    fx_compiled = fx_plain.detach().clone().requires_grad_(True)
    emb_compiled = emb_plain.detach().clone().requires_grad_(True)

    torch.manual_seed(11)
    out_plain = compiled_plain(fx_plain, embedding=emb_plain)
    out_plain.square().mean().backward()
    torch.manual_seed(11)
    out_compiled = compiled_checkpointed(fx_compiled, embedding=emb_compiled)
    out_compiled.square().mean().backward()
    torch.testing.assert_close(out_compiled, out_plain, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fx_compiled.grad, fx_plain.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(emb_compiled.grad, emb_plain.grad, atol=1e-6, rtol=1e-5)
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


def test_transolver_activation_checkpointing_amp(device):
    """Checkpointed blocks support mixed-precision training."""
    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
        activation_checkpointing=True,
    ).to(device)
    fx = torch.randn(2, 16, 2, device=device)
    embedding = torch.randn(2, 16, 3, device=device)
    assert validate_amp(model, (fx, embedding), iterations=1)


def test_transolver2d_forward(device):
    """Test Transolver2D forward pass"""
    torch.manual_seed(0)
    # Construct Transolver model
    model = Transolver(
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
        use_te=False,
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
        file_name="models/transolver/data/transolver2d_output.pth",
        atol=2e-3,
    )


def test_transolver_irregular_forward(device):
    """Test Transolver Irregular forward pass"""
    torch.manual_seed(0)
    # Construct Transolver model
    model = Transolver(
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
        use_te=False,
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
        file_name="models/transolver/data/transolver_irregular_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "spatial",
    [(16, 16), (8, 8, 8)],
    ids=["structured_2d", "structured_3d"],
)
def test_transolver_structured_nonunified_spatial_embedding(device, spatial):
    """Structured (unified_pos=False) models accept spatially-shaped embeddings.

    Regression test: a spatially-shaped embedding ``(B, *spatial, C_emb)`` must
    be flattened internally to align with ``fx`` rather than crashing in the
    concatenation. Also checks that passing a spatial embedding is equivalent
    to passing its pre-flattened ``(B, N, C_emb)`` form.
    """
    torch.manual_seed(0)
    batch_size, functional_dim, embedding_dim, out_dim = 2, 3, 4, 2

    model = Transolver(
        functional_dim=functional_dim,
        out_dim=out_dim,
        embedding_dim=embedding_dim,
        structured_shape=spatial,
        unified_pos=False,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        use_te=False,
    ).to(device)
    model.eval()

    fx_spatial = torch.randn(batch_size, *spatial, functional_dim).to(device)
    emb_spatial = torch.randn(batch_size, *spatial, embedding_dim).to(device)

    # Spatially-shaped inputs: output should keep fx's spatial layout.
    out_spatial = model(fx_spatial, embedding=emb_spatial)
    assert out_spatial.shape == (batch_size, *spatial, out_dim)

    # Pre-flattened inputs should give an identical result (same row-major flatten).
    fx_flat = fx_spatial.reshape(batch_size, -1, functional_dim)
    emb_flat = emb_spatial.reshape(batch_size, -1, embedding_dim)
    out_flat = model(fx_flat, embedding=emb_flat)
    assert out_flat.shape == (batch_size, fx_flat.shape[1], out_dim)
    assert torch.allclose(
        out_spatial.reshape(batch_size, -1, out_dim), out_flat, atol=1e-6
    )


def test_transolver_optims(device):
    """Test transolver optimizations"""

    def setup_model():
        """Setups up fresh transolver model and inputs for each optim test"""

        model = Transolver(
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
            use_te=False,
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


@requires_module("transformer_engine")
def test_transolver_te(pytestconfig):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)

    kwargs = dict(
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
        use_te=True,
    )
    model = Transolver(**kwargs).to("cuda")

    bsize = 4

    embedding = torch.randn(bsize, 12345, 3).to("cuda")
    functional_input = torch.randn(bsize, 12345, 2).to("cuda")

    assert validate_forward_accuracy(
        model,
        (
            embedding,
            functional_input,
        ),
        file_name="models/transolver/data/transolver_irregular_te_output.pth",
        atol=1e-3,
    )


@requires_module("transformer_engine")
def test_transolver_te_activation_checkpointing(monkeypatch):
    """Checkpointed TE blocks preserve outputs and gradients."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Transformer Engine disables the fused bias+GELU path while executing a
    # non-reentrant checkpoint. Construct both comparison models with that
    # path disabled so this test compares checkpointing rather than different
    # TE kernels.
    monkeypatch.setenv("NVTE_BIAS_GELU_NVFUSION", "0")
    torch.manual_seed(0)

    kwargs = dict(
        structured_shape=None,
        n_layers=2,
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
        use_te=True,
    )
    plain = Transolver(**kwargs, activation_checkpointing=False).to("cuda")
    checkpointed = Transolver(**kwargs, activation_checkpointing=True).to("cuda")
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    # Compare the actual checkpointed backward path on a small workload.
    fx_plain = torch.randn(2, 512, 2, device="cuda", requires_grad=True)
    emb_plain = torch.randn(2, 512, 3, device="cuda", requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    emb_checkpointed = emb_plain.detach().clone().requires_grad_(True)

    out_plain = plain(fx_plain, emb_plain)
    out_plain.square().mean().backward()
    out_checkpointed = checkpointed(fx_checkpointed, emb_checkpointed)
    out_checkpointed.square().mean().backward()

    torch.testing.assert_close(out_checkpointed, out_plain, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        fx_checkpointed.grad, fx_plain.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        emb_checkpointed.grad, emb_plain.grad, atol=1e-5, rtol=1e-4
    )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-5, rtol=1e-4)


@requires_module("transformer_engine")
def test_transolver_te_fp8_activation_checkpointing(monkeypatch):
    """Checkpointed TE blocks preserve FP8 outputs and gradients."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import transformer_engine.pytorch as te_runtime
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    fp8_available, reason = te_runtime.is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason)

    # Keep the reference and checkpointed executions on the same deterministic
    # TE kernel path. TE also applies this restriction internally during
    # non-reentrant recomputation.
    monkeypatch.setenv("NVTE_BIAS_GELU_NVFUSION", "0")

    kwargs = dict(
        structured_shape=None,
        n_layers=2,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=2,
        functional_dim=2,
        embedding_dim=14,
        out_dim=16,
        slice_num=16,
        ref=1,
        unified_pos=False,
        use_te=True,
    )
    dtype = torch.bfloat16
    torch.manual_seed(0)
    plain = Transolver(**kwargs, activation_checkpointing=False).to(
        device="cuda", dtype=dtype
    )
    checkpointed = Transolver(**kwargs, activation_checkpointing=True).to(
        device="cuda", dtype=dtype
    )
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    fx_plain = torch.randn(2, 32, 2, device="cuda", dtype=dtype, requires_grad=True)
    emb_plain = torch.randn(2, 32, 14, device="cuda", dtype=dtype, requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    emb_checkpointed = emb_plain.detach().clone().requires_grad_(True)

    def run(model, functional_input, embedding):
        FP8GlobalStateManager.reset()
        with te_runtime.autocast(enabled=True, recipe=te_recipe.DelayedScaling()):
            output = model(functional_input, embedding)
        output.float().square().mean().backward()
        return output

    out_plain = run(plain, fx_plain, emb_plain)
    out_checkpointed = run(checkpointed, fx_checkpointed, emb_checkpointed)

    # Match Transformer Engine's own full-recompute FP8 tolerances.
    tolerances = dict(atol=0.0675, rtol=0.125)
    torch.testing.assert_close(out_checkpointed, out_plain, **tolerances)
    torch.testing.assert_close(fx_checkpointed.grad, fx_plain.grad, **tolerances)
    torch.testing.assert_close(emb_checkpointed.grad, emb_plain.grad, **tolerances)
    _assert_parameter_gradients_close(checkpointed, plain, **tolerances)


def test_transolver_checkpoint(device):
    """Test transolver checkpoint save/load"""
    # Construct transolver models
    model_1 = Transolver(
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
        use_te=False,
    ).to(device)

    model_2 = Transolver(
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
        use_te=False,
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


def test_transolver_activation_checkpointing_serialization(tmp_path):
    """The checkpointing policy round-trips without adding state-dict keys."""
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
    default_model = Transolver(**kwargs)
    checkpointed_model = Transolver(
        **kwargs, activation_checkpointing=True, checkpointing_ratio=0.5
    )
    checkpointed_model.load_state_dict(default_model.state_dict())
    assert checkpointed_model.state_dict().keys() == default_model.state_dict().keys()

    checkpoint_path = tmp_path / "transolver_checkpointed.mdlus"
    checkpointed_model.save(checkpoint_path)
    restored = Module.from_checkpoint(checkpoint_path)
    assert isinstance(restored, Transolver)
    assert restored._activation_checkpointing_ratio == 0.5
    assert restored.state_dict().keys() == default_model.state_dict().keys()

    # Simulate constructor metadata from a checkpoint written before the new
    # optional argument existed. Instantiation must fall back to disabled.
    legacy_args = {
        **default_model._args,
        "__args__": default_model._args["__args__"].copy(),
    }
    legacy_args["__args__"].pop("activation_checkpointing")
    legacy_args["__args__"].pop("checkpointing_ratio")
    legacy_restored = Module.instantiate(legacy_args)
    assert isinstance(legacy_restored, Transolver)
    assert legacy_restored._activation_checkpointing_ratio == 0.0


def test_transolver_legacy_full_object_pickle_defaults_checkpointing_off():
    """Full-object pickles made before the new attribute remain executable."""
    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        structured_shape=None,
        use_te=False,
    )
    fx = torch.randn(2, 16, 2, requires_grad=True)
    embedding = torch.randn(2, 16, 3, requires_grad=True)
    expected = model(fx, embedding=embedding).detach()

    # Loading a full-object pickle does not invoke ``__init__``. Removing the
    # field reproduces an object serialized by the pre-checkpointing class.
    del model._activation_checkpointing_ratio
    restored = pickle.loads(pickle.dumps(model))  # noqa: S301 - trusted local fixture

    assert not hasattr(restored, "_activation_checkpointing_ratio")
    restored_fx = fx.detach().clone().requires_grad_(True)
    restored_embedding = embedding.detach().clone().requires_grad_(True)
    actual = restored(restored_fx, embedding=restored_embedding)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert restored_fx.grad is not None
    assert restored_embedding.grad is not None
    assert all(parameter.grad is not None for parameter in restored.parameters())


@check_ort_version()
def test_transolver_deploy(device):
    """Test transolver deployment support"""
    # Construct transolver model
    model = Transolver(
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
        use_te=False,
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
            invar,
            invar,
        ),
        1e-2,
        1e-2,
    )
