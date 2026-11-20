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

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.models.transolver import Transolver
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


def test_transolver_activation_checkpointing_matches_outputs_and_gradients(device):
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
        structured_shape=(4, 6),
        use_te=False,
        time_input=True,
        plus=True,
    )
    torch.manual_seed(1)
    plain = Transolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = Transolver(**kwargs, activation_checkpointing=True).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    spatial = (4, 6)
    fx_plain = torch.randn(2, *spatial, 2, device=device, requires_grad=True)
    emb_plain = torch.randn(2, *spatial, 3, device=device, requires_grad=True)
    fx_checkpointed = fx_plain.detach().clone().requires_grad_(True)
    emb_checkpointed = emb_plain.detach().clone().requires_grad_(True)
    time_plain = torch.rand(2, device=device, requires_grad=True)
    time_checkpointed = time_plain.detach().clone().requires_grad_(True)

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
    torch.testing.assert_close(
        time_checkpointed.grad, time_plain.grad, atol=1e-6, rtol=1e-5
    )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


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


def test_transolver_activation_checkpointing_torch_compile(device):
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
        structured_shape=None,
        use_te=False,
        plus=False,
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

    spatial = (16,)
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
    """New and legacy checkpoint formats preserve checkpointing behavior."""
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

    fx = torch.randn(2, 16, 2, requires_grad=True)
    embedding = torch.randn(2, 16, 3, requires_grad=True)
    expected = default_model(fx, embedding=embedding).detach()

    # Loading a full-object pickle does not invoke ``__init__``. Removing the
    # field reproduces an object serialized by the pre-checkpointing class.
    del default_model._activation_checkpointing_ratio
    restored = pickle.loads(  # noqa: S301 - trusted local fixture
        pickle.dumps(default_model)
    )

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
