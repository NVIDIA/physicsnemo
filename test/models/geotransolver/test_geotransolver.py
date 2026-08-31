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
import importlib
import pickle
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from physicsnemo.core.module import Module
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.models.geotransolver import geotransolver as geotransolver_module
from physicsnemo.models.geotransolver.geotransolver import (
    GeoTransolver,
)
from test.common import (  # noqa E402
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_cuda_graphs,
    validate_forward_accuracy,
    validate_jit,
)
from test.conftest import requires_module

# =============================================================================
# GeoTransolver End-to-End Model Tests
# =============================================================================


def _flatten_outputs(output):
    return (output,) if isinstance(output, torch.Tensor) else tuple(output)


def _output_loss(output):
    return sum(tensor.square().mean() for tensor in _flatten_outputs(output))


def _assert_parameter_gradients_close(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    atol: float,
    rtol: float,
):
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


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA"])
@pytest.mark.parametrize("use_geometry", [False, True])
@pytest.mark.parametrize("use_global", [False, True])
def test_geotransolver_forward(device, attention_type, use_geometry, use_global):
    """Test GeoTransolver model forward pass with optional geometry and global context."""
    torch.manual_seed(42)

    batch_size = 2
    n_tokens = 100
    n_geom_tokens = 345
    n_global = 5
    geometry_dim = 3
    global_dim = 16

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=geometry_dim if use_geometry else None,
        global_dim=global_dim if use_global else None,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
        attention_type=attention_type,
    ).to(device)

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    local_positions = local_emb[:, :, :3]
    kwargs = {}
    if use_geometry:
        kwargs["geometry"] = torch.randn(batch_size, n_geom_tokens, geometry_dim).to(
            device
        )
    if use_global:
        kwargs["global_embedding"] = torch.randn(batch_size, n_global, global_dim).to(
            device
        )

    outputs = model(local_emb, local_positions, **kwargs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


def test_geotransolver_forward_returns_embedding_states(device):
    """Test returning geometry and global context embedding states."""
    torch.manual_seed(42)

    batch_size = 2
    n_tokens = 100
    n_geom_tokens = 345
    n_global = 5
    n_hidden = 64
    n_head = 4
    slice_num = 8

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=n_hidden,
        dropout=0.0,
        n_head=n_head,
        act="gelu",
        mlp_ratio=2,
        slice_num=slice_num,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    local_emb = torch.randn(batch_size, n_tokens, 32, device=device)
    geometry = torch.randn(batch_size, n_geom_tokens, 3, device=device)
    global_emb = torch.randn(batch_size, n_global, 16, device=device)

    outputs, embedding_states = model(
        local_emb,
        global_embedding=global_emb,
        geometry=geometry,
        return_embedding_states=True,
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()
    assert isinstance(embedding_states, torch.Tensor)
    assert embedding_states.shape == (
        batch_size,
        n_head,
        slice_num,
        2 * (n_hidden // n_head),
    )
    assert not torch.isnan(embedding_states).any()


def _small_model(device, **overrides):
    """Small GeoTransolver for the return-option tests."""
    kwargs = dict(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    )
    kwargs.update(overrides)
    return GeoTransolver(**kwargs).to(device)


def test_geotransolver_return_options(device):
    """Return flags preserve signatures and checkpointed gradients.

    All four flag combinations must agree on the prediction; the full return
    tuple must also remain numerically equivalent under checkpointing.
    """
    torch.manual_seed(42)

    batch_size = 2
    n_tokens = 100
    n_hidden = 64
    model = _small_model(device, n_hidden=n_hidden).eval()

    local_emb = torch.randn(batch_size, n_tokens, 32, device=device)
    geometry = torch.randn(batch_size, 235, 3, device=device)
    global_emb = torch.randn(batch_size, 5, 16, device=device)
    inputs = dict(global_embedding=global_emb, geometry=geometry)

    with torch.no_grad():
        plain = model(local_emb, **inputs)
        out_states, states = model(local_emb, **inputs, return_embedding_states=True)
        out_feats, point_features = model(
            local_emb, **inputs, return_point_features=True
        )
        out_both, states_both, feats_both = model(
            local_emb,
            **inputs,
            return_embedding_states=True,
            return_point_features=True,
        )

    assert isinstance(plain, torch.Tensor)
    assert plain.shape == (batch_size, n_tokens, 4)

    # Point features are the pre-readout latents, so n_hidden wide here.
    assert point_features.shape == (batch_size, n_tokens, n_hidden)
    assert feats_both.shape == point_features.shape
    assert states_both.shape == states.shape

    # The flags must not perturb the prediction, in any combination.
    for other in (out_states, out_feats, out_both):
        assert torch.equal(plain, other)

    checkpointed = _small_model(
        device,
        n_hidden=n_hidden,
        activation_checkpointing=True,
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    )
    checkpointed.load_state_dict(model.state_dict())
    model.train()
    checkpointed.train()
    local_plain = local_emb.detach().clone().requires_grad_(True)
    geometry_plain = geometry.detach().clone().requires_grad_(True)
    global_plain = global_emb.detach().clone().requires_grad_(True)
    local_checkpointed = local_plain.detach().clone().requires_grad_(True)
    geometry_checkpointed = geometry_plain.detach().clone().requires_grad_(True)
    global_checkpointed = global_plain.detach().clone().requires_grad_(True)

    plain_values = model(
        local_plain,
        geometry=geometry_plain,
        global_embedding=global_plain,
        return_embedding_states=True,
        return_point_features=True,
    )
    checkpointed_values = checkpointed(
        local_checkpointed,
        geometry=geometry_checkpointed,
        global_embedding=global_checkpointed,
        return_embedding_states=True,
        return_point_features=True,
    )
    for actual, expected in zip(checkpointed_values, plain_values):
        torch.testing.assert_close(actual, expected)

    sum(value.square().mean() for value in plain_values).backward()
    sum(value.square().mean() for value in checkpointed_values).backward()
    torch.testing.assert_close(local_checkpointed.grad, local_plain.grad)
    torch.testing.assert_close(geometry_checkpointed.grad, geometry_plain.grad)
    torch.testing.assert_close(global_checkpointed.grad, global_plain.grad)
    _assert_parameter_gradients_close(checkpointed, model, atol=1e-6, rtol=1e-5)


def test_geotransolver_return_flags_are_keyword_only(device):
    """The two return flags cannot be passed positionally."""
    model = _small_model(device)
    local_emb = torch.randn(1, 16, 32, device=device)

    with pytest.raises(TypeError):
        model(local_emb, None, None, None, None, True)


def test_geotransolver_point_features_with_local_features(device):
    """Local features widen the point features by n_hidden_local per radius."""
    torch.manual_seed(42)

    n_hidden = 64
    n_hidden_local = 32
    radii = [0.05, 0.25]
    model = _small_model(
        device,
        n_hidden=n_hidden,
        include_local_features=True,
        radii=radii,
        neighbors_in_radius=[8, 32],
        n_hidden_local=n_hidden_local,
    ).eval()

    batch_size = 1
    n_tokens = 100
    local_emb = torch.randn(batch_size, n_tokens, 32, device=device)

    with torch.no_grad():
        outputs, point_features = model(
            local_emb,
            local_positions=local_emb[:, :, :3],
            global_embedding=torch.randn(batch_size, 5, 16, device=device),
            geometry=torch.randn(batch_size, 235, 3, device=device),
            return_point_features=True,
        )

    assert outputs.shape == (batch_size, n_tokens, 4)
    assert point_features.shape == (
        batch_size,
        n_tokens,
        n_hidden + n_hidden_local * len(radii),
    )
    assert not torch.isnan(point_features).any()


def test_geotransolver_point_features_tuple_inputs(device):
    """Tuple inputs return one point-feature tensor per stream."""
    torch.manual_seed(42)

    functional_dims = (32, 48)
    out_dims = (4, 6)
    n_hidden = 64
    model = _small_model(
        device,
        functional_dim=functional_dims,
        out_dim=out_dims,
        n_hidden=n_hidden,
    ).eval()

    batch_size = 2
    n_tokens = (100, 150)
    local_embs = tuple(
        torch.randn(batch_size, n, d, device=device)
        for n, d in zip(n_tokens, functional_dims)
    )

    with torch.no_grad():
        outputs, point_features = model(
            local_embs,
            local_positions=tuple(emb[:, :, :3] for emb in local_embs),
            global_embedding=torch.randn(batch_size, 5, 16, device=device),
            geometry=torch.randn(batch_size, 235, 3, device=device),
            return_point_features=True,
        )

    assert len(outputs) == len(point_features) == 2
    for i, n in enumerate(n_tokens):
        assert outputs[i].shape == (batch_size, n, out_dims[i])
        assert point_features[i].shape == (batch_size, n, n_hidden)


def test_geotransolver_forward_tuple_inputs(device):
    """Test GeoTransolver model forward pass with tuple inputs/outputs (multi-head)."""
    torch.manual_seed(42)

    functional_dims = (32, 48)
    out_dims = (4, 6)

    model = GeoTransolver(
        functional_dim=functional_dims,
        out_dim=out_dims,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    n_geom = 235
    n_global = 5

    local_emb_1 = torch.randn(batch_size, n_tokens_1, functional_dims[0]).to(device)
    local_emb_2 = torch.randn(batch_size, n_tokens_2, functional_dims[1]).to(device)
    local_positions_1 = local_emb_1[:, :, :3]
    local_positions_2 = local_emb_2[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        (local_emb_1, local_emb_2),
        local_positions=(local_positions_1, local_positions_2),
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert len(outputs) == 2
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs[0].shape == (batch_size, n_tokens_1, out_dims[0])
    assert outputs[1].shape == (batch_size, n_tokens_2, out_dims[1])
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


def test_geotransolver_forward_with_local_features(device, pytestconfig):
    """Test GeoTransolver model forward pass with local features (BQ warp)."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=True,
        radii=[0.05, 0.25],
        neighbors_in_radius=[8, 32],
        n_hidden_local=32,
    ).to(device)

    batch_size = 1
    n_tokens = 100
    n_global = 5
    n_geom = 235

    # For local features, the first 3 channels of local_emb should be coordinates
    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    local_positions = local_emb[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


@requires_module("warp")
def test_geotransolver_full_component_checkpointing_with_local_features(device):
    """Full checkpointing preserves local-feature outputs and gradients."""
    kwargs = dict(
        functional_dim=6,
        out_dim=4,
        geometry_dim=3,
        global_dim=3,
        n_layers=2,
        n_hidden=32,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=4,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=True,
        radii=[0.25],
        neighbors_in_radius=[8],
        n_hidden_local=16,
    )
    torch.manual_seed(42)
    plain = GeoTransolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = GeoTransolver(
        **kwargs,
        activation_checkpointing=True,
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    ).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    batch_size = 2
    local_plain = torch.randn(batch_size, 32, 6, device=device, requires_grad=True)
    geometry_plain = torch.randn(batch_size, 50, 3, device=device, requires_grad=True)
    global_plain = torch.randn(batch_size, 2, 3, device=device, requires_grad=True)
    local_checkpointed = local_plain.detach().clone().requires_grad_(True)
    geometry_checkpointed = geometry_plain.detach().clone().requires_grad_(True)
    global_checkpointed = global_plain.detach().clone().requires_grad_(True)

    output_plain = plain(
        local_plain,
        local_positions=local_plain[:, :, :3],
        global_embedding=global_plain,
        geometry=geometry_plain,
    )
    output_checkpointed = checkpointed(
        local_checkpointed,
        local_positions=local_checkpointed[:, :, :3],
        global_embedding=global_checkpointed,
        geometry=geometry_checkpointed,
    )
    assert output_plain.shape == (batch_size, 32, 4)
    assert torch.isfinite(output_plain).all()
    torch.testing.assert_close(output_checkpointed, output_plain)
    output_plain.square().mean().backward()
    output_checkpointed.square().mean().backward()
    torch.testing.assert_close(local_checkpointed.grad, local_plain.grad)
    torch.testing.assert_close(geometry_checkpointed.grad, geometry_plain.grad)
    torch.testing.assert_close(global_checkpointed.grad, global_plain.grad)
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


# =============================================================================
# Forward Accuracy Tests (reproducibility)
# =============================================================================


def test_geotransolver_forward_accuracy_basic(device):
    """Test GeoTransolver basic forward pass accuracy."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    local_positions = local_emb[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    assert validate_forward_accuracy(
        model,
        (local_emb, local_positions, global_emb, geometry),
        file_name="models/geotransolver/data/geotransolver_basic_output.pth",
        atol=1e-3,
    )


def test_geotransolver_forward_accuracy_tuple(device):
    """Test GeoTransolver forward pass accuracy with tuple inputs."""
    torch.manual_seed(42)

    functional_dims = (32, 48)
    out_dims = (4, 6)

    model = GeoTransolver(
        functional_dim=functional_dims,
        out_dim=out_dims,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    n_global = 5
    n_geom = 235

    local_emb_1 = torch.randn(batch_size, n_tokens_1, functional_dims[0]).to(device)
    local_emb_2 = torch.randn(batch_size, n_tokens_2, functional_dims[1]).to(device)

    local_positions_1 = local_emb_1[:, :, :3]
    local_positions_2 = local_emb_2[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    assert validate_forward_accuracy(
        model,
        (
            (local_emb_1, local_emb_2),
            (local_positions_1, local_positions_2),
            global_emb,
            geometry,
        ),
        file_name="models/geotransolver/data/geotransolver_tuple_output.pth",
        atol=2e-3,
    )


# =============================================================================
# Optimization Tests
# =============================================================================


def test_geotransolver_optimizations(device):
    """Test GeoTransolver optimizations (CUDA graphs, JIT, AMP, combo)."""
    torch.manual_seed(42)

    def setup_model():
        """Setup fresh GeoTransolver model and inputs for each optimization test."""
        model = GeoTransolver(
            functional_dim=32,
            out_dim=4,
            geometry_dim=3,
            global_dim=16,
            n_layers=2,
            n_hidden=64,
            dropout=0.0,
            n_head=4,
            act="gelu",
            mlp_ratio=2,
            slice_num=8,
            use_te=False,
            time_input=False,
            plus=False,
            include_local_features=False,
        ).to(device)

        batch_size = 2
        n_tokens = 100
        n_global = 5

        local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
        geometry = torch.randn(batch_size, n_tokens, 3).to(device)
        global_emb = torch.randn(batch_size, n_global, 16).to(device)
        local_positions = local_emb[:, :, :3]
        return model, local_emb, local_positions, global_emb, geometry

    # Check CUDA graphs
    model, local_emb, local_positions, global_emb, geometry = setup_model()

    assert validate_cuda_graphs(
        model,
        (local_emb, local_positions, global_emb, geometry),
    )

    # Check JIT
    model, local_emb, local_positions, global_emb, geometry = setup_model()
    assert validate_jit(
        model,
        (local_emb, local_positions, global_emb, geometry),
    )

    # Check AMP
    model, local_emb, local_positions, global_emb, geometry = setup_model()
    assert validate_amp(
        model,
        (local_emb, local_positions, global_emb, geometry),
    )

    # Check Combo
    model, local_emb, local_positions, global_emb, geometry = setup_model()
    assert validate_combo_optims(
        model,
        (local_emb, local_positions, global_emb, geometry),
    )


# =============================================================================
# Transformer Engine Tests
# =============================================================================


@requires_module("transformer_engine")
def test_geotransolver_te_basic(device, pytestconfig):
    """Test GeoTransolver with Transformer Engine backend."""
    torch.manual_seed(42)

    if device == "cpu":
        pytest.skip("TE Tests require cuda.")

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=True,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)
    local_positions = local_emb[:, :, :3]

    outputs = model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


@requires_module("transformer_engine")
def test_geotransolver_te_gale_fa(device):
    """Test GeoTransolver with the GALE_FA backend and Transformer Engine.

    Exercises the TE attention path in GALE_FA (both the FLARE self-attention
    passes and the context cross-attention run through te.DotProductAttention).
    """
    torch.manual_seed(42)

    if device == "cpu":
        pytest.skip("TE Tests require cuda.")

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=True,
        time_input=False,
        plus=False,
        include_local_features=False,
        attention_type="GALE_FA",
    ).to(device)

    assert model.blocks[0].Attn.use_te is True

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)
    local_positions = local_emb[:, :, :3]

    outputs = model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


@requires_module("transformer_engine")
def test_geotransolver_te_activation_checkpointing(device, monkeypatch):
    """Full checkpointing preserves TE GALE_FA outputs and gradients."""
    if device == "cpu":
        pytest.skip("TE Tests require cuda.")

    monkeypatch.setenv("NVTE_BIAS_GELU_NVFUSION", "0")
    kwargs = dict(
        functional_dim=6,
        out_dim=4,
        geometry_dim=3,
        global_dim=3,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=16,
        use_te=True,
        time_input=False,
        plus=False,
        include_local_features=False,
        attention_type="GALE_FA",
    )
    torch.manual_seed(42)
    plain = GeoTransolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = GeoTransolver(
        **kwargs,
        activation_checkpointing=True,
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    ).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    local_plain = torch.randn(2, 64, 6, device=device, requires_grad=True)
    geometry_plain = torch.randn(2, 80, 3, device=device, requires_grad=True)
    global_plain = torch.randn(2, 4, 3, device=device, requires_grad=True)
    local_checkpointed = local_plain.detach().clone().requires_grad_(True)
    geometry_checkpointed = geometry_plain.detach().clone().requires_grad_(True)
    global_checkpointed = global_plain.detach().clone().requires_grad_(True)

    output_plain = plain(
        local_plain, geometry=geometry_plain, global_embedding=global_plain
    )
    output_checkpointed = checkpointed(
        local_checkpointed,
        geometry=geometry_checkpointed,
        global_embedding=global_checkpointed,
    )
    output_plain.square().mean().backward()
    output_checkpointed.square().mean().backward()

    torch.testing.assert_close(output_checkpointed, output_plain, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        local_checkpointed.grad, local_plain.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        geometry_checkpointed.grad, geometry_plain.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        global_checkpointed.grad, global_plain.grad, atol=1e-5, rtol=1e-4
    )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-5, rtol=1e-4)


# =============================================================================
# Checkpoint Tests
# =============================================================================


def test_geotransolver_legacy_checkpoint_class_path():
    """Test resolving the model class path stored by experimental checkpoints."""
    from physicsnemo.experimental.models.geotransolver import (
        GeoTransolver as LegacyPackageGeoTransolver,
    )
    from physicsnemo.experimental.models.geotransolver.geotransolver import (
        GeoTransolver as LegacyModuleGeoTransolver,
    )

    assert LegacyPackageGeoTransolver is GeoTransolver
    assert LegacyModuleGeoTransolver is GeoTransolver


def test_geotransolver_legacy_import_paths():
    """Test the component import paths used before the move out of experimental."""
    import physicsnemo.models.geotransolver as production_pkg

    # Drop the cached legacy modules so the shim warning fires again.
    for module_name in list(sys.modules):
        if module_name.startswith("physicsnemo.experimental.models.geotransolver"):
            del sys.modules[module_name]

    with pytest.warns(
        LegacyFeatureWarning, match=re.escape("physicsnemo.models.geotransolver")
    ):
        legacy_pkg = importlib.import_module(
            "physicsnemo.experimental.models.geotransolver"
        )

    for legacy_name in legacy_pkg.__all__:
        # The move out of experimental renamed GALE_block to GALEBlock.
        production_name = "GALEBlock" if legacy_name == "GALE_block" else legacy_name
        assert getattr(legacy_pkg, legacy_name) is getattr(
            production_pkg, production_name
        ), f"'{legacy_name}' does not resolve to production '{production_name}'"


def test_geotransolver_checkpoint(device):
    """Test GeoTransolver checkpoint save/load."""
    torch.manual_seed(42)

    model_1 = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    model_2 = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_tokens, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)
    local_positions = local_emb[:, :, :3]
    assert validate_checkpoint(
        model_1,
        model_2,
        (local_emb, local_positions, global_emb, geometry),
    )


def test_geotransolver_checkpoint_tuple(device):
    """Test GeoTransolver checkpoint save/load with tuple inputs."""
    torch.manual_seed(42)

    functional_dims = (32, 48)
    out_dims = (4, 6)

    model_1 = GeoTransolver(
        functional_dim=functional_dims,
        out_dim=out_dims,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    model_2 = GeoTransolver(
        functional_dim=functional_dims,
        out_dim=out_dims,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    n_global = 5

    local_emb_1 = torch.randn(batch_size, n_tokens_1, functional_dims[0]).to(device)
    local_emb_2 = torch.randn(batch_size, n_tokens_2, functional_dims[1]).to(device)
    geometry = torch.randn(batch_size, n_tokens_1, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    assert validate_checkpoint(
        model_1,
        model_2,
        ((local_emb_1, local_emb_2), (None, None), global_emb, geometry),
    )


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_geotransolver_invalid_hidden_head_dims():
    """Test that GeoTransolver raises error for incompatible hidden/head dimensions."""
    with pytest.raises(ValueError, match="n_hidden % n_head == 0"):
        GeoTransolver(
            functional_dim=32,
            out_dim=4,
            n_hidden=65,  # Not divisible by n_head=4
            n_head=4,
            use_te=False,
        )


def test_geotransolver_mismatched_functional_out_dims():
    """Test that GeoTransolver raises error for mismatched functional/out dim lengths."""
    with pytest.raises(
        ValueError, match="functional_dim and out_dim must be the same length"
    ):
        GeoTransolver(
            functional_dim=(32, 48),
            out_dim=(4,),  # Length mismatch
            use_te=False,
        )


def test_geotransolver_structured_rejects_local_features():
    """Ball-query local features are incompatible with structured_shape."""
    with pytest.raises(ValueError, match="include_local_features=True"):
        GeoTransolver(
            functional_dim=8,
            out_dim=1,
            structured_shape=(4, 4),
            include_local_features=True,
            geometry_dim=2,
            use_te=False,
        )


def test_geotransolver_structured_2d_forward(device):
    """Structured 2D: spatial input (B,H,W,C) and flattened (B,N,C); optional geometry."""
    torch.manual_seed(0)
    H, W = 4, 4
    model = GeoTransolver(
        functional_dim=3,
        out_dim=2,
        structured_shape=(H, W),
        geometry_dim=2,
        global_dim=None,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        mlp_ratio=2,
        use_te=False,
    ).to(device)
    B = 2
    x4 = torch.randn(B, H, W, 3, device=device)
    g = torch.randn(B, H, W, 2, device=device)
    y4 = model(x4, geometry=g)
    assert y4.shape == (B, H, W, 2)
    assert not torch.isnan(y4).any()

    x3 = x4.reshape(B, H * W, 3)
    g3 = g.reshape(B, H * W, 2)
    y3 = model(x3, geometry=g3)
    assert y3.shape == (B, H * W, 2)

    y_none = model(x4)
    assert y_none.shape == (B, H, W, 2)


def test_geotransolver_structured_3d_forward(device):
    """Structured 3D voxel input (B,H,W,D,C)."""
    torch.manual_seed(1)
    H, W, Dg = 2, 2, 2
    model = GeoTransolver(
        functional_dim=4,
        out_dim=1,
        structured_shape=(H, W, Dg),
        n_layers=1,
        n_hidden=32,
        n_head=4,
        slice_num=4,
        mlp_ratio=2,
        use_te=False,
    ).to(device)
    B = 1
    x = torch.randn(B, H, W, Dg, 4, device=device)
    y = model(x)
    assert y.shape == (B, H, W, Dg, 1)


def test_geotransolver_structured_global_context(device):
    """Structured grid with global embedding context."""
    torch.manual_seed(2)
    H, W = 4, 4
    model = GeoTransolver(
        functional_dim=2,
        out_dim=1,
        structured_shape=(H, W),
        geometry_dim=2,
        global_dim=8,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        mlp_ratio=2,
        use_te=False,
    ).to(device)
    B = 2
    x = torch.randn(B, H, W, 2, device=device)
    geo = torch.randn(B, H, W, 2, device=device)
    glob = torch.randn(B, 3, 8, device=device)
    y = model(x, geometry=geo, global_embedding=glob)
    assert y.shape == (B, H, W, 1)


# =============================================================================
# Activation Function Tests
# =============================================================================


@pytest.mark.parametrize("activation", ["gelu", "relu", "tanh", "silu"])
def test_geotransolver_activations(device, activation):
    """Test GeoTransolver with different activation functions."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act=activation,
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_global = 5
    n_geom = 235

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        local_emb, local_positions=None, global_embedding=global_emb, geometry=geometry
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


# =============================================================================
# Shape and Configuration Tests
# =============================================================================


@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_geotransolver_different_depths(device, n_layers):
    """Test GeoTransolver with different numbers of layers."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=n_layers,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        local_emb, local_positions=None, global_embedding=global_emb, geometry=geometry
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


@pytest.mark.parametrize("slice_num", [4, 16, 32])
def test_geotransolver_different_slice_nums(device, slice_num):
    """Test GeoTransolver with different numbers of physical state slices."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=64,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=2,
        slice_num=slice_num,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        local_emb, local_positions=None, global_embedding=global_emb, geometry=geometry
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


@pytest.mark.parametrize("n_hidden,n_head", [(64, 4), (128, 8), (256, 8)])
def test_geotransolver_different_hidden_sizes(device, n_hidden, n_head):
    """Test GeoTransolver with different hidden dimensions and head counts."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=3,
        global_dim=16,
        n_layers=2,
        n_hidden=n_hidden,
        dropout=0.0,
        n_head=n_head,
        act="gelu",
        mlp_ratio=2,
        slice_num=8,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=False,
    ).to(device)

    batch_size = 2
    n_tokens = 100
    n_geom = 235
    n_global = 5

    local_emb = torch.randn(batch_size, n_tokens, 32).to(device)
    geometry = torch.randn(batch_size, n_geom, 3).to(device)
    global_emb = torch.randn(batch_size, n_global, 16).to(device)

    outputs = model(
        local_emb, local_positions=None, global_embedding=global_emb, geometry=geometry
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs[0]).any()


# =============================================================================
# Model Metadata Tests
# =============================================================================


def test_geotransolver_metadata():
    """Test GeoTransolver model metadata."""
    model = GeoTransolver(
        functional_dim=32,
        out_dim=4,
        use_te=False,
    )

    assert model.meta.name == "GeoTransolver"
    assert model.meta.amp is True
    assert model.__name__ == "GeoTransolver"


# =============================================================================
# Batched local-features tests (B > 1)
# =============================================================================


def test_geotransolver_local_features_batch_gt_1(device):
    """GeoTransolver with local features should work with batch_size > 1."""
    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=16,
        out_dim=4,
        geometry_dim=3,
        global_dim=8,
        n_layers=1,
        n_hidden=32,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=1,
        slice_num=4,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=True,
        radii=[0.25],
        neighbors_in_radius=[8],
        n_hidden_local=16,
    ).to(device)

    batch_size = 2
    n_tokens = 32
    n_geom = 50
    n_global = 2

    local_emb = torch.randn(batch_size, n_tokens, 16, device=device)
    local_positions = local_emb[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3, device=device)
    global_emb = torch.randn(batch_size, n_global, 8, device=device)

    outputs = model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, n_tokens, 4)
    assert not torch.isnan(outputs).any()


def test_geotransolver_local_features_compile(device):
    """GeoTransolver with local features should be compilable (max_points path)."""
    if "cuda" in device:
        pytest.skip("Skipping GeoTransolver torch.compile on CUDA")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    torch.manual_seed(42)

    model = GeoTransolver(
        functional_dim=16,
        out_dim=4,
        geometry_dim=3,
        global_dim=8,
        n_layers=1,
        n_hidden=32,
        dropout=0.0,
        n_head=4,
        act="gelu",
        mlp_ratio=1,
        slice_num=4,
        use_te=False,
        time_input=False,
        plus=False,
        include_local_features=True,
        radii=[0.25],
        neighbors_in_radius=[8],
        n_hidden_local=16,
    ).to(device)

    batch_size = 2
    n_tokens = 32
    n_geom = 50
    n_global = 2

    local_emb = torch.randn(batch_size, n_tokens, 16, device=device)
    local_positions = local_emb[:, :, :3]
    geometry = torch.randn(batch_size, n_geom, 3, device=device)
    global_emb = torch.randn(batch_size, n_global, 8, device=device)

    eager_out = model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    compiled_model = torch.compile(model)
    compiled_out = compiled_model(
        local_emb,
        local_positions=local_positions,
        global_embedding=global_emb,
        geometry=geometry,
    )

    assert compiled_out.shape == eager_out.shape
    assert not torch.isnan(compiled_out).any()


# =============================================================================
# Activation Checkpointing Tests
# =============================================================================


@pytest.mark.parametrize(
    "components,error",
    [
        ([], ValueError),
        ("blocks", TypeError),
        ([["blocks"]], TypeError),
        (["unknown"], ValueError),
    ],
)
def test_geotransolver_activation_checkpointing_rejects_invalid_components(
    components, error
):
    with pytest.raises(error, match="activation_checkpointing_components"):
        GeoTransolver(
            functional_dim=3,
            out_dim=2,
            n_hidden=16,
            n_head=4,
            slice_num=4,
            use_te=False,
            activation_checkpointing=True,
            activation_checkpointing_components=components,
        )


@pytest.mark.parametrize(
    "functional_dim,out_dim,attention_type,components,concrete_dropout",
    [
        (3, 2, "GALE", ("blocks",), False),
        (
            (3, 5),
            (2, 4),
            "GALE_FA",
            ("context", "preprocess", "blocks", "output"),
            True,
        ),
    ],
    ids=["blocks_gale", "full_gale_fa"],
)
def test_geotransolver_activation_checkpointing_matches_outputs_and_gradients(
    device,
    functional_dim,
    out_dim,
    attention_type,
    components,
    concrete_dropout,
):
    kwargs = dict(
        functional_dim=functional_dim,
        out_dim=out_dim,
        geometry_dim=3,
        global_dim=2,
        n_layers=3,
        n_hidden=16,
        dropout=0.1,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        use_te=False,
        plus=True,
        attention_type=attention_type,
        concrete_dropout=concrete_dropout,
    )
    torch.manual_seed(1)
    plain = GeoTransolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = GeoTransolver(
        **kwargs,
        activation_checkpointing=True,
        activation_checkpointing_components=components,
    ).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()

    dims = (functional_dim,) if isinstance(functional_dim, int) else functional_dim
    plain_inputs = tuple(
        torch.randn(2, 24 + i * 4, dim, device=device, requires_grad=True)
        for i, dim in enumerate(dims)
    )
    checkpointed_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in plain_inputs
    )
    geometry_plain = torch.randn(2, 28, 3, device=device, requires_grad=True)
    global_plain = torch.randn(2, 3, 2, device=device, requires_grad=True)
    geometry_checkpointed = geometry_plain.detach().clone().requires_grad_(True)
    global_checkpointed = global_plain.detach().clone().requires_grad_(True)

    plain_arg = plain_inputs[0] if len(plain_inputs) == 1 else plain_inputs
    checkpointed_arg = (
        checkpointed_inputs[0] if len(checkpointed_inputs) == 1 else checkpointed_inputs
    )
    torch.manual_seed(7)
    output_plain = plain(
        plain_arg,
        geometry=geometry_plain,
        global_embedding=global_plain,
    )
    torch.manual_seed(7)
    output_checkpointed = checkpointed(
        checkpointed_arg,
        geometry=geometry_checkpointed,
        global_embedding=global_checkpointed,
    )
    for actual, expected in zip(
        _flatten_outputs(output_checkpointed), _flatten_outputs(output_plain)
    ):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    _output_loss(output_plain).backward()
    _output_loss(output_checkpointed).backward()
    for actual, expected in zip(checkpointed_inputs, plain_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        geometry_checkpointed.grad, geometry_plain.grad, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        global_checkpointed.grad, global_plain.grad, atol=1e-6, rtol=1e-5
    )
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


def test_geotransolver_activation_checkpointing_recomputes_selected_blocks(
    monkeypatch,
):
    model = GeoTransolver(
        functional_dim=3,
        out_dim=2,
        geometry_dim=3,
        n_layers=4,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        use_te=False,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
    )
    model.train()
    call_counts = [0] * len(model.blocks)
    for block_idx, block in enumerate(model.blocks):
        original_forward = block.forward

        def counting_forward(fx, context, idx=block_idx, forward=original_forward):
            call_counts[idx] += 1
            return forward(fx, context)

        monkeypatch.setattr(block, "forward", counting_forward)

    local_embedding = torch.randn(2, 16, 3)
    geometry = torch.randn(2, 16, 3)
    model(local_embedding, geometry=geometry).square().mean().backward()
    assert call_counts == [2, 1, 2, 1]


def test_geotransolver_zero_block_ratio_checkpoints_selected_components(monkeypatch):
    """A zero block ratio does not disable other checkpoint boundaries."""
    model = GeoTransolver(
        functional_dim=3,
        out_dim=2,
        geometry_dim=3,
        global_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        use_te=False,
        activation_checkpointing=True,
        checkpointing_ratio=0.0,
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    )
    model.train()
    calls = []

    def fake_te_checkpoint(function, *args, **kwargs):
        calls.append((len(args), kwargs))
        return function(*args)

    monkeypatch.setattr(
        geotransolver_module,
        "te",
        SimpleNamespace(checkpoint=fake_te_checkpoint),
        raising=False,
    )
    model.use_te = True
    model(
        torch.randn(2, 16, 3),
        geometry=torch.randn(2, 20, 3),
        global_embedding=torch.randn(2, 2, 2),
    )

    assert calls == [
        (4, {"use_reentrant": False}),
        (1, {"use_reentrant": False}),
        (1, {"use_reentrant": False}),
    ]


def test_geotransolver_activation_checkpointing_torch_compile(device):
    components = ("context", "preprocess", "blocks", "output")
    kwargs = dict(
        functional_dim=3,
        out_dim=2,
        geometry_dim=3,
        global_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        mlp_ratio=2,
        slice_num=4,
        use_te=False,
        attention_type="GALE_FA",
    )
    plain = GeoTransolver(**kwargs, activation_checkpointing=False).to(device)
    checkpointed = GeoTransolver(
        **kwargs,
        activation_checkpointing=True,
        activation_checkpointing_components=components,
    ).to(device)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled_plain = torch.compile(plain, backend=backend, fullgraph=True)
    compiled_checkpointed = torch.compile(checkpointed, backend=backend, fullgraph=True)
    local_plain = torch.randn(2, 16, 3, device=device, requires_grad=True)
    geometry_plain = torch.randn(2, 16, 3, device=device, requires_grad=True)
    global_plain = torch.randn(2, 2, 2, device=device, requires_grad=True)
    local_checkpointed = local_plain.detach().clone().requires_grad_(True)
    geometry_checkpointed = geometry_plain.detach().clone().requires_grad_(True)
    global_checkpointed = global_plain.detach().clone().requires_grad_(True)

    output_plain = compiled_plain(
        local_plain, geometry=geometry_plain, global_embedding=global_plain
    )
    output_checkpointed = compiled_checkpointed(
        local_checkpointed,
        geometry=geometry_checkpointed,
        global_embedding=global_checkpointed,
    )
    torch.testing.assert_close(output_checkpointed, output_plain)
    output_plain.square().mean().backward()
    output_checkpointed.square().mean().backward()
    torch.testing.assert_close(local_checkpointed.grad, local_plain.grad)
    torch.testing.assert_close(geometry_checkpointed.grad, geometry_plain.grad)
    torch.testing.assert_close(global_checkpointed.grad, global_plain.grad)
    _assert_parameter_gradients_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


def test_geotransolver_activation_checkpointing_serialization(tmp_path):
    """Checkpoint settings and Hydra component lists round-trip safely."""
    components = OmegaConf.create(["context", "output"])
    kwargs = dict(
        functional_dim=3,
        out_dim=2,
        geometry_dim=3,
        n_layers=2,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        use_te=False,
    )
    default_model = GeoTransolver(**kwargs)
    checkpointed_model = GeoTransolver(
        **kwargs,
        activation_checkpointing=True,
        checkpointing_ratio=0.0,
        activation_checkpointing_components=components,
    )
    checkpointed_model.load_state_dict(default_model.state_dict())
    assert checkpointed_model.state_dict().keys() == default_model.state_dict().keys()

    checkpoint_path = tmp_path / "geotransolver_checkpointed.mdlus"
    checkpointed_model.save(checkpoint_path)
    restored = Module.from_checkpoint(checkpoint_path)
    assert isinstance(restored, GeoTransolver)
    assert restored._activation_checkpointing_enabled is True
    assert restored._activation_checkpointing_ratio == 0.0
    assert restored._activation_checkpointing_components == frozenset(components)

    legacy_args = {
        **default_model._args,
        "__args__": default_model._args["__args__"].copy(),
    }
    legacy_args["__args__"].pop("activation_checkpointing")
    legacy_args["__args__"].pop("checkpointing_ratio")
    legacy_args["__args__"].pop("activation_checkpointing_components")
    legacy_restored = Module.instantiate(legacy_args)
    assert isinstance(legacy_restored, GeoTransolver)
    assert legacy_restored._activation_checkpointing_enabled is False
    assert legacy_restored._activation_checkpointing_ratio == 0.0
    assert legacy_restored._activation_checkpointing_components == frozenset({"blocks"})

    local_embedding = torch.randn(2, 16, 3, requires_grad=True)
    geometry = torch.randn(2, 16, 3, requires_grad=True)
    expected = default_model(local_embedding, geometry=geometry).detach()

    del default_model._activation_checkpointing_enabled
    del default_model._activation_checkpointing_ratio
    del default_model._activation_checkpointing_components
    restored = pickle.loads(  # noqa: S301 - trusted local fixture
        pickle.dumps(default_model)
    )
    restored_local = local_embedding.detach().clone().requires_grad_(True)
    restored_geometry = geometry.detach().clone().requires_grad_(True)
    actual = restored(restored_local, geometry=restored_geometry)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert restored_local.grad is not None
    assert restored_geometry.grad is not None
    assert all(parameter.grad is not None for parameter in restored.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_geotransolver_activation_checkpointing_reduces_peak_cuda_memory():
    def peak_step_bytes(
        activation_checkpointing,
        activation_checkpointing_components=("blocks",),
    ):
        torch.cuda.empty_cache()
        model = GeoTransolver(
            functional_dim=6,
            out_dim=4,
            geometry_dim=3,
            global_dim=3,
            n_layers=6,
            n_hidden=128,
            n_head=8,
            mlp_ratio=4,
            slice_num=32,
            use_te=False,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_components=activation_checkpointing_components,
        ).to("cuda")
        model.train()
        local_embedding = torch.randn(1, 4096, 6, device="cuda")
        geometry = torch.randn(1, 4096, 3, device="cuda")
        global_embedding = torch.randn(1, 4, 3, device="cuda")

        # Warm up each policy before resetting the allocator peak so one-time
        # kernel initialization is excluded symmetrically.
        model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        ).square().mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        ).square().mean().backward()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline
        del model, local_embedding, geometry, global_embedding
        gc.collect()
        torch.cuda.empty_cache()
        return peak_delta

    plain_peak = peak_step_bytes(False)
    full_peak = peak_step_bytes(True, ("context", "preprocess", "blocks", "output"))
    assert full_peak < plain_peak


# =============================================================================
# Checkpoint Tests
# =============================================================================
