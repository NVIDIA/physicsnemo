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
import torch

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
