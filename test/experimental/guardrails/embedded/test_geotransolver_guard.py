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

from physicsnemo.experimental.guardrails.embedded import (
    GuardedGeoTransolver,
    OODGuardConfig,
    attach_ood_guard,
)
from physicsnemo.models.geotransolver.geotransolver import GeoTransolver

# =============================================================================
# GuardedGeoTransolver wrapper: OOD guard decoupled from the model
# =============================================================================


def _make_model(device, *, geometry_dim=3, global_dim=16):
    """Minimal GeoTransolver used by the guard-wrapper tests."""
    return GeoTransolver(
        functional_dim=32,
        out_dim=4,
        geometry_dim=geometry_dim,
        global_dim=global_dim,
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


def _inputs(device, batch_size=2):
    local_emb = torch.randn(batch_size, 50, 32, device=device)
    return {
        "local_embedding": local_emb,
        "local_positions": local_emb[:, :, :3].contiguous(),
        "global_embedding": torch.randn(batch_size, 1, 16, device=device),
        "geometry": torch.randn(batch_size, 80, 3, device=device),
    }


def test_model_has_no_embedded_guard():
    """The upstreamed model must not expose any embedded guard surface."""
    model = _make_model("cpu")
    assert not hasattr(model, "ood_guard")


def test_wrapper_infers_dims_and_attaches_guard():
    """Dims are inferred from the model's context_builder submodules."""
    model = _make_model("cpu")
    guarded = GuardedGeoTransolver(model, OODGuardConfig(buffer_size=8))
    # geometry_embed_dim == dim_head == n_hidden // n_head == 64 // 4 == 16.
    assert guarded.ood_guard.geo_embeddings.shape[1] == 16
    # global_dim == 16 (the configured global embedding channel dim).
    assert guarded.ood_guard.global_min.shape[0] == 16


def test_wrapper_collect_then_check_runs(device):
    """Train forward collects into the guard; eval forward checks without crashing."""
    torch.manual_seed(42)
    model = _make_model(device)
    guarded = GuardedGeoTransolver(
        model, OODGuardConfig(buffer_size=8, knn_k=3, sensitivity=1.5)
    ).to(device)
    assert guarded.ood_guard.global_min.device == torch.device(device)
    assert guarded.ood_guard.geo_embeddings.device == torch.device(device)

    batch_size = 2
    inputs = _inputs(device, batch_size)

    guarded.train()
    out = guarded(**inputs)
    assert out.shape == (batch_size, 50, 4)
    # The geometry latent was captured via the forward hook and collected.
    assert guarded.ood_guard.geo_ptr.item() == batch_size
    assert not torch.isinf(guarded.ood_guard.global_min).any()

    guarded.eval()
    _ = guarded(**inputs)  # runs checks; must not raise


def test_wrapper_forwards_output_unchanged(device):
    """Wrapping must not alter the model's output for a given input."""
    torch.manual_seed(0)
    model = _make_model(device)
    model.eval()
    inputs = _inputs(device)
    with torch.no_grad():
        ref = model(**inputs)

    guarded = GuardedGeoTransolver(model, OODGuardConfig(buffer_size=8)).to(device)
    guarded.eval()
    with torch.no_grad():
        wrapped = guarded(**inputs)

    torch.testing.assert_close(ref, wrapped)


def test_attach_ood_guard_alias():
    """The functional alias returns a configured wrapper."""
    model = _make_model("cpu")
    guarded = attach_ood_guard(model, OODGuardConfig(buffer_size=8))
    assert isinstance(guarded, GuardedGeoTransolver)


def test_wrapper_without_any_surface_raises():
    """Wrapping a model with neither surface enabled raises a clear error."""
    model = _make_model("cpu", geometry_dim=None, global_dim=None)
    with pytest.raises(ValueError, match="nothing to watch"):
        GuardedGeoTransolver(model, OODGuardConfig(buffer_size=8))
