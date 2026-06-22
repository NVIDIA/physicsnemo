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

import torch

# Import datapipes first: it resolves a pre-existing nn.functional <-> datapipes
# circular import that otherwise fails when context_projector is the first
# physicsnemo import in the process (e.g. running this file standalone).
import physicsnemo.datapipes  # noqa: F401
from physicsnemo.experimental.models.geotransolver.context_projector import (
    ContextProjector,
    MultiScaleFeatureExtractor,
)

# =============================================================================
# ContextProjector Tests
# =============================================================================


def test_context_projector_forward(device):
    """Test ContextProjector forward pass."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    projector = ContextProjector(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)

    slice_tokens = projector(x)

    # Output shape: [Batch, Heads, Slice_num, dim_head]
    assert slice_tokens.shape == (batch_size, heads, slice_num, dim_head)
    assert not torch.isnan(slice_tokens).any()


# =============================================================================
# MultiScaleFeatureExtractor consolidation tests
# =============================================================================


def _make_extractor(device):
    """Build a small MultiScaleFeatureExtractor for the consolidation tests."""
    extractor = MultiScaleFeatureExtractor(
        geometry_dim=3,
        radii=[0.5, 1.0],
        neighbors_in_radius=[4, 8],
        hidden_dim=16,
        n_head=4,
        dim_head=8,
        dropout=0.0,
        slice_num=8,
        use_te=False,
        plus=False,
    ).to(device)
    extractor.eval()
    return extractor


def test_same_coords_guard(device):
    """``_same_coords`` detects aliasing views but not equal-valued copies."""
    base = torch.randn(64, 3, device=device)

    # Identical object.
    assert MultiScaleFeatureExtractor._same_coords(base, base)

    # Distinct view objects over the same storage (mirrors the recipe collate,
    # which unsqueezes geometry and local_positions separately).
    a = base.unsqueeze(0)
    b = base.unsqueeze(0)
    assert a is not b
    assert MultiScaleFeatureExtractor._same_coords(a, b)

    # Equal values but distinct storage -> must NOT be treated as aliased.
    assert not MultiScaleFeatureExtractor._same_coords(base, base.clone())


def test_extract_context_and_local_aliased_matches_two_pass(device):
    """Fast path (aliased inputs) equals the separate context/local methods."""
    torch.manual_seed(0)
    extractor = _make_extractor(device)
    x = torch.randn(2, 64, 3, device=device)

    with torch.no_grad():
        context, local = extractor.extract_context_and_local(x, x)
        context_ref = extractor.extract_context_features(x, x)
        local_ref = extractor.extract_local_features(x, x)

    assert len(context) == len(context_ref) == extractor.num_scales
    for got, ref in zip(context, context_ref):
        assert torch.equal(got, ref)
    assert torch.equal(local, local_ref)


def test_extract_context_and_local_distinct_matches_two_pass(device):
    """Fallback path (distinct inputs) preserves the asymmetric semantics."""
    torch.manual_seed(1)
    extractor = _make_extractor(device)
    spatial = torch.randn(2, 64, 3, device=device)
    geometry = torch.randn(2, 64, 3, device=device)

    # Sanity: the guard must reject these so the fallback path is taken.
    assert not extractor._same_coords(spatial, geometry)

    with torch.no_grad():
        context, local = extractor.extract_context_and_local(spatial, geometry)
        context_ref = extractor.extract_context_features(spatial, geometry)
        local_ref = extractor.extract_local_features(spatial, geometry)

    for got, ref in zip(context, context_ref):
        assert torch.equal(got, ref)
    assert torch.equal(local, local_ref)


def test_extract_context_and_local_reuses_radius_search(device, monkeypatch):
    """Aliased inputs issue one radius_search per scale; distinct inputs issue two."""
    # Imported lazily: importing ball_query before the model package is fully
    # initialized trips a pre-existing nn.functional <-> datapipes import cycle.
    import physicsnemo.nn.module.ball_query as ball_query_mod

    extractor = _make_extractor(device)

    call_count = {"n": 0}
    original = ball_query_mod.radius_search

    def counting_radius_search(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ball_query_mod, "radius_search", counting_radius_search)

    x = torch.randn(2, 64, 3, device=device)
    with torch.no_grad():
        extractor.extract_context_and_local(x, x)
    # One ball query per scale (consolidated), not two.
    assert call_count["n"] == extractor.num_scales

    call_count["n"] = 0
    spatial = torch.randn(2, 64, 3, device=device)
    geometry = torch.randn(2, 64, 3, device=device)
    with torch.no_grad():
        extractor.extract_context_and_local(spatial, geometry)
    # Fallback runs context + local separately: two ball queries per scale.
    assert call_count["n"] == 2 * extractor.num_scales
