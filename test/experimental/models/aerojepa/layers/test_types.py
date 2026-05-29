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

"""Tests for AeroJEPA core token dataclasses."""

import torch

from physicsnemo.experimental.models.aerojepa.layers import EncoderOutput, TokenSet


def test_tokenset_unbatched_construction(device):
    """Unbatched TokenSet exposes the right attributes and properties."""
    ts = TokenSet(
        features=torch.randn(10, 16, device=device),
        coords=torch.randn(10, 3, device=device),
    )
    assert ts.features.shape == (10, 16)
    assert ts.coords.shape == (10, 3)
    assert ts.mask is None
    assert ts.global_token is None
    assert ts.aux == {}
    assert ts.is_batched is False
    assert ts.token_dim == 16


def test_tokenset_batched_construction(device):
    """Batched TokenSet flips ``is_batched`` and keeps the last-axis token_dim."""
    ts = TokenSet(
        features=torch.zeros(2, 10, 32, device=device),
        coords=torch.zeros(2, 10, 3, device=device),
        mask=torch.ones(2, 10, device=device, dtype=torch.bool),
        global_token=torch.zeros(2, 32, device=device),
        aux={"layer": 0},
    )
    assert ts.is_batched is True
    assert ts.token_dim == 32
    assert ts.mask.shape == (2, 10)
    assert ts.global_token.shape == (2, 32)
    assert ts.aux == {"layer": 0}


def test_tokenset_aux_default_factory_is_independent():
    """Two TokenSets created without aux must not share the dict instance."""
    ts1 = TokenSet(features=torch.zeros(2, 4), coords=torch.zeros(2, 3))
    ts2 = TokenSet(features=torch.zeros(2, 4), coords=torch.zeros(2, 3))
    ts1.aux["k"] = "v"
    assert ts2.aux == {}


def test_tokenset_with_updates_returns_new_instance(device):
    """`with_updates` returns a new TokenSet without mutating the original."""
    ts = TokenSet(
        features=torch.zeros(1, 4, 8, device=device),
        coords=torch.zeros(1, 4, 3, device=device),
    )
    mask = torch.ones(1, 4, device=device, dtype=torch.bool)
    ts2 = ts.with_updates(mask=mask)
    assert ts2 is not ts
    assert ts2.mask is mask
    assert ts.mask is None
    assert ts2.features is ts.features
    assert ts2.coords is ts.coords


def test_tokenset_with_updates_replaces_only_given_fields(device):
    """Fields left as None are kept; supplied fields override."""
    ts = TokenSet(
        features=torch.zeros(4, 8, device=device),
        coords=torch.zeros(4, 3, device=device),
        global_token=torch.ones(8, device=device),
        aux={"a": 1},
    )
    new_feat = torch.ones(4, 8, device=device)
    ts2 = ts.with_updates(features=new_feat)
    assert torch.equal(ts2.features, new_feat)
    assert ts2.coords is ts.coords
    assert ts2.global_token is ts.global_token
    assert ts2.aux == {"a": 1}
    # Default aux behavior shallow-copies so future mutation of ts.aux does not
    # affect ts2.aux.
    ts.aux["b"] = 2
    assert ts2.aux == {"a": 1}


def test_tokenset_with_updates_aux_explicit_override(device):
    """Passing an explicit aux replaces (not merges)."""
    ts = TokenSet(
        features=torch.zeros(2, 4, device=device),
        coords=torch.zeros(2, 3, device=device),
        aux={"a": 1},
    )
    ts2 = ts.with_updates(aux={"b": 2})
    assert ts2.aux == {"b": 2}
    assert ts.aux == {"a": 1}


def test_encoder_output_construction(device):
    """EncoderOutput wraps a TokenSet and surfaces optional global token / aux."""
    ts = TokenSet(
        features=torch.zeros(2, 10, 64, device=device),
        coords=torch.zeros(2, 10, 3, device=device),
    )
    out = EncoderOutput(tokens=ts)
    assert out.tokens is ts
    assert out.global_token is None
    assert out.aux == {}

    out2 = EncoderOutput(
        tokens=ts,
        global_token=torch.zeros(2, 64, device=device),
        aux={"attn": "weights"},
    )
    assert out2.global_token.shape == (2, 64)
    assert out2.aux == {"attn": "weights"}


def test_encoder_output_aux_default_factory_is_independent():
    """EncoderOutput's default aux must be per-instance, like TokenSet's."""
    ts = TokenSet(features=torch.zeros(1, 4), coords=torch.zeros(1, 3))
    out1 = EncoderOutput(tokens=ts)
    out2 = EncoderOutput(tokens=ts)
    out1.aux["k"] = "v"
    assert out2.aux == {}
