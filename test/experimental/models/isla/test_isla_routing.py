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

"""Bounded (cosine) routing logits, and the softmax over points that normalizes them."""

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._blocks import (
    MLP_RATIO,
    ROUTING_TEMPERATURE,
    _RoutingLogits,
    _SliceBlock,
    _softmax_over_points,
)
from test.experimental.models.isla._meshes import outputs, soup


def test_routing_logits_are_bounded_and_bias_free():
    """The routing has no bias parameter, and its cosine logits stay within the
    temperature however large the seeds, the maps, the gain and the state grow."""
    torch.manual_seed(0)
    r = _RoutingLogits(64, 32)
    assert r.weight.shape == (32, 64)
    # the seeds, the key and value maps of the prototype synthesis and its gain; no bias anywhere
    assert set(dict(r.named_parameters())) == {
        "weight",
        "k_proj.weight",
        "v_proj.weight",
        "gain",
    }
    with torch.no_grad():
        r.weight.mul_(1000.0)  # however large the seeds grow ...
        r.k_proj.weight.mul_(1000.0)
        r.v_proj.weight.mul_(1000.0)
        r.gain.fill_(1000.0)  # ... or the synthesis maps ...
        x = torch.randn(2, 50, 64) * 100.0  # ... or the state
        out = r(x, torch.zeros(2, 50, 1))
    assert ROUTING_TEMPERATURE == 10.0
    assert out.abs().max() <= ROUTING_TEMPERATURE + 1e-5


def test_softmax_over_points_matches_middle_dim_softmax():
    """The point softmax (reduction along a contiguous last dimension after a
    transpose) is the same arithmetic as a softmax over the middle dimension: exact in
    float64, at roundoff in float32."""
    x = torch.randn(2, 500, 64, dtype=torch.float64)
    assert torch.allclose(_softmax_over_points(x), torch.softmax(x, dim=1), atol=1e-14)
    assert torch.allclose(
        _softmax_over_points(x).sum(dim=1),
        torch.ones(2, 64, dtype=torch.float64),
        atol=1e-14,
    )
    x32 = x.float()
    rel = (
        (_softmax_over_points(x32) - torch.softmax(x32, dim=1)).norm()
        / torch.softmax(x32, dim=1).norm()
    ).item()
    assert rel < 1e-6


class _Record(TorchFunctionMode):
    """Records every softmax input and the einsum output that sums ``n_hat``."""

    def __init__(self, n_hat):
        super().__init__()
        self.n_hat, self.softmax_in, self.m_sum = n_hat, [], None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is torch.softmax:
            self.softmax_in.append(args[0])
        if func is torch.einsum and args[-1] is self.n_hat:
            self.m_sum = out
        return out


def test_routing_path_is_float32_under_bf16_autocast():
    """Under bfloat16 autocast the attention logits before their softmax, the routing
    logits and the slice mean normal's weighted sum are float32, as on the compiled
    path; in float32 the logits are the plain formula, bitwise."""
    torch.manual_seed(0)
    block, n = _SliceBlock(32, 8, 1), 100
    h, r = torch.randn(1, n, 32), torch.randn(1, n, 3)
    log_w = -3.0 * torch.rand(1, n, 1)
    n_hat = F.normalize(torch.randn(1, n, 3), dim=-1)
    g_hat = F.normalize(torch.randn(1, 1, 1, 3), dim=-1).expand(1, n, 1, 3)
    for autocast in (False, True):
        rec = _Record(n_hat)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), rec:
            x = block.norm_assign(h)
            protos = block.assign.prototypes(x, log_w)
            logits = block.assign.logits(x, protos)
            block(h, log_w, r, n_hat, g_hat, 1e-12)
        att = rec.softmax_in[0]  # the first softmax is the prototypes' attention
        assert logits.dtype == att.dtype == rec.m_sum.dtype == torch.float32
        if autocast:  # scaled and biased in float32, not rounded to bfloat16 first
            assert not torch.equal(att, att.bfloat16().float())
        else:
            cos = torch.einsum(
                "bnh,bsh->bns", F.normalize(x, dim=-1), F.normalize(protos, dim=-1)
            )
            assert torch.equal(logits, ROUTING_TEMPERATURE * cos)


def test_routing_temperature_and_mlp_ratio_reach_every_block():
    """ISLA's ``routing_temperature`` bounds every block's routing logits and the head
    frame's, and its ``mlp_ratio`` sets the width of every block's two MLPs; their
    defaults are the blocks' constants. With the same parameters a model of
    another temperature routes, and so predicts, differently."""
    io = dict(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=16,
        n_layers=2,
        n_slices=4,
    )
    assert (ROUTING_TEMPERATURE, MLP_RATIO) == (10.0, 4)
    torch.manual_seed(0)
    default = ISLA(**io).eval()
    m = ISLA(**io, routing_temperature=2.5, mlp_ratio=2).eval()
    routings = [b.assign for b in m.blocks] + [m.frame_assign]
    assert [r.temperature for r in routings] == [2.5] * 3
    widths = [(b.slice_mlp[1].out_features, b.mlp[0].out_features) for b in m.blocks]
    assert widths == [(32, 32)] * 2
    assert [b.assign.temperature for b in default.blocks] == [10.0] * 2
    assert default.blocks[0].mlp[0].out_features == 4 * 16
    x = torch.randn(1, 50, 16) * 100.0
    with torch.no_grad():
        assert (
            float(m.blocks[0].assign(x, torch.zeros(1, 50, 1)).abs().max())
            <= 2.5 + 1e-6
        )
    cooled = ISLA(**io, routing_temperature=2.5).eval()
    cooled.load_state_dict(default.state_dict())
    gen = torch.Generator().manual_seed(3)
    pts = torch.randn(40, 3, generator=gen)
    nrm = F.normalize(torch.randn(40, 3, generator=gen), dim=-1)
    kw = dict(
        boundary_meshes={"wall": soup(pts, nrm, torch.rand(40, generator=gen) + 0.5)},
        prediction_boundaries=["wall"],
        global_data={"g": F.normalize(torch.randn(3, generator=gen), dim=0)},
    )
    with torch.no_grad():
        a, b = outputs(default(**kw)), outputs(cooled(**kw))
    assert not torch.allclose(a["p"], b["p"])
