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

"""The token axis's padding to a multiple of ``_TOKEN_ALIGNMENT``: padding tokens of
no measure that nothing reads leave every output and gradient unchanged."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import physicsnemo.experimental.models.isla.model as isla_model
from physicsnemo.experimental.models.isla import ISLA
from test.experimental.models.isla._meshes import cloud, live, outputs, same, soup

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
D = torch.float64


@pytest.mark.parametrize("task", ["surface", "interior_dtt"])
def test_token_padding_leaves_outputs_and_gradients_unchanged(task, monkeypatch):
    """At token counts that are not multiples of _TOKEN_ALIGNMENT, the outputs and every parameter
    gradient equal those of the unpadded network (alignment 1) to float64 roundoff,
    and the outputs keep the given count."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(203)
    mesh = soup(pts, nrm, w)
    q = None if task == "surface" else torch.randn(30, 3, dtype=D) * 2.0
    kw = {} if task == "surface" else {"interior": True}
    m = ISLA(
        output_schema=PV,
        boundary_schemas={"wall": {}},
        global_schema=G,
        hidden=32,
        n_layers=2,
        n_slices=8,
        **kw,
    ).double()
    live(m)  # the interior task's boundary read contributes (a surface model has none)
    assert (203 + (0 if q is None else 30)) % isla_model._TOKEN_ALIGNMENT

    def run(alignment):
        monkeypatch.setattr(isla_model, "_TOKEN_ALIGNMENT", alignment)
        m.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                prediction_boundaries=None if q is not None else ["wall"],
                global_data={"g": drv},
            )
        )
        sum(
            (v * torch.linspace(-1, 1, v.numel(), dtype=D).view_as(v)).sum()
            for v in out.values()
        ).backward()
        return out, {
            k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None
        }

    out1, grad1 = run(1)
    out8, grad8 = run(8)
    assert out8["p"].shape == (203 if q is None else 30,)
    assert same(out8, out1, atol=1e-12)
    assert grad8.keys() == grad1.keys()
    for k in grad1:
        assert torch.allclose(grad8[k], grad1[k], atol=1e-11, rtol=1e-10), k


class _AtenOps(TorchDispatchMode):
    """The names of the aten operators that run under it."""

    def __init__(self):
        super().__init__()
        self.names = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("task", ["surface", "interior"])
def test_an_aligned_token_count_runs_no_padding(task):
    """In eager mode, at a multiple of _TOKEN_ALIGNMENT tokens, the forward and backward
    run no padding op (an unaligned count does), so they are those of the network
    without padding."""
    nq = 0 if task == "surface" else 32
    m = ISLA(
        output_schema=PV,
        boundary_schemas={"wall": {}},
        global_schema=G,
        hidden=32,
        n_layers=2,
        n_slices=8,
        interior=task == "interior",
    ).double()

    def ops(n_boundary):
        pts, nrm, w, drv = cloud(n_boundary)
        q = torch.randn(nq, 3, dtype=D) if nq else None
        with _AtenOps() as mode:
            out = outputs(
                m(
                    boundary_meshes={"wall": soup(pts, nrm, w)},
                    prediction_points=q,
                    prediction_boundaries=None if nq else ["wall"],
                    global_data={"g": drv},
                )
            )
            sum(v.sum() for v in out.values()).backward()
        return mode.names

    aligned = 4 * isla_model._TOKEN_ALIGNMENT
    if task == "surface":
        assert "constant_pad_nd" in ops(aligned - nq + 1)
        assert "constant_pad_nd" not in ops(aligned - nq)
        return
    ### The interior task's forest build runs constant_pad_nd ops of its own, so the interior
    ### task is checked by the token count its blocks see, the boundary tokens alone
    ### (the points read the blocks' slices): no padding token at an aligned count, a
    ### padded axis at an unaligned one.
    seen = []
    hook = m.blocks[0].register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(args[0].shape[1]), with_kwargs=True
    )
    try:
        ops(aligned + 1)
        ops(aligned)
    finally:
        hook.remove()
    assert seen == [aligned + isla_model._TOKEN_ALIGNMENT, aligned]
