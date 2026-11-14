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

"""Cases shared by the ISLA kernel tests: the kernel basis's geometry, plan and MLP,
its chunked basis and gradients, a move of a case to a device and dtype, the 2x-error
rule of a fast kernel against its eager reference, a fused-read case, and a CPU
stand-in read for the fused read's basis-gradient sink."""

import types

import torch

from physicsnemo.experimental.models.isla import _fused_read as fused_read
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._tree_read import (
    _KernelBasis,
    kernel_geo_width,
)

D = torch.float64


def _basis_case(dim, n_vecs, n_pairs=20000, seed=0):
    """One case's geometry for the kernel basis (float64, CPU; two global directions,
    ``n_vecs`` rank-1 inputs per source), a random plan of ``n_pairs`` pairs and a
    basis MLP."""
    g = torch.Generator().manual_seed(seed)
    nq, ns = 300, 500

    def unit(*s):
        return torch.nn.functional.normalize(
            torch.randn(*s, generator=g, dtype=D), dim=-1
        )

    geometry = dict(
        query_r=torch.randn(nq, dim, generator=g, dtype=D),
        query_n=unit(nq, dim),
        query_wall=torch.rand(nq, generator=g, dtype=D) + 0.01,
        src_r=torch.randn(ns, dim, generator=g, dtype=D),
        src_n=unit(ns, dim),
        src_vecs=torch.randn(ns, n_vecs, dim, generator=g, dtype=D) if n_vecs else None,
        g_unit=unit(2, dim),
        eps=1e-12,
    )
    plan = types.SimpleNamespace(
        qi=torch.randint(0, nq, (n_pairs,), generator=g).int(),
        bj=torch.randint(0, ns, (n_pairs,), generator=g).int(),
    )
    return geometry, plan, _KernelBasis(kernel_geo_width(2, n_vecs)).double()


def _to(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device, dtype) if x.is_floating_point() else x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device, dtype) for k, v in x.items()}
    if isinstance(x, types.SimpleNamespace):
        return types.SimpleNamespace(**_to(vars(x), device, dtype))
    return x


def _basis_step(basis, plan, geometry, seed, autocast=False, weights=False):
    """The chunked basis of every pair from the plan, and its MLP's gradients for the
    basis gradient ``seed``."""
    basis.zero_grad(set_to_none=True)
    dev = geometry["query_r"].device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=autocast):
        w = tree_read.fused_basis.mlp_weights(basis) if weights else None
        out = tree_read._chunked_basis_from_plan(
            basis, plan, 0, int(plan.qi.shape[0]), 0, 0, geometry, w
        )
    out.backward(seed.to(out.dtype))
    return [out.detach()] + [p.grad.clone() for p in basis.parameters()]


def _two_x(name, fast, eager, exact, floor):
    """The 2x-error rule for one tensor (relative to the float64 tensor's largest entry)."""
    fast, eager, exact = (t.detach().double().cpu() for t in (fast, eager, exact))
    scale = exact.abs().max()
    d_fast, d_eager = (
        (fast - exact).abs().max() / scale,
        (eager - exact).abs().max() / scale,
    )
    assert d_fast <= 2 * d_eager + floor, (name, float(d_fast), float(d_eager))


def _read_case(seed):
    """A fused-read case (float64, CPU): a ragged plan of 37 points and 50 sources
    with a point without pairs and one source read by half the pairs, the read's
    leaves (basis, weights, q, k, v), per-source log measures and an output
    gradient."""
    g = torch.Generator().manual_seed(seed)
    n_q, n_b, h, kc, c, f = 37, 50, 8, 4, 8, 16
    counts = torch.randint(0, 5000, (n_q,), generator=g)
    counts[3] = 0
    p = int(counts.sum())
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b - 5, (p,), generator=g)
    bj[: p // 2] = 0
    leaves = [
        torch.randn(p, f, generator=g, dtype=D),
        0.3 * torch.randn(h, f, generator=g, dtype=D),
        torch.randn(n_q, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, c, generator=g, dtype=D),
    ]
    logm = -torch.rand(n_b, generator=g, dtype=D) * 8
    out_grad = torch.randn(n_q, h, c, generator=g, dtype=D)
    return types.SimpleNamespace(
        n_q=n_q,
        n_b=n_b,
        counts=counts,
        qi=qi,
        bj=bj,
        leaves=leaves,
        logm=logm,
        out_grad=out_grad,
    )


class _SinkRead(torch.autograd.Function):
    """``basis @ w.T`` handing its basis gradient to the fused read's sink, as the
    fused read's backward does (a CPU stand-in for its Triton kernels)."""

    @staticmethod
    def forward(ctx, basis, token, grad, w):
        ctx.save_for_backward(basis, w)
        ctx.grad, ctx.sink_node = grad, token.grad_fn
        return basis @ w.t()

    @staticmethod
    def backward(ctx, dout):
        basis, w = ctx.saved_tensors
        acc = fused_read._basis_grad_target(ctx, basis)
        if acc is not None:
            acc += dout @ w
        return None, None if acc is None else basis.new_empty(0), None, dout.t() @ basis
