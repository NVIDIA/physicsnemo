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

"""Kernel-read cases shared by the ISLA kernel tests: one case's geometry, plan and
basis MLP for the kernel basis, its chunked basis and gradients, a move of a case to a
device and dtype, and the 2x-error rule of a fast kernel against its eager reference."""

import types

import torch

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
    print(f"{name}: fast {d_fast:.2e}, eager {d_eager:.2e}")
    assert d_fast <= 2 * d_eager + floor, (name, float(d_fast), float(d_eager))
