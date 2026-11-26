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

"""Replicated-Q / sharded-KV SDPA: parity against unsharded SDPA across
backends (``PHYSICSNEMO_DP_SDPA_BACKEND``), dtypes, uneven shards, head dims."""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

ENV = "PHYSICSNEMO_DP_SDPA_BACKEND"


def _ap():
    from physicsnemo.domain_parallel.shard_utils import attention_patches as ap

    return ap


def _backend_helper():
    return _ap()._replicated_q_sdpa_backend


def torch_can_use_flash(q, k, v) -> bool:
    r"""PyTorch's own flash eligibility for these shapes (with grad)."""
    if not q.is_cuda:
        return False
    params = torch.backends.cuda.SDPAParams(
        *(t.detach().requires_grad_(True) for t in (q, k, v)), None, 0.0, False, False
    )
    return torch.backends.cuda.can_use_flash_attention(params)


@pytest.fixture
def fresh_backend_cache():
    r"""Clear the cached flash capability probe around a test."""
    _ap()._flash_attention_supported.cache_clear()
    yield
    _ap()._flash_attention_supported.cache_clear()


def _grad_rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["auto", "flash", "efficient"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("head_dim", [16, 64])
@pytest.mark.parametrize("seq_len_kv", [1001, 4099, 20001])
def test_replicated_q_flash_parity(
    distributed_mesh, monkeypatch, backend, dtype, batch_size, head_dim, seq_len_kv
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if backend == "flash" and dtype == torch.float32:
        pytest.skip("flash-ineligible: fp32 (forced flash raises by contract)")
    monkeypatch.setenv(ENV, backend)

    dm = DistributedManager()
    H, Sq = 8, 64
    torch.manual_seed(1234)  # same data on every rank
    mk = lambda s: torch.randn(  # noqa: E731
        batch_size, H, s, head_dim, device=dm.device, dtype=dtype
    )
    q, k, v = mk(Sq), mk(seq_len_kv), mk(seq_len_kv)

    # Unsharded reference.
    rq, rk, rv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(rq, rk, rv)
    ref.float().square().sum().backward()

    sq = scatter_tensor(q, 0, distributed_mesh, (Replicate(),), requires_grad=True)
    sk = scatter_tensor(k, 0, distributed_mesh, (Shard(2),), requires_grad=True)
    sv = scatter_tensor(v, 0, distributed_mesh, (Shard(2),), requires_grad=True)

    flash_ok = dtype != torch.float32 and torch_can_use_flash(q, k, v)
    if backend == "flash" and not flash_ok:
        pytest.skip("flash-ineligible: torch.backends.cuda.can_use_flash_attention")
    got = _ap()._replicated_q_sdpa_backend(q, k, v, None, 0.0, False)
    assert got == ("flash" if flash_ok and backend != "efficient" else "efficient")

    out = F.scaled_dot_product_attention(sq, sk, sv)
    out.float().square().sum().backward()

    if dtype == torch.float32:
        rtol = atol = 1e-5
        gtol = 1e-5
    else:
        rtol, atol, gtol = 2e-2, 2e-3, 2e-2

    torch.testing.assert_close(out.full_tensor(), ref.detach(), rtol=rtol, atol=atol)
    for name, s, r in (("q", sq, rq), ("k", sk, rk), ("v", sv, rv)):
        g = s.grad.full_tensor()
        err = _grad_rel_err(g, r.grad)
        assert err <= gtol, f"grad_{name} rel err {err:.3e} > {gtol}"


def test_replicated_q_backend_selector(monkeypatch, fresh_backend_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    helper = _backend_helper()
    dev = "cuda"
    q = torch.randn(1, 8, 64, 16, device=dev, dtype=torch.float16)
    k = torch.randn(1, 8, 100, 16, device=dev, dtype=torch.float16)
    flash_ok = torch_can_use_flash(q, k, k)
    monkeypatch.setenv(ENV, "auto")
    assert helper(q, k, k, None, 0.0, False) == ("flash" if flash_ok else "efficient")
    monkeypatch.setenv(ENV, "efficient")
    assert helper(q, k, k, None, 0.0, False) == "efficient"
    monkeypatch.setenv(ENV, "auto")
    assert helper(q.float(), k.float(), k.float(), None, 0.0, False) == "efficient"
    # The raw flash op does not pad the head dim.
    q12, k12 = q[..., :12], k[..., :12]
    assert helper(q12, k12, k12, None, 0.0, False) == "efficient"
    monkeypatch.setenv(ENV, "flash")
    with pytest.raises(RuntimeError, match="ineligible"):
        helper(q.float(), k.float(), k.float(), None, 0.0, False)
    monkeypatch.setenv(ENV, "bogus")
    with pytest.raises(ValueError):
        helper(q, k, k, None, 0.0, False)


def test_replicated_q_backend_respects_torch_capability(
    monkeypatch, fresh_backend_cache
):
    r"""Where PyTorch says flash cannot run (e.g. sm75, a build without flash),
    ``auto`` keeps the efficient kernel and forcing ``flash`` raises."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    helper = _backend_helper()
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 100, 64, device="cuda", dtype=torch.bfloat16)
    calls = []

    def no_flash(params, debug=False):
        calls.append(params)
        return False

    monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", no_flash)
    monkeypatch.setenv(ENV, "auto")
    assert helper(q, k, k, None, 0.0, False) == "efficient"
    assert helper(q, k, k, None, 0.0, False) == "efficient"
    assert len(calls) == 1, "capability probe must be cached per configuration"
    monkeypatch.setenv(ENV, "flash")
    with pytest.raises(RuntimeError, match="ineligible"):
        helper(q, k, k, None, 0.0, False)
    monkeypatch.setenv(ENV, "efficient")
    assert helper(q, k, k, None, 0.0, False) == "efficient"


def test_replicated_q_backend_respects_sdpa_kernel(monkeypatch, fresh_backend_cache):
    r"""An ``sdpa_kernel`` context that excludes flash keeps ``auto`` off flash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    helper = _backend_helper()
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 100, 64, device="cuda", dtype=torch.float16)
    monkeypatch.setenv(ENV, "auto")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert helper(q, k, k, None, 0.0, False) == "efficient"
        monkeypatch.setenv(ENV, "flash")
        with pytest.raises(RuntimeError, match="disabled"):
            helper(q, k, k, None, 0.0, False)
    monkeypatch.setenv(ENV, "auto")
    expected = "flash" if torch_can_use_flash(q, k, k) else "efficient"
    assert helper(q, k, k, None, 0.0, False) == expected
