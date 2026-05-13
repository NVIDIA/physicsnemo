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

r"""Lightweight unit tests for :mod:`physicsnemo.nn.functional.natten`.

Validates that the ``na1d``, ``na2d``, and ``na3d`` wrappers:

* produce bit-identical results to the underlying ``natten.functional`` calls,
* preserve tensor shapes through forward and backward,
* correctly dispatch through ``__torch_function__`` for tensor subclasses,
* degenerate to standard scaled dot-product attention when the kernel window
  covers the entire spatial extent.
"""

import pytest
import torch
import torch.nn.functional as F

from test.conftest import requires_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DispatchRecorder(torch.Tensor):
    """Minimal tensor subclass that records ``__torch_function__`` calls."""

    dispatched_funcs: list = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        cls.dispatched_funcs.append(func)
        plain_args = tuple(
            a.as_subclass(torch.Tensor) if isinstance(a, cls) else a for a in args
        )
        return func(*plain_args, **kwargs)


def _sdpa_reference(q, k, v):
    """Compute standard scaled dot-product attention over the last two dims.

    Expects (B, *spatial, heads, D) layout.  Flattens spatial dims so every
    token attends to every other token, then reshapes back.
    """
    *leading, heads, d = q.shape
    B = leading[0]
    spatial = leading[1:]
    S = 1
    for s in spatial:
        S *= s
    # -> (B, heads, S, D) for F.scaled_dot_product_attention
    q_flat = q.reshape(B, S, heads, d).permute(0, 2, 1, 3)
    k_flat = k.reshape(B, S, heads, d).permute(0, 2, 1, 3)
    v_flat = v.reshape(B, S, heads, d).permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat)
    # -> (B, *spatial, heads, D)
    return out.permute(0, 2, 1, 3).reshape(*leading, heads, d)


# ---------------------------------------------------------------------------
# 1-D neighbourhood attention
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA1D:
    """Unit tests for :func:`physicsnemo.nn.functional.natten.na1d`."""

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_matches_natten_directly(self, device, kernel_size, dilation):
        """Wrapper output must be identical to ``natten.functional.na1d``."""
        import natten.functional as nf

        from physicsnemo.nn.functional.natten import na1d

        B, L, H, D = 2, 16, 4, 8
        q = torch.randn(B, L, H, D, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected = nf.na1d(q, k, v, kernel_size, dilation=dilation)
        actual = na1d(q, k, v, kernel_size, dilation=dilation)

        torch.testing.assert_close(actual, expected)

    def test_output_shape(self, device):
        """Output shape must equal the query shape."""
        from physicsnemo.nn.functional.natten import na1d

        B, L, H, D = 1, 12, 2, 16
        q = torch.randn(B, L, H, D, device=device)
        out = na1d(q, q, q, kernel_size=3)
        assert out.shape == q.shape

    def test_backward(self, device):
        """Gradients must flow back through all three inputs."""
        from physicsnemo.nn.functional.natten import na1d

        B, L, H, D = 1, 12, 2, 8
        q = torch.randn(B, L, H, D, device=device, requires_grad=True)
        k = torch.randn(B, L, H, D, device=device, requires_grad=True)
        v = torch.randn(B, L, H, D, device=device, requires_grad=True)

        try:
            out = na1d(q, k, v, kernel_size=3)
        except NotImplementedError as e:
            # natten's default CPU backend is FlexAttention, whose
            # forward refuses requires_grad inputs on CPU.  Skip only
            # in that specific case -- if natten picks a different
            # backend (e.g. "reference") that supports CPU backward,
            # the test should still run.
            if "FlexAttention does not support backward on CPU" in str(e):
                pytest.skip(
                    "natten chose FlexAttention backend; CPU backward unsupported"
                )
            raise
        out.sum().backward()

        for name, t in [("q", q), ("k", k), ("v", v)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.shape == t.shape

    def test_torch_function_dispatch(self, device):
        """``__torch_function__`` must be invoked for tensor subclasses."""
        from physicsnemo.nn.functional.natten import na1d

        B, L, H, D = 1, 8, 2, 8
        q = torch.randn(B, L, H, D, device=device).as_subclass(_DispatchRecorder)
        k = torch.randn(B, L, H, D, device=device).as_subclass(_DispatchRecorder)
        v = torch.randn(B, L, H, D, device=device).as_subclass(_DispatchRecorder)

        _DispatchRecorder.dispatched_funcs.clear()
        na1d(q, k, v, kernel_size=3)

        assert na1d in _DispatchRecorder.dispatched_funcs

    def test_full_window_matches_sdpa(self, device):
        """When kernel covers the entire sequence, NA degenerates to SDPA."""
        from physicsnemo.nn.functional.natten import na1d

        B, L, H, D = 2, 7, 2, 8
        q = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # kernel_size == L means every token sees every other token
        na_out = na1d(q, k, v, kernel_size=L)
        sdpa_out = _sdpa_reference(q, k, v)

        torch.testing.assert_close(na_out, sdpa_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2-D neighbourhood attention
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA2D:
    """Unit tests for :func:`physicsnemo.nn.functional.natten.na2d`."""

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_matches_natten_directly(self, device, kernel_size, dilation):
        """Wrapper output must be identical to ``natten.functional.na2d``."""
        import natten.functional as nf

        from physicsnemo.nn.functional.natten import na2d

        B, Ht, W, H, D = 2, 16, 16, 4, 8
        q = torch.randn(B, Ht, W, H, D, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected = nf.na2d(q, k, v, kernel_size, dilation=dilation)
        actual = na2d(q, k, v, kernel_size, dilation=dilation)

        torch.testing.assert_close(actual, expected)

    def test_output_shape(self, device):
        """Output shape must equal the query shape."""
        from physicsnemo.nn.functional.natten import na2d

        B, Ht, W, H, D = 1, 6, 6, 2, 16
        q = torch.randn(B, Ht, W, H, D, device=device)
        out = na2d(q, q, q, kernel_size=3)
        assert out.shape == q.shape

    def test_backward(self, device):
        """Gradients must flow back through all three inputs."""
        from physicsnemo.nn.functional.natten import na2d

        B, Ht, W, H, D = 1, 6, 6, 2, 8
        q = torch.randn(B, Ht, W, H, D, device=device, requires_grad=True)
        k = torch.randn(B, Ht, W, H, D, device=device, requires_grad=True)
        v = torch.randn(B, Ht, W, H, D, device=device, requires_grad=True)

        try:
            out = na2d(q, k, v, kernel_size=3)
        except NotImplementedError as e:
            # natten's default CPU backend is FlexAttention, whose
            # forward refuses requires_grad inputs on CPU.  Skip only
            # in that specific case -- if natten picks a different
            # backend (e.g. "reference") that supports CPU backward,
            # the test should still run.
            if "FlexAttention does not support backward on CPU" in str(e):
                pytest.skip(
                    "natten chose FlexAttention backend; CPU backward unsupported"
                )
            raise
        out.sum().backward()

        for name, t in [("q", q), ("k", k), ("v", v)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.shape == t.shape

    def test_torch_function_dispatch(self, device):
        """``__torch_function__`` must be invoked for tensor subclasses."""
        from physicsnemo.nn.functional.natten import na2d

        B, Ht, W, H, D = 1, 4, 4, 2, 8
        q = torch.randn(B, Ht, W, H, D, device=device).as_subclass(_DispatchRecorder)
        k = torch.randn(B, Ht, W, H, D, device=device).as_subclass(_DispatchRecorder)
        v = torch.randn(B, Ht, W, H, D, device=device).as_subclass(_DispatchRecorder)

        _DispatchRecorder.dispatched_funcs.clear()
        na2d(q, k, v, kernel_size=3)

        assert na2d in _DispatchRecorder.dispatched_funcs

    def test_full_window_matches_sdpa(self, device):
        """When kernel covers the full spatial extent, NA degenerates to SDPA."""
        from physicsnemo.nn.functional.natten import na2d

        B, Ht, W, H, D = 2, 5, 5, 2, 8
        q = torch.randn(B, Ht, W, H, D, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        na_out = na2d(q, k, v, kernel_size=max(Ht, W))
        sdpa_out = _sdpa_reference(q, k, v)

        torch.testing.assert_close(na_out, sdpa_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3-D neighbourhood attention
# ---------------------------------------------------------------------------


@requires_module("natten")
class TestNA3D:
    """Unit tests for :func:`physicsnemo.nn.functional.natten.na3d`."""

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_matches_natten_directly(self, device, kernel_size):
        """Wrapper output must be identical to ``natten.functional.na3d``."""
        import natten.functional as nf

        from physicsnemo.nn.functional.natten import na3d

        B, X, Y, Z, H, D = 1, 16, 16, 16, 2, 8
        q = torch.randn(B, X, Y, Z, H, D, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected = nf.na3d(q, k, v, kernel_size)
        actual = na3d(q, k, v, kernel_size)

        torch.testing.assert_close(actual, expected)

    def test_output_shape(self, device):
        """Output shape must equal the query shape."""
        from physicsnemo.nn.functional.natten import na3d

        B, X, Y, Z, H, D = 1, 4, 4, 4, 2, 8
        q = torch.randn(B, X, Y, Z, H, D, device=device)
        out = na3d(q, q, q, kernel_size=3)
        assert out.shape == q.shape

    def test_backward(self, device):
        """Gradients must flow back through all three inputs."""
        from physicsnemo.nn.functional.natten import na3d

        B, X, Y, Z, H, D = 1, 4, 4, 4, 2, 8
        q = torch.randn(B, X, Y, Z, H, D, device=device, requires_grad=True)
        k = torch.randn(B, X, Y, Z, H, D, device=device, requires_grad=True)
        v = torch.randn(B, X, Y, Z, H, D, device=device, requires_grad=True)

        try:
            out = na3d(q, k, v, kernel_size=3)
        except NotImplementedError as e:
            # natten's default CPU backend is FlexAttention, whose
            # forward refuses requires_grad inputs on CPU.  Skip only
            # in that specific case -- if natten picks a different
            # backend (e.g. "reference") that supports CPU backward,
            # the test should still run.
            if "FlexAttention does not support backward on CPU" in str(e):
                pytest.skip(
                    "natten chose FlexAttention backend; CPU backward unsupported"
                )
            raise
        out.sum().backward()

        for name, t in [("q", q), ("k", k), ("v", v)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.shape == t.shape

    def test_torch_function_dispatch(self, device):
        """``__torch_function__`` must be invoked for tensor subclasses."""
        from physicsnemo.nn.functional.natten import na3d

        B, X, Y, Z, H, D = 1, 4, 4, 4, 2, 8
        q = torch.randn(B, X, Y, Z, H, D, device=device).as_subclass(_DispatchRecorder)
        k = torch.randn(B, X, Y, Z, H, D, device=device).as_subclass(_DispatchRecorder)
        v = torch.randn(B, X, Y, Z, H, D, device=device).as_subclass(_DispatchRecorder)

        _DispatchRecorder.dispatched_funcs.clear()
        na3d(q, k, v, kernel_size=3)

        assert na3d in _DispatchRecorder.dispatched_funcs

    def test_full_window_matches_sdpa(self, device):
        """When kernel covers the full spatial extent, NA degenerates to SDPA."""
        from physicsnemo.nn.functional.natten import na3d

        B, X, Y, Z, H, D = 1, 5, 5, 5, 2, 8
        q = torch.randn(B, X, Y, Z, H, D, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        na_out = na3d(q, k, v, kernel_size=max(X, Y, Z))
        sdpa_out = _sdpa_reference(q, k, v)

        torch.testing.assert_close(na_out, sdpa_out, atol=1e-5, rtol=1e-5)
