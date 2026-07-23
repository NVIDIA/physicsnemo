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

from physicsnemo.nn.functional import segment_softmax
from physicsnemo.nn.functional.segments import SegmentSoftmax


def _reference_segment_softmax(logits: torch.Tensor, offsets: torch.Tensor):
    chunks = []
    offsets_cpu = offsets.detach().cpu()
    for i in range(int(offsets_cpu.numel()) - 1):
        start = int(offsets_cpu[i].item())
        end = int(offsets_cpu[i + 1].item())
        if end > start:
            chunks.append(torch.softmax(logits[start:end], dim=0))
    if not chunks:
        return logits.clone()
    return torch.cat(chunks, dim=0)


def _assert_segment_sums_are_one(weights: torch.Tensor, offsets: torch.Tensor):
    offsets_cpu = offsets.detach().cpu()
    for i in range(int(offsets_cpu.numel()) - 1):
        start = int(offsets_cpu[i].item())
        end = int(offsets_cpu[i + 1].item())
        if end > start:
            sums = weights[start:end].sum(dim=0)
            torch.testing.assert_close(sums, torch.ones_like(sums))


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_segment_softmax_known_answer(device: str, implementation: str):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    logits = torch.tensor([2.0, 1.0, 0.0, 4.0, 2.0], device=device)
    offsets = torch.tensor([0, 3, 5], device=device, dtype=torch.int64)

    out = segment_softmax(logits, offsets, implementation=implementation)
    expected = torch.cat(
        [
            torch.softmax(logits[:3], dim=0),
            torch.softmax(logits[3:], dim=0),
        ]
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    _assert_segment_sums_are_one(out, offsets)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_segment_softmax_empty_segments(device: str, implementation: str):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    logits = torch.tensor([[1.0, 0.0], [2.0, 4.0], [-1.0, 3.0]], device=device)
    offsets = torch.tensor([0, 0, 2, 2, 3], device=device, dtype=torch.int64)

    out = segment_softmax(logits, offsets, implementation=implementation)
    expected = _reference_segment_softmax(logits, offsets)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    _assert_segment_sums_are_one(out, offsets)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("tail_shape", [(), (4,), (2, 3)])
def test_segment_softmax_trailing_dims(device, implementation, tail_shape):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    torch.manual_seed(12)
    offsets = torch.tensor([0, 1, 4, 4, 9, 10], device=device, dtype=torch.int64)
    logits = torch.randn((10, *tail_shape), device=device)

    out = segment_softmax(logits, offsets, implementation=implementation)
    expected = _reference_segment_softmax(logits, offsets)
    assert out.shape == logits.shape
    assert out.dtype == logits.dtype
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    _assert_segment_sums_are_one(out, offsets)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_segment_softmax_stable_large_logits(device, implementation):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    logits = torch.tensor(
        [[10000.0, -10000.0], [9999.0, -9999.0], [0.0, 0.0]],
        device=device,
    )
    offsets = torch.tensor([0, 2, 3], device=device, dtype=torch.int64)

    out = segment_softmax(logits, offsets, implementation=implementation)
    expected = _reference_segment_softmax(logits, offsets)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_segment_softmax_offset_dtypes(device: str, offset_dtype: torch.dtype):
    logits = torch.randn(6, 3, device=device)
    offsets = torch.tensor([0, 2, 6], device=device, dtype=offset_dtype)
    out = segment_softmax(logits, offsets, implementation="torch")
    expected = _reference_segment_softmax(logits, offsets)
    torch.testing.assert_close(out, expected)


def test_segment_softmax_backward_torch_matches_reference(device: str):
    torch.manual_seed(4)
    offsets = torch.tensor([0, 2, 5, 9], device=device, dtype=torch.int64)
    logits = torch.randn(9, 3, device=device, dtype=torch.float64, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    weight = torch.randn_like(logits)

    out = segment_softmax(logits, offsets, implementation="torch")
    ref = _reference_segment_softmax(ref_logits, offsets)
    (out * weight).sum().backward()
    (ref * weight).sum().backward()

    torch.testing.assert_close(logits.grad, ref_logits.grad, atol=1e-8, rtol=1e-8)


def test_segment_softmax_dispatch_cpu_matches_torch():
    torch.manual_seed(8)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
    logits = torch.randn(6, 3)

    out_default = segment_softmax(logits, offsets)
    out_torch = segment_softmax(logits, offsets, implementation="torch")
    torch.testing.assert_close(out_default, out_torch)


def test_segment_softmax_backend_forward_parity(device: str):
    if "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    torch.manual_seed(9)
    offsets = torch.tensor([0, 3, 8, 8, 11, 17], device=device, dtype=torch.int64)
    logits = torch.randn(17, 5, device=device, dtype=torch.float32)

    out_torch = segment_softmax(logits, offsets, implementation="torch")
    out_warp = segment_softmax(logits, offsets, implementation="warp")
    SegmentSoftmax.compare_forward(out_warp, out_torch)


def test_segment_softmax_backend_backward_parity(device: str):
    if "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    torch.manual_seed(10)
    offsets = torch.tensor([0, 2, 6, 7, 13], device=device, dtype=torch.int64)
    logits_torch = torch.randn(13, 4, device=device, dtype=torch.float32)
    logits_warp = logits_torch.detach().clone()
    logits_torch.requires_grad_(True)
    logits_warp.requires_grad_(True)
    weight = torch.randn_like(logits_torch)

    out_torch = segment_softmax(logits_torch, offsets, implementation="torch")
    out_warp = segment_softmax(logits_warp, offsets, implementation="warp")
    (out_torch * weight).sum().backward()
    (out_warp * weight).sum().backward()

    torch.testing.assert_close(logits_warp.grad, logits_torch.grad, atol=1e-5, rtol=1e-5)


def test_segment_softmax_empty_input(device: str):
    logits = torch.empty(0, 3, device=device)
    offsets = torch.tensor([0, 0, 0], device=device, dtype=torch.int64)
    out = segment_softmax(logits, offsets, implementation="torch")
    assert out.shape == logits.shape
    assert out.numel() == 0


def test_segment_softmax_empty_trailing_dimension(device: str):
    logits = torch.empty(4, 0, device=device)
    offsets = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)
    out = segment_softmax(logits, offsets, implementation="torch")
    assert out.shape == logits.shape
    assert out.numel() == 0


def test_segment_softmax_error_handling(device: str):
    logits = torch.randn(4, device=device)
    with pytest.raises(ValueError, match="rank-1"):
        segment_softmax(logits, torch.tensor([[0, 4]], device=device))
    with pytest.raises(ValueError, match="int32 or int64"):
        segment_softmax(logits, torch.tensor([0.0, 4.0], device=device))
    with pytest.raises(ValueError, match="start at 0"):
        segment_softmax(logits, torch.tensor([1, 4], device=device))
    with pytest.raises(ValueError, match="offsets\\[-1\\]"):
        segment_softmax(logits, torch.tensor([0, 3], device=device))
    with pytest.raises(ValueError, match="monotonically"):
        segment_softmax(logits, torch.tensor([0, 3, 2, 4], device=device))
    with pytest.raises(ValueError, match="floating point"):
        segment_softmax(torch.arange(4, device=device), torch.tensor([0, 4], device=device))


def test_segment_softmax_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(SegmentSoftmax.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    out = SegmentSoftmax.dispatch(*args, implementation="torch", **kwargs)
    assert out.shape == args[0].shape


def test_segment_softmax_make_inputs_backward(device: str):
    label, args, kwargs = next(iter(SegmentSoftmax.make_inputs_backward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    logits, _ = args
    assert logits.requires_grad
    out = SegmentSoftmax.dispatch(*args, implementation="torch", **kwargs)
    assert out.shape == logits.shape


def test_segment_softmax_compare_forward_contract(device: str):
    _, args, kwargs = next(iter(SegmentSoftmax.make_inputs_forward(device)))
    out = SegmentSoftmax.dispatch(*args, implementation="torch", **kwargs)
    SegmentSoftmax.compare_forward(out, out.detach().clone())


def test_segment_softmax_compare_backward_contract(device: str):
    _, args, kwargs = next(iter(SegmentSoftmax.make_inputs_backward(device)))
    logits, _ = args
    out = SegmentSoftmax.dispatch(*args, implementation="torch", **kwargs)
    out.square().mean().backward()
    SegmentSoftmax.compare_backward(logits.grad, logits.grad.detach().clone())


def test_segment_softmax_opcheck(device: str):
    if "cpu" in device:
        pytest.skip("warp segment_softmax backend is CUDA-only")
    from physicsnemo.nn.functional.segments.segment_softmax._warp_impl import (
        segment_softmax as segment_softmax_warp,
    )

    logits = torch.randn(7, 2, device=device, dtype=torch.float32, requires_grad=True)
    offsets = torch.tensor([0, 2, 7], device=device, dtype=torch.int64)
    torch.library.opcheck(segment_softmax_warp, (logits, offsets))
