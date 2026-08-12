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

r"""Direct contract tests for the autograd boundary guards in
``shard_utils/grad_ops.py``.

``GradReducer`` distributed behavior is covered in
``test_async_collective_gradients.py`` and ``test_compile_ops.py``;
``ContiguousGrad`` is collective-free and tested here single-rank.
"""

import torch

from physicsnemo.domain_parallel.shard_utils.grad_ops import ContiguousGrad


def test_contiguous_grad_forward_is_identity():
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = ContiguousGrad.apply(x)

    assert torch.equal(y, x)
    # Alias, not a copy: forward must not materialize anything.
    assert y.data_ptr() == x.data_ptr()


def test_contiguous_grad_backward_makes_grad_contiguous():
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = ContiguousGrad.apply(x)

    # A permuted (non-contiguous) incoming gradient, as attention kernels
    # produce (BSHD memory viewed as BHSD).
    grad = torch.randn(2, 8, 4).permute(0, 2, 1)
    assert not grad.is_contiguous()

    y.backward(gradient=grad)

    assert x.grad is not None
    assert x.grad.is_contiguous()
    torch.testing.assert_close(x.grad, grad)


def test_contiguous_grad_passes_contiguous_grad_through():
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = ContiguousGrad.apply(x)

    grad = torch.randn(2, 4, 8)
    y.backward(gradient=grad)

    assert x.grad.is_contiguous()
    torch.testing.assert_close(x.grad, grad)


def test_contiguous_grad_preserves_non_contiguous_forward_layout():
    # The guard normalizes gradients, not activations: a non-contiguous
    # forward input passes through with its layout intact.
    x = torch.randn(2, 8, 4).permute(0, 2, 1)
    y = ContiguousGrad.apply(x)

    assert not y.is_contiguous()
    assert torch.equal(y, x)
