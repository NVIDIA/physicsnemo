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

"""Single-process tests for the tensor helpers in ``physicsnemo.distributed.utils``."""

import pytest
import torch

from physicsnemo.distributed.utils import (
    compute_split_shapes,
    pad_helper,
    split_tensor_along_dim,
    truncate_helper,
)


@pytest.mark.parametrize(
    "size, num_chunks, expected",
    [
        (8, 4, [2, 2, 2, 2]),
        (10, 4, [3, 3, 3, 1]),
        (10, 6, [1, 1, 1, 1, 1, 5]),
        (7, 1, [7]),
    ],
)
def test_compute_split_shapes(size, num_chunks, expected):
    assert compute_split_shapes(size, num_chunks) == expected


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
def test_pad_helper_zero_pads_requested_dim(device, dim):
    x = torch.randn(2, 3, 4, device=device)
    out = pad_helper(x, dim, x.shape[dim] + 2)
    expected_shape = list(x.shape)
    expected_shape[dim] += 2
    assert list(out.shape) == expected_shape
    kept = out.narrow(dim, 0, x.shape[dim])
    torch.testing.assert_close(kept, x)
    assert torch.all(out.narrow(dim, x.shape[dim], 2) == 0)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_pad_helper_conj_pads_requested_dim(device, dim):
    x = torch.randn(3, 4, 5, dtype=torch.complex64, device=device)
    pad = 2
    out = pad_helper(x, dim, x.shape[dim] + pad, mode="conj")
    # The padded entries are the conjugate of entries 1..pad, in reverse order.
    expected = torch.cat(
        [x, torch.flip(torch.conj(x.narrow(dim, 1, pad)), dims=[dim])], dim=dim
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_truncate_helper(device, dim):
    x = torch.randn(3, 4, 5, device=device)
    out = truncate_helper(x, dim, 2)
    torch.testing.assert_close(out, x.narrow(dim, 0, 2))


def test_truncate_helper_preserves_channels_last(device):
    x = torch.randn(2, 3, 4, 4, device=device).to(memory_format=torch.channels_last)
    out = truncate_helper(x, 1, 2)
    assert out.is_contiguous(memory_format=torch.channels_last)


def test_split_tensor_along_dim(device):
    x = torch.arange(10, device=device).reshape(1, 10)
    chunks = split_tensor_along_dim(x, 1, 4)
    assert [c.shape[1] for c in chunks] == [3, 3, 3, 1]
    torch.testing.assert_close(torch.cat(chunks, dim=1), x)


def test_split_tensor_along_dim_errors():
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="cannot be split along 2"):
        split_tensor_along_dim(x, 2, 2)
    with pytest.raises(ValueError, match="cannot split dim 1 of size 3 into 4"):
        split_tensor_along_dim(x, 1, 4)
