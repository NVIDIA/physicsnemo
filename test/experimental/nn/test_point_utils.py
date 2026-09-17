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

"""Tests for the generic batching/mask/k-NN helpers under experimental/nn."""

import pytest
import torch

from physicsnemo.experimental.nn import (
    chunked_knn_indices,
    compute_batch_offset_step,
    counts_to_mask,
    flatten_batched_coords,
    flatten_padded_batch,
    gather_rows,
    masked_mean,
    unflatten_to_padded,
)

# ---------------------------------------------------------------------------
# gather_rows
# ---------------------------------------------------------------------------


def test_gather_rows_basic(device):
    """``gather_rows`` indexes into the first axis and broadcasts trailing dims."""
    x = torch.arange(20, device=device, dtype=torch.float32).reshape(10, 2)
    idx = torch.tensor([[0, 1], [2, 3]], device=device)
    out = gather_rows(x, idx)
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0, 0], x[0])
    assert torch.equal(out[1, 1], x[3])


# ---------------------------------------------------------------------------
# counts_to_mask
# ---------------------------------------------------------------------------


def test_counts_to_mask(device):
    """Counts become left-aligned True positions in each row."""
    counts = torch.tensor([3, 1, 5], device=device)
    m = counts_to_mask(counts)
    assert m.shape == (3, 5)
    assert m.sum().item() == 9
    assert m[1].tolist() == [True, False, False, False, False]


def test_counts_to_mask_rank_check():
    """A rank-2 counts tensor is rejected."""
    with pytest.raises(ValueError, match="rank-1"):
        counts_to_mask(torch.zeros(2, 3, dtype=torch.long))


# ---------------------------------------------------------------------------
# flatten_padded_batch / unflatten_to_padded
# ---------------------------------------------------------------------------


def test_flatten_unflatten_round_trip(device):
    """Round-trip: padded → flat → padded recovers valid entries."""
    x = torch.randn(2, 5, 3, device=device)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]],
        device=device,
    )
    flat = flatten_padded_batch(x, mask)
    back = unflatten_to_padded(flat, mask)
    assert flat.shape == (7, 3)
    assert torch.equal(x[mask], back[mask])
    # Padding positions in the unflatten output are zero.
    assert torch.equal(back[~mask], torch.zeros_like(back[~mask]))


def test_flatten_padded_batch_shape_mismatch_raises():
    """Mismatching mask vs ``x`` is rejected."""
    with pytest.raises(ValueError, match="x.shape"):
        flatten_padded_batch(torch.zeros(2, 5, 3), torch.zeros(3, 5, dtype=torch.bool))


def test_unflatten_to_padded_row_count_check():
    """The flat row count must match ``mask.sum()``."""
    flat = torch.zeros(2, 4)
    mask = torch.tensor([[True, True], [True, False]])
    with pytest.raises(ValueError, match="flat row count must match"):
        unflatten_to_padded(flat, mask)


# ---------------------------------------------------------------------------
# compute_batch_offset_step + flatten_batched_coords
# ---------------------------------------------------------------------------


def test_compute_batch_offset_step_unbatched(device):
    """Rank-2 coords return the safe step from coordinate extents."""
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    step = compute_batch_offset_step(coords)
    assert step == pytest.approx(4.0 * 2.0, abs=1e-6)


def test_compute_batch_offset_step_empty_falls_back():
    """All-False mask returns the ``1.0`` fallback."""
    coords = torch.zeros(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    assert compute_batch_offset_step(coords, mask) == 1.0


def test_compute_batch_offset_step_rank_check():
    """Wrong rank coords are rejected."""
    with pytest.raises(ValueError, match="rank-2/3"):
        compute_batch_offset_step(torch.zeros(4))


def test_flatten_batched_coords_shapes(device):
    """Flatten preserves the valid count and emits per-batch shift."""
    coords = torch.randn(2, 4, 3, device=device)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]], device=device
    )
    flat, flat_off, batch_ids = flatten_batched_coords(coords, mask, offset_step=10.0)
    assert flat.shape == (5, 3)
    assert flat_off.shape == (5, 3)
    assert batch_ids.shape == (5,)
    # Per-batch offset on the first coord axis only.
    diff = flat_off[:, 0] - flat[:, 0]
    expected_diff = batch_ids.to(dtype=flat.dtype) * 10.0
    assert torch.allclose(diff, expected_diff)
    # Other axes are unchanged.
    assert torch.equal(flat[:, 1:], flat_off[:, 1:])


# ---------------------------------------------------------------------------
# chunked_knn_indices
# ---------------------------------------------------------------------------


def test_chunked_knn_indices_cpu(device):
    """Auto-backend on CPU tensors returns a valid index tensor of the right shape."""
    q = torch.randn(50, 3, device=device)
    k = torch.randn(20, 3, device=device)
    idx = chunked_knn_indices(
        query_coords=q, key_coords=k, k=4, chunk_size=16, dilation=1
    )
    assert idx.shape == (50, 4)
    assert idx.dtype == torch.long
    assert int(idx.min().item()) >= 0
    assert int(idx.max().item()) < 20


def test_chunked_knn_indices_validation():
    """Empty inputs / wrong rank / non-positive sizes are rejected."""
    with pytest.raises(ValueError, match="rank-2"):
        chunked_knn_indices(
            query_coords=torch.zeros(5),
            key_coords=torch.zeros(5, 3),
            k=2,
            chunk_size=8,
        )
    with pytest.raises(ValueError, match="at least one point"):
        chunked_knn_indices(
            query_coords=torch.zeros(0, 3),
            key_coords=torch.zeros(5, 3),
            k=2,
            chunk_size=8,
        )
    with pytest.raises(ValueError, match="must be positive"):
        chunked_knn_indices(
            query_coords=torch.zeros(5, 3),
            key_coords=torch.zeros(5, 3),
            k=0,
            chunk_size=8,
        )


# ---------------------------------------------------------------------------
# masked_mean
# ---------------------------------------------------------------------------


def test_masked_mean_unmasked(device):
    """``mask=None`` reduces to a plain mean along the token axis."""
    x = torch.randn(8, 5, device=device)
    m = masked_mean(x, None)
    assert m.shape == (1, 5)
    assert torch.allclose(m, x.mean(dim=0, keepdim=True))


def test_masked_mean_unmasked_rank3_keeps_dim(device):
    """Rank-3 ``mask=None`` returns ``(B, 1, F)`` (matching the masked path)."""
    x = torch.randn(3, 8, 5, device=device)
    m = masked_mean(x, None)
    assert m.shape == (3, 1, 5)
    assert torch.allclose(m, x.mean(dim=1, keepdim=True))


def test_masked_mean_masked(device):
    """Only ``mask==True`` rows contribute."""
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    mask = torch.tensor([True, False, True], device=device)
    m = masked_mean(x, mask)
    # (1+3)/2 = 2 per channel.
    assert torch.allclose(m, torch.tensor([[2.0, 2.0]], device=device))
