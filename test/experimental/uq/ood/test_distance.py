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

"""Generic distance scoring, application decisions, and fitted-state persistence."""

import io

import pytest
import torch

from physicsnemo.experimental.uq.ood import KNNScorer


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k", [1, 4, 20])
def test_chunked_leave_one_out_matches_dense(device, normalize, k, monkeypatch):
    """Bound every distance block and exclude only the matching reference row."""
    reference = torch.randn(11, 7, device=device, dtype=torch.float64)
    reference[1] = reference[0]  # Duplicate rows must remain legitimate neighbors.
    query = torch.randn(9, 7, device=device, dtype=torch.float64)
    ref = (
        reference / (reference.norm(dim=1, keepdim=True) + 1e-8)
        if normalize
        else reference
    )
    q = query / (query.norm(dim=1, keepdim=True) + 1e-8) if normalize else query
    dense = torch.cdist(ref, ref)
    dense.fill_diagonal_(float("inf"))
    expected_reference = dense.topk(min(k, 10), largest=False).values.mean(1)
    expected_query = torch.cdist(q, ref).topk(min(k, 11), largest=False).values.mean(1)
    original = torch.cdist
    calls = []

    def bounded_cdist(x, y, **kwargs):
        assert x.shape[0] <= 3 and y.shape[0] <= 4
        calls.append((x.shape[0], y.shape[0]))
        return original(x, y, **kwargs)

    monkeypatch.setattr(torch, "cdist", bounded_cdist)
    scorer = KNNScorer(k, normalize, query_chunk_size=3, reference_chunk_size=4)
    scorer.fit(reference)
    cached = scorer._normalized_reference
    torch.testing.assert_close(scorer.reference_scores(), expected_reference)
    torch.testing.assert_close(scorer.score(query), expected_query)
    assert scorer._normalized_reference is cached and len(calls) > 1
    restored = KNNScorer(query_chunk_size=3, reference_chunk_size=4)
    restored.load_state_dict(scorer.state_dict())
    torch.testing.assert_close(restored.reference_scores(), expected_reference)


def test_leave_one_out_requires_two_samples():
    scorer = KNNScorer()
    with pytest.raises(RuntimeError, match="fit"):
        scorer.reference_scores()
    scorer.fit(torch.ones(1, 2))
    with pytest.raises(ValueError, match="two"):
        scorer.reference_scores()


def test_normalization_setting_and_saved_state_remain_consistent():
    scorer = KNNScorer(k=1, normalize=False)
    scorer.fit(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    query = torch.tensor([[1.0, 0.0]])
    scorer.normalize = True
    torch.testing.assert_close(scorer.score(query), torch.zeros(1))
    restored = KNNScorer()
    restored.load_state_dict(scorer.state_dict())
    torch.testing.assert_close(restored.score(query), scorer.score(query))
    scorer.normalize = False
    torch.testing.assert_close(scorer.score(query), torch.ones(1))


@pytest.mark.parametrize("name", ["query_chunk_size", "reference_chunk_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_chunk_size(name, value):
    with pytest.raises(ValueError, match=name):
        KNNScorer(**{name: value})


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k", [1, 3, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distance_matches_dense_reference(device, normalize, k, dtype):
    """Compare dispatched nearest neighbors with an independent dense distance."""
    reference = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], device=device, dtype=dtype
    )
    query = torch.tensor(
        [[1.0, 1.0], [-2.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype
    )
    scorer = KNNScorer(k=k, normalize=normalize)
    scorer.fit(reference)
    if normalize:
        reference = reference / (reference.norm(dim=1, keepdim=True) + 1e-8)
        query_for_ref = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    else:
        query_for_ref = query
    expected = (
        torch.cdist(query_for_ref, reference)
        .sort(dim=1)
        .values[:, : min(k, 3)]
        .mean(dim=1)
    )
    torch.testing.assert_close(scorer.score(query), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scoring_and_policy_round_trip(device, dtype):
    """Scoring and restored decisions match an independent normalized distance."""
    torch.manual_seed(42)
    reference = torch.randn(15, 4, device=device, dtype=dtype)
    query = torch.randn(7, 4, device=device, dtype=dtype)
    scorer = KNNScorer(k=3)
    scorer.fit(reference)
    reference32, query32 = reference.float(), query.float()
    reference32 = reference32 / (reference32.norm(dim=1, keepdim=True) + 1e-8)
    query32 = query32 / (query32.norm(dim=1, keepdim=True) + 1e-8)
    expected = torch.cdist(query32, reference32).sort(dim=1).values[:, :3].mean(1)
    torch.testing.assert_close(scorer.score(query), expected)
    # A real weights-only serialization round trip, including application policy.
    threshold = expected.median()
    stream = io.BytesIO()
    torch.save({"scorer": scorer.state_dict(), "threshold": threshold}, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True, map_location=device)
    restored = KNNScorer(k=1, normalize=False)
    restored.load_state_dict(state["scorer"])
    scores = restored.score(query)
    torch.testing.assert_close(scores, expected)
    assert torch.equal(scores > state["threshold"], expected > threshold)
    assert restored.k == 3 and restored.normalize is True


def test_fit_and_snapshot_are_detached_copies(device):
    """Mutating training tensors or saved snapshots cannot corrupt the scorer."""
    reference = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0]], device=device, requires_grad=True
    )
    query = torch.tensor([[1.0, 0.0]], device=device, requires_grad=True)
    scorer = KNNScorer(k=1, normalize=False)
    scorer.fit(reference)
    with torch.no_grad():
        reference.add_(100)
    saved = scorer.state_dict()
    saved["reference"].add_(100)
    scores = scorer.score(query)
    torch.testing.assert_close(scores, torch.ones(1, device=device))
    assert not scores.requires_grad
    assert not scorer.state_dict()["reference"].requires_grad


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_reject_invalid_k(bad):
    """Invalid neighbor counts fail at construction."""
    with pytest.raises(ValueError, match="positive integer"):
        KNNScorer(k=bad)


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(2),
        torch.ones(0, 2),
        torch.ones(2, 0),
        torch.ones(2, 2, dtype=torch.int64),
        torch.tensor([[float("nan"), 0.0]]),
    ],
)
def test_invalid_fit_preserves_reference(bad):
    """A failed fit leaves a previously usable scorer intact."""
    scorer = KNNScorer(k=1, normalize=False)
    scorer.fit(torch.zeros(1, 2))
    with pytest.raises(ValueError):
        scorer.fit(bad)
    torch.testing.assert_close(scorer.score(torch.ones(1, 2)), torch.tensor([2**0.5]))


def test_lifecycle_and_validation():
    """Empty queries are valid; missing state and incompatible queries are not."""
    scorer = KNNScorer()
    with pytest.raises(RuntimeError, match="fit"):
        scorer.score(torch.ones(1, 2))
    with pytest.raises(RuntimeError, match="fit"):
        scorer.state_dict()
    scorer.fit(torch.zeros(1, 2))
    assert scorer.score(torch.empty(0, 2)).shape == (0,)
    with pytest.raises(ValueError, match="dimension"):
        scorer.score(torch.ones(1, 3))
    with pytest.raises(ValueError, match="finite"):
        scorer.score(torch.tensor([[float("inf"), 0.0]]))
    with pytest.raises(ValueError, match="overflow"):
        scorer.score(torch.full((1, 2), 1e100, dtype=torch.float64))
    state = scorer.state_dict()
    state["version"] = 2
    with pytest.raises(ValueError, match="version"):
        scorer.load_state_dict(state)
    state = scorer.state_dict()
    state["reference"] = torch.ones(0, 2)
    with pytest.raises(ValueError):
        scorer.load_state_dict(state)
    assert scorer.k == 10
    torch.testing.assert_close(scorer.score(torch.zeros(1, 2)), torch.zeros(1))
