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
