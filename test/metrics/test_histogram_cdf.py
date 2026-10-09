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

import physicsnemo.metrics.general.histogram as hist


def test_cdf_accumulates_over_all_inputs(device):
    torch.manual_seed(0)
    x = torch.randn((10, 3, 4), device=device)
    y = torch.randn((5, 3, 4), device=device)

    bin_edges, cdf = hist.cdf(x, y, bins=10)

    # Every value is a fraction of all 15 samples, so nothing exceeds one.
    assert torch.all(cdf <= 1.0)
    assert torch.allclose(cdf[-1], torch.ones_like(cdf[-1]))

    # Reference built from the concatenated samples.
    xy = torch.cat((x, y), dim=0)
    total = xy.shape[0]
    cumulative = [
        (xy < bin_edges[i + 1]).sum(dim=0) for i in range(bin_edges.shape[0] - 2)
    ]
    cumulative.append(torch.full(xy.shape[1:], total, device=device))
    reference = torch.stack(cumulative) / total
    assert torch.allclose(cdf, reference)


@pytest.mark.parametrize("cdf", [False, True])
def test_bin_reductions_accumulate_into_existing_counts(device, cdf):
    torch.manual_seed(0)
    x = torch.randn((7, 3, 4), device=device)
    bin_edges = hist.linspace(x.min(dim=0)[0], x.max(dim=0)[0], 10)
    number_of_bins = bin_edges.shape[0] - 1
    existing = torch.randint(0, 5, (number_of_bins, 3, 4), device=device)

    if cdf:
        low_memory = hist._low_memory_bin_reduction_cdf
        high_memory = hist._high_memory_bin_reduction_cdf
    else:
        low_memory = hist._low_memory_bin_reduction_counts
        high_memory = hist._high_memory_bin_reduction_counts

    counts_low = low_memory(x, bin_edges, existing.clone(), number_of_bins)
    counts_high = high_memory(x, bin_edges, existing.clone(), number_of_bins)

    # Both routines add to the counts they are given and have to agree.
    assert torch.equal(counts_low, counts_high)
    if cdf:
        assert torch.equal(counts_high[-1], existing[-1] + x.shape[0])


def test_histogram_update_keeps_number_of_bins_consistent(device):
    torch.manual_seed(0)
    x = torch.randn((10, 3, 4), device=device)
    H = hist.Histogram((1, 3, 4), bins=10, device=device)
    H(x)
    assert H.number_of_bins == 10

    # Shifted data lies outside the current range, so the bin edges get extended.
    bin_edges, counts = H.update(x + 10.0)

    assert H.number_of_bins == bin_edges.shape[0] - 1
    assert counts.shape[0] == H.number_of_bins

    # A later call reuses number_of_bins and has to keep the same number of bins.
    _, counts = H(x)
    assert counts.shape[0] == bin_edges.shape[0] - 1
