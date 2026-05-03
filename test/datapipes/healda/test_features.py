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
"""Tests for observation metadata featurization (standard and extended).

The Triton kernel tests require CUDA and validate that the Triton
implementation matches the reference Python implementation.
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from physicsnemo.experimental.datapipes.healda.transforms import (  # noqa: E402
    obs_features as standard,
)
from physicsnemo.experimental.datapipes.healda.transforms import (  # noqa: E402
    obs_features_ext as extended,
)


def _make_obs_data(n, device, include_lat=False):
    g = torch.Generator(device=device)
    g.manual_seed(42)

    height = torch.rand(n, device=device, generator=g) * 50000
    pressure = torch.rand(n, device=device, generator=g) * 1100
    scan_angle = torch.rand(n, device=device, generator=g) * 100 - 50
    sat_zenith_angle = torch.rand(n, device=device, generator=g) * 120 - 60
    sol_zenith_angle = torch.rand(n, device=device, generator=g) * 160 + 10

    # Conv/sat split: NaN height -> satellite, valid height -> conventional
    is_sat = torch.rand(n, device=device, generator=g) < 0.4
    height[is_sat] = float("nan")
    pressure[is_sat] = float("nan")
    scan_angle[~is_sat] = float("nan")
    sat_zenith_angle[~is_sat] = float("nan")
    sol_zenith_angle[~is_sat] = float("nan")

    data = dict(
        target_time_sec=torch.full(
            (n,), 1_700_000_000, dtype=torch.int64, device=device
        ),
        time=torch.full(
            (n,), 1_700_000_100_000_000_000, dtype=torch.int64, device=device
        ),
        lon=torch.rand(n, device=device, generator=g) * 360 - 180,
        height=height,
        pressure=pressure,
        scan_angle=scan_angle,
        sat_zenith_angle=sat_zenith_angle,
        sol_zenith_angle=sol_zenith_angle,
    )
    if include_lat:
        data["lat"] = torch.rand(n, device=device, generator=g) * 180 - 90
    return data


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel"
)
@pytest.mark.parametrize("n", [0, 1, 137, 10_000])
def test_standard_triton_matches_reference(n):
    device = torch.device("cuda")
    data = _make_obs_data(max(n, 1), device)
    if n == 0:
        data = {k: v[:0] for k, v in data.items()}

    ref = standard._compute_unified_metadata_reference(**data)
    triton_out = standard.compute_unified_metadata(**data)

    assert ref.shape == triton_out.shape == (n, standard.N_FEATURES)
    if n > 0:
        torch.testing.assert_close(ref, triton_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel"
)
@pytest.mark.parametrize("n", [0, 1, 137, 10_000])
def test_extended_triton_matches_reference(n):
    device = torch.device("cuda")
    data = _make_obs_data(max(n, 1), device, include_lat=True)
    if n == 0:
        data = {k: val[:0] for k, val in data.items()}

    ref = extended._compute_unified_metadata_reference(**data)
    triton_out = extended.compute_unified_metadata(**data)

    assert ref.shape == triton_out.shape == (n, extended.N_FEATURES)
    if n > 0:
        torch.testing.assert_close(ref, triton_out, atol=1e-5, rtol=1e-5)
