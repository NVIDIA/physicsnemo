# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import random

import pytest
import torch

from physicsnemo.models.layers import WeightFactLinear


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_weight_fact(device):
    """Test weight fact"""

    in_features = random.randint(1, 8)
    out_features = random.randint(1, 8)
    # Construct FC model
    wfact = WeightFactLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, in_features).to(device)
    outvar = wfact(invar)
    assert outvar.shape == (bsize, out_features)
    assert wfact.g.shape == (out_features, 1)
    assert wfact.v.shape == (out_features, in_features)
    assert (
        wfact.extra_repr()
        == f"in_features={in_features}, out_features={out_features}, bias={True}, mean = {1.0}, stddev = {0.1}"
    )
