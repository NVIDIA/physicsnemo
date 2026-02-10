# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from physicsnemo.experimental.models.healda import ScatterAggregator
from test import common


def test_scatter_aggregator_forward(device):
    """Test ScatterAggregator forward pass."""
    torch.manual_seed(0)

    in_dim = 8
    out_dim = 16
    nchannel = 3
    nplatform = 2
    npix = 50

    model = ScatterAggregator(
        in_dim=in_dim,
        out_dim=out_dim,
        nchannel=nchannel,
        nplatform=nplatform,
        npix=npix,
    ).to(device)
    model.eval()

    n_obs = 100
    nbatch = 2
    nbuckets = nchannel * nplatform
    obs_features = torch.randn(n_obs, in_dim).to(device)
    batch_idx = torch.randint(0, nbatch, (n_obs,)).to(device)
    pix = torch.randint(0, npix, (n_obs,)).to(device)
    bucket_id = torch.randint(0, nbuckets, (n_obs,)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (obs_features, batch_idx, pix, bucket_id, nbatch),
        file_name="models/healda/data/scatter_aggregator_output.pth",
        atol=1e-3,
    )


def test_scatter_aggregator_empty_cells(device):
    """Test ScatterAggregator handles sparse observations."""
    torch.manual_seed(0)

    model = ScatterAggregator(in_dim=4, out_dim=8, nchannel=2, nplatform=2, npix=10).to(
        device
    )

    n_obs = 1
    obs_features = torch.randn(n_obs, 4, device=device)
    batch_idx = torch.zeros(n_obs, dtype=torch.long, device=device)
    pix = torch.arange(n_obs, device=device)
    bucket_id = torch.zeros(n_obs, dtype=torch.long, device=device)

    output = model(obs_features, batch_idx, pix, bucket_id, nbatch=1)

    assert output.shape == (1, 10, 8)
    assert torch.isfinite(output).all()
