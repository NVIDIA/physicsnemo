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
import torch_geometric

from physicsnemo.models.dgcnn.dgcnn_compensation import DGCNN

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("knn_cnt", [5, 20])
@pytest.mark.parametrize("sample_pts", [100, 1000, 10000])
def test_dgcnn_forward(device, knn_cnt, sample_pts):
    """Test model forward pass"""
    torch.manual_seed(0)
    # Construct dgcnn model
    model = DGCNN(k=knn_cnt, aggr="max").to(device)

    bsize = 2
    in_pts = torch.randn(bsize, sample_pts, 3).to(device)
    in_pts = in_pts.contiguous()
    edge_index = torch_geometric.nn.knn_graph(in_pts, knn_cnt)
    invar = torch_geometric.data.Data(x=in_pts, edge_index=edge_index)
    assert common.validate_forward_accuracy(
        model,
        (invar,),
        file_name=f"dgcnn_k{knn_cnt}_pts{sample_pts}_output.pth",
        atol=1e-4,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("knn_cnt", [5, 20])
@pytest.mark.parametrize("sample_pts", [100, 1000, 10000])
def test_dgcnn_checkpoint(device, knn_cnt, sample_pts):
    """Test model checkpoint save/load"""
    torch.manual_seed(0)
    # Construct dgcnn model
    model_1 = DGCNN(k=knn_cnt, aggr="max").to(device)

    model_2 = DGCNN(k=knn_cnt, aggr="max").to(device)

    bsize = random.randint(1, 2)
    in_pts = torch.randn(bsize, sample_pts, 3).to(device)
    edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(in_pts), knn_cnt)
    invar = torch_geometric.data.Data(x=in_pts, edge_index=edge_index)

    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("knn_cnt", [5, 20])
@pytest.mark.parametrize("sample_pts", [100, 1000, 10000])
def test_dgcnn_optimizations(device, knn_cnt, sample_pts):
    """Test model optimizations"""

    def setup_model():
        "Sets up fresh model for each optimization test"
        # Construct dgcnn model
        model = DGCNN(k=knn_cnt, aggr="max").to(device)

        bsize = 2
        in_pts = torch.randn(bsize, sample_pts, 3).to(device)
        edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(in_pts), knn_cnt)
        invar = torch_geometric.data.Data(x=in_pts, edge_index=edge_index)

        return model, invar

    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
