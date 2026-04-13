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

"""
Example: Optimized Physics-AI Inference with TensorRT and Warp
--------------------------------------------------------------
This example demonstrates a hybrid inference pipeline that leverages:
1. NVIDIA Warp for high-performance geometric processing (neighbor search).
2. TensorRT (via Torch-TensorRT) for accelerated neural network execution.

The model is a simplified point-cloud processor that finds neighbors using Warp
and then processes the local geometry using a TensorRT-optimized MLP.
"""

import time
import torch
import torch.nn as nn
import warp as wp
import numpy as np
from physicsnemo.utils.inference import compile_to_trt, is_trt_available
from physicsnemo.models.figconvnet.warp_neighbor_search import radius_search_warp

# 1. Define a Simple Neural Network Module
class GeometryProcessor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def run_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available. This example requires a GPU for Warp and TensorRT.")
        return

    # Initialize Warp
    wp.init()

    # 2. Setup Data
    num_points = 10000
    points = torch.randn(num_points, 3, device=device)
    queries = torch.randn(1000, 3, device=device)
    radius = 0.1

    # 3. Geometric Processing with Warp (Neighbor Search)
    print(f"Finding neighbors for {queries.shape[0]} queries in {points.shape[0]} points using Warp...")
    start_time = time.time()
    # neighbor_index: [total_neighbors], neighbor_dist: [total_neighbors], neighbor_offset: [num_queries + 1]
    neighbor_index, neighbor_dist, neighbor_offset = radius_search_warp(
        points, queries, radius, device=device.type
    )
    wp_time = time.time() - start_time
    print(f"Warp neighbor search took: {wp_time:.4f}s")

    # 4. Neural Network Optimization with TensorRT
    model = GeometryProcessor(input_dim=3).to(device).eval()
    
    # Example input for TensorRT compilation signature
    # We'll process one neighbor at a time or in batch. 
    # For simplicity, let's assume we process the relative coordinates of all neighbors.
    example_input = torch.randn(1, 3, device=device) 

    if is_trt_available():
        print("Compiling GeometryProcessor to TensorRT...")
        try:
            trt_model = compile_to_trt(
                model, 
                input_signature=[example_input],
                enabled_precisions={torch.float32}
            )
            process_func = trt_model
        except Exception as e:
            print(f"TensorRT compilation failed, falling back to eager mode: {e}")
            process_func = model
    else:
        print("Torch-TensorRT not found, using eager PyTorch.")
        process_func = model

    # 5. Hybrid Inference Loop
    print("Running hybrid inference...")
    # For the sake of the example, we just process the first query's neighbors
    q_idx = 0
    start_idx = neighbor_offset[q_idx].item()
    end_idx = neighbor_offset[q_idx+1].item()
    
    neighbor_indices = neighbor_index[start_idx:end_idx]
    if len(neighbor_indices) > 0:
        neighbor_coords = points[neighbor_indices]
        relative_coords = neighbor_coords - queries[q_idx]
        
        # Inference using TensorRT-optimized module
        with torch.no_grad():
            output = process_func(relative_coords)
        
        print(f"Query {q_idx} has {len(neighbor_indices)} neighbors.")
        print(f"Output features shape: {output.shape}")
    else:
        print(f"Query {q_idx} has no neighbors within radius {radius}.")

    print("Example completed successfully!")

if __name__ == "__main__":
    run_example()
