# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch

from physicsnemo.nn.functional import radius_search

from utils import Meter

try:
    import warp as wp
except ImportError:
    print(
        """NVIDIA WARP is required for this datapipe. This package is under the
NVIDIA Source Code License (NVSCL). To install use:

pip install warp-lang
"""
    )
    raise SystemExit(1)


def test_warp(
    points: torch.Tensor,
    radius: float,
    min_tries: int,
):
    timer = Meter("Warp")
    for i in range(min_tries):
        with timer:
            _ = radius_search(
                points=points,
                queries=points,
                radius=radius,
            )
            torch.cuda.synchronize()
    return timer.min_time, timer.max_allocated_memory


if __name__ == "__main__":
    wp.init()
    torch.cuda.init()
    torch.manual_seed(42)

    min_tries = 5
    Ns = [100_000, 1_000_000, 2_000_000, 3_000_000]
    radii = [0.02, 0.01, 0.001, 0.0001]
    device = "cuda"

    for N in Ns:
        warp_times = []
        warp_max_mems = []

        points = torch.rand([N, 3]).to(device)

        for radius in radii:
            # try:
            min_time, max_mem = test_warp(points, radius, min_tries)
            warp_times.append(min_time)
            warp_max_mems.append(max_mem)

        print(f"\n\nResults for {N} points")
        # Print table
        print("Radius\tWarp Min Time (sec)\tWarp Max Mem (MB)")
        for i in range(len(radii)):
            print(f"{radii[i]}\t{warp_times[i]}\t{warp_max_mems[i] / 1024 / 1024}")
