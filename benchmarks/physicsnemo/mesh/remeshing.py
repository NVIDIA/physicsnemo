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

"""ASV benchmarks for CUDA surface remeshing."""

import torch

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import remesh


class RemeshBenchmark:
    """Measure warmed, end-to-end CUDA remeshing.

    Setup performs one untimed call so imports, Warp kernel compilation, and
    allocator initialization do not distort steady-state measurements.
    """

    params = [4, 5, 6, 7, 8, 9]
    param_names = ["subdivisions"]
    number = 1
    repeat = (5, 9, 45.0)
    warmup_time = 0
    timeout = 180

    def setup(self, subdivisions: int) -> None:
        """Create an icosphere with an 8:1 target vertex reduction."""
        if not torch.cuda.is_available():
            raise NotImplementedError("the CUDA remeshing benchmark requires CUDA")

        self.mesh = sphere_icosahedral.load(
            subdivisions=subdivisions,
            device="cuda",
        )
        self.n_clusters = max(3, self.mesh.n_points // 8)
        self.output = remesh(self.mesh, self.n_clusters)
        torch.cuda.synchronize()

    def time_remesh(self, subdivisions: int) -> None:
        """Time complete clustering, projection, and topology reconstruction."""
        self.output = remesh(self.mesh, self.n_clusters)
        torch.cuda.synchronize(self.mesh.points.device)

    def track_output_vertices(self, subdivisions: int) -> int:
        """Track the realized output vertex count after cleanup."""
        return self.output.n_points

    def track_output_triangles(self, subdivisions: int) -> int:
        """Track the realized output triangle count after cleanup."""
        return self.output.n_cells
