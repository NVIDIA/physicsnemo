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

from physicsnemo.nn.functional.neighbors.radius_search._compact_cell_points_impl import (
    _load_compact_cell_points_kernel_source,
)


def test_compact_cell_kernel_source_resource():
    source = _load_compact_cell_points_kernel_source()

    assert source.startswith("// SPDX-FileCopyrightText:")
    for kernel_name in (
        "count_point_cells",
        "scatter_point_bins",
        "radius_search",
    ):
        assert f"void {kernel_name}(" in source
