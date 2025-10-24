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

import torch
from torch import Tensor

from physicsnemo.utils.graphcast.graph_utils import deg2rad


def normalized_grid_cell_area(lat: Tensor, unit="deg") -> Tensor:
    """Normalized area of the latitude-longitude grid cell"""
    if unit == "deg":
        lat = deg2rad(lat)
    area = torch.abs(torch.cos(lat))
    return area / torch.mean(area)
