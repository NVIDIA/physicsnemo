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


import pytest
import torch

from physicsnemo.metrics.diffusion import calculate_fid_from_inception_stats


def test_fid_calculation():
    mu = torch.Tensor([1.0, 2.0])
    sigma = torch.Tensor([[1.0, 0.5], [0.5, 2.0]])
    mu_ref = torch.Tensor([0.0, 1.0])
    sigma_ref = torch.Tensor([[2.0, 0.3], [0.3, 1.5]])

    fid = calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)
    expected_fid = 2.234758220608337

    assert pytest.approx(fid, abs=1e-4) == expected_fid
