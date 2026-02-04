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

from typing import List

import numpy as np
import torch


def fix_random_seeds(seed=0):
    """Fix random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_random_input(input_res, dim):
    """Create random input for testing"""
    return torch.randn(1, dim, *input_res)


def compare_quantiles(
    t: torch.Tensor, ref: torch.Tensor, quantiles: List[float], tolerances: List[float]
):
    """Utility function which compares a tensor against a reference based on tolanceres
    on desired quantiles. Comparing different algorithms in FP32, and especially FP16
    and BF16, is often hard through e.g. ``torch.allclose`` as absolute differences
    in a element-to-element comparison fail when some outliers are present. It sometimes
    is better to compare quantiles with different tolerances allowing some outliers while
    enforcing stricter absolute tolerances for most of the elements.

    Parameters
    ----------
    t : torch.Tensor
        tensor to be compared against reference
    ref : torch.Tensor
        tensor acting as a reference
    quantiles : List[float]
        list of floats defining quantiles of interest, e.g. [0.25, 0.5, 0.75]
        indicates a comparison of the 25%, the 50%, and the 75% quantile.
    tolerances : List[float]
        list of floats indicating the absolute corresponding tolerances for
        all individual quantiles passed into this function
    """
    assert len(quantiles) == len(tolerances)
    diff = torch.abs(ref.float() - t.float()).contiguous().view(-1)
    for i, q in enumerate(quantiles):
        diff_q = torch.quantile(diff, q=q, interpolation="midpoint").item()
        tol = tolerances[i]
        msg = f"For the quantile q={q}, expected a numerical difference of at most {tol}, but observed {diff_q}"
        assert diff_q < tol, msg
