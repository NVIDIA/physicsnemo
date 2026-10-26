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

"""GMM iteration-limit validation before fitting mutates model state."""

import pytest
import torch

from physicsnemo.experimental.uq.ood._gmm import TorchGMM


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
def test_invalid_iteration_limit_is_rejected_before_initialization(device, max_iter):
    """Invalid iteration budgets raise a clear error without fitting parameters."""
    model = TorchGMM(n_components=1, max_iter=max_iter, device=device)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        model.fit(torch.randn(10, 3, device=device))
    assert model.weights_ is None
    assert model.n_iter_ == 0


def test_one_iteration_is_supported(device):
    """The smallest valid iteration budget still yields a usable fitted model."""
    model = TorchGMM(n_components=1, max_iter=1, device=device)
    reference = torch.randn(10, 3, device=device)
    model.fit(reference)
    assert model.n_iter_ == 1
    assert torch.isfinite(model.score(reference)).all()
