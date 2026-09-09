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

"""Tests for VariationalGPHead's constructor contract.

The scalar head predates these tests; what is covered here is the part of its
signature that the field head's tests also pin down, so the two stay in step.
"""

import pytest
import torch

pytest.importorskip("gpytorch", reason="VariationalGPHead requires gpytorch")

from physicsnemo.experimental.uq import VariationalGPHead  # noqa: E402

INPUT_DIM = 16
N_TRAIN = 3200


def test_predict_returns_one_value_per_sample():
    """The scalar head collapses the embedding to a single prediction."""
    head = VariationalGPHead(input_dim=INPUT_DIM, n_inducing=32, n_train=N_TRAIN)
    assert head.predict(torch.randn(4, INPUT_DIM)).mean.shape == (4,)


def test_n_train_is_required():
    """n_train scales the ELBO, so omitting it must not fall back to a guess."""
    with pytest.raises(TypeError):
        VariationalGPHead(input_dim=INPUT_DIM, n_inducing=32)


def test_tuning_knobs_are_keyword_only():
    """Only input_dim is positional, so a config block cannot be mis-ordered."""
    with pytest.raises(TypeError):
        VariationalGPHead(INPUT_DIM, N_TRAIN)


@pytest.mark.parametrize("matern_nu", [0.5, 1.5, 2.5])
def test_matern_order_is_configurable(matern_nu):
    """The kernel's smoothness order is exposed, and reaches the kernel."""
    head = VariationalGPHead(
        input_dim=INPUT_DIM, n_inducing=16, n_train=N_TRAIN, matern_nu=matern_nu
    )
    assert head.gp_layer.covar_module.base_kernel.nu == matern_nu
