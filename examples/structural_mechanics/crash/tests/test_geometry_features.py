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

"""One-shot feature method preserves predictions and ordinary forward hooks."""

import os
import sys

import pytest
import torch

# Match the sibling rollout test's import path for the example modules.
THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipe import SimSample  # noqa: E402
from rollout import GeoTransolverOneShot  # noqa: E402


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_oneshot_geometry_features(device):
    """The separate method exposes features with unchanged rollout values."""
    torch.manual_seed(11)
    model = (
        GeoTransolverOneShot(
            num_time_steps=3,
            functional_dim=3,
            out_dim=6,
            geometry_dim=3,
            n_layers=1,
            n_hidden=16,
            n_head=4,
            slice_num=4,
            use_te=False,
        )
        .to(device)
        .eval()
    )
    sample = SimSample(
        node_features={"coords": torch.randn(11, 3, device=device)},
        node_target=torch.zeros(11, 2, 3, device=device),
    )
    observed = []
    hook = model.register_forward_hook(
        lambda module, args, output: observed.append(output)
    )
    with torch.no_grad():
        ordinary = model(sample, {})
        prediction, tokens = model.forward_with_geometry_features(sample, {})
        expected_tokens = model.context_builder.geometry_tokenizer(
            sample.node_features["coords"].unsqueeze(0)
        )
    hook.remove()
    assert len(observed) == 1
    assert isinstance(observed[0], torch.Tensor)
    assert prediction.shape == (11, 2, 3)
    torch.testing.assert_close(prediction, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    assert tokens.mean(dim=(1, 2)).shape == (1, 4)
    assert not tokens.requires_grad
