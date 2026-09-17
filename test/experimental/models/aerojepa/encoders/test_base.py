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

"""Tests for the AeroJEPA encoder ABCs."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.encoders.base import (
    BaseContextEncoder,
    BaseTargetEncoder,
)


def test_base_context_encoder_is_abstract():
    """Direct instantiation of BaseContextEncoder is rejected."""
    with pytest.raises(TypeError, match="abstract"):
        BaseContextEncoder()


def test_base_target_encoder_is_abstract():
    """Direct instantiation of BaseTargetEncoder is rejected."""
    with pytest.raises(TypeError, match="abstract"):
        BaseTargetEncoder()


def test_supports_batched_forward_default():
    """``supports_batched_forward`` defaults to ``False`` on both ABCs."""
    assert BaseContextEncoder.supports_batched_forward is False
    assert BaseTargetEncoder.supports_batched_forward is False


def test_context_forward_batched_default_raises():
    """``forward_batched`` raises ``NotImplementedError`` mentioning the subclass."""

    class DummyCtx(BaseContextEncoder):
        def forward(self, *, context_pos, context_feat):
            return None

    d = DummyCtx()
    with pytest.raises(NotImplementedError, match="DummyCtx"):
        d.forward_batched(
            context_pos=torch.zeros(1, 4, 3),
            context_feat=torch.zeros(1, 4, 2),
            context_pos_n=torch.tensor([4]),
        )


def test_target_forward_batched_default_raises():
    """``forward_batched`` on the target ABC raises with the subclass name."""

    class DummyTgt(BaseTargetEncoder):
        def forward(self, *, surface_pos, surface_main_feat, volume_pos, volume_feat):
            return None

    d = DummyTgt()
    with pytest.raises(NotImplementedError, match="DummyTgt"):
        d.forward_batched(
            surface_pos=torch.zeros(1, 4, 3),
            surface_main_feat=torch.zeros(1, 4, 2),
            surface_pos_n=torch.tensor([4]),
            volume_pos=torch.zeros(1, 6, 3),
            volume_feat=torch.zeros(1, 6, 2),
            volume_pos_n=torch.tensor([6]),
        )


def test_context_forward_signature_no_gen_params():
    """``BaseContextEncoder.forward`` no longer takes ``gen_params``."""
    assert "gen_params" not in BaseContextEncoder.forward.__code__.co_varnames


def test_target_forward_signature_no_gen_params_no_context_tokens():
    """``BaseTargetEncoder.forward`` does not take ``gen_params`` or ``context_tokens``."""
    assert "gen_params" not in BaseTargetEncoder.forward.__code__.co_varnames
    assert "context_tokens" not in BaseTargetEncoder.forward.__code__.co_varnames
