# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: S101
import models
from utils.checkpointing import Checkpoint

from physicsnemo.models.healda import ModelConfigV1


def _assert_state_dict_equal(d1, d2):
    assert set(d1) == set(d2)
    for k in d1:
        assert d1[k].equal(d2[k])


def test_checkpointing(tmp_path):
    config = ModelConfigV1(
        architecture="dit-test",
    )
    model = models.get_model(config)

    state_dict = model.state_dict()
    with Checkpoint(tmp_path / "test.checkpoint", "w") as checkpoint:
        checkpoint.write_model(model)
        checkpoint.write_model_config(config)

    with Checkpoint(tmp_path / "test.checkpoint", "r") as checkpoint:
        model = checkpoint.read_model()
        assert config == checkpoint.read_model_config()
        _assert_state_dict_equal(model.state_dict(), state_dict)
