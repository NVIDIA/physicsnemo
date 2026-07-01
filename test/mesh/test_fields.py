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

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import (
    flatten_rank_spec,
    rank_counts,
    ranks_from_tensordict,
    validate_data_contains_ranks,
)


def _mixed_fields(n: int = 4) -> TensorDict:
    return TensorDict(
        {
            "wind": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "state": {"temperature": torch.arange(n, dtype=torch.float32) + 20},
            "displacement": -torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "pressure": torch.arange(n, dtype=torch.float32) + 100,
            "ignored": torch.ones(n),
        },
        batch_size=[n],
    )


def test_rank_spec_helpers():
    ranks = {"wind": 1, "state": {"temperature": 0}, "pressure": 0}
    assert flatten_rank_spec(ranks) == {
        "wind": 1,
        "state.temperature": 0,
        "pressure": 0,
    }
    assert rank_counts(ranks) == {0: 2, 1: 1}
    assert ranks_from_tensordict(_mixed_fields()) == {
        "wind": 1,
        "state": {"temperature": 0},
        "displacement": 1,
        "pressure": 0,
        "ignored": 0,
    }


def test_validate_data_contains_ranks_reports_schema_errors():
    data = TensorDict({"pressure": torch.zeros(3, 2)}, batch_size=[3])
    with pytest.raises(ValueError) as error:
        validate_data_contains_ranks(
            data=data,
            declared_ranks={"pressure": 0, "velocity": 1},
            source_label="boundary data",
        )
    assert str(error.value) == (
        "boundary data does not contain its declared rank spec:\n"
        "  - missing leaf 'velocity' (declared rank 1)\n"
        "  - rank mismatch for 'pressure': declared 0, got 1"
    )
