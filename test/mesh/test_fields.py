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

from physicsnemo.mesh import FieldLayout, ScalarVectorFields
from physicsnemo.mesh.fields import (
    flatten_rank_spec,
    rank_counts,
    ranks_from_tensordict,
    validate_data_contains_ranks,
    validate_rank_spec,
)


def _mixed_fields(n: int = 4) -> TensorDict:
    return TensorDict(
        {
            # Deliberately not in lexicographic order.
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

def test_validate_rank_spec_and_data_reports_schema_errors():
    validate_rank_spec({"scalar": 0, "tensor": 2})
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        validate_rank_spec({"tensor": 2}, allowed_ranks=(0, 1))
    with pytest.raises(TypeError, match="must be an integer"):
        validate_rank_spec({"bad": True})
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_rank_spec({"bad": -1})

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


def test_field_layout_pack_is_deterministic_and_round_trips():
    ranks = {
        "wind": 1,
        "state": {"temperature": 0},
        "pressure": 0,
        "displacement": 1,
    }
    reordered_ranks = {
        "pressure": 0,
        "displacement": 1,
        "state": {"temperature": 0},
        "wind": 1,
    }
    data = _mixed_fields()
    layout = FieldLayout(ranks, spatial_dim=3)
    reordered_layout = FieldLayout(reordered_ranks, spatial_dim=3)

    assert layout.scalar_names == ("pressure", "state.temperature")
    assert layout.vector_names == ("displacement", "wind")
    assert layout.flat_rank_spec == {
        "displacement": 1,
        "pressure": 0,
        "state.temperature": 0,
        "wind": 1,
    }

    packed = layout.pack(data)
    reordered = reordered_layout.pack(data)
    torch.testing.assert_close(packed.scalars, reordered.scalars)
    torch.testing.assert_close(packed.vectors, reordered.vectors)
    torch.testing.assert_close(
        packed.scalars,
        torch.stack((data["pressure"], data["state", "temperature"]), dim=-1),
    )
    torch.testing.assert_close(
        packed.vectors,
        torch.stack((data["displacement"], data["wind"]), dim=-2),
    )

    unpacked = layout.unpack(packed)
    assert set(unpacked.keys()) == {"displacement", "pressure", "state", "wind"}
    for key in ("pressure", ("state", "temperature"), "displacement", "wind"):
        torch.testing.assert_close(unpacked[key], data[key])


def test_field_layout_preserves_polar_vector_transformation():
    data = _mixed_fields()
    layout = FieldLayout(
        {"pressure": 0, "wind": 1, "displacement": 1},
        spatial_dim=3,
    )
    packed = layout.pack(data)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_data = data.clone()
    rotated_data["wind"] = data["wind"] @ rotation.T
    rotated_data["displacement"] = data["displacement"] @ rotation.T
    rotated = layout.pack(rotated_data)

    torch.testing.assert_close(rotated.scalars, packed.scalars)
    torch.testing.assert_close(rotated.vectors, packed.vectors @ rotation.T)


@pytest.mark.parametrize(
    ("ranks", "expected_scalar_shape", "expected_vector_shape"),
    [
        ({"pressure": 0}, (4, 1), (4, 0, 3)),
        ({"wind": 1}, (4, 0), (4, 1, 3)),
    ],
)
def test_field_layout_supports_scalar_or_vector_only(
    ranks, expected_scalar_shape, expected_vector_shape
):
    layout = FieldLayout(ranks, spatial_dim=3)
    packed = layout.pack(_mixed_fields())
    assert packed.scalars.shape == expected_scalar_shape
    assert packed.vectors.shape == expected_vector_shape
    round_trip = layout.pack(layout.unpack(packed))
    torch.testing.assert_close(round_trip.scalars, packed.scalars)
    torch.testing.assert_close(round_trip.vectors, packed.vectors)


def test_field_layout_rejects_non_vector_rank_and_wrong_vector_dimension():
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        FieldLayout({"stress": 2}, spatial_dim=3)

    layout = FieldLayout({"velocity": 1}, spatial_dim=3)
    wrong_dim = TensorDict({"velocity": torch.zeros(5, 2)}, batch_size=[5])
    with pytest.raises(ValueError, match=r"must have shape \(5, 3\)"):
        layout.pack(wrong_dim)


def test_field_layout_validates_packed_shapes_and_dtype():
    layout = FieldLayout({"pressure": 0, "velocity": 1}, spatial_dim=3)
    with pytest.raises(ValueError, match="vectors must have shape"):
        layout.unpack(
            ScalarVectorFields(
                scalars=torch.empty(5, 1),
                vectors=torch.empty(4, 1, 3),
            )
        )
    with pytest.raises(ValueError, match="must have the same dtype"):
        layout.unpack(
            ScalarVectorFields(
                scalars=torch.empty(5, 1, dtype=torch.float32),
                vectors=torch.empty(5, 1, 3, dtype=torch.float64),
            )
        )
