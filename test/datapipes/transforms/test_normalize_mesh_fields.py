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

"""Tests for the NormalizeMeshFields transform and its inverses."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
from physicsnemo.mesh import Mesh


def _surface_mesh_with_fields() -> Mesh:
    """A 2-triangle 3D mesh with a scalar pressure and a vector wss field."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return Mesh(
        points=points,
        cells=cells,
        cell_data={
            ### Two physical-units cell-centered fields: pressure (scalar)
            ### and wall shear stress (3-vector). Values chosen to be far
            ### from the normalizer's mean / std so a forward + inverse
            ### round-trip is a non-trivial check (vs all-zeros, which
            ### would round-trip via zero regardless of correctness).
            "pressure": torch.tensor([1500.0, -200.0], dtype=torch.float32),
            "wss": torch.tensor(
                [[5.0, -1.0, 0.5], [-3.0, 2.0, -0.25]],
                dtype=torch.float32,
            ),
        },
    )


def _stats_for_pressure_and_wss() -> dict[str, dict]:
    """Plausible normalization stats for the two fields above."""
    return {
        "pressure": {
            "type": "scalar",
            "mean": -100.0,
            "std": 800.0,
        },
        "wss": {
            "type": "vector",
            "mean": [0.5, -0.25, 0.1],
            "std": [3.0, 2.0, 0.4],
        },
    }


class TestNormalizeMeshFieldsInverseTd:
    """Round-trip tests for the per-field TensorDict inverse."""

    def test_inverse_td_round_trip_recovers_physical_units(self):
        """``inverse_td(__call__(mesh).cell_data)`` should match the original cell_data."""
        mesh = _surface_mesh_with_fields()
        normalizer = NormalizeMeshFields(
            association="cell_data", fields=_stats_for_pressure_and_wss()
        )
        normalized_mesh = normalizer(mesh)

        ### Inverse the normalized cell_data TensorDict back to physical
        ### units. ``select`` keeps only the keys we care about; this
        ### mirrors how the recipe's `to_physical_units` consumes
        ### per-field predictions rather than a full mesh section.
        roundtripped = normalizer.inverse_td(
            normalized_mesh.cell_data.select("pressure", "wss")
        )

        torch.testing.assert_close(roundtripped["pressure"], mesh.cell_data["pressure"])
        torch.testing.assert_close(roundtripped["wss"], mesh.cell_data["wss"])

    def test_inverse_td_passes_through_unknown_fields(self):
        """Leaves not present in the stats dict must round-trip unchanged."""
        normalizer = NormalizeMeshFields(
            association="cell_data",
            fields={"pressure": _stats_for_pressure_and_wss()["pressure"]},
        )
        td = TensorDict(
            {
                "pressure": torch.tensor([0.5, -1.0]),
                ### Field not in normalizer stats; must be returned untouched.
                "extra_unknown": torch.tensor([42.0, -42.0]),
            },
            batch_size=[2],
        )
        out = normalizer.inverse_td(td)
        torch.testing.assert_close(out["extra_unknown"], td["extra_unknown"])

    def test_inverse_td_does_not_mutate_input(self):
        """Original TensorDict must be untouched (clone semantics)."""
        normalizer = NormalizeMeshFields(
            association="cell_data", fields=_stats_for_pressure_and_wss()
        )
        td = TensorDict(
            {
                "pressure": torch.tensor([0.5, -1.0]),
                "wss": torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
            },
            batch_size=[2],
        )
        td_before = td.clone()
        _ = normalizer.inverse_td(td)
        ### Both leaves must be byte-identical to their pre-call state.
        torch.testing.assert_close(td["pressure"], td_before["pressure"])
        torch.testing.assert_close(td["wss"], td_before["wss"])


class TestNormalizeMeshFieldsInverseConsistency:
    """``inverse_td`` and ``inverse_tensor`` must agree element-wise."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_td_and_tensor_inverses_agree(self, dtype):
        """Concatenating then `inverse_tensor` should match `inverse_td` per-field."""
        normalizer = NormalizeMeshFields(
            association="cell_data", fields=_stats_for_pressure_and_wss()
        )
        target_config = {"pressure": "scalar", "wss": "vector"}

        ### Build a normalized TensorDict directly, then concatenate to
        ### ``(N, C=4)`` in target_config order. C = 1 (pressure) + 3 (wss).
        n = 4
        normalized_td = TensorDict(
            {
                "pressure": torch.randn(n, dtype=dtype),
                "wss": torch.randn(n, 3, dtype=dtype),
            },
            batch_size=[n],
        )
        ### Match the channel ordering inverse_tensor expects: pressure
        ### gets a synthetic trailing axis so cat alongside wss works.
        flat = torch.cat(
            [normalized_td["pressure"].unsqueeze(-1), normalized_td["wss"]], dim=-1
        )

        td_recovered = normalizer.inverse_td(normalized_td)
        flat_recovered = normalizer.inverse_tensor(flat, target_config)

        torch.testing.assert_close(td_recovered["pressure"], flat_recovered[..., 0])
        torch.testing.assert_close(td_recovered["wss"], flat_recovered[..., 1:4])
