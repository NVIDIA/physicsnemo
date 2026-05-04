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

"""Numerical equivalence + field_weights correctness for the new dict-based loss.

The previous (B, N, C)-tensor loss used:

- For scalar fields: ``F.huber_loss(pred[..., spec.start:end].squeeze(-1),
                                     target[..., spec.start:end].squeeze(-1))``.
- For vector fields: per-component sum of mean Huber.
- Total: ``sum(per_field_losses) / total_channels``.

The new dict-based ``LossCalculator`` should produce the same total when
``field_weights`` is None (or all 1.0). This module verifies that and the
correctness of ``field_weights``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from loss import DEFAULT_HUBER_DELTA, LossCalculator
from utils import parse_target_config


def _legacy_total_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_config: dict[str, str],
    n_spatial_dims: int = 3,
) -> torch.Tensor:
    """Reference implementation matching the pre-refactor (B, N, C) loss formula.

    Mirrors the pre-refactor `LossCalculator` exactly:
    - Scalar fields: ``F.huber_loss(pred_field, target_field, reduction='mean')``
      with the channel dim squeezed off (so input is (B, N)).
    - Vector fields: sum across components of per-component mean Huber.
    - Total: sum(per-field) / total_channels.
    """
    specs = parse_target_config(target_config, n_spatial_dims=n_spatial_dims)
    total_channels = sum(s.dim for s in specs)
    total = torch.zeros((), dtype=pred.dtype, device=pred.device)
    for spec in specs:
        p = pred[..., spec.start_index : spec.end_index]
        t = target[..., spec.start_index : spec.end_index]
        if spec.field_type == "scalar":
            total = total + F.huber_loss(
                p.squeeze(-1),
                t.squeeze(-1),
                reduction="mean",
                delta=DEFAULT_HUBER_DELTA,
            )
        else:  # vector: per-component mean, summed
            for i in range(spec.dim):
                total = total + F.huber_loss(
                    p[..., i], t[..., i], reduction="mean", delta=DEFAULT_HUBER_DELTA
                )
    return total / total_channels


def _split_concat_tensor(
    tensor: torch.Tensor,
    target_config: dict[str, str],
    n_spatial_dims: int = 3,
) -> dict[str, torch.Tensor]:
    """Slice a (B, N, C) tensor by FieldSpec into the dict the new loss wants."""
    specs = parse_target_config(target_config, n_spatial_dims=n_spatial_dims)
    out: dict[str, torch.Tensor] = {}
    for spec in specs:
        slice_ = tensor[..., spec.start_index : spec.end_index]
        if spec.field_type == "scalar":
            slice_ = slice_.squeeze(-1)
        out[spec.name] = slice_
    return out


### ---------------------------------------------------------------------------
### Equivalence with legacy formula (no field_weights)
### ---------------------------------------------------------------------------


class TestLegacyEquivalence:
    """Tests for legacy equivalence."""

    @pytest.mark.parametrize(
        "target_config",
        [
            {"pressure": "scalar", "wss": "vector"},
            {"velocity": "vector", "pressure": "scalar", "nut": "scalar"},
            {"phi": "scalar"},
            {"u": "vector"},
            {
                "pressure": "scalar",
                "temperature": "scalar",
                "density": "scalar",
                "velocity": "vector",
                "tau_wall": "vector",
            },
        ],
        ids=["pressure_wss", "vel_p_nut", "scalar_only", "vector_only", "highlift"],
    )
    def test_huber_equivalence(self, target_config):
        ### Pin RNG so the comparison is deterministic.
        """Huber equivalence."""
        torch.manual_seed(123)

        specs = parse_target_config(target_config)
        total_channels = sum(s.dim for s in specs)

        ### Concatenated (1, N, C) tensors -- the old loss interface.
        pred = torch.randn(1, 50, total_channels)
        target = torch.randn(1, 50, total_channels)

        ### Legacy reference total.
        legacy_total = _legacy_total_huber(pred, target, target_config)

        ### New dict-based loss with default (no) weights.
        new_loss = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights=None,
        )
        pred_dict = _split_concat_tensor(pred, target_config)
        target_dict = _split_concat_tensor(target, target_config)
        new_total, new_dict = new_loss(pred_dict, target_dict)

        ### Bit-exact (modulo floating-point reductions in the same order).
        assert torch.allclose(new_total, legacy_total, atol=1e-7, rtol=1e-6), (
            f"new={float(new_total):.10f} legacy={float(legacy_total):.10f}"
        )
        assert "loss/total" in new_dict
        assert torch.allclose(
            new_dict["loss/total"], legacy_total, atol=1e-7, rtol=1e-6
        )

    @pytest.mark.parametrize("loss_type", ["huber", "mse"])
    def test_per_field_keys_match_target_config(self, loss_type):
        """Per field keys match target config."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_dict = {
            "pressure": torch.randn(1, 30),
            "wss": torch.randn(1, 30, 3),
        }
        target_dict = {
            "pressure": torch.randn(1, 30),
            "wss": torch.randn(1, 30, 3),
        }
        lc = LossCalculator(target_config=target_config, loss_type=loss_type)
        _, ldict = lc(pred_dict, target_dict)
        ### Per-field entries plus loss/total.
        assert set(ldict) == {"loss/pressure", "loss/wss", "loss/total"}


### ---------------------------------------------------------------------------
### field_weights correctness
### ---------------------------------------------------------------------------


class TestFieldWeights:
    """Tests for field weights."""

    def test_uniform_weights_match_no_weights(self):
        """field_weights={...: 1.0} is a no-op vs default (None)."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_dict = {
            "pressure": torch.randn(1, 50),
            "wss": torch.randn(1, 50, 3),
        }
        target_dict = {
            "pressure": torch.randn(1, 50),
            "wss": torch.randn(1, 50, 3),
        }
        no_weights = LossCalculator(target_config=target_config, loss_type="huber")
        unit_weights = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights={"pressure": 1.0, "wss": 1.0},
        )
        a, _ = no_weights(pred_dict, target_dict)
        b, _ = unit_weights(pred_dict, target_dict)
        assert torch.allclose(a, b, atol=1e-7)

    def test_single_field_weight_scales_linearly(self):
        """Weighting one field by k scales its per-field contribution by k."""
        torch.manual_seed(0)
        target_config = {"pressure": "scalar", "wss": "vector"}
        pred_dict = {
            "pressure": torch.randn(1, 50),
            "wss": torch.randn(1, 50, 3),
        }
        target_dict = {
            "pressure": torch.randn(1, 50),
            "wss": torch.randn(1, 50, 3),
        }
        baseline = LossCalculator(target_config=target_config, loss_type="huber")
        boosted = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights={"pressure": 1.0, "wss": 100.0},
        )
        _, base_dict = baseline(pred_dict, target_dict)
        _, boost_dict = boosted(pred_dict, target_dict)
        ### Per-field loss values are weighted in the dict (loss_dict shows
        ### the WEIGHTED per-field loss).
        assert torch.allclose(
            boost_dict["loss/wss"], 100.0 * base_dict["loss/wss"], atol=1e-6
        )
        assert torch.allclose(boost_dict["loss/pressure"], base_dict["loss/pressure"])

    def test_total_weighted_matches_explicit_sum(self):
        """Total = sum(weighted per-field losses) / total_channels."""
        torch.manual_seed(42)
        target_config = {"a": "scalar", "b": "vector", "c": "scalar"}
        n_pts = 40
        pred_dict = {
            "a": torch.randn(1, n_pts),
            "b": torch.randn(1, n_pts, 3),
            "c": torch.randn(1, n_pts),
        }
        target_dict = {k: torch.randn_like(v) for k, v in pred_dict.items()}
        weights = {"a": 0.5, "b": 2.0, "c": 1.5}
        lc = LossCalculator(
            target_config=target_config,
            loss_type="huber",
            field_weights=weights,
        )
        total, dct = lc(pred_dict, target_dict)
        expected = (dct["loss/a"] + dct["loss/b"] + dct["loss/c"]) / lc.total_channels
        assert torch.allclose(total, expected, atol=1e-7)

    def test_unknown_field_name_raises(self):
        """Unknown field name raises."""
        with pytest.raises(ValueError, match="references unknown fields"):
            LossCalculator(
                target_config={"pressure": "scalar"},
                loss_type="huber",
                field_weights={"not_a_field": 2.0},
            )


### ---------------------------------------------------------------------------
### Shape-agnostic invariance: same loss for (1, N, C) and (N, C) inputs
### ---------------------------------------------------------------------------


class TestShapeAgnostic:
    """Tests for shape agnostic."""

    def test_with_or_without_batch_dim(self):
        """With or without batch dim."""
        torch.manual_seed(7)
        target_config = {"pressure": "scalar", "wss": "vector"}
        ### Mesh-input style (no batch dim).
        pred_no_batch = {
            "pressure": torch.randn(80),
            "wss": torch.randn(80, 3),
        }
        target_no_batch = {
            "pressure": torch.randn(80),
            "wss": torch.randn(80, 3),
        }
        ### Same data with a leading batch dim of 1.
        pred_with_batch = {k: v.unsqueeze(0) for k, v in pred_no_batch.items()}
        target_with_batch = {k: v.unsqueeze(0) for k, v in target_no_batch.items()}

        lc = LossCalculator(target_config=target_config, loss_type="huber")
        loss_no_batch, _ = lc(pred_no_batch, target_no_batch)
        loss_with_batch, _ = lc(pred_with_batch, target_with_batch)
        assert torch.allclose(loss_no_batch, loss_with_batch, atol=1e-7)
