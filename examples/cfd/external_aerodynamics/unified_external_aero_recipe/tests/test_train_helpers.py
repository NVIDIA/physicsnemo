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

"""Unit tests for `src/train.py`'s private TensorDict-aware helpers and for `src/output_normalize.py`.

``TensorDict`` is not a ``dict`` subclass, so the bare
``isinstance(obj, dict)`` branches in the recipe's recursive helpers
must be paired with explicit ``isinstance(obj, TensorDict)`` branches
for TD inputs to be walked at all. These tests pin that explicit
handling for:

- :func:`train._walk_batch_for_logging`: must yield ``(name, tensor)``
  pairs from TensorDict leaves -- including correctly producing dotted
  paths for nested TDs via ``TD.flatten_keys('.')``.
- :func:`output_normalize.normalize_output_to_tensordict`: routes a
  model output (``Mesh`` or ``(B, N, C)`` tensor) to a per-target
  TensorDict, with clear error messages on shape / channel-count
  mismatches.
- :func:`train._reduce_and_average`: averages rank-local loss / metric
  sums over the global sample count (used per step and per epoch); its
  single-process path must equal plain ``total_loss / n`` + per-leaf
  ``sum / n`` averaging.

(The analogous tests for the shared, tensorboard-free
:func:`utils.recursive_to_device` live in ``test_utils.py``, outside
this module's tensorboard skip guard.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
### `output_normalize` itself is tensorboard-free, so we import it
### directly (no skip).
pytest.importorskip("tensorboard")

from loss import LossCalculator  # noqa: E402
from metrics import MetricCalculator  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from output_normalize import normalize_output_to_tensordict  # noqa: E402
from train import (  # noqa: E402  -- after the skip guard
    _reduce_and_average,
    _run_epoch,
    _skip_nan,
    _walk_batch_for_logging,
    forward_pass,
)

from physicsnemo.mesh import Mesh  # noqa: E402  -- after the importorskip guard

### ---------------------------------------------------------------------------
### forward_pass: the batch's target_measure reaches the loss and the metrics
### ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measure_weighted,expected_mse,expected_mae", [(True, 1.0, 0.5), (False, 2.0, 1.0)]
)
def test_forward_pass_weights_loss_and_metrics_by_target_measure(
    measure_weighted, expected_mse, expected_mae
):
    """The erroneous point holds a quarter of the measure: weighted MSE is 1 and
    MAE 0.5, against 2 and 1 per point."""
    predictions = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    target_config = {"pressure": "scalar"}
    batch = {
        "forward_kwargs": {"input": predictions},
        "targets": TensorDict({"pressure": torch.ones(1, 2)}, batch_size=[1, 2]),
        "target_measure": torch.tensor([[3.0, 1.0]]),
    }
    loss, _, metrics = forward_pass(
        batch,
        torch.nn.Identity(),
        "float32",
        LossCalculator(target_config, "mse", measure_weighted=measure_weighted),
        MetricCalculator(
            target_config, metrics=["mae"], measure_weighted=measure_weighted
        ),
        output_type="tensors",
        target_config=target_config,
    )
    assert loss.item() == pytest.approx(expected_mse)
    assert metrics["pressure_mae"].item() == pytest.approx(expected_mae)


### ---------------------------------------------------------------------------
### _walk_batch_for_logging
### ---------------------------------------------------------------------------


class TestWalkBatchForLogging:
    """Tests for `_walk_batch_for_logging`."""

    def test_yields_from_tensordict_leaves(self):
        """Bare TD input yields one entry per leaf with the leaf path."""
        td = TensorDict(
            {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
            batch_size=[5],
        )

        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"pressure", "wss"}
        assert items["pressure"].shape == torch.Size([5])
        assert items["wss"].shape == torch.Size([5, 3])

    def test_dict_containing_tensordict_yields_dotted_keys(self):
        """Nested dict -> TD -> leaves: keys come back dot-joined."""
        batch = {
            "targets": TensorDict(
                {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
                batch_size=[5],
            ),
        }

        items = dict(_walk_batch_for_logging(batch))
        ### Without the TD branch in the walker, neither `targets.pressure`
        ### nor `targets.wss` would appear in the output.
        assert set(items) == {"targets.pressure", "targets.wss"}
        assert items["targets.pressure"].shape == torch.Size([5])

    def test_walk_handles_nested_tensordict_via_flatten_keys(self):
        """A TD nested under another TD: ``flatten_keys`` produces dotted paths.

        This exercises the idiomatic-TD path: ``flatten_keys('.')`` on a
        nested TD returns a flat TD whose keys are dotted leaf paths.
        Without that delegation, a manual ``.items()`` walk would still
        work for flat TDs but would silently mishandle nested ones.
        """
        td = TensorDict(
            {
                "scalar": torch.zeros(3),
                "nested": TensorDict({"x": torch.zeros(3)}, batch_size=[3]),
            },
            batch_size=[3],
        )
        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"scalar", "nested.x"}
        ### And under a plain dict prefix, paths cascade correctly:
        items_with_prefix = dict(_walk_batch_for_logging({"targets": td}))
        assert set(items_with_prefix) == {"targets.scalar", "targets.nested.x"}


### ---------------------------------------------------------------------------
### normalize_output_to_tensordict
### ---------------------------------------------------------------------------


class TestNormalizeOutputToTensordict:
    """Tests for `normalize_output_to_tensordict`."""

    def test_tensors_output_three_dim_splits_correctly(self):
        """Standard (B, N, total_C) output splits into per-field leaves."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        out = torch.randn(1, 50, 4)  # 1 scalar + 1 vector(3) = 4 channels
        td = normalize_output_to_tensordict(out, target_config, "tensors")
        assert tuple(td["pressure"].shape) == (1, 50)  # squeezed scalar
        assert tuple(td["wss"].shape) == (1, 50, 3)
        assert td.batch_size == torch.Size([1, 50])

    def test_tensors_output_two_dim_raises_clearly(self):
        """Two-D output (missing channel dim) raises a clear shape error.

        A ``(B, N)`` output for a single-scalar target is a config bug:
        without the explicit ``ndim < 3`` guard the per-element axis ``N``
        gets compared to the expected channel count ``C``, yielding a
        confusing "channel dim ``N`` does not match expected ``1``" error.
        The guard surfaces the actual problem (missing trailing channel
        dimension) directly.
        """
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50)
        with pytest.raises(ValueError, match=r"expects a \(B, N, C\) tensor"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_tensors_output_channel_mismatch_still_raises(self):
        """Three-D output with wrong channel count still raises the channel error."""
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50, 3)  # expected 1 channel
        with pytest.raises(ValueError, match="does not match the expected"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_mesh_output_extracts_target_fields(self):
        """Mesh output: ``point_data.select(*target_config)`` keeps batch_size [N]."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        mesh = Mesh(
            points=torch.randn(7, 3),
            point_data={
                "pressure": torch.randn(7),
                "wss": torch.randn(7, 3),
                ### A non-target field that must NOT appear in the result.
                "extra": torch.randn(7),
            },
        )
        td = normalize_output_to_tensordict(mesh, target_config, "mesh")
        assert set(td.keys()) == {"pressure", "wss"}
        assert td.batch_size == torch.Size([7])

    def test_mesh_output_missing_target_raises(self):
        """Missing target field on a Mesh output is reported clearly."""
        target_config = {"pressure": "scalar"}
        mesh = Mesh(points=torch.randn(7, 3), point_data={"other": torch.randn(7)})
        with pytest.raises(KeyError, match="missing target fields"):
            normalize_output_to_tensordict(mesh, target_config, "mesh")


### ---------------------------------------------------------------------------
### _reduce_and_average
### ---------------------------------------------------------------------------


class TestReduceAndAverage:
    """Tests for `_reduce_and_average` (single-process path).

    The distributed branch is gated on an initialized process group with
    ``world_size > 1``; with no group initialized these tests exercise the
    pure-local path, which must stay equivalent to the previous
    ``total_loss / n`` + per-leaf ``sum / n`` averaging it replaced. The
    collective branch mirrors the already-shipped ``infer._allreduce_sums``
    and is validated by inspection.
    """

    @staticmethod
    def _epoch_sums() -> tuple[TensorDict, TensorDict]:
        """A representative pair of 0-D (epoch-accumulated) sum TensorDicts."""
        losses_td = TensorDict(
            {"pressure": torch.tensor(6.0), "wss": torch.tensor(9.0)},
        )
        metrics_td = TensorDict(
            {"pressure_l2": torch.tensor(3.0), "wss_mae": torch.tensor(12.0)},
        )
        return losses_td, metrics_td

    def test_single_process_divides_sums_by_local_count(self):
        """No process group: global average == local sum / n_local.

        ``loss_sum`` is passed as a 0-D tensor (matching the on-device epoch
        accumulator); the reducer returns Python floats.
        """
        losses_td, metrics_td = self._epoch_sums()
        avg_loss, avg_losses, avg_metrics = _reduce_and_average(
            torch.tensor(15.0), losses_td, metrics_td, 3, device="cpu"
        )
        assert avg_loss == pytest.approx(5.0)
        assert avg_losses == pytest.approx({"pressure": 2.0, "wss": 3.0})
        assert avg_metrics == pytest.approx({"pressure_l2": 1.0, "wss_mae": 4.0})

    def test_none_sentinel_returns_loss_only(self):
        """The "no steps seeded" sentinel (either TD ``None``) yields (loss / n, {}, {})."""
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(8.0), None, None, 2, device="cpu"
        )
        assert loss == pytest.approx(4.0)
        assert losses == {} and metrics == {}
        ### A single ``None`` is enough to trip the sentinel.
        losses_td, _ = self._epoch_sums()
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(8.0), losses_td, None, 2, device="cpu"
        )
        assert loss == pytest.approx(4.0)
        assert losses == {} and metrics == {}

    def test_metric_counts_skip_nan_metrics(self):
        """With counts, a NaN metric (no finite target) is left out of its mean."""
        steps = [
            TensorDict({"a_l2": torch.tensor(1.0), "b_l2": torch.tensor(b)})
            for b in (2.0, float("nan"), 4.0)
        ]
        parts = [_skip_nan(m) for m in steps]
        sums = parts[0][0] + parts[1][0] + parts[2][0]
        counts = parts[0][1] + parts[1][1] + parts[2][1]
        _, _, means = _reduce_and_average(
            torch.tensor(0.0), TensorDict(), sums, 3, device="cpu", metric_counts=counts
        )
        assert means == pytest.approx({"a_l2": 1.0, "b_l2": 3.0})
        ### One step whose field has no finite target: NaN, not 0.
        _, _, step = _reduce_and_average(
            torch.tensor(0.0),
            TensorDict(),
            parts[1][0],
            1,
            device="cpu",
            metric_counts=parts[1][1],
        )
        assert step["a_l2"] == 1.0 and step["b_l2"] != step["b_l2"]

    def test_zero_local_count_avoids_zero_division(self):
        """``n_local == 0`` (a step-less epoch) divides by 1, not 0."""
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(7.0), None, None, 0, device="cpu"
        )
        assert loss == pytest.approx(7.0)
        assert losses == {} and metrics == {}

    def test_val_epoch_skips_a_field_without_finite_targets(self):
        """A batch whose field is all NaN leaves that field's epoch metric alone."""
        targets = {"a": "scalar", "b": "scalar"}

        def batch(b: float) -> dict:
            values = {"a": torch.ones(1, 4), "b": torch.full((1, 4), b)}
            return {
                "forward_kwargs": {"input": torch.zeros(1, 4, 2)},
                "targets": TensorDict(values, batch_size=[1, 4]),
            }

        _, averages = _run_epoch(
            [batch(2.0), batch(float("nan"))],
            torch.nn.Identity(),
            LossCalculator(targets, mask_nonfinite_targets=True),
            MetricCalculator(targets, metrics=["mae"], mask_nonfinite_targets=True),
            SimpleNamespace(info=lambda *_: None),
            0,
            OmegaConf.create({"precision": "float32", "profile": False}),
            SimpleNamespace(rank=0, device="cpu"),
            mode="val",
            output_type="tensors",
            target_config=targets,
        )
        assert averages["b_mae"] == pytest.approx(2.0)
        assert averages["a_mae"] == pytest.approx(1.0)
