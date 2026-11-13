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

"""Unit tests for the shared tensorboard-free helpers in `src/utils.py` / `src/metrics.py`.

These helpers are used by both entry points (`train.py` / `infer.py`),
and -- unlike `train.py` itself -- import without the `tensorboard`
package, so this module carries no skip guard.

``TensorDict`` is not a ``dict`` subclass, so the bare
``isinstance(obj, dict)`` branch in :func:`utils.recursive_to_device`
must be paired with an explicit ``isinstance(obj, TensorDict)`` branch
for TD inputs to be walked at all. The device tests pin that: a freshly
built ``TensorDict(..., batch_size=[N])`` has ``device is None``, and
``td.to("cpu")`` updates ``.device``, so ``result.device == cpu`` fails
if the walker skips the TD branch.
"""

from __future__ import annotations

import json

import pytest
import torch
from metrics import DEFAULT_METRICS, MetricCalculator, resolve_metrics
from omegaconf import OmegaConf
from tensordict import TensorDict
from utils import (
    get_autocast_context,
    make_jsonl_logger,
    recursive_to_device,
    resolve_dict,
)

### ---------------------------------------------------------------------------
### recursive_to_device
### ---------------------------------------------------------------------------


class TestRecursiveToDevice:
    """Tests for `recursive_to_device`."""

    def test_tensordict_input_moves_to_device(self):
        """Bare TD input goes through `.to(device)`."""
        cpu = torch.device("cpu")
        td = TensorDict(
            {"pressure": torch.zeros(4), "wss": torch.zeros(4, 3)},
            batch_size=[4],
        )
        ### Baseline: TD with no explicit device has .device is None.
        assert td.device is None

        result = recursive_to_device(td, cpu)
        assert isinstance(result, TensorDict)
        ### `.to(cpu)` sets `.device`, so a non-None `.device` here is
        ### proof the walker recursed into the TD branch (a skipped TD
        ### would leave `.device` at its initial `None`).
        assert result.device == cpu
        assert result["pressure"].device == cpu
        assert result["wss"].device == cpu
        assert set(result.keys()) == {"pressure", "wss"}

    def test_dict_with_nested_tensordict(self):
        """Plain dict containing a TD: walker recurses into the dict, then
        the TD branch picks up the inner TD."""
        cpu = torch.device("cpu")
        batch = {
            "forward_kwargs": {"x": torch.zeros(2, 3)},
            "targets": TensorDict({"pressure": torch.zeros(4)}, batch_size=[4]),
        }
        assert batch["targets"].device is None

        result = recursive_to_device(batch, cpu)
        assert isinstance(result, dict)
        assert isinstance(result["targets"], TensorDict)
        assert result["targets"].device == cpu
        assert result["forward_kwargs"]["x"].device == cpu


### ---------------------------------------------------------------------------
### JSONL logging
### ---------------------------------------------------------------------------


def test_make_jsonl_logger_writes_timestamped_line(tmp_path):
    """Logger appends one JSON object per call, each stamped with a 'ts' field."""
    path = tmp_path / "metrics.jsonl"
    log = make_jsonl_logger(path)
    log({"phase": "infer_summary", "value": 1.5})
    log({"phase": "infer_step", "step": 0})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["phase"] == "infer_summary"
    assert first["value"] == 1.5
    assert "ts" in first


### ---------------------------------------------------------------------------
### Metrics resolution
### ---------------------------------------------------------------------------


def test_resolve_metrics_default_and_override():
    """A configured metrics list passes through; an absent one -> DEFAULT_METRICS."""
    assert resolve_metrics(OmegaConf.create({"metrics": ["l2"]})) == ["l2"]
    assert resolve_metrics(OmegaConf.create({})) == list(DEFAULT_METRICS)


def test_resolve_metrics_rejects_unknown_name():
    """A typo'd metric name fails fast in resolve_metrics, not deep in MetricCalculator."""
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_metrics(OmegaConf.create({"metrics": ["l2", "rmse"]}))


def test_metric_calculator_expected_keys_match_call():
    """`expected_keys` returns exactly the key set `__call__` produces.

    `infer.py` zero-fills its running sums from `expected_keys` so every
    rank packs the same tensor length into the final all-reduce, even on
    a rank whose sampler shard was empty -- so the two key sets must
    never drift.
    """
    target_config = {"pressure": "scalar", "wss": "vector"}
    calc = MetricCalculator(target_config=target_config, metrics=["l2", "mae"])
    td = TensorDict(
        {"pressure": torch.randn(10), "wss": torch.randn(10, 3)}, batch_size=[10]
    )
    computed = set(calc(td, td.clone()).keys())
    assert set(calc.expected_keys()) == computed


def test_vector_aggregate_metric_is_direction_sensitive():
    """The bare-name vector aggregate must penalize direction errors.

    Adversarial case: ``pred = -target`` has the correct magnitude at
    every point but maximally wrong direction. The aggregate ``wss_*``
    metrics must say so (relative l1/l2 of a sign flip is exactly 2).
    Before 2026-07 the bare-name aggregate was a magnitude comparison,
    so this prediction scored 0 -- the headline vector metric could not
    see direction errors at all. That broken behavior is not retained
    under a separate metric key.
    """
    t = torch.randn(64, 3)
    calc = MetricCalculator({"wss": "vector"})
    out = calc(
        TensorDict({"wss": -t}, batch_size=[]),
        TensorDict({"wss": t}, batch_size=[]),
    )
    assert out["wss_l1"].item() == pytest.approx(2.0, rel=1e-5)
    assert out["wss_l2"].item() == pytest.approx(2.0, rel=1e-5)
    assert out["wss_mae"].item() == pytest.approx(
        (2 * torch.mean(torch.abs(t.flatten()))).item(), rel=1e-5
    )
    assert not any("_mag_" in key for key in out.keys())


def test_vector_aggregate_l2_is_frobenius():
    """The bare-name l2 aggregate is the whole-field relative norm.

    Pins the reduction axes: the field is flattened before the relative
    norm, so the result is ``||pred - target||_F / ||target||_F`` --
    not a mean of per-point ratios, which would blow up wherever the
    target magnitude is near zero (e.g. WSS at stagnation points).
    """
    torch.manual_seed(0)
    t = torch.randn(128, 3)
    p = t + 0.1 * torch.randn(128, 3)
    calc = MetricCalculator({"wss": "vector"}, metrics=["l2"])
    out = calc(
        TensorDict({"wss": p}, batch_size=[]),
        TensorDict({"wss": t}, batch_size=[]),
    )
    expected = (torch.linalg.norm(p - t) / torch.linalg.norm(t)).item()
    assert out["wss_l2"].item() == pytest.approx(expected, rel=1e-5)


def _weighted_metric_reference(p, t, m, metric):
    """Hand-written measure-weighted metric over the last (point) axis."""
    if metric == "mae":
        return (m * (p - t).abs()).sum() / m.sum()
    if metric == "l1":
        return (m * (p - t).abs()).sum() / (m * t.abs()).sum()
    return ((m * (p - t) ** 2).sum() / (m * t**2).sum()).sqrt()


def _measure_case(seed: int = 3):
    torch.manual_seed(seed)
    pred = TensorDict(
        {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)},
        batch_size=[1, 50],
    )
    target = TensorDict(
        {"pressure": torch.randn(1, 50), "wss": torch.randn(1, 50, 3)},
        batch_size=[1, 50],
    )
    return pred, target, torch.rand(1, 50) * 5 + 0.1


@pytest.mark.parametrize("metric", ["mae", "l1", "l2"])
def test_metrics_with_measure_match_hand_computed_weighted_values(metric):
    """Scalar, per-component and vector-aggregate metrics are measure-weighted."""
    pred, target, m = _measure_case()
    calc = MetricCalculator({"pressure": "scalar", "wss": "vector"}, metrics=[metric])
    out = calc(pred, target, m)

    expected = {"pressure": (pred["pressure"], target["pressure"], m)}
    for i, comp in enumerate("xyz"):
        expected[f"wss_{comp}"] = (pred["wss"][..., i], target["wss"][..., i], m)
    m_rep = m.unsqueeze(-1).expand(1, 50, 3)
    expected["wss"] = (pred["wss"], target["wss"], m_rep)
    for name, (p, t, w) in expected.items():
        ref = _weighted_metric_reference(p, t, w, metric)
        torch.testing.assert_close(out[f"{name}_{metric}"], ref, rtol=1e-5, atol=0)


def test_metrics_switched_off_ignore_the_measure_bitwise():
    """``measure_weighted=False`` gives the unweighted metrics bit for bit."""
    pred, target, m = _measure_case()
    calc = MetricCalculator(
        {"pressure": "scalar", "wss": "vector"}, measure_weighted=False
    )
    assert "measure_weighted=False" in repr(calc)
    out, ref = calc(pred, target, m), calc(pred, target)
    for key in ref.keys():
        assert torch.equal(out[key], ref[key])


def test_metrics_with_equal_measures_match_unweighted():
    """Equal measures on every point reproduce the unweighted metrics."""
    pred, target, _ = _measure_case()
    calc = MetricCalculator({"pressure": "scalar", "wss": "vector"})
    out = calc(pred, target, torch.full((1, 50), 3.0))
    ref = calc(pred, target)
    for key in ref.keys():
        torch.testing.assert_close(out[key], ref[key])


### ---------------------------------------------------------------------------
### Autocast precision contract
### ---------------------------------------------------------------------------


def test_get_autocast_context_rejects_float8():
    """fp8 is scoped out of this recipe: it must fail fast, not silently no-op.

    TE fp8 needs every GEMM dimension (the point count, the target out_dim)
    divisible by 16, which this recipe does not pad, so ``precision=float8``
    is rejected at the autocast boundary rather than erroring deep in a TE
    layer (or no-op'ing on a non-TE model).
    """
    with pytest.raises(NotImplementedError, match="float8"):
        get_autocast_context("float8")


def test_get_autocast_context_rejects_unknown_precision():
    """A typo'd precision (e.g. 'bf16') fails fast instead of silently running fp32."""
    with pytest.raises(ValueError, match="bf16"):
        get_autocast_context("bf16")


### ---------------------------------------------------------------------------
### Config helpers
### ---------------------------------------------------------------------------


def test_resolve_dict():
    """A populated key -> a plain dict; an empty or missing key -> None."""
    cfg = OmegaConf.create({"a": {"b": 1}, "empty": {}})
    assert resolve_dict(cfg, "a") == {"b": 1}
    assert resolve_dict(cfg, "empty") is None  # empty -> None
    assert resolve_dict(cfg, "missing") is None
