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
- :func:`train._run_epoch`: per-step records are read back without a host
  sync, and every step is logged once, in order, with the values the
  per-step blocking readback gave.

(The analogous tests for the shared, tensorboard-free
:func:`utils.recursive_to_device` live in ``test_utils.py``, outside
this module's tensorboard skip guard.)
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
### `output_normalize` itself is tensorboard-free, so we import it
### directly (no skip).
pytest.importorskip("tensorboard")

import train  # noqa: E402  -- monkeypatch module globals in focused tests
from output_normalize import normalize_output_to_tensordict  # noqa: E402
from train import (  # noqa: E402  -- after the skip guard
    _reduce_and_average,
    _run_epoch,
    _walk_batch_for_logging,
)

from physicsnemo.mesh import Mesh  # noqa: E402  -- after the importorskip guard

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

    def test_zero_local_count_avoids_zero_division(self):
        """``n_local == 0`` (a step-less epoch) divides by 1, not 0."""
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(7.0), None, None, 0, device="cpu"
        )
        assert loss == pytest.approx(7.0)
        assert losses == {} and metrics == {}


### ---------------------------------------------------------------------------
### _run_epoch: per-step readback
### ---------------------------------------------------------------------------

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


def _run_test_epoch(
    monkeypatch, loader, device, *, mode, profile=False, reference=True
):
    """Run one :func:`train._run_epoch` with a one-weight model and SGD.

    ``train.forward_pass`` is patched to return step-dependent loss and
    metric values; returns ``(jsonl records, expected per-step records)``,
    where the expected values come from the per-step blocking readback
    (``_reduce_and_average`` with ``n_samples=1``) and the scheduler's lr
    right after each step. ``reference=False`` skips that readback, which
    syncs.
    """
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    expected = []

    def forward(batch, *args, **kwargs):
        loss = (model.weight.sum() - batch["k"]) ** 2 / 3
        values = TensorDict({"loss/test": loss.detach() * 7})
        metrics = TensorDict({"test_l2": loss.detach().sqrt()})
        if reference:
            expected.append(
                _reduce_and_average(loss.detach(), values, metrics, 1, device=device)
            )
        return loss, values, metrics

    monkeypatch.setattr(train, "forward_pass", forward)
    records = []
    cfg = OmegaConf.create(
        {
            "precision": "float32",
            "profile": profile,
            "training": {"scheduler_update_mode": "step"},
            "logging": {"log_every_n_steps": 4},
        }
    )
    _run_epoch(
        loader,
        model,
        None,
        None,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        0,
        cfg,
        SimpleNamespace(rank=0, world_size=1, device=torch.device(device)),
        mode=mode,
        output_type="tensors",
        target_config={"pressure": "scalar"},
        optimizer=optimizer,
        scheduler=scheduler,
        log_jsonl=records.append,
    )
    return records, expected


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["train", "val"])
def test_every_step_logged_once_in_order_with_blocking_values(
    monkeypatch, device, mode
):
    """Each of 10 steps (log period 4, so the last period of 2 is drained at
    the epoch's end) is logged once, in order, with the per-step blocking
    readback's loss, losses and metrics, bitwise (single process, so the
    all-reduce is the identity on both paths)."""
    loader = [{"k": float(k)} for k in range(10)]
    records, expected = _run_test_epoch(monkeypatch, loader, device, mode=mode)
    steps = [r for r in records if r["phase"] == f"{mode}_step"]
    key = "global_step" if mode == "train" else "val_step"
    assert [r[key] for r in steps] == list(range(10))
    for r, (loss, losses, metrics) in zip(steps, expected, strict=True):
        assert r["loss"] == loss
        assert {k: r[k] for k in losses} == losses
        assert {k: r[k] for k in metrics} == metrics
    assert records[-1]["phase"] == f"{mode}_summary"


def test_lr_is_read_at_its_own_step(monkeypatch):
    """The lagged TensorBoard ``iteration/lr`` of step i is the lr right after
    step i's scheduler step, not the lr when the record is logged."""
    scalars = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, global_step: scalars.append(
            (tag, value, global_step)
        )
    )
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    def forward(batch, *args, **kwargs):
        loss = model.weight.sum()
        values = TensorDict({"loss/test": loss.detach()})
        return loss, values, values.clone()

    monkeypatch.setattr(train, "forward_pass", forward)
    _run_epoch(
        [{}] * 7,
        model,
        None,
        None,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        0,
        OmegaConf.create(
            {
                "precision": "float32",
                "profile": False,
                "training": {"scheduler_update_mode": "step"},
                "logging": {"log_every_n_steps": 3},
            }
        ),
        SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu")),
        mode="train",
        output_type="tensors",
        target_config={"pressure": "scalar"},
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
    )
    lrs = [(step, value) for tag, value, step in scalars if tag == "iteration/lr"]
    assert lrs == [(i, 0.1 * 0.5 ** (i + 1)) for i in range(7)]


def test_profile_stop_drains_its_records(monkeypatch):
    """Under ``cfg.profile`` the loop stops after step ``_PROFILE_MAX_STEPS``
    and still logs every step it ran."""
    loader = [{"k": float(k)} for k in range(train._PROFILE_MAX_STEPS + 5)]
    records, _ = _run_test_epoch(monkeypatch, loader, "cpu", mode="train", profile=True)
    steps = [r["global_step"] for r in records if r["phase"] == "train_step"]
    assert steps == list(range(train._PROFILE_MAX_STEPS + 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_steps_do_not_sync_with_the_device(monkeypatch):
    """From the first batch to the loader's end, no step syncs the host with
    the device: torch's sync debug mode raises on any flagged sync, and
    explicit event, stream and device waits (which it does not flag, such as
    TensorDict's blocking D2H) are counted. Only the epoch's end may wait."""
    waits = []
    in_steps = False
    for owner, name in (
        (torch.cuda.Event, "synchronize"),
        (torch.cuda.Stream, "synchronize"),
        (torch.cuda, "synchronize"),
    ):
        original = getattr(owner, name)

        def counted(*args, _original=original, _name=f"{owner}.{name}", **kwargs):
            if in_steps:
                waits.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    class Loader:
        """Twelve batches, with the sync checks on while the loop runs."""

        def __len__(self):
            return 12

        def __iter__(self):
            nonlocal in_steps
            in_steps = True
            torch.cuda.set_sync_debug_mode("error")
            try:
                for k in range(12):
                    yield {"k": float(k)}
            finally:
                torch.cuda.set_sync_debug_mode("default")
                in_steps = False

    records, _ = _run_test_epoch(
        monkeypatch, Loader(), "cuda", mode="train", reference=False
    )
    assert waits == []
    steps = [r["global_step"] for r in records if r["phase"] == "train_step"]
    assert steps == list(range(12))


def test_step_loop_has_no_host_readback():
    """No ``.cpu()``, ``.item()``, ``.tolist()`` or ``.numpy()`` call remains in
    ``_run_epoch`` or ``_start_readback``, and the step loop does not call the
    blocking ``_reduce_and_average``. Host values are read only from landed
    copies (``_collect_landed``) and in the epoch summary."""

    def called(node):
        return {
            call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Attribute, ast.Name))
        }

    readbacks = {"cpu", "item", "tolist", "numpy"}
    for fn in (train._run_epoch, train._start_readback):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert not called(tree) & readbacks, fn.__name__
    tree = ast.parse(textwrap.dedent(inspect.getsource(train._run_epoch)))
    (loop,) = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert "_reduce_and_average" not in called(loop)
