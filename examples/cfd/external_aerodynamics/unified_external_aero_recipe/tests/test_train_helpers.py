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
- :func:`train._fail_together`: a loading, forward, or divergent-loss
  failure on one rank stops every rank, before DDP forward and before
  backward.
- :func:`train._finish_epoch`: the epoch-mode scheduler steps before the
  checkpoint is written and the final epoch is always saved, so a resumed
  run matches a continuous one.

(The analogous tests for the shared, tensorboard-free
:func:`utils.recursive_to_device` live in ``test_utils.py``, outside
this module's tensorboard skip guard.)
"""

from __future__ import annotations

from datetime import timedelta
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
    _fail_together,
    _finish_epoch,
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
### Rank-failure rendezvous and divergence guard
### ---------------------------------------------------------------------------


def _single_rank() -> SimpleNamespace:
    """Return a single-process stand-in for ``DistributedManager``."""
    return SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))


def _run_test_epoch(
    loader,
    model,
    *,
    mode,
    dist_manager,
    threshold=None,
    loss_calculator=None,
    metric_calculator=None,
):
    """Run one :func:`train._run_epoch` on a one-field tensor model with SGD."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = OmegaConf.create(
        {
            "precision": "float32",
            "profile": False,
            "training": {
                "scheduler_update_mode": "epoch",
                "divergence_loss_threshold": threshold,
            },
        }
    )
    return _run_epoch(
        loader,
        model,
        loss_calculator,
        metric_calculator,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        0,
        cfg,
        dist_manager,
        mode=mode,
        output_type="tensors",
        target_config={"pressure": "scalar"},
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )


def test_fail_together_single_process():
    """A healthy step is a no-op; a failure re-raises the original error with a note."""
    where = "train epoch 5, step 9 (data loading)"
    _fail_together(None, _single_rank(), where)

    error = ValueError("corrupt sample")
    with pytest.raises(ValueError) as exc:
        _fail_together(error, _single_rank(), where)
    assert exc.value is error
    assert exc.value.__notes__ == [f"Raised during {where}."]


@pytest.mark.parametrize(
    ("mode", "loss_value", "threshold", "raises"),
    [
        ("train", 1000.0, 1000.0, False),
        ("train", 1000.1, 1000.0, True),
        ("train", float("nan"), 1000.0, True),
        ("train", float("inf"), float("inf"), True),
        ("train", 5.0, float("inf"), False),
        ("val", float("nan"), 1000.0, False),
    ],
)
def test_divergence_guard_stops_before_backward(
    monkeypatch, mode, loss_value, threshold, raises
):
    """Bad training losses raise before backward; validation losses are unchecked."""
    model = torch.nn.Linear(1, 1, bias=False)

    def forward(*args, **kwargs):
        loss = model.weight.sum() * 0.0 + loss_value
        values = TensorDict({"loss/test": loss.detach()})
        return loss, values, values.clone()

    monkeypatch.setattr(train, "forward_pass", forward)
    if not raises:
        _run_test_epoch(
            [{}], model, mode=mode, dist_manager=_single_rank(), threshold=threshold
        )
        return
    with pytest.raises(RuntimeError, match="divergence_loss_threshold"):
        _run_test_epoch(
            [{}], model, mode=mode, dist_manager=_single_rank(), threshold=threshold
        )
    assert model.weight.grad is None


class _TwoStepLoader:
    """Yield one healthy batch, then optionally fail on a chosen rank."""

    def __init__(self, batch, *, fail):
        """Store the common batch and this rank's failure switch."""
        self.batch = batch
        self.fail = fail

    def __len__(self):
        """Report equal step counts on both ranks."""
        return 2

    def __iter__(self):
        """Fail after one complete step, when DDP may rebuild its buckets."""
        yield self.batch
        if self.fail:
            raise ValueError("corrupt second sample")
        yield self.batch


class _FlakyLinear(torch.nn.Module):
    """Linear model whose second forward raises or returns NaN when asked to."""

    def __init__(self, failure):
        """Store the failure (``None``, ``"raise"``, or ``"nan"``) for step 1."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        ### DDP broadcasts buffers at the start of every forward, a collective
        ### that a rank which failed during loading must not leave unmatched.
        self.register_buffer("marker", torch.ones(1))
        self.failure = failure
        self.calls = 0

    def forward(self, input):
        """Apply the linear layer, failing on the second call if configured."""
        self.calls += 1
        out = self.linear(input)
        if self.calls == 2 and self.failure == "raise":
            raise RuntimeError("simulated forward failure")
        if self.calls == 2 and self.failure == "nan":
            out = out * float("nan")
        return out


### (failing phase, mode, failing rank); the divergence guard is train-only.
_DDP_FAILURES = [
    (phase, mode, failing_rank)
    for phase in ("loading", "forward", "divergence")
    for mode in (("train",) if phase == "divergence" else ("train", "val"))
    for failing_rank in (0, 1)
]


def _ddp_failure_worker(rank, store_uri):
    """Run every ``_DDP_FAILURES`` case through the real epoch loop."""
    torch.set_num_threads(1)
    train.dist.init_process_group(
        "gloo",
        init_method=store_uri,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    batch = {
        "forward_kwargs": {"input": torch.ones(1, 3, 2)},
        "targets": TensorDict({"pressure": torch.ones(1, 3)}, batch_size=[1, 3]),
    }
    dist_manager = SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu"))
    try:
        for phase, mode, failing_rank in _DDP_FAILURES:
            fails = rank == failing_rank
            failure = {"forward": "raise", "divergence": "nan"}.get(phase)
            stage = "data loading" if phase == "loading" else "forward/loss"
            where = f"{mode} epoch 0, step 1 ({stage})"
            with pytest.raises((ValueError, RuntimeError)) as exc:
                _run_test_epoch(
                    _TwoStepLoader(batch, fail=fails and phase == "loading"),
                    torch.nn.parallel.DistributedDataParallel(
                        _FlakyLinear(failure if fails else None)
                    ),
                    mode=mode,
                    dist_manager=dist_manager,
                    threshold=1.0e6,
                    loss_calculator=train.LossCalculator(
                        {"pressure": "scalar"}, loss_type="mse"
                    ),
                    metric_calculator=train.MetricCalculator({"pressure": "scalar"}),
                )
            if fails:
                expected = {
                    "loading": "corrupt second sample",
                    "forward": "simulated forward failure",
                    "divergence": "divergence_loss_threshold",
                }[phase]
                assert expected in str(exc.value)
                assert exc.value.__notes__ == [f"Raised during {where}."]
            else:
                assert type(exc.value) is RuntimeError
                assert str(exc.value) == f"rank {failing_rank} failed during {where}"
    finally:
        train.dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
def test_ddp_failure_on_one_rank_stops_both(tmp_path):
    """Loading, forward, and divergence failures on either rank stop both ranks.

    One spawn runs every case: each failure leaves both ranks at the same
    rendezvous, so the process group stays usable for the next case.
    """
    torch.multiprocessing.spawn(
        _ddp_failure_worker,
        args=((tmp_path / "store").as_uri(),),
        nprocs=2,
        join=True,
    )


### ---------------------------------------------------------------------------
### Epoch completion / checkpoint ordering
### ---------------------------------------------------------------------------


class TestFinishEpoch:
    """Tests for scheduler/checkpoint ordering and the final checkpoint."""

    @staticmethod
    def _cfg(*, save_interval: int, scheduler_update_mode: str = "epoch"):
        return OmegaConf.create(
            {
                "training": {
                    "save_interval": save_interval,
                    "scheduler_update_mode": scheduler_update_mode,
                }
            }
        )

    def test_scheduler_steps_before_checkpoint(self, monkeypatch):
        """The epoch-mode scheduler steps first; the index counts completed epochs."""
        events = []
        monkeypatch.setattr(
            train,
            "save_checkpoint",
            lambda **kwargs: events.append(("save", kwargs["epoch"])),
        )
        _finish_epoch(
            epoch=0,
            num_epochs=10,
            cfg=self._cfg(save_interval=5),
            scheduler=SimpleNamespace(step=lambda: events.append("step")),
            ckpt_args={"path": "/unused"},
            normalizer=None,
            is_rank0=True,
        )
        assert events == ["step", ("save", 1)]

    @pytest.mark.parametrize(
        ("epoch", "is_rank0", "saved"),
        [(0, True, [1]), (24, True, []), (499, True, [500]), (499, False, [])],
    )
    def test_saves_periodic_and_final_epochs(self, monkeypatch, epoch, is_rank0, saved):
        """Rank 0 saves on the periodic cadence and after the final epoch."""
        saved_epochs = []
        monkeypatch.setattr(
            train,
            "save_checkpoint",
            lambda **kwargs: saved_epochs.append(kwargs["epoch"]),
        )
        _finish_epoch(
            epoch=epoch,
            num_epochs=500,
            cfg=self._cfg(save_interval=25, scheduler_update_mode="step"),
            scheduler=SimpleNamespace(step=lambda: pytest.fail("unexpected step")),
            ckpt_args={"path": "/unused"},
            normalizer=None,
            is_rank0=is_rank0,
        )
        assert saved_epochs == saved

    def test_resume_matches_continuous_training(self, tmp_path):
        """Saving after three epochs and resuming matches six continuous epochs."""
        features = torch.tensor(
            [[0.5, -1.0], [1.5, 0.25], [-0.75, 0.4]], dtype=torch.float64
        )
        targets = torch.tensor([[0.2], [-0.1], [0.7]], dtype=torch.float64)

        def build():
            torch.manual_seed(0)
            model = torch.nn.Linear(2, 1, dtype=torch.float64)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.2
            )
            return model, optimizer, scheduler

        def train_epochs(model, optimizer, scheduler, epochs, save_dir=None):
            lrs = []
            for epoch in epochs:
                lrs.append(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad()
                (model(features) - targets).square().sum().backward()
                optimizer.step()
                _finish_epoch(
                    epoch=epoch,
                    num_epochs=epochs.stop,
                    cfg=self._cfg(save_interval=100),
                    scheduler=scheduler,
                    ckpt_args={
                        "path": str(save_dir),
                        "models": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                    },
                    normalizer=None,
                    is_rank0=save_dir is not None,
                )
            return lrs

        model, optimizer, scheduler = build()
        expected_lrs = train_epochs(model, optimizer, scheduler, range(6))

        lrs = train_epochs(*build(), range(3), save_dir=tmp_path)
        resumed_model, resumed_optimizer, resumed_scheduler = build()
        start = train.load_checkpoint(
            path=str(tmp_path),
            models=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            device="cpu",
        )
        lrs += train_epochs(
            resumed_model, resumed_optimizer, resumed_scheduler, range(start, 6)
        )

        assert start == 3
        assert lrs == expected_lrs
        assert resumed_scheduler.state_dict() == scheduler.state_dict()
        assert torch.equal(resumed_model.weight, model.weight)
        assert torch.equal(resumed_model.bias, model.bias)
