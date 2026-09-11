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
- :func:`train._step_failure_barrier` / :func:`train._raise_if_divergent_loss`:
  loading failures rendezvous before DDP forward, and local forward/loss
  failures rendezvous before backward.
- :func:`train._finish_epoch` / :func:`train._reconcile_loaded_checkpoint`:
  the epoch-mode scheduler advances before the checkpoint is written, so a
  resumed scheduler/scaler state is restored; older checkpoints are migrated.
  Stochastic trajectory fidelity remains unverified without RNG restoration.

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
    _finish_epoch,
    _raise_if_divergent_loss,
    _step_failure_barrier,
    _reconcile_loaded_checkpoint,
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
### Loss divergence guard
### ---------------------------------------------------------------------------


class TestLossDivergenceGuard:
    """Tests for the synchronized, pre-backward loss guard."""

    @staticmethod
    def _dist_manager(*, world_size: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            rank=0, world_size=world_size, device=torch.device("cpu")
        )

    def test_finite_loss_at_threshold_passes(self):
        """The optional threshold is strict: equality is still accepted."""
        _raise_if_divergent_loss(
            torch.tensor(1000.0),
            mode="train",
            epoch=3,
            step=7,
            threshold=1000.0,
        )

    @pytest.mark.parametrize(
        ("loss", "message"),
        [
            (float("nan"), "non-finite"),
            (float("inf"), "non-finite"),
            (1000.1, "above 1000"),
        ],
    )
    def test_local_nonfinite_or_threshold_failure_raises(self, loss, message):
        """NaN, inf, and above-threshold losses each raise with a clear reason."""
        with pytest.raises(RuntimeError, match=message):
            _raise_if_divergent_loss(
                torch.tensor(loss),
                mode="val",
                epoch=2,
                step=4,
                threshold=1000.0,
            )

    def test_barrier_is_noop_on_healthy_step(self):
        """Single process, no error: the barrier returns without a collective."""
        _step_failure_barrier(
            None, mode="train", epoch=1, step=2, dist_manager=self._dist_manager()
        )

    def test_barrier_reraises_local_error_with_cause(self):
        """The failing rank's original exception is chained as ``__cause__``."""
        original = ValueError("corrupt sample")
        with pytest.raises(RuntimeError, match="step failed at epoch 5") as info:
            _step_failure_barrier(
                original,
                mode="train",
                epoch=5,
                step=9,
                dist_manager=self._dist_manager(),
            )
        assert info.value.__cause__ is original

    def test_barrier_synchronizes_peer_failure(self, monkeypatch):
        """A healthy rank raises too when the reduced any-rank flag is set."""

        def report_remote_failure(flag, op):
            assert op == torch.distributed.ReduceOp.MAX
            flag.fill_(1)

        monkeypatch.setattr(train.dist, "all_reduce", report_remote_failure)
        with pytest.raises(RuntimeError, match="peer rank failed"):
            _step_failure_barrier(
                None,
                mode="train",
                epoch=0,
                step=0,
                dist_manager=self._dist_manager(world_size=2),
            )

    def test_barrier_healthy_when_no_rank_failed(self, monkeypatch):
        """Multi-rank, all healthy: the reduced flag stays 0 and nothing raises."""

        def report_no_failure(flag, op):
            assert op == torch.distributed.ReduceOp.MAX

        monkeypatch.setattr(train.dist, "all_reduce", report_no_failure)
        _step_failure_barrier(
            None,
            mode="val",
            epoch=0,
            step=0,
            dist_manager=self._dist_manager(world_size=2),
        )

    @pytest.mark.parametrize("mode", ["train", "val"])
    def test_epoch_loop_checks_before_backward_in_both_modes(self, mode, monkeypatch):
        """A NaN from forward aborts train and val; train creates no gradient."""
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        nan = torch.tensor(float("nan"))

        def divergent_forward(*args, **kwargs):
            loss = model.weight.sum() * nan
            values = TensorDict({"loss/test": loss.detach()})
            return loss, values, values.clone()

        monkeypatch.setattr(train, "forward_pass", divergent_forward)
        cfg = OmegaConf.create(
            {
                "precision": "float32",
                "profile": False,
                "training": {
                    "scheduler_update_mode": "epoch",
                    "divergence_loss_threshold": 1.0e6,
                },
            }
        )
        kwargs = {}
        if mode == "train":
            kwargs = {"optimizer": optimizer, "scheduler": scheduler}

        with pytest.raises(RuntimeError, match=rf"{mode} step failed at epoch"):
            _run_epoch(
                [{}],
                model,
                None,
                None,
                SimpleNamespace(
                    info=lambda *args, **kwargs: None,
                    error=lambda *args, **kwargs: None,
                ),
                0,
                cfg,
                self._dist_manager(),
                mode=mode,
                output_type="tensors",
                target_config={"pressure": "scalar"},
                **kwargs,
            )
        assert model.weight.grad is None

    def test_default_null_threshold_runs_no_per_step_guard(self, monkeypatch):
        """With the knob unset, the loop must not pay the guard's host sync."""

        def forbidden_guard(*args, **kwargs):
            raise AssertionError("guard must not run when threshold is null")

        def finite_forward(*args, **kwargs):
            loss = torch.tensor(0.5, requires_grad=True)
            values = TensorDict({"loss/test": loss.detach()})
            return loss, values, values.clone()

        monkeypatch.setattr(train, "_raise_if_divergent_loss", forbidden_guard)
        monkeypatch.setattr(train, "forward_pass", finite_forward)
        cfg = OmegaConf.create(
            {
                "precision": "float32",
                "profile": False,
                "training": {
                    "scheduler_update_mode": "epoch",
                    "divergence_loss_threshold": None,
                },
            }
        )
        _run_epoch(
            [{}],
            torch.nn.Linear(1, 1),
            None,
            None,
            SimpleNamespace(
                info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None
            ),
            0,
            cfg,
            self._dist_manager(),
            mode="val",
            output_type="tensors",
            target_config={"pressure": "scalar"},
        )

    def test_invalid_threshold_fails_before_forward(self, monkeypatch):
        """A non-positive threshold is a config error caught before any step runs."""
        called = False

        def forward(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("forward should not run")

        monkeypatch.setattr(train, "forward_pass", forward)
        cfg = OmegaConf.create(
            {
                "precision": "float32",
                "profile": False,
                "training": {
                    "scheduler_update_mode": "epoch",
                    "divergence_loss_threshold": 0,
                },
            }
        )
        with pytest.raises(ValueError, match="must be positive"):
            _run_epoch(
                [{}],
                torch.nn.Linear(1, 1),
                None,
                None,
                SimpleNamespace(
                    info=lambda *args, **kwargs: None,
                    error=lambda *args, **kwargs: None,
                ),
                0,
                cfg,
                self._dist_manager(),
                mode="val",
                output_type="tensors",
                target_config={"pressure": "scalar"},
            )
        assert not called


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


def _ddp_loading_failure_worker(rank, store_uri, mode, failing_rank):
    """Exercise the real epoch loop with a rank-local second-load failure."""
    torch.set_num_threads(1)
    train.dist.init_process_group(
        "gloo",
        init_method=store_uri,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        base = torch.nn.Linear(2, 1)
        base.register_buffer("marker", torch.ones(1))
        model = torch.nn.parallel.DistributedDataParallel(base)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        batch = {
            "forward_kwargs": {"input": torch.ones(1, 3, 2)},
            "targets": TensorDict({"pressure": torch.ones(1, 3)}, batch_size=[1, 3]),
        }
        cfg = OmegaConf.create(
            {
                "precision": "float32",
                "profile": False,
                "training": {"scheduler_update_mode": "epoch"},
            }
        )
        expected = (
            "step failed at epoch 0, step 1"
            if rank == failing_rank
            else "peer rank failed during"
        )
        with pytest.raises(RuntimeError, match=expected) as exc:
            _run_epoch(
                _TwoStepLoader(batch, fail=rank == failing_rank),
                model,
                train.LossCalculator({"pressure": "scalar"}, loss_type="mse"),
                train.MetricCalculator({"pressure": "scalar"}),
                SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
                0,
                cfg,
                SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu")),
                mode=mode,
                output_type="tensors",
                target_config={"pressure": "scalar"},
                optimizer=optimizer,
                scheduler=scheduler,
            )
        assert "data loading" in str(exc.value)
        if rank == failing_rank:
            assert isinstance(exc.value.__cause__, ValueError)
            assert str(exc.value.__cause__) == "corrupt second sample"
        else:
            assert exc.value.__cause__ is None
    finally:
        train.dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("mode", ["train", "val"])
@pytest.mark.parametrize("failing_rank", [0, 1])
def test_ddp_loading_failure_precedes_forward(tmp_path, mode, failing_rank):
    """Both ranks raise the intended error instead of a mismatched-collective timeout."""
    torch.multiprocessing.spawn(
        _ddp_loading_failure_worker,
        args=((tmp_path / "store").as_uri(), mode, failing_rank),
        nprocs=2,
        join=True,
    )


### ---------------------------------------------------------------------------
### Epoch completion / checkpoint ordering
### ---------------------------------------------------------------------------


class TestFinishEpoch:
    """Tests for scheduler/checkpoint ordering and terminal persistence."""

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

    def test_scheduler_advances_before_checkpoint_and_epoch_is_next(self, monkeypatch):
        """Epoch-mode scheduler steps first; the saved epoch is the resume index."""
        events = []
        scheduler = SimpleNamespace(step=lambda: events.append("scheduler"))

        def save(**kwargs):
            metadata = kwargs["metadata"]["unified_external_aero_recipe"]
            events.append(
                ("checkpoint", kwargs["epoch"], metadata["scaler_state_saved"])
            )

        monkeypatch.setattr(train, "save_checkpoint", save)
        saved = _finish_epoch(
            epoch=4,
            num_epochs=5,
            cfg=self._cfg(save_interval=2),
            scheduler=scheduler,
            ckpt_args={"path": "/unused"},
            normalizer=None,
            is_rank0=True,
        )

        assert saved
        ### Epoch 4 is both periodic and terminal, but is persisted once as
        ### five completed epochs / the next resume index.
        assert events == ["scheduler", ("checkpoint", 5, False)]

    def test_nonperiodic_terminal_is_always_saved(self, monkeypatch):
        """The last epoch is checkpointed even when it is off the periodic cadence."""
        saved_epochs = []
        monkeypatch.setattr(
            train,
            "save_checkpoint",
            lambda **kwargs: saved_epochs.append(kwargs["epoch"]),
        )
        saved = _finish_epoch(
            epoch=4,
            num_epochs=5,
            cfg=self._cfg(save_interval=3, scheduler_update_mode="step"),
            scheduler=SimpleNamespace(step=lambda: pytest.fail("unexpected step")),
            ckpt_args={"path": "/unused"},
            normalizer=None,
            is_rank0=True,
        )
        assert saved
        assert saved_epochs == [5]

    def test_existing_periodic_cadence_is_preserved(self, monkeypatch):
        """The historical epoch-0 save remains checkpoint 1."""
        saved_epochs = []
        monkeypatch.setattr(
            train,
            "save_checkpoint",
            lambda **kwargs: saved_epochs.append(kwargs["epoch"]),
        )
        _finish_epoch(
            epoch=0,
            num_epochs=500,
            cfg=self._cfg(save_interval=25, scheduler_update_mode="step"),
            scheduler=SimpleNamespace(step=lambda: pytest.fail("unexpected step")),
            ckpt_args={"path": "/unused"},
            normalizer=None,
            is_rank0=True,
        )
        assert saved_epochs == [1]

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_real_save_resume_reports_state_fidelity(self, tmp_path, dropout):
        """Restore scheduler state without promising stochastic trajectory fidelity."""
        features = torch.tensor(
            [[0.5, -1.0], [1.5, 0.25], [-0.75, 0.4]], dtype=torch.float64
        )
        targets = torch.tensor([[0.2], [-0.1], [0.7]], dtype=torch.float64)

        def build_state():
            torch.manual_seed(1701)
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 4, dtype=torch.float64),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(4, 1, dtype=torch.float64),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.03, weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.2
            )
            return model, optimizer, scheduler

        def step_epoch(model, optimizer, scheduler, epoch, checkpoint_dir=None):
            optimizer.zero_grad()
            loss = (model(features) - targets).square().sum()
            loss.backward()
            optimizer.step()
            _finish_epoch(
                epoch=epoch,
                num_epochs=3 if checkpoint_dir is not None else 6,
                cfg=self._cfg(save_interval=1000),
                scheduler=scheduler,
                ckpt_args={
                    "path": checkpoint_dir or tmp_path / "unused",
                    "models": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
                normalizer=None,
                is_rank0=checkpoint_dir is not None,
            )
            return (
                loss.detach().clone(),
                torch.nn.utils.parameters_to_vector(model.parameters())
                .detach()
                .clone(),
                optimizer.param_groups[0]["lr"],
            )

        continuous_model, continuous_opt, continuous_sched = build_state()
        continuous_history = [
            step_epoch(continuous_model, continuous_opt, continuous_sched, epoch)
            for epoch in range(6)
        ]

        first_model, first_opt, first_sched = build_state()
        checkpoint_dir = tmp_path / "resume"
        resumed_history = [
            step_epoch(
                first_model,
                first_opt,
                first_sched,
                epoch,
                checkpoint_dir if epoch == 2 else None,
            )
            for epoch in range(3)
        ]
        resumed_model, resumed_opt, resumed_sched = build_state()
        metadata = {}
        loaded_epoch = train.load_checkpoint(
            path=checkpoint_dir,
            models=resumed_model,
            optimizer=resumed_opt,
            scheduler=resumed_sched,
            metadata_dict=metadata,
            device="cpu",
        )
        assert loaded_epoch == 3
        report = _reconcile_loaded_checkpoint(
            loaded_epoch=loaded_epoch,
            metadata=metadata,
            scheduler=resumed_sched,
            scheduler_update_mode="epoch",
            scaler=None,
            logger=SimpleNamespace(warning=lambda message: pytest.fail(message)),
        )
        assert not report["legacy_checkpoint"]
        assert report["scheduler_scaler_state_restored"]
        assert report["trajectory_exactness"] == "unverified"
        assert "RNG" in report["trajectory_exactness_reason"]
        assert "resume_exact" not in report
        resumed_history.extend(
            step_epoch(resumed_model, resumed_opt, resumed_sched, epoch)
            for epoch in range(loaded_epoch, 6)
        )

        assert continuous_sched.state_dict() == resumed_sched.state_dict()
        for epoch, (expected, actual) in enumerate(
            zip(continuous_history, resumed_history, strict=True)
        ):
            assert expected[2] == actual[2]
            if dropout == 0.0 or epoch < loaded_epoch:
                assert torch.equal(expected[0], actual[0])
                assert torch.equal(expected[1], actual[1])
        if dropout:
            assert not torch.equal(continuous_history[-1][1], resumed_history[-1][1])

    def test_no_checkpoint_does_not_report_restored_state(self):
        """A fresh run has no restored state or trajectory-fidelity assessment."""
        report = _reconcile_loaded_checkpoint(
            loaded_epoch=0,
            metadata={},
            scheduler=SimpleNamespace(step=lambda: pytest.fail("unexpected step")),
            scheduler_update_mode="epoch",
            scaler=None,
            logger=SimpleNamespace(warning=lambda message: pytest.fail(message)),
        )
        assert not report["checkpoint_found"]
        assert not report["scheduler_scaler_state_restored"]
        assert report["trajectory_exactness"] == "not_applicable"

    def test_legacy_epoch_checkpoint_scheduler_is_migrated(self, tmp_path):
        """Old pre-step scheduler state is advanced to the continuous state."""
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        ### Reproduce the old ordering after two completed epochs: epoch 1
        ### stepped, then checkpoint 2 was written before the epoch-2 step.
        optimizer.step()
        scheduler.step()
        train.save_checkpoint(
            path=tmp_path,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
        )

        resumed_model = torch.nn.Linear(1, 1)
        resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=1.0)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(
            resumed_optimizer, step_size=2, gamma=0.1
        )
        metadata = {}
        loaded_epoch = train.load_checkpoint(
            path=tmp_path,
            models=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            metadata_dict=metadata,
            device="cpu",
        )
        messages = []
        report = _reconcile_loaded_checkpoint(
            loaded_epoch=loaded_epoch,
            metadata=metadata,
            scheduler=resumed_scheduler,
            scheduler_update_mode="epoch",
            scaler=None,
            logger=SimpleNamespace(warning=messages.append),
        )

        assert resumed_scheduler.last_epoch == 2
        assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        assert report["legacy_scheduler_step_applied"]
        assert report["scheduler_scaler_state_restored"]
        assert report["trajectory_exactness"] == "unverified"
        assert len(messages) == 1

    def test_legacy_fp16_resume_reports_missing_scaler(self):
        """A pre-metadata checkpoint resumed under fp16 is flagged as non-exact."""
        messages = []
        scaler = SimpleNamespace()
        report = _reconcile_loaded_checkpoint(
            loaded_epoch=25,
            metadata={},
            scheduler=SimpleNamespace(step=lambda: None),
            scheduler_update_mode="step",
            scaler=scaler,
            logger=SimpleNamespace(warning=messages.append),
        )

        assert not report["scheduler_scaler_state_restored"]
        assert "scaler" in report["missing_state_reason"]
        assert len(messages) == 1

    def test_metadata_checkpoint_reports_missing_scaler(self):
        """A post-fix fp32 checkpoint resumed under fp16 is flagged, not migrated."""
        messages = []
        report = _reconcile_loaded_checkpoint(
            loaded_epoch=4,
            metadata={"unified_external_aero_recipe": {"scaler_state_saved": False}},
            scheduler=SimpleNamespace(step=lambda: pytest.fail("unexpected step")),
            scheduler_update_mode="epoch",
            scaler=SimpleNamespace(),
            logger=SimpleNamespace(warning=messages.append),
        )

        assert not report["legacy_checkpoint"]
        assert not report["legacy_scheduler_step_applied"]
        assert not report["scheduler_scaler_state_restored"]
        assert len(messages) == 1
