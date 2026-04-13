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

"""Tests for MSEDSMLoss and WeightedMSEDSMLoss."""

import pytest
import torch

from physicsnemo.diffusion.metrics.losses import MSEDSMLoss, WeightedMSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_losses_"
BATCH = 4
LR = 1e-2
TRAIN_STEPS = 2

SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VENoiseScheduler, {}, "ve"),
    (VPNoiseScheduler, {}, "vp"),
]

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
]

PREDICTION_TYPES = ["x0", "score"]


# =============================================================================
# Helpers
# =============================================================================


def _first_param(model):
    """Return a clone of the first parameter."""
    return next(model.parameters()).detach().clone()


def _make_loss(model, scheduler, prediction_type):
    """Create an MSEDSMLoss with the given prediction type."""
    kwargs = {}
    if prediction_type == "score":
        kwargs["score_to_x0_fn"] = scheduler.score_to_x0
    return MSEDSMLoss(model, scheduler, prediction_type=prediction_type, **kwargs)


def _make_weighted_loss(model, scheduler, prediction_type):
    """Create a WeightedMSEDSMLoss with the given prediction type."""
    kwargs = {}
    if prediction_type == "score":
        kwargs["score_to_x0_fn"] = scheduler.score_to_x0
    return WeightedMSEDSMLoss(
        model, scheduler, prediction_type=prediction_type, **kwargs
    )


def _run_training_loop(loss_fn, model, x0, condition=None, steps=TRAIN_STEPS):
    """Run a minimal training loop and return per-step loss + param snapshots."""
    losses = []
    params = []
    for _ in range(steps):
        loss = loss_fn(x0, condition=condition)
        loss.backward()
        losses.append(loss.detach().cpu())
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= LR * p.grad
                    p.grad = None
        params.append(_first_param(model).cpu())
    return losses, params


def _run_weighted_training_loop(
    loss_fn, model, x0, weight, condition=None, steps=TRAIN_STEPS
):
    """Run a minimal training loop with weighted loss."""
    losses = []
    params = []
    for _ in range(steps):
        loss = loss_fn(x0, weight=weight, condition=condition)
        loss.backward()
        losses.append(loss.detach().cpu())
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= LR * p.grad
                    p.grad = None
        params.append(_first_param(model).cpu())
    return losses, params


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Tests for loss constructor and validation."""

    def test_mse_constructor(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = EDMNoiseScheduler()
        loss_fn = MSEDSMLoss(model, scheduler)
        assert loss_fn.model is model
        assert loss_fn.noise_scheduler is scheduler

    def test_weighted_mse_constructor(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = EDMNoiseScheduler()
        loss_fn = WeightedMSEDSMLoss(model, scheduler)
        assert loss_fn.model is model
        assert loss_fn.noise_scheduler is scheduler

    def test_invalid_prediction_type(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        with pytest.raises(ValueError, match="prediction_type"):
            MSEDSMLoss(model, EDMNoiseScheduler(), prediction_type="bad")

    def test_score_requires_fn(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        with pytest.raises(ValueError, match="score_to_x0_fn"):
            MSEDSMLoss(model, EDMNoiseScheduler(), prediction_type="score")

    def test_invalid_reduction(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        with pytest.raises(ValueError, match="reduction"):
            MSEDSMLoss(model, EDMNoiseScheduler(), reduction="bad")

    def test_reduction_none(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = EDMNoiseScheduler()
        loss_fn = MSEDSMLoss(model, scheduler, reduction="none")
        x0 = make_input((BATCH, 3, 16))
        out = loss_fn(x0)
        assert out.shape == x0.shape

    def test_reduction_sum(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = EDMNoiseScheduler()
        loss_fn = MSEDSMLoss(model, scheduler, reduction="sum")
        x0 = make_input((BATCH, 3, 16))
        out = loss_fn(x0)
        assert out.ndim == 0


# =============================================================================
# Non-Regression Tests — MSEDSMLoss
# =============================================================================


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestMSEDSMLossNonRegression:
    """Non-regression training loop tests for MSEDSMLoss."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = sched_cls(**sched_kwargs)
        loss_fn = _make_loss(model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        param_before = _first_param(model).cpu()

        losses, params = _run_training_loop(loss_fn, model, x0)

        for loss_val in losses:
            assert loss_val.ndim == 0 and torch.isfinite(loss_val)
        assert not torch.equal(param_before, params[0])
        assert not torch.equal(params[0], params[1])

        if "cuda" in str(device):
            ref_file = (
                f"{REF_PREFIX}mse_{sched_name}_{spatial_name}_{prediction_type}.pth"
            )
            ref = load_or_create_reference(ref_file, None)
            assert losses[0].shape == ref["loss_0"].shape
            assert params[0].shape == ref["param_0"].shape
        else:
            ref_file = (
                f"{REF_PREFIX}mse_{sched_name}_{spatial_name}_{prediction_type}.pth"
            )
            ref = load_or_create_reference(
                ref_file,
                lambda: {
                    "loss_0": losses[0],
                    "loss_1": losses[1],
                    "param_0": params[0],
                    "param_1": params[1],
                },
            )
            compare_outputs(losses[0], ref["loss_0"], **tolerances)
            compare_outputs(losses[1], ref["loss_1"], **tolerances)
            compare_outputs(params[0], ref["param_0"], **tolerances)
            compare_outputs(params[1], ref["param_1"], **tolerances)


# =============================================================================
# Non-Regression Tests — WeightedMSEDSMLoss
# =============================================================================


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestWeightedMSEDSMLossNonRegression:
    """Non-regression training loop tests for WeightedMSEDSMLoss."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = sched_cls(**sched_kwargs)
        loss_fn = _make_weighted_loss(model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        # Zero out half the spatial dimensions
        weight.narrow(-1, 0, shape[-1] // 2).zero_()
        param_before = _first_param(model).cpu()

        losses, params = _run_weighted_training_loop(loss_fn, model, x0, weight)

        for loss_val in losses:
            assert loss_val.ndim == 0 and torch.isfinite(loss_val)
        assert not torch.equal(param_before, params[0])
        assert not torch.equal(params[0], params[1])

        if "cuda" in str(device):
            ref_file = (
                f"{REF_PREFIX}wmse_{sched_name}_{spatial_name}_{prediction_type}.pth"
            )
            ref = load_or_create_reference(ref_file, None)
            assert losses[0].shape == ref["loss_0"].shape
            assert params[0].shape == ref["param_0"].shape
        else:
            ref_file = (
                f"{REF_PREFIX}wmse_{sched_name}_{spatial_name}_{prediction_type}.pth"
            )
            ref = load_or_create_reference(
                ref_file,
                lambda: {
                    "loss_0": losses[0],
                    "loss_1": losses[1],
                    "param_0": params[0],
                    "param_1": params[1],
                },
            )
            compare_outputs(losses[0], ref["loss_0"], **tolerances)
            compare_outputs(losses[1], ref["loss_1"], **tolerances)
            compare_outputs(params[0], ref["param_0"], **tolerances)
            compare_outputs(params[1], ref["param_1"], **tolerances)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through the losses."""

    @pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
    def test_mse_gradient_flow(self, device, prediction_type):
        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = EDMNoiseScheduler()
        loss_fn = _make_loss(model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        loss = loss_fn(x0)
        loss.backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    @pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
    def test_weighted_mse_gradient_flow(self, device, prediction_type):
        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = EDMNoiseScheduler()
        loss_fn = _make_weighted_loss(model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, :3] = 0.0
        loss = loss_fn(x0, weight=weight)
        loss.backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad


# =============================================================================
# Compile Tests
# =============================================================================


COMPILE_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VPNoiseScheduler, {}, "vp"),
]


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    COMPILE_CONFIGS,
    ids=[c[2] for c in COMPILE_CONFIGS],
)
class TestMSEDSMLossCompile:
    """Double-call compile tests for MSEDSMLoss."""

    def test_compile(
        self,
        deterministic_settings,
        device,
        prediction_type,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        """Compiled loss produces finite output and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = sched_cls(**sched_kwargs)
        loss_fn = _make_loss(model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)

        compiled_loss_fn = torch.compile(loss_fn, fullgraph=True)

        # First call — triggers tracing
        loss_1 = compiled_loss_fn(x0)
        assert loss_1.ndim == 0 and torch.isfinite(loss_1)

        # Second call — must reuse the graph
        loss_2 = compiled_loss_fn(x0)
        assert loss_2.ndim == 0 and torch.isfinite(loss_2)


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    COMPILE_CONFIGS,
    ids=[c[2] for c in COMPILE_CONFIGS],
)
class TestWeightedMSEDSMLossCompile:
    """Double-call compile tests for WeightedMSEDSMLoss."""

    def test_compile(
        self,
        deterministic_settings,
        device,
        prediction_type,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        """Compiled weighted loss produces finite output and graph is reused."""
        torch._dynamo.config.error_on_recompile = True

        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = sched_cls(**sched_kwargs)
        loss_fn = _make_weighted_loss(model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, :3] = 0.0

        compiled_loss_fn = torch.compile(loss_fn, fullgraph=True)

        # First call — triggers tracing
        loss_1 = compiled_loss_fn(x0, weight=weight)
        assert loss_1.ndim == 0 and torch.isfinite(loss_1)

        # Second call — must reuse the graph
        loss_2 = compiled_loss_fn(x0, weight=weight)
        assert loss_2.ndim == 0 and torch.isfinite(loss_2)
