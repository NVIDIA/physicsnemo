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

"""Tests for MSEDSMLoss, WeightedMSEDSMLoss, FlowMatchingLoss, and
WeightedFlowMatchingLoss."""

import pytest
import torch

from physicsnemo.diffusion.metrics.losses import (
    FlowMatchingLoss,
    MSEDSMLoss,
    WeightedFlowMatchingLoss,
    WeightedMSEDSMLoss,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    RectifiedFlowNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.preconditioners import EDMPreconditioner

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

# sigma_data must be consistent between EDMPreconditioner and EDMNoiseScheduler
# to mirror the realistic SDA recipe pattern.
SIGMA_DATA = 1.0

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
]

PREDICTION_TYPES = ["x0", "score", "epsilon", "flow"]

RF_SCHEDULER_KWARGS = {
    "x0": {"t_min": 1e-3},
    "score": {},
    "epsilon": {},
    "flow": {},
}

LOSS_CONFIGS = [
    (MSEDSMLoss, EDMNoiseScheduler, {}, "mse_edm"),
    (MSEDSMLoss, VENoiseScheduler, {}, "mse_ve"),
    (MSEDSMLoss, VPNoiseScheduler, {}, "mse_vp"),
    (FlowMatchingLoss, RectifiedFlowNoiseScheduler, {}, "fm"),
]

WEIGHTED_LOSS_CONFIGS = [
    (WeightedMSEDSMLoss, EDMNoiseScheduler, {}, "wmse_edm"),
    (WeightedMSEDSMLoss, VENoiseScheduler, {}, "wmse_ve"),
    (WeightedMSEDSMLoss, VPNoiseScheduler, {}, "wmse_vp"),
    (WeightedFlowMatchingLoss, RectifiedFlowNoiseScheduler, {}, "wfm"),
]

# Subset for compile tests
COMPILE_LOSS_CONFIGS = [
    (MSEDSMLoss, EDMNoiseScheduler, {}, "mse_edm"),
    (MSEDSMLoss, VPNoiseScheduler, {}, "mse_vp"),
    (FlowMatchingLoss, RectifiedFlowNoiseScheduler, {}, "fm"),
]

WEIGHTED_COMPILE_LOSS_CONFIGS = [
    (WeightedMSEDSMLoss, EDMNoiseScheduler, {}, "wmse_edm"),
    (WeightedMSEDSMLoss, VPNoiseScheduler, {}, "wmse_vp"),
    (WeightedFlowMatchingLoss, RectifiedFlowNoiseScheduler, {}, "wfm"),
]

_LOSS_IDS = [c[3] for c in LOSS_CONFIGS]
_WEIGHTED_LOSS_IDS = [c[3] for c in WEIGHTED_LOSS_CONFIGS]
_COMPILE_LOSS_IDS = [c[3] for c in COMPILE_LOSS_CONFIGS]
_WEIGHTED_COMPILE_LOSS_IDS = [c[3] for c in WEIGHTED_COMPILE_LOSS_CONFIGS]
_SPATIAL_IDS = [c[0] for c in SPATIAL_CONFIGS]


# =============================================================================
# Helpers
# =============================================================================


def _first_param(model):
    """Return a clone of the first parameter."""
    return next(model.parameters()).detach().clone()


def _make_scheduler(sched_cls, sched_kwargs, prediction_type):
    """Instantiate a scheduler, applying the RF per-prediction-type kwargs."""
    kwargs = dict(sched_kwargs)
    if sched_cls is RectifiedFlowNoiseScheduler:
        kwargs.update(RF_SCHEDULER_KWARGS[prediction_type])
    return sched_cls(**kwargs)


def _make_loss(loss_cls, model, scheduler, prediction_type):
    """Create a loss of type *loss_cls* with the given prediction type.

    Wires the conversion callback required by ``prediction_type`` from the
    scheduler's conversion methods: to-x0 conversions for the DSM losses,
    to-flow conversions for the flow matching losses.
    """
    kwargs = {}
    if loss_cls in (MSEDSMLoss, WeightedMSEDSMLoss):
        if prediction_type == "score":
            kwargs["score_to_x0_fn"] = scheduler.score_to_x0
        elif prediction_type == "epsilon":
            kwargs["epsilon_to_x0_fn"] = scheduler.epsilon_to_x0
        elif prediction_type == "flow":
            kwargs["flow_to_x0_fn"] = scheduler.flow_to_x0
    else:
        kwargs["x0_to_flow_fn"] = scheduler.x0_to_flow
        if prediction_type == "score":
            kwargs["score_to_flow_fn"] = scheduler.score_to_flow
        elif prediction_type == "epsilon":
            kwargs["epsilon_to_flow_fn"] = lambda eps, x_t, t: scheduler.x0_to_flow(
                scheduler.epsilon_to_x0(eps, x_t, t), x_t, t
            )
    return loss_cls(model, scheduler, prediction_type=prediction_type, **kwargs)


def _make_preconditioned_model(predictor_cls, predictor_kwargs, seed=0):
    """Wrap predictor_cls in EDMPreconditioner with sigma_data=SIGMA_DATA.

    sigma_data must be consistent with the EDMNoiseScheduler used in the loss
    to mirror the realistic SDA recipe pattern.
    """
    inner = instantiate_model_deterministic(
        predictor_cls, seed=seed, **predictor_kwargs
    )
    return EDMPreconditioner(inner, sigma_data=SIGMA_DATA)


def _run_training_loop(
    loss_fn, model, x0, weight=None, condition=None, steps=TRAIN_STEPS
):
    """Run a minimal training loop and return per-step loss + param snapshots.

    Passes ``weight`` through to the loss call when provided (weighted
    losses); omits it otherwise.
    """
    loss_kwargs = {} if weight is None else {"weight": weight}
    losses = []
    params = []
    for _ in range(steps):
        loss = loss_fn(x0, condition=condition, **loss_kwargs)
        loss.backward()
        losses.append(loss.detach().cpu())
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= LR * p.grad
                    p.grad = None
        params.append(_first_param(model).cpu())
    return losses, params


def _check_non_regression(losses, params, param_before, ref_file, device, tolerances):
    """Assert finite training-loop invariants and compare against goldens."""
    for loss_val in losses:
        assert loss_val.ndim == 0 and torch.isfinite(loss_val)
    assert not torch.equal(param_before, params[0])
    assert not torch.equal(params[0], params[1])

    if "cuda" in str(device):
        ref = load_or_create_reference(ref_file, None)
        assert losses[0].shape == ref["loss_0"].shape
        assert params[0].shape == ref["param_0"].shape
    else:
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


def _half_masked_weight(x0, shape):
    """Weight of ones with the first half of the last dimension zeroed."""
    weight = torch.ones_like(x0)
    weight.narrow(-1, 0, shape[-1] // 2).zero_()
    return weight


def _assert_has_grad(model):
    """Assert at least one parameter received a finite gradient."""
    has_grad = any(
        p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters()
    )
    assert has_grad


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

    @pytest.mark.parametrize(
        "prediction_type,missing_fn",
        [
            ("score", "score_to_x0_fn"),
            ("epsilon", "epsilon_to_x0_fn"),
            ("flow", "flow_to_x0_fn"),
        ],
    )
    def test_requires_conversion_fn(self, prediction_type, missing_fn):
        """Non-x0 prediction types require the matching conversion callback."""
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        with pytest.raises(ValueError, match=missing_fn):
            MSEDSMLoss(model, EDMNoiseScheduler(), prediction_type=prediction_type)

    def test_epsilon_constructor(self):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = EDMNoiseScheduler()
        loss_fn = MSEDSMLoss(
            model,
            scheduler,
            prediction_type="epsilon",
            epsilon_to_x0_fn=scheduler.epsilon_to_x0,
        )
        assert loss_fn.model is model

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


@pytest.mark.parametrize(
    "loss_cls",
    [FlowMatchingLoss, WeightedFlowMatchingLoss],
    ids=["fm", "wfm"],
)
class TestFlowMatchingLossConstructor:
    """Constructor and validation tests shared by both flow matching losses."""

    def test_constructor(self, loss_cls):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = loss_cls(model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow)
        assert loss_fn.model is model
        assert loss_fn.noise_scheduler is scheduler

    def test_requires_x0_to_flow_fn(self, loss_cls):
        """The constructor demands x0_to_flow_fn: it computes the flow target."""
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match="x0_to_flow_fn"):
            loss_cls(model, scheduler)

    @pytest.mark.parametrize(
        "prediction_type,missing_fn",
        [
            ("score", "score_to_flow_fn"),
            ("epsilon", "epsilon_to_flow_fn"),
        ],
    )
    def test_requires_conversion_fn(self, loss_cls, prediction_type, missing_fn):
        """score/epsilon prediction types require the matching callback."""
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match=missing_fn):
            loss_cls(
                model,
                scheduler,
                prediction_type=prediction_type,
                x0_to_flow_fn=scheduler.x0_to_flow,
            )

    def test_invalid_prediction_type(self, loss_cls):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match="prediction_type"):
            loss_cls(
                model,
                scheduler,
                prediction_type="bad",
                x0_to_flow_fn=scheduler.x0_to_flow,
            )

    def test_invalid_reduction(self, loss_cls):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match="reduction"):
            loss_cls(
                model,
                scheduler,
                x0_to_flow_fn=scheduler.x0_to_flow,
                reduction="bad",
            )

    def test_reduction_none(self, loss_cls):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = loss_cls(
            model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow, reduction="none"
        )
        x0 = make_input((BATCH, 3, 16))
        kwargs = (
            {"weight": torch.ones_like(x0)}
            if loss_cls is WeightedFlowMatchingLoss
            else {}
        )
        out = loss_fn(x0, **kwargs)
        assert out.shape == x0.shape

    def test_reduction_sum(self, loss_cls):
        model = instantiate_model_deterministic(FlatLinearX0Predictor, features=48)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = loss_cls(
            model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow, reduction="sum"
        )
        x0 = make_input((BATCH, 3, 16))
        kwargs = (
            {"weight": torch.ones_like(x0)}
            if loss_cls is WeightedFlowMatchingLoss
            else {}
        )
        out = loss_fn(x0, **kwargs)
        assert out.ndim == 0

    def test_binary_mask_zeroes_masked_region(self, loss_cls):
        """A zero weight fully excludes masked elements from the loss."""
        if loss_cls is not WeightedFlowMatchingLoss:
            pytest.skip("weight argument only exists on the weighted loss")
        model = instantiate_model_deterministic(Conv2dX0Predictor, seed=0, channels=3)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = loss_cls(
            model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow, reduction="none"
        )
        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED)
        weight = torch.ones_like(x0)
        weight[:, :, :, :3] = 0.0
        t = scheduler.sample_time(BATCH)
        out = loss_fn(x0, weight=weight, t=t)
        assert torch.equal(out[:, :, :, :3], torch.zeros_like(out[:, :, :, :3]))


# =============================================================================
# Non-Regression Tests — non-weighted losses (DSM and flow matching)
# =============================================================================


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "loss_cls,sched_cls,sched_kwargs,loss_name",
    LOSS_CONFIGS,
    ids=_LOSS_IDS,
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=_SPATIAL_IDS,
)
class TestLossNonRegression:
    """Non-regression training loop tests for the non-weighted losses."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        loss_cls,
        sched_cls,
        sched_kwargs,
        loss_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        param_before = _first_param(model).cpu()

        losses, params = _run_training_loop(loss_fn, model, x0)

        ref_file = f"{REF_PREFIX}{loss_name}_{spatial_name}_{prediction_type}.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=_SPATIAL_IDS,
)
class TestMSEDSMLossWithPreconditioner:
    """Non-regression tests for MSEDSMLoss with EDMPreconditioner as the model.

    Tests the full pipeline: EDMPreconditioner(backbone) + EDMNoiseScheduler
    (with consistent sigma_data) + MSEDSMLoss. This verifies that the wrapping
    order and parameter alignment produce a stable training signal.
    """

    def test_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = _make_preconditioned_model(predictor_cls, predictor_kwargs).to(device)
        # sigma_data must match the preconditioner to ensure consistent noise scaling.
        scheduler = EDMNoiseScheduler(sigma_data=SIGMA_DATA)
        loss_fn = _make_loss(MSEDSMLoss, model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        param_before = _first_param(model).cpu()

        losses, params = _run_training_loop(loss_fn, model, x0)

        ref_file = f"{REF_PREFIX}precond_edm_{spatial_name}_{prediction_type}.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


# =============================================================================
# Non-Regression Tests — weighted losses (DSM and flow matching)
# =============================================================================


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "loss_cls,sched_cls,sched_kwargs,loss_name",
    WEIGHTED_LOSS_CONFIGS,
    ids=_WEIGHTED_LOSS_IDS,
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=_SPATIAL_IDS,
)
class TestWeightedLossNonRegression:
    """Non-regression training loop tests for the weighted losses."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        loss_cls,
        sched_cls,
        sched_kwargs,
        loss_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        weight = _half_masked_weight(x0, shape)
        param_before = _first_param(model).cpu()

        losses, params = _run_training_loop(loss_fn, model, x0, weight=weight)

        ref_file = f"{REF_PREFIX}{loss_name}_{spatial_name}_{prediction_type}.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=_SPATIAL_IDS,
)
class TestWeightedMSEDSMLossWithPreconditioner:
    """Non-regression tests for WeightedMSEDSMLoss with EDMPreconditioner as the model.

    Same intent as TestMSEDSMLossWithPreconditioner but exercises the weighted
    variant with a partial spatial mask.
    """

    def test_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        prediction_type,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        model = _make_preconditioned_model(predictor_cls, predictor_kwargs).to(device)
        scheduler = EDMNoiseScheduler(sigma_data=SIGMA_DATA)
        loss_fn = _make_loss(WeightedMSEDSMLoss, model, scheduler, prediction_type)

        x0 = make_input(shape, seed=GLOBAL_SEED, device=device)
        weight = _half_masked_weight(x0, shape)
        param_before = _first_param(model).cpu()

        losses, params = _run_training_loop(loss_fn, model, x0, weight=weight)

        ref_file = (
            f"{REF_PREFIX}weighted_precond_edm_{spatial_name}_{prediction_type}.pth"
        )
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through all four losses."""

    @pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
    @pytest.mark.parametrize(
        "loss_cls,sched_cls,sched_kwargs,loss_name",
        LOSS_CONFIGS,
        ids=_LOSS_IDS,
    )
    def test_gradient_flow(
        self, device, prediction_type, loss_cls, sched_cls, sched_kwargs, loss_name
    ):
        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        loss = loss_fn(x0)
        loss.backward()
        _assert_has_grad(model)

    @pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
    @pytest.mark.parametrize(
        "loss_cls,sched_cls,sched_kwargs,loss_name",
        WEIGHTED_LOSS_CONFIGS,
        ids=_WEIGHTED_LOSS_IDS,
    )
    def test_weighted_gradient_flow(
        self, device, prediction_type, loss_cls, sched_cls, sched_kwargs, loss_name
    ):
        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, :3] = 0.0
        loss = loss_fn(x0, weight=weight)
        loss.backward()
        _assert_has_grad(model)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.usefixtures("nop_compile")
@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "loss_cls,sched_cls,sched_kwargs,loss_name",
    COMPILE_LOSS_CONFIGS,
    ids=_COMPILE_LOSS_IDS,
)
class TestLossCompile:
    """Double-call compile tests for the non-weighted losses."""

    def test_compile(
        self,
        deterministic_settings,
        device,
        prediction_type,
        loss_cls,
        sched_cls,
        sched_kwargs,
        loss_name,
    ):
        """Compiled loss produces finite output and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)

        compiled_loss_fn = torch.compile(loss_fn, fullgraph=True)

        # First call — triggers tracing
        loss_1 = compiled_loss_fn(x0)
        assert loss_1.ndim == 0 and torch.isfinite(loss_1)

        # Second call — must reuse the graph
        loss_2 = compiled_loss_fn(x0)
        assert loss_2.ndim == 0 and torch.isfinite(loss_2)


@pytest.mark.usefixtures("nop_compile")
@pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
@pytest.mark.parametrize(
    "loss_cls,sched_cls,sched_kwargs,loss_name",
    WEIGHTED_COMPILE_LOSS_CONFIGS,
    ids=_WEIGHTED_COMPILE_LOSS_IDS,
)
class TestWeightedLossCompile:
    """Double-call compile tests for the weighted losses."""

    def test_compile(
        self,
        deterministic_settings,
        device,
        prediction_type,
        loss_cls,
        sched_cls,
        sched_kwargs,
        loss_name,
    ):
        """Compiled weighted loss produces finite output and graph is reused."""
        torch._dynamo.config.error_on_recompile = True

        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = _make_scheduler(sched_cls, sched_kwargs, prediction_type)
        loss_fn = _make_loss(loss_cls, model, scheduler, prediction_type)

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


# =============================================================================
# FlowMatchingLoss — Sampling Round-Trip
# =============================================================================


class TestFlowMatchingLossSamplingRoundTrip:
    """End-to-end sanity check: train briefly, then sample from the result."""

    def test_train_then_sample(self, device):
        from physicsnemo.diffusion.samplers import sample

        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = FlowMatchingLoss(model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow)

        x0 = make_input((BATCH, 3, 8, 6), seed=GLOBAL_SEED, device=device)
        for _ in range(TRAIN_STEPS):
            loss = loss_fn(x0)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= LR * p.grad
                        p.grad = None

        num_steps = 4
        t_steps = scheduler.timesteps(num_steps, device=device)
        xN = scheduler.init_latents((3, 8, 6), t_steps[0].expand(BATCH), device=device)
        denoiser = scheduler.get_denoiser(flow_predictor=model)
        with torch.no_grad():
            samples = sample(denoiser, xN, scheduler, num_steps=num_steps)
        assert samples.shape == (BATCH, 3, 8, 6)
        assert torch.isfinite(samples).all()
