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

"""Tests for multi-diffusion losses."""

import pytest
import torch

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionFlowMatchingLoss,
    MultiDiffusionModel2D,
    MultiDiffusionMSEDSMLoss,
    MultiDiffusionWeightedFlowMatchingLoss,
    MultiDiffusionWeightedMSEDSMLoss,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    RectifiedFlowNoiseScheduler,
)
from physicsnemo.diffusion.preconditioners import EDMPreconditioner

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)
from .test_multi_diffusion_models import (
    BATCH,
    CHANNELS,
    IMG_H,
    IMG_H_NS,
    IMG_W,
    IMG_W_NS,
    INPUT_SHAPE,
    PATCH_NUM,
    PATCH_SHAPE,
    PATCH_SHAPE_NS,
    _create_md_model,
    _make_condition,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_multi_diffusion_losses_"
LR = 1e-2
TRAIN_STEPS = 2

# sigma_data must be consistent between EDMPreconditioner and EDMNoiseScheduler
# to mirror the realistic SDA recipe pattern.
SIGMA_DATA = 1.0

LOSS_FAMILIES = [
    (
        MultiDiffusionMSEDSMLoss,
        EDMNoiseScheduler,
        "x0",
        "mse",
    ),
    (
        MultiDiffusionFlowMatchingLoss,
        RectifiedFlowNoiseScheduler,
        "flow",
        "fm",
    ),
]

WEIGHTED_LOSS_FAMILIES = [
    (
        MultiDiffusionWeightedMSEDSMLoss,
        EDMNoiseScheduler,
        "x0",
        "wmse",
    ),
    (
        MultiDiffusionWeightedFlowMatchingLoss,
        RectifiedFlowNoiseScheduler,
        "flow",
        "wfm",
    ),
]

FLOW_LOSS_CLASSES = [
    MultiDiffusionFlowMatchingLoss,
    MultiDiffusionWeightedFlowMatchingLoss,
]

# (config_name, prediction_type, img_shape, patch_shape, tag)
LOSS_CONFIGS = [
    ("uncond", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_x0_sq"),
    ("cond_patch", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "cond_patch_x0_sq"),
    ("cond_vec_img", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "cond_vec_x0_sq"),
    ("posembd_learn", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "posembd_x0_sq"),
    ("uncond", "score", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_score_sq"),
    ("cond_patch", "score", (IMG_H, IMG_W), PATCH_SHAPE, "cond_patch_score_sq"),
    ("uncond", "epsilon", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_eps_sq"),
    ("cond_patch", "epsilon", (IMG_H, IMG_W), PATCH_SHAPE, "cond_patch_eps_sq"),
    ("uncond", "flow", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_flow_sq"),
    ("cond_patch", "flow", (IMG_H, IMG_W), PATCH_SHAPE, "cond_patch_flow_sq"),
    ("uncond", "x0", (IMG_H_NS, IMG_W_NS), PATCH_SHAPE_NS, "uncond_x0_ns"),
    ("cond_patch", "x0", (IMG_H_NS, IMG_W_NS), PATCH_SHAPE_NS, "cond_patch_x0_ns"),
]

COMPILE_LOSS_CONFIGS = [
    ("uncond", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_x0_sq"),
    ("cond_patch", "x0", (IMG_H, IMG_W), PATCH_SHAPE, "cond_patch_x0_sq"),
    ("uncond", "score", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_score_sq"),
    ("uncond", "epsilon", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_eps_sq"),
    ("uncond", "flow", (IMG_H, IMG_W), PATCH_SHAPE, "uncond_flow_sq"),
]

_LOSS_IDS = [c[4] for c in LOSS_CONFIGS]
_COMPILE_LOSS_IDS = [c[4] for c in COMPILE_LOSS_CONFIGS]
_LOSS_FAMILY_IDS = [c[3] for c in LOSS_FAMILIES]
_WEIGHTED_LOSS_FAMILY_IDS = [c[3] for c in WEIGHTED_LOSS_FAMILIES]


# =============================================================================
# Helpers
# =============================================================================


def _first_param(model: MultiDiffusionModel2D) -> torch.Tensor:
    """Return a clone of the first parameter of the wrapped inner model."""
    return next(model.model.parameters()).detach().clone()


def _make_scheduler(scheduler_cls, prediction_type):
    """Instantiate a scheduler for the requested prediction type."""
    kwargs = {}
    if scheduler_cls is RectifiedFlowNoiseScheduler and prediction_type == "x0":
        kwargs["t_min"] = 1e-3
    return scheduler_cls(**kwargs)


def _make_loss(loss_cls, md, scheduler, prediction_type, target_type):
    """Create a configured loss with the requested prediction type."""
    kwargs = {}
    if target_type == "flow":
        kwargs["x0_to_flow_fn"] = scheduler.x0_to_flow
        if prediction_type == "score":
            kwargs["score_to_flow_fn"] = scheduler.score_to_flow
        elif prediction_type == "epsilon":
            kwargs["epsilon_to_flow_fn"] = lambda eps, x_t, t: scheduler.x0_to_flow(
                scheduler.epsilon_to_x0(eps, x_t, t), x_t, t
            )
    else:
        if prediction_type == "score":
            kwargs["score_to_x0_fn"] = scheduler.score_to_x0
        elif prediction_type == "epsilon":
            kwargs["epsilon_to_x0_fn"] = scheduler.epsilon_to_x0
        elif prediction_type == "flow":
            kwargs["flow_to_x0_fn"] = scheduler.flow_to_x0
    return loss_cls(md, scheduler, prediction_type=prediction_type, **kwargs)


def _run_training_loop(
    loss_fn, md_model, x0, condition, weight=None, steps=TRAIN_STEPS
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
            for p in md_model.parameters():
                if p.grad is not None:
                    p -= LR * p.grad
                    p.grad = None
        params.append(_first_param(md_model).cpu())
    return losses, params


def _check_non_regression(losses, params, param_before, ref_file, device, tolerances):
    """Assert finite training-loop invariants and compare against goldens.

    On CUDA, the noise scheduler's internal RNG (sample_time, add_noise)
    produces a different random stream than on CPU even with the same seed,
    so only shapes and finiteness get verified there. Full value comparison
    happens on CPU only.
    """
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


def _create_preconditioned_md_model(seed=0):
    """Full realistic pipeline: EDMPreconditioner(backbone) inside MultiDiffusionModel2D.

    sigma_data stays consistent between the preconditioner and the
    EDMNoiseScheduler used in the loss, mirroring the SDA recipe pattern.
    """
    backbone = instantiate_model_deterministic(
        Conv2dX0Predictor, seed=seed, channels=CHANNELS
    )
    precond = EDMPreconditioner(backbone, sigma_data=SIGMA_DATA)
    return MultiDiffusionModel2D(model=precond, global_spatial_shape=(IMG_H, IMG_W))


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Tests for loss constructor and public attributes."""

    def test_mse_constructor(self):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = EDMNoiseScheduler()
        loss_fn = MultiDiffusionMSEDSMLoss(md, scheduler)
        assert loss_fn.model is md
        assert loss_fn.noise_scheduler is scheduler

    def test_weighted_mse_constructor(self):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = EDMNoiseScheduler()
        loss_fn = MultiDiffusionWeightedMSEDSMLoss(md, scheduler)
        assert loss_fn.model is md
        assert loss_fn.noise_scheduler is scheduler

    @pytest.mark.parametrize("loss_cls", FLOW_LOSS_CLASSES, ids=["fm", "wfm"])
    def test_flow_constructor(self, loss_cls):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = RectifiedFlowNoiseScheduler()
        loss_fn = loss_cls(md, scheduler, x0_to_flow_fn=scheduler.x0_to_flow)
        assert loss_fn.model is md
        assert loss_fn.noise_scheduler is scheduler

    def test_invalid_prediction_type(self):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        with pytest.raises(ValueError, match="prediction_type"):
            MultiDiffusionMSEDSMLoss(md, EDMNoiseScheduler(), prediction_type="bad")

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
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        with pytest.raises(ValueError, match=missing_fn):
            MultiDiffusionMSEDSMLoss(
                md, EDMNoiseScheduler(), prediction_type=prediction_type
            )

    @pytest.mark.parametrize("loss_cls", FLOW_LOSS_CLASSES, ids=["fm", "wfm"])
    def test_flow_requires_x0_to_flow_fn(self, loss_cls):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        with pytest.raises(ValueError, match="x0_to_flow_fn"):
            loss_cls(md, RectifiedFlowNoiseScheduler())

    @pytest.mark.parametrize("loss_cls", FLOW_LOSS_CLASSES, ids=["fm", "wfm"])
    @pytest.mark.parametrize(
        "prediction_type,missing_fn",
        [
            ("score", "score_to_flow_fn"),
            ("epsilon", "epsilon_to_flow_fn"),
        ],
    )
    def test_flow_requires_conversion_fn(self, loss_cls, prediction_type, missing_fn):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match=missing_fn):
            loss_cls(
                md,
                scheduler,
                prediction_type=prediction_type,
                x0_to_flow_fn=scheduler.x0_to_flow,
            )

    @pytest.mark.parametrize("loss_cls", FLOW_LOSS_CLASSES, ids=["fm", "wfm"])
    def test_flow_invalid_prediction_type(self, loss_cls):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = RectifiedFlowNoiseScheduler()
        with pytest.raises(ValueError, match="prediction_type"):
            loss_cls(
                md,
                scheduler,
                prediction_type="bad",
                x0_to_flow_fn=scheduler.x0_to_flow,
            )

    def test_epsilon_constructor(self):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = EDMNoiseScheduler()
        loss_fn = MultiDiffusionMSEDSMLoss(
            md,
            scheduler,
            prediction_type="epsilon",
            epsilon_to_x0_fn=scheduler.epsilon_to_x0,
        )
        assert loss_fn.model is md

    @pytest.mark.parametrize(
        "loss_cls,scheduler_cls,target_type,loss_name",
        LOSS_FAMILIES + WEIGHTED_LOSS_FAMILIES,
        ids=_LOSS_FAMILY_IDS + _WEIGHTED_LOSS_FAMILY_IDS,
    )
    def test_invalid_reduction(self, loss_cls, scheduler_cls, target_type, loss_name):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = scheduler_cls()
        kwargs = (
            {"x0_to_flow_fn": scheduler.x0_to_flow} if target_type == "flow" else {}
        )
        with pytest.raises(ValueError, match="reduction"):
            loss_cls(md, scheduler, reduction="bad", **kwargs)

    @pytest.mark.parametrize(
        "loss_cls,scheduler_cls,target_type,loss_name,weighted",
        [(*config, False) for config in LOSS_FAMILIES]
        + [(*config, True) for config in WEIGHTED_LOSS_FAMILIES],
        ids=_LOSS_FAMILY_IDS + _WEIGHTED_LOSS_FAMILY_IDS,
    )
    def test_no_patching_raises(
        self,
        loss_cls,
        scheduler_cls,
        target_type,
        loss_name,
        weighted,
    ):
        """Calling a loss without setting a patching strategy must fail."""
        md = _create_md_model("uncond")
        scheduler = _make_scheduler(scheduler_cls, target_type)
        loss_fn = _make_loss(
            loss_cls,
            md,
            scheduler,
            prediction_type=target_type,
            target_type=target_type,
        )
        x0 = torch.randn(*INPUT_SHAPE)
        kwargs = {"weight": torch.ones_like(x0)} if weighted else {}
        with pytest.raises(RuntimeError, match="patching"):
            loss_fn(x0, **kwargs)


# =============================================================================
# Non-Regression Tests: Unweighted Losses
# =============================================================================


@pytest.mark.parametrize(
    "config_name,prediction_type,img_shape,patch_shape,tag",
    LOSS_CONFIGS,
    ids=_LOSS_IDS,
)
@pytest.mark.parametrize(
    "loss_cls,scheduler_cls,target_type,loss_name",
    LOSS_FAMILIES,
    ids=_LOSS_FAMILY_IDS,
)
class TestLossNonRegression:
    """Non-regression training loop tests for unweighted losses."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        prediction_type,
        img_shape,
        patch_shape,
        tag,
        loss_cls,
        scheduler_cls,
        target_type,
        loss_name,
    ):
        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_random_patching(patch_shape=patch_shape, patch_num=PATCH_NUM)
        scheduler = _make_scheduler(scheduler_cls, prediction_type)
        loss_fn = _make_loss(loss_cls, md, scheduler, prediction_type, target_type)

        H, W = img_shape
        x0 = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        condition = _make_condition(config_name, img_shape=img_shape, device=device)
        param_before = _first_param(md).cpu()

        losses, params = _run_training_loop(loss_fn, md, x0, condition)

        ref_file = f"{REF_PREFIX}{loss_name}_{tag}_train.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


# =============================================================================
# Non-Regression Tests: Weighted Losses
# =============================================================================


@pytest.mark.parametrize(
    "config_name,prediction_type,img_shape,patch_shape,tag",
    LOSS_CONFIGS,
    ids=_LOSS_IDS,
)
@pytest.mark.parametrize(
    "loss_cls,scheduler_cls,target_type,loss_name",
    WEIGHTED_LOSS_FAMILIES,
    ids=_WEIGHTED_LOSS_FAMILY_IDS,
)
class TestWeightedLossNonRegression:
    """Non-regression training loop tests for weighted losses."""

    def test_training_loop(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        prediction_type,
        img_shape,
        patch_shape,
        tag,
        loss_cls,
        scheduler_cls,
        target_type,
        loss_name,
    ):
        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_random_patching(patch_shape=patch_shape, patch_num=PATCH_NUM)
        scheduler = _make_scheduler(scheduler_cls, prediction_type)
        loss_fn = _make_loss(loss_cls, md, scheduler, prediction_type, target_type)

        H, W = img_shape
        x0 = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, : W // 2] = 0.0
        condition = _make_condition(config_name, img_shape=img_shape, device=device)
        param_before = _first_param(md).cpu()

        losses, params = _run_training_loop(loss_fn, md, x0, condition, weight=weight)

        ref_file = f"{REF_PREFIX}{loss_name}_{tag}_train.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.usefixtures("nop_compile")
@pytest.mark.parametrize(
    "config_name,prediction_type,img_shape,patch_shape,tag",
    COMPILE_LOSS_CONFIGS,
    ids=_COMPILE_LOSS_IDS,
)
@pytest.mark.parametrize(
    "loss_cls,scheduler_cls,target_type,loss_name",
    LOSS_FAMILIES,
    ids=_LOSS_FAMILY_IDS,
)
class TestLossCompile:
    """Verify internal _CompiledPatchX compilation is reused across calls."""

    def test_internal_compile_no_recompile(
        self,
        deterministic_settings,
        device,
        config_name,
        prediction_type,
        img_shape,
        patch_shape,
        tag,
        loss_cls,
        scheduler_cls,
        target_type,
        loss_name,
    ):
        """The internally compiled patch_x graph is reused across patch resets."""
        torch._dynamo.config.error_on_recompile = True

        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_random_patching(patch_shape=patch_shape, patch_num=PATCH_NUM)
        scheduler = _make_scheduler(scheduler_cls, prediction_type)
        loss_fn = _make_loss(loss_cls, md, scheduler, prediction_type, target_type)

        H, W = img_shape
        x0 = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        condition = _make_condition(config_name, img_shape=img_shape, device=device)

        # First call triggers internal _CompiledPatchX compilation
        loss_1 = loss_fn(x0, condition=condition)
        assert loss_1.ndim == 0 and torch.isfinite(loss_1)

        # Patch resets must reuse the compiled graph
        loss_2 = loss_fn(x0, condition=condition)
        assert loss_2.ndim == 0 and torch.isfinite(loss_2)


@pytest.mark.usefixtures("nop_compile")
@pytest.mark.parametrize(
    "config_name,prediction_type,img_shape,patch_shape,tag",
    COMPILE_LOSS_CONFIGS,
    ids=_COMPILE_LOSS_IDS,
)
@pytest.mark.parametrize(
    "loss_cls,scheduler_cls,target_type,loss_name",
    WEIGHTED_LOSS_FAMILIES,
    ids=_WEIGHTED_LOSS_FAMILY_IDS,
)
class TestWeightedLossCompile:
    """Verify internal _CompiledPatchX compilation is reused across calls."""

    def test_internal_compile_no_recompile(
        self,
        deterministic_settings,
        device,
        config_name,
        prediction_type,
        img_shape,
        patch_shape,
        tag,
        loss_cls,
        scheduler_cls,
        target_type,
        loss_name,
    ):
        """The internally compiled patch_x graph is reused across patch resets."""
        torch._dynamo.config.error_on_recompile = True

        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_random_patching(patch_shape=patch_shape, patch_num=PATCH_NUM)
        scheduler = _make_scheduler(scheduler_cls, prediction_type)
        loss_fn = _make_loss(loss_cls, md, scheduler, prediction_type, target_type)

        H, W = img_shape
        x0 = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, : W // 2] = 0.0
        condition = _make_condition(config_name, img_shape=img_shape, device=device)

        # First call triggers internal _CompiledPatchX compilation
        loss_1 = loss_fn(x0, weight=weight, condition=condition)
        assert loss_1.ndim == 0 and torch.isfinite(loss_1)

        # Patch resets must reuse the compiled graph
        loss_2 = loss_fn(x0, weight=weight, condition=condition)
        assert loss_2.ndim == 0 and torch.isfinite(loss_2)


# =============================================================================
# Combined Workflow Tests: EDMPreconditioner inside MultiDiffusionModel2D
# =============================================================================


class TestMSEDSMLossWithPreconditionedInnerModel:
    """Non-regression tests for MultiDiffusionMSEDSMLoss with a preconditioned inner model.

    Verifies the critical wrapping order: EDMPreconditioner is applied *inside*
    MultiDiffusionModel2D, which is the pattern used in the realistic SDA recipe.
    sigma_data is consistently set in both the preconditioner and the scheduler.
    """

    def test_non_regression(self, deterministic_settings, device, tolerances):
        md = _create_preconditioned_md_model().to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        # sigma_data must match EDMPreconditioner to ensure consistent noise scaling.
        scheduler = EDMNoiseScheduler(sigma_data=SIGMA_DATA)
        loss_fn = MultiDiffusionMSEDSMLoss(md, scheduler)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        param_before = _first_param(md).cpu()

        losses, params = _run_training_loop(loss_fn, md, x0, condition=None)

        ref_file = f"{REF_PREFIX}precond_edm_train.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )


class TestWeightedMSEDSMLossWithPreconditionedInnerModel:
    """Non-regression tests for MultiDiffusionWeightedMSEDSMLoss with a preconditioned inner model.

    Same intent as TestMSEDSMLossWithPreconditionedInnerModel but with a
    spatial weight mask.
    """

    def test_non_regression(self, deterministic_settings, device, tolerances):
        md = _create_preconditioned_md_model().to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        scheduler = EDMNoiseScheduler(sigma_data=SIGMA_DATA)
        loss_fn = MultiDiffusionWeightedMSEDSMLoss(md, scheduler)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        weight = torch.ones_like(x0)
        weight[:, :, :, : IMG_W // 2] = 0.0
        param_before = _first_param(md).cpu()

        losses, params = _run_training_loop(
            loss_fn, md, x0, condition=None, weight=weight
        )

        ref_file = f"{REF_PREFIX}weighted_precond_edm_train.pth"
        _check_non_regression(
            losses, params, param_before, ref_file, device, tolerances
        )
