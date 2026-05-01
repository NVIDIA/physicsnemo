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

"""Tests for MultiDiffusionPredictor."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
from physicsnemo.diffusion.multi_diffusion.predictor import MultiDiffusionPredictor

from .conftest import GLOBAL_SEED
from .helpers import (
    compare_outputs,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_input,
)
from .test_multi_diffusion_models import (
    BATCH,
    CHANNELS,
    GRID_CONFIGS,
    IMG_H,
    IMG_W,
    INPUT_SHAPE,
    MD_CONFIGS,
    PATCH_NUM,
    PATCH_SHAPE,
    _create_md_model,
    _create_md_model_edm_precond,
    _make_condition,
)

REF_PREFIX = "test_multi_diffusion_predictor_"


# =============================================================================
# Helpers
# =============================================================================


def _create_predictor(
    config_name,
    img_shape=(IMG_H, IMG_W),
    patch_shape=PATCH_SHAPE,
    overlap_pix=0,
    boundary_pix=0,
    device="cpu",
    fuse=True,
    seed=0,
):
    """Create a MultiDiffusionPredictor for the given config."""
    md = _create_md_model(config_name, img_shape=img_shape, seed=seed).to(device)
    md.set_grid_patching(
        patch_shape=patch_shape,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
        fuse=fuse,
    )
    condition = _make_condition(config_name, img_shape=img_shape, device=device)
    return MultiDiffusionPredictor(md, condition=condition, fuse=fuse)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Constructor tests covering the predictor's public contract.

    These tests deliberately avoid asserting on private attributes such as
    the pre-patched caches; the correctness of those caches is exercised end
    to end by the non-regression tests.
    """

    @pytest.mark.parametrize(
        "setup",
        ["none", "random"],
        ids=["no_patching", "random_patching"],
    )
    def test_requires_grid_patching(self, setup):
        """Predictor construction raises when grid patching is not active."""
        md = _create_md_model("uncond")
        if setup == "random":
            md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        with pytest.raises(RuntimeError, match="grid patching"):
            MultiDiffusionPredictor(md)

    @pytest.mark.parametrize("fuse", [True, False], ids=["fuse_true", "fuse_false"])
    @pytest.mark.parametrize(
        "config_name",
        [c[0] for c in MD_CONFIGS],
        ids=[c[0] for c in MD_CONFIGS],
    )
    def test_public_api(self, config_name, fuse):
        """Every MD_CONFIG constructs cleanly and exposes the documented API."""
        pred = _create_predictor(config_name, fuse=fuse)
        # .fuse property reports the value passed to the constructor
        assert pred.fuse is fuse
        # .model is the MultiDiffusionModel2D the predictor wraps
        assert isinstance(pred.model, MultiDiffusionModel2D)
        # .fuse setter round-trips and is reflected on the wrapped model
        pred.fuse = not fuse
        assert pred.fuse is (not fuse)
        assert pred.model._fuse is (not fuse)


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "config_name",
    [c[0] for c in MD_CONFIGS],
    ids=[c[0] for c in MD_CONFIGS],
)
class TestNonRegression:
    """Non-regression tests for MultiDiffusionPredictor forward pass."""

    def test_forward_fuse_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Forward with fuse=True returns (B, C, H, W) and matches reference."""
        pred = _create_predictor(config_name, device=device, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)

        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_no_fuse_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Forward with fuse=False returns (P*B, C, Hp, Wp) and matches reference."""
        pred = _create_predictor(config_name, device=device, fuse=False)
        P = pred._P
        Hp, Wp = PATCH_SHAPE

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (P * BATCH, CHANNELS, Hp, Wp)

        ref_file = f"{REF_PREFIX}{config_name}_nofuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,grid_name",
        GRID_CONFIGS,
        ids=[c[4] for c in GRID_CONFIGS],
    )
    def test_forward_grid_configs(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        grid_name,
    ):
        """Forward with various grid patching configs matches reference."""
        H, W = img_shape
        pred = _create_predictor(
            config_name,
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
            device=device,
            fuse=True,
        )

        x = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (BATCH, CHANNELS, H, W)

        ref_file = f"{REF_PREFIX}{config_name}_{grid_name}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Predictor from loaded checkpoint matches fresh-instantiation reference."""

        def create_fn():
            return _create_md_model(config_name)

        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        condition = _make_condition(config_name, device=device)
        pred = MultiDiffusionPredictor(md, condition=condition, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)

        # Reuse golden file from test_forward_fuse_non_regression
        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through the predictor."""

    def test_gradient_flow_fuse(self, device):
        pred = _create_predictor("uncond", device=device, fuse=True)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        pred(x, t).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_gradient_flow_no_fuse(self, device):
        pred = _create_predictor("uncond", device=device, fuse=False)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        pred(x, t).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_gradient_flow_conditional(self, device):
        md = _create_md_model("cond_patch").to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        cond_img = make_input(INPUT_SHAPE, seed=99, device=device).requires_grad_(True)
        condition = TensorDict({"image": cond_img}, batch_size=[BATCH])
        pred = MultiDiffusionPredictor(md, condition=condition, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        pred(x, t).sum().backward()
        assert x.grad is not None
        assert cond_img.grad is not None

    def test_gradient_flow_posembd(self, device):
        md = _create_md_model("posembd_learn").to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        condition = _make_condition("posembd_learn", device=device)
        pred = MultiDiffusionPredictor(md, condition=condition, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        pred(x, t).sum().backward()
        assert x.grad is not None
        assert md.pos_embd.grad is not None


# =============================================================================
# torch.compile Tests
# =============================================================================

COMPILE_CONFIGS = ["uncond", "cond_patch", "cond_interp", "cond_vec_img"]


@pytest.mark.parametrize("config_name", COMPILE_CONFIGS, ids=COMPILE_CONFIGS)
class TestCompile:
    """torch.compile compatibility tests for MultiDiffusionPredictor."""

    def test_compiled_forward_fuse(self, device, config_name):
        """Compiled predictor (fuse=True) matches eager; no recompile on second call."""
        torch._dynamo.config.error_on_recompile = True

        pred = _create_predictor(config_name, device=device, fuse=True)
        compiled_pred = torch.compile(pred, fullgraph=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = torch.rand(BATCH, device=device)

        out_eager = pred(x, t)
        out_compiled = compiled_pred(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        out_compiled_2 = compiled_pred(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2)

    def test_compiled_forward_no_fuse(self, device, config_name):
        """Compiled predictor (fuse=False) matches eager; no recompile on second call."""
        torch._dynamo.config.error_on_recompile = True

        pred = _create_predictor(config_name, device=device, fuse=False)
        compiled_pred = torch.compile(pred, fullgraph=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = torch.rand(BATCH, device=device)

        out_eager = pred(x, t)
        out_compiled = compiled_pred(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        out_compiled_2 = compiled_pred(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2)


# =============================================================================
# Combined Workflow Tests — EDMPreconditioner as inner model
# =============================================================================


class TestWithPreconditionedInnerModel:
    """Tests for MultiDiffusionPredictor wrapping an EDMPreconditioner."""

    def test_forward_non_regression(self, deterministic_settings, device, tolerances):
        """Forward with EDMPreconditioner inner model matches reference."""
        md = _create_md_model_edm_precond().to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        pred = MultiDiffusionPredictor(md, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)

        ref_file = f"{REF_PREFIX}edm_precond_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)
