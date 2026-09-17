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

import warnings

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
from physicsnemo.diffusion.multi_diffusion.predictor import MultiDiffusionPredictor
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

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
    PATCH_SHAPE,
    _create_md_model,
    _create_md_model_edm_precond,
    _make_condition,
)

REF_PREFIX = "test_multi_diffusion_predictor_"

CONFIGS = [c[0] for c in MD_CONFIGS] + ["edm_precond"]

# chunk_size and use_checkpointing change the memory schedule, not the output,
# so every mode must reproduce the same golden.
# (mode_id, chunk_size, use_checkpointing)
EXEC_MODES = [
    ("plain", None, False),
    ("chunked", 3, False),
    ("checkpointed", None, True),
]

PREDICTION_TYPES = ["score", "epsilon"]

COMPILE_CONFIGS = ["uncond", "cond_patch", "cond_interp", "cond_vec_img"]


# =============================================================================
# Helpers
# =============================================================================


def _create_md_for_config(config_name, img_shape=(IMG_H, IMG_W), seed=0):
    """Build the inner MultiDiffusionModel2D for a config name.

    ``"edm_precond"`` wraps an EDMPreconditioner inner model (mirroring the SDA
    recipe); every other name is a standard MD_CONFIGS entry.
    """
    if config_name == "edm_precond":
        return _create_md_model_edm_precond(img_shape=img_shape, seed=seed)
    return _create_md_model(config_name, img_shape=img_shape, seed=seed)


def _create_predictor(
    config_name,
    img_shape=(IMG_H, IMG_W),
    patch_shape=PATCH_SHAPE,
    overlap_pix=0,
    boundary_pix=0,
    device="cpu",
    fuse=True,
    seed=0,
    chunk_size=None,
    use_checkpointing=False,
    prediction_type="x0",
    scheduler=None,
):
    """Create a MultiDiffusionPredictor for the given config.

    Unified factory for every predictor test. ``config_name`` selects the inner
    model (any MD_CONFIGS name or ``"edm_precond"``); ``chunk_size`` and
    ``use_checkpointing`` toggle the memory schedule; ``prediction_type`` plus a
    ``scheduler`` wire the score/epsilon-to-x0 conversion.
    """
    md = _create_md_for_config(config_name, img_shape=img_shape, seed=seed).to(device)
    md.set_grid_patching(
        patch_shape=patch_shape,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
        fuse=fuse,
    )
    condition = (
        None
        if config_name == "edm_precond"
        else _make_condition(config_name, img_shape=img_shape, device=device)
    )
    conv_kwargs = {}
    if prediction_type == "score":
        conv_kwargs = {
            "prediction_type": "score",
            "score_to_x0_fn": scheduler.score_to_x0,
        }
    elif prediction_type == "epsilon":
        conv_kwargs = {
            "prediction_type": "epsilon",
            "epsilon_to_x0_fn": scheduler.epsilon_to_x0,
        }
    pred = MultiDiffusionPredictor(
        md,
        condition=condition,
        fuse=fuse,
        chunk_size=chunk_size,
        use_checkpointing=use_checkpointing,
        **conv_kwargs,
    )
    pred.set_patching(overlap_pix=overlap_pix, boundary_pix=boundary_pix)
    return pred


# =============================================================================
# Constructor Tests
# =============================================================================


@pytest.mark.parametrize("fuse", [True, False], ids=["fuse_true", "fuse_false"])
@pytest.mark.parametrize("config_name", CONFIGS, ids=CONFIGS)
class TestConstructor:
    """Constructor tests covering the predictor's public contract.

    These tests deliberately avoid asserting on private attributes such as the
    pre-patched caches; the correctness of those caches is exercised end to end
    by the non-regression tests.
    """

    def test_public_api(self, config_name, fuse):
        """Every config constructs cleanly and exposes the documented API."""
        pred = _create_predictor(config_name, fuse=fuse)
        # .fuse property reports the value passed to the constructor
        assert pred.fuse is fuse
        # .model is the MultiDiffusionModel2D the predictor wraps
        assert isinstance(pred.model, MultiDiffusionModel2D)
        # .patch_shape reports the saved patch geometry
        assert pred.patch_shape == PATCH_SHAPE
        # .fuse setter round-trips on the predictor
        pred.fuse = not fuse
        assert pred.fuse is (not fuse)


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize("config_name", CONFIGS, ids=CONFIGS)
class TestNonRegression:
    """Non-regression tests for the predictor's public methods.

    ``config_name`` is shared by every method and lives on the class.
    chunk_size and use_checkpointing are exercised as execution-mode
    parameterization (EXEC_MODES) on the methods that support them: since they
    only change the memory schedule, every mode must reproduce the same golden
    as the plain fused / per-patch call. ``chunks`` and ``patch_fn`` /
    ``fuse_fn`` are tested as their own methods, mirroring how ``forward`` is
    tested (fresh model and checkpoint).
    """

    @pytest.mark.parametrize(
        "mode_id,chunk_size,use_checkpointing",
        EXEC_MODES,
        ids=[m[0] for m in EXEC_MODES],
    )
    def test_forward_fuse_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        mode_id,
        chunk_size,
        use_checkpointing,
    ):
        """Forward with fuse=True returns (B, C, H, W) and matches reference."""
        pred = _create_predictor(
            config_name,
            device=device,
            fuse=True,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
        )

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)

        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    @pytest.mark.parametrize(
        "mode_id,chunk_size,use_checkpointing",
        EXEC_MODES,
        ids=[m[0] for m in EXEC_MODES],
    )
    def test_forward_no_fuse_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        mode_id,
        chunk_size,
        use_checkpointing,
    ):
        """Forward with fuse=False returns (P*B, C, Hp, Wp) and matches reference."""
        pred = _create_predictor(
            config_name,
            device=device,
            fuse=False,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
        )
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

    def test_chunks_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """chunks() per-patch outputs match the no-fuse golden, and fusing them
        matches the fused golden — implicitly enforcing chunks/__call__ agreement.
        """
        pred = _create_predictor(config_name, device=device, fuse=False, chunk_size=3)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        x0_chunks = [x0_c for _, x0_c, _, _ in pred.chunks(x, t)]
        out_patched = torch.cat(x0_chunks, dim=0)  # (P*B, C, Hp, Wp)

        # Each chunk must match its slice of the per-patch (no-fuse) golden.
        nofuse_ref = load_or_create_reference(
            f"{REF_PREFIX}{config_name}_nofuse.pth",
            lambda: {"out": out_patched.cpu()},
        )
        compare_outputs(out_patched, nofuse_ref["out"], **tolerances)

        # Fusing the chunks must match the fused golden (== fused __call__).
        out_fused = pred.fuse_fn(out_patched)
        fuse_ref = load_or_create_reference(
            f"{REF_PREFIX}{config_name}_fuse.pth",
            lambda: {"out": out_fused.cpu()},
        )
        compare_outputs(out_fused, fuse_ref["out"], **tolerances)

    def test_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Predictor from loaded checkpoint matches fresh-instantiation reference."""
        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(
            ckpt_file, lambda: _create_md_for_config(config_name)
        ).to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        condition = (
            None
            if config_name == "edm_precond"
            else _make_condition(config_name, device=device)
        )
        pred = MultiDiffusionPredictor(md, condition=condition, fuse=True)
        pred.set_patching(overlap_pix=0, boundary_pix=0)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)

        # Reuse golden file from test_forward_fuse_non_regression
        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_chunks_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """chunks() from a loaded checkpoint matches the fused forward reference."""
        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(
            ckpt_file, lambda: _create_md_for_config(config_name)
        ).to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=False)
        condition = (
            None
            if config_name == "edm_precond"
            else _make_condition(config_name, device=device)
        )
        pred = MultiDiffusionPredictor(
            md, condition=condition, fuse=False, chunk_size=3
        )
        pred.set_patching(overlap_pix=0, boundary_pix=0)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        x0_chunks = [x0_c for _, x0_c, _, _ in pred.chunks(x, t)]
        out = pred.fuse_fn(torch.cat(x0_chunks, dim=0))

        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    @pytest.mark.parametrize("prediction_type", PREDICTION_TYPES, ids=PREDICTION_TYPES)
    def test_forward_prediction_type(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        prediction_type,
    ):
        """score / epsilon prediction_type applies the conversion to x0."""
        scheduler = EDMNoiseScheduler()
        pred = _create_predictor(
            config_name,
            device=device,
            fuse=True,
            prediction_type=prediction_type,
            scheduler=scheduler,
        )

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = pred(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)

        ref_file = f"{REF_PREFIX}{config_name}_pred_{prediction_type}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_patch_fn(self, deterministic_settings, device, tolerances, config_name):
        """patch_fn output matches reference."""
        pred = _create_predictor(config_name, device=device)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        out = pred.patch_fn(x)
        P = pred._P
        assert out.shape == (P * BATCH, CHANNELS, *PATCH_SHAPE)
        ref_file = f"{REF_PREFIX}{config_name}_patch_fn.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize("config_name", CONFIGS, ids=CONFIGS)
class TestConsistency:
    """Equivalent code paths produce identical results."""

    def test_fuse_fn_roundtrip(self, device, config_name):
        """fuse_fn(patch_fn(x)) reconstructs x on a no-overlap grid."""
        pred = _create_predictor(
            config_name, device=device, overlap_pix=0, boundary_pix=0
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        recon = pred.fuse_fn(pred.patch_fn(x))
        torch.testing.assert_close(recon, x, atol=1e-4, rtol=1e-4)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through the predictor's public methods.

    These methods verify specific gradient paths (a learnable condition, the
    positional embedding) and so use the config each path requires; there is no
    single shared parameterization to hoist to the class.
    """

    @pytest.mark.parametrize(
        "mode_id,chunk_size,use_checkpointing",
        EXEC_MODES,
        ids=[m[0] for m in EXEC_MODES],
    )
    def test_gradient_flow_fuse(self, device, mode_id, chunk_size, use_checkpointing):
        """Gradient flows through forward (fuse) for every execution mode."""
        pred = _create_predictor(
            "uncond",
            device=device,
            fuse=True,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
        )
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
        pred.set_patching(overlap_pix=0, boundary_pix=0)

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
        pred.set_patching(overlap_pix=0, boundary_pix=0)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        pred(x, t).sum().backward()
        assert x.grad is not None
        assert md.pos_embd.grad is not None

    def test_chunks_gradient_flow(self, device):
        """Gradient flows through a chunks() reconstruct + fuse."""
        pred = _create_predictor("uncond", device=device, fuse=False, chunk_size=3)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        x0_chunks = [x0_c for _, x0_c, _, _ in pred.chunks(x, t)]
        pred.fuse_fn(torch.cat(x0_chunks, dim=0)).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_patch_fuse_gradient_flow(self, device):
        """Gradient flows through patch_fn and fuse_fn."""
        pred = _create_predictor("uncond", device=device)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        pred.fuse_fn(pred.patch_fn(x)).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()


# =============================================================================
# torch.compile Tests
# =============================================================================


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

    def test_compiled_chunks(self, device, config_name):
        """Compiled chunks() reconstruct matches eager; no recompile on second call."""
        torch._dynamo.config.error_on_recompile = True

        pred = _create_predictor(config_name, device=device, fuse=False, chunk_size=3)

        def reconstruct(xi, ti):
            x0_chunks = [x0_c for _, x0_c, _, _ in pred.chunks(xi, ti)]
            return pred.fuse_fn(torch.cat(x0_chunks, dim=0))

        compiled_fn = torch.compile(reconstruct, fullgraph=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = torch.rand(BATCH, device=device)

        out_eager = reconstruct(x, t)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2)

    def test_compiled_patch_fuse(self, device, config_name):
        """Compiled patch_fn / fuse_fn round-trip matches eager."""
        torch._dynamo.config.error_on_recompile = True

        pred = _create_predictor(config_name, device=device)

        def roundtrip(xi):
            return pred.fuse_fn(pred.patch_fn(xi))

        compiled_fn = torch.compile(roundtrip, fullgraph=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        torch.testing.assert_close(roundtrip(x), compiled_fn(x))
        torch.testing.assert_close(roundtrip(x), compiled_fn(x))


# =============================================================================
# Diagnostics Tests — raised errors and warnings
# =============================================================================


class TestDiagnostics:
    """Tests for the predictor's error and warning paths."""

    def test_set_patching_requires_patch_shape(self):
        """set_patching raises when no patch_shape is available or provided."""
        md = _create_md_model("uncond")
        pred = MultiDiffusionPredictor(md)
        with pytest.raises(RuntimeError, match="patch_shape"):
            pred.set_patching(overlap_pix=0, boundary_pix=0)

    def test_call_requires_set_patching(self):
        """__call__ raises when set_patching has not been called."""
        md = _create_md_model("uncond")
        pred = MultiDiffusionPredictor(md)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED)
        t = torch.rand(BATCH)
        with pytest.raises(RuntimeError, match="set_patching"):
            pred(x, t)

    def test_patch_fn_requires_set_patching(self, device):
        """patch_fn raises when set_patching has not been called."""
        md = _create_md_model("uncond").to(device)
        pred = MultiDiffusionPredictor(md)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        with pytest.raises(RuntimeError, match="set_patching"):
            pred.patch_fn(x)

    def test_chunks_requires_chunk_size(self, device):
        """chunks() raises when chunk_size was not set at construction."""
        pred = _create_predictor("uncond", device=device)  # chunk_size=None
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        with pytest.raises(RuntimeError, match="chunk_size"):
            next(pred.chunks(x, t))

    def test_fuse_fn_requires_divisible(self, device):
        """fuse_fn rejects a tensor whose row count is not a multiple of P."""
        pred = _create_predictor("uncond", device=device)
        P = pred._P
        bad = make_input((P * BATCH + 1, CHANNELS, *PATCH_SHAPE), seed=1, device=device)
        with pytest.raises(ValueError, match="divisible"):
            pred.fuse_fn(bad)

    def test_score_requires_conversion_fn(self):
        """score prediction_type without score_to_x0_fn raises."""
        md = _create_md_model("uncond")
        with pytest.raises(ValueError, match="score_to_x0_fn"):
            MultiDiffusionPredictor(md, prediction_type="score")

    def test_epsilon_requires_conversion_fn(self):
        """epsilon prediction_type without epsilon_to_x0_fn raises."""
        md = _create_md_model("uncond")
        with pytest.raises(ValueError, match="epsilon_to_x0_fn"):
            MultiDiffusionPredictor(md, prediction_type="epsilon")

    def test_invalid_prediction_type(self):
        """An unknown prediction_type raises."""
        md = _create_md_model("uncond")
        with pytest.raises(ValueError, match="prediction_type"):
            MultiDiffusionPredictor(md, prediction_type="bogus")

    def test_patch_shape_override_warns(self, device):
        """Overriding the saved patch_shape emits a warning."""
        md = _create_md_model("uncond").to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        pred = MultiDiffusionPredictor(md)
        with pytest.warns(UserWarning, match="patch_shape"):
            pred.set_patching(overlap_pix=0, boundary_pix=0, patch_shape=(4, 4))

    def test_global_shape_override_warns(self, device):
        """Overriding the saved global_spatial_shape emits a warning."""
        md = _create_md_model("uncond").to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        pred = MultiDiffusionPredictor(md)
        with pytest.warns(UserWarning, match="global"):
            pred.set_patching(overlap_pix=0, boundary_pix=0, global_shape=(8, 8))

    def test_matching_overrides_do_not_warn(self, device):
        """Passing the saved geometry explicitly must not warn."""
        md = _create_md_model("uncond").to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)
        pred = MultiDiffusionPredictor(md)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            pred.set_patching(
                overlap_pix=0,
                boundary_pix=0,
                patch_shape=PATCH_SHAPE,
                global_shape=(IMG_H, IMG_W),
            )
        override_warnings = [w for w in records if "Overriding saved" in str(w.message)]
        assert not override_warnings
