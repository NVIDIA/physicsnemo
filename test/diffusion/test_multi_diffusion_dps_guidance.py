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

"""Tests for patch-local multi-diffusion DPS guidance."""

import pytest
import torch
import torch._functorch.config as _functorch_config

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionDataConsistencyDPSGuidance,
    MultiDiffusionDPSGuidance,
    MultiDiffusionDPSScorePredictor,
    MultiDiffusionModelConsistencyDPSGuidance,
    MultiDiffusionPredictor,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from .helpers import (
    compare_outputs,
    load_or_create_reference,
    make_input,
)
from .test_multi_diffusion_models import (
    BATCH,
    CHANNELS,
    IMG_H,
    IMG_W,
    PATCH_SHAPE,
    _create_md_model,
    _make_condition,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_multi_diffusion_dps_guidance_"

# P*B = 8 patch rows for the default grid; chunk_size=3 gives a ragged final chunk.
CHUNK_SIZE = 3
STATE_SHAPE = (BATCH, CHANNELS, IMG_H, IMG_W)
OBS_SHAPE = (BATCH, 1, IMG_H, IMG_W)

_COMPILE_XFAIL = (
    "Multi-chunk DPS score predictor recompiles per slice_start (a Python int "
    "Dynamo specializes on) under error_on_recompile."
)

# (config_name, description)
GUIDANCE_CONFIGS = [
    ("data_l2", "DataConsistency L2"),
    ("data_l2_sda", "DataConsistency L2 + SDA"),
    ("data_l1_sda", "DataConsistency L1 + SDA"),
    ("model_l2", "ModelConsistency L2"),
    ("model_l2_sda", "ModelConsistency L2 + SDA"),
    ("model_l1_sda", "ModelConsistency L1 + SDA"),
]

MULTI_GUIDANCE_CONFIGS = [
    ("two_data", "Two DataConsistency guidances"),
    ("data_and_model", "DataConsistency + ModelConsistency"),
]

# A guidance call can run over the full P*B batch at once (slice_start=None) or
# chunk by chunk; both must produce the same patch-local guidance term.
GUIDANCE_MODES = ["full", "chunked"]


# =============================================================================
# Helpers
# =============================================================================


def _nonlinear_obs_op(x):
    """Patch-local observation operator: select first channel and square."""
    return x[:, :1] ** 2


def _make_mask(device, points=((2, 3), (5, 10), (11, 4), (13, 12))):
    """Boolean mask (B, C, H, W) with one observed pixel per patch quadrant."""
    mask = torch.zeros(STATE_SHAPE, dtype=torch.bool, device=device)
    for h, w in points:
        mask[:, :, h, w] = True
    return mask


def _make_predictor(
    config_name="uncond",
    chunk_size=None,
    overlap_pix=0,
    boundary_pix=0,
    device="cpu",
    seed=0,
):
    """Create a grid-patched MultiDiffusionPredictor for guidance tests."""
    md = _create_md_model(config_name, seed=seed).to(device)
    md.set_grid_patching(
        patch_shape=PATCH_SHAPE,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
        fuse=True,
    )
    condition = _make_condition(config_name, device=device)
    pred = MultiDiffusionPredictor(
        md, condition=condition, fuse=True, chunk_size=chunk_size
    )
    pred.set_patching(overlap_pix=overlap_pix, boundary_pix=boundary_pix)
    return pred


def _make_guidance(config_name, predictor, device, seed=310, create_graph=False):
    """Build a single patch-local guidance object from a config name."""
    scheduler = EDMNoiseScheduler()
    if config_name == "data_l2":
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=seed, device=device)
        return MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y,
            std_y=0.1,
            create_graph=create_graph,
        )
    elif config_name == "data_l2_sda":
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=seed, device=device)
        return MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "data_l1_sda":
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=seed, device=device)
        return MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y,
            std_y=0.1,
            norm=1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "model_l2":
        y = make_input(OBS_SHAPE, seed=seed, device=device)
        return MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            create_graph=create_graph,
        )
    elif config_name == "model_l2_sda":
        y = make_input(OBS_SHAPE, seed=seed, device=device)
        return MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "model_l1_sda":
        y = make_input(OBS_SHAPE, seed=seed, device=device)
        return MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            norm=1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    raise ValueError(f"Unknown guidance config: {config_name}")


def _make_multi_guidance(
    config_name,
    predictor,
    device,
    seed=420,
    create_graph=False,
    retain_all=False,
):
    """Build a list of patch-local guidance objects for multi-guidance tests.

    When create_graph=True (for gradient flow tests), retain_graph=True is set
    on ALL guidances because loss.backward() also needs to traverse the forward
    graph after all autograd.grad calls. When retain_all=True (for compile
    tests), all guidances share the same retain_graph=True to avoid torch.compile
    guard failures from differing attribute values. Otherwise, only non-last
    guidances need retain_graph=True.
    """
    scheduler = EDMNoiseScheduler()
    retain_last = create_graph or retain_all
    if config_name == "two_data":
        mask1 = _make_mask(device, points=((2, 3), (5, 10)))
        mask2 = _make_mask(device, points=((11, 4), (13, 12)))
        y = make_input(STATE_SHAPE, seed=seed, device=device)
        g1 = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask1,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=create_graph,
        )
        g2 = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask2,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=retain_last,
            create_graph=create_graph,
        )
        return [g1, g2]
    elif config_name == "data_and_model":
        mask = _make_mask(device)
        y_data = make_input(STATE_SHAPE, seed=seed, device=device)
        g1 = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y_data,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=create_graph,
        )
        y_model = make_input(OBS_SHAPE, seed=seed + 1, device=device)
        g2 = MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y_model,
            std_y=0.1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=retain_last,
            create_graph=create_graph,
        )
        return [g1, g2]
    raise ValueError(f"Unknown multi-guidance config: {config_name}")


def _make_dps_score_predictor(predictor, guidances):
    """Build a MultiDiffusionDPSScorePredictor from a predictor and guidances."""
    scheduler = EDMNoiseScheduler()
    return MultiDiffusionDPSScorePredictor(
        x0_predictor=predictor,
        x0_to_score_fn=scheduler.x0_to_score,
        guidances=guidances,
    )


def _eval_guidance(g, predictor, device, seed=311, mode="full", chunk_size=CHUNK_SIZE):
    """Evaluate a guidance over all P*B patches, full-batch or chunk by chunk.

    Both modes return the (P*B, C, Hp, Wp) guidance term and must agree. Chunked
    mode mirrors how ``MultiDiffusionPredictor.chunks()`` feeds the guidance: an
    independent leaf per chunk with ``x_0`` recomputed from it and the chunk's
    ``slice_start`` forwarded so the guidance slices its own pre-patched data.
    """
    x_global = make_input(STATE_SHAPE, seed=seed, device=device)
    x_patched = predictor.patch_fn(x_global)
    pb = x_patched.shape[0]
    t = make_input((pb,), seed=seed + 1, device=device).abs() + 0.1
    if mode == "full":
        x = x_patched.detach().requires_grad_(True)
        x_0 = x * 0.9
        return g(x, t, x_0, slice_start=None)
    chunk_outs = []
    for s in range(0, pb, chunk_size):
        e = min(s + chunk_size, pb)
        xc = x_patched[s:e].detach().requires_grad_(True)
        x0c = xc * 0.9
        chunk_outs.append(g(xc, t[s:e], x0c, slice_start=s))
    return torch.cat(chunk_outs, dim=0)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestMultiDiffusionDPSScorePredictorConstructor:
    """Constructor contract of MultiDiffusionDPSScorePredictor."""

    def test_single_guidance_stored_as_list(self, device):
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        g = _make_guidance("data_l2", predictor, device)
        dps = _make_dps_score_predictor(predictor, g)
        assert dps.x0_predictor is predictor
        assert isinstance(dps.guidances, list)
        assert len(dps.guidances) == 1

    def test_multiple_guidances_stored_as_list(self, device):
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        guidances = _make_multi_guidance("two_data", predictor, device)
        dps = _make_dps_score_predictor(predictor, guidances)
        assert len(dps.guidances) == 2


class TestDataConsistencyConstructor:
    """Constructor contract of MultiDiffusionDataConsistencyDPSGuidance."""

    def test_default_attributes(self, device):
        predictor = _make_predictor("uncond", device=device)
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=1, device=device)
        g = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor, mask=mask, y=y, std_y=0.1
        )
        assert g.predictor is predictor
        assert g.std_y == pytest.approx(0.1)
        assert g.gamma == pytest.approx(0.0)
        assert g.fuse is False
        assert g.retain_graph is False
        assert g.create_graph is False

    def test_custom_attributes(self, device):
        scheduler = EDMNoiseScheduler()
        predictor = _make_predictor("uncond", device=device)
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=1, device=device)
        g = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y,
            std_y=0.5,
            norm=1,
            gamma=2.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            fuse=True,
            retain_graph=True,
            create_graph=True,
        )
        assert g.std_y == pytest.approx(0.5)
        assert g.gamma == pytest.approx(2.0)
        assert g.fuse is True
        assert g.retain_graph is True
        assert g.create_graph is True

    def test_satisfies_protocol(self, device):
        predictor = _make_predictor("uncond", device=device)
        g = _make_guidance("data_l2", predictor, device)
        assert isinstance(g, MultiDiffusionDPSGuidance)


class TestModelConsistencyConstructor:
    """Constructor contract of MultiDiffusionModelConsistencyDPSGuidance."""

    def test_default_attributes(self, device):
        predictor = _make_predictor("uncond", device=device)
        y = make_input(OBS_SHAPE, seed=1, device=device)
        g = MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
        )
        assert g.predictor is predictor
        assert g.std_y == pytest.approx(0.1)
        assert g.gamma == pytest.approx(0.0)
        assert g.fuse is False
        assert g.retain_graph is False
        assert g.create_graph is False

    def test_custom_attributes(self, device):
        scheduler = EDMNoiseScheduler()
        predictor = _make_predictor("uncond", device=device)
        y = make_input(OBS_SHAPE, seed=1, device=device)
        g = MultiDiffusionModelConsistencyDPSGuidance(
            predictor=predictor,
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.5,
            norm=1,
            gamma=2.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            fuse=True,
            retain_graph=True,
            create_graph=True,
        )
        assert g.std_y == pytest.approx(0.5)
        assert g.gamma == pytest.approx(2.0)
        assert g.fuse is True
        assert g.retain_graph is True
        assert g.create_graph is True

    def test_satisfies_protocol(self, device):
        predictor = _make_predictor("uncond", device=device)
        g = _make_guidance("model_l2", predictor, device)
        assert isinstance(g, MultiDiffusionDPSGuidance)


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "guidance_config,guidance_desc",
    GUIDANCE_CONFIGS,
    ids=[c[0] for c in GUIDANCE_CONFIGS],
)
class TestGuidanceNonRegression:
    """Non-regression for a direct guidance call and the guided score predictor."""

    @pytest.mark.parametrize("mode", GUIDANCE_MODES, ids=GUIDANCE_MODES)
    def test_guidance_call(
        self,
        deterministic_settings,
        device,
        tolerances,
        guidance_config,
        guidance_desc,
        mode,
    ):
        """Guidance term matches reference; full-batch and chunked modes agree.

        Both modes are compared to the same golden, so the ``chunked`` case
        doubles as the full-vs-chunked (``slice_start``) consistency check.
        """
        predictor = _make_predictor("uncond", device=device)
        g = _make_guidance(guidance_config, predictor, device, seed=310)

        out = _eval_guidance(g, predictor, device, seed=311, mode=mode)
        P = predictor._P
        assert out.shape == (P * BATCH, CHANNELS, *PATCH_SHAPE)
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}{guidance_config}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_dps_score_predictor(
        self,
        deterministic_settings,
        device,
        tolerances,
        guidance_config,
        guidance_desc,
    ):
        """Guided score at global resolution matches reference."""
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        g = _make_guidance(guidance_config, predictor, device, seed=410)
        dps = _make_dps_score_predictor(predictor, g)

        x = make_input(STATE_SHAPE, seed=411, device=device)
        t = make_input((BATCH,), seed=412, device=device).abs() + 0.1
        # Recommended inference pattern: under no_grad the predictor re-enables
        # autograd locally for the guidance gradient but returns a detached
        # score, so no graph compounds across solver steps.
        with torch.no_grad():
            out = dps(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)
        assert out.requires_grad is False
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}dps_{guidance_config}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
class TestMultiGuidanceNonRegression:
    """Non-regression for a multi-guidance MultiDiffusionDPSScorePredictor."""

    def test_dps_score_predictor(
        self,
        deterministic_settings,
        device,
        tolerances,
        multi_config,
        multi_desc,
    ):
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        guidances = _make_multi_guidance(multi_config, predictor, device, seed=420)
        dps = _make_dps_score_predictor(predictor, guidances)

        x = make_input(STATE_SHAPE, seed=421, device=device)
        t = make_input((BATCH,), seed=422, device=device).abs() + 0.1
        out = dps(x, t)
        assert out.shape == (BATCH, CHANNELS, IMG_H, IMG_W)
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}multi_{multi_config}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
class TestGradientFlow:
    """Gradients flow through MultiDiffusionDPSScorePredictor to model params."""

    def test_backward_through_dps(self, device, multi_config, multi_desc):
        """Backward through the guided score reaches the wrapped model parameters.

        Requires create_graph=True on all guidances so that autograd.grad creates
        a differentiable graph, and retain_graph=True on all guidances so that
        the subsequent loss.backward() can traverse the forward graph again.
        """
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        guidances = _make_multi_guidance(
            multi_config, predictor, device, seed=440, create_graph=True
        )
        dps = _make_dps_score_predictor(predictor, guidances)

        x = make_input(STATE_SHAPE, seed=441, device=device)
        t = make_input((BATCH,), seed=442, device=device).abs() + 0.1
        out = dps(x, t)
        out.sum().backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in predictor.model.parameters()
        )
        assert has_grad


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "guidance_config,guidance_desc",
    GUIDANCE_CONFIGS,
    ids=[c[0] for c in GUIDANCE_CONFIGS],
)
@pytest.mark.usefixtures("nop_compile")
class TestCompileSingleGuidance:
    """torch.compile tests for MultiDiffusionDPSScorePredictor, single guidance."""

    @pytest.mark.xfail(reason=_COMPILE_XFAIL, strict=False)
    def test_compile(self, device, guidance_config, guidance_desc):
        torch._dynamo.config.error_on_recompile = True

        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        g = _make_guidance(guidance_config, predictor, device, seed=500)
        dps = _make_dps_score_predictor(predictor, g)
        x = make_input(STATE_SHAPE, seed=501, device=device)
        t = make_input((BATCH,), seed=502, device=device).abs() + 0.1

        # fullgraph=False: like DPSScorePredictor, the predictor uses
        # x.detach().requires_grad_(True) and an internal autograd.grad.
        compiled_fn = torch.compile(lambda xi, ti: dps(xi, ti))

        out_eager = dps(x, t)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
@pytest.mark.usefixtures("nop_compile")
class TestCompileMultiGuidance:
    """torch.compile tests for MultiDiffusionDPSScorePredictor, multiple guidances."""

    @pytest.mark.xfail(reason=_COMPILE_XFAIL, strict=False)
    def test_compile(self, device, monkeypatch, multi_config, multi_desc):
        torch._dynamo.config.error_on_recompile = True
        monkeypatch.setattr(_functorch_config, "donated_buffer", False)

        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        guidances = _make_multi_guidance(
            multi_config, predictor, device, seed=510, retain_all=True
        )
        dps = _make_dps_score_predictor(predictor, guidances)
        x = make_input(STATE_SHAPE, seed=511, device=device)
        t = make_input((BATCH,), seed=512, device=device).abs() + 0.1

        compiled_fn = torch.compile(lambda xi, ti: dps(xi, ti))

        out_eager = dps(x, t)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2, atol=1e-4, rtol=1e-4)


# =============================================================================
# Diagnostics Tests — raised errors
# =============================================================================


class TestDiagnostics:
    """Tests for the error paths of the guided score predictor and guidances."""

    def test_requires_multi_diffusion_predictor(self, device):
        """A non-MultiDiffusionPredictor x0_predictor raises TypeError."""
        scheduler = EDMNoiseScheduler()

        def not_md(x, t):
            return x

        with pytest.raises(TypeError, match="MultiDiffusionPredictor"):
            MultiDiffusionDPSScorePredictor(
                x0_predictor=not_md,
                x0_to_score_fn=scheduler.x0_to_score,
                guidances=(lambda x, t, x_0, slice_start=None: torch.zeros_like(x_0)),
            )

    def test_requires_chunk_size(self, device):
        """A predictor without chunk_size set raises ValueError."""
        predictor = _make_predictor("uncond", chunk_size=None, device=device)
        g = _make_guidance("data_l2", predictor, device)
        scheduler = EDMNoiseScheduler()
        with pytest.raises(ValueError, match="chunk_size"):
            MultiDiffusionDPSScorePredictor(
                x0_predictor=predictor,
                x0_to_score_fn=scheduler.x0_to_score,
                guidances=g,
            )

    def test_data_consistency_gamma_requires_sigma_fn(self, device):
        predictor = _make_predictor("uncond", device=device)
        mask = _make_mask(device)
        y = make_input(STATE_SHAPE, seed=1, device=device)
        with pytest.raises(ValueError, match="sigma_fn"):
            MultiDiffusionDataConsistencyDPSGuidance(
                predictor=predictor, mask=mask, y=y, std_y=0.1, gamma=1.0
            )

    def test_model_consistency_gamma_requires_sigma_fn(self, device):
        predictor = _make_predictor("uncond", device=device)
        y = make_input(OBS_SHAPE, seed=1, device=device)
        with pytest.raises(ValueError, match="sigma_fn"):
            MultiDiffusionModelConsistencyDPSGuidance(
                predictor=predictor,
                observation_operator=_nonlinear_obs_op,
                y=y,
                std_y=0.1,
                gamma=1.0,
            )

    def test_score_predictor_inference_mode_raises(self, device):
        """The score predictor refuses to run under torch.inference_mode()."""
        predictor = _make_predictor("uncond", chunk_size=CHUNK_SIZE, device=device)
        g = _make_guidance("data_l2", predictor, device, seed=600)
        dps = _make_dps_score_predictor(predictor, g)

        x = make_input(STATE_SHAPE, seed=601, device=device)
        t = make_input((BATCH,), seed=602, device=device).abs() + 0.1
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="inference mode"):
                dps(x, t)

    def test_guidance_inference_mode_raises(self, device):
        """An individual guidance refuses to run under torch.inference_mode()."""
        predictor = _make_predictor("uncond", device=device)
        g = _make_guidance("data_l2", predictor, device, seed=600)
        x_global = make_input(STATE_SHAPE, seed=601, device=device)
        x = predictor.patch_fn(x_global).detach().requires_grad_(True)
        pb = x.shape[0]
        t = make_input((pb,), seed=602, device=device).abs() + 0.1
        x_0 = x * 0.9
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="inference mode"):
                g(x, t, x_0)
