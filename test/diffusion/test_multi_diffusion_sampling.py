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

"""End-to-end sampling tests for MultiDiffusionPredictor with sample()."""

import pytest
import torch

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionDataConsistencyDPSGuidance,
    MultiDiffusionDPSScorePredictor,
    MultiDiffusionModelConsistencyDPSGuidance,
    MultiDiffusionPredictor,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample

from .helpers import (
    compare_outputs,
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
    PATCH_SHAPE,
    PATCH_SHAPE_NS,
    _create_md_model,
    _make_condition,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_multi_diffusion_sampling_"
NUM_STEPS = 4
NUM_STEPS_SHORT = 2
GUIDED_CHUNK_SIZE = 3

SAMPLER_CPU_TOLERANCES = {"atol": 20.0, "rtol": 5e-2}

# Common sampling config, shared by plain and guided methods.
# (md_config, img_shape, patch_shape, overlap_pix, boundary_pix, config_tag)
SAMPLE_CONFIGS = [
    ("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "uncond_sq_nooverlap"),
    ("cond_patch", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "cond_patch_sq_nooverlap"),
    ("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 2, 0, "uncond_sq_overlap2"),
    ("uncond", (IMG_H_NS, IMG_W_NS), PATCH_SHAPE_NS, 0, 0, "uncond_ns_nooverlap"),
    ("posembd_sin", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "posembd_sin_sq_nooverlap"),
]

SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VENoiseScheduler, {}, "ve"),
    (VPNoiseScheduler, {}, "vp"),
]

# Deterministic solvers only (no stochastic churn)
SOLVER_CONFIGS = [
    ("euler", None, "euler"),
    ("heun", None, "heun"),
]

# Guided sub-parameterization (patch-local DPS guidance), crossed with the
# common config only in the guided methods.
GUIDANCE_CONFIGS = ["data_l2", "model_l2_sda"]

# Common config for compile (kept small) and for gradient flow (single).
COMPILE_SAMPLE_CONFIGS = [
    ("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, EDMNoiseScheduler, {}, "euler", None),
    (
        "cond_patch",
        (IMG_H, IMG_W),
        PATCH_SHAPE,
        0,
        0,
        VENoiseScheduler,
        {},
        "heun",
        None,
    ),
]
GRAD_SAMPLE_CONFIGS = [
    ("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, EDMNoiseScheduler, {}, "euler", None),
]

_GUIDED_COMPILE_XFAIL = (
    "Guided DPS sampling does not compile fullgraph: the guidance runs an "
    "internal autograd.grad and the per-chunk slice_start (a Python int Dynamo "
    "specializes on) forces recompiles."
)


# =============================================================================
# Helpers
# =============================================================================


def _torch_version_ge_2_10() -> bool:
    """True when the installed torch version is >= 2.10."""
    parts = torch.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 10)


def _make_sampling_components(
    md_config,
    img_shape,
    patch_shape,
    overlap_pix,
    boundary_pix,
    sched_cls,
    sched_kwargs,
    device,
    num_steps=NUM_STEPS,
    guidance_config=None,
    chunk_size=GUIDED_CHUNK_SIZE,
    create_graph=False,
    retain_graph=False,
):
    """Create scheduler, md model, predictor, denoiser, and initial latent.

    With ``guidance_config=None`` the denoiser wraps the plain x0-predictor.
    Otherwise a patch-local DPS guidance is built and the denoiser wraps a
    MultiDiffusionDPSScorePredictor (the predictor then needs ``chunk_size``).
    """
    scheduler = sched_cls(**sched_kwargs)
    md = _create_md_model(md_config, img_shape=img_shape).to(device)
    md.set_grid_patching(
        patch_shape=patch_shape,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
        fuse=True,
    )
    condition = _make_condition(md_config, img_shape=img_shape, device=device)

    if guidance_config is None:
        predictor = MultiDiffusionPredictor(md, condition=condition, fuse=True)
        predictor.set_patching(overlap_pix=overlap_pix, boundary_pix=boundary_pix)
        denoiser = scheduler.get_denoiser(x0_predictor=predictor)
    else:
        predictor = MultiDiffusionPredictor(
            md, condition=condition, fuse=True, chunk_size=chunk_size
        )
        predictor.set_patching(overlap_pix=overlap_pix, boundary_pix=boundary_pix)
        H, W = img_shape
        if guidance_config == "data_l2":
            mask = torch.zeros((BATCH, CHANNELS, H, W), dtype=torch.bool, device=device)
            mask[:, :, 2, 3] = True
            mask[:, :, H - 2, W - 2] = True
            y = make_input((BATCH, CHANNELS, H, W), seed=300, device=device)
            guidance = MultiDiffusionDataConsistencyDPSGuidance(
                predictor=predictor,
                mask=mask,
                y=y,
                std_y=0.1,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )
        elif guidance_config == "model_l2_sda":
            y = make_input((BATCH, 1, H, W), seed=300, device=device)
            guidance = MultiDiffusionModelConsistencyDPSGuidance(
                predictor=predictor,
                observation_operator=lambda xx: xx[:, :1],
                y=y,
                std_y=0.1,
                gamma=0.5,
                sigma_fn=scheduler.sigma,
                alpha_fn=scheduler.alpha,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )
        else:
            raise ValueError(f"Unknown guidance config: {guidance_config}")
        dps = MultiDiffusionDPSScorePredictor(
            x0_predictor=predictor,
            x0_to_score_fn=scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = scheduler.get_denoiser(score_predictor=dps, denoising_type="ode")

    H, W = img_shape
    shape = (BATCH, CHANNELS, H, W)
    t_steps = scheduler.timesteps(num_steps, device=device)
    tN = t_steps[0].expand(shape[0])
    xN = make_input(shape, seed=200, device=device) * tN.view(-1, 1, 1, 1)

    return scheduler, md, predictor, denoiser, xN


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_key,solver_options,solver_name",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,config_tag",
    SAMPLE_CONFIGS,
    ids=[c[5] for c in SAMPLE_CONFIGS],
)
class TestMultiDiffusionSampleNonRegression:
    """Non-regression tests for sample() (plain x0 and guided DPS paths)."""

    def test_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_tag,
        sched_cls,
        sched_kwargs,
        sched_name,
        solver_key,
        solver_options,
        solver_name,
    ):
        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
        )
        H, W = img_shape
        shape = (BATCH, CHANNELS, H, W)

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS,
            solver=solver_key,
            solver_options=solver_options,
        )

        assert x0.shape == torch.Size(shape)
        assert torch.isfinite(x0).all()

        if "cuda" not in str(device):
            ref_file = f"{REF_PREFIX}{config_tag}_{sched_name}_{solver_name}.pth"
            ref = load_or_create_reference(ref_file, lambda: {"x0": x0.cpu()})
            compare_outputs(x0, ref["x0"], **SAMPLER_CPU_TOLERANCES)

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_guided_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_tag,
        sched_cls,
        sched_kwargs,
        sched_name,
        solver_key,
        solver_options,
        solver_name,
        guidance_config,
    ):
        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            guidance_config=guidance_config,
        )
        H, W = img_shape
        shape = (BATCH, CHANNELS, H, W)

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS,
            solver=solver_key,
            solver_options=solver_options,
        )

        assert x0.shape == torch.Size(shape)
        assert torch.isfinite(x0).all()

        if "cuda" not in str(device):
            ref_file = (
                f"{REF_PREFIX}{config_tag}_{sched_name}_{solver_name}"
                f"_{guidance_config}.pth"
            )
            ref = load_or_create_reference(ref_file, lambda: {"x0": x0.cpu()})
            compare_outputs(x0, ref["x0"], **SAMPLER_CPU_TOLERANCES)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,sched_cls,sched_kwargs,"
    "solver_key,solver_options",
    COMPILE_SAMPLE_CONFIGS,
    ids=[f"{c[0]}_{c[7]}" for c in COMPILE_SAMPLE_CONFIGS],
)
class TestMultiDiffusionSampleCompile:
    """Compiled denoiser inside the sample() loop (plain and guided)."""

    def test_compiled_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
    ):
        torch._dynamo.config.error_on_recompile = True

        scheduler, md, predictor, denoiser_eager, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )
        denoiser_compiled = torch.compile(denoiser_eager, fullgraph=True)

        with torch.no_grad():
            x0_eager = sample(
                denoiser_eager,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )
            x0_compiled = sample(
                denoiser_compiled,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )
        torch.testing.assert_close(x0_eager, x0_compiled, atol=0.5, rtol=0.3)

        with torch.no_grad():
            x0_compiled_2 = sample(
                denoiser_compiled,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)

    @pytest.mark.xfail(reason=_GUIDED_COMPILE_XFAIL, strict=False)
    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_compiled_guided_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
        guidance_config,
    ):
        torch._dynamo.config.error_on_recompile = True

        scheduler, md, predictor, denoiser_eager, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
        )
        denoiser_compiled = torch.compile(denoiser_eager, fullgraph=True)

        with torch.no_grad():
            x0_eager = sample(
                denoiser_eager,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )
            x0_compiled = sample(
                denoiser_compiled,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )
        torch.testing.assert_close(x0_eager, x0_compiled, atol=0.5, rtol=0.3)


@pytest.mark.xfail(
    _torch_version_ge_2_10(),
    reason=(
        "torch>=2.10 inductor codegen segfaults when compiling the full "
        "sample() call through MultiDiffusionPredictor (SIGSEGV at the C level, "
        "which brings down the pytest process), so it is not run on "
        "torch>=2.10. The per-step denoiser compile still runs."
    ),
    strict=False,
    run=False,
)
@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,sched_cls,sched_kwargs,"
    "solver_key,solver_options",
    COMPILE_SAMPLE_CONFIGS,
    ids=[f"{c[0]}_{c[7]}" for c in COMPILE_SAMPLE_CONFIGS],
)
class TestMultiDiffusionFullSamplerCompile:
    """Compile the entire sample() call (plain and guided)."""

    def test_compiled_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
    ):
        torch._dynamo.config.error_on_recompile = True

        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )

        def do_sample(x):
            return sample(
                denoiser,
                x,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )

        compiled_sample = torch.compile(do_sample, fullgraph=True)

        with torch.no_grad():
            x0_compiled = compiled_sample(xN)
        assert x0_compiled.shape == xN.shape
        assert torch.isfinite(x0_compiled).all()

        with torch.no_grad():
            x0_compiled_2 = compiled_sample(xN)
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)

        with torch.no_grad():
            x0_eager = do_sample(xN)
        torch.testing.assert_close(x0_eager, x0_compiled, atol=2.0, rtol=2.0)

    @pytest.mark.xfail(reason=_GUIDED_COMPILE_XFAIL, strict=False)
    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_compiled_guided_sample(
        self,
        deterministic_settings,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
        guidance_config,
    ):
        torch._dynamo.config.error_on_recompile = True

        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
        )

        def do_sample(x):
            return sample(
                denoiser,
                x,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )

        compiled_sample = torch.compile(do_sample, fullgraph=True)
        with torch.no_grad():
            x0_compiled = compiled_sample(xN)
        assert torch.isfinite(x0_compiled).all()


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,sched_cls,sched_kwargs,"
    "solver_key,solver_options",
    GRAD_SAMPLE_CONFIGS,
    ids=[f"{c[0]}_{c[7]}" for c in GRAD_SAMPLE_CONFIGS],
)
class TestMultiDiffusionSampleGradientFlow:
    """Gradients flow through the sampling loop to the wrapped model parameters."""

    def test_backward_through_sampling(
        self,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
    ):
        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=solver_key,
            solver_options=solver_options,
        )
        x0.sum().backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in md.parameters()
        )
        assert has_grad

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_backward_through_guided_sampling(
        self,
        device,
        md_config,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        sched_cls,
        sched_kwargs,
        solver_key,
        solver_options,
        guidance_config,
    ):
        # create_graph/retain_graph let the post-sample backward traverse the
        # per-step autograd.grad graph the guidance builds.
        scheduler, md, predictor, denoiser, xN = _make_sampling_components(
            md_config,
            img_shape,
            patch_shape,
            overlap_pix,
            boundary_pix,
            sched_cls,
            sched_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
            create_graph=True,
            retain_graph=True,
        )

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=solver_key,
            solver_options=solver_options,
        )
        x0.sum().backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in md.parameters()
        )
        assert has_grad
