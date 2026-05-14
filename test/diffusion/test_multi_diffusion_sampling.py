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

from physicsnemo.diffusion.multi_diffusion import MultiDiffusionPredictor
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample

from .conftest import GLOBAL_SEED
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

# Sampler tolerances — looser than single-op tests (errors accumulate over steps)
SAMPLER_CPU_TOLERANCES = {"atol": 20.0, "rtol": 5e-2}

# (md_config, img_shape, patch_shape, overlap_pix, boundary_pix, config_tag)
MULTI_DIFFUSION_SAMPLE_CONFIGS = [
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


# =============================================================================
# Helpers
# =============================================================================


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
):
    """Create scheduler, md model, predictor, denoiser, and initial latent."""
    scheduler = sched_cls(**sched_kwargs)
    md = _create_md_model(md_config, img_shape=img_shape).to(device)
    md.set_grid_patching(
        patch_shape=patch_shape,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
        fuse=True,
    )
    condition = _make_condition(md_config, img_shape=img_shape, device=device)
    predictor = MultiDiffusionPredictor(md, condition=condition, fuse=True)
    predictor.set_patching(overlap_pix=overlap_pix, boundary_pix=boundary_pix)
    denoiser = scheduler.get_denoiser(x0_predictor=predictor)

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
    MULTI_DIFFUSION_SAMPLE_CONFIGS,
    ids=[c[5] for c in MULTI_DIFFUSION_SAMPLE_CONFIGS],
)
class TestMultiDiffusionSampleNonRegression:
    """Non-regression tests for sample() with MultiDiffusionPredictor."""

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


# =============================================================================
# Compile Tests
# =============================================================================

# Subset for compile tests: keep small to avoid long CI times
COMPILE_SAMPLE_CONFIGS = [
    ("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "uncond_sq_nooverlap"),
    ("cond_patch", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "cond_patch_sq_nooverlap"),
]

COMPILE_SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VENoiseScheduler, {}, "ve"),
]

COMPILE_SOLVER_CONFIGS = [
    ("euler", None, "euler"),
    ("heun", None, "heun"),
]


@pytest.mark.parametrize(
    "solver_key,solver_options,solver_name",
    COMPILE_SOLVER_CONFIGS,
    ids=[c[2] for c in COMPILE_SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    COMPILE_SCHEDULER_CONFIGS,
    ids=[c[2] for c in COMPILE_SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,config_tag",
    COMPILE_SAMPLE_CONFIGS,
    ids=[c[5] for c in COMPILE_SAMPLE_CONFIGS],
)
class TestMultiDiffusionSampleCompile:
    """torch.compile tests: compiled denoiser in sample() loop.

    Mirrors ``test_samplers.py::TestSampleCompile`` but with
    ``MultiDiffusionPredictor`` supplying the x0 predictor inside the
    denoiser. Compiling the denoiser closure (not the predictor instance) is
    the same pattern used across the rest of the diffusion test suite.
    """

    def test_compiled_denoiser_in_sample(
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
        """Compiled denoiser produces same output as eager; graph reused on second call."""
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
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_eager = sample(
                denoiser_eager,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )

            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_compiled = sample(
                denoiser_compiled,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )

        torch.testing.assert_close(x0_eager, x0_compiled, atol=0.5, rtol=0.3)

        # Second compiled call, must reuse the graph (error_on_recompile guards this)
        with torch.no_grad():
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_compiled_2 = sample(
                denoiser_compiled,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options,
            )

        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)


# =============================================================================
# Full Sampler Compile Tests
# =============================================================================


def _torch_version_ge_2_10() -> bool:
    """True when the installed torch version is >= 2.10."""
    parts = torch.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 10)


@pytest.mark.xfail(
    _torch_version_ge_2_10(),
    reason=(
        "torch>=2.10 inductor codegen segfaults when compiling the full "
        "sample() call through MultiDiffusionPredictor. The crash is at the "
        "C level (SIGSEGV), which would bring down the whole pytest process, "
        "so the test is not run on torch>=2.10. The per-step denoiser compile "
        "(TestMultiDiffusionSampleCompile) still runs cleanly. Revisit once "
        "the upstream issue is resolved."
    ),
    strict=False,
    run=False,
)
@pytest.mark.parametrize(
    "solver_key,solver_options,solver_name",
    COMPILE_SOLVER_CONFIGS,
    ids=[c[2] for c in COMPILE_SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    COMPILE_SCHEDULER_CONFIGS,
    ids=[c[2] for c in COMPILE_SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,config_tag",
    COMPILE_SAMPLE_CONFIGS,
    ids=[c[5] for c in COMPILE_SAMPLE_CONFIGS],
)
class TestMultiDiffusionFullSamplerCompile:
    """Compile the entire ``sample()`` call (not just the denoiser).

    Mirrors ``test_samplers.py::TestFullSamplerCompile``: wraps the
    ``sample(denoiser, xN, scheduler, ..., solver=...)`` invocation in a
    closure and compiles the whole thing with ``fullgraph=True``. Verifies
    the graph is reused on a second call, and that the compiled output
    matches eager within a loose tolerance.
    """

    def test_compiled_sample(
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

        # Second call must reuse the graph (error_on_recompile guards this)
        with torch.no_grad():
            x0_compiled_2 = compiled_sample(xN)
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)

        # Eager vs compiled match, with a loose tolerance
        with torch.no_grad():
            x0_eager = do_sample(xN)
        torch.testing.assert_close(x0_eager, x0_compiled, atol=2.0, rtol=2.0)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_key,solver_options,solver_name",
    [("euler", None, "euler")],
    ids=["euler"],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    [(EDMNoiseScheduler, {}, "edm")],
    ids=["edm"],
)
@pytest.mark.parametrize(
    "md_config,img_shape,patch_shape,overlap_pix,boundary_pix,config_tag",
    [("uncond", (IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "uncond_sq_nooverlap")],
    ids=["uncond_sq_nooverlap"],
)
class TestMultiDiffusionSampleGradientFlow:
    """Tests that gradients flow through the sampling loop to model parameters."""

    def test_backward_through_sampling(
        self,
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
