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

"""Tests for diffusion model sampling interface."""

import pytest
import torch

from physicsnemo.diffusion.guidance import (
    DataConsistencyDPSGuidance,
    DPSScorePredictor,
    ModelConsistencyDPSGuidance,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.samplers.solvers import (
    DPMSolverPlusPlus2M,
    EulerSolver,
    HeunSolver,
)

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    Conv3dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    gpu_rng_roundtrip,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_samplers_"
BATCH = 2
NUM_STEPS = 2
NUM_STEPS_SHORT = 2

# Sampler non-regression tolerances: looser than single-op tests, since errors
# accumulate over solver steps and across CPU ISAs.
SAMPLER_CPU_TOLERANCES = {"atol": 20.0, "rtol": 5e-2}
SAMPLER_GPU_TOLERANCES = {"atol": 20.0, "rtol": 5e-2}

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VENoiseScheduler, {}, "ve"),
    (VPNoiseScheduler, {}, "vp"),
]

PREDICTOR_TYPES = ["x0", "score", "epsilon"]

# Guided sub-parameterization (DPS guidance), crossed with the common config
# only in the guided methods. Guided sampling uses the euler solver.
GUIDANCE_CONFIGS = ["data_l2", "model_l2_sda"]


class _CustomEulerSolver:
    """User-defined solver implementing the Solver protocol from scratch."""

    def __init__(self, denoiser):
        self.denoiser = denoiser

    def step(self, x, t_cur, t_next):
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))
        d = self.denoiser(x, t_cur)
        return x + (t_next_bc - t_cur_bc) * d


# (solver_key, solver_options, sampler_name, uses_rng). "_custom_euler" maps to
# a _CustomEulerSolver instance via _make_solver_arg.
SAMPLER_CONFIGS = [
    ("euler", {}, "euler", False),
    ("heun", {}, "heun", False),
    ("heun", {"alpha": 0.5}, "heun_midpoint", False),
    ("_custom_euler", {}, "custom_euler", False),
    (
        "edm_stochastic_euler",
        {"S_churn": 20, "num_steps": NUM_STEPS},
        "stoch_euler",
        True,
    ),
    (
        "edm_stochastic_heun",
        {"S_churn": 20, "num_steps": NUM_STEPS},
        "stoch_heun",
        True,
    ),
]

TIME_EVAL_INDICES = [0, 1]


def _make_sampling_components(
    sched_cls,
    sched_kwargs,
    shape,
    predictor_cls,
    predictor_kwargs,
    device,
    seed=0,
    num_steps=NUM_STEPS,
    predictor_type="x0",
    guidance_config=None,
    create_graph=False,
    retain_graph=False,
):
    """Create scheduler, model, denoiser, and initial latents.

    With ``guidance_config`` set, the model is wrapped in a DPSScorePredictor
    with a DPS guidance and the denoiser uses the score path; otherwise the
    denoiser uses ``predictor_type`` (x0 / score / epsilon).
    """
    scheduler = sched_cls(**sched_kwargs)
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    if guidance_config is not None:
        if guidance_config == "data_l2":
            mask = torch.zeros(shape, dtype=torch.bool, device=device)
            flat = mask.view(shape[0], -1)
            flat[:, 0] = True
            flat[:, flat.shape[1] // 2] = True
            y = make_input(shape, seed=300, device=device)
            guidance = DataConsistencyDPSGuidance(
                mask=mask,
                y=y,
                std_y=0.1,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )
        elif guidance_config == "model_l2_sda":
            obs_shape = (shape[0], 1, *shape[2:])
            y = make_input(obs_shape, seed=300, device=device)
            guidance = ModelConsistencyDPSGuidance(
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
        dps = DPSScorePredictor(
            x0_predictor=model,
            x0_to_score_fn=scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = scheduler.get_denoiser(score_predictor=dps, denoising_type="ode")
    elif predictor_type == "score":
        denoiser = scheduler.get_denoiser(score_predictor=model, denoising_type="ode")
    elif predictor_type == "epsilon":
        denoiser = scheduler.get_denoiser(epsilon_predictor=model, denoising_type="ode")
    else:
        denoiser = scheduler.get_denoiser(x0_predictor=model, denoising_type="ode")
    t_steps = scheduler.timesteps(num_steps, device=device)
    tN = t_steps[0].expand(shape[0])
    xN = make_input(shape, seed=200, device=device) * tN.view(
        -1, *([1] * (len(shape) - 1))
    )
    return scheduler, model, denoiser, xN


def _make_solver_arg(solver_key, solver_options, denoiser):
    """Build the solver argument for sample() from config fields."""
    if solver_key == "_custom_euler":
        return _CustomEulerSolver(denoiser), None
    return solver_key, solver_options or None


# =============================================================================
# Non-Regression Tests
# =============================================================================


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
class TestSampleNonRegression:
    """Non-regression tests for sample() (plain and guided)."""

    @pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
    @pytest.mark.parametrize(
        "solver_key,solver_options,sampler_name,uses_rng",
        SAMPLER_CONFIGS,
        ids=[c[2] for c in SAMPLER_CONFIGS],
    )
    def test_sample(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
    ):
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            predictor_type=predictor_type,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        if "cuda" in str(device) and uses_rng:

            def fn():
                return sample(
                    denoiser,
                    xN,
                    scheduler,
                    NUM_STEPS,
                    solver=solver_arg,
                    solver_options=opts,
                )

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
        elif "cuda" in str(device) or uses_rng:
            x0 = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
            )
            assert x0.shape == shape
            assert torch.isfinite(x0).all()
        else:
            x0 = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
            )
            assert x0.shape == shape
            assert torch.isfinite(x0).all()
            ref_file = f"{REF_PREFIX}{sampler_name}_{sched_name}_{spatial_name}_{predictor_type}pred.pth"
            ref = load_or_create_reference(ref_file, lambda: {"x0": x0.cpu()})
            compare_outputs(x0, ref["x0"], **SAMPLER_CPU_TOLERANCES)

    @pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
    @pytest.mark.parametrize(
        "solver_key,solver_options,sampler_name,uses_rng",
        SAMPLER_CONFIGS,
        ids=[c[2] for c in SAMPLER_CONFIGS],
    )
    def test_sample_with_time_eval(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
    ):
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            predictor_type=predictor_type,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        if "cuda" in str(device) and uses_rng:

            def fn():
                results = sample(
                    denoiser,
                    xN,
                    scheduler,
                    NUM_STEPS,
                    solver=solver_arg,
                    solver_options=opts,
                    time_eval=TIME_EVAL_INDICES,
                )
                return torch.stack(results)

            stacked = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
        elif "cuda" in str(device) or uses_rng:
            results = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
                time_eval=TIME_EVAL_INDICES,
            )
            stacked = torch.stack(results)
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
            assert torch.isfinite(stacked).all()
        else:
            results = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
                time_eval=TIME_EVAL_INDICES,
            )
            stacked = torch.stack(results)
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
            assert torch.isfinite(stacked).all()
            ref_file = f"{REF_PREFIX}{sampler_name}_{sched_name}_{spatial_name}_{predictor_type}pred_teval.pth"
            ref = load_or_create_reference(ref_file, lambda: {"stacked": stacked.cpu()})
            compare_outputs(stacked, ref["stacked"], **SAMPLER_CPU_TOLERANCES)

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_guided_sample(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        guidance_config,
    ):
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            guidance_config=guidance_config,
        )

        x0 = sample(denoiser, xN, scheduler, NUM_STEPS, solver="euler")

        assert x0.shape == shape
        assert torch.isfinite(x0).all()

        if "cuda" not in str(device):
            ref_file = (
                f"{REF_PREFIX}guided_{spatial_name}_{sched_name}_{guidance_config}.pth"
            )
            ref = load_or_create_reference(ref_file, lambda: {"x0": x0.cpu()})
            compare_outputs(x0, ref["x0"], **SAMPLER_CPU_TOLERANCES)


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
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
class TestSampleConsistency:
    """Tests that equivalent argument combinations produce identical results."""

    def test_time_steps_vs_num_steps(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing explicit time_steps from scheduler.timesteps(N) should match
        passing num_steps=N to let sample() generate them internally."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )
        t_steps = scheduler.timesteps(NUM_STEPS_SHORT, device=device, dtype=xN.dtype)

        x0_via_num_steps = sample(
            denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler"
        )
        x0_via_time_steps = sample(
            denoiser, xN, scheduler, num_steps=0, time_steps=t_steps, solver="euler"
        )
        compare_outputs(x0_via_time_steps, x0_via_num_steps, atol=1e-6, rtol=1e-6)

    def test_solver_string_vs_instance(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing solver="euler" should match passing solver=EulerSolver(denoiser)."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )

        x0_via_string = sample(denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler")
        x0_via_instance = sample(
            denoiser, xN, scheduler, NUM_STEPS_SHORT, solver=EulerSolver(denoiser)
        )
        compare_outputs(x0_via_instance, x0_via_string, atol=1e-6, rtol=1e-6)

    def test_solver_options_vs_instance(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing solver="heun" + solver_options={"alpha": 0.5} should match
        passing solver=HeunSolver(denoiser, alpha=0.5)."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )

        x0_via_options = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="heun",
            solver_options={"alpha": 0.5},
        )
        x0_via_instance = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=HeunSolver(denoiser, alpha=0.5),
        )
        compare_outputs(x0_via_instance, x0_via_options, atol=1e-6, rtol=1e-6)

    def test_custom_solver_vs_euler(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """User-defined _CustomEulerSolver should match built-in EulerSolver."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )

        x0_builtin = sample(denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler")
        x0_custom = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=_CustomEulerSolver(denoiser),
        )
        compare_outputs(x0_custom, x0_builtin, **tolerances)


# =============================================================================
# Validation / Error Tests
# =============================================================================


class TestSampleValidation:
    """Tests for sample() argument validation and error handling."""

    def test_solver_options_with_instance_raises(self, device):
        shape = (BATCH, 3, 8, 6)
        scheduler, _, denoiser, xN = _make_sampling_components(
            EDMNoiseScheduler, {}, shape, Conv2dX0Predictor, {"channels": 3}, device
        )
        with pytest.raises(ValueError, match="solver_options"):
            sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=EulerSolver(denoiser),
                solver_options={"alpha": 0.5},
            )

    def test_unknown_solver_string_raises(self, device):
        shape = (BATCH, 3, 8, 6)
        scheduler, _, denoiser, xN = _make_sampling_components(
            EDMNoiseScheduler, {}, shape, Conv2dX0Predictor, {"channels": 3}, device
        )
        with pytest.raises(ValueError, match="Unknown solver"):
            sample(denoiser, xN, scheduler, NUM_STEPS, solver="nonexistent")


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.usefixtures("nop_compile")
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
class TestSampleCompile:
    """torch.compile tests: compiled denoiser passed to sample() (plain and guided)."""

    @pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
    @pytest.mark.parametrize(
        "solver_key,solver_options,sampler_name,uses_rng",
        SAMPLER_CONFIGS,
        ids=[c[2] for c in SAMPLER_CONFIGS],
    )
    def test_compiled_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
    ):
        """Sampling with a compiled denoiser matches eager; graph reused on 2nd call."""
        torch._dynamo.config.error_on_recompile = True

        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )
        compiled_denoiser = torch.compile(denoiser, fullgraph=True)

        solver_eager, opts_eager = _make_solver_arg(
            solver_key, solver_options, denoiser
        )
        solver_compiled, opts_compiled = _make_solver_arg(
            solver_key, solver_options, compiled_denoiser
        )

        with torch.no_grad():
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_eager = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_eager,
                solver_options=opts_eager,
            )
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_compiled = sample(
                compiled_denoiser,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_compiled,
                solver_options=opts_compiled,
            )
        torch.testing.assert_close(x0_eager, x0_compiled, atol=0.5, rtol=0.3)

        with torch.no_grad():
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_compiled_2 = sample(
                compiled_denoiser,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_compiled,
                solver_options=opts_compiled,
            )
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_compiled_guided_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        guidance_config,
    ):
        """Compiled guided (DPS) denoiser matches eager; graph reused on 2nd call.

        fullgraph=False because the guidance computes an internal autograd.grad
        (a graph break) inside the denoiser.
        """
        torch._dynamo.config.error_on_recompile = True

        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
        )
        compiled_denoiser = torch.compile(denoiser)

        with torch.no_grad():
            x0_eager = sample(denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler")
            x0_compiled = sample(
                compiled_denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler"
            )
        torch.testing.assert_close(x0_eager, x0_compiled, atol=0.5, rtol=0.3)

        with torch.no_grad():
            x0_compiled_2 = sample(
                compiled_denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler"
            )
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)


# =============================================================================
# Full Sampler Compile Tests
# =============================================================================


@pytest.mark.usefixtures("nop_compile")
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
class TestFullSamplerCompile:
    """Compile the entire sample() call (plain and guided)."""

    @pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
    @pytest.mark.parametrize(
        "solver_key,solver_options,sampler_name,uses_rng",
        SAMPLER_CONFIGS,
        ids=[c[2] for c in SAMPLER_CONFIGS],
    )
    def test_compiled_sample(
        self,
        deterministic_settings,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
    ):
        """torch.compile(sample(...)) traces and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )

        # Custom solver instances are exercised in TestSampleCompile; here we
        # test string-based solver dispatch through compile.
        if solver_key == "_custom_euler":
            pytest.skip("Custom solver instances are tested in TestSampleCompile")

        def do_sample(x):
            return sample(
                denoiser,
                x,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_key,
                solver_options=solver_options or None,
            )

        compiled_sample = torch.compile(do_sample, fullgraph=True)

        with torch.no_grad():
            x0_compiled = compiled_sample(xN)
        assert x0_compiled.shape == shape
        assert torch.isfinite(x0_compiled).all()

        with torch.no_grad():
            x0_compiled_2 = compiled_sample(xN)
        assert x0_compiled_2.shape == shape
        assert torch.isfinite(x0_compiled_2).all()

        if not uses_rng:
            with torch.no_grad():
                x0_eager = do_sample(xN)
            torch.testing.assert_close(x0_eager, x0_compiled, atol=2.0, rtol=2.0)

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_compiled_guided_sample(
        self,
        deterministic_settings,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        guidance_config,
    ):
        """Guided: torch.compile(sample(...)) over the DPS path traces and reuses
        the graph. fullgraph=False because the guidance computes an internal
        autograd.grad inside the denoiser."""
        torch._dynamo.config.error_on_recompile = True

        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
        )

        def do_sample(x):
            return sample(denoiser, x, scheduler, NUM_STEPS_SHORT, solver="euler")

        compiled_sample = torch.compile(do_sample)

        with torch.no_grad():
            x0_compiled = compiled_sample(xN)
        assert x0_compiled.shape == shape
        assert torch.isfinite(x0_compiled).all()

        with torch.no_grad():
            x0_compiled_2 = compiled_sample(xN)
        torch.testing.assert_close(x0_compiled, x0_compiled_2, atol=0.5, rtol=0.3)

        with torch.no_grad():
            x0_eager = do_sample(xN)
        torch.testing.assert_close(x0_eager, x0_compiled, atol=2.0, rtol=2.0)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


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
class TestGradientFlow:
    """Gradients flow through the sampling loop to model parameters (plain and guided)."""

    @pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
    @pytest.mark.parametrize(
        "solver_key,solver_options,sampler_name,uses_rng",
        SAMPLER_CONFIGS,
        ids=[c[2] for c in SAMPLER_CONFIGS],
    )
    def test_backward_through_sampling(
        self,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
    ):
        scheduler, model, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            predictor_type=predictor_type,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=solver_arg,
            solver_options=opts,
        )
        x0.sum().backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS, ids=GUIDANCE_CONFIGS)
    def test_backward_through_guided_sampling(
        self,
        device,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        sched_cls,
        sched_kwargs,
        sched_name,
        guidance_config,
    ):
        # create_graph/retain_graph let the post-sample backward traverse the
        # per-step autograd.grad graph the guidance builds.
        scheduler, model, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
            guidance_config=guidance_config,
            create_graph=True,
            retain_graph=True,
        )

        x0 = sample(denoiser, xN, scheduler, NUM_STEPS_SHORT, solver="euler")
        x0.sum().backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad


# =============================================================================
# DPM-Solver++(2M) Sampler Integration
# =============================================================================


@pytest.mark.usefixtures("deterministic_settings")
class TestDPMSolverPlusPlus2MSampling:
    """sample() integration for the stateful multistep solver."""

    SHAPE = (BATCH, 3, 8, 6)

    def _components(self, device, sched_cls=EDMNoiseScheduler, num_steps=NUM_STEPS):
        return _make_sampling_components(
            sched_cls,
            {},
            self.SHAPE,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
            num_steps=num_steps,
        )

    def test_entry_reset_discards_hand_driven_state(self, device):
        """sample() must clear history it did not create.

        The exit reset alone is not enough: a caller may drive ``step`` by hand
        -- a documented use -- and then pass the same instance to ``sample``.
        Without the reset at entry, that stale data prediction is extrapolated
        into the first step of the new trajectory.
        """
        scheduler, _, denoiser, xN = self._components(device, num_steps=6)
        solver = DPMSolverPlusPlus2M(denoiser)

        # Prime with a non-terminal step from an unrelated trajectory.
        solver.step(
            xN,
            torch.full((BATCH,), 40.0, device=device),
            torch.full((BATCH,), 12.0, device=device),
        )
        assert solver._D_prev is not None

        primed = sample(denoiser, xN, scheduler, 6, solver=solver)
        fresh = sample(denoiser, xN, scheduler, 6, solver=DPMSolverPlusPlus2M(denoiser))
        torch.testing.assert_close(primed, fresh, rtol=0, atol=0)

        # The cached latent-sized tensor must not outlive the trajectory either.
        assert solver._D_prev is None
        assert solver._h_prev is None

    def test_reset_hook_requires_the_marker(self, device):
        """sample() must not call reset() on a solver that did not opt in.

        ``reset`` is a common method name, so a pre-existing user-defined solver
        may already have one meaning something else entirely. The lifecycle call
        is gated on ``_requires_state_reset`` so that such a solver keeps its
        previous behavior.
        """
        scheduler, _, denoiser, xN = self._components(device)

        class _UnrelatedReset:
            """Solver whose reset() is its own business, not sampler lifecycle."""

            def __init__(self, den):
                self.denoiser, self.reset_calls = den, 0

            def reset(self):
                self.reset_calls += 1

            def step(self, x, t_cur, t_next):
                shape = (-1,) + (1,) * (x.ndim - 1)
                return x + (
                    t_next.reshape(shape) - t_cur.reshape(shape)
                ) * self.denoiser(x, t_cur)

        solver = _UnrelatedReset(denoiser)
        assert not hasattr(solver, "_requires_state_reset")
        sample(denoiser, xN, scheduler, NUM_STEPS, solver=solver)
        assert solver.reset_calls == 0

        # The shipped solver does opt in, so it is still reset.
        opted_in = DPMSolverPlusPlus2M(denoiser)
        sample(denoiser, xN, scheduler, NUM_STEPS, solver=opted_in)
        assert opted_in._D_prev is None

    def test_history_released_after_error(self, device):
        """Cached history must also be released when sampling raises."""
        scheduler, _, denoiser, xN = self._components(device)

        calls = []

        def failing_denoiser(x, t):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("boom")
            return denoiser(x, t)

        solver = DPMSolverPlusPlus2M(failing_denoiser)
        with pytest.raises(RuntimeError, match="boom"):
            sample(failing_denoiser, xN, scheduler, 4, solver=solver)
        assert solver._D_prev is None
        assert solver._h_prev is None

    def test_multistep_path_is_exercised(self, device):
        """Guard against a trajectory that never leaves the first-order path.

        With ``num_steps=2`` the first step has no history and the second lands
        on ``t = 0``, where the update returns the data prediction and discards
        the extrapolation -- so the result matches Euler up to floating-point
        round-off and proves
        nothing about the multistep coefficients. At least three steps are
        needed for the second-order path to affect the output.
        """
        scheduler, _, denoiser, xN = self._components(device, num_steps=4)

        def relative_gap(num_steps):
            dpm = sample(denoiser, xN, scheduler, num_steps, solver="dpmpp_2m")
            # The registry key and a hand-built instance must agree exactly:
            # sample() constructs the solver itself on the string path, and only
            # the instance path additionally has state to reset.
            by_instance = sample(
                denoiser, xN, scheduler, num_steps, solver=DPMSolverPlusPlus2M(denoiser)
            )
            torch.testing.assert_close(dpm, by_instance, rtol=0, atol=0)
            euler = sample(denoiser, xN, scheduler, num_steps, solver="euler")
            return float((dpm - euler).abs().max() / dpm.abs().max())

        # Two steps: identical up to floating-point round-off.
        assert relative_gap(2) < 1e-4
        # Three steps: the multistep update reaches the output.
        assert relative_gap(3) > 1e-2

    @pytest.mark.usefixtures("nop_compile")
    def test_compiled_sample(self, device):
        """The whole trajectory compiles fullgraph and reuses its graph."""
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()

        scheduler, _, denoiser, xN = self._components(device, num_steps=4)
        solver = DPMSolverPlusPlus2M(denoiser)

        def do_sample(x):
            return sample(denoiser, x, scheduler, 4, solver=solver)

        compiled = torch.compile(do_sample, fullgraph=True)
        with torch.no_grad():
            first = compiled(xN)
        torch._dynamo.config.error_on_recompile = True
        try:
            with torch.no_grad():
                second = compiled(xN)
        finally:
            torch._dynamo.config.error_on_recompile = False

        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        # Compiling must not change the trajectory. Comparing the two compiled
        # runs above only shows the graph is reused, not that it is right.
        eager = sample(denoiser, xN, scheduler, 4, solver=DPMSolverPlusPlus2M(denoiser))
        torch.testing.assert_close(first, eager, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "sched_cls",
        [VPNoiseScheduler, VENoiseScheduler],
        ids=["vp", "ve"],
    )
    def test_string_dispatch_configures_linear_gaussian_schedule(
        self, device, sched_cls
    ):
        """Selecting by name must configure the solver from the scheduler.

        Shape and finiteness alone would not detect the schedule functions being
        dropped, since the EDM defaults also produce a finite result. The string
        result is therefore compared against an explicitly configured instance,
        and shown to differ from the EDM defaults. Only non-EDM schedules are
        used: on EDM the injected functions equal the defaults, so the
        comparison would hold even if the injection were removed.
        """
        scheduler, _, denoiser, xN = self._components(
            device, sched_cls=sched_cls, num_steps=6
        )
        by_name = sample(denoiser, xN, scheduler, 6, solver="dpmpp_2m")
        configured = sample(
            denoiser,
            xN,
            scheduler,
            6,
            solver=DPMSolverPlusPlus2M(
                denoiser,
                alpha_fn=scheduler.alpha,
                sigma_fn=scheduler.sigma,
                alpha_dot_fn=scheduler.alpha_dot,
                sigma_dot_fn=scheduler.sigma_dot,
            ),
        )
        torch.testing.assert_close(by_name, configured, rtol=0, atol=0)

        # The defaults integrate a different ODE here, so the equality above
        # would be vacuous if the schedule functions were ignored.
        edm_defaults = sample(
            denoiser, xN, scheduler, 6, solver=DPMSolverPlusPlus2M(denoiser)
        )
        assert not torch.allclose(by_name, edm_defaults)

    def test_schedule_functions_taken_from_wrapped_scheduler(self, device):
        """A wrapper delegating to an inner scheduler must be unwrapped.

        ``DomainParallelNoiseScheduler`` exposes the schedule functions only via
        its inner scheduler, so the wrapped result must equal the unwrapped one.
        """
        scheduler, _, denoiser, xN = self._components(device)

        class _Wrapper:
            """Minimal wrapper exposing inner_scheduler and timesteps."""

            def __init__(self, inner):
                self._inner = inner

            @property
            def inner_scheduler(self):
                return self._inner

            def timesteps(self, *args, **kwargs):
                return self._inner.timesteps(*args, **kwargs)

        wrapped = sample(
            denoiser, xN, _Wrapper(scheduler), NUM_STEPS, solver="dpmpp_2m"
        )
        direct = sample(denoiser, xN, scheduler, NUM_STEPS, solver="dpmpp_2m")
        torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)

    def test_scheduler_without_schedule_functions_is_rejected(self, device):
        """A scheduler missing the schedule functions must fail clearly."""
        scheduler, _, denoiser, xN = self._components(device)

        class _Incomplete:
            def __init__(self, inner):
                self._inner = inner

            def timesteps(self, *args, **kwargs):
                return self._inner.timesteps(*args, **kwargs)

            def alpha(self, t):
                return torch.ones_like(t)

        with pytest.raises(ValueError, match="sigma"):
            sample(denoiser, xN, _Incomplete(scheduler), NUM_STEPS, solver="dpmpp_2m")

        class _NoScheduleFns(_Incomplete):
            alpha = None

        # Providing none of the four must be rejected here rather than falling
        # through to the EDM defaults, which would silently integrate the wrong
        # ODE. The constructor's partial-set guard cannot catch this case.
        with pytest.raises(ValueError, match="alpha, alpha_dot, sigma, sigma_dot"):
            sample(
                denoiser, xN, _NoScheduleFns(scheduler), NUM_STEPS, solver="dpmpp_2m"
            )

    def test_scheduler_overrides_conflicting_solver_options(self, device):
        """The scheduler is authoritative and the caller's dict is untouched.

        The time-steps come from the scheduler, so schedule functions describing
        anything else would silently disagree with them.
        """
        scheduler, _, denoiser, xN = self._components(device)
        conflicting = {
            "alpha_fn": lambda t: torch.full_like(t, 2.0),
            "sigma_fn": lambda t: 3.0 * t,
            "alpha_dot_fn": torch.zeros_like,
            "sigma_dot_fn": lambda t: torch.full_like(t, 3.0),
        }
        opts = dict(conflicting)
        out = sample(
            denoiser, xN, scheduler, NUM_STEPS, solver="dpmpp_2m", solver_options=opts
        )
        assert opts.keys() == conflicting.keys()
        assert all(opts[k] is conflicting[k] for k in conflicting)

        expected = sample(denoiser, xN, scheduler, NUM_STEPS, solver="dpmpp_2m")
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "sched_cls", [VPNoiseScheduler, VENoiseScheduler], ids=["vp", "ve"]
    )
    @pytest.mark.usefixtures("nop_compile")
    def test_compiled_sample_with_general_schedule(self, device, sched_cls):
        """Bound scheduler methods must remain traceable under fullgraph."""
        torch._dynamo.reset()
        scheduler, _, denoiser, xN = self._components(
            device, sched_cls=sched_cls, num_steps=4
        )

        def do_sample(x):
            return sample(denoiser, x, scheduler, 4, solver="dpmpp_2m")

        with torch.no_grad():
            compiled = torch.compile(do_sample, fullgraph=True)(xN)
            eager = do_sample(xN)
        assert torch.isfinite(compiled).all()
        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("guidance_config", GUIDANCE_CONFIGS)
    def test_dps_guidance_reuse_is_clean(self, device, guidance_config):
        """Guided sampling must be repeatable on one solver instance.

        DPS denoisers attach a per-step autograd graph, so the cached data
        prediction is the one place a guided run could retain state or leak a
        graph into the next trajectory.
        """
        scheduler, _, denoiser, xN = _make_sampling_components(
            EDMNoiseScheduler,
            {},
            self.SHAPE,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
            num_steps=4,
            guidance_config=guidance_config,
        )
        solver = DPMSolverPlusPlus2M(denoiser)

        with torch.no_grad():
            first = sample(denoiser, xN, scheduler, 4, solver=solver)
            second = sample(denoiser, xN, scheduler, 4, solver=solver)

        assert first.shape == self.SHAPE
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert solver._D_prev is None
        assert solver._h_prev is None
        # Under no_grad the result must not carry a graph from the guidance.
        assert not first.requires_grad
        assert first.grad_fn is None
