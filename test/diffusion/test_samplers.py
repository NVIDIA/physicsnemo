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

from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.samplers.solvers import (
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
NUM_STEPS = 4
NUM_STEPS_SHORT = 2

# Sampler non-regression tolerances — looser than single-op tests because
# errors accumulate over multiple solver steps across different CPU ISAs.
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

PREDICTOR_TYPES = ["x0", "score"]


class _CustomEulerSolver:
    """User-defined solver implementing the Solver protocol from scratch."""

    def __init__(self, denoiser):
        self.denoiser = denoiser

    def step(self, x, t_cur, t_next):
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))
        d = self.denoiser(x, t_cur)
        return x + (t_next_bc - t_cur_bc) * d


# (solver_key, solver_options, sampler_name, uses_rng)
# "_custom_euler" is handled specially to create a _CustomEulerSolver instance.
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

TIME_EVAL_INDICES = [0, 1, 3]


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
):
    """Create scheduler, model, denoiser, and initial latents."""
    scheduler = sched_cls(**sched_kwargs)
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    if predictor_type == "score":
        denoiser = scheduler.get_denoiser(score_predictor=model, denoising_type="ode")
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


@pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
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
    """Non-regression tests for sample() across all sampler configs."""

    def test_sample(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
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

    def test_sample_with_time_eval(
        self,
        deterministic_settings,
        device,
        tolerances,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
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
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
        x0_via_time_steps = sample(
            denoiser,
            xN,
            scheduler,
            num_steps=0,
            time_steps=t_steps,
            solver="euler",
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

        x0_via_string = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
        x0_via_instance = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=EulerSolver(denoiser),
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

        x0_builtin = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
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
            EDMNoiseScheduler,
            {},
            shape,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
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
            EDMNoiseScheduler,
            {},
            shape,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
        )
        with pytest.raises(ValueError, match="Unknown solver"):
            sample(denoiser, xN, scheduler, NUM_STEPS, solver="nonexistent")


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
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
    """torch.compile tests: compiled denoiser passed to sample()."""

    def test_compiled_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Sampling with a compiled denoiser matches eager sampling.

        Also makes a second compiled call to verify the graph is reused,
        with error_on_recompile to catch unexpected graph breaks.
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
            predictor_type=predictor_type,
        )
        compiled_denoiser = torch.compile(denoiser, fullgraph=True)

        solver_eager, opts_eager = _make_solver_arg(
            solver_key,
            solver_options,
            denoiser,
        )
        solver_compiled, opts_compiled = _make_solver_arg(
            solver_key,
            solver_options,
            compiled_denoiser,
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

        # Second compiled call — must reuse the graph (error_on_recompile guards this)
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


# =============================================================================
# Full Sampler Compile Tests
# =============================================================================


@pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
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
    """Compile the entire sample() call and verify double-call graph reuse."""

    def test_compiled_sample(
        self,
        deterministic_settings,
        device,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
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

        # _custom_euler uses an instance, not a string — skip it here since
        # we test string-based solver dispatch through compile.
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

        # Second call — must reuse the graph
        with torch.no_grad():
            x0_compiled_2 = compiled_sample(xN)
        assert x0_compiled_2.shape == shape
        assert torch.isfinite(x0_compiled_2).all()

        # For deterministic solvers, also verify eager-vs-compiled match
        if not uses_rng:
            with torch.no_grad():
                x0_eager = do_sample(xN)
            torch.testing.assert_close(x0_eager, x0_compiled, atol=2.0, rtol=2.0)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize("predictor_type", PREDICTOR_TYPES, ids=PREDICTOR_TYPES)
@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
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
    """Tests that gradients flow through the sampling loop to model parameters."""

    def test_backward_through_sampling(
        self,
        device,
        predictor_type,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
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
        loss = x0.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad
