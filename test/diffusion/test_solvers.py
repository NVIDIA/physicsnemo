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

"""Tests for diffusion ODE/SDE solvers."""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import (
    DPMPlusPlus2M,
    DPMPlusPlus2MUniC2,
    EDMStochasticEulerSolver,
    EDMStochasticExponentialEulerSolver,
    EDMStochasticHeunSolver,
    EulerSolver,
    ExponentialEulerSolver,
    HeunSolver,
    Solver,
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

REF_PREFIX = "test_solvers_"
BATCH = 2

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

# (solver_cls, solver_kwargs, solver_name, uses_rng, time_scale)
# The solver constructor receives solver_kwargs after `denoiser`. The
# "_use_*" keys are sentinels resolved by _make_solver_and_denoiser: they
# select the noise scheduler of the config and the schedule callbacks built
# from it. time_scale scales the step times, so that the tests stay within
# the valid time range of bounded schedules (VP).
# The configs of the exponential and DPM-Solver++(2M) solvers mirror the
# docstring examples of their classes.
SOLVER_CONFIGS = [
    (EulerSolver, {}, "euler", False, 1.0),
    (HeunSolver, {}, "heun", False, 1.0),
    (HeunSolver, {"alpha": 0.5}, "heun_midpoint", False, 1.0),
    (EDMStochasticEulerSolver, {"S_churn": 0}, "stoch_euler_nochurn", False, 1.0),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_euler_churn",
        True,
        1.0,
    ),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10, "_use_edm_sigma_fns": True},
        "stoch_euler_sigmafns",
        True,
        1.0,
    ),
    (EDMStochasticHeunSolver, {"S_churn": 0}, "stoch_heun_nochurn", False, 1.0),
    (
        EDMStochasticHeunSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_heun_churn",
        True,
        1.0,
    ),
    # EDM schedule with the affine coefficients of the x0-parameterization
    (
        ExponentialEulerSolver,
        {"_use_linear_fn": True, "_use_slope_fn": True},
        "exponential_euler",
        False,
        1.0,
    ),
    # DDIM sampler for distilled few-step models: VP schedule with an
    # x0-parameterization
    (
        ExponentialEulerSolver,
        {"_use_vp_scheduler": True, "_use_linear_fn": True, "_use_slope_fn": True},
        "exponential_euler_ddim",
        False,
        0.1,
    ),
    # EDM-style churn on top of the exponential Euler update
    (
        EDMStochasticExponentialEulerSolver,
        {
            "S_churn": 40,
            "num_steps": 18,
            "_use_linear_fn": True,
            "_use_slope_fn": True,
        },
        "stoch_exp_euler_churn",
        True,
        1.0,
    ),
    # Stochastic DDIM (full noise renewal) for distilled few-step and
    # consistency models: VP schedule with its noise-level callbacks
    (
        EDMStochasticExponentialEulerSolver,
        {
            "renoise": 1.0,
            "_use_vp_scheduler": True,
            "_use_linear_fn": True,
            "_use_slope_fn": True,
            "_use_sigma_fns": True,
        },
        "stoch_exp_euler_renoise",
        True,
        0.1,
    ),
    # Classical two-step Adams-Bashforth: default callbacks
    (DPMPlusPlus2M, {}, "dpmpp_2m_ab2", False, 1.0),
    # Original DPM-Solver++(2M): log-SNR extrapolation coordinate
    (
        DPMPlusPlus2M,
        {"_use_linear_fn": True, "_use_slope_fn": True, "_use_log_snr_lambda": True},
        "dpmpp_2m",
        False,
        1.0,
    ),
    # Corrected two-step Adams-Bashforth: default callbacks
    (DPMPlusPlus2MUniC2, {}, "dpmpp_2m_unic2_default", False, 1.0),
    # DPM-Solver++(2M) with the UniC-2 corrector: log-SNR extrapolation
    # coordinate
    (
        DPMPlusPlus2MUniC2,
        {"_use_linear_fn": True, "_use_slope_fn": True, "_use_log_snr_lambda": True},
        "dpmpp_2m_unic2",
        False,
        1.0,
    ),
]


def _identity_denoiser(x, t):
    return x


def _make_solver_and_denoiser(
    solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
):
    """Create a solver and its deterministic x0-parameterized ODE denoiser,
    resolving the "_use_*" sentinels of the config."""
    kwargs = dict(solver_kwargs)
    if kwargs.pop("_use_vp_scheduler", False):
        scheduler = VPNoiseScheduler()
    else:
        scheduler = EDMNoiseScheduler()
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=0,
        **predictor_kwargs,
    ).to(device)
    denoiser = scheduler.get_denoiser(x0_predictor=model, denoising_type="ode")
    if kwargs.pop("_use_edm_sigma_fns", False):
        kwargs["sigma_fn"] = scheduler.sigma
        kwargs["sigma_inv_fn"] = scheduler.sigma_inv
        kwargs["diffusion_fn"] = scheduler.diffusion
    if kwargs.pop("_use_sigma_fns", False):
        kwargs["sigma_fn"] = scheduler.sigma
        kwargs["sigma_inv_fn"] = scheduler.sigma_inv
        kwargs["alpha_fn"] = scheduler.alpha
    if kwargs.pop("_use_linear_fn", False):
        (
            kwargs["bias_fn"],
            kwargs["bias_int_fn"],
            slope_fn,
        ) = scheduler.get_linear_denoiser(prediction_type="x0")
        if kwargs.pop("_use_slope_fn", False):
            kwargs["slope_fn"] = slope_fn
    if kwargs.pop("_use_log_snr_lambda", False):
        kwargs["lambda_fn"] = lambda t: torch.log(scheduler.snr(t))
    return solver_cls(denoiser, **kwargs), denoiser


# =============================================================================
# Constructor Tests
# =============================================================================


class TestEulerSolverConstructor:
    """Tests for EulerSolver constructor."""

    def test_attributes(self):
        solver = EulerSolver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)


class TestHeunSolverConstructor:
    """Tests for HeunSolver constructor."""

    def test_default_alpha(self):
        solver = HeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)

    def test_custom_alpha(self):
        solver = HeunSolver(_identity_denoiser, alpha=0.5)
        assert solver.alpha == pytest.approx(0.5)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=1.5)


class TestEDMStochasticEulerSolverConstructor:
    """Tests for EDMStochasticEulerSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticEulerSolver(_identity_denoiser)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.S_noise == pytest.approx(1.0)
        assert solver.num_steps == 18

    def test_sigma_fn_validation(self):
        def sigma_only(t):
            return t

        with pytest.raises(ValueError, match="sigma_fn and sigma_inv_fn"):
            EDMStochasticEulerSolver(_identity_denoiser, sigma_fn=sigma_only)


class TestEDMStochasticHeunSolverConstructor:
    """Tests for EDMStochasticHeunSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticHeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)
        assert solver.S_churn == pytest.approx(0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            EDMStochasticHeunSolver(_identity_denoiser, alpha=0.0)


class TestExponentialEulerSolverConstructor:
    """Tests for ExponentialEulerSolver constructor."""

    def test_default_attributes(self):
        solver = ExponentialEulerSolver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        # Default bias and antiderivative are zero and default slope is one
        # (explicit Euler)
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.bias_fn(t) == 0)
        assert torch.all(solver.bias_int_fn(t) == 0)
        assert torch.all(solver.slope_fn(t) == 1)

    def test_custom_bias_and_slope_fns(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        def minus_t_antideriv(t):
            return -t

        def two_coeff(t):
            return 2 * torch.ones_like(t)

        solver = ExponentialEulerSolver(
            _identity_denoiser,
            bias_fn=minus_one_coeff,
            bias_int_fn=minus_t_antideriv,
            slope_fn=two_coeff,
        )
        assert solver.bias_fn is minus_one_coeff
        assert solver.bias_int_fn is minus_t_antideriv
        assert solver.slope_fn is two_coeff

    def test_bias_fn_validation(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        with pytest.raises(ValueError, match="bias_int_fn"):
            ExponentialEulerSolver(_identity_denoiser, bias_fn=minus_one_coeff)
        with pytest.raises(ValueError, match="bias_int_fn"):
            ExponentialEulerSolver(_identity_denoiser, bias_int_fn=minus_one_coeff)


class TestEDMStochasticExponentialEulerSolverConstructor:
    """Tests for EDMStochasticExponentialEulerSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticExponentialEulerSolver(_identity_denoiser)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.renoise == pytest.approx(0.0)
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.bias_fn(t) == 0)
        assert torch.all(solver.bias_int_fn(t) == 0)
        assert torch.all(solver.slope_fn(t) == 1)
        assert torch.all(solver.alpha_fn(t) == 1)

    def test_sigma_fn_validation(self):
        def sigma_only(t):
            return t

        with pytest.raises(ValueError, match="sigma_fn and sigma_inv_fn"):
            EDMStochasticExponentialEulerSolver(_identity_denoiser, sigma_fn=sigma_only)

    def test_bias_fn_validation(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        with pytest.raises(ValueError, match="bias_int_fn"):
            EDMStochasticExponentialEulerSolver(
                _identity_denoiser, bias_fn=minus_one_coeff
            )

    def test_invalid_renoise(self):
        with pytest.raises(ValueError, match="renoise"):
            EDMStochasticExponentialEulerSolver(_identity_denoiser, renoise=1.5)
        with pytest.raises(ValueError, match="renoise"):
            EDMStochasticExponentialEulerSolver(_identity_denoiser, renoise=-0.1)


class TestDPMPlusPlus2MConstructor:
    """Tests for DPMPlusPlus2M constructor."""

    def test_default_attributes(self):
        solver = DPMPlusPlus2M(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        # Default bias is zero and default slope is one with antiderivative t
        # (classical two-step method); the default extrapolation coordinate
        # is diffusion time
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.bias_fn(t) == 0)
        assert torch.all(solver.bias_int_fn(t) == 0)
        assert torch.all(solver.slope_fn(t) == 1)
        assert torch.all(solver.lambda_fn(t) == t)

    def test_custom_bias_and_slope_fns(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        def minus_t_antideriv(t):
            return -t

        def two_coeff(t):
            return 2 * torch.ones_like(t)

        solver = DPMPlusPlus2M(
            _identity_denoiser,
            bias_fn=minus_one_coeff,
            bias_int_fn=minus_t_antideriv,
            slope_fn=two_coeff,
        )
        assert solver.bias_fn is minus_one_coeff
        assert solver.bias_int_fn is minus_t_antideriv
        assert solver.slope_fn is two_coeff

    def test_bias_only_slope_default(self):
        """Without a slope callback, the solver uses a constant slope."""

        def minus_one_coeff(t):
            return -torch.ones_like(t)

        def minus_t_antideriv(t):
            return -t

        solver = DPMPlusPlus2M(
            _identity_denoiser,
            bias_fn=minus_one_coeff,
            bias_int_fn=minus_t_antideriv,
        )
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.slope_fn(t) == 1)

    def test_bias_fn_validation(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        with pytest.raises(ValueError, match="bias_int_fn"):
            DPMPlusPlus2M(_identity_denoiser, bias_fn=minus_one_coeff)

    def test_custom_lambda_fn(self):
        def neg_log_coord(t):
            return -torch.log(t)

        solver = DPMPlusPlus2M(_identity_denoiser, lambda_fn=neg_log_coord)
        assert solver.lambda_fn is neg_log_coord


class TestDPMPlusPlus2MUniC2Constructor:
    """Tests for DPMPlusPlus2MUniC2 constructor."""

    def test_default_attributes(self):
        solver = DPMPlusPlus2MUniC2(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        # Default bias is zero and default slope is one with antiderivative t
        # (corrected classical two-step method); the default extrapolation
        # coordinate is diffusion time
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.bias_fn(t) == 0)
        assert torch.all(solver.bias_int_fn(t) == 0)
        assert torch.all(solver.slope_fn(t) == 1)
        assert torch.all(solver.lambda_fn(t) == t)

    def test_custom_bias_and_slope_fns(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        def minus_t_antideriv(t):
            return -t

        def two_coeff(t):
            return 2 * torch.ones_like(t)

        solver = DPMPlusPlus2MUniC2(
            _identity_denoiser,
            bias_fn=minus_one_coeff,
            bias_int_fn=minus_t_antideriv,
            slope_fn=two_coeff,
        )
        assert solver.bias_fn is minus_one_coeff
        assert solver.bias_int_fn is minus_t_antideriv
        assert solver.slope_fn is two_coeff

    def test_bias_only_slope_default(self):
        """Without a slope callback, the solver uses a constant slope."""

        def minus_one_coeff(t):
            return -torch.ones_like(t)

        def minus_t_antideriv(t):
            return -t

        solver = DPMPlusPlus2MUniC2(
            _identity_denoiser,
            bias_fn=minus_one_coeff,
            bias_int_fn=minus_t_antideriv,
        )
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.slope_fn(t) == 1)

    def test_bias_fn_validation(self):
        def minus_one_coeff(t):
            return -torch.ones_like(t)

        with pytest.raises(ValueError, match="bias_int_fn"):
            DPMPlusPlus2MUniC2(_identity_denoiser, bias_fn=minus_one_coeff)

    def test_custom_lambda_fn(self):
        def neg_log_coord(t):
            return -torch.log(t)

        solver = DPMPlusPlus2MUniC2(_identity_denoiser, lambda_fn=neg_log_coord)
        assert solver.lambda_fn is neg_log_coord


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng,time_scale",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestStepNonRegression:
    """Non-regression tests for solver step() across all solver configs."""

    def test_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        time_scale,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        solver, _ = _make_solver_and_denoiser(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0 * time_scale] * shape[0], device=device)
        t_next = torch.tensor([2.5 * time_scale] * shape[0], device=device)

        ref_file = f"{REF_PREFIX}{solver_name}_{spatial_name}_step.pth"
        if "cuda" in str(device) and uses_rng:

            def fn():
                return solver.step(x, t_cur, t_next)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["x_next"].shape
        else:
            x_next = solver.step(x, t_cur, t_next)
            assert x_next.shape == shape
            ref = load_or_create_reference(ref_file, lambda: {"x_next": x_next.cpu()})
            compare_outputs(x_next, ref["x_next"], **tolerances)

    def test_step_to_zero(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        time_scale,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Step to t=0 should produce finite output."""
        solver, _ = _make_solver_and_denoiser(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )

        x = make_input(shape, seed=101, device=device)
        t_cur = torch.tensor([1.0 * time_scale] * shape[0], device=device)
        t_next = torch.tensor([0.0] * shape[0], device=device)

        x_next = solver.step(x, t_cur, t_next)
        assert x_next.shape == shape
        assert torch.isfinite(x_next).all()

    def test_zero_churn_matches_deterministic(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        time_scale,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Stochastic solvers with S_churn=0 should match their deterministic counterpart."""
        if solver_name == "stoch_euler_nochurn":
            det_cls = EulerSolver
        elif solver_name == "stoch_heun_nochurn":
            det_cls = HeunSolver
        else:
            pytest.skip("Only applies to zero-churn stochastic configs")

        stoch_solver, denoiser = _make_solver_and_denoiser(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )
        det_solver = det_cls(denoiser)

        x = make_input(shape, seed=120, device=device)
        t_cur = torch.tensor([5.0 * time_scale] * shape[0], device=device)
        t_next = torch.tensor([2.5 * time_scale] * shape[0], device=device)

        x_stoch = stoch_solver.step(x, t_cur, t_next)
        x_det = det_solver.step(x, t_cur, t_next)
        compare_outputs(x_stoch, x_det, **tolerances)


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng,time_scale",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestStepConsistency:
    """Consistency of a single step against the known solution of a trivial
    linear ODE, exercising the solvers alone (no noise schedule, no golden
    files)."""

    @pytest.mark.parametrize("t_end_frac", [1e-3, 0.5], ids=["large_step", "half_step"])
    def test_single_step_matches_exact_solution(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        time_scale,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
        t_end_frac,
    ):
        """One step of the trivial ODE dx/dt = (1 / t) x lands on the exact
        solution x(t1) = (t1 / t0) x(t0), which is linear in time, for both
        a large step to near zero and a half step."""
        if uses_rng:
            pytest.skip("Noise injection has no deterministic reference solution")

        def trivial_denoiser(x, t):
            """RHS of the trivial linear ODE dx/dt = (1 / t) x, whose
            solution is linear in time: x(t1) = (t1 / t0) x(t0)."""
            return x / t.reshape((-1,) + (1,) * (x.ndim - 1))

        # Resolve the "_use_*" sentinels of the config with the exact
        # decomposition of this ODE (bias a(t) = 1 / t, slope b(t) = 0)
        # instead of noise-scheduler callbacks, so the test exercises the
        # solver alone
        kwargs = dict(solver_kwargs)
        kwargs.pop("_use_vp_scheduler", False)
        if kwargs.pop("_use_edm_sigma_fns", False):
            kwargs["sigma_fn"] = lambda t: t
            kwargs["sigma_inv_fn"] = lambda sigma: sigma
            kwargs["diffusion_fn"] = lambda x, t: 2 * t.reshape(
                (-1,) + (1,) * (x.ndim - 1)
            )
        if kwargs.pop("_use_sigma_fns", False):
            kwargs["sigma_fn"] = lambda t: t
            kwargs["sigma_inv_fn"] = lambda sigma: sigma
            kwargs["alpha_fn"] = lambda t: torch.ones_like(t)
        if kwargs.pop("_use_linear_fn", False):
            kwargs["bias_fn"] = lambda t: 1 / t
            kwargs["bias_int_fn"] = torch.log
            if kwargs.pop("_use_slope_fn", False):
                kwargs["slope_fn"] = lambda t: torch.zeros_like(t)
        if kwargs.pop("_use_log_snr_lambda", False):
            kwargs["lambda_fn"] = lambda t: -torch.log(t)
        solver = solver_cls(trivial_denoiser, **kwargs)

        x = make_input(shape, seed=102, device=device)
        t_cur = torch.tensor([1.0 * time_scale] * shape[0], device=device)
        t_next = torch.tensor([t_end_frac * time_scale] * shape[0], device=device)

        x_next = solver.step(x, t_cur, t_next)
        compare_outputs(x_next, t_end_frac * x, **tolerances)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng,time_scale",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
@pytest.mark.usefixtures("nop_compile")
class TestStepCompile:
    """Compile tests for solver step() over a multi-step trajectory."""

    def test_compiled_step(
        self,
        deterministic_settings,
        device,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        time_scale,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """A fresh compiled solver steps a trajectory without caller-side
        priming and reuses the steady-state graph."""
        torch._dynamo.config.error_on_recompile = False

        solver, _ = _make_solver_and_denoiser(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )
        compiled_step = torch.compile(solver.step, fullgraph=True)

        x = make_input(shape, seed=100, device=device)
        # Consecutive times of a single trajectory: multistep solvers cache
        # history across calls
        t_traj = [
            torch.tensor([t * time_scale] * shape[0], device=device)
            for t in (7.5, 5.0, 2.5, 1.0)
        ]

        # The first two calls may each compile one specialization: multistep
        # solvers build their history caches on the first step and update
        # them in place afterwards
        outs = []
        with torch.no_grad():
            outs.append(compiled_step(x, t_traj[0], t_traj[1]))
            outs.append(compiled_step(outs[-1], t_traj[1], t_traj[2]))

        # Steady state: every later call must reuse the graph
        torch._dynamo.config.error_on_recompile = True
        with torch.no_grad():
            outs.append(compiled_step(outs[-1], t_traj[2], t_traj[3]))

        for out in outs:
            assert out.shape == shape
            assert torch.isfinite(out).all()

        # For deterministic solvers, verify eager-vs-compiled match over the
        # whole trajectory with a fresh instance
        if not uses_rng:
            solver_eager, _ = _make_solver_and_denoiser(
                solver_cls,
                solver_kwargs,
                shape,
                predictor_cls,
                predictor_kwargs,
                device,
            )
            x_eager = x
            with torch.no_grad():
                for out, t_a, t_b in zip(outs, t_traj[:-1], t_traj[1:]):
                    x_eager = solver_eager.step(x_eager, t_a, t_b)
                    torch.testing.assert_close(x_eager, out)
