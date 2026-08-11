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

import math

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers.solvers import (
    DPMSolverPlusPlus2M,
    EDMStochasticEulerSolver,
    EDMStochasticHeunSolver,
    EulerSolver,
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

# (solver_cls, solver_kwargs, solver_name, uses_rng)
# solver_kwargs are passed to the solver constructor after `denoiser`.
# "_use_edm_sigma_fns" is a sentinel handled by _make_solver.
SOLVER_CONFIGS = [
    (EulerSolver, {}, "euler", False),
    (HeunSolver, {}, "heun", False),
    (HeunSolver, {"alpha": 0.5}, "heun_midpoint", False),
    (EDMStochasticEulerSolver, {"S_churn": 0}, "stoch_euler_nochurn", False),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_euler_churn",
        True,
    ),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10, "_use_edm_sigma_fns": True},
        "stoch_euler_sigmafns",
        True,
    ),
    (EDMStochasticHeunSolver, {"S_churn": 0}, "stoch_heun_nochurn", False),
    (
        EDMStochasticHeunSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_heun_churn",
        True,
    ),
    # Stateful: the parameterized golden files below take a single step from a
    # fresh solver, so they pin only its first-order fallback. The multistep
    # coefficients are covered by TestDPMSolverPlusPlus2M.
    (DPMSolverPlusPlus2M, {}, "dpmpp_2m", False),
]


def _make_denoiser(shape, predictor_cls, predictor_kwargs, device, seed=0):
    """Create a deterministic ODE denoiser from an x0-predictor via EDM scheduler."""
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    scheduler = EDMNoiseScheduler()
    return scheduler.get_denoiser(x0_predictor=model, denoising_type="ode"), model


def _identity_denoiser(x, t):
    return x


def _make_solver(solver_cls, solver_kwargs, denoiser):
    """Create a solver, injecting EDM sigma callbacks if requested."""
    kwargs = dict(solver_kwargs)
    if kwargs.pop("_use_edm_sigma_fns", False):
        edm = EDMNoiseScheduler()
        kwargs["sigma_fn"] = edm.sigma
        kwargs["sigma_inv_fn"] = edm.sigma_inv
        kwargs["diffusion_fn"] = edm.diffusion
    return solver_cls(denoiser, **kwargs)


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


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
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
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

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
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Step to t=0 should produce finite output."""
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=101, device=device)
        t_cur = torch.tensor([1.0] * shape[0], device=device)
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

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        stoch_solver = _make_solver(solver_cls, solver_kwargs, denoiser)
        det_solver = det_cls(denoiser)

        x = make_input(shape, seed=120, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_stoch = stoch_solver.step(x, t_cur, t_next)
        x_det = det_solver.step(x, t_cur, t_next)
        compare_outputs(x_stoch, x_det, **tolerances)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
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
    """Double-call compile tests for solver step()."""

    def test_compiled_step(
        self,
        deterministic_settings,
        device,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Compiled step traces without error and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        # Warm stateful solvers so this generic test covers steady-state graph
        # reuse. Bootstrap compilation is covered by
        # TestDPMSolverPlusPlus2M.test_compile_from_fresh_state_matches_eager.
        if getattr(solver, "_requires_state_reset", False):
            with torch.no_grad():
                solver.step(x, t_cur, t_next)

        compiled_step = torch.compile(solver.step, fullgraph=True)

        with torch.no_grad():
            out_compiled = compiled_step(x, t_cur, t_next)
        assert out_compiled.shape == shape
        assert torch.isfinite(out_compiled).all()

        # Second call — must reuse the graph
        with torch.no_grad():
            out_compiled_2 = compiled_step(x, t_cur, t_next)
        assert out_compiled_2.shape == shape
        assert torch.isfinite(out_compiled_2).all()

        # For deterministic solvers, also verify eager-vs-compiled match
        if not uses_rng:
            with torch.no_grad():
                out_eager = solver.step(x, t_cur, t_next)
            torch.testing.assert_close(out_eager, out_compiled)


# =============================================================================
# DPM-Solver++(2M) Specific Tests
# =============================================================================


def _analytic_denoiser(scale: float = 1.0):
    r"""ODE right-hand side for a Gaussian data distribution with standard deviation ``scale``.

    For :math:`p(x) = \mathcal{N}(0, s^2)` the optimal denoiser is
    :math:`D(x, t) = x s^2 / (s^2 + t^2)`, and the probability-flow ODE has the
    closed-form solution :math:`x(t) = C \sqrt{s^2 + t^2}`. This gives an exact
    reference trajectory to measure the convergence order against.
    """

    def denoiser(x, t):
        t_bc = t.reshape((-1,) + (1,) * (x.ndim - 1))
        D = x * scale**2 / (scale**2 + t_bc**2)
        return (x - D) / t_bc

    return denoiser


def _exact_solution(x_init, t_init, t_final, scale=1.0):
    """Exact PF-ODE solution for the Gaussian data distribution of ``_analytic_denoiser``."""
    return x_init * math.sqrt(scale**2 + t_final**2) / math.sqrt(scale**2 + t_init**2)


class TestDPMSolverPlusPlus2MConstructor:
    """Tests for DPMSolverPlusPlus2M constructor."""

    def test_default_attributes(self):
        solver = DPMSolverPlusPlus2M(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        # Without schedule functions the solver uses the EDM schedule.
        t = torch.tensor(3.0)
        assert solver.alpha_fn(t) == torch.ones_like(t)
        assert solver.sigma_fn(t) == t
        assert solver.alpha_dot_fn(t) == torch.zeros_like(t)
        assert solver.sigma_dot_fn(t) == torch.ones_like(t)


@pytest.mark.usefixtures("deterministic_settings")
class TestDPMSolverPlusPlus2M:
    """Correctness, statefulness and compile behavior of DPM-Solver++(2M)."""

    def test_dbar_is_linear_extrapolation(self, device):
        """The multistep coefficients must extrapolate D(lambda) to lambda + h/2.

        This pins the direction of the step-size ratio ``r = h_prev / h``.
        The test is only sensitive to it on a *non-uniform* ladder: when
        ``h_prev == h`` the correct and inverted coefficients coincide exactly.
        """
        denoiser = _analytic_denoiser()
        solver = DPMSolverPlusPlus2M(denoiser)

        # h_prev = log(2), h = log(4): deliberately non-uniform in lambda.
        t0, t1, t2 = 8.0, 4.0, 1.0
        shape = (BATCH, 3, 8, 6)
        x0 = make_input(shape, seed=7, device=device)

        def as_t(v):
            return torch.full((BATCH,), v, device=device)

        def data_pred(x, t):
            t_bc = torch.full((BATCH,) + (1,) * (len(shape) - 1), t, device=device)
            return x - t_bc * denoiser(x, as_t(t))

        D_prev = data_pred(x0, t0)
        x1 = solver.step(x0, as_t(t0), as_t(t1))
        D = data_pred(x1, t1)

        # Linear interpolant through (lambda_prev, D_prev) and (lambda_cur, D),
        # evaluated at the midpoint lambda_cur + h / 2.
        lam = [-math.log(t) for t in (t0, t1, t2)]
        h_prev, h = lam[1] - lam[0], lam[2] - lam[1]
        coeff = h / (2.0 * h_prev)
        D_bar = (1.0 + coeff) * D - coeff * D_prev

        ratio = t2 / t1
        expected = ratio * x1 + (1.0 - ratio) * D_bar

        x2 = solver.step(x1, as_t(t1), as_t(t2))
        torch.testing.assert_close(x2, expected, rtol=1e-5, atol=1e-6)

    def test_first_step_matches_euler(self, device):
        """With no history the update reduces to an explicit Euler step in sigma."""
        denoiser = _analytic_denoiser()
        shape = (BATCH, 3, 8, 6)
        x = make_input(shape, seed=11, device=device)
        t_cur = torch.full((BATCH,), 5.0, device=device)
        t_next = torch.full((BATCH,), 2.5, device=device)

        dpm = DPMSolverPlusPlus2M(denoiser).step(x, t_cur, t_next)
        euler = EulerSolver(denoiser).step(x, t_cur, t_next)
        torch.testing.assert_close(dpm, euler, rtol=1e-5, atol=1e-6)

    @staticmethod
    def _integrate(solver_cls, num_steps, device, scale=1.0, t_max=80.0, t_min=2e-3):
        """Integrate the analytic PF-ODE on a Karras rho=7 ladder."""
        rho = 7.0
        ts = [
            (
                t_max ** (1 / rho)
                + i / (num_steps - 1) * (t_min ** (1 / rho) - t_max ** (1 / rho))
            )
            ** rho
            for i in range(num_steps)
        ]
        solver = solver_cls(_analytic_denoiser(scale))
        x = make_input((1, 4), seed=3, device=device) * t_max
        for t_cur, t_next in zip(ts[:-1], ts[1:]):
            x = solver.step(
                x,
                torch.full((1,), t_cur, device=device),
                torch.full((1,), t_next, device=device),
            )
        exact = _exact_solution(
            make_input((1, 4), seed=3, device=device) * t_max, t_max, t_min, scale
        )
        return float((x - exact).abs().max())

    def test_convergence_and_accuracy_vs_euler(self, device):
        """Verify asymptotic convergence and better accuracy than Euler at equal NFE.

        Only the asymptotic range is asserted: on a coarse ladder a wrong scheme
        can be accidentally more accurate through error cancellation, so a
        single-step-count threshold would not discriminate.
        """
        errors = [
            self._integrate(DPMSolverPlusPlus2M, n, device) for n in (8, 16, 32, 64)
        ]
        assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), (
            f"error did not decrease monotonically: {errors}"
        )

        euler = self._integrate(EulerSolver, 64, device)
        assert errors[-1] < euler / 5.0, (
            f"dpmpp_2m={errors[-1]:.3e} vs euler={euler:.3e}"
        )

    def test_reset_restores_first_order_path(self, device):
        denoiser = _analytic_denoiser()
        solver = DPMSolverPlusPlus2M(denoiser)
        shape = (BATCH, 3, 8, 6)
        x_init = make_input(shape, seed=5, device=device)
        ts = [40.0, 12.0, 3.0, 0.4]

        def run():
            solver.reset()
            x = x_init
            for t_cur, t_next in zip(ts[:-1], ts[1:]):
                x = solver.step(
                    x,
                    torch.full((BATCH,), t_cur, device=device),
                    torch.full((BATCH,), t_next, device=device),
                )
            return x

        first, second = run(), run()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        # Without the reset the stale history changes the result.
        x = x_init
        for t_cur, t_next in zip(ts[:-1], ts[1:]):
            x = solver.step(
                x,
                torch.full((BATCH,), t_cur, device=device),
                torch.full((BATCH,), t_next, device=device),
            )
        assert not torch.allclose(x, first)

    def test_one_denoiser_call_per_step(self, device):
        calls = []
        inner = _analytic_denoiser()

        def counting_denoiser(x, t):
            calls.append(1)
            return inner(x, t)

        solver = DPMSolverPlusPlus2M(counting_denoiser)
        x = make_input((BATCH, 3, 8, 6), seed=13, device=device)
        ts = [40.0, 12.0, 3.0, 0.4, 0.0]
        for t_cur, t_next in zip(ts[:-1], ts[1:]):
            x = solver.step(
                x,
                torch.full((BATCH,), t_cur, device=device),
                torch.full((BATCH,), t_next, device=device),
            )
        assert len(calls) == len(ts) - 1

    def test_rounded_duplicate_timesteps_stay_finite(self, device):
        """A ladder that collides only after casting must not poison the cache.

        Two adjacent timesteps that round to the same low-precision value give a
        zero step size in lambda, which makes the multistep coefficient
        singular. That step must be the identity and the next must fall back to
        first order. This is how the case arises in practice: ``sample`` casts
        the timesteps to the latent dtype, so a fine ladder loses distinctions
        the schedule intended. Exact repeats also occur at full precision -- the
        iDDPM ladder yields them at large step counts even in float64.
        """
        denoiser = _analytic_denoiser()
        solver = DPMSolverPlusPlus2M(denoiser)
        x = make_input((BATCH, 3, 8, 6), seed=17, device=device).to(torch.bfloat16)

        # Strictly decreasing in float32; ts[1] and ts[2] collide in bfloat16.
        ts = [40.0, 12.01, 12.0, 3.0, 0.4, 0.0]
        cast = [
            torch.full((BATCH,), t, device=device, dtype=torch.bfloat16) for t in ts
        ]
        assert all(a > b for a, b in zip(ts[:-1], ts[1:])), "ladder must decrease"
        assert torch.equal(cast[1], cast[2]), "ts[1] and ts[2] must collide in bfloat16"

        collision = 1  # index of the step whose endpoints collide after casting
        for i, (t_cur, t_next) in enumerate(zip(cast[:-1], cast[1:])):
            x_prev = x
            x = solver.step(x, t_cur, t_next)
            assert torch.isfinite(x).all(), (
                f"non-finite at step {i} ({ts[i]}->{ts[i + 1]})"
            )
            if i == collision:
                # A zero-length step must be exactly the identity.
                torch.testing.assert_close(x, x_prev, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "sched_cls", [VPNoiseScheduler, VENoiseScheduler], ids=["vp", "ve"]
    )
    def test_converges_on_vp_and_ve_schedules(self, device, sched_cls):
        """Second-order behavior is not specific to the EDM parameterization.

        Results are compared with the exact solution because solvers can use
        different terminal updates. The window is deliberately narrow and the
        assertions broad: coarse ladders are pre-asymptotic, and on VE the error
        changes sign near 150 steps, so an order estimated across that crossing
        is meaningless.
        """
        sched = sched_cls()
        scale = 1.0

        def x0_predictor(x, t):
            t_bc = t.reshape((-1,) + (1,) * (x.ndim - 1))
            a, sg = sched.alpha(t_bc), sched.sigma(t_bc)
            return x * a * scale**2 / (a**2 * scale**2 + sg**2)

        denoiser = sched.get_denoiser(x0_predictor=x0_predictor, denoising_type="ode")

        errors = []
        for num_steps in (24, 32, 48, 64):
            ts = sched.timesteps(num_steps, device=device, dtype=torch.float64)
            a0, s0 = sched.alpha(ts[0]), sched.sigma(ts[0])
            aT, sT = sched.alpha(ts[-1]), sched.sigma(ts[-1])
            scale0 = float(torch.sqrt(a0**2 * scale**2 + s0**2))
            xT = make_input((1, 64), seed=41, device=device).double() * scale0
            exact = xT * float(torch.sqrt(aT**2 * scale**2 + sT**2)) / scale0

            solver = DPMSolverPlusPlus2M(
                denoiser,
                alpha_fn=sched.alpha,
                sigma_fn=sched.sigma,
                alpha_dot_fn=sched.alpha_dot,
                sigma_dot_fn=sched.sigma_dot,
            )
            x = xT
            for t_cur, t_next in zip(ts[:-1], ts[1:]):
                x = solver.step(x, t_cur.expand(1), t_next.expand(1))
            errors.append(float((x - exact).abs().max() / exact.abs().max()))

        assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), (
            f"error did not decrease monotonically: {errors}"
        )
        # Refining 24 -> 64 steps is a factor 8/3; a second-order method gains
        # roughly (8/3)^2 ~ 7x. Assert well below that to stay robust.
        assert errors[0] / errors[-1] > 3.0, f"convergence too slow: {errors}"

    def test_rejects_partial_schedule_functions(self):
        """Partially supplied schedule functions must be rejected.

        Accepting a subset would silently combine a custom schedule with the
        EDM defaults for the rest, which integrates a different ODE than the
        caller intended.
        """
        denoiser = _analytic_denoiser()
        with pytest.raises(ValueError, match="together or not at all"):
            DPMSolverPlusPlus2M(denoiser, sigma_fn=lambda t: t)

    def test_terminal_step_scales_the_data_prediction_by_alpha(self):
        """The final step must return ``alpha_next * D``, not ``D``.

        Every shipped scheduler has ``alpha == 1`` at the zero-noise endpoint
        and reaches ``sigma == 0`` only at ``t == 0``, so neither the factor
        nor the detection on ``sigma`` rather than ``t`` is observable there.
        This schedule has ``alpha(1) = 1.5`` and ``sigma(1) = 0``, separating
        both.
        """
        rhs = torch.full((1, 4), 0.25)
        solver = DPMSolverPlusPlus2M(
            lambda x, t: rhs,
            alpha_fn=lambda t: 1.0 + t / 2.0,
            sigma_fn=lambda t: t - 1.0,
            alpha_dot_fn=lambda t: torch.full_like(t, 0.5),
            sigma_dot_fn=torch.ones_like,
        )
        x = torch.linspace(-1.0, 1.0, 4).reshape(1, 4)
        # One ordinary step first: without history the extrapolation equals D,
        # and the general branch would land on the same value as the terminal
        # one, so detecting the final step on sigma would not be observable.
        x = solver.step(x, torch.tensor([3.0]), torch.tensor([2.0]))
        # D = (sigma_dot x - sigma rhs) / (alpha sigma_dot - sigma alpha_dot)
        # is (x - rhs) / 1.5 at t = 2, so alpha_next * D is exactly x - rhs.
        x_next = solver.step(x, torch.tensor([2.0]), torch.tensor([1.0]))
        torch.testing.assert_close(x_next, x - rhs)

    def test_zero_sigma_is_the_identity_and_differentiable(self, device):
        """Repeated and zero timesteps stay finite and differentiable.

        The denoiser returns ``(x - D) / t`` and is singular at zero, so it runs
        on a surrogate time and its result is discarded; the step is the
        identity. A ladder rounded to a low-precision dtype can underflow to
        zero before its final entry, so this is reachable in practice.
        """
        denoiser = _analytic_denoiser()
        solver = DPMSolverPlusPlus2M(denoiser)
        x = make_input((BATCH, 3, 8, 6), seed=29, device=device).requires_grad_(True)

        out = x
        for t_cur, t_next in zip(
            [40.0, 12.0, 12.0, 3.0, 0.0], [12.0, 12.0, 3.0, 0.0, 0.0]
        ):
            before = out
            out = solver.step(
                out,
                torch.full((BATCH,), t_cur, device=device),
                torch.full((BATCH,), t_next, device=device),
            )
            assert torch.isfinite(out).all(), f"non-finite at t={t_cur}->{t_next}"

        # The final entry steps from t_cur == 0, which must be an exact no-op.
        torch.testing.assert_close(out, before, rtol=0, atol=0)

        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile_from_fresh_state_matches_eager(self, device):
        """Compiling ``step`` from the reset state must reproduce an eager run.

        The shared TestStepCompile warms the solver first, so the first-order
        bootstrap is only ever traced here.
        """
        torch._dynamo.reset()
        denoiser = _analytic_denoiser()
        ts = [80.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.0]

        def run(solver, compile_step):
            # step_fn must be bound to the same solver that is reset here, or
            # the trajectory runs on another instance's history.
            step_fn = (
                torch.compile(solver.step, fullgraph=True)
                if compile_step
                else solver.step
            )
            solver.reset()
            x = make_input((BATCH, 3, 8, 6), seed=23, device=device)
            for t_cur, t_next in zip(ts[:-1], ts[1:]):
                x = step_fn(
                    x,
                    torch.full((BATCH,), t_cur, device=device),
                    torch.full((BATCH,), t_next, device=device),
                )
            return x

        with torch.no_grad():
            compiled = run(DPMSolverPlusPlus2M(denoiser), compile_step=True)
            eager = run(DPMSolverPlusPlus2M(denoiser), compile_step=False)

        assert torch.isfinite(compiled).all()
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)
