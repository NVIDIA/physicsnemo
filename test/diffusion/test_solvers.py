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

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.samplers import (
    DPMPlusPlus2M,
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

# (solver_cls, solver_kwargs, solver_name, uses_rng)
# solver_kwargs are passed to the solver constructor after `denoiser`.
# "_use_edm_sigma_fns" and "_use_linear_fn" are sentinels handled by
# _make_solver: they select EDM schedule callbacks and the linear coefficient
# of the semi-linear split.
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
    (
        ExponentialEulerSolver,
        {"_use_linear_fn": True},
        "exponential_euler",
        False,
    ),
    (
        EDMStochasticExponentialEulerSolver,
        {"S_churn": 0, "_use_linear_fn": True},
        "stoch_exp_euler_nochurn",
        False,
    ),
    (
        EDMStochasticExponentialEulerSolver,
        {
            "S_churn": 40,
            "num_steps": 10,
            "_use_edm_sigma_fns": True,
            "_use_linear_fn": True,
        },
        "stoch_exp_euler_churn",
        True,
    ),
    (
        EDMStochasticExponentialEulerSolver,
        {"S_churn": 0, "renoise": 1.0, "_use_linear_fn": True},
        "stoch_exp_euler_renoise",
        True,
    ),
    (DPMPlusPlus2M, {"_use_linear_fn": True}, "dpmpp_2m", False),
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


def _minus_one_coeff(t):
    return -torch.ones_like(t)


def _make_solver(solver_cls, solver_kwargs, denoiser):
    """Create a solver, resolving the "_use_*" sentinels."""
    kwargs = dict(solver_kwargs)
    if kwargs.pop("_use_edm_sigma_fns", False):
        edm = EDMNoiseScheduler()
        kwargs["sigma_fn"] = edm.sigma
        kwargs["sigma_inv_fn"] = edm.sigma_inv
        kwargs["diffusion_fn"] = edm.diffusion
    if kwargs.pop("_use_linear_fn", False):
        kwargs["linear_fn"] = EDMNoiseScheduler().get_linear_denoiser(
            prediction_type="x0"
        )
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


class TestExponentialEulerSolverConstructor:
    """Tests for ExponentialEulerSolver constructor."""

    def test_default_attributes(self):
        solver = ExponentialEulerSolver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        # Default linear coefficient is zero (explicit Euler)
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.linear_fn(t) == 0)

    def test_custom_linear_fn(self):
        solver = ExponentialEulerSolver(_identity_denoiser, linear_fn=_minus_one_coeff)
        assert solver.linear_fn is _minus_one_coeff


class TestEDMStochasticExponentialEulerSolverConstructor:
    """Tests for EDMStochasticExponentialEulerSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticExponentialEulerSolver(_identity_denoiser)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.renoise == pytest.approx(0.0)
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.linear_fn(t) == 0)

    def test_sigma_fn_validation(self):
        def sigma_only(t):
            return t

        with pytest.raises(ValueError, match="sigma_fn and sigma_inv_fn"):
            EDMStochasticExponentialEulerSolver(_identity_denoiser, sigma_fn=sigma_only)

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
        # Default linear coefficient is zero (classical two-step method)
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.linear_fn(t) == 0)

    def test_custom_linear_fn(self):
        solver = DPMPlusPlus2M(_identity_denoiser, linear_fn=_minus_one_coeff)
        assert solver.linear_fn is _minus_one_coeff


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestConsistency:
    """Cross-solver consistency checks on closed-form identities."""

    @staticmethod
    def _score_components(shape, predictor_cls, predictor_kwargs, device):
        """Score-parameterized denoiser and linear coefficient on EDM."""
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = EDMNoiseScheduler()

        def score_pred(x, t):
            return scheduler.x0_to_score(model(x, t), x, t)

        denoiser = scheduler.get_denoiser(score_predictor=score_pred)
        linear_fn = scheduler.get_linear_denoiser(prediction_type="score")
        return scheduler, model, denoiser, linear_fn

    def test_exponential_euler_matches_euler_on_edm(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """With a score parameterization on EDM, the linear coefficient
        vanishes, so the exponential Euler update must coincide with the
        explicit Euler update on the same denoiser."""
        _, _, denoiser, linear_fn = self._score_components(
            shape, predictor_cls, predictor_kwargs, device
        )
        exp_solver = ExponentialEulerSolver(denoiser, linear_fn=linear_fn)
        euler_solver = EulerSolver(denoiser)

        x = make_input(shape, seed=130, device=device)
        for t_cur_val, t_next_val in [(5.0, 2.5), (1.0, 0.0)]:
            t_cur = torch.tensor([t_cur_val] * shape[0], device=device)
            t_next = torch.tensor([t_next_val] * shape[0], device=device)
            x_exp = exp_solver.step(x, t_cur, t_next)
            x_euler = euler_solver.step(x, t_cur, t_next)
            compare_outputs(x_exp, x_euler, **tolerances)

    def test_exponential_euler_matches_renoised_prediction_on_edm(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """With a score parameterization on EDM, the exponential Euler step
        equals re-noising the data prediction with the noise inferred from
        the current state (the DDIM update)."""
        scheduler, model, denoiser, linear_fn = self._score_components(
            shape, predictor_cls, predictor_kwargs, device
        )
        solver = ExponentialEulerSolver(denoiser, linear_fn=linear_fn)

        x = make_input(shape, seed=131, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_solver = solver.step(x, t_cur, t_next)

        x0 = model(x, t_cur)
        eps = scheduler.x0_to_epsilon(x0, x, t_cur)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_next_bc = t_next.reshape(expected_shape)
        x_renoised = scheduler.alpha(t_next_bc) * x0 + scheduler.sigma(t_next_bc) * eps
        compare_outputs(x_solver, x_renoised, **tolerances)

    def test_renoise_full_restart_returns_data_prediction_at_zero_noise(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """At t_next = 0 the arrival noise level is zero, so the fully
        re-noised step returns the data prediction exactly."""
        _, model, denoiser, linear_fn = self._score_components(
            shape, predictor_cls, predictor_kwargs, device
        )
        solver = EDMStochasticExponentialEulerSolver(
            denoiser, linear_fn=linear_fn, S_churn=0, renoise=1.0
        )

        x = make_input(shape, seed=135, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([0.0] * shape[0], device=device)
        compare_outputs(solver.step(x, t_cur, t_next), model(x, t_cur), **tolerances)

    def test_dpmpp_first_step_matches_exponential_euler(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """The first DPM-Solver++(2M) step has no history, so it equals a
        first-order exponential Euler step on the same semi-linear split."""
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        linear_fn = EDMNoiseScheduler().get_linear_denoiser(prediction_type="x0")
        dpmpp = DPMPlusPlus2M(denoiser, linear_fn=linear_fn)
        exp_euler = ExponentialEulerSolver(denoiser, linear_fn=linear_fn)

        x = make_input(shape, seed=136, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)
        compare_outputs(
            dpmpp.step(x, t_cur, t_next),
            exp_euler.step(x, t_cur, t_next),
            **tolerances,
        )

    def test_dpmpp_reset_restores_first_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """reset() clears the history: the next step reproduces a fresh
        instance's first step."""
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = DPMPlusPlus2M(
            denoiser,
            linear_fn=EDMNoiseScheduler().get_linear_denoiser(prediction_type="x0"),
        )

        x = make_input(shape, seed=137, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_first = solver.step(x, t_cur, t_next)
        solver.step(x_first, t_next, torch.tensor([1.0] * shape[0], device=device))
        solver.reset()
        x_after_reset = solver.step(x, t_cur, t_next)
        compare_outputs(x_after_reset, x_first, **tolerances)


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
            det_cls, det_kwargs = EulerSolver, {}
        elif solver_name == "stoch_heun_nochurn":
            det_cls, det_kwargs = HeunSolver, {}
        elif solver_name == "stoch_exp_euler_nochurn":
            det_cls, det_kwargs = ExponentialEulerSolver, {"_use_linear_fn": True}
        else:
            pytest.skip("Only applies to zero-churn stochastic configs")

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        stoch_solver = _make_solver(solver_cls, solver_kwargs, denoiser)
        det_solver = _make_solver(det_cls, det_kwargs, denoiser)

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
        t_prev = torch.tensor([7.5] * shape[0], device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        def prime(s):
            # Multistep solvers specialize on their empty history; give them
            # one eager step so that compilation traces the steady state
            if hasattr(s, "reset"):
                with torch.no_grad():
                    s.step(x, t_prev, t_cur)

        prime(solver)
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
        # against a fresh instance primed the same way
        if not uses_rng:
            solver_eager = _make_solver(solver_cls, solver_kwargs, denoiser)
            prime(solver_eager)
            with torch.no_grad():
                out_eager = solver_eager.step(x, t_cur, t_next)
            torch.testing.assert_close(out_eager, out_compiled)
