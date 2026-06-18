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

"""Static signature-contract tests for the dataclass-driven entry points.

These tests parse the project's source files with ``ast`` (no runtime imports
of third-party-dependent modules) and check that the major refactored
entry points expose the parameters the rest of the codebase expects.

The point is to catch silent breakage of the kind that has bitten this
project before: a function gets a renamed/removed parameter but a caller
still passes the old name. By asserting on parameter *names*, these tests
fail loudly the moment the contract drifts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))


def _function_params(file_rel: str, func_name: str) -> list[str]:
    """Return the positional + keyword parameter names of *func_name* in *file_rel*.

    Parameters
    ----------
    file_rel : str
        Path to a Python source file, relative to ``src/``.
    func_name : str
        Name of the top-level function (or class method) to inspect.
    """
    src = (SRC_ROOT / file_rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return [a.arg for a in node.args.args] + [
                a.arg for a in node.args.kwonlyargs
            ]
    raise AssertionError(f"{func_name} not found in {file_rel}")


# ---------------------------------------------------------------------------
# train.py: training_step / _validation_step_impl must accept ``physics`` and
# ``norm`` so callers can pass the dataclass-bundled physics constants.
# ---------------------------------------------------------------------------


def test_training_step_accepts_physics_and_norm():
    """Test training step accepts physics and norm."""
    params = _function_params("train.py", "training_step")
    assert "physics" in params, "training_step must accept the PhysicsParams bundle"
    assert "norm" in params, "training_step must accept the NormParams bundle"


def test_validation_step_impl_accepts_physics_and_norm():
    """Test validation step impl accepts physics and norm."""
    params = _function_params("train.py", "_validation_step_impl")
    assert "physics" in params
    assert "norm" in params


def test_training_step_keeps_existing_positional_contract():
    """The first 14 args are still the legacy contract used by run_training_loop's call site."""
    params = _function_params("train.py", "training_step")
    expected_prefix = [
        "model",
        "inputin",
        "inputin_p",
        "TARGETS",
        "cfg",
        "device",
        "input_keys",
        "output_keys_saturation",
        "steppi",
        "output_variables",
        "training_step_metrics",
        "neededM",
        "neededMx",
        "epoch",
    ]
    assert params[: len(expected_prefix)] == expected_prefix


# ---------------------------------------------------------------------------
# forward/utils/sequential/training_function.py: run_training_loop must accept
# the full dataclass bundle so train.py's main() compiles.
# ---------------------------------------------------------------------------


def test_run_training_loop_accepts_dataclasses():
    """Test run training loop accepts dataclasses."""
    params = _function_params(
        "forward/utils/sequential/training_function.py", "run_training_loop"
    )
    for required in (
        "dist",
        "logger",
        "cfg",
        "mlflow",
        "use_epoch",
        "pde_method",
        "models",
        "loaders",
        "keys",
        "physics",
        "norm",
        "optimizers",
        "schedulers",
        "state",
    ):
        assert required in params, f"run_training_loop must accept {required!r}"


# ---------------------------------------------------------------------------
# inverse/history_matching.py: run_history_matching_loop is the inverse-side
# entry point and must accept the full dataclass bundle.
# ---------------------------------------------------------------------------


def test_run_history_matching_loop_accepts_dataclasses():
    """Test run history matching loop accepts dataclasses."""
    params = _function_params(
        "inverse/history_matching.py", "run_history_matching_loop"
    )
    for required in (
        "dist",
        "logger",
        "cfg",
        "device",
        "oldfolder",
        "gpu_available",
        "iteration_converged",
        "iteration_count",
        "input_variables",
        "output_variables",
        "ensemble",
        "ensemblep",
        "ensemblef",
        "CDd",
        "True_mat",
        "perturbations",
        "grid",
        "norm",
        "perm",
        "ens",
        "well",
        "surrogate",
        "inversion",
        "time_arr",
        "flow",
    ):
        assert required in params, f"run_history_matching_loop must accept {required!r}"


# ---------------------------------------------------------------------------
# inverse/utils/ensemble_results.py: process_final_results must accept the
# inverse-config dataclasses.
# ---------------------------------------------------------------------------


def test_process_final_results_accepts_dataclasses():
    """Test process final results accepts dataclasses."""
    params = _function_params(
        "inverse/utils/ensemble_results.py", "process_final_results"
    )
    for required in (
        "grid",
        "norm",
        "perm",
        "ens",
        "well",
        "surrogate",
        "time_arr",
        "flow",
    ):
        assert required in params


# ---------------------------------------------------------------------------
# compare_and_analyze_results was reduced from 68 positional params to 9
# dataclass-bundled groups. Lock in the new contract.
# ---------------------------------------------------------------------------


def test_setup_models_and_data_takes_dataclass_bundles():
    """``setup_models_and_data`` was reduced from 23 positional args to 9."""
    params = _function_params(
        "inverse/utils/ensemble_generation.py", "setup_models_and_data"
    )
    assert params == [
        "input_variables",
        "output_variables",
        "runtime",
        "grid",
        "well",
        "priors",
        "Ne",
        "minK",
        "maxK",
    ], (
        "setup_models_and_data must take exactly the 9 dataclass-bundled args; "
        f"got {params}"
    )


def test_compare_and_analyze_results_takes_dataclass_bundles():
    """Test compare and analyze results takes dataclass bundles."""
    params = _function_params(
        "compare/sequential/utils/misc_utils.py", "compare_and_analyze_results"
    )
    assert params == [
        "timing",
        "grid",
        "fields",
        "wells",
        "well_results",
        "runtime",
        "norms",
        "surrogate",
        "flow",
    ], (
        "compare_and_analyze_results must take exactly the 9 dataclass bundles; "
        f"got {params}"
    )


# ---------------------------------------------------------------------------
# Forward_model_ensemble — the central physics-surrogate entry point. Verify
# its parameter list still matches what the inverse + compare paths pass.
# ---------------------------------------------------------------------------


def test_forward_model_ensemble_signature():
    """Test forward model ensemble signature."""
    params = _function_params("utils/ccr_utils.py", "Forward_model_ensemble")
    # Spot-check the most-relied-on parameter names. If any of these drift,
    # the dataclass-driven callers in run_history_matching_loop and
    # process_final_results will silently break.
    for required in (
        "N",
        "x_true",
        "steppi",
        "min_inn_fcn",
        "max_inn_fcn",
        "target_min",
        "target_max",
        "minK",
        "maxK",
        "minT",
        "maxT",
        "minP",
        "maxP",
        "models",
        "device",
        "min_out_fcn",
        "max_out_fcn",
        "Time",
        "active_cells_ensemble",
        "Trainmoe",
        "num_cores",
        "pred_type",
        "oldfolder",
        "degg",
        "experts",
        "min_out_fcn2",
        "max_out_fcn2",
        "min_inn_fcn2",
        "max_inn_fcn2",
        "producer_wells",
        "unique_entries",
        "output_variables",
        "well_measurements",
        "cfg",
        "N_pr",
        "lenwels",
        "active_mask_3d",
        "awater",
        "agas",
        "aoil",
        "aqq",
        "nx",
        "ny",
        "nz",
        "minQ",
        "maxQ",
        "minQw",
        "maxQw",
        "minQg",
        "maxQg",
    ):
        assert required in params, f"Forward_model_ensemble must accept {required!r}"


# ---------------------------------------------------------------------------
# pyproject.toml is parseable and configures ruff sensibly.
# ---------------------------------------------------------------------------


def test_pyproject_toml_is_valid_and_configures_ruff():
    """Test pyproject toml is valid and configures ruff."""
    pyproject = SRC_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml should exist at the project root"

    if sys.version_info >= (3, 11):
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    else:
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("tomllib/tomli not available on this Python")
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    ruff = data.get("tool", {}).get("ruff", {})
    assert ruff.get("line-length") == 100
    lint = ruff.get("lint", {})
    # The disabled rules document our domain choices; if they go missing it's
    # almost certainly an accidental config regression.
    for rule in ("E501", "S101", "S301", "S311"):
        assert rule in lint.get("ignore", []), f"{rule} should be ignored project-wide"
    # Tests must keep their per-file ignore for ``assert`` and unused fixtures.
    assert "tests/*.py" in lint.get("per-file-ignores", {})


# ---------------------------------------------------------------------------
# The two relative-permeability polynomial bindings are passed to
# train_polynomial_models with the *correct* SWOW vs. SWOG tables. This was
# the cited correctness bug in Peter's review; lock it down with a static
# check so it cannot regress.
# ---------------------------------------------------------------------------


def test_train_polynomial_models_uses_matching_tables():
    """Test train polynomial models uses matching tables."""
    src = (SRC_ROOT / "train.py").read_text(encoding="utf-8")
    # Both lines must coexist in the file. If a future refactor drops one or
    # swaps SWOW/SWOG between the two assignments, this test fails.
    assert "params1_swow, params2_swow = train_polynomial_models(SWOW," in src, (
        "SWOW polynomial bindings must use the SWOW table"
    )
    assert "params1_swog, params2_swog = train_polynomial_models(SWOG," in src, (
        "SWOG polynomial bindings must use the SWOG table"
    )


# ---------------------------------------------------------------------------
# Subprocess hardening: no remaining ``shell=True`` outside the explicitly
# audited locations, and no remaining ``os.system(...)`` calls.
# ---------------------------------------------------------------------------


def test_no_unaudited_shell_true():
    """``shell=True`` should not appear in any actual subprocess call.

    AST-based check: only flags ``shell=True`` when it is a real keyword
    argument to a Call node, not a substring inside a comment, docstring,
    or this very test file.
    """
    offenders = []
    for py in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts or py.name.startswith("_"):
            continue
        if py.resolve() == Path(__file__).resolve():
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        offenders.extend(
            f"{py.relative_to(SRC_ROOT).as_posix()}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "shell"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
        )
    assert offenders == [], f"shell=True still present in: {offenders}"


def test_no_os_system_calls():
    """``os.system(...)`` is replaced with subprocess + shlex throughout."""
    offenders = []
    for py in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts or py.name.startswith("_"):
            continue
        src = py.read_text(encoding="utf-8")
        # Only flag actual call sites, not the substring inside docstrings.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (
                    isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "os"
                    and f.attr == "system"
                ):
                    offenders.append(
                        f"{py.relative_to(SRC_ROOT).as_posix()}:{node.lineno}"
                    )
    assert offenders == [], f"os.system(...) call(s) still present: {offenders}"
