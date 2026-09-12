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
"""
Regression tests for gray-box closure discovery + physics verifiers.

The tests exercise the *mechanism* through the standalone reference
implementation (numpy only), so they run in CI without a GPU/torch environment.
An additional, guarded test checks that the PhysicsNeMo verifier helpers import
and construct when PhysicsNeMo is available.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from graybox_reference import run_wedge  # noqa: E402


@pytest.fixture(scope="module")
def metrics():
    """Run the reference mechanism once and share the metrics across tests."""
    # small ensemble / short training keeps the test fast but stable
    return run_wedge(seed_count=3, iters=2500)


def test_symbolic_seam_addresses_the_unknown_term(metrics):
    """The discoverable term must be located on the Add.make_args seam (index 1)."""
    assert metrics["discover_idx"] == [1]


def test_verifier_reduces_non_identifiability(metrics):
    """The physics verifier must shrink the ensemble spread in the unobserved region."""
    assert metrics["gaming_ver"] < metrics["gaming_no"] / 2.0


def test_verifier_recovers_out_of_coverage_closure(metrics):
    """With verifiers, the recovered closure must be close to the true one everywhere."""
    assert metrics["err_ver"] < 0.06
    # and clearly better than the unconstrained fit
    assert metrics["err_ver"] < metrics["err_no"]


def test_physicsnemo_verifier_helpers_importable():
    """Check the PhysicsNeMo verifier helpers import when PhysicsNeMo is available."""
    pytest.importorskip("physicsnemo")  # skipped unless PhysicsNeMo is installed
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "..", "physicsnemo", "sym", "eq"),
    )
    # verifiers.py ships in a follow-up PR; skip here rather than fail until it lands.
    verifiers = pytest.importorskip("verifiers")
    assert hasattr(verifiers, "equilibrium_verifier")
    assert hasattr(verifiers, "SignConstraint")
