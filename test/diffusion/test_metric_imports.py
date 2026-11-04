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

"""Public import migration and numerical consistency for diffusion metrics."""

import importlib
import subprocess
import sys

import pytest
import torch

from physicsnemo.diffusion.metrics import (
    calculate_fid_from_inception_stats,
    wasserstein_from_normal,
)
from physicsnemo.diffusion.metrics.wasserstein import (
    wasserstein_from_normal as canonical,
)
from physicsnemo.metrics.general.wasserstein import (
    wasserstein_from_normal as original,
)

REMOVED_METRICS = (
    ("rank_probability_score", "calibration"),
    ("crps", "crps"),
    ("kcrps", "crps"),
    ("EnsembleMetrics", "ensemble_metrics"),
    ("Mean", "ensemble_metrics"),
    ("Variance", "ensemble_metrics"),
    ("entropy_from_counts", "entropy"),
    ("relative_entropy_from_counts", "entropy"),
    ("Histogram", "histogram"),
    ("cdf", "histogram"),
    ("histogram", "histogram"),
    ("normal_cdf", "histogram"),
    ("normal_pdf", "histogram"),
    ("power_spectrum", "power_spectrum"),
    ("wasserstein_from_cdf", "wasserstein"),
    ("wasserstein_from_samples", "wasserstein"),
)


class TestDiagnostics:
    @pytest.mark.parametrize("name,module", REMOVED_METRICS)
    def test_removed_metric_import(self, name, module):
        replacement = f"from physicsnemo.metrics.general.{module} import {name}"
        with pytest.warns(FutureWarning, match=replacement):
            with pytest.raises(ImportError, match=replacement):
                __import__("physicsnemo.diffusion.metrics", fromlist=[name])
        assert callable(
            getattr(
                importlib.import_module(f"physicsnemo.metrics.general.{module}"), name
            )
        )

    def test_unknown_attribute(self):
        metrics = importlib.import_module("physicsnemo.diffusion.metrics")
        with pytest.raises(AttributeError, match="unknown_metric"):
            getattr(metrics, "unknown_metric")

    def test_imports_without_models_or_general_metrics(self):
        # A fresh interpreter catches eager dependencies hidden by pytest collection.
        code = """
import importlib.abc
import sys

class BlockUmbrellaImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("physicsnemo.models", "physicsnemo.metrics")):
            raise ImportError(f"Unexpected dependency: {fullname}")

sys.meta_path.insert(0, BlockUmbrellaImports())
from physicsnemo.diffusion.metrics import wasserstein_from_normal
from physicsnemo.diffusion.utils import ConcatConditionWrapper
"""
        result = subprocess.run(  # noqa: S603, fixed test script
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestConsistency:
    def test_wasserstein_import_identity(self):
        assert wasserstein_from_normal is canonical is original

    @pytest.mark.parametrize("batched", [False, True])
    def test_gaussian_distance_and_fid(self, batched):
        mu0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        mu1 = torch.tensor([3.0, 5.0], dtype=torch.float64)
        sigma0 = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
        sigma1 = torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64))
        expected = torch.tensor(27.0, dtype=torch.float64)
        if batched:
            mu0, mu1 = mu0.expand(3, -1), mu1.expand(3, -1)
            sigma0, sigma1 = sigma0.expand(3, -1, -1), sigma1.expand(3, -1, -1)
            expected = expected.expand(3)
        torch.testing.assert_close(
            wasserstein_from_normal(mu0, sigma0, mu1, sigma1), expected
        )
        torch.testing.assert_close(
            calculate_fid_from_inception_stats(mu0, sigma0, mu1, sigma1), expected
        )

    def test_univariate_distance(self):
        actual = wasserstein_from_normal(
            torch.tensor(0.0), torch.tensor(4.0), torch.tensor(3.0), torch.tensor(9.0)
        )
        torch.testing.assert_close(actual, torch.tensor(10.0))
