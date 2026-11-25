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

import warnings

from .fid import calculate_fid_from_inception_stats
from .legacy_losses import (
    EDMLoss,
    EDMLossLogUniform,
    EDMLossSR,
    RegressionLoss,
    RegressionLossCE,
    ResidualLoss,
    VELoss,
    VELoss_dfsr,
    VPLoss,
)
from .losses import (
    FlowMatchingLoss,
    MSEDSMLoss,
    WeightedFlowMatchingLoss,
    WeightedMSEDSMLoss,
)
from .wasserstein import wasserstein_from_normal

_REMOVED_METRICS = {
    "rank_probability_score": "calibration",
    "crps": "crps",
    "kcrps": "crps",
    "EnsembleMetrics": "ensemble_metrics",
    "StreamingStatistic": "ensemble_metrics",
    "Mean": "ensemble_metrics",
    "Variance": "ensemble_metrics",
    "entropy_from_counts": "entropy",
    "relative_entropy_from_counts": "entropy",
    "Histogram": "histogram",
    "cdf": "histogram",
    "histogram": "histogram",
    "normal_cdf": "histogram",
    "normal_pdf": "histogram",
    "power_spectrum": "power_spectrum",
    "wasserstein_from_cdf": "wasserstein",
    "wasserstein_from_samples": "wasserstein",
}


def __getattr__(name: str) -> None:
    """Explain replacement imports for removed general metric re-exports."""
    if name in _REMOVED_METRICS:
        replacement = "StreamingStatistic" if name == "EnsembleMetrics" else name
        message = (
            f"physicsnemo.diffusion.metrics.{name} has been removed. "
            f"Use 'from physicsnemo.metrics.general.{_REMOVED_METRICS[name]} "
            f"import {replacement}' instead."
        )
        warnings.warn(message, FutureWarning, stacklevel=2)
        raise ImportError(message)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
