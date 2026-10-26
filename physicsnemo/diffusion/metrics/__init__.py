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

from physicsnemo.core.warnings import LegacyFeatureWarning

# NOTE: re-import these general metrics for increased visibility in the
# diffusion subpackage
from physicsnemo.metrics.general.calibration import rank_probability_score
from physicsnemo.metrics.general.crps import crps, kcrps
from physicsnemo.metrics.general.ensemble_metrics import (
    Mean,
    StreamingStatistic,
    Variance,
)
from physicsnemo.metrics.general.entropy import (
    entropy_from_counts,
    relative_entropy_from_counts,
)
from physicsnemo.metrics.general.histogram import (
    Histogram,
    cdf,
    histogram,
    normal_cdf,
    normal_pdf,
)
from physicsnemo.metrics.general.power_spectrum import power_spectrum
from physicsnemo.metrics.general.wasserstein import (
    wasserstein_from_cdf,
    wasserstein_from_normal,
    wasserstein_from_samples,
)

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


def __getattr__(name):
    """Keep the original base-class import available without eager warnings."""
    if name == "EnsembleMetrics":
        warnings.warn(
            "EnsembleMetrics is deprecated; use StreamingStatistic instead.",
            LegacyFeatureWarning,
            stacklevel=2,
        )
        return StreamingStatistic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
