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

"""Uncertainty quantification modules (experimental).

This subpackage provides UQ building blocks that can be attached to encoder
backbones to produce calibrated uncertainty estimates.  Currently available:

* :class:`VariationalGPHead` — variational Gaussian process head backed by
  GPyTorch.  Requires ``gpytorch`` (``pip install gpytorch``).
"""

from physicsnemo.core.version_check import check_version_spec

_GPYTORCH_AVAILABLE = check_version_spec("gpytorch", hard_fail=False)

if _GPYTORCH_AVAILABLE:
    from .variational_gp_head import GPPrediction, VariationalGPHead
