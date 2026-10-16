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

r"""Nonconformity scores for conformal prediction on spatio-temporal fields.

A score measures, per element, how far a target falls from a model's
prediction, and turns a threshold into an interval ``(lo, hi)``.

Pick the score by what the model outputs:
:class:`~physicsnemo.experimental.uq.conformal.AbsoluteErrorScore` for a
point prediction,
:class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore` for a
mean plus a standard deviation in ``aux["sigma"]``, and
:class:`~physicsnemo.experimental.uq.conformal.QuantileRegressionScore` for
quantile heads in ``aux["lo"]`` and ``aux["hi"]``.
"""

from .scores import (
    AbsoluteErrorScore,
    NormalizedErrorScore,
    QuantileRegressionScore,
)

__all__ = [
    "AbsoluteErrorScore",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
]
