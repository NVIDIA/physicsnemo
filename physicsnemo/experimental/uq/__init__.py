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

This subpackage collects UQ building blocks that attach to existing backbones.
It holds the Gaussian-process heads below; other UQ methods are expected to join
them as the area develops.

Gaussian-process heads
----------------------
Both are variational Gaussian processes with inducing points, and they differ in
what constitutes a data point, and therefore in what they predict:

* :class:`VariationalGPHead` — takes one pooled embedding per sample and
  predicts a *scalar* target with uncertainty.
* :class:`FieldVariationalGPHead` — takes *per-point* features and predicts a
  multi-channel *field*, with one Gaussian posterior per point per channel.

Each returns a distance-aware predictive variance from a single forward pass,
without ensembling or sampling.  That variance is a posterior variance rather
than a calibrated error bar: how well its scale matches observed error depends
on the training recipe and on the shift between training and evaluation data, so
check it against held-out data, and recalibrate, before reading it
quantitatively.

Both heads require ``gpytorch``: ``pip install gpytorch``, or the ``uq-extras``
optional dependency group (``pip install nvidia-physicsnemo[uq-extras]``).
"""

from physicsnemo.core.version_check import check_version_spec

_GPYTORCH_AVAILABLE = check_version_spec("gpytorch", hard_fail=False)

if _GPYTORCH_AVAILABLE:
    from .field_variational_gp_head import (
        FieldVariationalGPHead,
        FieldVariationalGPPrediction,
    )
    from .variational_gp_head import GPPrediction, VariationalGPHead
