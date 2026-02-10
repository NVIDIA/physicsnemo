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

"""Dtype-aware numerical tolerances for mesh computations.

A hardcoded absolute tolerance like ``1e-10`` is wrong for meshes whose
coordinates live at a scale far from unity.  For float64 in particular,
``1e-10`` is millions of times larger than machine precision and corrupts
results on micro- or nanoscale geometries.

This module provides :func:`safe_eps`, which returns a floor value derived
from the dtype alone, chosen so that:

- It is small enough to never activate on any physically meaningful mesh.
- Its reciprocal squared does not overflow (important for inverse-distance
  weights raised to power 2).

Concretely, ``safe_eps(dtype) = torch.finfo(dtype).tiny ** 0.25``:

==========  =============  =============================
dtype       ``safe_eps``   ``1 / safe_eps ** 2``
==========  =============  =============================
float32     ~3.3e-10       ~9.2e+18  (well below 3.4e38)
float64     ~1.2e-77       ~6.7e+153 (well below 1.8e308)
==========  =============  =============================
"""

import torch


def safe_eps(dtype: torch.dtype) -> float:
    """Return a dtype-aware safe epsilon for preventing division by zero.

    This replaces all hardcoded ``1e-10`` clamp floors in the mesh module.
    The returned value is:

    - Small enough to leave any physically meaningful quantity untouched.
    - Large enough that ``1 / safe_eps(dtype) ** 2`` does not overflow.

    Parameters
    ----------
    dtype : torch.dtype
        The floating-point dtype (e.g. ``torch.float32``,
        ``torch.float64``).

    Returns
    -------
    float
        A small positive floor value equal to
        ``torch.finfo(dtype).tiny ** 0.25``.
    """
    return torch.finfo(dtype).tiny ** 0.25
