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
- ``1 / safe_eps(dtype)`` does not overflow in the dtype's arithmetic.

Concretely, ``safe_eps(dtype) = min(tiny ** 0.25, machine_eps)``:

==========  =============  ============  ======================
dtype       ``safe_eps``   ``1 / eps``   note
==========  =============  ============  ======================
float16     ~9.8e-4        ~1.0e+3       capped at machine eps
bfloat16    ~3.3e-10       ~3.0e+9       tiny ** 0.25
float32     ~3.3e-10       ~3.0e+9       tiny ** 0.25
float64     ~1.2e-77       ~8.2e+76      tiny ** 0.25
==========  =============  ============  ======================

For float32 and wider types, ``1 / safe_eps ** 2`` also does not overflow,
which is useful when inverse-distance weights are squared.  Float16 has too
little dynamic range to satisfy both constraints simultaneously; the cap at
machine epsilon keeps the clamp floor small enough to be transparent for
values that are numerically meaningful in that dtype.
"""

import torch
from jaxtyping import Float


def safe_eps(dtype: torch.dtype) -> float:
    """Return a dtype-aware safe epsilon for preventing division by zero.

    This replaces all hardcoded ``1e-10`` clamp floors in the mesh module.
    The returned value is:

    - Small enough to leave any physically meaningful quantity untouched.
    - Large enough that ``1 / safe_eps(dtype)`` does not overflow.

    For types with wide exponent range (float32, float64, bfloat16) the
    formula ``tiny ** 0.25`` additionally guarantees that
    ``1 / safe_eps ** 2`` does not overflow.  For float16, whose 5-bit
    exponent cannot satisfy both constraints, the result is capped at
    machine epsilon to avoid corrupting mesh quantities.

    Parameters
    ----------
    dtype : torch.dtype
        The floating-point dtype (e.g. ``torch.float32``,
        ``torch.float64``).

    Returns
    -------
    float
        ``min(torch.finfo(dtype).tiny ** 0.25, torch.finfo(dtype).eps)``.
    """
    info = torch.finfo(dtype)
    return min(info.tiny**0.25, info.eps)


def default_length_tolerance(
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
) -> float:
    """Length tolerance relative to the mesh size, for the repair operations.

    ``max(1e-6, 100 * eps)`` times the bounding-box diagonal, where ``eps`` is
    the machine epsilon of the point dtype (so about ``1.2e-5`` for float32),
    and at least the smallest normal number of that dtype, so that points
    that all coincide still merge.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape ``(n_points, n_spatial_dims)``.

    Returns
    -------
    float
        The tolerance, or ``0.0`` when there are no points.
    """
    if len(points) == 0:
        return 0.0
    info = torch.finfo(points.dtype)
    diagonal = (points.amax(dim=0) - points.amin(dim=0)).norm().item()
    return max(max(1e-6, 100 * info.eps) * diagonal, info.tiny)
