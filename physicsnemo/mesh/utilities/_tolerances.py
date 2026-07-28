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
from the dtype alone, and :func:`safe_normalize`, which normalizes vectors
with no clamp floor at all -- no single epsilon is correct across every dtype
and mesh scale. ``safe_eps`` is chosen so that:

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


def safe_normalize(
    vectors: Float[torch.Tensor, "..."],
    dim: int,
) -> Float[torch.Tensor, "..."]:
    r"""Scale vectors to unit length along ``dim`` without an epsilon clamp.

    ``torch.nn.functional.normalize`` divides by the norm clamped below at a
    hardcoded ``eps=1e-12``, which is wrong in three separate ways:

    - In ``float16`` that floor is not representable and rounds to zero, so a
      degenerate cell computes ``0 / 0`` and its normal becomes NaN.
    - The floor is *absolute*, so a genuine norm below ``1e-12`` is silently
      substituted. In ``float32`` and ``float64`` alike this yields non-unit
      normals once mesh feature size falls below roughly ``1e-6`` -- exactly
      the micro- and nanoscale case this module exists to protect.
    - Squaring components to form the norm overflows to ``inf`` for large
      inputs, and ``v / inf`` silently returns a *zero* normal for a
      perfectly well-conditioned cell.

    No epsilon fixes all three: in ``float16`` every representable floor,
    :func:`safe_eps` included, is large enough to shorten valid normals.
    Instead each vector is divided by its own largest component before the
    norm is taken. The rescaled vector's largest component is exactly one, so
    its norm always lies in :math:`[1, \sqrt{n}]` for :math:`n` components --
    it can neither overflow nor underflow -- and only an exactly zero vector
    still needs a guard.

    Parameters
    ----------
    vectors : Float[torch.Tensor, "..."]
        Vectors to normalize, of any shape.
    dim : int
        Dimension holding the vector components. Deliberately has no default:
        ``torch.nn.functional.normalize`` defaults to ``dim=1`` while mesh code
        wants ``dim=-1``, and a silent mismatch would be easy to miss at a
        call site.

    Returns
    -------
    Float[torch.Tensor, "..."]
        Unit vectors along ``dim``, with the shape and dtype of ``vectors``.
        Rows that are exactly zero are returned as zero vectors -- the
        convention the mesh normal APIs report for degenerate (zero-area)
        cells and for points with no incident cell.

    Notes
    -----
    Finite input always produces finite output, since the rescaled norm is
    bounded below by one. Non-finite input is not repaired: a single ``inf``
    or ``NaN`` component makes the whole vector NaN, rather than the partially
    finite result ``torch.nn.functional.normalize`` happens to produce.

    Examples
    --------
    >>> v = torch.tensor([[3.0e-13, 4.0e-13], [0.0, 0.0]])
    >>> safe_normalize(v, dim=-1)
    tensor([[0.6000, 0.8000],
            [0.0000, 0.0000]])
    """
    ### ``amax`` rejects a zero-size reduction dim, and there is nothing to
    ### normalize in that case anyway.
    if vectors.shape[dim] == 0:
        return vectors

    ### Rescaling adds a pass over ``vectors`` relative to a plain clamp and
    ### divide, making this memory-bound kernel roughly 2-4x slower on its own.
    ### That is a deliberate correctness-over-speed trade, not an oversight:
    ### removing the rescale reintroduces the overflow and small-norm errors
    ### described above, which ``TestSafeNormalize`` pins down. Mesh normals
    ### are cached per ``Mesh``, so the cost is paid once per mesh and stays
    ### sub-millisecond even at multi-million cells.
    scale = vectors.abs().amax(dim=dim, keepdim=True)
    is_zero = scale == 0
    scaled = vectors / scale.masked_fill(is_zero, 1)
    norm = scaled.norm(dim=dim, keepdim=True)
    return scaled / norm.masked_fill(is_zero, 1)
