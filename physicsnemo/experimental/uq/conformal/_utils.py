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

r"""Private helpers shared by the conformal package.

Outward rounding and scalar checks used by the nonconformity scores. A
score or interval endpoint must never round to the wrong side, or a target
the threshold admits could fall outside its interval, so
:func:`cast_directed` rounds outward whenever a cast loses precision.
"""

import math

import torch
from torch import Tensor


def cast_directed(t: Tensor, dtype: torch.dtype, *, up: bool) -> Tensor:
    """Cast to ``dtype``, rounding toward ``+inf`` (``up=True``) or ``-inf``.

    Outward rounding keeps a threshold from landing below its order statistic.
    """
    cast = t.to(dtype)
    if dtype == t.dtype or not (t.is_floating_point() and cast.is_floating_point()):
        return cast
    exact = t.to(torch.float64)
    got = cast.to(torch.float64)
    wrong_side = got < exact if up else got > exact
    toward = torch.inf if up else -torch.inf
    bumped = torch.nextafter(cast, torch.full_like(cast, toward))
    return torch.where(wrong_side, bumped, cast)


def clamp_min_floor(t: Tensor, eps: float) -> Tensor:
    """Clamp ``t`` to at least ``max(eps, finfo(t.dtype).tiny)``.

    ``eps`` alone can underflow to zero in float16.
    """
    if not math.isfinite(eps):
        raise ValueError(f"eps must be finite, got {eps}.")
    floor = eps
    if t.is_floating_point():
        info = torch.finfo(t.dtype)
        if eps > info.max:
            raise ValueError(
                f"eps={eps} exceeds the largest finite {t.dtype} value "
                f"({info.max}); use a smaller eps or a wider dtype."
            )
        floor = max(eps, float(info.tiny))
    return t.clamp_min(floor)


def positive_finite_float(value: object, name: str) -> float:
    """Coerce a strategy scalar (``eps``, ``value``) to a positive finite float."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive finite value, got {value!r}."
        ) from exc
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"{name} must be a positive finite value, got {value}.")
    return value
