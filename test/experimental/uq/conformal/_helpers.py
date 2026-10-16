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

"""Shared containment assertions for the conformal suite.

Plain module attributes, imported explicitly
(``from test.experimental.uq.conformal._helpers import ...``).
"""

from __future__ import annotations

import torch

__all__ = [
    "assert_admitted_covered",
]


def assert_admitted_covered(score, pred, target, threshold, aux) -> None:
    """Score-level property: ``score <= threshold`` implies ``lo <= target <= hi``."""
    s = score.score(pred, target, aux=aux)
    admitted = torch.isfinite(s) & (s <= threshold) & torch.isfinite(target)
    lo, hi = score.interval(pred, threshold, aux=aux)
    assert lo.dtype == pred.dtype and hi.dtype == pred.dtype
    bad = admitted & ~((target >= lo) & (target <= hi))
    assert not bad.any(), (
        f"{int(bad.sum())} score-admitted target(s) excluded; first at "
        f"index {int(bad.nonzero()[0])}"
    )
