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

"""Material-property transform for radiation-transport surrogates.

The shipped zarr stores carry precomputed cross-section fields
(``sigma_a``, ``sigma_s``, ``sigma_t``, ``Q``) per cell. ``MaterialPropertyExtractor``
stacks them into a single ``physical_properties`` tensor of shape ``(N, 4)``
in the order ``[sigma_a, sigma_s, sigma_t, Q]``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from transforms import Transform


class MaterialPropertyExtractor(Transform):
    """Stack precomputed sigma fields into a per-cell ``(N, 4)`` tensor.

    Q must be present in the source data; it may be all-zero for source-free
    regimes (e.g., hohlraum).
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        for key in ("sigma_a", "sigma_s", "sigma_t", "Q"):
            if key not in data:
                raise KeyError(
                    f"Zarr store is missing required field {key!r}. "
                    "All four fields (sigma_a, sigma_s, sigma_t, Q) must be precomputed."
                )

        data["physical_properties"] = torch.stack(
            [data["sigma_a"], data["sigma_s"], data["sigma_t"], data["Q"]],
            dim=-1,
        ).to(dtype=torch.float32)
        return data


__all__ = ["MaterialPropertyExtractor"]
