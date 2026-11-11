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

"""Move Python values to a device without a device synchronization."""

from collections.abc import Sequence

import torch


def to_device(
    value: float | Sequence[float] | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``torch.as_tensor(value, device=device, dtype=dtype)``, without a host sync.

    Tensors convert as with :func:`torch.as_tensor`, which returns them unchanged
    when they already have ``device`` and ``dtype``. Python numbers and sequences
    are first built in fresh pageable host memory, even under a CUDA default
    device. CUDA stages that memory before an asynchronous copy returns, so
    ``non_blocking`` avoids the device synchronization that a blocking
    host-to-device copy makes.
    """
    if isinstance(value, torch.Tensor):
        return torch.as_tensor(value, device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device="cpu").to(
        device, non_blocking=True
    )
