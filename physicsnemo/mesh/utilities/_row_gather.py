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

"""Row gathers that stay cheap on memory-mapped tensors."""

import torch

### Largest row span (in bytes) read as one contiguous slice before falling
### back to a plain fancy-index gather.
RANGE_READ_MAX_BYTES = 256 * 2**20


def gather_rows(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Return ``t[idx]`` as a fresh tensor, reading memmap rows in one range when cheap.

    A fancy-index gather on a memory-mapped tensor faults one page per row,
    which on a network file system means one small random read per row. When
    the referenced rows span at most :data:`RANGE_READ_MAX_BYTES`, the whole
    range is read with a single sequential slice and indexed in memory instead.
    Otherwise this is exactly ``t[idx]``. ``idx`` may be in any order and may
    contain duplicates; the result matches ``t[idx]`` row for row.
    """
    if idx.numel() == 0 or t.ndim == 0:
        return t[idx]
    lo = int(idx.min())
    hi = int(idx.max()) + 1
    row_bytes = t.element_size() * (t[0].numel() if t.ndim > 1 else 1)
    if (hi - lo) * row_bytes <= RANGE_READ_MAX_BYTES:
        return t[lo:hi][idx - lo]
    return t[idx]
