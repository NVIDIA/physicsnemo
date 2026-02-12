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

from typing import Any, Iterable, Mapping

import itertools
import io
import numpy as np

import torch
import webdataset as wds


def split_by_node_equal(
    src: Iterable,
    drop_last: bool = False,
    group: "torch.distributed.ProcessGroup" = None,
):
    """Splits input iterable into equal-sized chunks according to multiprocessing configuration.

    Similar to `Webdataset.split_by_node`, but the resulting split is equal-sized.
     Now supports multi-GPU execution.
    """
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info(group=group)

    worker = 0 if worker is None else worker
    num_workers = max(1, num_workers)
    g_worker = rank * num_workers + worker  # Global worker id.
    g_world = world_size * num_workers  # Total number of global workers.

    it = iter(src)
    for chunk in iter(lambda: list(itertools.islice(it, g_world)), []):
        n = len(chunk)
        if n < g_world:  # Tail chunk.
            if not drop_last and g_worker < n:
                yield chunk[g_worker]
            return
        yield chunk[g_worker]


def from_numpy(sample: Mapping[str, Any], key: str):
    """Loads numpy objects from .npy, .npz or pickled files."""

    np_obj = np.load(io.BytesIO(sample[key]), allow_pickle=True)
    return {k: np_obj[k] for k in np_obj.files}
