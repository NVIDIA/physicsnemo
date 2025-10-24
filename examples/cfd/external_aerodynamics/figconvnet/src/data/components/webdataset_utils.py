# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
    """

    rank, world_size, *_ = wds.utils.pytorch_worker_info(group=group)
    cur = iter(src)
    while len(next_items := list(itertools.islice(cur, world_size))) == world_size:
        yield next_items[rank]

    tail_size = len(next_items)
    assert tail_size < world_size
    # If drop_last is not set, handle the tail.
    if not drop_last and tail_size > 0:
        yield next_items[rank % tail_size]


def from_numpy(sample: Mapping[str, Any], key: str):
    """Loads numpy objects from .npy, .npz or pickled files."""

    np_obj = np.load(io.BytesIO(sample[key]), allow_pickle=True)
    return {k: np_obj[k] for k in np_obj.files}
