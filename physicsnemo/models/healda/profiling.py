# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
from contextlib import contextmanager, nullcontext

import torch

NVTX_ENABLED = os.environ.get("HEALDA_NVTX", "0") == "1"


def nvtx(func=None, *, enabled: bool | None = None):
    def decorator(fn):
        use_nvtx = NVTX_ENABLED if enabled is None else enabled
        if not use_nvtx:
            return fn

        tag = fn.__module__ + ":" + fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(tag)
            out = fn(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return out

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@contextmanager
def _nvtx_range_impl(tag: str):
    torch.cuda.nvtx.range_push(tag)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_range(tag: str, enabled: bool | None = None):
    use_nvtx = NVTX_ENABLED if enabled is None else enabled
    if use_nvtx:
        return _nvtx_range_impl(tag)
    return nullcontext()
