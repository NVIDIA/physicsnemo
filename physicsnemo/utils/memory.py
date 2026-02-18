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

import importlib
import os

import torch

from physicsnemo.core.version_check import check_version_spec

RMM_AVAILABLE = check_version_spec("rmm", "2.6.0", hard_fail=False)
CUPY_AVAILABLE = check_version_spec("cupy", "12.0.0", hard_fail=False)

"""
Using a unifed gpu memory provider, we consolidate the pool into just a
single allocator for cupy/rapids and torch.  Ideally, we add warp to this someday.

To use this, you need to add the following to your code at or near the top
(before allocating any GPU memory):

```python
from physicsnemo.utils.memory import unified_gpu_memory
```

"""


def srt2bool(val: str):
    if isinstance(val, bool):
        return val
    if val.lower() in ["true", "1", "yes", "y"]:
        return True
    elif val.lower() in ["false", "0", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {val}")


DISABLE_RMM = srt2bool(os.environ.get("PHYSICSNEMO_DISABLE_RMM", False))


def _setup_unified_gpu_memory():
    # Skip if RMM is disabled
    if RMM_AVAILABLE and not DISABLE_RMM:
        rmm = importlib.import_module("rmm")
        rmm_torch_allocator = importlib.import_module(
            "rmm.allocators.torch"
        ).rmm_torch_allocator

        # First, determine the local rank so that we allocate on the right device.
        # These are meant to be tested in the same order as DistributedManager
        # We can't actually initialize it, though, since we have to unify mallocs
        # before torch init.
        PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD = os.environ.get(
            "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD", None
        )
        if PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD is None:
            # default to 0:
            local_rank = 0

            # Update if a variable sets the local rank:
            for method in ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"]:
                if os.environ.get(method) is not None:
                    local_rank = int(os.environ.get(method))
                    break

        else:
            if PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD == "ENV":
                local_rank = int(os.environ.get("LOCAL_RANK"))
            elif PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD == "SLURM":
                local_rank = int(os.environ.get("SLURM_LOCALID"))
            elif PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD == "OPENMPI":
                local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
            else:
                raise ValueError(
                    f"Unknown initialization method: {PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD}"
                )

        # Initialize RMM
        rmm.reinitialize(
            pool_allocator=True, devices=local_rank, initial_pool_size="1024MB"
        )

        # Set PyTorch allocator if available
        # from rmm.allocators.torch import rmm_torch_allocator

        if torch.cuda.is_available():
            torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

        # Set CuPy allocator if available
        if CUPY_AVAILABLE:
            cupy = importlib.import_module("cupy")
            # from rmm.allocators.cupy import rmm_cupy_allocator
            rmm_cupy_allocator = importlib.import_module(
                "rmm.allocators.torch"
            ).rmm_cupy_allocator

            cupy.cuda.set_allocator(rmm_cupy_allocator)


# This is what gets executed when someone does "from memory import unified_gpu_memory"


def __getattr__(name):
    if name == "unified_gpu_memory":
        return _setup_unified_gpu_memory()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
