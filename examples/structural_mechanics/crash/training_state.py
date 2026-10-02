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

"""Per-rank random state for reproducible epoch-boundary crash restarts."""

import os
import random

import numpy as np
import torch


def configure_deterministic_training(enabled):
    """Configure before CUDA initialization; fail closed on unsupported kernels.

    Saving RNG state is insufficient for CUDA scatter/attention reductions. Keep
    this opt-in so historical recipes retain their original execution policy.
    """
    if not enabled:
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Unsupported deterministic CUBLAS_WORKSPACE_CONFIG")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state(generators=None):
    """Capture Python, NumPy, Torch, CUDA, and named loader RNG states."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "generators": {
            name: gen.get_state() for name, gen in (generators or {}).items()
        },
    }


def restore_rng_state(state, generators=None):
    """Restore checkpoint RNG states and require matching loader generators."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires a CUDA device")
        torch.cuda.set_rng_state(state["cuda"].cpu())
    if set(state["generators"]) != set(generators or {}):
        raise ValueError("DataLoader generator contract changed at restart")
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name].cpu())


def gather_rng_states(generators=None):
    """Every rank must call, before rank zero writes the shared checkpoint."""
    local = capture_rng_state(generators)
    if not torch.distributed.is_initialized():
        return [local]
    states = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(states, local)
    return states


def restore_rank_rng(states, rank, world_size, generators=None):
    """Restore this rank only after checking the saved distributed world size."""
    if len(states) != world_size or not 0 <= rank < world_size:
        raise ValueError("RNG checkpoint world size does not match this restart")
    restore_rng_state(states[rank], generators)
