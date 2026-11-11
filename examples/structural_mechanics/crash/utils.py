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

"""Shared helpers of the crash recipes: global features, optimizers, warm starts,
distributed dataset startup and per-rank random state."""

import json
import math
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from physicsnemo.optim import CombinedOptimizer


def load_global_features(json_path: str) -> dict[str, dict[str, float]]:
    """
    Load global features JSON once.

    Returns:
        dict[str, dict[str, float]]:
            Mapping run_id -> global feature dict
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Global features file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Global features JSON must be a dict keyed by run_id")

    # Optional: sanity check values
    for run_id, features in data.items():
        if not isinstance(features, dict):
            raise TypeError(f"Global features for run '{run_id}' must be a dict")

    return data


def get_global_features_for_run(
    all_global_features: dict[str, dict[str, float]],
    run_id: str,
) -> dict[str, float]:
    """
    Fetch global features for a single run.

    Args:
        all_global_features: output of load_global_features
        run_id: key identifying the run (e.g. derived from filename)

    Returns:
        dict[str, float]: global scalar features for this run
    """
    try:
        return all_global_features[run_id]
    except KeyError:
        raise KeyError(f"run_id '{run_id}' not found in global features file")


def build_muon_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build Muon + AdamW combined optimizer (Muon for 2D params, AdamW for others).

    Muon requires PyTorch >= 2.9. Pass the underlying model (unwrap DDP if needed).
    Frozen parameters (a residual model's teacher) are not optimized.
    ``training.muon_min_dim`` (default 0): 2-D parameters whose smaller side is
    below it go to AdamW instead of being orthogonalized by Muon (tiny matrices
    such as an 8 x 32 per-head bias projection).
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "Muon optimizer requires PyTorch >= 2.9. "
            "Install a newer PyTorch or use optimizer=adam."
        )
    base_model = model.module if hasattr(model, "module") else model
    min_dim = int(cfg.training.get("muon_min_dim", 0))
    named = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
    muon_params = [p for _, p in named if p.ndim == 2 and min(p.shape) >= min_dim]
    other_params = [
        p for _, p in named if not (p.ndim == 2 and min(p.shape) >= min_dim)
    ]
    weight_decay = cfg.training.get("optimizer_weight_decay", 1e-4)
    lr = cfg.training.start_lr
    if muon_params and other_params:
        return CombinedOptimizer(
            [
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                torch.optim.AdamW(
                    other_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1.0e-8,
                ),
            ]
        )
    elif muon_params:
        return torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
    else:
        return torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay)


def _is_mdlus_archive(path: str) -> bool:
    """A physicsnemo ``.mdlus`` archive: a zip, or the tar of older releases, with ``model.pt``."""
    import tarfile
    import zipfile

    # Tar first: a tar archive holding a zip member passes the zip sniff as
    # well. A torch.save file is a zip too, without model.pt.
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            return "model.pt" in archive.getnames()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return "model.pt" in archive.namelist()
    return False


def load_warm_start(model: torch.nn.Module, path) -> dict[str, list[str]]:
    """Initialise ``model`` from another run's ``.mdlus`` (or state-dict ``.pt``) file.

    A ``.mdlus`` archive (zip, or the tar format of releases before 2.2.0) is
    loaded with :func:`physicsnemo.utils.load_model_weights`; a ``torch.save``
    file may hold a state dict or a checkpoint dict with ``model_state_dict``.
    Every tensor of the file must exist in the model and every tensor of the
    model must be in the file. Returns the loaded key list. Used to initialise
    a run from a finished one (``training.warm_start_model``), by the evaluators
    and by the corrector to load its frozen teacher.
    """
    from physicsnemo.utils import load_model_weights

    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"warm start file {path} does not exist")
    if _is_mdlus_archive(path):
        try:
            load_model_weights(model, path)
        except RuntimeError as error:  # the strict load names the mismatched keys
            raise ValueError(
                f"warm start {path} does not match the model: {error}"
            ) from error
        return {"loaded": sorted(model.state_dict().keys())}
    # a torch.save file: a state dict or a checkpoint dict
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            f"warm start {path} has unexpected tensors not in the model: "
            f"{sorted(result.unexpected_keys)[:8]}"
        )
    if result.missing_keys:
        raise ValueError(
            f"warm start {path} is missing tensors of the model: "
            f"{sorted(result.missing_keys)[:8]}"
        )
    return {"loaded": sorted(state.keys())}


# ----- distributed dataset startup (rank zero publishes, the others load) -----


def initialize_datasets(build, timeout_seconds=3600):
    """Run ``build(stats_mode)`` on every rank, publishing rank-zero data first.

    ``build`` must initialize BOTH training and validation datasets. Rank zero
    alone publishes normalization files and warms shared geometry caches. All
    ranks finish before returning to any GPU collectives. CPU failure messages
    are propagated collectively; a process crash is bounded by the Gloo timeout.
    This does not increase the training process group's NCCL timeout.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Dataset startup timeout must be finite and positive")
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return build("compute")
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
    result = None
    try:
        status = [None]
        if dist.get_rank() == 0:
            try:
                result = build("compute")
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(status, src=0, group=group)
        if status[0] is not None:
            raise RuntimeError(f"Rank-zero dataset startup failed: {status[0]}")
        failure = None
        if dist.get_rank() != 0:
            try:
                result = build("load")
            except Exception as error:
                failure = f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure, group=group)
        if any(failures):
            raise RuntimeError(f"Dataset startup failed: {failures}")
        return result
    finally:
        dist.destroy_process_group(group)


# ----- per-rank random state for reproducible epoch-boundary restarts -----


def configure_deterministic_training(enabled):
    """Configure before CUDA initialization; fail closed on unsupported kernels.

    Saving RNG state is insufficient for CUDA scatter/attention reductions. Keep
    this opt-in; the released recipes train without it.
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


def restore_rank_rng(states, rank, world_size, generators=None, *, required=True):
    """Restore this rank's RNG state after checking the saved distributed world size.

    Returns ``True`` when the state was restored. At another world size the
    saved streams cannot be replayed: with ``required`` (a reproducible or
    deterministic restart) that raises, otherwise nothing is restored and
    ``False`` is returned so the caller can warn and go on.
    """
    if len(states) != world_size or not 0 <= rank < world_size:
        if required:
            raise ValueError(
                f"RNG checkpoint holds {len(states)} rank states but this restart "
                f"has {world_size} ranks; training.reproducible_sampling and "
                "training.deterministic need the world size of the checkpoint "
                "(training.restore_rng=false loads the weights without the streams)"
            )
        return False
    restore_rng_state(states[rank], generators)
    return True
