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

r"""Measure the per-geometry inference FLOPs of an AeroJEPA model.

Reports the paper's ``Mean TFLOPs`` column: the compute to run one full
inference on a single geometry -- encode the geometry once, predict the
target tokens, and decode the surface field over the entire ``128 x 256``
query grid.

The forward path mirrors :func:`inference._predict_one_case` exactly, so
the count reflects the real inference cost. FLOPs depend only on the
architecture and the input point counts, not on the trained weights, so a
checkpoint is optional (a random-init model gives the same count).

FLOPs are counted with ``torch.profiler`` (``with_flops=True``), which
sums the FLOPs of the standard aten ops that dominate the cost (the
attention / MLP matmuls). Custom point ops (kNN, farthest-point sampling)
have no FLOP formula and are not counted, so the figure is a lower bound
that tracks the matmul-dominated total the paper reports.

Usage (in the PhysicsNeMo container, on a GPU node)::

    python -m src.postprocessing.superwing_flops \
        data.path=/path/to/SuperWing_Dataset \
        checkpoint=outputs/<run>/checkpoints/best.pt   # optional

    # Average over more geometries (default 1):
    python -m src.postprocessing.superwing_flops \
        data.path=/path/to/SuperWing_Dataset flops_num_cases=8
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile

from inference import (
    _build_test_loader,
    _ensure_superwing_artifacts,
    _load_checkpoint,
    _predict_one_case,
    _slice_batch_sample,
)
from src.training import move_batch_to_device, set_seed

log = logging.getLogger(__name__)


def _forward_flops(
    *,
    model: torch.nn.Module,
    sample: dict,
    device: torch.device,
    precision: str,
    chunk_size: int,
) -> int:
    """Return summed FLOPs for one geometry's full-grid inference forward."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True) as prof:
        _predict_one_case(
            model=model,
            sample=sample,
            device=device,
            precision=precision,
            chunk_size=chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
    return sum(int(evt.flops) for evt in prof.key_averages() if evt.flops)


@hydra.main(
    config_path="../../conf",
    config_name="config_inference",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point -- see module docstring."""
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    precision = str(cfg.precision)
    chunk_size = int(cfg.decoder_chunk_size)
    num_cases = int(cfg.get("flops_num_cases", 1))

    split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    loader = _build_test_loader(
        cfg.data,
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
    )

    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    ckpt_path = Path(str(cfg.checkpoint)) if cfg.get("checkpoint") else None
    if ckpt_path is not None and ckpt_path.exists():
        _load_checkpoint(
            model=model, ckpt_path=ckpt_path, use_ema=bool(cfg.use_ema), device=device
        )
    else:
        log.info("No checkpoint loaded; FLOPs depend only on architecture + shapes.")

    per_case: list[float] = []
    for case_idx, batch in enumerate(loader):
        if case_idx >= num_cases:
            break
        batch = move_batch_to_device(batch, device)
        sample = _slice_batch_sample(batch, 0)
        flops = _forward_flops(
            model=model,
            sample=sample,
            device=device,
            precision=precision,
            chunk_size=chunk_size,
        )
        tflops = flops / 1e12
        per_case.append(tflops)
        log.info("case %d: %.4f TFLOPs", case_idx, tflops)

    if not per_case:
        raise RuntimeError("No test cases available to measure FLOPs.")
    mean = sum(per_case) / len(per_case)
    log.info(
        "Mean inference cost over %d case(s): %.4f TFLOPs per geometry "
        "(matmul FLOPs only; custom point ops not counted).",
        len(per_case),
        mean,
    )
    print(f"Mean TFLOPs (per geometry): {mean:.4f}")


if __name__ == "__main__":
    main()
