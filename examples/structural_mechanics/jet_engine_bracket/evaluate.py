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

r"""Evaluate a trained (and optionally LoRA-adapted) model on every node of a split, in float32.

Per sample: relative L2 error of the displacement vector and of the von Mises stress, and the
stress :math:`R^2`. Writes ``<output>_cases.csv`` and ``<output>_summary.json`` (mean and median
per load case and overall).

    python evaluate.py data.root=<data>/deepjeb                                        # DeepJEB test
    python evaluate.py data.root=<data>/simjeb data.split=splits/simjeb.json output=outputs/deepjeb/evaluation/simjeb_test
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from bracket_io import load_split
from datapipe import BracketDataset, load_scales
from nastran import LOADS
from omegaconf import DictConfig
from utils import case_metrics, predict, resolve

import physicsnemo
from physicsnemo.experimental.models.geotransolver_sdf import GeoTransolverSDF
from physicsnemo.experimental.peft import load_adapter


def load_model(
    path: str | Path, adapter: str | Path | None, device: torch.device
) -> torch.nn.Module:
    r"""A saved model, with a LoRA adapter of ``finetune.py`` applied if given, in evaluation mode."""
    model = physicsnemo.Module.from_checkpoint(str(resolve(path)))
    if adapter:
        load_adapter(model, str(resolve(adapter)))
    return model.to(device).eval()


def evaluate(
    model: torch.nn.Module,
    dataset: BracketDataset,
    device: torch.device,
    precision: str = "float32",
) -> list[dict]:
    r"""One row of :func:`utils.case_metrics` per sample, with its ``shape_id`` and ``load``."""
    data = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    rows = []
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = (
        precision != "float32"
    )  # strict float32
    try:
        for batch in data:
            prediction = predict(model, batch, device, precision)
            rows.append(
                {
                    "shape_id": int(batch["shape_id"]),
                    "load": LOADS[int(batch["load_id"])],
                    **case_metrics(prediction, batch["target"][0].to(device)),
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    return rows


def summarize(rows: list[dict]) -> dict:
    r"""Mean and median of every metric, per load case and over all samples."""
    metrics = [k for k in rows[0] if k not in ("shape_id", "load")]
    groups = {
        "all": rows,
        **{load: [r for r in rows if r["load"] == load] for load in LOADS},
    }
    return {
        name: {
            "samples": len(g),
            **{
                f"{m}_{s}": float(f([r[m] for r in g]))
                for m in metrics
                for s, f in (("mean", np.mean), ("median", np.median))
            },
        }
        for name, g in groups.items()
        if g
    }


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    r"""Evaluate a model, optionally with a LoRA adapter (Hydra entry point, ``conf/evaluate.yaml``)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = (
        False  # strict float32
    )
    split = load_split(resolve(cfg.data.split))
    subsets = (
        ("train", "val", "test") if cfg.data.subset == "all" else (cfg.data.subset,)
    )
    shapes = sorted(s for k in subsets for s in split[k])[: cfg.data.max_designs]
    model = load_model(cfg.checkpoint, cfg.adapter, device)
    dataset = BracketDataset(
        cfg.data.root,
        shapes,
        load_scales(resolve(cfg.data.scales)),
        coordinate_scale_mm=cfg.data.coordinate_scale_mm,
        token_voxel_mm=cfg.data.token_voxel_mm,
        sdf=isinstance(model, GeoTransolverSDF),
    )
    rows = evaluate(model, dataset, device)
    output = resolve(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{output}_cases.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    if (
        device.type == "cuda"
    ):  # peak memory of the evaluation, for the hardware notes of the README
        summary["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 2
        )
    Path(f"{output}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["all"]))


if __name__ == "__main__":
    main()
