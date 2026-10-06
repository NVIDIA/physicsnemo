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

"""Evaluate a trained Rotor37 run on complete surface meshes."""

import csv
import json
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_class, to_absolute_path
from models import predict
from objectives import FieldDecoder
from omegaconf import DictConfig, OmegaConf
from prepare_data import FIELD_NAMES, GLOBAL_NAMES
from train import make_dataset, strict_float32

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger

STRONGEST_FRACTION = 0.01


def pressure_jump_errors(prediction, target, edges, scale):
    """Return squared normalized errors of signed pressure differences.

    The second result marks the ``ceil(0.01 E)`` edges with the largest
    reference differences, with ties broken by edge order.
    """
    reference = target[edges[:, 0]] - target[edges[:, 1]]
    predicted = prediction[edges[:, 0]] - prediction[edges[:, 1]]
    squared = ((predicted - reference) / scale) ** 2
    count = int(np.ceil(STRONGEST_FRACTION * len(edges)))
    strongest = np.zeros(len(edges), dtype=bool)
    strongest[np.argsort(-np.abs(reference), kind="stable")[:count]] = True
    return squared, strongest


def case_metrics(fields, globals_, arrays, edges, pressure_scale):
    """Measure the errors of one predicted case against its reference."""
    fields = fields.astype(np.float64)
    target = arrays["fields"].astype(np.float64)
    error = fields - target
    row = {}
    for channel, name in enumerate(FIELD_NAMES):
        row[f"{name}_relative_l2"] = float(
            np.linalg.norm(error[:, channel]) / np.linalg.norm(target[:, channel])
        )
        row[f"{name}_mse"] = float(np.mean(error[:, channel] ** 2))
        row[f"{name}_mae"] = float(np.mean(np.abs(error[:, channel])))
    squared, strongest = pressure_jump_errors(
        fields[:, 1], target[:, 1], edges, pressure_scale
    )
    row["jump_mse"] = float(squared.mean())
    row["strongest_jump_mse"] = float(squared[strongest].mean())
    for name, value, truth in zip(GLOBAL_NAMES, globals_, arrays["globals"]):
        row[f"{name}_prediction"] = float(value)
        row[f"{name}_truth"] = float(truth)
    return row


def summarize(rows):
    """Aggregate per-case errors with equal weight for every case."""
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary = {
        "cases": len(rows),
        "fields": {
            name: {
                "relative_l2": mean[f"{name}_relative_l2"],
                "rmse": float(np.sqrt(mean[f"{name}_mse"])),
                "mae": mean[f"{name}_mae"],
            }
            for name in FIELD_NAMES
        },
        "pressure_jump": {
            "rmse": float(np.sqrt(mean["jump_mse"])),
            "strongest_rmse": float(np.sqrt(mean["strongest_jump_mse"])),
        },
        "globals": {},
    }
    for name in GLOBAL_NAMES:
        prediction = np.array([row[f"{name}_prediction"] for row in rows])
        truth = np.array([row[f"{name}_truth"] for row in rows])
        error = prediction - truth
        summary["globals"][name] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error**2) / np.sum((truth - truth.mean()) ** 2)),
        }
    return summary


def evaluate(cfg, run_dir, log):
    """Evaluate the final checkpoint of a run on ``cfg.evaluation.split``."""
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    run_cfg.data.data_dir = cfg.data.data_dir
    data_dir = Path(to_absolute_path(cfg.data.data_dir))
    for name in ("manifest.json", "stats.json"):
        if (data_dir / name).read_bytes() != (run_dir / name).read_bytes():
            raise ValueError(f"{data_dir / name} differs from the trained run")
    DistributedManager.initialize()
    device = DistributedManager().device
    split = cfg.evaluation.split
    dataset = make_dataset(run_cfg, run_dir, split)
    decoder = FieldDecoder(dataset.basis, dataset.stats).to(device)
    epoch = run_cfg.training.num_epochs
    network = get_class(run_cfg.model._target_)
    model = network.from_checkpoint(
        str(run_dir / "checkpoints" / f"{network.__name__}.0.{epoch}.mdlus")
    )
    model = model.to(device).eval()
    output_dir = run_dir / "evaluation" / split
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "predictions").mkdir(parents=True)
    labeled = split != "official_test"
    exports = set(cfg.evaluation.export_samples)
    pressure_scale = dataset.stats["fields"]["std"][1]
    rows = []
    with strict_float32(), torch.no_grad():
        for index, sample_id in enumerate(dataset.sample_ids):
            batch = {
                key: value[None].to(device)
                for key, value in dataset[index].items()
                if isinstance(value, torch.Tensor)
            }
            prediction = predict(model, batch, decoder)
            fields = decoder.physical(prediction.coefficients)[0].cpu().numpy()
            globals_ = dataset.denormalize_globals(
                prediction.globals[0].cpu().numpy().astype(np.float64)
            )
            arrays = dataset.load(index)
            if labeled:
                row = case_metrics(
                    fields, globals_, arrays, dataset.basis["edges"], pressure_scale
                )
                rows.append({"sample_id": sample_id, **row})
            if not labeled or sample_id in exports:
                export = {
                    "points": arrays["points"],
                    "quads": arrays["quads"],
                    "field_prediction": fields,
                    "global_prediction": globals_,
                    "pressure_scale": pressure_scale,
                }
                if labeled:
                    export["field_truth"] = arrays["fields"]
                    export["global_truth"] = arrays["globals"]
                np.savez_compressed(
                    output_dir / "predictions" / f"sample_{sample_id:06d}.npz", **export
                )
    if labeled:
        summary = {"split": split, "epoch": epoch, **summarize(rows)}
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output_dir / "cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    log.info(f"Evaluated {len(dataset)} {split} cases in {output_dir}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate the run selected by ``output_dir``."""
    torch.set_num_threads(cfg.num_threads)
    evaluate(cfg, Path(to_absolute_path(cfg.output_dir)), PythonLogger("rotor37"))


if __name__ == "__main__":
    main()
