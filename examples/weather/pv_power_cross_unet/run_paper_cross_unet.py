# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Cross-Unet-only reproduction sweeps for the PV-power paper tables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from train_real_cross_unet import (
    PAPER_HORIZONS,
    RealPVDataConfig,
    RealPVDataset,
    compute_prediction_metrics,
    paper_seq_len_for_horizon,
    predict,
    train,
)


@dataclass(frozen=True)
class PaperExperiment:
    """Single Cross-Unet paper-table experiment."""

    station: str
    source: str
    horizon_label: str

    @property
    def pred_len(self) -> int:
        return PAPER_HORIZONS[self.horizon_label]

    @property
    def seq_len(self) -> int:
        return paper_seq_len_for_horizon(self.pred_len)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.station, self.source, self.horizon_label)


CSV_COLUMNS = [
    "station",
    "source",
    "horizon_label",
    "seq_len",
    "pred_len",
    "seed",
    "mae",
    "mse",
    "r2",
    "paper_mae",
    "paper_mse",
    "paper_r2",
    "pass_mae",
    "pass_mse",
    "pass_r2",
    "param_count",
]


# Fill this dictionary when exact paper table constants are available. The
# runner also accepts an external CSV with matching key columns and
# paper_mae/paper_mse/paper_r2 fields.
PAPER_TARGETS: dict[tuple[str, str, str], dict[str, float]] = {}

PAPER_PARAM_TARGETS_M: dict[str, float] = {
    "4h": 5.851,
    "12h": 5.851,
    "1d": 5.866,
    "4d": 5.866,
    "7d": 5.897,
}


def build_experiments(phase: str) -> list[PaperExperiment]:
    """Return representative or full Cross-Unet-only experiment lists."""
    short = "4h"
    long = "7d"
    if phase == "representative":
        return [
            PaperExperiment("S-1", "nwp", short),
            PaperExperiment("S-1", "nwp", long),
            PaperExperiment("S-1", "satellite", short),
            PaperExperiment("KDASC", "satellite", long),
            PaperExperiment("S-1", "ai", short),
            PaperExperiment("S-1", "ai", long),
        ]

    experiments: list[PaperExperiment] = []
    for station in ["S-1", "S-2", "S-3", "S-4"]:
        experiments.extend(
            PaperExperiment(station, "nwp", horizon) for horizon in PAPER_HORIZONS
        )
    for station in ["S-1", "S-2", "S-3", "S-4", "KDASC"]:
        experiments.extend(
            PaperExperiment(station, "satellite", horizon) for horizon in PAPER_HORIZONS
        )
    for station in ["S-1", "S-2", "S-3", "S-4"]:
        experiments.extend(
            PaperExperiment(station, "ai", horizon) for horizon in PAPER_HORIZONS
        )
    return experiments


def filter_experiments(
    experiments: list[PaperExperiment], cases_csv: Path | None
) -> list[PaperExperiment]:
    """Restrict experiments to an optional station/source/horizon CSV."""
    if cases_csv is None:
        return experiments
    df = pd.read_csv(cases_csv)
    wanted = {
        (row["station"], row["source"], row["horizon_label"])
        for _, row in df.iterrows()
    }
    return [experiment for experiment in experiments if experiment.key in wanted]


def load_target_csv(path: Path | None) -> dict[tuple[str, str, str], dict[str, float]]:
    """Load optional paper targets from a CSV file."""
    targets = dict(PAPER_TARGETS)
    if path is None:
        return targets
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        key = (row["station"], row["source"], row["horizon_label"])
        targets[key] = {
            "mae": float(row["paper_mae"]),
            "mse": float(row["paper_mse"]),
            "r2": float(row["paper_r2"]),
        }
    return targets


def pass_with_tolerance(
    metric: str, value: float, paper_value: float | None
) -> bool | None:
    """Return whether a metric is within the requested 5 percent tolerance."""
    if paper_value is None or np.isnan(paper_value):
        return None
    if metric == "r2":
        return value >= paper_value * 0.95
    return value <= paper_value * 1.05


def update_row_with_target(
    row: dict[str, Any],
    targets: dict[tuple[str, str, str], dict[str, float]],
) -> dict[str, Any]:
    """Attach paper targets and tolerance checks to an existing metrics row."""
    updated = dict(row)
    target = targets.get(row_key(updated), {})
    paper_mae = target.get("mae")
    paper_mse = target.get("mse")
    paper_r2 = target.get("r2")
    updated["paper_mae"] = "" if paper_mae is None else f"{paper_mae:.8g}"
    updated["paper_mse"] = "" if paper_mse is None else f"{paper_mse:.8g}"
    updated["paper_r2"] = "" if paper_r2 is None else f"{paper_r2:.8g}"
    updated["pass_mae"] = (
        pass_with_tolerance("mae", float(updated["mae"]), paper_mae)
        if updated.get("mae", "") != ""
        else None
    )
    updated["pass_mse"] = (
        pass_with_tolerance("mse", float(updated["mse"]), paper_mse)
        if updated.get("mse", "") != ""
        else None
    )
    updated["pass_r2"] = (
        pass_with_tolerance("r2", float(updated["r2"]), paper_r2)
        if updated.get("r2", "") != ""
        else None
    )
    return updated


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["station"], row["source"], row["horizon_label"])


def read_existing_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as fp:
        return {row_key(row): row for row in csv.DictReader(fp)}


def write_metrics(path: Path, rows: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(
                {column: rows[key].get(column, "") for column in CSV_COLUMNS}
            )


def write_report(path: Path, rows: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-Unet Paper Reproduction",
        "",
        "| Station | Source | Horizon | MAE | MSE | R2 | Paper MAE | Paper MSE | Paper R2 | Pass | Params |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for key in sorted(rows):
        row = rows[key]
        pass_values = [row.get("pass_mae"), row.get("pass_mse"), row.get("pass_r2")]
        if any(value in ("", None) for value in pass_values):
            result_text = "target missing"
        else:
            result_text = (
                "yes" if all(str(value) == "True" for value in pass_values) else "no"
            )
        lines.append(
            "| {station} | {source} | {horizon_label} | {mae} | {mse} | {r2} | "
            "{paper_mae} | {paper_mse} | {paper_r2} | {result_text} | {param_count} |".format(
                result_text=result_text,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Parameter Count Check",
            "",
            "Table F7 reports Cross-Unet parameters in millions for the AI-weather setting.",
            "",
            "| Horizon | Measured Params (M) | Paper Params (M) | Within 5% |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    seen_horizons = sorted(
        {key[2] for key in rows}, key=list(PAPER_PARAM_TARGETS_M).index
    )
    for horizon in seen_horizons:
        paper_params_m = PAPER_PARAM_TARGETS_M[horizon]
        ai_rows = [
            row
            for key, row in rows.items()
            if key[1] == "ai" and key[2] == horizon and row.get("param_count", "") != ""
        ]
        source_rows = ai_rows or [
            row
            for key, row in rows.items()
            if key[2] == horizon and row.get("param_count", "") != ""
        ]
        if not source_rows:
            measured_text = ""
            params_status = "target pending"
        else:
            measured_m = float(source_rows[0]["param_count"]) / 1_000_000.0
            measured_text = f"{measured_m:.3f}"
            params_status = "yes" if measured_m <= paper_params_m * 1.05 else "no"
        lines.append(
            f"| {horizon} | {measured_text} | {paper_params_m:.3f} | {params_status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def target_range_from_checkpoint(checkpoint_path: Path, dataset_root: Path) -> float:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_scaler = checkpoint["target_scaler"]
    if target_scaler.get("kind", "standard") == "minmax":
        target_range = float(np.asarray(target_scaler["std"], dtype=np.float64)[-1])
    else:
        data_config = checkpoint["data_config"]
        range_ds = RealPVDataset(
            RealPVDataConfig(
                dataset_root=dataset_root,
                station_name=data_config["station_name"],
                seq_len=data_config["seq_len"],
                pred_len=data_config["pred_len"],
                weather_source=data_config["weather_source"],
                normalization="minmax",
            ),
            split="train",
        )
        target_range = float(range_ds.target_scaler.std[-1])
    if target_range <= 1.0e-12:
        return 1.0
    return target_range


def prediction_metrics(
    path: Path, checkpoint_path: Path, metric_scale: str, dataset_root: Path
) -> dict[str, float]:
    df = pd.read_csv(path)
    predictions = df["prediction"].to_numpy(dtype=np.float64)
    targets = df["target"].to_numpy(dtype=np.float64)
    if metric_scale == "normalized":
        target_range = target_range_from_checkpoint(checkpoint_path, dataset_root)
        predictions = predictions / target_range
        targets = targets / target_range
    return compute_prediction_metrics(predictions, targets)


def parameter_count(checkpoint_path: Path) -> int:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(sum(value.numel() for value in checkpoint["model_state"].values()))


def make_cfg(args: argparse.Namespace, experiment: PaperExperiment):
    script_dir = Path(__file__).resolve().parent
    cfg = OmegaConf.load(script_dir / "conf" / "real_data.yaml")
    run_dir = (
        args.output_dir
        / "runs"
        / experiment.source
        / experiment.station
        / experiment.horizon_label
    )
    prediction_path = run_dir / "predictions" / "predictions.csv"

    cfg.dataset_root = str(args.dataset_root)
    cfg.station_name = experiment.station
    cfg.weather_source = experiment.source
    cfg.seq_len = experiment.seq_len
    cfg.pred_len = experiment.pred_len
    cfg.paper_preset = True
    cfg.output_dir = str(run_dir)
    cfg.prediction_path = str(prediction_path)
    cfg.metrics_csv_path = str(args.metrics_csv)
    cfg.metric_report_path = str(args.report)
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.max_valid_samples is not None:
        cfg.max_valid_samples = args.max_valid_samples
    if args.max_predict_samples is not None:
        cfg.max_predict_samples = args.max_predict_samples
    return cfg, run_dir, prediction_path


def run_experiment(
    args: argparse.Namespace,
    experiment: PaperExperiment,
    targets: dict[tuple[str, str, str], dict[str, float]],
) -> dict[str, Any]:
    cfg, run_dir, prediction_path = make_cfg(args, experiment)
    checkpoint_path = (
        run_dir
        / experiment.station
        / "checkpoints"
        / f"real_cross_unet_{experiment.station}.pt"
    )
    if not args.predict_only:
        train(cfg)
    predict(cfg)

    metrics = prediction_metrics(
        prediction_path, checkpoint_path, args.metric_scale, args.dataset_root
    )
    row = {
        "station": experiment.station,
        "source": experiment.source,
        "horizon_label": experiment.horizon_label,
        "seq_len": experiment.seq_len,
        "pred_len": experiment.pred_len,
        "seed": 2021,
        "mae": f"{metrics['mae']:.8g}",
        "mse": f"{metrics['mse']:.8g}",
        "r2": f"{metrics['r2']:.8g}",
        "param_count": parameter_count(checkpoint_path),
    }
    return update_row_with_target(row, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["representative", "full"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "paper_repro",
    )
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--paper-targets-csv", type=Path, default=None)
    parser.add_argument("--cases-csv", type=Path, default=None)
    parser.add_argument(
        "--metric-scale",
        choices=["raw", "normalized"],
        default="normalized",
        help="Compare paper MAE/MSE on raw values or normalized by target range.",
    )
    parser.add_argument(
        "--update-targets-only",
        action="store_true",
        help="Only refresh paper target columns and the Markdown report.",
    )
    parser.add_argument(
        "--rerun-below-r2",
        type=float,
        default=None,
        help="With --resume, rerun completed rows whose current R2 is below this value.",
    )
    parser.add_argument(
        "--only-rerun-existing",
        action="store_true",
        help="Only consider rows already present in metrics.csv; useful with --rerun-below-r2.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-predict-samples", type=int, default=None)
    args = parser.parse_args()
    args.metrics_csv = args.metrics_csv or args.output_dir / "metrics.csv"
    args.report = args.report or args.output_dir / "report.md"
    return args


def main() -> None:
    args = parse_args()
    rows = read_existing_metrics(args.metrics_csv)
    targets = load_target_csv(args.paper_targets_csv)
    if args.update_targets_only:
        rows = {key: update_row_with_target(row, targets) for key, row in rows.items()}
        write_metrics(args.metrics_csv, rows)
        write_report(args.report, rows)
        return
    for experiment in filter_experiments(build_experiments(args.phase), args.cases_csv):
        existing_row = rows.get(experiment.key)
        if args.only_rerun_existing and existing_row is None:
            continue
        should_rerun_low_r2 = (
            args.rerun_below_r2 is not None
            and existing_row is not None
            and existing_row.get("r2", "") != ""
            and float(existing_row["r2"]) < args.rerun_below_r2
        )
        if args.resume and experiment.key in rows and not should_rerun_low_r2:
            continue
        rows[experiment.key] = run_experiment(args, experiment, targets)
        write_metrics(args.metrics_csv, rows)
        write_report(args.report, rows)


if __name__ == "__main__":
    main()
