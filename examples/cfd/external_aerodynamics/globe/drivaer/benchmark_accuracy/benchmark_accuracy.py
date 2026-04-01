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

r"""Benchmark GLOBE accuracy on DrivAerML against GeoTransolver Table 1 metrics.

Evaluates a trained GLOBE model on the DrivAerML test set (48 validation
designs), computing the *exact* metrics reported in GeoTransolver (arXiv
2512.20399, Table 1):

- **p_s**: surface pressure relative L1 error (%)
- **tau_w**: wall shear stress relative L1 error (%)
- **C_D**: drag coefficient R² across test designs
- **C_L**: lift coefficient R² across test designs

Supports multi-GPU inference via ``torchrun``: samples are distributed
round-robin across ranks, each rank loads the model independently, and
results are gathered on rank 0.

Metric Definitions (matching ``transformer_models/src/metrics.py``)
-------------------------------------------------------------------

For each test sample *b* with *N_b* surface points:

.. math::

    \text{relL1}_{p_s}^{(b)} =
        \frac{\sum_{i=1}^{N_b} |C_{p,i}^{\text{pred}} - C_{p,i}^{\text{true}}|}
             {\sum_{i=1}^{N_b} |C_{p,i}^{\text{true}}|}

    \text{relL1}_{\tau_w}^{(b)} =
        \frac{\sum_{i=1}^{N_b} \bigl|\|\mathbf{C}_{f,i}^{\text{pred}}\|
              - \|\mathbf{C}_{f,i}^{\text{true}}\|\bigr|}
             {\sum_{i=1}^{N_b} \|\mathbf{C}_{f,i}^{\text{true}}\|}

Then ``p_s = mean_b(relL1_p_s) * 100`` and ``tau_w = mean_b(relL1_tau_w) * 100``.

Why These Metrics Are Directly Comparable
-----------------------------------------

The GeoTransolver pipeline stores surface fields as ``pMeanTrim / (rho * U^2)``
and ``wallShearStressMeanTrim / (rho * U^2)``; GLOBE uses ``C_p = CpMeanTrim``
and ``C_f = wallShearStressMeanTrim / (0.5 * U^2)``. These differ by constant
factors (``1/(2*rho)`` for pressure, ``1/(2*rho)`` for shear). Since the
relative L1 metric is **scale-invariant** -- multiplying all values by a
constant *k* yields ``k * sum|diff| / (k * sum|true|)`` = same ratio -- the
metric value is identical regardless of which nondimensionalization is used,
as long as both pred and true share the same representation (which they do).

Usage
-----

    # Single GPU:
    uv run benchmark_accuracy/benchmark_accuracy.py

    # Multi-GPU (4 GPUs on one node):
    uv run torchrun --nproc-per-node 4 benchmark_accuracy/benchmark_accuracy.py

    # With explicit output directory:
    uv run benchmark_accuracy/benchmark_accuracy.py --output-dir output/my_run
"""

import csv
import logging
import math
import os
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed
import yaml
from dataset import (
    DrivAerMLDataSet,
    DrivAerMLSample,
    compute_surface_force_coefficients,
)

from physicsnemo.core import get_physicsnemo_pkg_info
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh import Mesh
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SampleResult:
    """Per-sample benchmark metrics."""

    run_name: str
    n_points: int
    rel_l1_cp: float
    rel_l1_cf: float
    cd_pred: float
    cd_true: float
    cl_pred: float
    cl_true: float
    inference_s: float


@dataclass
class AggregateResult:
    """Aggregated benchmark metrics across all test samples."""

    n_samples: int
    p_s_pct: float
    tau_w_pct: float
    cd_r2: float
    cl_r2: float


# ═══════════════════════════════════════════════════════════════════════════
# Metric functions
# ═══════════════════════════════════════════════════════════════════════════


def relative_l1_scalar(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Per-sample relative L1 for a scalar field.

    Computes ``sum(|pred - true|) / sum(|true|)`` over all surface points.
    Matches ``l1_pressure_surf`` from ``transformer_models/src/metrics.py``.

    Args:
        pred: Predicted scalar values, shape ``(n_points,)``.
        true: Ground-truth scalar values, shape ``(n_points,)``.

    Returns:
        Relative L1 ratio (dimensionless, not in %).
    """
    return (torch.abs(pred - true).sum() / torch.abs(true).sum()).item()


def relative_l1_vector_magnitude(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Per-sample relative L1 for the magnitude of a vector field.

    First computes the Euclidean magnitude of each vector, then applies the
    relative L1 formula to the magnitudes: ``sum(|mag_pred - mag_true|) /
    sum(mag_true)``. Matches ``l1_wall_shear_stress`` from
    ``transformer_models/src/metrics.py``.

    Args:
        pred: Predicted vectors, shape ``(n_points, 3)``.
        true: Ground-truth vectors, shape ``(n_points, 3)``.

    Returns:
        Relative L1 ratio (dimensionless, not in %).
    """
    mag_pred = torch.linalg.norm(pred, dim=-1)
    mag_true = torch.linalg.norm(true, dim=-1)
    return (torch.abs(mag_pred - mag_true).sum() / mag_true.sum()).item()


def r_squared(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Coefficient of determination (R²) between predicted and true values.

    Args:
        pred: Predicted values, shape ``(n_samples,)``.
        true: Ground-truth values, shape ``(n_samples,)``.

    Returns:
        R² value. 1.0 indicates perfect prediction.
    """
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


# ═══════════════════════════════════════════════════════════════════════════
# Sample-level inference and metrics
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_sample(
    model: GLOBE,
    sample: DrivAerMLSample,
    *,
    device: torch.device,
    n_faces_per_boundary: int,
) -> SampleResult:
    """Run inference on one test sample and compute all metrics.

    Loads the full-resolution prediction mesh (all surface vertices), runs
    GLOBE inference, and computes relative L1 errors for C_p and |C_f| as
    well as integrated force coefficients (Cd, Cl).

    Args:
        model: Trained GLOBE model in eval mode.
        sample: Preprocessed DrivAerML sample (CPU). The vehicle boundary
            will be created by subsampling prediction_mesh.
        device: GPU device for inference.
        n_faces_per_boundary: Number of boundary faces for the vehicle mesh.

    Returns:
        :class:`SampleResult` with all per-sample metrics.
    """
    ### Create vehicle boundary mesh (same subsample logic as __getitem__).
    ### Floor boundaries (slip_floor, no_slip_floor) are already present
    ### from preprocessing / cache and must be kept for models trained with
    ### domain floor BCs.
    sample.boundary_meshes["vehicle"] = DrivAerMLDataSet.subsample_mesh(
        sample.prediction_mesh, n_faces_per_boundary
    )

    ### Precompute boundary lazy geometry (same as prepare(), but without
    ### subsampling prediction points - we want full resolution)
    for mesh in sample.boundary_meshes.values():
        _ = mesh.cell_centroids
        _ = mesh.cell_areas
        _ = mesh.cell_normals

    sample = sample.to(device)

    ### Run inference
    t0 = perf_counter()
    with torch.no_grad():
        pred_mesh = model(
            prediction_points=sample.prediction_mesh.points,
            boundary_meshes=sample.boundary_meshes,
            reference_lengths=sample.reference_lengths,
        )
    torch.cuda.synchronize(device)
    inference_s = perf_counter() - t0

    ### Move to CPU for metric computation (avoids GPU memory pressure
    ### when processing many samples sequentially)
    pred_mesh = pred_mesh.to("cpu")
    sample = sample.to("cpu")

    ### Extract predicted and true fields
    cp_pred: torch.Tensor = pred_mesh.point_data["C_p"]
    cp_true: torch.Tensor = sample.prediction_mesh.point_data["C_p"]
    cf_pred: torch.Tensor = pred_mesh.point_data["C_f"]
    cf_true: torch.Tensor = sample.prediction_mesh.point_data["C_f"]

    ### Relative L1 metrics
    rel_l1_cp = relative_l1_scalar(cp_pred, cp_true)
    rel_l1_cf = relative_l1_vector_magnitude(cf_pred, cf_true)

    ### Force coefficients via surface integration.
    ### Only valid at full resolution (needs cell connectivity spanning all
    ### prediction points). When subsampled, the mesh is a point cloud with
    ### no cells, so force integration is skipped.
    true_coeffs = sample.aero_coefficients
    has_cells = sample.prediction_mesh.n_cells > 0

    if has_cells:
        a_ref = float(sample.dimensional_constants["A_ref"])
        pred_surface = Mesh(
            points=sample.prediction_mesh.points,
            cells=sample.prediction_mesh.cells,
            point_data=pred_mesh.point_data,
        )
        pred_coeffs = compute_surface_force_coefficients(pred_surface, a_ref)
        cd_pred = float(pred_coeffs["Cd"])
        cl_pred = float(pred_coeffs["Cl"])
    else:
        cd_pred = float("nan")
        cl_pred = float("nan")

    return SampleResult(
        run_name="",
        n_points=int(cp_pred.shape[0]),
        rel_l1_cp=rel_l1_cp,
        rel_l1_cf=rel_l1_cf,
        cd_pred=cd_pred,
        cd_true=float(true_coeffs["Cd"]),
        cl_pred=cl_pred,
        cl_true=float(true_coeffs["Cl"]),
        inference_s=inference_s,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_results(results: list[SampleResult]) -> AggregateResult:
    """Compute Table 1-style aggregate metrics from per-sample results.

    Args:
        results: Per-sample results from :func:`evaluate_sample`.

    Returns:
        Aggregated metrics matching GeoTransolver Table 1 format.
    """
    n = len(results)
    p_s_pct = sum(r.rel_l1_cp for r in results) / n * 100
    tau_w_pct = sum(r.rel_l1_cf for r in results) / n * 100

    ### Force coefficients: only include samples with valid (non-NaN) values.
    ### NaN occurs when max_prediction_points is used (point cloud, no cells).
    valid = [r for r in results if not math.isnan(r.cd_pred)]
    if valid:
        cd_pred = torch.tensor([r.cd_pred for r in valid])
        cd_true = torch.tensor([r.cd_true for r in valid])
        cl_pred = torch.tensor([r.cl_pred for r in valid])
        cl_true = torch.tensor([r.cl_true for r in valid])
        cd_r2 = r_squared(cd_pred, cd_true)
        cl_r2 = r_squared(cl_pred, cl_true)
    else:
        cd_r2 = float("nan")
        cl_r2 = float("nan")

    return AggregateResult(
        n_samples=n,
        p_s_pct=p_s_pct,
        tau_w_pct=tau_w_pct,
        cd_r2=cd_r2,
        cl_r2=cl_r2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════

H_LINE = "\u2500"


def print_table1(agg: AggregateResult) -> None:
    """Print results in the same format as GeoTransolver Table 1."""
    w = 60
    print(f"\n{'=' * w}")
    print(f"  GLOBE DrivAerML Accuracy ({agg.n_samples} test designs)")
    print(f"{'=' * w}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {H_LINE * 30} {H_LINE * 12}")
    print(f"  {'p_s  relative L1 (%%)':<30} {agg.p_s_pct:>11.2f}%")
    print(f"  {'tau_w relative L1 (%%)':<30} {agg.tau_w_pct:>11.2f}%")
    print(f"  {'C_D R^2':<30} {agg.cd_r2:>12.3f}")
    print(f"  {'C_L R^2':<30} {agg.cl_r2:>12.3f}")
    print(f"{'=' * w}")

    print("\n  GeoTransolver reference (Table 1):")
    print(f"  {H_LINE * 44}")
    print(f"  {'p_s':<30} {'2.86%':>12}")
    print(f"  {'tau_w':<30} {'4.90%':>12}")
    print(f"  {'C_D R^2':<30} {'0.996':>12}")
    print(f"  {'C_L R^2':<30} {'0.991':>12}")
    print()


def print_per_sample(results: list[SampleResult]) -> None:
    """Print per-sample metrics table."""
    print(
        f"\n  {'Sample':<12} {'N pts':>8} {'p_s L1%':>8} "
        f"{'tau_w L1%':>9} {'Cd pred':>8} {'Cd true':>8} "
        f"{'Cl pred':>8} {'Cl true':>8} {'Time(s)':>8}"
    )
    print(
        f"  {H_LINE * 12} {H_LINE * 8} {H_LINE * 8} "
        f"{H_LINE * 9} {H_LINE * 8} {H_LINE * 8} "
        f"{H_LINE * 8} {H_LINE * 8} {H_LINE * 8}"
    )
    for r in results:
        print(
            f"  {r.run_name:<12} {r.n_points:>8,} {r.rel_l1_cp * 100:>7.2f}% "
            f"{r.rel_l1_cf * 100:>8.2f}% {r.cd_pred:>8.4f} {r.cd_true:>8.4f} "
            f"{r.cl_pred:>8.4f} {r.cl_true:>8.4f} {r.inference_s:>7.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    n_faces_per_boundary: int | None = None,
    max_prediction_points: int | None = None,
    results_dir: Path | None = None,
    verbose: bool = True,
):
    """Benchmark GLOBE accuracy on DrivAerML test set.

    Evaluates a trained GLOBE model on all 48 DrivAerML validation designs
    and reports metrics matching GeoTransolver Table 1 (arXiv 2512.20399).

    Supports multi-GPU inference via ``torchrun``: the 48 test samples are
    distributed round-robin across ranks, and results are gathered on rank 0.

    Args:
        data_dir: Path to the DrivAerML dataset root (containing ``run_N/``
            subdirectories). Falls back to ``DRIVAER_DATA_DIR`` env var.
        output_dir: Path to the trained GLOBE output directory containing
            ``best_model.mdlus`` and ``hyperparameters.yaml``. If ``None``,
            uses ``GLOBE_OUTPUT_DIR`` env var or auto-detects the most recent
            output subdirectory.
        n_faces_per_boundary: Override for the vehicle boundary face count.
            If ``None``, reads from ``hyperparameters.yaml``.
        max_prediction_points: If set, randomly subsample each prediction
            mesh to this many points. ``None`` (default) uses all surface
            vertices for full-resolution evaluation.
        results_dir: Directory for output files (timestamped YAML + CSV).
            Defaults to ``benchmark_accuracy/`` (this script's directory).
        verbose: Print per-sample results table.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(1)

    ### [Distributed Setup]
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.cuda.set_device(device)

    is_rank0 = dist.rank == 0
    if is_rank0:
        logging.basicConfig(level=logging.INFO)
    else:
        warnings.filterwarnings("ignore")
        logging.disable(logging.ERROR)

    logger = PythonLogger("globe.drivaer.benchmark_accuracy")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    torch.set_float32_matmul_precision("high")

    ### Resolve data directory
    if data_dir is None:
        if _data_env := os.environ.get("DRIVAER_DATA_DIR"):
            data_dir = Path(_data_env)
        else:
            if is_rank0:
                print(
                    "ERROR: DrivAerML data directory not specified. "
                    "Pass --data-dir or set DRIVAER_DATA_DIR.",
                    file=sys.stderr,
                )
            sys.exit(1)
    data_dir = Path(data_dir)

    ### Resolve output directory (same pattern as inference.py)
    drivaer_root = Path(__file__).parent.parent
    if output_dir is None:
        if _env_dir := os.environ.get("GLOBE_OUTPUT_DIR"):
            output_dir = Path(_env_dir)
        else:
            _output_root = drivaer_root / "output"
            output_subdirs = [d for d in _output_root.iterdir() if d.is_dir()]
            if not output_subdirs:
                if is_rank0:
                    print(
                        f"ERROR: No output directories in {_output_root}",
                        file=sys.stderr,
                    )
                sys.exit(1)
            output_dir = max(output_subdirs, key=lambda d: d.stat().st_mtime)
    logger0.info(f"Output directory: {output_dir}")

    ### Load hyperparameters and model (every rank loads independently)
    hp_path = output_dir / "hyperparameters.yaml"
    if not hp_path.exists():
        if is_rank0:
            print(f"ERROR: {hp_path} not found.", file=sys.stderr)
        sys.exit(1)
    hyperparameters = yaml.safe_load(hp_path.read_text())

    if n_faces_per_boundary is None:
        n_faces_per_boundary = hyperparameters.get("n_faces_per_boundary", 80_000)
    logger0.info(f"Boundary faces: {n_faces_per_boundary:,}")

    best_model_path = output_dir / "best_model.mdlus"
    if not best_model_path.exists():
        if is_rank0:
            print(f"ERROR: {best_model_path} not found.", file=sys.stderr)
        sys.exit(1)

    model = GLOBE.from_checkpoint(best_model_path).to(device)
    model.eval()
    logger0.info(f"Loaded model from {best_model_path.name}")

    ### Set up result cache directory for resumability.
    ### Each completed sample is saved to cache/{run_name}.pt so
    ### the job can be resubmitted with --dependency=singleton and pick
    ### up where it left off.
    if results_dir is None:
        results_dir = Path(__file__).parent
    results_dir = Path(results_dir)
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ### Load validation sample paths and distribute across ranks
    data_cache_dir = drivaer_root / "cache"
    all_val_paths = DrivAerMLDataSet.get_split_paths(data_dir, "validation")
    my_val_paths = all_val_paths[dist.rank :: dist.world_size]

    logger0.info(
        f"Test samples: {len(all_val_paths)} total, "
        f"{dist.world_size} GPU{'s' if dist.world_size > 1 else ''}"
    )

    if max_prediction_points is not None:
        logger0.info(f"Max prediction points per sample: {max_prediction_points:,}")

    ### Evaluate this rank's subset of samples (with resume support)
    my_results: list[SampleResult] = []

    for i, sample_path in enumerate(my_val_paths):
        run_name = sample_path.name

        ### Check per-sample cache for previously completed results
        sample_result_path = cache_dir / f"{run_name}.pt"
        if sample_result_path.exists():
            result = torch.load(sample_result_path, weights_only=False)
            logger.info(
                f"[rank {dist.rank}] "
                f"[{i + 1}/{len(my_val_paths)}] {run_name} (cached)"
            )
            my_results.append(result)
            continue

        logger.info(
            f"[rank {dist.rank}] "
            f"[{i + 1}/{len(my_val_paths)}] {run_name}..."
        )

        ### Load full-resolution sample from data cache or raw data
        cache_pt = (data_cache_dir / sample_path.name).with_suffix(".pt")
        if cache_pt.exists():
            sample: DrivAerMLSample = torch.load(cache_pt, weights_only=False)
        else:
            sample = DrivAerMLDataSet.preprocess(sample_path)

        ### Optionally subsample prediction points. Force coefficient
        ### integration requires cell connectivity spanning all prediction
        ### points, so it is only valid at full resolution.
        if max_prediction_points is not None:
            n_pts = min(max_prediction_points, sample.prediction_mesh.n_points)
            mask = torch.randperm(sample.prediction_mesh.n_points)[:n_pts]
            sample.prediction_mesh = (
                sample.prediction_mesh.to_point_cloud().slice_points(mask)
            )

        result = evaluate_sample(
            model,
            sample,
            device=device,
            n_faces_per_boundary=n_faces_per_boundary,
        )
        result.run_name = run_name

        ### Save to per-sample cache for resumability
        torch.save(result, sample_result_path)

        logger.info(
            f"[rank {dist.rank}]   "
            f"p_s={result.rel_l1_cp * 100:.2f}%  "
            f"tau_w={result.rel_l1_cf * 100:.2f}%  "
            f"Cd={result.cd_pred:.4f} (true {result.cd_true:.4f})  "
            f"[{result.inference_s:.1f}s, {result.n_points:,} pts]"
        )
        my_results.append(result)

    ### Gather results on rank 0
    if dist.world_size > 1:
        gathered: list[list[SampleResult] | None] = [None] * dist.world_size
        torch.distributed.gather_object(
            my_results, gathered if is_rank0 else None, dst=0
        )
        if is_rank0:
            results: list[SampleResult] = []
            for shard in gathered:
                if shard is not None:
                    results.extend(shard)
        else:
            return
    else:
        results = my_results

    ### Aggregate and report (rank 0 only from here)
    agg = aggregate_results(results)
    print_table1(agg)

    if verbose:
        print_per_sample(results)

    total_time = sum(r.inference_s for r in results)
    logger0.info(f"Total inference time: {total_time:.1f}s")

    pkg_info = get_physicsnemo_pkg_info()
    gpu_name = torch.cuda.get_device_name(device)

    provenance = {
        "timestamp_utc": timestamp,
        "evaluated_output_dir": str(output_dir),
        "evaluated_output_name": output_dir.name,
        "best_model_path": str(best_model_path),
        "data_dir": str(data_dir),
        "physicsnemo_version": pkg_info.get("version"),
        "physicsnemo_git_hash": pkg_info.get("git_hash"),
        "gpu": gpu_name,
        "world_size": dist.world_size,
        "n_faces_per_boundary_eval": n_faces_per_boundary,
        "max_prediction_points": max_prediction_points,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "hostname": os.environ.get("SLURM_NODELIST", os.uname().nodename),
    }

    output_doc = {
        "provenance": provenance,
        "training_hyperparameters": hyperparameters,
        "metrics": asdict(agg),
        "per_sample": [asdict(r) for r in results],
    }

    stem = f"results_{timestamp}"
    yaml_path = results_dir / f"{stem}.yaml"
    csv_path = results_dir / f"{stem}.csv"

    yaml_path.write_text(
        yaml.safe_dump(output_doc, default_flow_style=False, sort_keys=False)
    )
    logger0.info(f"Results + provenance: {yaml_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    logger0.info(f"Per-sample CSV: {csv_path}")

    ### Clean up per-sample cache now that final outputs are written.
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
