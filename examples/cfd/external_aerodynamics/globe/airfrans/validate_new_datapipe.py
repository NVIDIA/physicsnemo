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

"""
Validate the new AirFRANS VTK datapipe against the old monolithic pipeline.

Loads the same VTK sample directories through both pipelines and compares
mapped output fields with configurable tolerances. Reports per-field error
statistics and PASS/FAIL status.

Usage
-----
    python validate.py \\
        --data-dir /path/to/vtk \\
        --task full --split train --n-samples 3

Known differences
-----------------
1. The old pipeline patches ``grad_C_p`` magnitudes > 20 to NaN separately;
   the new pipeline only patches via the C_pt > 1.02 threshold. The validation
   script accounts for this by comparing only mutually-finite values and
   reporting NaN-mask disagreements.

2. Airfoil normal computation uses a slightly different PyVista code path
   between old (``internal.sample(target=airfoil_polydata)``) and new
   (``compute_mesh_quantities`` which extracts surface, computes normals,
   then samples back). Small numerical differences are expected on derived
   force fields.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

from dataset import AirFRANSDataSet, AirFRANSSample
from pipeline import (
    AirFRANSVTKReader,
    ComputeAirfoilNormals,
    ComputeForceCoefficients,
    ComputeFreestreamQuantities,
    ComputeGradients,
    NondimensionalizeFields,
    PatchNonPhysicalValues,
)

# Maps new pipeline field names -> old pipeline field names.
# "points" is a special case extracted from sample.interior_mesh.points;
# all others come from sample.interior_mesh.point_data[old_key].
FIELD_MAP: dict[str, str] = {
    "points": "points",
    "U/|U_inf|": "U/|U_inf|",
    "DeltaU/|U_inf|": "ΔU/|U_inf|",
    "C_p": "C_p",
    "C_pt": "C_pt",
    "ln(1+nut/nu)": "ln(1+nut/nu)",
    "grad_C_p_chord": "∇C_p*chord",
    "C_F_shear": "C_F,shear",
    "C_F_pressure": "C_F,pressure",
    "C_F": "C_F",
}


def compare_tensors(
    new: torch.Tensor,
    old: torch.Tensor,
    atol: float,
) -> dict[str, object]:
    """Compare two tensors element-wise, returning error statistics."""
    result: dict[str, object] = {}

    if new.shape != old.shape:
        result["status"] = "FAIL"
        result["reason"] = f"shape mismatch: new={list(new.shape)} vs old={list(old.shape)}"
        return result

    result["shape"] = list(new.shape)

    new_nan = torch.isnan(new)
    old_nan = torch.isnan(old)
    nan_agree = (new_nan == old_nan).all().item()
    result["nan_mask_agree"] = nan_agree

    new_nan_count = new_nan.sum().item()
    old_nan_count = old_nan.sum().item()
    result["new_nan_count"] = new_nan_count
    result["old_nan_count"] = old_nan_count

    both_finite = ~new_nan & ~old_nan
    n_finite = both_finite.sum().item()
    result["n_finite"] = n_finite

    if n_finite == 0:
        result["status"] = "PASS" if nan_agree else "WARN"
        result["reason"] = "no finite values to compare"
        return result

    diff = (new[both_finite] - old[both_finite]).abs()
    result["max_abs_err"] = diff.max().item()
    result["mean_abs_err"] = diff.mean().item()

    old_abs = old[both_finite].abs()
    nonzero = old_abs > 1e-12
    if nonzero.any():
        rel_err = diff[nonzero] / old_abs[nonzero]
        result["max_rel_err"] = rel_err.max().item()
    else:
        result["max_rel_err"] = float("nan")

    passes = result["max_abs_err"] <= atol
    result["status"] = "PASS" if passes else "FAIL"

    return result


def run_old_pipeline(sample_path: Path) -> AirFRANSSample:
    """Run the old monolithic pipeline on a single sample."""
    return AirFRANSDataSet.preprocess(
        sample_path, patch_out_nonphysical_values=True
    )


def run_new_pipeline(reader: AirFRANSVTKReader, index: int) -> dict[str, torch.Tensor]:
    """Run the new modular pipeline on a single sample."""
    data, _meta = reader[index]

    transforms = [
        ComputeGradients(),
        ComputeAirfoilNormals(),
        ComputeFreestreamQuantities(),
        NondimensionalizeFields(),
        ComputeForceCoefficients(),
        PatchNonPhysicalValues(threshold=1.02),
    ]
    for t in transforms:
        data = t(data)

    return {k: data[k] for k in data.keys()}


def get_old_tensor(sample: AirFRANSSample, old_key: str) -> torch.Tensor:
    """Extract a field from the old pipeline's AirFRANSSample."""
    if old_key == "points":
        return sample.interior_mesh.points
    return sample.interior_mesh.point_data[old_key]


def print_field_table(results: list[tuple[str, dict[str, object]]]) -> None:
    hdr_fmt = "{:<22s} {:>12s} {:>12s} {:>12s} {:>12s} {:>8s} {:>7s}"
    row_fmt = "{:<22s} {:>12s} {:>12s} {:>12s} {:>12s} {:>8s} {:>7s}"
    header = hdr_fmt.format(
        "Field", "MaxAbsErr", "MeanAbsErr", "MaxRelErr", "Shape", "NaN ok?", "Status"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for field_name, r in results:
        if "reason" in r and "max_abs_err" not in r:
            print(
                row_fmt.format(
                    field_name,
                    "---",
                    "---",
                    "---",
                    str(r.get("shape", "?")),
                    "---",
                    str(r["status"]),
                )
            )
            if "reason" in r:
                print(f"  Note: {r['reason']}")
            continue

        shape_str = "x".join(str(s) for s in r.get("shape", []))
        nan_ok = "yes" if r.get("nan_mask_agree", False) else "NO"
        print(
            row_fmt.format(
                field_name,
                f"{r.get('max_abs_err', 0):.2e}",
                f"{r.get('mean_abs_err', 0):.2e}",
                f"{r.get('max_rel_err', 0):.2e}",
                shape_str,
                nan_ok,
                str(r["status"]),
            )
        )
        if not r.get("nan_mask_agree", True):
            print(
                f"  NaN counts: new={r['new_nan_count']}, old={r['old_nan_count']}"
            )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate new VTK datapipe against old pipeline"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="full")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for PASS/FAIL per field",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    sample_paths = AirFRANSDataSet.get_split_paths(
        data_dir, task=args.task, split=args.split
    )
    n = min(args.n_samples, len(sample_paths))

    reader = AirFRANSVTKReader(
        data_dir=data_dir, task=args.task, split=args.split
    )

    print(f"=== AirFRANS Datapipe Validation ===")
    print(f"  Data dir:   {data_dir}")
    print(f"  Task:       {args.task}")
    print(f"  Split:      {args.split}")
    print(f"  Tolerance:  {args.atol}")
    print(f"  Samples:    {n} / {len(sample_paths)}")
    print()

    aggregate_status: dict[str, list[str]] = defaultdict(list)

    for i in range(n):
        sample_path = sample_paths[i]
        print(f"--- Sample {i}: {sample_path.name} ---")

        old_sample = run_old_pipeline(sample_path)
        new_data = run_new_pipeline(reader, i)

        sample_results: list[tuple[str, dict[str, object]]] = []

        for new_key, old_key in FIELD_MAP.items():
            if new_key not in new_data:
                sample_results.append(
                    (new_key, {"status": "SKIP", "reason": "not in new output"})
                )
                aggregate_status[new_key].append("SKIP")
                continue

            try:
                old_tensor = get_old_tensor(old_sample, old_key)
            except KeyError:
                sample_results.append(
                    (new_key, {"status": "SKIP", "reason": f"'{old_key}' not in old sample"})
                )
                aggregate_status[new_key].append("SKIP")
                continue

            new_tensor = new_data[new_key]
            if isinstance(old_tensor, torch.Tensor):
                old_tensor = old_tensor.float()
            if isinstance(new_tensor, torch.Tensor):
                new_tensor = new_tensor.float()

            r = compare_tensors(new_tensor, old_tensor, args.atol)
            sample_results.append((new_key, r))
            aggregate_status[new_key].append(r["status"])

        print_field_table(sample_results)
        print()

    # --- Aggregate summary ---
    print("=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    all_pass = True
    for field, statuses in aggregate_status.items():
        fail_count = sum(1 for s in statuses if s == "FAIL")
        pass_count = sum(1 for s in statuses if s == "PASS")
        skip_count = sum(1 for s in statuses if s == "SKIP")
        warn_count = sum(1 for s in statuses if s == "WARN")
        overall = "PASS" if fail_count == 0 else "FAIL"
        if overall == "FAIL":
            all_pass = False
        parts = []
        if pass_count:
            parts.append(f"{pass_count} pass")
        if fail_count:
            parts.append(f"{fail_count} FAIL")
        if warn_count:
            parts.append(f"{warn_count} warn")
        if skip_count:
            parts.append(f"{skip_count} skip")
        print(f"  {field:<22s}  {overall:>6s}  ({', '.join(parts)})")

    print()
    if all_pass:
        print("ALL FIELDS PASS across all samples.")
    else:
        print("SOME FIELDS FAILED. See per-sample tables above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
