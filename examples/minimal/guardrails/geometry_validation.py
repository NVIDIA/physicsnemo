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
Geometry Guardrail Example with DrivAerML and AhmedML Datasets

This example demonstrates geometry guardrails using real-world automotive datasets:
- DrivAerML: 500 parametrically morphed DrivAer vehicle variants
- AhmedML: 500 Ahmed body variations

The example runs three experiments:
1. GMM: Train on DrivAerML, test on DrivAerML validation
2. PCE: Train on DrivAerML, test on DrivAerML validation
3. GMM Cross-dataset: Train on DrivAerML, test on AhmedML
"""

import multiprocessing as mp
from pathlib import Path

from physicsnemo.experimental.guardrails import GeometryGuardrail


def prepare_datasets(
    train_dir: Path,
    val_dir: Path,
    ahmedml_dir: Path,
) -> None:
    """Verify that datasets are properly downloaded and organized."""
    if not train_dir.exists() or not list(train_dir.glob("drivaer_*.stl")):
        raise FileNotFoundError(f"DrivAerML training directory not found: {train_dir}")
    if not val_dir.exists() or not list(val_dir.glob("drivaer_*.stl")):
        raise FileNotFoundError(f"DrivAerML validation directory not found: {val_dir}")
    if not ahmedml_dir.exists() or not list(ahmedml_dir.glob("ahmed_*.stl")):
        raise FileNotFoundError(f"AhmedML directory not found: {ahmedml_dir}")


def fit_and_score(
    train_dir: Path,
    test_dir: Path,
    method: str,
    experiment_name: str,
    model_path: Path,
    device: str,
    gmm_components: int = 1,
    pce_components: int | None = None,
) -> dict:
    """Train a guardrail and evaluate on test data."""
    print(f"\n{experiment_name} ({method.upper()})")

    # Create guardrail
    if method == "gmm":
        guardrail = GeometryGuardrail(
            method="gmm",
            gmm_components=gmm_components,
            warn_pct=99.0,
            reject_pct=99.9,
            device=device,
        )
    else:  # pce
        guardrail = GeometryGuardrail(
            method="pce",
            pce_components=3,
            warn_pct=99.0,
            reject_pct=99.9,
            device=device,
        )

    # Train
    guardrail.fit_from_dir(train_dir, n_workers=mp.cpu_count() - 1)
    guardrail.save(model_path)

    # Evaluate
    results = guardrail.query_from_dir(test_dir, n_workers=mp.cpu_count() - 1)

    # Compute statistics
    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    reject_count = sum(1 for r in results if r["status"] == "REJECT")
    total = len(results)

    print(
        f"OK: {ok_count} ({100 * ok_count / total:.1f}%) | "
        f"WARN: {warn_count} ({100 * warn_count / total:.1f}%) | "
        f"REJECT: {reject_count} ({100 * reject_count / total:.1f}%)"
    )

    return {
        "method": experiment_name,
        "total": total,
        "ok": ok_count,
        "warn": warn_count,
        "reject": reject_count,
        "results": results,
    }


def main():
    """Main execution function."""
    data_dir = Path("data")
    train_dir = data_dir / "drivaerml_train"
    val_dir = data_dir / "drivaerml_val"
    ahmedml_dir = data_dir / "ahmedml"
    device = "cuda"

    prepare_datasets(train_dir, val_dir, ahmedml_dir)

    # Experiment 1: GMM - DrivAerML train → DrivAerML validation
    stats1 = fit_and_score(
        train_dir=train_dir,
        test_dir=val_dir,
        method="gmm",
        experiment_name="GMM - DrivAerML Train → DrivAerML Validation",
        model_path=Path("drivaerml_gmm.npz"),
        device=device,
        gmm_components=1,
    )

    # Experiment 2: PCE - DrivAerML train → DrivAerML validation
    stats2 = fit_and_score(
        train_dir=train_dir,
        test_dir=val_dir,
        method="pce",
        experiment_name="PCE - DrivAerML Train → DrivAerML Validation",
        model_path=Path("drivaerml_pce.npz"),
        device=device,
    )

    # Experiment 3: GMM - DrivAerML train → AhmedML (cross-dataset)
    stats3 = fit_and_score(
        train_dir=train_dir,
        test_dir=ahmedml_dir,
        method="gmm",
        experiment_name="GMM - DrivAerML Train → AhmedML (Cross-Dataset)",
        model_path=Path("drivaerml_gmm.npz"),
        device=device,
        gmm_components=1,
    )

    print("All experiments completed")


if __name__ == "__main__":
    main()
