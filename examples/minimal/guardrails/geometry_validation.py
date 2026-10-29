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

import json
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv

from physicsnemo.experimental.uq.ood import DensityScorer
from physicsnemo.experimental.uq.ood.geometry import FeatureSchema, extract_features
from physicsnemo.mesh.io import from_pyvista


def _extract_stl(path: Path):
    """Load a surface using the shared mesh I/O and optional descriptor utility."""
    try:
        mesh = from_pyvista(pv.read(path), manifold_dim=2)
        return path.name, extract_features(mesh), None
    except (OSError, ValueError, RuntimeError) as error:
        return path.name, None, str(error)


def directory_features(directory: Path, n_workers: int = 1):
    """Own file discovery, extraction workers, and invalid-file handling in the recipe."""
    paths = sorted(directory.glob("*.stl"))
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_workers == 1:
        records = list(map(_extract_stl, paths))
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            records = pool.map(_extract_stl, paths)
    names, values = [], []
    for name, value, error in records:
        if error is None:
            names.append(name)
            values.append(value)
        else:
            warnings.warn(f"Skipping invalid mesh {name}: {error}", stacklevel=2)
    if not values:
        raise RuntimeError(f"No valid STL files found in {directory}")
    return names, np.stack(values)


def save_scorer(
    path: Path, scorer: DensityScorer, feature_identity: dict, policy: dict
):
    """Save NumPy state plus application feature identity and decision thresholds."""
    state = scorer.get_state()
    arrays = {
        key: value for key, value in state.items() if isinstance(value, np.ndarray)
    }
    config = {key: value for key, value in state.items() if key not in arrays}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": config,
                "features": feature_identity,
                "policy": policy,
            },
            indent=2,
        )
    )


def load_scorer(path: Path, feature_identity: dict, device: str):
    """Reject mismatched descriptors or units before restoring the fitted scorer."""
    saved = json.loads(path.with_suffix(".json").read_text())
    if saved["features"] != feature_identity:
        raise ValueError("Saved feature schema, units, or preprocessing do not match")
    with np.load(path, allow_pickle=False) as arrays:
        state = saved["config"] | {key: arrays[key] for key in arrays.files}
    scorer = DensityScorer()
    scorer.set_state(state, device=device)
    return scorer, saved["policy"]


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
    pce_components: int | None = 3,
    n_workers: int = 1,
    length_unit: str = "m",
) -> dict:
    """Fit reusable scores; keep directory handling and acceptance policy explicit."""
    print(f"\n{experiment_name} ({method.upper()})")

    scorer = DensityScorer(
        method=method,
        gmm_components=gmm_components,
        pce_components=pce_components,
        device=device,
    )
    identity = {
        "names": list(FeatureSchema.names),
        "version": FeatureSchema.version,
        "hash": FeatureSchema.hash,
        "length_unit": length_unit,
        "preprocessing": "raw geometry descriptors",
    }
    policy = {"warn_pct": 99.0, "reject_pct": 99.9}
    _, reference = directory_features(train_dir, n_workers)
    scorer.fit(reference)
    save_scorer(model_path, scorer, identity, policy)
    scorer, policy = load_scorer(model_path, identity, device)

    names, query = directory_features(test_dir, n_workers)
    scores = scorer.score(query)
    percentiles = scorer.percentiles(scores)
    results = [
        {
            "name": name,
            "score": float(score),
            "percentile": float(percentile),
            "status": "REJECT"
            if percentile >= policy["reject_pct"]
            else "WARN"
            if percentile >= policy["warn_pct"]
            else "OK",
        }
        for name, score, percentile in zip(names, scores.cpu().numpy(), percentiles)
    ]

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
    n_workers = max(1, min(8, mp.cpu_count() - 1))

    prepare_datasets(train_dir, val_dir, ahmedml_dir)

    # Experiment 1: GMM - DrivAerML train → DrivAerML validation
    fit_and_score(
        train_dir=train_dir,
        test_dir=val_dir,
        method="gmm",
        experiment_name="GMM - DrivAerML Train → DrivAerML Validation",
        model_path=Path("drivaerml_gmm.npz"),
        device=device,
        n_workers=n_workers,
        gmm_components=1,
    )

    # Experiment 2: PCE - DrivAerML train → DrivAerML validation
    fit_and_score(
        train_dir=train_dir,
        test_dir=val_dir,
        method="pce",
        experiment_name="PCE - DrivAerML Train → DrivAerML Validation",
        model_path=Path("drivaerml_pce.npz"),
        device=device,
        n_workers=n_workers,
    )

    # Experiment 3: GMM - DrivAerML train → AhmedML (cross-dataset)
    fit_and_score(
        train_dir=train_dir,
        test_dir=ahmedml_dir,
        method="gmm",
        experiment_name="GMM - DrivAerML Train → AhmedML (Cross-Dataset)",
        model_path=Path("drivaerml_gmm.npz"),
        device=device,
        n_workers=n_workers,
        gmm_components=1,
    )

    print("All experiments completed")


if __name__ == "__main__":
    main()
