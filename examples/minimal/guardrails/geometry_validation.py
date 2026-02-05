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
Geometry Guardrail Example

This example demonstrates how to use geometry guardrails for validating
CAD/STL files in a production workflow.
"""

import multiprocessing as mp
from pathlib import Path

from physicsnemo.experimental.guardrails import GeometryGuardrail
from physicsnemo.experimental.guardrails.geometry import is_fast_reader_available


def check_optimizations():
    """Check and report available performance optimizations."""
    # Check for GPU
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    # Check for fast reader
    has_fast_reader = is_fast_reader_available()

    return {
        "gpu": has_gpu,
        "fast_reader": has_fast_reader,
        "device": "cuda" if has_gpu else "cpu",
    }


def train_guardrail(train_dir: Path, model_path: Path, device: str = "cpu"):
    """
    Train a geometry guardrail from a directory of STL files.

    Parameters
    ----------
    train_dir : Path
        Directory containing training STL files (known-good geometries).
    model_path : Path
        Path where the trained model will be saved.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
    """
    print(f"\n{'='*60}")
    print("Training Geometry Guardrail")
    print(f"{'='*60}")
    print(f"Training data: {train_dir}")
    print(f"Device: {device}")
    print(f"Workers: {mp.cpu_count() - 1}")

    # Create guardrail
    guardrail = GeometryGuardrail(
        n_components=1,  # Single Gaussian (unimodal assumption)
        warn_pct=99.0,  # Flag top 1% as warnings
        reject_pct=99.9,  # Flag top 0.1% as rejections
        device=device,
        random_state=42,
    )

    # Train from directory
    guardrail.fit_from_dir(
        train_dir,
        n_workers=mp.cpu_count() - 1,
        chunksize=8,
    )

    # Save model
    guardrail.save(model_path)
    print(f"\n✓ Model trained and saved to {model_path}")


def validate_geometries(test_dir: Path, model_path: Path, device: str = "cpu"):
    """
    Validate geometries using a trained guardrail.

    Parameters
    ----------
    test_dir : Path
        Directory containing STL files to validate.
    model_path : Path
        Path to the trained guardrail model.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    dict
        Validation statistics and results.
    """
    print(f"\n{'='*60}")
    print("Validating Geometries")
    print(f"{'='*60}")
    print(f"Test data: {test_dir}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    # Load guardrail
    guardrail = GeometryGuardrail.load(model_path, device=device)

    # Validate all geometries
    results = guardrail.query_from_dir(
        test_dir,
        n_workers=mp.cpu_count() - 1,
        chunksize=8,
    )

    # Compute statistics
    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    reject_count = sum(1 for r in results if r["status"] == "REJECT")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {len(results)} geometries validated")
    print(f"  OK:     {ok_count:3d} ({100*ok_count/len(results):.1f}%)")
    print(f"  WARN:   {warn_count:3d} ({100*warn_count/len(results):.1f}%)")
    print(f"  REJECT: {reject_count:3d} ({100*reject_count/len(results):.1f}%)")
    print(f"{'='*60}\n")

    # Show detailed results
    for r in sorted(results, key=lambda x: x["percentile"], reverse=True):
        status_icon = {"OK": "✓", "WARN": "⚠", "REJECT": "✗"}[r["status"]]
        print(
            f"{status_icon} {r['name']:40s} "
            f"{r['percentile']:6.2f}% "
            f"[{r['status']:6s}]"
        )

    # Highlight rejected geometries
    if reject_count > 0:
        print(f"\n⚠ WARNING: {reject_count} geometries flagged for manual review")
        print("These may be corrupted, out-of-spec, or highly anomalous.")

    return {
        "total": len(results),
        "ok": ok_count,
        "warn": warn_count,
        "reject": reject_count,
        "results": results,
    }


def main():
    """Main execution function."""
    # Example paths (modify these for your data)
    train_dir = Path("data/train_geometries")
    test_dir = Path("data/test_geometries")
    model_path = Path("geometry_guardrail.npz")

    # Check available optimizations
    opts = check_optimizations()
    print(f"\n{'='*60}")
    print("Performance Optimizations")
    print(f"{'='*60}")
    print(f"GPU Available: {'✓' if opts['gpu'] else '✗'}")
    print(f"Fast STL Reader: {'✓' if opts['fast_reader'] else '✗'}")
    print(f"Selected Device: {opts['device']}")

    # Check if model exists, otherwise train
    if model_path.exists():
        print(f"\n✓ Using existing model: {model_path}")
    else:
        print(f"\n✗ No model found at {model_path}")
        if not train_dir.exists():
            print(f"\nERROR: Training directory not found: {train_dir}")
            print("\nPlease create the following directories:")
            print(f"  - {train_dir}  (known-good geometries for training)")
            print(f"  - {test_dir}   (geometries to validate)")
            return

        print("\nTraining new model...")
        train_guardrail(train_dir, model_path, device=opts["device"])

    # Validate test geometries
    if not test_dir.exists():
        print(f"\nERROR: Test directory not found: {test_dir}")
        print(f"Please create {test_dir} and add STL files to validate.")
        return

    stats = validate_geometries(test_dir, model_path, device=opts["device"])

    # Exit with appropriate status
    if stats["reject"] > 0:
        print("\n⚠ Some geometries were rejected. Review flagged files.")
        exit(1)
    else:
        print("\n✓ All geometries passed validation.")
        exit(0)


if __name__ == "__main__":
    main()
