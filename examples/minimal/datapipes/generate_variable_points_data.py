# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Script to generate synthetic data with variable number of points per sample.

Each sample has a semi-random number of points (between min_points and max_points),
while maintaining consistent feature dimensions across fields.

Supports .npz, and zarr storage formats.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_shapes(shapes_str: str) -> Dict[str, Tuple[int, ...]]:
    """
    Parse shape specification from command line.

    Expected format: "key1:dim1,dim2 key2:dim1"
    Example: "velocity:3 pressure:1 temperature:1"

    The shape specifies the feature dimensions (excluding the points dimension).
    For example, "velocity:3" means each point has 3 velocity components.

    Parameters
    ----------
    shapes_str : str
        Space-separated key:shape pairs where shape is comma-separated feature dimensions

    Returns
    -------
    Dict[str, Tuple[int, ...]]
        Dictionary mapping field names to feature dimension tuples
    """
    shapes = {}
    for item in shapes_str.split():
        if ":" not in item:
            raise ValueError(
                f"Invalid shape specification: {item}. Expected format: key:dim1,dim2,..."
            )

        key, dims_str = item.split(":", 1)
        try:
            dims = tuple(int(d) for d in dims_str.split(","))
        except ValueError as e:
            raise ValueError(f"Invalid dimensions for key '{key}': {dims_str}") from e

        shapes[key] = dims

    return shapes


def generate_point_counts(
    num_samples: int, min_points: int, max_points: int, seed: int = 42
) -> np.ndarray:
    """
    Generate random point counts for each sample.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate point counts for
    min_points : int
        Minimum number of points per sample
    max_points : int
        Maximum number of points per sample
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of point counts with shape (num_samples,)
    """
    rng = np.random.RandomState(seed)
    point_counts = rng.randint(min_points, max_points + 1, size=num_samples)
    return point_counts


def generate_variable_sample(
    num_points: int, shapes: Dict[str, Tuple[int, ...]], rng: np.random.RandomState
) -> Dict[str, np.ndarray]:
    """
    Generate a single sample with variable number of points.

    Parameters
    ----------
    num_points : int
        Number of points for this sample
    shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping field names to feature dimension tuples
    rng : np.random.RandomState
        Random number generator

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping field names to generated data arrays
        Each array has shape (num_points, *feature_dims)
    """
    sample_data = {}

    for key, feature_dims in shapes.items():
        full_shape = (num_points,) + feature_dims
        # Generate random data in from a normal distribution, mean 17 and std 3
        sample_data[key] = rng.normal(17.0, 3.0, size=full_shape).astype(np.float32)

    return sample_data


def save_npz(
    num_samples: int,
    point_counts: np.ndarray,
    shapes: Dict[str, Tuple[int, ...]],
    output_dir: Path,
    seed: int,
):
    """
    Save data as separate .npz files per sample with variable points.

    Each sample is saved as an .npz file containing all fields.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    point_counts : np.ndarray
        Array of point counts for each sample
    shapes : Dict[str, Tuple[int, ...]]
        Dictionary of feature dimensions for each field
    output_dir : Path
        Output directory
    seed : int
        Random seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    total_size = 0
    for i in range(num_samples):
        num_points = point_counts[i]
        sample_data = generate_variable_sample(num_points, shapes, rng)

        # Save to individual file
        filepath = output_dir / f"sample_{i:06d}.npz"
        np.savez(filepath, **sample_data)

        # Track size
        total_size += sum(array.nbytes for array in sample_data.values())

        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"  Saved {i + 1}/{num_samples} samples...")

    print(
        f"Saved {num_samples} samples as .npz files ({total_size / 1e6:.2f} MB total)"
    )


def save_zarr(
    num_samples: int,
    point_counts: np.ndarray,
    shapes: Dict[str, Tuple[int, ...]],
    output_dir: Path,
    seed: int,
):
    """
    Save data as separate zarr directories per sample with variable points.

    Each sample is saved in its own zarr directory containing all fields.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    point_counts : np.ndarray
        Array of point counts for each sample
    shapes : Dict[str, Tuple[int, ...]]
        Dictionary of feature dimensions for each field
    output_dir : Path
        Output directory
    seed : int
        Random seed for reproducibility
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for zarr storage backend. Install with: pip install zarr"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    total_size = 0
    for i in range(num_samples):
        num_points = point_counts[i]
        sample_data = generate_variable_sample(num_points, shapes, rng)

        # Create zarr group for this sample
        sample_dir = output_dir / f"sample_{i:06d}.zarr"
        root = zarr.open_group(str(sample_dir), mode="w")

        # Save all fields for this sample
        for key, array in sample_data.items():
            root.create_dataset(
                key,
                data=array,
                shape=array.shape,
            )

        # Track size
        total_size += sum(array.nbytes for array in sample_data.values())

        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"  Saved {i + 1}/{num_samples} samples...")

    print(
        f"Saved {num_samples} samples as zarr directories ({total_size / 1e6:.2f} MB total)"
    )


def save_metadata(
    output_dir: Path,
    num_samples: int,
    shapes: Dict[str, Tuple[int, ...]],
    point_counts: np.ndarray,
    min_points: int,
    max_points: int,
    backend: str,
):
    """
    Save metadata about the generated dataset.

    Parameters
    ----------
    output_dir : Path
        Output directory
    num_samples : int
        Number of samples generated
    shapes : Dict[str, Tuple[int, ...]]
        Feature dimensions of each field
    point_counts : np.ndarray
        Array of point counts for each sample
    min_points : int
        Minimum number of points
    max_points : int
        Maximum number of points
    backend : str
        Storage backend used
    """
    metadata = {
        "num_samples": num_samples,
        "feature_shapes": {k: list(v) for k, v in shapes.items()},
        "point_counts": point_counts.tolist(),
        "min_points": int(min_points),
        "max_points": int(max_points),
        "mean_points": float(np.mean(point_counts)),
        "backend": backend,
        "variable_points": True,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with variable number of points per sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 samples with variable points (30k-100k), 3D velocity and scalar pressure
  python generate_variable_points_data.py -n 100 -s "velocity:3 pressure:1" -b npz -o output/

  # Generate 50 samples with 10k-50k points
  python generate_variable_points_data.py -n 50 -s "coords:3 features:8" --min-points 10000 --max-points 50000 -b zarr

  # Generate point cloud data with normals
  python generate_variable_points_data.py -n 200 -s "xyz:3 normal:3 color:3" -b zarr -o pointcloud_data/
        """,
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        required=True,
        help='Space-separated key:shape pairs for feature dimensions (e.g., "velocity:3 pressure:1")',
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=30000,
        help="Minimum number of points per sample (default: 30000)",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=100000,
        help="Maximum number of points per sample (default: 100000)",
    )

    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["npz", "zarr"],
        default="npz",
        help="Storage backend to use (default: npz)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="synthetic_variable_data",
        help="Output directory (default: synthetic_variable_data)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate point range
    if args.min_points >= args.max_points:
        raise ValueError(
            f"min_points ({args.min_points}) must be less than max_points ({args.max_points})"
        )

    # Parse shapes
    print("Parsing shape specifications...")
    shapes = parse_shapes(args.shapes)
    print(f"Feature shapes: {shapes}")
    print(f"Point range: {args.min_points:,} to {args.max_points:,}")

    # Generate point counts for each sample
    print(f"\nGenerating point counts for {args.num_samples} samples...")
    point_counts = generate_point_counts(
        args.num_samples, args.min_points, args.max_points, seed=args.seed
    )
    print(f"Mean points per sample: {np.mean(point_counts):.0f}")
    print(f"Total points across all samples: {np.sum(point_counts):,}")

    # Save data
    output_dir = Path(args.output)
    print(f"\nSaving data to {output_dir} using backend '{args.backend}'...")

    if args.backend == "npz":
        save_npz(args.num_samples, point_counts, shapes, output_dir, args.seed)
    elif args.backend == "zarr":
        save_zarr(args.num_samples, point_counts, shapes, output_dir, args.seed)

    # Save metadata
    save_metadata(
        output_dir,
        args.num_samples,
        shapes,
        point_counts,
        args.min_points,
        args.max_points,
        args.backend,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
