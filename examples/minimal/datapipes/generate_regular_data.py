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
Script to generate synthetic data with configurable shapes and storage backends.

Supports  .npz, and zarr storage formats.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_shapes(shapes_str: str) -> Dict[str, Tuple[int, ...]]:
    """
    Parse shape specification from command line.

    Expected format: "key1:dim1,dim2,dim3 key2:dim1,dim2"
    Example: "velocity:100,64,64 pressure:100,32,32"

    Parameters
    ----------
    shapes_str : str
        Space-separated key:shape pairs where shape is comma-separated dimensions

    Returns
    -------
    Dict[str, Tuple[int, ...]]
        Dictionary mapping field names to shape tuples
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


def generate_synthetic_data(
    num_samples: int, shapes: Dict[str, Tuple[int, ...]], seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic random data.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping field names to shape tuples (per sample)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping field names to generated data arrays
        Each array has shape (num_samples, *shape)
    """
    rng = np.random.RandomState(seed)
    data = {}

    for key, shape in shapes.items():
        full_shape = (num_samples,) + shape
        # Generate random data in range [-7, 13]
        data[key] = rng.uniform(-7.0, 13.0, size=full_shape).astype(np.float32)
        print(f"Generated '{key}' with shape {full_shape}")

    return data


def save_npz(data: Dict[str, np.ndarray], output_dir: Path):
    """
    Save data as separate .npz files per sample.

    Each sample is saved as an .npz file containing all fields.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary of arrays to save, each with shape (num_samples, ...)
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get number of samples (assumes all fields have the same number)
    num_samples = next(iter(data.values())).shape[0]

    for i in range(num_samples):
        # Extract this sample from all fields
        sample_data = {key: array[i] for key, array in data.items()}

        # Save to individual file
        filepath = output_dir / f"sample_{i:06d}.npz"
        np.savez(filepath, **sample_data)

    total_size = sum(array.nbytes for array in data.values())
    print(
        f"Saved {num_samples} samples as .npz files ({total_size / 1e6:.2f} MB total)"
    )


def save_zarr(data: Dict[str, np.ndarray], output_dir: Path):
    """
    Save data as separate zarr directories per sample.

    Each sample is saved in its own zarr directory containing all fields.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary of arrays to save, each with shape (num_samples, ...)
    output_dir : Path
        Output directory
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for zarr storage backend. Install with: pip install zarr"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get number of samples (assumes all fields have the same number)
    num_samples = next(iter(data.values())).shape[0]

    for i in range(num_samples):
        # Create zarr group for this sample
        sample_dir = output_dir / f"sample_{i:06d}.zarr"
        # zarr v3 compatible: pass path directly instead of DirectoryStore
        root = zarr.open_group(str(sample_dir), mode="w")

        # Save all fields for this sample
        for key, array in data.items():
            sample_data = array[i]
            root.create_dataset(
                key,
                data=sample_data,
                shape=sample_data.shape,
            )

    total_size = sum(array.nbytes for array in data.values())
    print(
        f"Saved {num_samples} samples as zarr directories ({total_size / 1e6:.2f} MB total)"
    )


def save_metadata(
    output_dir: Path, num_samples: int, shapes: Dict[str, Tuple[int, ...]], backend: str
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
        Shapes of the generated data
    backend : str
        Storage backend used
    """
    metadata = {
        "num_samples": num_samples,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "backend": backend,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with configurable shapes and storage backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 samples with two fields, save as .npz
  python generate_data.py -n 100 -s "velocity:64,64,64 pressure:32,32,32" -b npz -o output/

  # Generate 200 samples, save as zarr
  python generate_data.py -n 200 -s "u:128,128 v:128,128" -b zarr -o zarr_data/
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
        help='Space-separated key:shape pairs (e.g., "field1:100,200 field2:64,64,64")',
    )

    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["npz", "zarr"],
        default="zarr",
        help="Storage backend to use (default: zarr)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="synthetic_data",
        help="Output directory (default: synthetic_data)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Parse shapes
    print("Parsing shape specifications...")
    shapes = parse_shapes(args.shapes)
    print(f"Shapes: {shapes}")

    # Generate data
    print(f"\nGenerating {args.num_samples} samples...")
    data = generate_synthetic_data(args.num_samples, shapes, seed=args.seed)

    # Save data
    output_dir = Path(args.output)
    print(f"\nSaving data to {output_dir} using backend '{args.backend}'...")

    if args.backend == "npz":
        save_npz(data, output_dir)
    elif args.backend == "zarr":
        save_zarr(data, output_dir)

    # Save metadata
    save_metadata(output_dir, args.num_samples, shapes, args.backend)

    print("\nDone!")


if __name__ == "__main__":
    main()
