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

"""Prepare the downloaded Rotor37 dataset and fit every training-only transform."""

import argparse
import io
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
import yaml
from basis import fit_transforms
from tqdm import tqdm

DATASET_ID = "PLAID-datasets/Rotor37"
FIELD_NAMES = ("Density", "Pressure", "Temperature")
GLOBAL_NAMES = ("Massflow", "Compression_ratio", "Efficiency")
CONDITION_NAMES = ("Omega", "P")
NORMAL_NAMES = ("NormalsX", "NormalsY", "NormalsZ")


class NumpyUnpickler(pickle.Unpickler):
    """Decode the source NumPy arrays without importing arbitrary globals."""

    ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
    }

    def find_class(self, module, name):
        """Resolve only the NumPy constructors used by the source samples."""
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unsupported pickle global {module}.{name}")
        return self.ALLOWED[module, name]


def child(node, name):
    """Return the unique child of a CGNS tree node with the given name."""
    matches = [entry for entry in node[2] if entry[0] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one CGNS child {name}, found {len(matches)}")
    return matches[0]


def decode_sample(payload, labeled):
    """Read the surface mesh and its vertex data from one source sample."""
    sample = NumpyUnpickler(io.BytesIO(payload)).load()
    if set(sample["meshes"]) != {0.0}:
        raise ValueError("Expected one steady mesh")
    zone = child(child(sample["meshes"][0.0], "Base_2_3"), "Zone")
    coordinates = child(zone, "GridCoordinates")
    point_data = child(zone, "PointData")
    if child(point_data, "GridLocation")[1].tobytes().decode() != "Vertex":
        raise ValueError("Expected fields located at mesh vertices")
    elements = child(zone, "Elements_QUAD_4")
    if int(elements[1][0]) != 7:
        raise ValueError("Expected CGNS QUAD_4 surface elements")
    points = np.column_stack(
        [child(coordinates, f"Coordinate{axis}")[1] for axis in "XYZ"]
    ).astype(np.float32)
    quads = child(elements, "ElementConnectivity")[1].reshape(-1, 4) - 1
    if quads.min() < 0 or quads.max() >= len(points):
        raise ValueError("Mesh connectivity references an invalid vertex")
    scalars = sample["scalars"]
    arrays = {
        "points": points,
        "normals": np.column_stack(
            [child(point_data, name)[1] for name in NORMAL_NAMES]
        ).astype(np.float32),
        "quads": quads.astype(np.int32),
        "conditions": np.asarray(
            [scalars[name] for name in CONDITION_NAMES], np.float32
        ),
    }
    present = {entry[0] for entry in point_data[2]} & set(FIELD_NAMES)
    present |= set(scalars) & set(GLOBAL_NAMES)
    if labeled:
        if present != set(FIELD_NAMES) | set(GLOBAL_NAMES):
            raise ValueError("A labeled sample is missing target values")
        arrays["fields"] = np.column_stack(
            [child(point_data, name)[1] for name in FIELD_NAMES]
        ).astype(np.float32)
        arrays["globals"] = np.asarray(
            [scalars[name] for name in GLOBAL_NAMES], np.float32
        )
    elif present:
        raise ValueError("An official test sample contains target values")
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}")
        if name in ("normals", "fields") and values.shape != points.shape:
            raise ValueError(f"{name} do not align with the mesh vertices")
    return arrays


def make_splits(official_splits, seed=42, validation_fraction=0.1, test_fraction=0.1):
    """Partition the labeled source cases into training, validation and test."""
    labeled = np.asarray(official_splits["train_1000"], dtype=np.int64)
    official_test = [int(sample_id) for sample_id in official_splits["test"]]
    ids = labeled.tolist() + official_test
    if len(set(ids)) != len(ids):
        raise ValueError("Official splits must contain distinct sample IDs")
    shuffled = np.random.default_rng(seed).permutation(labeled)
    n_validation = round(len(shuffled) * validation_fraction)
    n_test = round(len(shuffled) * test_fraction)
    if min(n_validation, n_test) < 1 or n_validation + n_test >= len(shuffled):
        raise ValueError("Each split must contain at least one labeled case")
    return {
        "train": sorted(map(int, shuffled[n_validation + n_test :])),
        "validation": sorted(map(int, shuffled[:n_validation])),
        "test": sorted(map(int, shuffled[n_validation : n_validation + n_test])),
        "official_test": official_test,
    }


def write_samples(raw_dir, output_dir, seed):
    """Decode every source sample and write it with the split manifest."""
    card = (raw_dir / "README.md").read_text()
    metadata = yaml.safe_load(card.split("---", 2)[1])["dataset_info"]
    official = metadata["description"]["split"]
    splits = make_splits(official, seed)
    labeled = set(official["train_1000"])
    shards = sorted((raw_dir / "data").glob("all_samples-*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    if set(range(rows)) != labeled | set(official["test"]):
        raise ValueError("Source rows must match the official sample IDs")
    (output_dir / "samples").mkdir(parents=True)
    samples = {}
    sample_id = 0
    with tqdm(total=rows, desc="Preparing Rotor37") as progress:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=1, columns=["sample"]
            ):
                arrays = decode_sample(batch.column(0)[0].as_py(), sample_id in labeled)
                path = f"samples/{sample_id:06d}.npz"
                np.savez_compressed(output_dir / path, **arrays)
                samples[str(sample_id)] = {
                    "path": path,
                    "labeled": sample_id in labeled,
                }
                sample_id += 1
                progress.update()
    manifest = {
        "source": {"repository": DATASET_ID, "license": "CC-BY-SA-4.0"},
        "split_seed": seed,
        "splits": splits,
        "fields": list(FIELD_NAMES),
        "globals": list(GLOBAL_NAMES),
        "conditions": list(CONDITION_NAMES),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(data_dir, seed=42):
    """Convert ``data_dir/raw`` into the complete ``data_dir/processed`` directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "processed"
    if not (raw_dir / "README.md").is_file() or not list(
        raw_dir.glob("data/*.parquet")
    ):
        raise FileNotFoundError(
            f"Place the downloaded dataset in {raw_dir}, with README.md and data/*.parquet"
        )
    if output_dir.exists():
        raise FileExistsError(f"Prepared data already exist at {output_dir}")
    with TemporaryDirectory(prefix=".rotor37-", dir=data_dir) as staging:
        staging = Path(staging)
        manifest = write_samples(raw_dir, staging, seed)
        fit_transforms(staging)
        staging.rename(output_dir)
    sizes = ", ".join(f"{name} {len(ids)}" for name, ids in manifest["splits"].items())
    print(f"Prepared {output_dir} with {sizes} cases")
    return output_dir


def main():
    """Prepare Rotor37 from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rotor37"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_dir, args.seed)


if __name__ == "__main__":
    main()
