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
Standalone benchmark: physicsnemo.mesh memmap (.pmsh + MeshReader) vs zarr
as the storage backend for a surface-CFD training datapipe.

Models the unified_external_aero_recipe surface path: per epoch, each sample
is a random contiguous block of ``subsample_n_cells`` cells (plus their
vertex positions and cell fields) read from a full-resolution surface mesh,
returned as a ``physicsnemo.mesh.Mesh``.

Three readers over the same logical data:

- ``pmsh``:      the library ``MeshReader`` over ``Mesh.save()`` memmap stores
                 (the shipping pipeline).
- ``zarr``:      prototype zarr-python reader; one group per sample, cell-axis
                 chunks sized to the subsample block.
- ``zarr-ts``:   same stores read through tensorstore with concurrent
                 per-field async reads.

The zarr variants here are self-contained prototypes that predate (and
motivated) the library's ``ZarrMeshReader``; they isolate storage-format
effects from the library reader's feature set. They exploit
locality-ordered points (cell block -> contiguous point range + offset
remap) instead of MeshReader's unique()-based compaction; per-sample CPU
differences are small next to I/O but are noted in the results. Benchmark
the shipped reader end-to-end with
``benchmark_recipe_pipeline_mesh_vs_zarr.py``.

Synthetic data is a locality-ordered triangle soup with smooth,
spatially-correlated fields so compression ratios are CFD-plausible; pure
random data would understate compressed-zarr throughput.

Usage (synthetic data):
    python benchmarks/physicsnemo/datapipes/benchmark_mesh_vs_zarr.py \
        --data-dir /path/on/target/filesystem [--n-samples 16] \
        [--n-cells 1000000] [--subsample 200000] [--workers 1 8] \
        [--json-out results.json]

Usage (real DrivAerML .pdmsh dataset):
    python benchmarks/physicsnemo/datapipes/benchmark_mesh_vs_zarr.py \
        --data-dir <scratch dir> --real-from <dataset dir with run_*/> \
        --runs 8 [--subsample 200000] [--workers 1 8]

Real mode benchmarks the shipping path (MeshReader over the original
``domain_run_*.pdmsh`` vehicle boundary, via symlinks) against converted
stores. DrivAerML's point ordering has no locality (a contiguous 200k-cell
block references ~583k points scattered over the full 8.8M-point array), so
the zarr conversion denormalizes to a per-cell vertex soup: ~+20% raw bytes
for fully contiguous reads. A ``pmsh soup`` control store (same layout,
memmap format) separates layout effects from format/reader effects.

Cold-cache passes evict pages via posix_fadvise(DONTNEED) per file
(best-effort; on network filesystems client caching may persist -- results
label cold passes accordingly).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

FIELDS = ("pressure", "wss", "normals")  # synthetic cell_data fields
REAL_FIELDS = (
    "CpMeanTrim",
    "pMeanTrim",
    "pPrime2MeanTrim",
    "wallShearStressMeanTrim",
)  # DrivAerML vehicle-boundary cell_data fields


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _make_sample(rng: np.random.Generator, n_cells: int) -> dict[str, np.ndarray]:
    """Locality-ordered triangle soup on a bumpy sphere with smooth fields."""
    # Coherent parameterization -> consecutive cells are spatially adjacent,
    # matching a curated mesh whose points were ordered for block reads.
    t = np.linspace(0.0, 1.0, n_cells, dtype=np.float32)
    theta = t * np.pi
    phi = np.mod(t * n_cells / 137.0, 1.0).astype(np.float32) * 2.0 * np.pi
    r = 1.0 + 0.1 * np.sin(7 * theta) * np.cos(5 * phi)
    centers = np.stack(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ],
        axis=1,
    ).astype(np.float32)

    # Three vertices jittered around each center: triangle soup, points
    # ordered cell-major so cell block [s, s+k) references points [3s, 3(s+k)).
    jitter = rng.normal(scale=2e-3, size=(n_cells, 3, 3)).astype(np.float32)
    points = (centers[:, None, :] + jitter).reshape(-1, 3)
    cells = np.arange(3 * n_cells, dtype=np.int64).reshape(n_cells, 3)

    # Smooth fields + small noise: realistic compressibility.
    noise = rng.normal(scale=0.01, size=n_cells).astype(np.float32)
    pressure = (np.sin(3 * theta) * np.cos(2 * phi) + noise).astype(np.float32)
    wss = np.stack(
        [
            np.sin(2 * theta) + noise,
            np.cos(3 * phi) + noise,
            np.sin(theta + phi) + noise,
        ],
        axis=1,
    ).astype(np.float32)
    normals = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    return {
        "points": points,
        "cells": cells,
        "pressure": pressure,
        "wss": wss,
        "normals": normals.astype(np.float32),
    }


def _write_pmsh(sample: dict[str, np.ndarray], path: Path) -> None:
    from physicsnemo.mesh import Mesh

    mesh = Mesh(
        points=torch.from_numpy(sample["points"]),
        cells=torch.from_numpy(sample["cells"]),
        cell_data={k: torch.from_numpy(sample[k]) for k in FIELDS},
    )
    mesh.save(str(path))


def _write_zarr(
    sample: dict[str, np.ndarray], path: Path, subsample: int, compress: bool
) -> None:
    import zarr

    group = zarr.open_group(str(path), mode="w")
    if compress:
        from zarr.codecs import BloscCodec

        compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    else:
        compressors = None
    for name, arr in sample.items():
        # Cell-axis chunks match the subsample block so one sample read
        # touches at most two chunks per field. Points are cell-major.
        chunk0 = 3 * subsample if name == "points" else subsample
        chunks = (min(chunk0, arr.shape[0]),) + arr.shape[1:]
        z = group.create_array(
            name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunks,
            compressors=compressors,
        )
        z[:] = arr


def generate(
    data_dir: Path, n_samples: int, n_cells: int, subsample: int, seed: int
) -> None:
    stores = {
        "pmsh": data_dir / "pmsh",
        "zarr_zstd": data_dir / "zarr_zstd",
        "zarr_raw": data_dir / "zarr_raw",
    }
    for d in stores.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        sample = _make_sample(rng, n_cells)
        _write_pmsh(sample, stores["pmsh"] / f"sample_{i:04d}.pmsh")
        _write_zarr(
            sample, stores["zarr_zstd"] / f"sample_{i:04d}.zarr", subsample, True
        )
        _write_zarr(
            sample, stores["zarr_raw"] / f"sample_{i:04d}.zarr", subsample, False
        )
        print(f"  generated sample {i + 1}/{n_samples}", flush=True)


def convert_real(dataset_dir: Path, data_dir: Path, runs: int, subsample: int) -> None:
    """Convert DrivAerML vehicle boundaries to soup-layout benchmark stores.

    Creates:
      pmsh_orig/   symlinks to the original vehicle-boundary tensordicts
                   (the shipping MeshReader path, read in place)
      pmsh_soup/   denormalized soup layout, Mesh.save memmap
      zarr_zstd/   soup layout, zarr + blosc-zstd
      zarr_raw/    soup layout, zarr uncompressed
    """
    from physicsnemo.mesh import Mesh

    run_dirs = sorted(
        (d for d in dataset_dir.glob("run_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )[:runs]
    if len(run_dirs) < runs:
        raise ValueError(f"only {len(run_dirs)} run_* dirs under {dataset_dir}")

    stores = {n: data_dir / n for n in ("pmsh_orig", "pmsh_soup", "zarr_zstd", "zarr_raw")}
    for d in stores.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for i, run in enumerate(run_dirs):
        pdmsh = sorted(run.glob("domain_*.pdmsh"))[0]
        vehicle = pdmsh / "_tensordict" / "boundaries" / "vehicle"
        (stores["pmsh_orig"] / f"sample_{i:04d}.pmsh").symlink_to(vehicle)

        mesh = Mesh.load(str(vehicle))
        # Denormalize: per-cell vertex soup -> all reads become contiguous.
        soup_points = mesh.points[mesh.cells].reshape(-1, 3).contiguous().numpy()
        sample = {
            "points": soup_points,
            "cells": np.arange(soup_points.shape[0], dtype=np.int64).reshape(-1, 3),
        }
        for k in REAL_FIELDS:
            sample[k] = mesh.cell_data[k].contiguous().numpy()
        _write_pmsh_soup(sample, stores["pmsh_soup"] / f"sample_{i:04d}.pmsh")
        _write_zarr(sample, stores["zarr_zstd"] / f"sample_{i:04d}.zarr", subsample, True)
        _write_zarr(sample, stores["zarr_raw"] / f"sample_{i:04d}.zarr", subsample, False)
        print(f"  converted {run.name} ({i + 1}/{runs}): "
              f"{mesh.n_cells:,} cells, {mesh.n_points:,} points", flush=True)


def _write_pmsh_soup(sample: dict[str, np.ndarray], path: Path) -> None:
    from physicsnemo.mesh import Mesh

    mesh = Mesh(
        points=torch.from_numpy(sample["points"]),
        cells=torch.from_numpy(sample["cells"]),
        cell_data={k: torch.from_numpy(sample[k]) for k in REAL_FIELDS},
    )
    mesh.save(str(path))


# ---------------------------------------------------------------------------
# Readers (each: __len__, __getitem__(i) -> Mesh with `subsample` cells)
# ---------------------------------------------------------------------------


class PmshReaderAdapter:
    """The shipping path: library MeshReader over memmap stores."""

    def __init__(
        self, path: Path, subsample: int, seed: int, fields: tuple[str, ...] = FIELDS
    ) -> None:
        from physicsnemo.datapipes.readers.mesh import MeshReader

        self.field_names = fields
        self._reader = MeshReader(
            path, pattern="*.pmsh", subsample_n_cells=subsample
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        self._reader.set_generator(gen)

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, index: int):
        mesh, _ = self._reader[index]
        return mesh

    def files(self) -> list[Path]:
        return list(self._reader._paths)


class _ZarrMeshReaderBase:
    """Prototype zarr mesh reader: block-subsample semantics of MeshReader.

    Reads a random contiguous cell block, the corresponding contiguous
    point range (locality-ordered points), remaps indices by offset, and
    builds a Mesh -- the dependent two-stage read the library ZarrReader
    cannot express.
    """

    def __init__(
        self, path: Path, subsample: int, seed: int, fields: tuple[str, ...] = FIELDS
    ) -> None:
        self._paths = sorted(Path(path).glob("*.zarr"))
        if not self._paths:
            raise ValueError(f"no zarr groups under {path}")
        self.field_names = fields
        self._subsample = subsample
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __len__(self) -> int:
        return len(self._paths)

    def _block(self, n_cells: int) -> slice:
        k = self._subsample
        if n_cells <= k:
            return slice(0, n_cells)
        start = torch.randint(0, n_cells - k + 1, (1,), generator=self._gen).item()
        return slice(start, start + k)

    def _read_arrays(self, index: int, sl: slice) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _n_cells(self, index: int) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        from physicsnemo.mesh import Mesh

        sl = self._block(self._n_cells(index))
        arrays = self._read_arrays(index, sl)
        cells = torch.from_numpy(arrays["cells"]) - 3 * sl.start
        return Mesh(
            points=torch.from_numpy(arrays["points"]),
            cells=cells,
            cell_data={k: torch.from_numpy(arrays[k]) for k in self.field_names},
        )

    def files(self) -> list[Path]:
        return list(self._paths)


class ZarrMeshReader(_ZarrMeshReaderBase):
    """zarr-python backend, sequential per-field reads, cached stores."""

    def __init__(
        self, path: Path, subsample: int, seed: int, fields: tuple[str, ...] = FIELDS
    ) -> None:
        super().__init__(path, subsample, seed, fields)
        import zarr

        self._groups = [zarr.open_group(str(p), mode="r") for p in self._paths]

    def _n_cells(self, index: int) -> int:
        return self._groups[index]["cells"].shape[0]

    def _read_arrays(self, index: int, sl: slice) -> dict[str, np.ndarray]:
        g = self._groups[index]
        out = {name: g[name][sl] for name in ("cells",) + self.field_names}
        out["points"] = g["points"][3 * sl.start : 3 * sl.stop]
        return out


class TensorstoreMeshReader(_ZarrMeshReaderBase):
    """tensorstore backend: all per-field reads issued concurrently."""

    def __init__(
        self, path: Path, subsample: int, seed: int, fields: tuple[str, ...] = FIELDS
    ) -> None:
        super().__init__(path, subsample, seed, fields)
        import tensorstore as ts

        self._arrays: list[dict[str, ts.TensorStore]] = []
        for p in self._paths:
            handles = {}
            for name in ("points", "cells") + self.field_names:
                spec = {
                    "driver": "zarr3",
                    "kvstore": {"driver": "file", "path": str(p / name)},
                }
                handles[name] = ts.open(spec, open=True).result()
            self._arrays.append(handles)

    def _n_cells(self, index: int) -> int:
        return self._arrays[index]["cells"].shape[0]

    def _read_arrays(self, index: int, sl: slice) -> dict[str, np.ndarray]:
        h = self._arrays[index]
        futures = {name: h[name][sl].read() for name in ("cells",) + self.field_names}
        futures["points"] = h["points"][3 * sl.start : 3 * sl.stop].read()
        return {name: fut.result() for name, fut in futures.items()}


# ---------------------------------------------------------------------------
# Cache eviction + measurement
# ---------------------------------------------------------------------------


def evict_pages(root: Path) -> int:
    """Best-effort page-cache eviction for every file under root.

    Follows directory symlinks so pmsh_orig stores (symlinks into the
    source dataset) evict the real files' pages.
    """
    n = 0
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                n += 1
            except OSError:
                pass
    return n


def run_pass(reader, order: list[int], workers: int) -> dict:
    latencies: list[float] = []

    fields = getattr(reader, "field_names", FIELDS)

    def task(i: int) -> float:
        t0 = time.perf_counter()
        mesh = reader[i]
        # Touch the data so lazy/memmap tensors actually materialize; a real
        # pipeline consumes every field downstream.
        total = float(mesh.points.sum()) + float(mesh.cells.sum())
        for k in fields:
            total += float(mesh.cell_data[k].sum())
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    if workers == 1:
        latencies = [task(i) for i in order]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(task, order))
    wall = time.perf_counter() - t0

    return {
        "wall_s": wall,
        "samples_per_s": len(order) / wall,
        "latency_p50_ms": 1e3 * statistics.median(latencies),
        "latency_p95_ms": 1e3
        * sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def store_size_mb(root: Path) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total / 1e6


def logical_sample_mb(subsample: int, real: bool = False) -> float:
    """Raw bytes one sample's slices span (i64 cells, f32 everything else).

    Real mode: soup points (3k x 3 f32), cells (k x 3 i64), fields = three
    scalars + one 3-vector = 24 B/cell. The pmsh_orig variant reads the same
    logical bytes but its point gather is scattered (page-granular I/O is
    far larger cold).
    """
    cells = subsample * 3 * 8
    points = 3 * subsample * 3 * 4
    fields = subsample * (24 if real else 28)
    return (cells + points + fields) / 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-cells", type=int, default=1_000_000)
    ap.add_argument(
        "--subsample", type=int, default=200_000,
        help="cells per training sample (recipe sampling_resolution)",
    )
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--epochs", type=int, default=2,
                    help="warm passes after the cold pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse existing stores under --data-dir")
    ap.add_argument("--real-from", type=Path, default=None,
                    help="DrivAerML dataset dir (run_*/domain_*.pdmsh); "
                         "benchmarks real data instead of synthetic")
    ap.add_argument("--runs", type=int, default=8,
                    help="number of run_* cases to use in --real-from mode")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    real = args.real_from is not None
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_generate:
        if real:
            print(f"Converting {args.runs} DrivAerML runs from {args.real_from} ...")
            convert_real(args.real_from, args.data_dir, args.runs, args.subsample)
        else:
            print(f"Generating {args.n_samples} samples x {args.n_cells} cells ...")
            generate(args.data_dir, args.n_samples, args.n_cells, args.subsample,
                     args.seed)

    fields = REAL_FIELDS if real else FIELDS
    variants = {}
    if real:
        variants["pmsh original (MeshReader, shipping path)"] = (
            "pmsh_orig",
            lambda p: PmshReaderAdapter(p, args.subsample, args.seed, fields))
        variants["pmsh soup (MeshReader, relaid-out)"] = (
            "pmsh_soup",
            lambda p: PmshReaderAdapter(p, args.subsample, args.seed, fields))
    else:
        variants["pmsh (MeshReader/memmap)"] = (
            "pmsh", lambda p: PmshReaderAdapter(p, args.subsample, args.seed, fields))
    variants.update({
        "zarr+zstd (zarr-python)": (
            "zarr_zstd",
            lambda p: ZarrMeshReader(p, args.subsample, args.seed, fields)),
        "zarr+zstd (tensorstore)": (
            "zarr_zstd",
            lambda p: TensorstoreMeshReader(p, args.subsample, args.seed, fields)),
        "zarr raw (zarr-python)": (
            "zarr_raw",
            lambda p: ZarrMeshReader(p, args.subsample, args.seed, fields)),
        "zarr raw (tensorstore)": (
            "zarr_raw",
            lambda p: TensorstoreMeshReader(p, args.subsample, args.seed, fields)),
    })

    sample_mb = logical_sample_mb(args.subsample, real)
    print(f"\nLogical bytes per sample read: {sample_mb:.1f} MB")
    results: dict[str, dict] = {"config": vars(args) | {"data_dir": str(args.data_dir)}}
    rng = np.random.default_rng(args.seed)

    for label, (store_name, factory) in variants.items():
        store = args.data_dir / store_name
        reader = factory(store)
        size = store_size_mb(store)
        results[label] = {"store_mb": size, "passes": {}}
        print(f"\n=== {label}  (store: {size:.0f} MB on disk) ===")

        for workers in args.workers:
            order = list(rng.permutation(len(reader)))
            evicted = evict_pages(store)
            cold = run_pass(reader, order, workers)
            warm_list = [
                run_pass(reader, list(rng.permutation(len(reader))), workers)
                for _ in range(args.epochs)
            ]
            warm = min(warm_list, key=lambda r: r["wall_s"])
            for name, r in (("cold", cold), ("warm", warm)):
                r["read_mb_per_s"] = sample_mb * len(reader) / r["wall_s"]
            results[label]["passes"][f"workers={workers}"] = {
                "cold": cold, "warm": warm, "files_evicted": evicted,
            }
            print(
                f"  workers={workers:<2d} "
                f"cold: {cold['samples_per_s']:6.2f} samp/s "
                f"({cold['read_mb_per_s']:7.0f} MB/s, "
                f"p50 {cold['latency_p50_ms']:6.1f} ms) | "
                f"warm: {warm['samples_per_s']:6.2f} samp/s "
                f"({warm['read_mb_per_s']:7.0f} MB/s, "
                f"p50 {warm['latency_p50_ms']:6.1f} ms)"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
