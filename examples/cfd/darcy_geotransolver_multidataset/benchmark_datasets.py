# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load and benchmark each dataset (numpy and hdf5) separately.
# Usage: python benchmark_datasets.py data.numpy_path=/path/to/npz data.hdf5_path=/path/to/h5

import time
from collections import defaultdict

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from physicsnemo import datapipes

def _bench_dataset(name: str, dataset, n_iters: int = 2, n_samples=None) -> None:
    """Run n_iters passes over the dataset (or first n_samples per pass) and report throughput."""
    n = len(dataset)
    if n == 0:
        print(f"  {name}: empty, skip")
        return
    count = n if n_samples is None else min(n_samples, n)

    # Warmup
    for i in range(min(3, count)):
        data_dict, meta = dataset[i]

    for key, val in data_dict.items():
        print(f"  Key {key} has shape {val.shape}")

    # Accumulate per-key running stats over the full dataset
    sums = defaultdict(lambda: 0.0)
    sq_sums = defaultdict(lambda: 0.0)
    counts = defaultdict(lambda: 0)

    start = time.perf_counter()
    for _ in range(n_iters):
        for i in range(count):
            data_dict, meta = dataset[i]
            for key, val in data_dict.items():
                val_f = val.float()
                sums[key] += val_f.sum().item()
                sq_sums[key] += (val_f ** 2).sum().item()
                counts[key] += val_f.numel()
    elapsed = time.perf_counter() - start

    total = n_iters * count
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  {name}: {total} loads in {elapsed:.3f}s -> {rate:.1f} samples/s (len={n})")

    for key in sums:
        mean = sums[key] / counts[key]
        std = ((sq_sums[key] / counts[key]) - mean ** 2) ** 0.5
        print(f"  {name}/{key}: mean={mean:.6g}, std={std:.6g} (over {counts[key]} elements)")


def _path_ok(path) -> bool:
    """True if path looks set (not OmegaConf missing placeholder)."""
    if path is None:
        return False
    s = str(path).strip()
    return s != "" and s != "???"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print("Config (full):")
    print(OmegaConf.to_yaml(cfg))
    print()
    n_iters = 1
    n_samples = getattr(cfg, "bench_n_samples", 300)  # optional: cap samples per pass

    print("Benchmarking individual datasets:\n")

    for i, ds_cfg in enumerate(cfg.multi_dataset.datasets):
        print(f"Benchmark Dataset {i}")
        ds = hydra.utils.instantiate(ds_cfg)
        _bench_dataset("name", ds, n_iters=n_iters, n_samples=n_samples)
        ds.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
    