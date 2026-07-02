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

"""Synthetic thermal data used by the FP-DDM example."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

_SIMPLEX_WORKERS = 20


def _smooth(array: np.ndarray, sigma: float, *, mode: str = "reflect") -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(array, sigma=sigma, mode=mode)
    except ImportError:
        max_kernel = min(array.shape)
        if max_kernel % 2 == 0:
            max_kernel -= 1
        kernel = min(max(3, 2 * round(sigma) + 1), max_kernel)
        if kernel < 3:
            return array.astype(np.float32, copy=True)
        padding = kernel // 2
        values = torch.from_numpy(array.astype(np.float32))[None, None]
        padding_mode = "circular" if mode == "wrap" else "reflect"
        values = F.pad(values, (padding,) * 4, mode=padding_mode)
        return F.avg_pool2d(values, kernel, stride=1)[0, 0].numpy()


def _normalize_to_range(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    a_min = float(array.min())
    a_max = float(array.max())
    if a_max == a_min:
        return np.full_like(array, 0.5 * (vmin + vmax), dtype=np.float32)
    unit = (array - a_min) / (a_max - a_min)
    return (vmin + (vmax - vmin) * unit).astype(np.float32)


def _smooth_random_map(size: int, vmin: float, vmax: float, sigma: float) -> np.ndarray:
    values = np.random.uniform(vmin, vmax, (size, size))
    return _smooth(values, sigma)


def _conductivity_map(
    size: int,
    vmin: float,
    vmax: float,
    method: str = "pixelwise_smooth",
    sigma: float = 6.0,
) -> np.ndarray:
    if method == "uniform":
        values = np.full((size, size), np.random.uniform(vmin, vmax), dtype=np.float32)
        return values[None]
    elif method == "pixelwise":
        values = np.random.uniform(vmin, vmax, (size, size))
    elif method == "pixelwise_smooth":
        values = _smooth_random_map(size, vmin, vmax, sigma)
    else:
        raise ValueError(f"Unknown conductivity generation method: {method}")
    return _normalize_to_range(values, vmin, vmax)[None]


def _boundary_temperature_map(
    size: int,
    vmin: float,
    vmax: float,
    method: str = "pixelwise_smooth",
    sigma: float = 6.0,
) -> np.ndarray:
    if method == "uniform":
        values = np.zeros((size, size), dtype=np.float32)
        values[0], values[-1], values[:, 0], values[:, -1] = np.random.uniform(
            vmin, vmax, 4
        )
    elif method == "pixelwise":
        values = np.random.uniform(vmin, vmax, (size, size))
    elif method == "pixelwise_smooth":
        values = _smooth_random_map(size, vmin, vmax, sigma)
        edge_range = (vmax - vmin) / 4.0
        values[0] += np.random.uniform(-edge_range, edge_range)
        values[-1] += np.random.uniform(-edge_range, edge_range)
        values[:, 0] += np.random.uniform(-edge_range, edge_range)
        values[:, -1] += np.random.uniform(-edge_range, edge_range)
    else:
        raise ValueError(f"Unknown boundary generation method: {method}")

    values = _normalize_to_range(values, vmin, vmax)
    values[1:-1, 1:-1] = 0.0
    return values[None]


def _heat_source_map(size: int, vmin: float, vmax: float, sigma: float) -> np.ndarray:
    if vmin == vmax:
        values = np.full((size, size), vmin, dtype=np.float32)
    else:
        values = _smooth_random_map(size, vmin, vmax, sigma)
        values = _normalize_to_range(values, vmin, vmax)
    return values[None]


class ThermalDataset(Dataset):
    """Generate local thermal problems for physics-informed FNO training.

    Each sample contains five channels: normalized-grid coordinates ``x`` and
    ``y``, conductivity, Dirichlet temperature values, and volumetric heat
    source. Temperature values are nonzero only on the patch boundary.
    """

    def __init__(self, config: Mapping[str, object]):
        """Initialize the synthetic dataset from a mapping-like config."""

        self.n_samples = int(config.get("n_samples", 0))
        self.grid_size = int(config.get("grid_size", 32))
        self.k_min = float(config.get("k_min", 1.0))
        self.k_max = float(config.get("k_max", 100.0))
        self.q_min = float(config.get("q_min", 0.0))
        self.q_max = float(config.get("q_max", 0.0))
        self.T_min = float(config.get("T_min", 300.0))
        self.T_max = float(config.get("T_max", 400.0))
        self.sigma = float(config.get("sigma", 10.0))
        self.k_method = str(config.get("k_generation_method", "pixelwise_smooth"))
        self.boundary_method = str(
            config.get("boundary_generation_method", "pixelwise_smooth")
        )
        self.use_on_the_fly = bool(config.get("use_on_the_fly", True))

        axis = np.linspace(0.0, 1.0, self.grid_size, dtype=np.float32)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        self.coords = torch.from_numpy(np.stack([xx, yy], axis=0))

        self.samples = None
        if not self.use_on_the_fly:
            self.samples = [self.generate_sample() for _ in range(self.n_samples)]

    def generate_sample(self) -> torch.Tensor:
        """Generate one five-channel local thermal problem."""

        conductivity = _conductivity_map(
            self.grid_size,
            self.k_min,
            self.k_max,
            self.k_method,
            self.sigma,
        )
        temperature_bc = _boundary_temperature_map(
            self.grid_size,
            self.T_min,
            self.T_max,
            self.boundary_method,
            self.sigma,
        )
        heat_source = _heat_source_map(
            self.grid_size, self.q_min, self.q_max, self.sigma
        )
        fields = torch.from_numpy(
            np.concatenate([conductivity, temperature_bc, heat_source], axis=0)
        )
        return torch.cat([self.coords, fields], dim=0).float()

    def __len__(self) -> int:
        """Return the configured number of samples."""

        return self.n_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a cached sample or generate one on demand."""

        if self.samples is None:
            return self.generate_sample()
        return self.samples[index]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def create_dataloaders(
    config: Mapping[str, object],
    batch_size: int,
    *,
    seed: int = 1234,
    num_workers: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic train, validation, and test data loaders."""

    total = int(config.get("n_samples", 0))
    if total < 3:
        raise ValueError("FP-DDM training requires at least three samples")
    n_val = max(1, int(0.05 * total))
    n_test = max(1, int(0.05 * total))
    n_train = total - n_val - n_test

    train_config = copy.deepcopy(dict(config))
    train_config.update(n_samples=n_train, use_on_the_fly=True)
    validation_config = copy.deepcopy(dict(config))
    validation_config.update(n_samples=n_val + n_test, use_on_the_fly=False)

    train_dataset = ThermalDataset(train_config)
    validation_and_test = ThermalDataset(validation_config)
    validation_dataset, test_dataset = random_split(
        validation_and_test,
        [n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 10)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }

    def make_loader(dataset, shuffle: bool) -> DataLoader:
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            )
        return DataLoader(
            dataset,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            **loader_kwargs,
        )

    return (
        make_loader(train_dataset, True),
        make_loader(validation_dataset, False),
        make_loader(test_dataset, False),
    )


def _fallback_layout_map(seed: int, size: int, octaves: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1337)
    result = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(max(1, octaves)):
        noise = rng.random((size, size), dtype=np.float32)
        sigma = max(0.5, size / (8.0 * 2**octave))
        result += amplitude * _smooth(noise, sigma, mode="wrap")
        total_amplitude += amplitude
        amplitude *= 0.5
    return result / total_amplitude


def _simplex_rows(args) -> tuple[int, np.ndarray]:
    from opensimplex import OpenSimplex

    seed, size, y0, y1, octaves, lacunarity, gain, warp_strength = args
    simplex = OpenSimplex(seed=seed + 1337)
    noise = simplex.noise2
    base_scale = max(8.0, size / 4.0)
    warp_scale = max(8.0, size / 8.0)
    block = np.zeros((y1 - y0, size), dtype=np.float32)
    for yi, y in enumerate(range(y0, y1)):
        for x in range(size):
            wx = noise(x / warp_scale, y / warp_scale)
            wy = noise((x + 1000) / warp_scale, (y + 1000) / warp_scale)
            xx = x / base_scale + warp_strength * wx
            yy = y / base_scale + warp_strength * wy
            amplitude, frequency, value = 1.0, 1.0, 0.0
            for _ in range(octaves):
                value += amplitude * (1.0 - abs(noise(xx * frequency, yy * frequency)))
                amplitude *= gain
                frequency *= lacunarity
            block[yi, x] = value
    return y0, block


def _layout_map(
    seed: int,
    size: int,
    octaves: int = 5,
    lacunarity: float = 2.5,
    gain: float = 0.3,
    warp_strength: float = 0.5,
) -> np.ndarray:
    try:
        from opensimplex import OpenSimplex
    except ImportError:
        return _normalize_to_range(_fallback_layout_map(seed, size, octaves), 0.0, 1.0)

    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    workers = min(_SIMPLEX_WORKERS, available, max(1, size // 8))
    bounds = np.linspace(0, size, workers + 1).astype(int)
    tasks = [
        (seed, size, bounds[i], bounds[i + 1], octaves, lacunarity, gain, warp_strength)
        for i in range(workers)
        if bounds[i] < bounds[i + 1]
    ]

    result = np.zeros((size, size), dtype=np.float32)
    if len(tasks) == 1:
        blocks = map(_simplex_rows, tasks)
    else:
        OpenSimplex(seed=seed + 1337).noise2(0.1, 0.2)
        with mp.Pool(len(tasks)) as pool:
            blocks = pool.map(_simplex_rows, tasks)
    for y0, block in blocks:
        result[y0 : y0 + block.shape[0]] = block
    return _normalize_to_range(result, 0.0, 1.0)


def make_layout_fields(
    config: Mapping[str, object],
    *,
    seed_k: int = 0,
    seed_q: int = 1,
    seed_temperature: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate conductivity, heat-source, and temperature layout fields."""

    size = int(config["grid_size"])
    conductivity = _normalize_to_range(
        _layout_map(seed_k, size), float(config["k_min"]), float(config["k_max"])
    )
    heat_source = _normalize_to_range(
        _layout_map(seed_q, size), float(config["q_min"]), float(config["q_max"])
    )
    temperature = _normalize_to_range(
        _layout_map(seed_temperature, size),
        float(config["T_min"]),
        float(config["T_max"]),
    )
    return tuple(
        torch.from_numpy(array).float()
        for array in (conductivity, heat_source, temperature)
    )
