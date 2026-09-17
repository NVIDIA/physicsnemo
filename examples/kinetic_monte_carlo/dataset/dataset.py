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

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Literal

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from physicsnemo.diffusion.utils.utils import InfiniteSampler

# Per-sample filename produced by the user's preprocessing: sample_<sim>_<ts>.pth
_SAMPLE_RE = re.compile(r"sample_(\d+)_(\d+)\.pth")

# TensorDict keys for the universal quantities present in every application.
# User-configured feature names must not collide with these.
_RESERVED_KEYS: frozenset[str] = frozenset(
    {"particle_coords", "delay", "particle_state", "time", "mesh_coords"}
)


class ParticlesDataset(Dataset):
    """Windows of consecutive particle snapshots for a kinetic-Monte-Carlo surrogate.

    Each sample is a contiguous window of ``n_steps`` snapshots from one
    simulation, paired with that simulation's static background mesh. A
    "timestep" here is one particle-creation event, not a fixed
    wall-clock interval. Particle tensors are zero-padded to
    ``num_particles_max`` and accompanied by a binary state tensor that
    marks real (1) vs padded (0) entries.

    ``data_dir`` is expected to contain two subtrees, ``samples/`` and
    ``maps/``, each grouped into one or more *geometry* subdirectories
    (the directory names are opaque and auto-discovered; they typically
    correspond to distinct simulation geometries). All geometries are
    pooled into a single bucket of samples:

    - ``samples/<geometry>/sample_<sim>_<ts>.pth``: one file per
      particle-creation event for every simulation. The file at index
      ``i`` carries the first ``i`` particles of that simulation in
      generation order, plus the generation time of the most-recently
      created particle. The index runs from ``0`` to ``N`` (inclusive,
      where ``N`` is the total number of events for the sim); ``ts=0`` is
      the initial state.
    - ``maps/<geometry>/maps_<sim>.pth``: one file per simulation
      describing the static background mesh.

    Each per-snapshot file holds a dict with keys ``features`` of shape
    ``(i, num_particle_features)`` for the columns ``(x, y, z,
    <particle_feature_names...>, delay)`` and ``time`` (scalar). Each
    maps file holds ``positions`` of shape ``(N, 3)`` and one
    ``(N,)`` field per entry of ``mesh_feature_names``.

    Each item is a ``(TensorDict, metadata)`` pair. The TensorDict
    exposes every quantity under its own key so consumers never index
    into a packed tensor: the universal keys ``particle_coords``
    ``(S, K, 3)``, ``delay`` ``(S, K)``, ``particle_state`` ``(S, K)``,
    ``time`` ``(S,)``, and ``mesh_coords`` ``(N, 3)``, plus one
    ``(S, K)`` tensor per ``particle_feature_names`` entry and one
    ``(N,)`` tensor per ``mesh_feature_names`` entry, keyed by name.

    Parameters
    ----------
    data_dir : path
        Directory containing the ``samples`` and ``maps`` subtrees
        described above.
    particle_feature_names : list of str
        Names of the scalar particle features stored between the ``(x, y,
        z)`` coordinates and the trailing ``delay`` column, in column
        order. Used to look up per-feature normalization stats.
    mesh_feature_names : list of str
        Names of the scalar mesh fields stored alongside ``positions`` in
        each maps file, in the order they are concatenated after the
        ``(x, y, z)`` coordinates.
    num_particles_max : int
        Cap on the number of particles per snapshot (the ``K`` dimension
        of the per-particle tensors). Choose it at least as large as the
        largest particle count in the data; any snapshot exceeding it
        raises a :class:`ValueError`.
    n_steps : int, optional
        Number of consecutive snapshots per sample window. Default ``2``.
    stats_file : path or None, optional
        Path to a stats.json with per-variable z-score statistics.
        Default ``None`` (``get_stats()`` then raises).
    phase : {"train", "test", "all"}, optional
        Which split to read. ``"all"`` (default) reads ``samples`` and
        ``maps`` directly under ``data_dir``; ``"train"`` / ``"test"`` read
        them from a ``train`` / ``test`` subdirectory of ``data_dir``.

    Examples
    --------
    >>> ds = ParticlesDataset(
    ...     "data/",
    ...     particle_feature_names=["ionization_energy"],
    ...     mesh_feature_names=["temperature", "voltage"],
    ...     num_particles_max=512,
    ... )
    >>> sample, meta = ds[0]
    >>> sample["particle_coords"].shape       # (n_steps, num_particles_max, 3)
    >>> sample["ionization_energy"].shape     # (n_steps, num_particles_max)
    >>> sample["mesh_coords"].shape           # (N, 3)
    >>> sample["temperature"].shape           # (N,)
    """

    def __init__(
        self,
        data_dir: Path | str,
        particle_feature_names: list[str],
        mesh_feature_names: list[str],
        num_particles_max: int,
        n_steps: int = 2,
        stats_file: Path | str | None = None,
        phase: Literal["train", "test", "all"] = "all",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._n_steps = n_steps
        self._num_particles_max = int(num_particles_max)
        self._particle_feature_names = list(particle_feature_names)
        self._mesh_feature_names = list(mesh_feature_names)
        clashes = _RESERVED_KEYS.intersection(
            self._particle_feature_names + self._mesh_feature_names
        )
        if clashes:
            raise ValueError(
                f"Feature names {sorted(clashes)} collide with the reserved "
                f"keys {sorted(_RESERVED_KEYS)}; rename them."
            )
        # (x, y, z) + named scalars + trailing delay.
        self._num_particle_features = 3 + len(self._particle_feature_names) + 1
        # (x, y, z) + named scalar fields.
        self._num_mesh_features = 3 + len(self._mesh_feature_names)
        # "all" reads the data root directly; "train"/"test" read a same-named
        # subdirectory of it (each still holding its own samples/ and maps/).
        root = self._data_dir if phase == "all" else self._data_dir / phase
        self._samples_root = root / "samples"
        self._maps_root = root / "maps"
        if not self._samples_root.exists() or not self._maps_root.exists():
            raise FileNotFoundError(
                f"Expected 'samples' and 'maps' subdirectories under {root} "
                f"({self._samples_root}, {self._maps_root}). See the README for the "
                "expected layout."
            )

        # Discover sims by scanning the sample files. For each
        # (geometry, sim_id) we record the max ts index found, which equals
        # the number of particle-creation events for that sim.
        self._per_sim_count: dict[tuple[str, int], int] = {}
        self._scan_samples()

        # Verify a maps file exists for each discovered simulation.
        self._validate_maps()

        # Build the flat sample index. With files indexed 0..N per sim and a
        # window of n_steps, the valid ts_id range is [0, N - n_steps + 1].
        self._samples: list[tuple[str, int, int]] = []
        for (geometry, sim_id), n_events in sorted(self._per_sim_count.items()):
            max_start = n_events - n_steps + 1
            if max_start < 0:
                continue
            for ts_id in range(0, max_start + 1):
                self._samples.append((geometry, sim_id, ts_id))
        if not self._samples:
            raise RuntimeError(
                f"No samples after windowing. n_steps={n_steps} probably "
                "exceeds the per-sim event count."
            )

        # Lazy per-(geometry, sim_id) cache for the static mesh tensors.
        self._mesh_cache: dict[tuple[str, int], dict[str, Tensor]] = {}

        # Optional normalization stats.
        if stats_file is not None:
            with open(stats_file, "r") as f:
                self._stats: dict | None = json.load(f)
        else:
            self._stats = None
            warnings.warn(
                "ParticlesDataset instantiated without a stats_file; "
                "get_stats() will raise.",
                stacklevel=2,
            )

    @property
    def num_particles_max(self) -> int:
        """Cap on particles per snapshot."""
        return self._num_particles_max

    @property
    def num_particle_features(self) -> int:
        """Number of per-particle feature columns ``(x, y, z, ..., delay)``."""
        return self._num_particle_features

    @property
    def num_mesh_features(self) -> int:
        """Number of per-mesh-point feature columns ``(x, y, z, ...)``."""
        return self._num_mesh_features

    def _scan_samples(self) -> None:
        """Populate ``_per_sim_count`` by scanning every geometry subtree.

        Each simulation's ``ts`` indices must form a contiguous
        ``range(0, max_ts + 1)``. A gap is raised here, at construction, with
        the offending simulation named, rather than surfacing later as a bare
        ``FileNotFoundError`` from ``torch.load`` mid-training.
        """
        for geometry_dir in sorted(self._samples_root.iterdir()):
            if not geometry_dir.is_dir():
                continue
            geometry = geometry_dir.name
            ts_by_sim: dict[int, set[int]] = {}
            for path in geometry_dir.glob("sample_*_*.pth"):
                m = _SAMPLE_RE.fullmatch(path.name)
                if m is None:
                    continue
                sim_id = int(m.group(1))
                ts = int(m.group(2))
                ts_by_sim.setdefault(sim_id, set()).add(ts)
            for sim_id, ts_set in ts_by_sim.items():
                max_ts = max(ts_set)
                if ts_set != set(range(max_ts + 1)):
                    missing = sorted(set(range(max_ts + 1)) - ts_set)
                    raise FileNotFoundError(
                        f"{geometry}/sim_{sim_id}: sample ts indices are not "
                        f"contiguous; missing {missing} (expected 0..{max_ts})."
                    )
                self._per_sim_count[(geometry, sim_id)] = max_ts

        if not self._per_sim_count:
            raise FileNotFoundError(
                f"No sample files found under {self._samples_root}."
            )

    def _validate_maps(self) -> None:
        """Verify a maps file exists for every discovered simulation."""
        for geometry, sim_id in sorted(self._per_sim_count):
            path = self._maps_root / geometry / f"maps_{sim_id}.pth"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing maps file for {geometry}/sim_{sim_id}: expected {path}"
                )

    def _get_mesh(self, geometry: str, sim_id: int) -> dict[str, Tensor]:
        """Return the cached static mesh tensors for one simulation.

        Keys are ``"mesh_coords"`` of shape ``(N, 3)`` plus one ``(N,)``
        tensor per entry of ``mesh_feature_names``.
        """
        key = (geometry, sim_id)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached
        path = self._maps_root / geometry / f"maps_{sim_id}.pth"
        maps = torch.load(path, weights_only=False)
        mesh: dict[str, Tensor] = {
            "mesh_coords": maps["positions"].to(torch.float32)  # (N, 3)
        }
        for name in self._mesh_feature_names:
            if name not in maps:
                raise KeyError(
                    f"Mesh field {name!r} not found in {path}; available keys: "
                    f"{sorted(k for k in maps if k != 'positions')}."
                )
            mesh[name] = maps[name].to(torch.float32).reshape(-1)  # (N,)
        self._mesh_cache[key] = mesh
        return mesh

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[TensorDict, dict]:
        geometry, sim_id, ts_id = self._samples[index]
        K = self._num_particles_max
        S = self._n_steps

        particle_coords = torch.zeros(S, K, 3, dtype=torch.float32)
        delay = torch.zeros(S, K, dtype=torch.float32)
        particle_state = torch.zeros(S, K, dtype=torch.float32)
        time = torch.zeros(S, dtype=torch.float32)
        particle_features = {
            name: torch.zeros(S, K, dtype=torch.float32)
            for name in self._particle_feature_names
        }
        source_files: list[str] = []

        for step in range(S):
            ts = ts_id + step
            path = self._samples_root / geometry / f"sample_{sim_id}_{ts}.pth"
            source_files.append(str(path))
            snap = torch.load(path, weights_only=False)
            feats = snap["features"].to(torch.float32)  # (n, 3 + P + 1)
            n = feats.shape[0]
            if n > K:
                raise ValueError(
                    f"Snapshot {geometry}/sim_{sim_id}/ts_{ts} has {n} particles "
                    f"> num_particles_max={K}. Increase the cap when constructing "
                    "the dataset."
                )
            if feats.shape[1] != self._num_particle_features:
                raise ValueError(
                    f"Snapshot {geometry}/sim_{sim_id}/ts_{ts} has "
                    f"{feats.shape[1]} feature columns, but the dataset expects "
                    f"{self._num_particle_features} (= 3 coords + "
                    f"{len(self._particle_feature_names)} named features + 1 delay)."
                )
            if n > 0:
                # Unpack the documented feature columns
                #   [x, y, z, <particle_feature_names...>, delay]
                # into the granular named entries below. Particle and mesh
                # coordinates are assumed to already share the same unit.
                particle_coords[step, :n] = feats[:, 0:3]
                for j, name in enumerate(self._particle_feature_names):
                    particle_features[name][step, :n] = feats[:, 3 + j]
                delay[step, :n] = feats[:, -1]
                particle_state[step, :n] = 1.0
            time[step] = snap["time"].to(torch.float32)

        sample = TensorDict(
            {
                "particle_coords": particle_coords,  # (S, K, 3)
                "particle_state": particle_state,  # (S, K)
                "delay": delay,  # (S, K)
                "time": time,  # (S,)
                **particle_features,  # one (S, K) tensor per particle feature name
                **self._get_mesh(geometry, sim_id),  # mesh_coords (N, 3) + fields (N,)
            },
            batch_size=[],
        )
        metadata = {
            "sim_id": sim_id,
            "geometry": geometry,
            "ts_id": ts_id,
            "source_files": source_files,
        }
        return sample, metadata

    def get_stats(self, var_name: str) -> tuple[float, float]:
        """Return ``(mean, std)`` from the loaded stats.json for ``var_name``."""
        if self._stats is None:
            raise RuntimeError(
                "No stats file was provided to ParticlesDataset; "
                "pass stats_file=... when constructing the dataset."
            )
        entry = self._stats[var_name]
        return float(entry["mean"]), float(entry["std"])

    def geometries(self) -> list[str]:
        """Return the sorted geometry-group names discovered on disk."""
        return sorted({geometry for geometry, _ in self._per_sim_count})

    def get_sim_indices(self, geometry: str, sim_id: int) -> list[int]:
        """Return ordered flat sample indices for one ``(geometry, sim_id)``."""
        indices = [
            i
            for i, (g, s, _) in enumerate(self._samples)
            if g == geometry and s == sim_id
        ]
        if not indices:
            available = sorted({(g, s) for g, s, _ in self._samples})
            raise ValueError(
                f"No samples found for (geometry={geometry!r}, sim_id={sim_id}). "
                f"Dataset contains {len(available)} (geometry, sim_id) pairs."
            )
        return indices

    def get_sim_ids(self, geometry: str) -> list[int]:
        """Return the sorted ``sim_id`` values available for one geometry group."""
        sim_ids = sorted(s for g, s in self._per_sim_count if g == geometry)
        if not sim_ids:
            raise ValueError(
                f"No simulations found for geometry={geometry!r}. "
                f"Available geometries: {self.geometries()}"
            )
        return sim_ids


def _collate(
    batch: list[tuple[TensorDict, dict]],
) -> tuple[TensorDict, list[dict]]:
    """Collate a list of ``(TensorDict, metadata)`` samples into a batched TD."""
    tds = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    return torch.stack(tds, dim=0), metas


class ParticlesDataPipe(DataLoader):
    """DDP-aware infinite DataLoader for particle-snapshot windows.

    Wraps :class:`ParticlesDataset` with an :class:`InfiniteSampler` so
    training is sample-count based rather than epoch based. ``start_idx``
    resumes from an arbitrary position after a checkpoint reload.
    ``get_stats()`` proxies to the inner dataset.

    Parameters
    ----------
    data_dir : path
        Directory containing the ``samples`` and ``maps`` subtrees.
    batch_size_per_device : int
        Per-rank batch size. The effective global batch is
        ``batch_size_per_device * world_size``.
    particle_feature_names : list of str
        Forwarded to :class:`ParticlesDataset`.
    mesh_feature_names : list of str
        Forwarded to :class:`ParticlesDataset`.
    num_particles_max : int
        Forwarded to :class:`ParticlesDataset`.
    n_steps : int, optional
        Number of consecutive snapshots per sample window. Default ``2``.
    stats_file : path or None, optional
        Path to a stats.json. Default ``None``.
    phase : {"train", "test", "all"}, optional
        Forwarded to :class:`ParticlesDataset`. Default ``"all"``.
    shuffle : bool, optional
        Whether the sampler shuffles indices. Default ``True``.
    num_workers : int, optional
        Number of DataLoader worker processes. Default ``4``.
    prefetch_factor : int, optional
        Samples loaded in advance per worker (used when ``num_workers > 0``).
        Default ``4``.
    process_rank : int, optional
        Rank of this process in the DDP group. Default ``0``.
    world_size : int, optional
        Total number of DDP ranks. Default ``1``.
    start_idx : int, optional
        Sample offset to resume from after a checkpoint reload. Default ``0``.
    seed : int, optional
        Seed for the :class:`InfiniteSampler` shuffle. Default ``0``.
    """

    def __init__(
        self,
        data_dir: Path | str,
        batch_size_per_device: int,
        particle_feature_names: list[str],
        mesh_feature_names: list[str],
        num_particles_max: int,
        n_steps: int = 2,
        stats_file: Path | str | None = None,
        phase: Literal["train", "test", "all"] = "all",
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        process_rank: int = 0,
        world_size: int = 1,
        start_idx: int = 0,
        seed: int = 0,
    ) -> None:
        dataset = ParticlesDataset(
            data_dir=data_dir,
            particle_feature_names=particle_feature_names,
            mesh_feature_names=mesh_feature_names,
            n_steps=n_steps,
            num_particles_max=num_particles_max,
            stats_file=stats_file,
            phase=phase,
        )
        sampler = InfiniteSampler(
            dataset=dataset,
            rank=process_rank,
            num_replicas=world_size,
            shuffle=shuffle,
            seed=seed,
            start_idx=start_idx,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size_per_device,
            sampler=sampler,
            collate_fn=_collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=0,
            persistent_workers=False,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(**loader_kwargs)

    def get_stats(self, var_name: str) -> tuple[float, float]:
        """Proxy to :meth:`ParticlesDataset.get_stats`."""
        return self.dataset.get_stats(var_name)
