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

r"""PyTorch dataset for the SuperWing surface-flow dataset.

Layout assumptions (mirrors the
`yunplus/SuperWing <https://huggingface.co/datasets/yunplus/SuperWing>`_
release):

* ``geom0.npy``       : ``(Nshape, 3, 128, 256)`` cell-centric mesh.
* ``origingeom.npy``  : ``(Nshape, 3, 129, 257)`` grid-point mesh
  (used for force / moment integration in the post-processing script).
* ``data.npy``        : ``(N, 3, 128, 256)`` surface fields
  (``Cp``, ``Cf_tau``, ``Cf_z``), cell-centric.
* ``index.npy``       : ``(N, 12)`` per-sample metadata: ``geom_idx``,
  ``cond_idx``, ``aoa``, ``mach``, ``ref_area``, ``half_span``,
  ``cl_solver``, ``cd_solver``, ``cm_solver``,
  ``cl_surface``, ``cd_surface``, ``cm_surface``.
* ``configs.dat``     : ``(Nshape, ~57)`` per-geometry design
  parameters (reserved for latent-space analysis; not a model input).

The dataset produces the keys the AeroJEPA trunk consumes —
``context_pos`` / ``context_feat`` for the geometry encoder,
``target_surface_pos`` / ``target_surface_main_feat`` for the target
encoder's surface half, empty ``target_volume_*`` tensors (SuperWing
is surface-only), ``query_pos`` / ``query_sdf`` / ``query_target`` for
the decoder and loss, and ``gen_params`` for the operating conditions.
The context and target subsamples are drawn independently from the
surface grid — the JEPA predictor learns to map between two views of
the same wing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


SUPERWING_TARGET_CHANNELS: tuple[str, ...] = ("Cp", "Cf_tau", "Cf_z")
SUPERWING_GRID_SHAPE: tuple[int, int] = (128, 256)
SUPERWING_ORIGIN_GRID_SHAPE: tuple[int, int] = (129, 257)
SUPERWING_INDEX_NUM_COLS: int = 12


@dataclass(frozen=True)
class SuperWingPaths:
    """File-path helper for a SuperWing root directory."""

    root_dir: str

    @property
    def index(self) -> str:
        """Absolute path to ``index.npy``."""
        return os.path.join(self.root_dir, "index.npy")

    @property
    def geom0(self) -> str:
        """Absolute path to ``geom0.npy`` (cell-centric mesh)."""
        return os.path.join(self.root_dir, "geom0.npy")

    @property
    def origingeom(self) -> str:
        """Absolute path to ``origingeom.npy`` (grid-point mesh)."""
        return os.path.join(self.root_dir, "origingeom.npy")

    @property
    def data(self) -> str:
        """Absolute path to ``data.npy`` (surface flow field tensor)."""
        return os.path.join(self.root_dir, "data.npy")


def _load_split_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_normalization_stats(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    if str(stats.get("schema", "")).lower() != "superwing":
        raise ValueError(
            f"Normalization stats at {path} are not for SuperWing "
            f"(schema={stats.get('schema')!r})."
        )
    return stats


class SuperWingDataset(Dataset):
    r"""SuperWing surface-flow dataset for the AeroJEPA recipe.

    Each ``__getitem__`` returns a dict of tensors and lightweight
    metadata. The shapes match the AeroJEPA trunk's input contract:

    * ``context_pos``               ``(N_ctx, 3)`` — context-encoder
      surface points (independent subsample from the target).
    * ``context_feat``              ``(N_ctx, 0)`` — empty auxiliary
      features (SDF / normals / n-dot-uinf are disabled in the
      tutorial config).
    * ``target_surface_pos``        ``(N_tgt, 3)`` — target-encoder
      surface points (independent subsample from the context).
    * ``target_surface_main_feat``  ``(N_tgt, 6)`` — ``xyz``
      concatenated with the per-point normalised surface field at the
      target subsample.
    * ``target_volume_pos``         ``(0, 3)`` — SuperWing has no
      volume.
    * ``target_volume_feat``        ``(0, 6)`` — matching
      target_surface_main_feat channel count.
    * ``query_pos``          ``(N_q, 3)`` — decoder query points
      (independently sampled at train; full ``128 * 256`` grid at eval
      when ``eval_full_grid_query=True``).
    * ``query_sdf``          ``(N_q, 1)`` — zeros (SuperWing has no SDF).
    * ``query_target``       ``(N_q, 3)`` — normalised
      ``(Cp, Cf_tau, Cf_z)`` used for the reconstruction loss.
    * ``gen_params``         ``(G,)``     — standardised
      ``(aoa, mach)`` by default.
    * Metadata: ``case_id``, ``aoa_deg``, ``mach``, ``ref_area``,
      ``half_span``, ``solver_coeffs``, ``surface_coeffs``,
      ``geom_idx``, ``cond_idx``, ``sample_idx``.
    * Optional: ``target_full`` and ``xyz_full`` when
      ``return_full_fields=True``; ``origingeom_full`` when
      ``return_origingeom=True``.

    Parameters
    ----------
    root_dir : str
        SuperWing dataset root (containing ``index.npy``, ``geom0.npy``,
        ``data.npy``, optionally ``origingeom.npy``).
    split : {"train", "val", "test", "all"}
        Which partition of the split manifest to expose.
    split_manifest_path : str
        Path to the manifest produced by
        :func:`build_superwing_split_manifest`.
    normalization_stats : dict, optional
        Pre-loaded stats dict. If ``None``, ``normalization_stats_path``
        must be provided.
    normalization_stats_path : str, optional
        Path to the JSON produced by
        :func:`compute_superwing_normalization_stats`.
    surface_points : int, optional
        Number of context-encoder surface points per sample. Default
        ``8192``.
    target_encoder_points : int or None, optional
        Number of target-encoder surface points per sample. Falls back
        to ``surface_points`` when ``None``. Drawn independently of the
        context subsample. Default ``None``.
    query_points : int, optional
        Number of decoder query points per sample at training. Default
        ``8192``.
    eval_full_grid_query : bool, optional
        If True, evaluation splits use the full ``128 * 256`` grid as
        the query set instead of subsampling. Default ``True``.
    return_origingeom : bool, optional
        Include the vertex-grid mesh (``origingeom.npy``) in the
        sample. Required by the CL / CD post-processing script.
        Default ``False``.
    return_full_fields : bool, optional
        Include the unnormalised ``(3, 128, 256)`` flow field and the
        full xyz mesh in the sample. Useful for visualisation.
        Default ``False``.
    deterministic_sampling : bool, optional
        If True, seed the per-sample RNG by ``sample_idx`` so subsamples
        are reproducible across epochs. Default ``True``.
    normalize_xyz : bool, optional
        Min-max normalise xyz into ``[-1, 1]`` using the stats. Default
        ``True``.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        split: str,
        split_manifest_path: str,
        normalization_stats: dict | None = None,
        normalization_stats_path: str | None = None,
        surface_points: int = 8192,
        target_encoder_points: int | None = None,
        query_points: int = 8192,
        eval_full_grid_query: bool = True,
        return_origingeom: bool = False,
        return_full_fields: bool = False,
        deterministic_sampling: bool = True,
        normalize_xyz: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(
                f"split must be one of: train/val/test/all (got {split!r})"
            )
        self.split = split
        self.paths = SuperWingPaths(root_dir=root_dir)
        for p in (self.paths.index, self.paths.geom0, self.paths.data):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing SuperWing file: {p}")

        manifest = _load_split_manifest(split_manifest_path)
        if split == "all":
            self.sample_indices: list[int] = sorted(
                int(v)
                for v in (
                    list(manifest.get("train_sample_idx", []))
                    + list(manifest.get("val_sample_idx", []))
                    + list(manifest.get("test_sample_idx", []))
                )
            )
        else:
            key = f"{split}_sample_idx"
            if key not in manifest:
                raise KeyError(f"Split manifest missing key '{key}'")
            self.sample_indices = [int(v) for v in manifest[key]]
        if not self.sample_indices:
            raise ValueError(
                f"Split '{split}' is empty in manifest {split_manifest_path}"
            )

        if normalization_stats is None:
            if normalization_stats_path is None:
                raise ValueError(
                    "SuperWingDataset requires either normalization_stats "
                    "or normalization_stats_path."
                )
            normalization_stats = _load_normalization_stats(normalization_stats_path)
        self.normalization_stats = normalization_stats

        self.target_mean = np.asarray(
            normalization_stats["target_mean"], dtype=np.float32
        )
        self.target_std = np.asarray(
            normalization_stats["target_std"], dtype=np.float32
        )
        self.gen_param_columns = np.asarray(
            normalization_stats["gen_params_columns"], dtype=np.int64
        )
        self.gen_param_names = list(normalization_stats["gen_params_names"])
        self.gen_param_mean = np.asarray(
            normalization_stats["gen_params_mean"], dtype=np.float32
        )
        self.gen_param_std = np.asarray(
            normalization_stats["gen_params_std"], dtype=np.float32
        )
        self.xyz_min = np.asarray(normalization_stats["xyz_min"], dtype=np.float32)
        self.xyz_max = np.asarray(normalization_stats["xyz_max"], dtype=np.float32)

        self.surface_points = int(surface_points)
        self.target_encoder_points = (
            int(target_encoder_points)
            if target_encoder_points is not None
            else int(surface_points)
        )
        self.query_points = int(query_points)
        self.eval_full_grid_query = bool(eval_full_grid_query)
        self.return_origingeom = bool(return_origingeom)
        self.return_full_fields = bool(return_full_fields)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.normalize_xyz = bool(normalize_xyz)

        # Memmap handles are created lazily so DataLoader workers fork
        # without inheriting open file descriptors.
        self._geom_mm: np.memmap | None = None
        self._origin_mm: np.memmap | None = None
        self._data_mm: np.memmap | None = None
        self._index: np.ndarray | None = None
        self._n_grid_points = SUPERWING_GRID_SHAPE[0] * SUPERWING_GRID_SHAPE[1]

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = np.load(self.paths.index)
            if int(self._index.shape[1]) != SUPERWING_INDEX_NUM_COLS:
                raise ValueError(
                    f"Unexpected index.npy shape {tuple(self._index.shape)}; "
                    f"expected (*, {SUPERWING_INDEX_NUM_COLS})."
                )
        if self._geom_mm is None:
            self._geom_mm = np.load(self.paths.geom0, mmap_mode="r")
        if self._data_mm is None:
            self._data_mm = np.load(self.paths.data, mmap_mode="r")
        if self.return_origingeom and self._origin_mm is None:
            if not os.path.exists(self.paths.origingeom):
                raise FileNotFoundError(
                    f"origingeom.npy missing: {self.paths.origingeom}"
                )
            self._origin_mm = np.load(self.paths.origingeom, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _make_rng(self, sample_idx: int) -> np.random.Generator:
        if self.deterministic_sampling:
            seed = (int(sample_idx) * 2654435761) & 0xFFFFFFFF
            return np.random.default_rng(seed)
        return np.random.default_rng()

    def _sample_indices(self, rng: np.random.Generator, target_n: int) -> np.ndarray:
        n_total = self._n_grid_points
        if target_n >= n_total:
            return np.arange(n_total, dtype=np.int64)
        return rng.choice(n_total, size=int(target_n), replace=False).astype(np.int64)

    def _normalize_xyz(self, xyz: np.ndarray) -> np.ndarray:
        if not self.normalize_xyz:
            return xyz.astype(np.float32, copy=False)
        denom = np.maximum(self.xyz_max - self.xyz_min, 1e-12)
        out = 2.0 * (xyz - self.xyz_min[None, :]) / denom[None, :] - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        out = (target - self.target_mean[None, :]) / np.maximum(
            self.target_std[None, :], 1e-12
        )
        return out.astype(np.float32, copy=False)

    def _normalize_gen_params(self, raw_gp: np.ndarray) -> np.ndarray:
        out = (raw_gp - self.gen_param_mean) / np.maximum(self.gen_param_std, 1e-12)
        return out.astype(np.float32, copy=False)

    def __getitem__(self, dataset_idx: int) -> dict[str, object]:
        self._ensure_loaded()
        assert self._index is not None
        assert self._geom_mm is not None
        assert self._data_mm is not None

        sample_idx = int(self.sample_indices[int(dataset_idx)])
        idx_row = self._index[sample_idx]
        geom_idx = int(idx_row[0])
        cond_idx = int(idx_row[1])

        # Geometry (cell-centric grid).
        geom_chw = np.asarray(self._geom_mm[geom_idx], dtype=np.float32)
        xyz_full = geom_chw.reshape(3, -1).T
        xyz_n_full = self._normalize_xyz(xyz_full)

        # Surface field.
        data_chw = np.asarray(self._data_mm[sample_idx], dtype=np.float32)
        target_flat = data_chw.reshape(3, -1).T
        target_n_full = self._normalize_target(target_flat)

        # Operating conditions.
        raw_gp = idx_row[self.gen_param_columns].astype(np.float32)
        gen_params = self._normalize_gen_params(raw_gp)

        rng = self._make_rng(sample_idx)
        # Context and target subsamples are drawn independently from the
        # surface grid (the JEPA paradigm: two views of the same wing).
        s_idx = self._sample_indices(rng, self.surface_points)
        t_idx = self._sample_indices(rng, self.target_encoder_points)

        use_full_grid = self.eval_full_grid_query and self.split != "train"
        q_idx = (
            np.arange(self._n_grid_points, dtype=np.int64)
            if use_full_grid
            else self._sample_indices(rng, self.query_points)
        )

        n_ctx = int(s_idx.shape[0])
        n_q = int(q_idx.shape[0])

        context_pos = xyz_n_full[s_idx]
        context_feat = np.zeros((n_ctx, 0), dtype=np.float32)
        target_surface_pos = xyz_n_full[t_idx]
        target_surface_main_feat = np.concatenate(
            [target_surface_pos, target_n_full[t_idx]], axis=-1
        )

        query_pos = xyz_n_full[q_idx]
        query_target = target_n_full[q_idx]

        empty_target_volume_pos = np.zeros((0, 3), dtype=np.float32)
        empty_target_volume_feat = np.zeros(
            (0, target_surface_main_feat.shape[-1]), dtype=np.float32
        )

        out: dict[str, object] = {
            "gen_params": torch.from_numpy(gen_params),
            "context_pos": torch.from_numpy(context_pos),
            "context_feat": torch.from_numpy(context_feat),
            "target_surface_pos": torch.from_numpy(target_surface_pos),
            "target_surface_main_feat": torch.from_numpy(target_surface_main_feat),
            "target_volume_pos": torch.from_numpy(empty_target_volume_pos),
            "target_volume_feat": torch.from_numpy(empty_target_volume_feat),
            "query_pos": torch.from_numpy(query_pos),
            "query_target": torch.from_numpy(query_target),
            "query_sdf": torch.zeros((n_q, 1), dtype=torch.float32),
            # SuperWing-specific metadata used by the postprocessing and
            # visualisation scripts.
            "case_id": f"geo{geom_idx:04d}_cond{cond_idx:d}_s{sample_idx:06d}",
            "geom_idx": int(geom_idx),
            "cond_idx": int(cond_idx),
            "sample_idx": int(sample_idx),
            "aoa_deg": float(idx_row[2]),
            "mach": float(idx_row[3]),
            "ref_area": float(idx_row[4]),
            "half_span": float(idx_row[5]),
            "solver_coeffs": torch.tensor(
                [float(idx_row[6]), float(idx_row[7]), float(idx_row[8])],
                dtype=torch.float32,
            ),
            "surface_coeffs": torch.tensor(
                [float(idx_row[9]), float(idx_row[10]), float(idx_row[11])],
                dtype=torch.float32,
            ),
        }
        if self.return_full_fields:
            out["target_full"] = torch.from_numpy(data_chw)
            out["target_full_normalized"] = torch.from_numpy(
                target_n_full.T.reshape(3, *SUPERWING_GRID_SHAPE).astype(np.float32)
            )
            out["xyz_full"] = torch.from_numpy(xyz_n_full)
            out["xyz_full_raw"] = torch.from_numpy(xyz_full)
        if self.return_origingeom:
            assert self._origin_mm is not None
            out["origingeom_full"] = torch.from_numpy(
                np.asarray(self._origin_mm[geom_idx], dtype=np.float32)
            )
        return out


_VARIABLE_KEYS: frozenset[str] = frozenset(
    {
        "context_pos",
        "context_feat",
        "target_surface_pos",
        "target_surface_main_feat",
        "target_volume_pos",
        "target_volume_feat",
        "query_pos",
        "query_sdf",
        "query_target",
    }
)


def superwing_collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    r"""Collate that stacks uniform tensors and pads variable-length ones.

    Variable-length keys (``context_*``, ``target_surface_*``,
    ``target_volume_*``, ``query_*``) get padded with zeros along dim 0
    to the batch's maximum and accompanied by a ``<key>_n`` tensor of
    valid counts — the format the AeroJEPA trunk's batched forward
    consumes. Non-tensor metadata (case ids, indices, floats) is
    collected into plain Python lists.

    Parameters
    ----------
    batch : Sequence[dict]
        Samples produced by :class:`SuperWingDataset`.

    Returns
    -------
    dict
        Collated batch.
    """
    out: dict[str, object] = {}
    keys = batch[0].keys()
    for k in keys:
        values = [sample[k] for sample in batch]
        if torch.is_tensor(values[0]):
            if k in _VARIABLE_KEYS:
                lengths = [int(v.shape[0]) for v in values]
                max_len = max(lengths)
                if all(length == max_len for length in lengths):
                    out[k] = torch.stack(values, dim=0)
                else:
                    padded = []
                    for v in values:
                        pad_n = max_len - int(v.shape[0])
                        if pad_n > 0:
                            pad_shape = (pad_n,) + tuple(v.shape[1:])
                            pad = torch.zeros(pad_shape, dtype=v.dtype)
                            v = torch.cat([v, pad], dim=0)
                        padded.append(v)
                    out[k] = torch.stack(padded, dim=0)
                out[f"{k}_n"] = torch.tensor(lengths, dtype=torch.long)
            else:
                out[k] = torch.stack(values, dim=0)
        else:
            out[k] = values
    return out
