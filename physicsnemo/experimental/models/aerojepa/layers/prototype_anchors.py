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

"""Prototype anchor building and loading for AeroJEPA.

The ``data_prototype_cluster`` tokenization strategy
(:class:`physicsnemo.experimental.models.aerojepa.layers.point_tokenizer.PointCloudTokenizer`)
uses a fixed set of 3D anchor points as token centers. This module builds
those anchors by sampling token coordinates from a training dataset (via
the same tokenizer) and running a chunked k-means to compress them to a
target count, then serializing the result to disk for later reload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .point_tokenizer import PointCloudTokenizer


def _concat_target_points(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    surface_pos = sample.get("target_surface_pos", sample["surface_pos"])
    surface_feat = sample.get(
        "target_surface_main_feat",
        sample["surface_main_feat"]
        if "surface_main_feat" in sample
        else sample["surface_feat"],
    )
    volume_pos = sample.get("target_volume_pos", sample["volume_pos"])
    volume_feat = sample.get("target_volume_feat", sample["volume_feat"])
    if int(surface_feat.shape[-1]) != int(volume_feat.shape[-1]):
        raise ValueError(
            "surface_main_feat and volume_feat must have matching feature dimensions."
        )
    positions = torch.cat([surface_pos, volume_pos], dim=0).detach().cpu().float()
    features = torch.cat([surface_feat, volume_feat], dim=0).detach().cpu().float()
    return positions, features


def _build_context_surface_features(
    *,
    surface_pos: torch.Tensor,
    surface_feat: torch.Tensor,
    use_sdf: bool = False,
    use_solid_normals: bool = False,
    use_solid_n_dot_uinf: bool = False,
) -> torch.Tensor:
    if not use_sdf and not use_solid_normals and not use_solid_n_dot_uinf:
        return surface_pos

    extra_needed = (
        (1 if use_sdf else 0)
        + (3 if use_solid_normals else 0)
        + (1 if use_solid_n_dot_uinf else 0)
    )
    if int(surface_feat.shape[-1]) < int(surface_pos.shape[-1]) + extra_needed:
        raise ValueError(
            "surface_feat does not contain enough channels for configured "
            "context prototype features."
        )

    parts = [surface_pos]
    if use_sdf:
        sdf_offset_from_end = (
            (3 if use_solid_normals else 0)
            + (1 if use_solid_n_dot_uinf else 0)
            + 1
        )
        sdf_start = -sdf_offset_from_end
        sdf_end = None if sdf_offset_from_end == 1 else -sdf_offset_from_end + 1
        parts.append(surface_feat[..., sdf_start:sdf_end])
    if use_solid_normals:
        start = int(surface_feat.shape[-1]) - (4 if use_solid_n_dot_uinf else 3)
        parts.append(surface_feat[..., start : start + 3])
    if use_solid_n_dot_uinf:
        parts.append(surface_feat[..., -1:])
    return torch.cat(parts, dim=-1)


def _concat_context_points(
    sample: dict[str, Any], context_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = sample["surface_pos"].detach().cpu().float()
    features = (
        _build_context_surface_features(
            surface_pos=sample["surface_pos"],
            surface_feat=sample["surface_feat"],
            use_sdf=bool(context_cfg.get("use_sdf", False)),
            use_solid_normals=bool(context_cfg.get("use_solid_normals", False)),
            use_solid_n_dot_uinf=bool(context_cfg.get("use_solid_n_dot_uinf", False)),
        )
        .detach()
        .cpu()
        .float()
    )
    return positions, features


def _farthest_point_sampling(
    points: torch.Tensor, num_samples: int, *, seed: int
) -> torch.Tensor:
    n = int(points.shape[0])
    if num_samples >= n:
        return torch.arange(n, dtype=torch.long)
    gen = torch.Generator(device=points.device)
    gen.manual_seed(int(seed))
    selected = torch.empty((num_samples,), device=points.device, dtype=torch.long)
    current = int(
        torch.randint(0, n, (1,), generator=gen, device=points.device).item()
    )
    selected[0] = current
    min_dist_sq = torch.full(
        (n,), float("inf"), device=points.device, dtype=points.dtype
    )
    for i in range(1, num_samples):
        ref = points[current : current + 1]
        dist_sq = torch.sum((points - ref) ** 2, dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        current = int(torch.argmax(min_dist_sq).item())
        selected[i] = current
    return selected


def _assign_points(
    points: torch.Tensor,
    centers: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assign_chunks = []
    dist_chunks = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        chunk = points[start : start + int(chunk_size)]
        dists = torch.cdist(chunk, centers)
        min_dist, assign = torch.min(dists, dim=1)
        assign_chunks.append(assign)
        dist_chunks.append(min_dist)
    return torch.cat(assign_chunks, dim=0), torch.cat(dist_chunks, dim=0)


def _run_chunked_kmeans(
    points: torch.Tensor,
    *,
    num_clusters: int,
    num_iters: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    if int(points.shape[0]) <= int(num_clusters):
        return points.clone()
    center_idx = _farthest_point_sampling(points, int(num_clusters), seed=seed)
    centers = points[center_idx].clone()
    for _ in range(max(1, int(num_iters))):
        assign, _ = _assign_points(points, centers, chunk_size=chunk_size)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(
            (int(num_clusters), 1), dtype=points.dtype, device=points.device
        )
        new_centers.index_add_(0, assign, points)
        counts.index_add_(
            0,
            assign,
            torch.ones(
                (int(points.shape[0]), 1), dtype=points.dtype, device=points.device
            ),
        )
        valid = counts.squeeze(-1) > 0
        new_centers[valid] = new_centers[valid] / counts[valid].clamp_min(1.0)
        if (~valid).any():
            refill_idx = _farthest_point_sampling(
                points, int((~valid).sum().item()), seed=seed + 17
            )
            new_centers[~valid] = points[refill_idx]
        centers = new_centers
    return centers


def load_target_prototype_anchors(path: str | Path) -> torch.Tensor:
    r"""Load prototype anchor coordinates from a saved ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file produced by
        :func:`build_target_prototype_anchors`.

    Returns
    -------
    torch.Tensor
        Anchor coordinates of shape ``(P, 3)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If the file's ``coords`` array does not have shape ``(P, 3)``.
    """
    npz = np.load(str(path), allow_pickle=False)
    coords = np.asarray(npz["coords"], dtype=np.float32)
    if coords.ndim != 2 or int(coords.shape[1]) != 3:
        raise ValueError(
            f"Prototype anchor file {path} must contain coords with shape [P, 3]."
        )
    return torch.from_numpy(coords)


def load_context_prototype_anchors(path: str | Path) -> torch.Tensor:
    r"""Load context-side prototype anchor coordinates.

    Alias of :func:`load_target_prototype_anchors`; the file layout is
    identical for context and target anchor files.
    """
    return load_target_prototype_anchors(path)


def _build_prototype_anchors_from_points(
    *,
    train_dataset,
    cfg: dict[str, Any],
    output_path: str | Path,
    seed: int,
    source_name: str,
    point_loader,
) -> torch.Tensor:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prototype_count = int(cfg.get("prototype_anchor_count", 3072))
    prototype_num_passes = int(cfg.get("prototype_num_passes", 1))
    max_candidate_coords = int(cfg.get("prototype_max_candidate_coords", 65536))
    kmeans_iters = int(cfg.get("prototype_kmeans_iters", 12))
    assignment_chunk_size = int(cfg.get("prototype_assignment_chunk_size", 4096))

    source_strategy = str(
        cfg.get("prototype_source_strategy", "voxel_fps_cluster")
    ).lower()
    source_max_tokens = int(
        cfg.get(
            "prototype_source_max_point_tokens",
            cfg.get("max_point_tokens", prototype_count),
        )
    )
    source_cluster_size = cfg.get(
        "prototype_source_cluster_size", cfg.get("tokenizer_cluster_size")
    )
    source_voxel_size = cfg.get(
        "prototype_source_voxel_size", cfg.get("tokenizer_voxel_size")
    )
    source_knn_chunk_size = int(
        cfg.get(
            "prototype_source_knn_chunk_size",
            cfg.get("tokenizer_knn_chunk_size", 128),
        )
    )

    tokenizer = PointCloudTokenizer(
        max_point_tokens=source_max_tokens,
        strategy=source_strategy,
        deterministic_eval=False,
        cluster_size=source_cluster_size,
        knn_chunk_size=source_knn_chunk_size,
        voxel_size=source_voxel_size,
    )
    tokenizer.train(True)

    all_coords = []
    for pass_idx in range(prototype_num_passes):
        for sample_idx in range(len(train_dataset)):
            sample = train_dataset[sample_idx]
            positions, features = point_loader(sample)
            token_positions, _, _ = tokenizer.tokenize_with_clusters(
                point_positions=positions,
                point_features=features,
            )
            all_coords.append(token_positions.detach().cpu())
        if pass_idx == 0:
            print(
                f"[PrototypeAnchors] collected first pass {source_name} token coords "
                f"from {len(train_dataset)} train cases."
            )

    if not all_coords:
        raise ValueError(
            f"No {source_name} token coordinates were collected for prototype anchors."
        )

    coords = torch.cat(all_coords, dim=0).float()
    if int(coords.shape[0]) > max_candidate_coords:
        gen = torch.Generator(device=coords.device)
        gen.manual_seed(int(seed))
        subset_idx = torch.randperm(
            int(coords.shape[0]), generator=gen, device=coords.device
        )[:max_candidate_coords]
        coords = coords[subset_idx]

    centers = _run_chunked_kmeans(
        coords,
        num_clusters=prototype_count,
        num_iters=kmeans_iters,
        seed=seed,
        chunk_size=assignment_chunk_size,
    ).cpu()

    centers_np = centers.numpy()
    order = np.lexsort((centers_np[:, 2], centers_np[:, 1], centers_np[:, 0]))
    centers_np = centers_np[order].astype(np.float32, copy=False)
    metadata = {
        "prototype_anchor_count": int(centers_np.shape[0]),
        "prototype_num_passes": prototype_num_passes,
        "source_name": source_name,
        "source_strategy": source_strategy,
        "source_max_point_tokens": source_max_tokens,
        "source_cluster_size": None
        if source_cluster_size is None
        else int(source_cluster_size),
        "source_voxel_size": None
        if source_voxel_size is None
        else [float(v) for v in source_voxel_size],
        "seed": int(seed),
        "kmeans_iters": int(kmeans_iters),
        "max_candidate_coords": int(max_candidate_coords),
        "train_cases": int(len(train_dataset)),
    }
    np.savez_compressed(
        out_path,
        coords=centers_np,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(
        f"[PrototypeAnchors] saved {centers_np.shape[0]} {source_name} anchors to {out_path}"
    )
    return torch.from_numpy(centers_np)


def build_context_prototype_anchors(
    *,
    train_dataset,
    context_cfg: dict[str, Any],
    output_path: str | Path,
    seed: int = 42,
) -> torch.Tensor:
    r"""Build context-side prototype anchors by k-means over training samples.

    Walks the training dataset, tokenizes each sample's surface points with
    a :class:`PointCloudTokenizer` configured by ``context_cfg``, collects
    the token coordinates, optionally subsamples the union to
    ``prototype_max_candidate_coords`` points, runs chunked k-means, sorts
    the centers lexicographically for determinism, and writes the result
    plus a JSON-encoded metadata blob to ``output_path``.

    Parameters
    ----------
    train_dataset : object
        A dataset-like object supporting ``len(...)`` and integer
        ``__getitem__`` that returns a dict containing at least
        ``surface_pos`` and ``surface_feat`` tensors.
    context_cfg : dict
        Configuration mapping. Recognized keys include
        ``prototype_anchor_count`` (default 3072),
        ``prototype_num_passes`` (default 1),
        ``prototype_max_candidate_coords`` (default 65536),
        ``prototype_kmeans_iters`` (default 12),
        ``prototype_assignment_chunk_size`` (default 4096),
        ``prototype_source_strategy`` (default ``"voxel_fps_cluster"``),
        ``prototype_source_max_point_tokens``,
        ``prototype_source_cluster_size``,
        ``prototype_source_voxel_size``,
        ``prototype_source_knn_chunk_size``,
        ``use_sdf`` / ``use_solid_normals`` / ``use_solid_n_dot_uinf``
        (for the surface-feature assembly).
    output_path : str or Path
        Where to write the ``.npz`` anchor file.
    seed : int, optional
        Seed for FPS and the candidate subsampling. Default 42.

    Returns
    -------
    torch.Tensor
        Anchor coordinates of shape ``(P, 3)``, dtype ``float32``.
    """
    return _build_prototype_anchors_from_points(
        train_dataset=train_dataset,
        cfg=context_cfg,
        output_path=output_path,
        seed=seed,
        source_name="context",
        point_loader=lambda sample: _concat_context_points(sample, context_cfg),
    )


def build_target_prototype_anchors(
    *,
    train_dataset,
    target_cfg: dict[str, Any],
    output_path: str | Path,
    seed: int = 42,
) -> torch.Tensor:
    r"""Build target-side prototype anchors by k-means over training samples.

    Same procedure as :func:`build_context_prototype_anchors`, but feeds
    the tokenizer with the concatenation of the sample's target surface
    points and target volume points (or the regular surface/volume points
    when target-specific keys are absent).

    Parameters
    ----------
    train_dataset : object
        Dataset-like object as in :func:`build_context_prototype_anchors`.
        Samples must additionally provide ``volume_pos`` and
        ``volume_feat`` tensors (target variants are used when present).
    target_cfg : dict
        Configuration mapping. Same keys as
        :func:`build_context_prototype_anchors` plus any tokenizer
        defaults.
    output_path : str or Path
        Where to write the ``.npz`` anchor file.
    seed : int, optional
        Seed for FPS and the candidate subsampling. Default 42.

    Returns
    -------
    torch.Tensor
        Anchor coordinates of shape ``(P, 3)``, dtype ``float32``.
    """
    return _build_prototype_anchors_from_points(
        train_dataset=train_dataset,
        cfg=target_cfg,
        output_path=output_path,
        seed=seed,
        source_name="target",
        point_loader=_concat_target_points,
    )


def ensure_context_prototype_anchors(
    *,
    train_dataset,
    context_cfg: dict[str, Any],
    seed: int = 42,
) -> torch.Tensor:
    r"""Load context prototype anchors, building them first if missing.

    Looks up ``context_cfg["prototype_anchor_path"]``. If the file exists
    and ``context_cfg["prototype_rebuild"]`` is falsy, the existing file is
    loaded; otherwise the anchors are built (and the existing file
    overwritten when present).

    Parameters
    ----------
    train_dataset : object
        Forwarded to :func:`build_context_prototype_anchors` when a
        rebuild is needed.
    context_cfg : dict
        Configuration mapping. Must contain ``prototype_anchor_path``;
        ``prototype_rebuild`` forces a rebuild even when the file exists.
    seed : int, optional
        Forwarded to the builder. Default 42.

    Returns
    -------
    torch.Tensor
        Anchor coordinates of shape ``(P, 3)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If ``prototype_anchor_path`` is not set.
    """
    output_path = context_cfg.get("prototype_anchor_path")
    if not output_path:
        raise ValueError(
            "context_encoder.prototype_anchor_path is required when "
            "tokenizer_strategy='data_prototype_cluster'."
        )
    rebuild = bool(context_cfg.get("prototype_rebuild", False))
    path = Path(output_path)
    if path.exists() and not rebuild:
        return load_context_prototype_anchors(path)
    return build_context_prototype_anchors(
        train_dataset=train_dataset,
        context_cfg=context_cfg,
        output_path=path,
        seed=seed,
    )


def ensure_target_prototype_anchors(
    *,
    train_dataset,
    target_cfg: dict[str, Any],
    seed: int = 42,
) -> torch.Tensor:
    r"""Load target prototype anchors, building them first if missing.

    Mirror of :func:`ensure_context_prototype_anchors` for the target
    branch.

    Parameters
    ----------
    train_dataset : object
        Forwarded to :func:`build_target_prototype_anchors` when a
        rebuild is needed.
    target_cfg : dict
        Configuration mapping. Must contain ``prototype_anchor_path``;
        ``prototype_rebuild`` forces a rebuild even when the file exists.
    seed : int, optional
        Forwarded to the builder. Default 42.

    Returns
    -------
    torch.Tensor
        Anchor coordinates of shape ``(P, 3)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If ``prototype_anchor_path`` is not set.
    """
    output_path = target_cfg.get("prototype_anchor_path")
    if not output_path:
        raise ValueError(
            "target_encoder.prototype_anchor_path is required when "
            "tokenizer_strategy='data_prototype_cluster'."
        )
    rebuild = bool(target_cfg.get("prototype_rebuild", False))
    path = Path(output_path)
    if path.exists() and not rebuild:
        return load_target_prototype_anchors(path)
    return build_target_prototype_anchors(
        train_dataset=train_dataset,
        target_cfg=target_cfg,
        output_path=path,
        seed=seed,
    )
