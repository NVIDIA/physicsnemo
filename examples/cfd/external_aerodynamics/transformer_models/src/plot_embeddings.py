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

#!/usr/bin/env python3
"""Visualize GeoTransolver embedding distributions from a combined-trained model.

Produces three figures:
  1. embedding_comparison.png  — per-sample line plot (first 10 dims, both poolings)
  2. embedding_histograms.png  — histogram grid of ALL reduced dimensions
  3. embedding_summary.png     — per-dimension mean±std bars and box plots

Usage (from transformer_models/src)::

    python plot_embeddings.py
    python plot_embeddings.py checkpoint_dir=/path/to/runs run_id=my_run

Output: saves to <output_dir>/<run_id>/
"""

import collections
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset
from physicsnemo.utils import load_checkpoint

from gp_utils import (
    apply_spectral_norm_to_model,
    cast_precisions,
    create_embedding_reduction,
    load_pretrained_model_only,
)

N_DIMS_LINE_PLOT = 10


def plot_line_comparison(
    mean_embeds: np.ndarray,
    attn_embeds: np.ndarray,
    pooling_type: str,
    out_dir: Path,
) -> None:
    """Per-sample line plot of the first N embedding dimensions (both poolings)."""
    n_samples = mean_embeds.shape[0]
    indices = np.arange(n_samples)

    fig, (ax_mean, ax_attn) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for d in range(min(N_DIMS_LINE_PLOT, mean_embeds.shape[1])):
        ax_mean.plot(
            indices,
            mean_embeds[:, d],
            "-o",
            ms=2,
            label=f"Dim {d}",
            alpha=0.8,
        )
    ax_mean.set_ylabel("Embedding value")
    ax_mean.set_title("GeoTransolver embeddings: mean pooling (first 10 dims)")
    ax_mean.legend(loc="upper right", ncol=2, fontsize=8)
    ax_mean.grid(True, alpha=0.3)

    for d in range(min(N_DIMS_LINE_PLOT, attn_embeds.shape[1])):
        ax_attn.plot(
            indices,
            attn_embeds[:, d],
            "-o",
            ms=2,
            label=f"Dim {d}",
            alpha=0.8,
        )
    ax_attn.set_xlabel("Sample index")
    ax_attn.set_ylabel("Embedding value")
    ax_attn.set_title(
        f"GeoTransolver embeddings: {pooling_type} pooling (first 10 dims)"
    )
    ax_attn.legend(loc="upper right", ncol=2, fontsize=8)
    ax_attn.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "embedding_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved line plot to {out_path}")
    plt.close(fig)


def plot_histogram_grid(
    attn_embeds: np.ndarray,
    pooling_type: str,
    out_dir: Path,
) -> None:
    """Histogram grid showing the distribution of every reduced dimension."""
    n_samples, n_dims = attn_embeds.shape
    n_cols = 8
    n_rows = int(np.ceil(n_dims / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for d in range(n_dims):
        r, c = divmod(d, n_cols)
        ax = axes[r, c]
        values = attn_embeds[:, d]
        ax.hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(
            values.mean(),
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"\u03bc={values.mean():.2f}",
        )
        ax.axvline(
            np.median(values),
            color="orange",
            linestyle=":",
            linewidth=1.2,
            label=f"med={np.median(values):.2f}",
        )
        ax.set_title(f"Dim {d}  (\u03c3={values.std():.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for d in range(n_dims, n_rows * n_cols):
        r, c = divmod(d, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Embedding dimension distributions — {pooling_type} pooling, "
        f"{n_dims} dims, {n_samples} val samples",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / "embedding_histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved histogram grid to {out_path}")
    plt.close(fig)


def plot_summary_stats(
    attn_embeds: np.ndarray,
    pooling_type: str,
    out_dir: Path,
) -> None:
    """Bar chart of per-dimension mean±std and box plots side-by-side."""
    n_dims = attn_embeds.shape[1]
    dims = np.arange(n_dims)
    means = attn_embeds.mean(axis=0)
    stds = attn_embeds.std(axis=0)

    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(18, 5))

    ax_bar.bar(
        dims,
        means,
        yerr=stds,
        capsize=3,
        color="steelblue",
        alpha=0.8,
        edgecolor="white",
    )
    ax_bar.set_xlabel("Embedding dimension")
    ax_bar.set_ylabel("Value")
    ax_bar.set_title(f"Per-dimension mean \u00b1 std ({pooling_type} pooling)")
    ax_bar.set_xticks(dims)
    ax_bar.grid(True, alpha=0.3, axis="y")

    bp = ax_box.boxplot(
        [attn_embeds[:, d] for d in range(n_dims)],
        labels=[str(d) for d in range(n_dims)],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax_box.set_xlabel("Embedding dimension")
    ax_box.set_ylabel("Value")
    ax_box.set_title(f"Per-dimension box plots ({pooling_type} pooling)")
    ax_box.tick_params(axis="x", labelsize=7)
    ax_box.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = out_dir / "embedding_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved summary stats to {out_path}")
    plt.close(fig)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="geotransolver_surface_gp",
)
def main(cfg: DictConfig) -> None:
    """Load model and plot embedding distributions from a trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir
    combined_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_combined"
    use_combined = Path(combined_ckpt_path).exists()

    if not use_combined:
        pretrained_ckpt_path = (
            getattr(
                cfg,
                "pretrained_checkpoint_path",
                None,
            )
            or f"{checkpoint_dir}/{cfg.run_id}/checkpoints"
        )
        gp_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_gp"

    print(
        f"Loading from {'combined' if use_combined else 'separate'} "
        f"checkpoint(s) under {checkpoint_dir}/{cfg.run_id}/"
    )

    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    surface_factors = {
        "mean": torch.from_numpy(np.load(norm_file)["mean"]).to(device),
        "std": torch.from_numpy(np.load(norm_file)["std"]).to(device),
    }

    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=None,
    )

    feat_dim = getattr(cfg, "embedding_feat_dim", 256)
    embed_dim = getattr(cfg, "embed_dim", 32)
    pooling_type = cfg.get("embedding_pooling", "attention")

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    sn_backbone = getattr(cfg, "spectral_norm_backbone", False)
    sn_coeff = getattr(cfg, "spectral_norm_coeff", 1.0)
    if sn_backbone:
        apply_spectral_norm_to_model(model, coeff=sn_coeff)
    model.to(device)

    use_spectral_norm = getattr(cfg, "spectral_norm_embedding", False)
    normalize_embeddings = getattr(cfg, "normalize_embeddings", False)
    embedding_target_scale = getattr(cfg, "embedding_target_scale", 1.0)
    embedding_reduction_model = create_embedding_reduction(
        pooling=pooling_type,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        spectral_norm=use_spectral_norm,
        normalize=normalize_embeddings,
        target_scale=embedding_target_scale,
    )
    embedding_reduction_model.to(device)

    if use_combined:
        load_checkpoint(
            path=combined_ckpt_path,
            models=[model, embedding_reduction_model],
            device=device,
        )
    else:
        load_pretrained_model_only(model, pretrained_ckpt_path)
        load_checkpoint(
            path=gp_ckpt_path,
            models=[embedding_reduction_model],
            device=device,
        )

    model.eval()
    embedding_reduction_model.eval()

    precision = getattr(cfg, "precision", "float32")

    mean_embeds_list = []
    attn_embeds_list = []

    with torch.no_grad():
        for batch in val_dataloader:
            features = batch["fx"]
            embeddings = batch["embeddings"]
            geometry = (
                cast_precisions(batch["geometry"], precision)
                if "geometry" in batch
                else None
            )
            features = cast_precisions(features, precision)
            embeddings = cast_precisions(embeddings, precision)
            local_positions = embeddings[:, :, :3]

            _, embedding_states = model(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
                return_embedding_states=True,
            )

            mean_pooled = embedding_states.flatten(1, 2).mean(dim=1)
            attn_pooled = embedding_reduction_model(embedding_states.flatten(1, 2))

            mean_embeds_list.append(mean_pooled.cpu().numpy())
            attn_embeds_list.append(attn_pooled.cpu().numpy())

    mean_embeds = np.concatenate(mean_embeds_list, axis=0)
    attn_embeds = np.concatenate(attn_embeds_list, axis=0)

    print(
        f"Collected embeddings — mean-pooled: {mean_embeds.shape}, "
        f"{pooling_type}-pooled: {attn_embeds.shape}"
    )

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_line_comparison(mean_embeds, attn_embeds, pooling_type, out_dir)
    plot_histogram_grid(attn_embeds, pooling_type, out_dir)
    plot_summary_stats(attn_embeds, pooling_type, out_dir)

    print(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
