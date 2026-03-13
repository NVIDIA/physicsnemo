# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load datapipes from Hydra config, iterate batches, and visualize input (x) and output (y).
# Usage: python load_and_visualize_data.py data.numpy_path=/path/to/npz data.hdf5_path=/path/to/h5

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from physicsnemo import datapipes

def _squeeze_2d(tensor):
    """Return (H, W) for (H,W), (1,H,W), (C,H,W) with C==1, or a single sample from (B,C,H,W)."""
    import torch

    t = tensor
    if t.dim() == 2:
        return t
    if t.dim() == 3 and t.shape[0] == 1:
        return t[0]
    if t.dim() == 3:
        return t[0]
    if t.dim() == 4:
        return t[0].squeeze(0) if t.shape[1] == 1 else t[0]
    return t.squeeze()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    dataloader = hydra.utils.instantiate(cfg.dataloader)

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input/output keys after config transforms (Rename: coeff/nu -> x, sol/tensor -> y)
    in_key = "x"
    out_key = "y"

    n_batches_show = min(3, len(dataloader))
    for batch_idx, batch_out in enumerate(dataloader):
        if batch_idx >= n_batches_show:
            break

        if isinstance(batch_out, tuple):
            data, meta_list = batch_out
        else:
            data = batch_out
            batch_size = next(
                (data[k].shape[0] for k in data.keys() if data[k].dim() >= 2),
                0,
            )
            meta_list = [{}] * batch_size

        if in_key not in data or out_key not in data:
            keys = [k for k in data.keys() if data[k].dim() >= 2]
            in_key_use = keys[0] if len(keys) > 0 else None
            out_key_use = keys[1] if len(keys) > 1 else keys[0]
        else:
            in_key_use = in_key
            out_key_use = out_key

        if in_key_use is None:
            continue

        B = data[in_key_use].shape[0]
        if B == 0:
            continue
        fig, axes = plt.subplots(B, 2, figsize=(6, 3 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        for b in range(B):
            in_grid = _squeeze_2d(data[in_key_use][b])
            out_grid = _squeeze_2d(data[out_key_use][b]) if out_key_use else None
            if hasattr(in_grid, "numpy"):
                in_grid = in_grid.detach().cpu().numpy()
            meta = meta_list[b] if b < len(meta_list) else {}
            ds_idx = meta.get("dataset_index", -1)

            ax_in = axes[b, 0]
            ax_in.imshow(in_grid)
            ax_in.set_title(f"Input ({in_key_use}) [b={b}, batch={batch_idx}, ds={ds_idx}]")
            ax_in.set_axis_off()

            ax_out = axes[b, 1]
            if out_grid is not None:
                if hasattr(out_grid, "numpy"):
                    out_grid = out_grid.detach().cpu().numpy()
                ax_out.imshow(out_grid)
            ax_out.set_title(f"Output ({out_key_use}) [b={b}]")
            ax_out.set_axis_off()

        fig.tight_layout()
        fig.savefig(out_dir / f"batch_{batch_idx:02d}.png", dpi=100)
        plt.close(fig)

    print(f"Saved {n_batches_show} batch figures to {out_dir}")


if __name__ == "__main__":
    main()
