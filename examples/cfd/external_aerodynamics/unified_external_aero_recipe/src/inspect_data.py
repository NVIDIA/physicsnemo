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
Inspect mesh datasets to determine coordinate ranges, field names, and shapes.

Loads one sample from each configured dataset via Hydra-instantiated readers
and prints a summary useful for determining the vertical axis and
understanding data layout.

Usage::

    python -m src.inspect_data
    python -m src.inspect_data --configs conf/dataset/drivaer_ml_surface.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.mesh import Mesh


CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf" / "dataset"

DEFAULT_CONFIGS = [
    CONFIG_DIR / "drivaer_ml_surface.yaml",
    CONFIG_DIR / "shift_suv_estate.yaml",
]


def _print_tensordict_fields(td, indent: int = 4) -> None:
    prefix = " " * indent
    if not td.keys():
        print(f"{prefix}(empty)")
        return
    for key in sorted(td.keys()):
        val = td[key]
        if hasattr(val, "shape"):
            print(f"{prefix}{key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"{prefix}{key}: {type(val).__name__}")


def inspect_mesh(name: str, mesh: Mesh) -> None:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    print(f"\nGeometry:")
    print(f"  points: {tuple(mesh.points.shape)}")
    print(f"  cells:  {tuple(mesh.cells.shape)}")
    print(f"  n_spatial_dims: {mesh.n_spatial_dims}")

    pts = mesh.points
    print(f"\nCoordinate ranges (for vertical axis determination):")
    for dim, label in enumerate(["X", "Y", "Z"][: pts.shape[1]]):
        col = pts[:, dim]
        print(
            f"  {label}: min={col.min().item():+.4f}  "
            f"max={col.max().item():+.4f}  "
            f"range={col.max().item() - col.min().item():.4f}  "
            f"mean={col.mean().item():+.4f}"
        )

    print(f"\npoint_data fields:")
    _print_tensordict_fields(mesh.point_data)

    print(f"\ncell_data fields:")
    _print_tensordict_fields(mesh.cell_data)

    print(f"\nglobal_data fields:")
    _print_tensordict_fields(mesh.global_data)

    for section_name, section in [
        ("point_data", mesh.point_data),
        ("cell_data", mesh.cell_data),
    ]:
        for key in sorted(section.keys()):
            val = section[key]
            if not hasattr(val, "shape"):
                continue
            flat = val.reshape(-1) if val.ndim > 1 else val
            print(
                f"\n  {section_name}.{key} value stats:"
                f"  min={flat.min().item():+.6g}"
                f"  max={flat.max().item():+.6g}"
                f"  mean={flat.float().mean().item():+.6g}"
                f"  std={flat.float().std().item():.6g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect mesh datasets from YAML configs"
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        type=str,
        default=None,
        help="YAML config file paths (default: all in conf/dataset/)",
    )
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.configs] if args.configs else DEFAULT_CONFIGS

    for path in config_paths:
        if not path.exists():
            print(f"Config not found: {path}")
            continue

        cfg = OmegaConf.load(path)
        name = OmegaConf.select(cfg, "name", default=path.stem)
        datadir = OmegaConf.select(cfg, "train_datadir", default="")

        if not Path(datadir).exists():
            print(f"\nSkipping {name}: path {datadir} does not exist")
            continue

        reader = hydra.utils.instantiate(cfg.pipeline.reader)
        print(f"\n{name}: found {len(reader)} samples")

        mesh, metadata = reader[0]
        print(f"  source: {metadata.get('source_path', 'unknown')}")
        inspect_mesh(name, mesh)


if __name__ == "__main__":
    with torch.no_grad():
        main()
