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

"""LoRA fine-tuning recipe for GeoTransolver (companion to ``train.py``).

A standalone entry point that lives alongside ``train.py`` and reuses this
example's configs (``conf/``) and data pipeline — without modifying
``train.py``. Workflow (see ``FINETUNE_LORA.md``):

  1. Build the model and load a PRETRAINED base checkpoint (``init_from``).
  2. Inject LoRA adapters and freeze the base (``apply_lora``).
  3. Train ONLY the adapters on a (small) target dataset, routing the LoRA
     parameters to AdamW via ``split_params_for_optimizer`` (never Muon).
  4. Save a small ``.mdlus`` adapter (``save_adapter``).

Run from the example root (same convention as ``train.py``)::

    python src/finetune.py init_from=/path/to/base_geotransolver.mdlus

Deploy / merge is demonstrated in ``src/deploy.py``.
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.peft import (
    apply_lora,
    print_trainable_parameters,
    save_adapter,
    split_params_for_optimizer,
)

logger = logging.getLogger("finetune_lora")


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward_loss(model, batch: dict) -> torch.Tensor:
    """GeoTransolver ("Typhon") forward + MSE, mirroring the geometry path in
    ``train.py::forward_pass``. (For full metrics/normalization you can import
    and reuse ``train.forward_pass`` instead; kept minimal here for clarity.)"""
    features = batch["fx"]  # global features
    embeddings = batch["embeddings"]  # local features
    targets = batch["fields"]
    geometry = batch.get("geometry")
    local_positions = embeddings[:, :, :3]
    outputs = model(
        global_embedding=features,
        local_embedding=embeddings,
        geometry=geometry,
        local_positions=local_positions,
    )
    return torch.nn.functional.mse_loss(outputs, targets)


def _load_norm_factors(cfg: DictConfig, device) -> dict:
    """Load the normalization .npz factors train.py uses for the active mode.

    Fails loudly if a required factor file is missing — training on unnormalized
    targets silently produces a useless adapter (train.py errors here too).
    """
    norm_dir = Path(cfg.data.get("normalization_dir", "src/"))
    needed = {
        "surface": ["surface"],
        "volume": ["volume"],
        "combined": ["surface", "volume"],
    }.get(cfg.data.get("mode", "surface"), ["surface"])
    factors: dict = {}
    for mode in needed:
        f = norm_dir / f"{mode}_fields_normalization.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"normalization factors not found: {f}. Set data.normalization_dir "
                f"to the directory holding {mode}_fields_normalization.npz; do not "
                f"fine-tune on unnormalized targets."
            )
        d = np.load(f)
        factors[mode] = {
            "mean": torch.from_numpy(d["mean"]).to(device),
            "std": torch.from_numpy(d["std"]).to(device),
        }
    return factors


@hydra.main(version_base=None, config_path="conf", config_name="finetune_lora")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Minimal float32 recipe: the mixed/fp8 path (autocast, fp8 input/output
    # padding, GradScaler) lives in train.py and is intentionally not wired here.
    if str(cfg.get("precision", "float32")) != "float32":
        raise NotImplementedError(
            f"finetune.py supports precision=float32 only (got {cfg.get('precision')!r}); "
            "use train.py's precision path for fp16/fp8."
        )

    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device
    is_rank0 = dm.rank == 0
    distributed = dm.world_size > 1

    # Build model and load the pretrained base to fine-tune from.
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    if cfg.get("init_from"):
        logger.info("loading base checkpoint: %s", cfg.init_from)
        model.load(str(cfg.init_from), map_location=device)
    else:
        logger.warning("no init_from set — fine-tuning from random init (demo only)")

    # Inject LoRA + freeze base.
    peft_cfg = hydra.utils.instantiate(cfg.peft)
    result = apply_lora(model, peft_cfg)
    if is_rank0:
        print_trainable_parameters(model)
        logger.info("LoRA: wrapped %d layers", result.n_wrapped)

    # split BEFORE DDP wrapping (operate on the real model holding the params).
    groups = split_params_for_optimizer(model)
    trainable = groups["lora"] + groups["extras"]
    # PEFT needs find_unused_parameters: frozen base params receive no gradient.
    if distributed:
        model = DDP(
            model,
            device_ids=[dm.local_rank],
            output_device=device,
            find_unused_parameters=True,
        )
    base = model.module if distributed else model

    # Optimizer: LoRA (+extras) → AdamW. NOT Muon (Newton-Schulz is degenerate
    # on rank-r factors; see plan §3.5). The base config's optimizer is AdamW.
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=trainable)

    # Data (reuses this example's library datapipe + normalization factors).
    norm = _load_norm_factors(cfg, device)
    train_loader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=norm.get("surface"),
        volume_factors=norm.get("volume"),
    )
    # Shard across ranks (mirrors train.py): without this every rank iterates the
    # FULL dataset and computes identical gradients — no data parallelism.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_loader,
        num_replicas=dm.world_size,
        rank=dm.rank,
        shuffle=True,
        drop_last=True,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / f"{cfg.run_id}.mdlus"
    save_interval = int(cfg.training.get("save_interval", 10))

    for epoch in range(int(cfg.training.num_epochs)):
        train_sampler.set_epoch(epoch)
        train_loader.dataset.set_indices(list(train_sampler))
        base.train()
        for i, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            loss = _forward_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if is_rank0 and i % 10 == 0:
                logger.info("epoch %d step %d loss %.6f", epoch, i, loss.item())
        if is_rank0 and (epoch + 1) % save_interval == 0:
            save_adapter(base, adapter_path)
            logger.info("saved adapter -> %s", adapter_path)

    if is_rank0:
        save_adapter(base, adapter_path)
        logger.info("done. adapter at %s", adapter_path)


if __name__ == "__main__":
    main()
