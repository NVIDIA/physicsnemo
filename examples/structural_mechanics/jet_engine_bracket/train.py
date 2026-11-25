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

r"""Recipe 1: train a GeoTransolver on DeepJEB.

Every epoch visits every training design under all four load cases, one sample per GPU and
step, every mesh node a query point. The optimizer is Muon for the weight matrices and AdamW for
the rest, with a linear warm-up and a cosine decay; the forward pass runs under bfloat16
autocast, and the gradient norm is clipped (a step whose norm is not finite is skipped). A
checkpoint is written after every epoch, and a rerun resumes from the latest one; the final
model is saved as ``<output_dir>/model.mdlus``.

    python train.py data.root=<data>/deepjeb
    torchrun --nproc_per_node 4 train.py data.root=<data>/deepjeb
    python train.py data.root=<data>/deepjeb data.max_designs=8 training.epochs=2   # smoke test
"""

from __future__ import annotations

import json
import time

import hydra
import torch
from bracket_io import load_split
from datapipe import BracketDataset, load_scales, model_inputs
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from utils import (
    autocast,
    case_metrics,
    muon_adamw,
    resolve,
    set_seed,
    warmup_cosine,
    weighted_mse,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint


def loader(
    dataset: BracketDataset,
    manager: DistributedManager,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    r"""One sample per step, sharded across the ranks."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=manager.world_size,
        rank=manager.rank,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def reduce_sum(
    values: list[float], device: torch.device, manager: DistributedManager
) -> list[float]:
    r"""Sum of ``values`` over the ranks."""
    total = torch.tensor(values, dtype=torch.float64, device=device)
    if manager.distributed:
        torch.distributed.all_reduce(total)
    return total.tolist()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data: DataLoader,
    device: torch.device,
    manager: DistributedManager,
    precision: str,
) -> dict[str, float]:
    r"""Mean per-sample relative L2 errors over the validation samples (all ranks)."""
    model.eval()
    sums = [0.0, 0.0, 0.0]
    for batch in data:
        with autocast(device, precision):
            prediction = model(**model_inputs(batch, device))[0].float()
        m = case_metrics(prediction, batch["target"][0].to(device))
        sums = [
            sums[0] + 1,
            sums[1] + m["displacement_rel_l2"],
            sums[2] + m["von_mises_rel_l2"],
        ]
    count, displacement, stress = reduce_sum(sums, device, manager)
    model.train()
    return {
        "displacement_rel_l2": displacement / count,
        "von_mises_rel_l2": stress / count,
    }


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    r"""Train the model on DeepJEB (Hydra entry point, ``conf/train.yaml``)."""
    DistributedManager.initialize()
    manager = DistributedManager()
    device, rank0 = manager.device, manager.rank == 0
    t = cfg.training
    set_seed(t.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    split, scales = (
        load_split(resolve(cfg.data.split)),
        load_scales(resolve(cfg.data.scales)),
    )
    n = cfg.data.max_designs
    common = dict(
        scales=scales,
        coordinate_scale_mm=cfg.data.coordinate_scale_mm,
        token_voxel_mm=cfg.data.token_voxel_mm,
        sdf=bool(cfg.model.get("sdf_spacings")),  # the surface, for GeoTransolverSDF
    )
    train_data = loader(
        BracketDataset(cfg.data.root, split["train"][:n], **common),
        manager,
        True,
        t.seed,
        cfg.data.num_workers,
    )
    val_data = loader(
        BracketDataset(cfg.data.root, split["val"][:n], **common),
        manager,
        False,
        t.seed,
        cfg.data.num_workers,
    )

    model = instantiate(cfg.model, _convert_="all").to(
        device
    )  # plain lists, so that the model arguments serialize
    network = (
        DistributedDataParallel(model, device_ids=[manager.local_rank])
        if manager.distributed
        else model
    )
    optimizer = muon_adamw(model, t.lr, t.weight_decay)
    steps = len(train_data)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: warmup_cosine(
            s, t.warmup_epochs * steps, t.epochs * steps, t.final_lr_ratio
        ),
    )

    output = resolve(cfg.output_dir)
    checkpoints = output / "checkpoints"
    start = (
        load_checkpoint(
            checkpoints,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if checkpoints.exists()
        else 0
    )
    if rank0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        params = sum(p.numel() for p in model.parameters())
        print(
            f"{len(train_data.dataset)} training and {len(val_data.dataset)} validation samples, {manager.world_size} ranks, "
            f"{params / 1e6:.2f} M parameters, starting at epoch {start}",
            flush=True,
        )
    log = (output / "train_log.jsonl").open("a") if rank0 else None

    for epoch in range(start, t.epochs):
        train_data.sampler.set_epoch(epoch)
        started, loss_sum, skipped = time.time(), 0.0, 0
        for step, batch in enumerate(train_data):
            target = batch["target"].to(device, non_blocking=True)
            with autocast(device, t.precision):
                prediction = network(**model_inputs(batch, device))
            loss = weighted_mse(prediction, target, t.loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            if torch.isfinite(
                norm
            ):  # the gradients are all-reduced, so every rank decides alike
                optimizer.step()
            else:
                skipped += 1
            scheduler.step()
            loss_sum += float(loss.detach())
            if rank0 and t.log_every and (step + 1) % t.log_every == 0:
                print(
                    f"epoch {epoch + 1} step {step + 1}/{steps} loss {loss_sum / (step + 1):.5f}",
                    flush=True,
                )
        total, count, skipped_all = reduce_sum(
            [loss_sum, steps, skipped], device, manager
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "skipped_steps": int(skipped_all),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - started, 1),
        }
        if (epoch + 1) % t.val_every == 0 or epoch + 1 == t.epochs:
            record["validation"] = validate(
                network, val_data, device, manager, t.precision
            )
        if rank0:
            save_checkpoint(
                checkpoints,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
            )
            for old in (
                checkpoints.iterdir()
            ):  # <name>.<rank>.<epoch>.<suffix>: keep the two latest epochs
                saved = old.name.split(".")[-2]
                if saved.isdigit() and int(saved) < epoch:
                    old.unlink()
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    if rank0:
        model.save(str(output / "model.mdlus"))
        print(f"saved {output / 'model.mdlus'}", flush=True)


if __name__ == "__main__":
    main()
