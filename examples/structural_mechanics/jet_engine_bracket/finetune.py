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

r"""Recipe 3: adapt the DeepJEB-trained model to SimJEB with LoRA (:mod:`physicsnemo.experimental.peft`).

The pre-trained model is frozen; low-rank adapters on every linear layer are the only trainable
parameters (``lora_B`` starts at zero, so training starts from the pre-trained model), trained
with AdamW on the SimJEB training designs, every node and all four load cases. Targets keep the
DeepJEB normalization, so the adapters learn only where the pre-trained prediction is wrong. The
loss is the per-sample relative squared error, the mean of its displacement and stress terms, so
that every design counts alike whatever its stress level.

The pre-trained and the adapted model are evaluated in float32 on every node of the validation
and test designs (``summary.json``). The adapter is saved as ``<output_dir>/adapter.lora``; a
rerun resumes from ``state.pt``.

    python finetune.py data.root=<data>/simjeb
    torchrun --nproc_per_node 4 finetune.py data.root=<data>/simjeb
"""

from __future__ import annotations

import json
import time

import hydra
import torch
from bracket_io import load_split
from datapipe import BracketDataset, load_scales, model_inputs
from evaluate import evaluate, load_model, summarize
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from train import loader, reduce_sum, validate
from utils import autocast, resolve, set_seed, warmup_cosine

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.geotransolver_sdf import GeoTransolverSDF
from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    save_adapter,
    split_params_for_optimizer,
)


def relative_squared_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Mean of the relative squared errors of displacement and stress of one sample, :math:`(1, N, 4)` tensors."""
    error, target = (prediction.float() - target) ** 2, target**2
    return 0.5 * (
        error[..., :3].sum() / target[..., :3].sum()
        + error[..., 3].sum() / target[..., 3].sum()
    )


@hydra.main(version_base=None, config_path="conf", config_name="finetune")
def main(cfg: DictConfig) -> None:
    r"""Adapt a pre-trained model to SimJEB with LoRA (Hydra entry point, ``conf/finetune.yaml``)."""
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
    model = load_model(cfg.checkpoint, None, device)
    datasets = {
        k: BracketDataset(
            cfg.data.root,
            split[k][:n],
            scales,
            coordinate_scale_mm=cfg.data.coordinate_scale_mm,
            token_voxel_mm=cfg.data.token_voxel_mm,
            sdf=isinstance(
                model, GeoTransolverSDF
            ),  # the surface, for GeoTransolverSDF
        )
        for k in ("train", "val", "test")
    }
    train_data = loader(datasets["train"], manager, True, t.seed, cfg.data.num_workers)
    val_data = loader(datasets["val"], manager, False, t.seed, cfg.data.num_workers)
    output = resolve(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = {}
    if rank0:  # the pre-trained model, before any adaptation
        summary["pretrained"] = {
            k: summarize(evaluate(model, datasets[k], device))["all"]
            for k in ("val", "test")
        }
        print(json.dumps({"pretrained": summary["pretrained"]}), flush=True)
    result = apply_lora(
        model,
        LoRAConfig(
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            target_filter=lambda name, module: isinstance(module, torch.nn.Linear),
        ),
    )
    model.train()
    trainable = split_params_for_optimizer(model)["lora"]
    optimizer = torch.optim.AdamW(trainable, lr=t.lr, weight_decay=0.0)
    steps = len(train_data)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: warmup_cosine(
            s, int(t.warmup_epochs * steps), t.epochs * steps, t.final_lr_ratio
        ),
    )
    network = (
        DistributedDataParallel(model, device_ids=[manager.local_rank])
        if manager.distributed
        else model
    )
    start, state_file = 0, output / "state.pt"
    if state_file.exists():
        state = torch.load(state_file, map_location=device, weights_only=False)
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["epoch"]
    if rank0:
        (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        print(
            f"LoRA rank {cfg.lora.rank} on {result.n_wrapped} linear layers, {sum(p.numel() for p in trainable) / 1e6:.2f} M trainable "
            f"parameters; {len(datasets['train'])} training samples, starting at epoch {start}",
            flush=True,
        )
    log = (output / "train_log.jsonl").open("a") if rank0 else None

    for epoch in range(start, t.epochs):
        train_data.sampler.set_epoch(epoch)
        started, loss_sum = time.time(), 0.0
        for step, batch in enumerate(train_data):
            with autocast(device, t.precision):
                prediction = network(**model_inputs(batch, device))
            loss = relative_squared_error(
                prediction, batch["target"].to(device, non_blocking=True)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, t.grad_clip)
            if torch.isfinite(norm):
                optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
            if rank0 and t.log_every and (step + 1) % t.log_every == 0:
                print(
                    f"epoch {epoch + 1} step {step + 1}/{steps} loss {loss_sum / (step + 1):.5f}",
                    flush=True,
                )
        total, count = reduce_sum([loss_sum, steps], device, manager)
        record = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - started, 1),
        }
        if (epoch + 1) % t.val_every == 0 or epoch + 1 == t.epochs:
            record["validation"] = validate(
                network, val_data, device, manager, t.precision
            )
        if rank0:
            adapter = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(
                {
                    "epoch": epoch + 1,
                    "adapter": adapter,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                state_file,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    if rank0:
        save_adapter(model, output / "adapter.lora")
        model.eval()
        summary["adapted"] = {
            k: summarize(evaluate(model, datasets[k], device))["all"]
            for k in ("val", "test")
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps({"adapted": summary["adapted"]}), flush=True)


if __name__ == "__main__":
    main()
