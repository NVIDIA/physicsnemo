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

"""Train a network on Rotor37 surface fields and compressor outputs."""

import csv
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
import torch.distributed as distributed
from dataset import Rotor37Dataset
from hydra.utils import instantiate, to_absolute_path
from models import build_model, predict
from objectives import FieldDecoder, loss_components
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

RUN_FILES = ("manifest.json", "stats.json", "basis.npz", "basis.json")


@contextmanager
def thread_limit(count):
    """Limit PyTorch and NumPy CPU threads."""
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        with threadpool_limits(limits=count, user_api="blas"):
            yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def strict_float32():
    """Compute float32 matrix products without TF32."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


def make_dataset(cfg, run_dir, split):
    """Encode a split with the transforms saved in a run directory."""
    return Rotor37Dataset(
        to_absolute_path(cfg.data.data_dir),
        split,
        basis_path=run_dir / "basis.npz",
        stats_path=run_dir / "stats.json",
        geometry_points=cfg.data.geometry_points,
    )


def to_device(batch, device):
    """Move the tensors of a batch to ``device``."""
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def check_dimensions(cfg, dataset, decoder):
    """Require model widths that match the encoded inputs and fitted bases."""
    features = len(dataset[0]["features"])
    outputs = 3 + decoder.coefficient_count
    model = cfg.model
    for key in ("functional_dim", "global_dim", "in_features"):
        if key in model and model[key] != features:
            raise ValueError(f"Set model.{key} to {features}")
    for key in ("out_dim", "out_features"):
        if key in model and model[key] != outputs:
            raise ValueError(f"Set model.{key} to {outputs} for the fitted bases")


def check_resume(cfg, run_dir, data_dir, world_size):
    """Require the configuration, data and process count of the saved run."""
    saved = OmegaConf.load(run_dir / "config.yaml")
    current = OmegaConf.to_container(cfg, resolve=True)
    previous = OmegaConf.to_container(saved, resolve=True)
    current["training"].pop("resume")
    previous["training"].pop("resume")
    if current != previous:
        raise ValueError("Resuming requires the configuration of the saved run")
    if json.loads((run_dir / "runtime.json").read_text())["processes"] != world_size:
        raise ValueError("Resuming requires the process count of the saved run")
    for name in ("manifest.json", "stats.json"):
        if (data_dir / name).read_bytes() != (run_dir / name).read_bytes():
            raise ValueError(f"{data_dir / name} differs from the saved run")


@torch.no_grad()
def validate(model, loader, decoder, device, world_size):
    """Return mean normalized field and compressor output errors."""
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    with strict_float32():
        for batch in loader:
            batch = to_device(batch, device)
            prediction = predict(model, batch, decoder)
            fields = (prediction.fields - batch["fields"]).square().mean(dim=(1, 2))
            globals_ = (prediction.globals - batch["globals"]).square().mean(dim=1)
            totals[0] += fields.sum()
            totals[1] += globals_.sum()
            totals[2] += len(fields)
    if world_size > 1:
        distributed.all_reduce(totals)
    model.train()
    field, global_ = (totals[:2] / totals[2]).tolist()
    return {"validation_field_loss": field, "validation_global_loss": global_}


def append_history(path, row):
    """Append one epoch to the CSV training history."""
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_history(path, last_epoch):
    """Drop history rows after the resumed checkpoint."""
    with path.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream) if int(row["epoch"]) <= last_epoch
        ]
    path.unlink()
    for row in rows:
        append_history(path, row)


def train(cfg):
    """Train one configuration and save checkpoints in ``cfg.output_dir``."""
    DistributedManager.initialize()
    dist = DistributedManager()
    device, rank, world_size = dist.device, dist.rank, dist.world_size
    log = RankZeroLoggingWrapper(PythonLogger("rotor37"), dist)
    run_dir = Path(to_absolute_path(cfg.output_dir))
    data_dir = Path(to_absolute_path(cfg.data.data_dir))
    checkpoint_dir = run_dir / "checkpoints"
    if cfg.training.resume:
        check_resume(cfg, run_dir, data_dir, world_size)
    elif (run_dir / "config.yaml").exists():
        raise FileExistsError(f"{run_dir} contains a run, set training.resume=true")
    if world_size > 1:
        distributed.barrier()
    if rank == 0 and not cfg.training.resume:
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            shutil.copy2(data_dir / name, run_dir / name)
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    if world_size > 1:
        distributed.barrier()

    torch.set_float32_matmul_precision("high")
    train_data = make_dataset(cfg, run_dir, "train")
    validation_data = make_dataset(cfg, run_dir, "validation")
    decoder = FieldDecoder(train_data.basis, train_data.stats).to(device)
    check_dimensions(cfg, train_data, decoder)
    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, seed=cfg.seed
    )
    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size, sampler=sampler
    )
    validation_loader = DataLoader(
        Subset(validation_data, range(rank, len(validation_data), world_size)),
        batch_size=cfg.training.batch_size,
    )
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs, eta_min=cfg.training.min_lr
    )
    first_epoch = 1
    if cfg.training.resume:
        first_epoch += load_checkpoint(
            str(checkpoint_dir),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if rank == 0:
            truncate_history(run_dir / "history.csv", first_epoch - 1)
    elif rank == 0:
        cuda = device.type == "cuda"
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if cuda else "cpu",
            "processes": world_size,
            "global_batch_size": cfg.training.batch_size * world_size,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    training_model = model
    if world_size > 1:
        training_model = DistributedDataParallel(
            model, device_ids=[dist.local_rank] if device.type == "cuda" else None
        )

    weights = dict(cfg.training.loss_weights)
    log.info(f"Training on {len(train_data)} cases with {world_size} processes")
    for epoch in range(first_epoch, cfg.training.num_epochs + 1):
        started = time.perf_counter()
        torch.manual_seed(cfg.seed + epoch * world_size + rank)
        sampler.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        totals = torch.zeros(len(weights) + 1, dtype=torch.float64, device=device)
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(training_model, batch, decoder)
            components = loss_components(prediction, batch, decoder)
            loss = sum(
                weight * components[name].mean() for name, weight in weights.items()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            totals[:-1] += torch.stack(
                [components[name].detach().sum() for name in weights]
            )
            totals[-1] += len(batch["features"])
        if world_size > 1:
            distributed.all_reduce(totals)
        means = dict(zip(weights, (totals[:-1] / totals[-1]).tolist()))
        row = {
            "epoch": epoch,
            "loss": sum(weights[name] * value for name, value in means.items()),
            **{f"{name}_loss": value for name, value in means.items()},
            "validation_field_loss": "",
            "validation_global_loss": "",
            "learning_rate": learning_rate,
        }
        last = epoch == cfg.training.num_epochs
        if epoch % cfg.training.validation_interval == 0 or last:
            row.update(validate(model, validation_loader, decoder, device, world_size))
        scheduler.step()
        row["seconds"] = time.perf_counter() - started
        if rank == 0:
            append_history(run_dir / "history.csv", row)
            if epoch % cfg.training.checkpoint_interval == 0 or last:
                save_checkpoint(
                    str(checkpoint_dir),
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                )
        message = f"Epoch {epoch} loss {row['loss']:.6f}"
        if row["validation_field_loss"] != "":
            message += f" validation field {row['validation_field_loss']:.6f}"
        log.info(message)
    if world_size > 1:
        distributed.barrier()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the configured network with the Hydra configuration."""
    with thread_limit(cfg.num_threads):
        train(cfg)


if __name__ == "__main__":
    main()
