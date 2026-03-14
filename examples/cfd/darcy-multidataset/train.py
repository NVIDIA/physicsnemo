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
#
# Training entrypoint for Darcy Transolver multi-dataset.
# Uses MultiDataset dataloader and Transolver with spatial shape matching
# target_size for permeability -> pressure. Muon used for 2D params when available.

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# This is needed for datapipe registry via hydra instantiation
from physicsnemo import datapipes
from physicsnemo.datapipes import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.optim import CombinedOptimizer
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger


class RelativeL2Loss:
    """Scale-invariant relative L2 loss: mean( ||pred - y||_2 / ||y||_2 )."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        diff = torch.norm(pred.reshape(B, -1) - target.reshape(B, -1), dim=1)
        ref = torch.norm(target.reshape(B, -1), dim=1)
        return torch.mean(diff / ref)


def make_spatial_positions(
    h: int,
    w: int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""Build a fixed 2D grid of normalized positions in :math:`[0,1]^2`, tiled over batch.

    Call once per run (same ``h, w`` as ``target_size`` / model ``structured_shape``).
    Use :meth:`torch.Tensor.expand` in the training loop if minibatch size can differ
    (e.g. last batch when ``drop_last`` is false).

    Parameters
    ----------
    h, w : int
        Grid height and width (row, col count), matching data spatial shape.
    batch_size : int
        Leading batch dimension (e.g. dataloader ``batch_size``).
    device : torch.device
        Where to place the tensor (e.g. training device).
    dtype : torch.dtype
        Floating dtype (e.g. ``torch.get_default_dtype()``).

    Returns
    -------
    torch.Tensor
        Shape :math:`(B, H, W, 2)` with last dimension ``(x, y)`` in index order
        ``ij`` (``y`` increases with row, ``x`` with column).
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    # (H, W, 2) then (1, H, W, 2) -> expand to (B, H, W, 2) without extra storage
    grid_hw2 = torch.stack([xx, yy], dim=-1)
    return grid_hw2.unsqueeze(0).expand(batch_size, -1, -1, -1)


def _normalize_for_image(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a tensor to [0, 1] for TensorBoard image logging."""
    t_min, t_max = t.min(), t.max()
    if t_max - t_min < 1e-8:
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def _log_sample_images(
    writer: SummaryWriter,
    tag_prefix: str,
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    epoch: int,
) -> None:
    """Log the first sample's x, y, and pred as grayscale images."""
    # x, y, pred arrive as (B, H, W, 1); take first sample -> (H, W, 1) -> (1, H, W)
    for name, tensor in [("x", x), ("y_true", y), ("y_pred", pred)]:
        img = _normalize_for_image(tensor[0].detach().squeeze(-1))  # (H, W)
        writer.add_image(f"{tag_prefix}/{name}", img.unsqueeze(0), epoch)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="darcy_transolver_multidataset")
    log.file_logging()

    # Multi-dataset Data Loader Instantiated from hydra:
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    dataset = dataloader.dataset
    n = len(dataset)
    val_fraction = getattr(cfg.training, "val_fraction", 0.2)
    split_seed = getattr(cfg.training, "split_seed", 42)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_subset, val_subset = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed)
    )
    train_sampler = SubsetRandomSampler(train_subset.indices)
    val_sampler = SubsetRandomSampler(val_subset.indices)
    loader_kw = {
        "batch_size": dataloader.batch_size,
        "collate_fn": dataloader.collate_fn,
        "prefetch_factor": dataloader.prefetch_factor,
        "num_streams": dataloader.num_streams,
        "use_streams": dataloader.use_streams,
    }
    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **loader_kw,
    )
    val_dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
        **loader_kw,
    )
    log.info(
        f"Train/val split: {n_train} train, {n_val} val (val_fraction={val_fraction}, seed={split_seed})"
    )

    h, w = int(cfg.target_size[0]), int(cfg.target_size[1])
    batch_size = int(cfg.dataloader.batch_size)
    spatial_positions = make_spatial_positions(
        h, w, batch_size=batch_size, device=dist.device, dtype=torch.get_default_dtype()
    )

    # spatial_positions: (B, H, W, 2). Expand/slice per step if batch size varies.
    log.info(
        f"Spatial positions grid shape {tuple(spatial_positions.shape)} on {dist.device}"
    )

    # Model (structured_shape from config matches target_size)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = hydra.utils.instantiate(model_cfg, _convert_="all").to(dist.device)

    def _compute_metrics(pred, y):
        loss = loss_fn(pred, y)
        with torch.no_grad():
            l2_err_sq = (pred - y).pow(2).sum()
            l2_ref_sq = y.pow(2).sum()
        return {"loss": loss, "l2_err_sq": l2_err_sq, "l2_ref_sq": l2_ref_sq}

    # Resolve forward call from config so the training loop never branches on
    # model type (isinstance); required for torch.compile.
    _model_target = OmegaConf.select(cfg, "model._target_", default="")
    model_name = _model_target.rsplit(".", 1)[-1] if _model_target else "unknown"
    if "geotransolver" in _model_target.lower():
        # Coming soon!
        def _forward(model, batch, positions):
            x = batch["x"].unsqueeze(-1)
            y = batch["y"].unsqueeze(-1)
            pred = model(local_embedding=positions, geometry=x)
            return _compute_metrics(pred, y), pred, x, y
    elif "transolver" in _model_target.lower():

        def _forward(model, batch, positions):
            x = batch["x"].unsqueeze(-1)
            y = batch["y"].unsqueeze(-1)
            pred = model(fx=x, embedding=positions)
            return _compute_metrics(pred, y), pred, x, y
    else:
        raise ValueError(
            f"Unsupported model _target_ {_model_target!r}; "
            "expected a class path containing 'Transolver' or 'GeoTransolver'."
        )

    tb_log_dir = f"runs/{model_name}_{h}x{w}"
    tb_tag = f"{model_name}_{h}x{w}"
    train_writer = SummaryWriter(log_dir=tb_log_dir + f"/{tb_tag}/train/")
    val_writer = SummaryWriter(log_dir=tb_log_dir + f"/{tb_tag}/val/")
    log.info(f"TensorBoard logging to {tb_log_dir}")

    if getattr(cfg.training, "compile", False):
        compile_mode = getattr(cfg.training, "compile_mode", "default")
        model = torch.compile(model, mode=compile_mode)
        log.info(f"Model compiled with mode={compile_mode}.")

    # Optimizer and scheduler
    opt_cfg = cfg.training.optimizer
    use_muon = getattr(cfg.training, "use_muon", False)
    if use_muon:
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        other_params = [p for p in model.parameters() if p.ndim != 2]
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizer = CombinedOptimizer(
            optimizers=[
                torch.optim.Muon(
                    muon_params,
                    lr=opt_cfg.lr,
                    weight_decay=weight_decay,
                ),
                torch.optim.Adam(
                    other_params,
                    lr=opt_cfg.lr,
                    weight_decay=weight_decay,
                ),
            ],
        )
        log.info("Using Muon for 2D params, Adam for rest.")
    elif opt_cfg.name == "Adam":
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt_cfg.lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    sch_cfg = cfg.training.scheduler
    if sch_cfg.name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.max_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )

    loss_fn = RelativeL2Loss()

    ckpt_args = {
        "path": f"./checkpoints/{model_name}",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": [model],
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    start_epoch = max(1, loaded_epoch + 1) if loaded_epoch else 1

    n_batches = len(train_dataloader)
    val_every = cfg.training.validation_every_epochs
    ckpt_every = cfg.training.checkpoint_every_epochs

    if start_epoch == 1:
        log.success("Training started...")
    else:
        log.warning(f"Resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        model.train()
        _zero = torch.tensor(0.0, device=dist.device)
        train_loss_sum = _zero.clone()
        train_l2_err_sq = _zero.clone()
        train_l2_ref_sq = _zero.clone()
        train_n = 0
        train_sample = None
        with LaunchLogger(
            "train", num_mini_batch=n_batches, epoch_alert_freq=1, epoch=epoch
        ) as logger:
            for batch, meta in train_dataloader:
                metrics, pred, x, y = _forward(model, batch, spatial_positions)

                optimizer.zero_grad()
                metrics["loss"].backward()
                optimizer.step()

                b = x.shape[0]
                train_loss_sum += metrics["loss"].detach() * b
                train_l2_err_sq += metrics["l2_err_sq"]
                train_l2_ref_sq += metrics["l2_ref_sq"]
                train_n += b
                train_sample = (x, y, pred)

                logger.log_minibatch({"loss": metrics["loss"].detach()})

            logger.log_epoch({"lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()

        if train_n > 0:
            avg_train_loss = (train_loss_sum / train_n).item()
            train_rel_l2 = (train_l2_err_sq.sqrt() / train_l2_ref_sq.sqrt()).item()
            train_writer.add_scalar(f"train/loss", avg_train_loss, epoch)
            train_writer.add_scalar(f"train/rel_l2", train_rel_l2, epoch)
        if train_sample is not None:
            _log_sample_images(train_writer, f"train", *train_sample, epoch)

        if epoch % val_every == 0:
            model.eval()
            val_loss_sum = _zero.clone()
            val_l2_err_sq = _zero.clone()
            val_l2_ref_sq = _zero.clone()
            val_n = 0
            val_sample = None
            with torch.no_grad():
                for batch, meta in val_dataloader:
                    metrics, pred, x, y = _forward(model, batch, spatial_positions)
                    b = x.shape[0]
                    val_loss_sum += metrics["loss"] * b
                    val_l2_err_sq += metrics["l2_err_sq"]
                    val_l2_ref_sq += metrics["l2_ref_sq"]
                    val_n += b
                    val_sample = (x, y, pred)
            if val_n > 0:
                avg_val_loss = (val_loss_sum / val_n).item()
                val_rel_l2 = (val_l2_err_sq.sqrt() / val_l2_ref_sq.sqrt()).item()
                val_writer.add_scalar(f"val/loss", avg_val_loss, epoch)
                val_writer.add_scalar(f"val/rel_l2", val_rel_l2, epoch)
                log.info(
                    f"Epoch {epoch} val_loss={avg_val_loss:.6f} val_rel_l2={val_rel_l2:.6f}"
                )
            if val_sample is not None:
                _log_sample_images(val_writer, f"val", *val_sample, epoch)
            model.train()

        if epoch % ckpt_every == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

    train_writer.close()
    val_writer.close()
    save_checkpoint(**ckpt_args, epoch=cfg.training.max_epochs)
    log.success("Training completed.")


if __name__ == "__main__":
    main()
