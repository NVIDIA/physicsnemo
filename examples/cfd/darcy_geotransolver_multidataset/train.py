# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Training entrypoint for Darcy Transolver multi-dataset.
# Uses MultiDataset dataloader and Transolver with spatial shape matching
# target_size for permeability -> pressure. Muon used for 2D params when available.

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import L1Loss, MSELoss

# This is needed for datapipe registry via hydra instantiation
from physicsnemo import datapipes

from physicsnemo.distributed import DistributedManager
from physicsnemo.optim import CombinedOptimizer
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger

# Muon is available in PyTorch >= 2.9
_Muon = getattr(torch.optim, "Muon", None)


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


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="darcy_transolver_multidataset")
    log.file_logging()

    # Multi-dataset Data Loader Instantiated from hydra:
    dataloader = hydra.utils.instantiate(cfg.dataloader)

    h, w = int(cfg.target_size[0]), int(cfg.target_size[1])
    batch_size = int(cfg.dataloader.batch_size)
    spatial_positions = make_spatial_positions(
        h, w, batch_size=batch_size, device=dist.device, dtype=torch.get_default_dtype()
    )
    
    # spatial_positions: (B, H, W, 2). Expand/slice per step if batch size varies.
    log.info(f"Spatial positions grid shape {tuple(spatial_positions.shape)} on {dist.device}")

    # Model (structured_shape from config matches target_size)
    model = hydra.utils.instantiate(cfg.model).to(dist.device)

    # Resolve forward call from config so the training loop never branches on
    # model type (isinstance); required for torch.compile.
    _model_target = OmegaConf.select(cfg, "model._target_", default="")
    if "geotransolver" in _model_target.lower():
        def _forward(model, x, positions):
            return model(local_embedding=x, geometry=positions)
    elif "transolver" in _model_target.lower():
        def _forward(model, x, positions):
            return model(fx=x, embedding=positions)
    else:
        raise ValueError(
            f"Unsupported model _target_ {_model_target!r}; "
            "expected a class path containing 'Transolver' or 'GeoTransolver'."
        )

    if getattr(cfg.training, "compile", False):
        compile_mode = getattr(cfg.training, "compile_mode", "default")
        model = torch.compile(model, mode=compile_mode)
        log.info(f"Model compiled with mode={compile_mode}.")

    # Optimizer and scheduler
    opt_cfg = cfg.training.optimizer
    use_muon = getattr(cfg.training, "use_muon", False) and _Muon is not None
    if getattr(cfg.training, "use_muon", False) and _Muon is None:
        log.warning("use_muon=true but torch.optim.Muon not available (pytorch>=2.9); using Adam.")
    if use_muon:
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        other_params = [p for p in model.parameters() if p.ndim != 2]
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizer = CombinedOptimizer(
            optimizers=[
                _Muon(
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    loss_name = getattr(cfg.training, "loss", "mse").lower()
    loss_fn = MSELoss() if loss_name == "mse" else L1Loss()

    ckpt_args = {
        "path": "./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    start_epoch = max(1, loaded_epoch + 1) if loaded_epoch else 1

    n_batches = len(dataloader)
    val_every = cfg.training.validation_every_epochs
    ckpt_every = cfg.training.checkpoint_every_epochs

    if start_epoch == 1:
        log.success("Training started...")
    else:
        log.warning(f"Resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        model.train()
        with LaunchLogger("train", num_mini_batch=n_batches, epoch_alert_freq=1, epoch=epoch) as logger:
            for batch, meta in dataloader:
                x, y = batch['x'], batch['y']
                
                x = x.unsqueeze(-1)
                y = y.unsqueeze(-1)

                # Match batch size in case last batch is smaller (e.g. drop_last=False)
                b = x.shape[0]

                pred = _forward(model, x, spatial_positions)

                # pred (B, H, W, 1); y (B, H, W, 1) for loss
                
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.log_minibatch({"loss": loss.detach()})

            logger.log_epoch({"lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()

        if epoch % ckpt_every == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_epochs)
    log.success("Training completed.")


if __name__ == "__main__":
    main()
