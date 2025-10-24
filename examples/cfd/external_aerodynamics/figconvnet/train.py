# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from functools import partial
import logging
import logging.config
import os
from timeit import default_timer

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed
import torch.utils
import torch.utils.data
import torchinfo

import warp as wp

from physicsnemo.distributed import DistributedManager

from src.utils import rank0
from src.utils.average_meter import AverageMeter, AverageMeterDict, Timer
from src.utils.loggers import init_logger
from src.utils.seed import set_seed
from src.utils.signal_handlers import SignalHandler
from physicsnemo.models.figconvnet.geometries import GridFeaturesMemoryFormat


logger = logging.getLogger("figconv")


def _delete_previous_checkpoints(config):
    checkpoints_to_delete = []
    for f in os.listdir(config.output):
        if f.startswith("model_") and f.endswith(".pth"):
            checkpoints_to_delete.append(f)
    checkpoints_to_delete.sort()
    checkpoints_to_delete = checkpoints_to_delete[: -config.train.num_checkpoints]
    logger.info(f"Deleting {len(checkpoints_to_delete)} checkpoints")
    for f in checkpoints_to_delete:
        try:
            os.remove(os.path.join(config.output, f))
        except FileNotFoundError:
            pass


@rank0
def _save_state(model, optimizer, scheduler, scaler, epoch, tot_iter, config):
    save_path = os.path.join(config.output, f"model_{epoch:05d}.pth")
    logger.info(f"Saving model at epoch {epoch} to {save_path}")
    state_dict = {
        "model": model.model().state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "tot_iter": tot_iter,
    }
    # Save the file with 0000X format
    torch.save(state_dict, save_path)
    _delete_previous_checkpoints(config)


def _resume_from_checkpoint(model, optimizer, scheduler, scaler, config):
    logger.info(f"Resuming from {config.output}")

    # Find the latest checkpoint
    checkpoints = []
    for f in os.listdir(config.output):
        if f.startswith("model_") and f.endswith(".pth"):
            checkpoints.append(f)
    checkpoints.sort()

    start_epoch = 0
    tot_iter = 0
    # Load if there is a checkpoint
    if len(checkpoints) == 0:
        logger.info("No checkpoints found")
    else:
        logger.info(f"Found {len(checkpoints)} checkpoints")
        logger.info(f"Loading {checkpoints[-1]}")
        checkpoint_path = os.path.join(config.output, checkpoints[-1])
        # Get rank if distributed
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.model().load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        tot_iter = checkpoint["tot_iter"]

        # Wait until all processes load the checkpoint.
        if DistributedManager().distributed:
            torch.distributed.barrier()

    return start_epoch, tot_iter


@torch.no_grad()
def eval(model, datamodule, config, loss_fn=None):
    model.eval()
    test_loader = datamodule.test_dataloader(
        batch_size=config.eval.batch_size, **config.eval.dataloader
    )
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    eval_timer = Timer()
    for i, data_dict in enumerate(test_loader):
        eval_timer.tic()
        out_dict = model.eval_dict(data_dict, loss_fn=loss_fn, datamodule=datamodule)
        out_dict["inference_time"] = eval_timer.toc()
        eval_meter.update(out_dict)
        if i % config.eval.plot_interval == 0:
            visualize_data_dicts.append(data_dict)
        if i % config.eval.print_interval == 0:
            # Print eval dict
            print_str = f"Eval {i}: "
            for k, v in eval_meter.avg.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if isinstance(v, float):
                    print_str += f"{k}: {v:.4f}, "
                else:
                    print_str += f"{k}: {v}, "
            logger.info(print_str)

    # Merge all dictionaries
    merged_image_dict = {}
    merged_point_cloud_dict = {}
    if hasattr(model, "image_pointcloud_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_pointcloud_dict(
                data_dict, datamodule=datamodule
            )
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v
            for k, v in pointcloud_dict.items():
                merged_point_cloud_dict[f"{k}_{i}"] = v
    elif hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_dict(
                data_dict, datamodule=datamodule
            )
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    # Aggregate all counts, sums, avgs, and private attributes if distributed
    eval_meter.all_gather_attributes()

    eval_dict = eval_meter.avg

    # Post process the eval dict
    if hasattr(model, "post_eval_epoch"):
        (
            eval_dict,
            merged_image_dict,
            merged_point_cloud_dict,
        ) = model.post_eval_epoch(
            eval_dict,
            merged_image_dict,
            merged_point_cloud_dict,
            eval_meter._private_attributes,
            datamodule,
        )

    model.train()
    return eval_dict, merged_image_dict, merged_point_cloud_dict


def train(config: DictConfig, signal_handler: SignalHandler):
    dist = DistributedManager()

    # Initialize the device. Allow device override only in non-distributed setting.
    device = dist.device if dist.distributed else torch.device(config.device)
    # Set default devices.
    torch.cuda.device(device)
    wp.init()
    wp.set_device(str(device))

    loggers = init_logger(config)
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(config, sort_keys=True)}")

    # Initialize the model
    model = instantiate(config.model)
    model = model.to(device)
    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0)}\n")

    # Enable DDP.
    if dist.distributed:
        # TODO(akamenev): make broadcast_buffers configurable
        # since some of the models use BatchNorm.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.device],
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
        logger.info("Initialized DDP.")
    # Set the original model getter to simplify access.
    assert not hasattr(model, "model")
    type(model).model = (lambda m: m.module) if dist.distributed else (lambda m: m)

    # Initialize the dataloaders
    datamodule = instantiate(config.data)
    train_loader = datamodule.train_dataloader(
        batch_size=config.train.batch_size, **config.train.dataloader
    )

    # Initialize the optimizer and scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer)

    # Initialize the loss function.
    loss_fn = instantiate(config.loss)
    if config.eval.loss is None:
        eval_loss_fn = loss_fn
    else:
        eval_loss_fn = instantiate(config.eval.loss)

    # Initialize AMP.
    scaler = instantiate(config.amp.scaler)
    autocast = partial(
        torch.cuda.amp.autocast,
        enabled=config.amp.enabled,
        dtype=hydra.utils.get_object(config.amp.autocast.dtype),
    )

    # Resume if resume is True
    start_epoch = 0
    tot_iter = 0
    if config.train.resume and os.path.exists(config.output):
        start_epoch, tot_iter = _resume_from_checkpoint(
            model, optimizer, scheduler, scaler, config
        )

    # Eval first for debugging
    if config.eval.run_eval_first and start_epoch == 0:
        eval_dict, eval_images, eval_point_clouds = eval(
            model.model(), datamodule, config, eval_loss_fn
        )
        for k, v in eval_dict.items():
            logger.info(f"First Eval: {k}: {v:.4f}")

    for ep in range(start_epoch, config.train.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        datamodule.set_epoch(train_loader, ep)

        for data_dict in train_loader:
            # Check if the signal is received
            if signal_handler.is_stopped():
                logger.debug("Signal received. Breaking the training loop.")
                break

            optimizer.zero_grad()

            with autocast():
                loss_dict = model.model().loss_dict(
                    data_dict, loss_fn=loss_fn, datamodule=datamodule
                )

            loss = 0
            for k, v in loss_dict.items():
                v = v * getattr(config, k + "_weight", 1)
                loss = loss + v.mean()

            # Assert loss is valid
            assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

            # Note: if AMP is disabled, the scaler will fall back to the default behavior.
            scaler.scale(loss).backward()

            # TODO(akamenev): grad clipping can be used not only in AMP.
            if config.amp.clip_grad:
                # Unscales the gradients of optimizer's assigned params in-place.
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.amp.grad_max_norm
                )

            # If optimizer's gradients were already unscaled, the scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            train_l2_meter.update(loss.item())
            loggers.log_scalar("train/iter_lr", scheduler.get_last_lr()[0], tot_iter)
            loggers.log_scalar("train/iter_loss", loss.item(), tot_iter)
            for k, v in loss_dict.items():
                loggers.log_scalar(f"train/{k}", v.item(), tot_iter)
            if tot_iter % config.train.print_interval == 0:
                print_str = f"Iter {tot_iter} loss: {loss.item():.4f}, "
                for k, v in loss_dict.items():
                    print_str += f"{k}: {v.item():.4f}, "  # only print the number
                logger.info(print_str)

            if config.train.lr_scheduler_mode == "iteration":
                scheduler.step()
            tot_iter += 1
            torch.cuda.empty_cache()

        if config.train.lr_scheduler_mode == "epoch":
            scheduler.step()
        t2 = default_timer()
        logger.info(
            f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}"
        )
        loggers.log_scalar("train/epoch_train_l2", train_l2_meter.avg, tot_iter)
        loggers.log_scalar("train/train_epoch_duration", t2 - t1, tot_iter)

        if (
            ep % config.eval.interval == 0
            or ep == config.train.num_epochs - 1
            and (not signal_handler.is_stopped())
        ):
            eval_dict, eval_images, eval_point_clouds = eval(
                model.model(), datamodule, config, eval_loss_fn
            )
            for k, v in eval_dict.items():
                logger.info(f"Epoch: {ep} {k}: {v:.4f}")
                loggers.log_scalar(f"eval/{k}", v, tot_iter)
            for k, v in eval_images.items():
                loggers.log_image(f"eval_vis/{k}", v, tot_iter)
            if config.log_pointcloud:
                for k, v in eval_point_clouds.items():
                    loggers.log_pointcloud(
                        f"eval_vis/{k}", v[..., :3], v[..., 3:], tot_iter
                    )

        # Save the weights, optimization state, and scheduler state into one file
        if ep % config.train.save_interval == 0 or signal_handler.is_stopped():
            # save the model
            _save_state(model, optimizer, scheduler, scaler, ep, tot_iter, config)

        # Exit the training loop if the signal handler is stopped.
        if signal_handler.is_stopped():
            break

    # Save the final model if the training loop was not stopped by the signal handler.
    if not signal_handler.is_stopped():
        _save_state(
            model,
            optimizer,
            scheduler,
            scaler,
            config.train.num_epochs - 1,
            tot_iter,
            config,
        )


def _slurm_setup(config: DictConfig) -> None:
    # Hydra config contains properly resolved absolute path.
    config.output = HydraConfig.get().runtime.output_dir

    # Detect if it is running on a SLURM cluster.
    if "SLURM_JOB_ID" in os.environ:
        # The output directory is set to simply ${output}/SLURM_JOB_ID.
        # config.output = os.path.join(config.output, os.environ["SLURM_JOB_ID"])
        # Check for the checkpoints and model_*.pth files in the output directory.
        if os.path.exists(config.output) and any(
            f.startswith("model_") and f.endswith(".pth")
            for f in os.listdir(config.output)
        ):
            config.train.resume = True


def _init_python_logging(config: DictConfig) -> None:
    if config.log_dir is None:
        config.log_dir = config.output
    else:
        config.log_dir = to_absolute_path(config.log_dir)

    # Make the log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # Set up Python loggers.
    if pylog_cfg := OmegaConf.select(config, "logging.python"):
        pylog_cfg.output = config.output
        pylog_cfg.rank = DistributedManager().rank
        # Enable logging only on rank 0, if requested.
        if pylog_cfg.rank0_only and pylog_cfg.rank != 0:
            pylog_cfg.handlers = {}
            pylog_cfg.loggers.figconv.handlers = []
        # Configure logging.
        logging.config.dictConfig(OmegaConf.to_container(pylog_cfg, resolve=True))


@hydra.main(version_base="1.3", config_path="configs", config_name="base")
def main(config: DictConfig):
    _slurm_setup(config)

    _init_python_logging(config)

    # Set the random seed.
    if config.seed is not None:
        set_seed(config.seed)

    with SignalHandler(status_path=config.signal_handler.status_path) as signal_handler:
        train(config, signal_handler)


def _init_hydra_resolvers():
    def res_mem_pair(
        fmt: str, dims: list[int, int, int]
    ) -> tuple[GridFeaturesMemoryFormat, tuple[int, int, int]]:
        return getattr(GridFeaturesMemoryFormat, fmt), tuple(dims)

    OmegaConf.register_new_resolver("res_mem_pair", res_mem_pair)


if __name__ == "__main__":
    DistributedManager.initialize()

    _init_hydra_resolvers()

    main()
