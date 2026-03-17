# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Train DA model for observation-to-state regression.

This is the training script for the HealDA data assimilation model.
It trains a ViT model to regress atmospheric state from observations.

Usage:
    python train_da.py --name era5-v2-dense-noInfill-10M-fusion512-lrObs1e-4
"""

import dataclasses
import functools
import json
import logging
import os
import warnings

import config.environment as config
import matplotlib.pyplot as plt
import models
import torch
import torch.utils
import torch.utils.data
import training.loop
from datasets.base import BatchInfo, TimeUnit, VariableConfig
from datasets.dataset import (
    VARIABLE_CONFIGS,
    get_batch_info,
    get_dataset,
    get_sensors_for_config,
)
from datasets.prefetch_map import prefetch_map
from datasets.round_robin import RoundRobinLoader
from datasets.samplers import (
    ChunkedDistributedSampler,
)
from datasets.sensors import (
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
)
from datasets.transform import TransformV2, collate
from physicsnemo.utils import load_checkpoint
from training import loop
from utils import distributed as dist
from utils.dataclass_parser import parse_args, parse_dict
from utils.signals import finish_before_quitting
from utils.visualizations import visualize

from config.model_config import ModelConfigV1, ObsConfig
from utils import profiling

logger = logging.getLogger(__name__)


def build_sensor_lists(
    sensor_names: list[str],
) -> tuple[list[int], list[int], list[str]]:
    """Build list-based sensor config from sensor names.

    Returns:
        (nchannel_per_sensor, nplatform_per_sensor, sensor_names)
    """
    nchannel_per_sensor = []
    nplatform_per_sensor = []
    for name in sensor_names:
        nchannel_per_sensor.append(SENSOR_CONFIGS[name].channels)
        nplatform_per_sensor.append(max(len(SENSOR_CONFIGS[name].platforms), 1))
    return nchannel_per_sensor, nplatform_per_sensor, list(sensor_names)


@dataclasses.dataclass
class DistributedConfig:
    rank: int
    world_size: int


@dataclasses.dataclass
class TrainingLoop(loop.TrainingLoopBase):
    """
    Training loop for observation-to-state DA model.

    valid_samples_per_season: the number of samples to use when making season
        average plots
    """

    valid_min_samples: int = 128

    # loss options
    loss_type: str = "mse"
    huber_delta: float = 0.1

    # data loader options
    dataloader_num_workers: int = 3
    dataloader_prefetch_factor: int = 8
    prefetch_to_gpu: bool = True

    label_dropout: float = 0.0
    era5_chunk_size: int = 48
    start_year: int = -1  # Filter training data to >= this year
    obs_config: ObsConfig = ObsConfig()

    # model configuration
    opt: str = "adamw"
    adam_eps: float = 1e-8
    adam_beta2: float = 0.95
    lr_obs: float = 1e-4
    weight_decay: float = 0.1
    weight_decay_biases: bool = True
    drop_path: float = 0.0
    p_dropout: float = 0.0
    architecture: str = "dit-l_reg_hpx6_per_sensor"
    as_vit: bool = False
    gradient_checkpointing: bool = False
    dit_qk_rms_norm: bool = False
    # Original model was trained with the HealDA defaults (4*hidden_size /
    # hidden_size = 4096/1024), but this puts >60% of total parameters in
    # the adaLN modulation MLPs.  Setting both to 128 yields equally fast
    # convergence and comparable skill at a fraction of the parameter cost.
    emb_channels: int | None = None
    noise_channels: int | None = None

    # When True, apply a custom gradient clipping schedule that
    # linearly decays the clip value from 1.0 → 0.015 over the
    # first 50k images, then keeps it at 0.015 afterwards.
    use_gradient_clip_schedule: bool = False

    # Obs embedder settings
    embed_dim: int = 32
    meta_dim: int = 28
    fusion_dim: int = 512
    freeze_obs_embed: bool = False
    freeze_transformer_blocks: bool = False
    freeze_decoder: bool = False
    freeze_pos_embedding: bool = False

    # change defaults for parameter norm logging
    log_parameter_norm: bool = False
    log_parameter_grad_norm: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._train_sampler = None
        self._test_sampler = None

    @property
    def variable_config(self) -> VariableConfig:
        return VARIABLE_CONFIGS["era5"]

    @functools.cached_property
    def batch_info(self) -> BatchInfo:
        return get_batch_info(
            config=self.variable_config,
            time_unit=TimeUnit.HOUR,
        )

    def resume_from_state(self, resume_state_dump, optimizer=True, wandb=False):
        super().resume_from_state(resume_state_dump, optimizer, wandb=wandb)
        self._load_wandb_id()
        dist.print0(f"Loaded checkpoint from {resume_state_dump}.")

    def _save_wandb_id(self):
        if self.wandb_id is not None:
            with open(os.path.join(self.run_dir, "wandb_id"), "w") as f:
                f.write(self.wandb_id)

    def _load_wandb_id(self):
        try:
            with open(os.path.join(self.run_dir, "wandb_id")) as f:
                self.wandb_id = f.read()
        except FileNotFoundError:
            pass

    def save_training_state(self):
        if dist.get_rank() != 0:
            return
        super().save_training_state()
        self._save_wandb_id()

    def save_network_snapshot(self):
        if dist.get_rank() != 0:
            return
        super().save_network_snapshot()

    @property
    def out_channels(self):
        return len(self.batch_info.channels)

    def setup(self):
        super().setup()
        self.net.gradient_checkpointing = self.gradient_checkpointing

    @profiling.nvtx
    @finish_before_quitting
    def step_optimizer(self, cur_nimg):
        """Optionally apply a scheduled gradient clipping value, then
        delegate to the base implementation for LR scheduling and optimizer step.
        """
        if self.use_gradient_clip_schedule:
            start_clip = 1.0
            end_clip = 0.015
            schedule_end = 50_000

            n = max(0, min(cur_nimg, schedule_end))
            if schedule_end > 0:
                frac = n / schedule_end
            else:
                frac = 1.0

            self.gradient_clip_max_norm = start_clip + (end_clip - start_clip) * frac

        super().step_optimizer(cur_nimg)

    @functools.cached_property
    def _data_transform(self):
        """Get the appropriate data transform."""
        return TransformV2(
            variable_config=self.variable_config,
            sensors=self._sensor_names,
        )

    @functools.cached_property
    def _sensor_names(self) -> list[str]:
        return get_sensors_for_config(self.obs_config)

    def get_dataset(self, train: bool):
        """Returns the dataset for training or validation."""
        return get_dataset(
            dataset="era5",
            split="train" if train else "test",
            transform=None,
            batch_transform=self._data_transform.transform,
            rank=self.distributed_config.rank,
            world_size=self.distributed_config.world_size,
            infinite=True,
            shuffle=True,
            chunk_size=self.era5_chunk_size,
            obs_config=self.obs_config,
            start_year=self.start_year,
            map_style=True,
        )

    @property
    def distributed_config(self) -> DistributedConfig:
        return DistributedConfig(dist.get_rank(), dist.get_world_size())

    def _create_dataloader(
        self,
        dataset,
        sampler,
        batch_size,
        num_workers=None,
        prefetch_factor=None,
        pin_memory=True,
    ):
        """Helper to create a DataLoader with common settings."""
        if num_workers is None:
            num_workers = self.dataloader_num_workers

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            in_order=True,
        )

    def _get_loader(self, dataset, batch_size, train: bool = True):
        workers = self.dataloader_num_workers
        prefetch_factor = self.dataloader_prefetch_factor
        if not train and workers != 0:
            workers = 1
            prefetch_factor = 4

        if isinstance(dataset, torch.utils.data.IterableDataset):
            # Iterable datasets don't use samplers
            loader = self._create_dataloader(
                dataset, sampler=None, batch_size=batch_size
            )

        else:
            # Round-robin loader: one dataloader per worker, each with its own chunk assignment.
            # This is optimal for chunked zarr data where sequential access within chunks is fast.
            #
            # Note: Currently restarts with same sample order every time (default seed).
            # If we drop chunking in the future, switch to RestartableDistributedSampler
            # with proper checkpointing for reproducible training restarts.
            num_loaders = max(workers, 1)
            dataloaders = []

            for worker_id in range(num_loaders):
                worker_sampler = ChunkedDistributedSampler(
                    dataset,
                    chunk_size=self.era5_chunk_size,
                    num_replicas=self.distributed_config.world_size * num_loaders,
                    rank=self.distributed_config.rank * num_loaders + worker_id,
                    shuffle=True,
                    shuffle_within_chunk=True,
                    drop_last=True,
                )

                worker_loader = self._create_dataloader(
                    dataset,
                    sampler=worker_sampler,
                    batch_size=batch_size,
                    num_workers=1,  # single worker per loader
                    prefetch_factor=prefetch_factor,
                )
                dataloaders.append(worker_loader)
            loader = RoundRobinLoader(dataloaders)

        # transferring the obs data from cpu -> gpu can be slow, so
        # running it in a separate thread using prefetch_map improves utilization
        if self.prefetch_to_gpu:
            loader = prefetch_map(loader, self._device_transform, queue_size=2)
        return loader

    def _device_transform(self, batch):
        """Transformations to occur on device in a separate thread. including device movement"""
        return self._data_transform.device_transform(batch, device=self.device)

    def _stage_dict_batch(self, batch):
        if isinstance(batch.get("obs_table"), tuple):
            return self._device_transform(batch)
        return super()._stage_dict_batch(batch)

    def get_data_loaders(self, batch_gpu):
        """Create train and test DataLoaders"""
        dataset = self.get_dataset(train=True)
        train_loader = self._get_loader(dataset, batch_size=batch_gpu, train=True)
        test_dataset = self.get_dataset(train=False)
        test_loader = self._get_loader(test_dataset, batch_size=batch_gpu, train=False)

        self._test_dataset = test_dataset
        return dataset, train_loader, test_loader

    def _step(
        self,
        *,
        train=True,
        plot_image=False,
        target: torch.Tensor,
        condition,
        second_of_day,
        day_of_year,
        obs,
        labels=None,
        return_both=False,
        timestamp=None,
        **batch,
    ):
        b, c, t, x = target.shape
        noise_labels = torch.zeros([b], device=target.device)

        pred = self.ddp(
            condition,
            noise_labels,
            **obs,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            class_labels=labels,
        )

        train_tag = "train" if train else "test"

        # log per channel norm of training target and prediction
        for c in range(len(self.batch_info.channels)):
            channel = self.batch_info.channels[c]
            self.log_metric(f"norm/{channel}/target", target[:, c].norm(), print=False)
            self.log_metric(
                f"norm/{channel}/pred_{train_tag}", pred[:, c].norm(), print=False
            )
            self.log_metric(
                f"max/{channel}/target", target[:, c].abs().max(), print=False
            )
            self.log_metric(
                f"max/{channel}/pred_{train_tag}", pred[:, c].abs().max(), print=False
            )

        mse = (target - pred) ** 2
        huber_loss = torch.nn.functional.huber_loss(
            target, pred, reduction="none", delta=self.huber_delta
        )

        self.log_metric(f"Loss/{train_tag}_mse", mse)
        self.log_metric(f"Loss/{train_tag}_huber", huber_loss)

        scales = torch.as_tensor(self.batch_info.scales)[:, None, None].to(self.device)
        centers = torch.as_tensor(self.batch_info.center)[:, None, None].to(self.device)

        metrics_pred = pred * scales + centers
        metrics_target = target * scales + centers
        # Compute MSE for logging purposes in physical units
        full_mse_physical = (metrics_target - metrics_pred) ** 2

        for c in range(len(self.batch_info.channels)):
            channel = self.batch_info.channels[c]
            this_rmse = torch.sqrt(full_mse_physical[:, c, -1].mean())
            self.log_metric(f"rmse/{channel}/{train_tag}", this_rmse)
            self.log_metric(
                f"huber/{channel}/{train_tag}",
                huber_loss[:, c, -1].mean(),
                print=False,
            )

            if plot_image and dist.get_rank() == 0:
                for name, field in zip(
                    ["prediction", "target"], [metrics_pred, metrics_target]
                ):
                    fig = plt.figure()
                    display_field = field[0, c, -1].cpu()
                    visualize(
                        display_field,
                        hpxpad=True,
                        title=channel,
                    )
                    self.writer.add_figure(
                        f"sample/{channel}/{name}", fig, global_step=self.cur_nimg
                    )
        loss = mse if self.loss_type == "mse" else huber_loss

        if train:
            self.log_metric("loss", loss, frequency="step")

        if return_both:
            return mse, huber_loss
        else:
            return loss

    def train_step(self, **batch):
        return self._step(train=True, **batch)

    def test_step(self, **batch):
        return self._step(train=False, **batch)

    @classmethod
    def loads(cls, s):
        return parse_dict(cls, json.loads(s))

    @property
    def model_config(self) -> ModelConfigV1:
        condition_channels = 2  # orog and lfrac static variables
        nchannel_per_sensor, nplatform_per_sensor, sensor_names_list = (
            build_sensor_lists(self._sensor_names)
        )

        return models.ModelConfigV1(
            architecture=self.architecture,
            condition_channels=condition_channels,
            out_channels=self.out_channels,
            label_dim=0,
            label_dropout=self.label_dropout,
            obs_config=self.obs_config,
            p_dropout=self.p_dropout,
            drop_path=self.drop_path,
            nchannel_per_sensor=nchannel_per_sensor,
            nplatform_per_sensor=nplatform_per_sensor,
            sensor_names=sensor_names_list,
            embed_dim=self.embed_dim,
            meta_dim=self.meta_dim,
            fusion_dim=self.fusion_dim,
            qk_rms_norm=self.dit_qk_rms_norm,
            as_vit=self.as_vit,
            emb_channels=self.emb_channels,
            noise_channels=self.noise_channels,
        )

    def _setup_networks(self):
        torch.manual_seed(self.seed)
        net = self.get_network()
        net.train()
        net.requires_grad_(True)
        net.to(self.device)

        if self.freeze_obs_embed:
            net.obs_embedder.requires_grad_(False)

        if self.freeze_transformer_blocks:
            for block in net.dit.blocks:
                block.requires_grad_(False)

        if self.freeze_pos_embedding:
            net.dit.tokenizer.requires_grad_(False)

        if self.freeze_decoder:
            net.dit.detokenizer.requires_grad_(False)

        self.net = net
        if dist.get_world_size() > 1:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device],
                broadcast_buffers=False,
            )
        else:
            self.ddp = self.net

    def get_optimizer(self, named_parameters):
        """Builds optimizer, applying differential learning rate to observation embeddings and transformer blocks"""
        named_params = list(named_parameters)
        # Separate the obs embedding and transformer parameters to apply
        # lower learning rate to the obs embedding
        obs_param_prefix = "obs_embedder."

        def _get_param_groups(params, lr):
            if self.weight_decay_biases:
                return ({"params": params, "lr": lr, "base_lr": lr},)

            weights, biases = [], []

            for param in params:
                if param.ndim > 1:
                    weights.append(param)
                else:
                    biases.append(param)

            return [
                {
                    "params": weights,
                    "lr": lr,
                    "base_lr": lr,
                    "weight_decay": self.weight_decay,
                },
                {"params": biases, "lr": lr, "base_lr": lr, "weight_decay": 0.0},
            ]

        xfmr_params, obs_params = [], []
        for name, param in named_params:
            if name.startswith(obs_param_prefix):
                obs_params.append(param)
            else:
                xfmr_params.append(param)

        param_groups = []
        if xfmr_params:
            param_groups.extend(_get_param_groups(xfmr_params, self.lr))
        if obs_params and not self.freeze_obs_embed:
            param_groups.extend(_get_param_groups(obs_params, self.lr_obs))

        if self.opt == "adamw":
            return torch.optim.AdamW(
                param_groups,
                betas=(0.9, self.adam_beta2),
                eps=self.adam_eps,
                weight_decay=self.weight_decay,
                fused=True,
            )
        else:
            return torch.optim.Adam(
                param_groups,
                betas=(0.9, self.adam_beta2),
                eps=self.adam_eps,
                fused=True,
            )

    def get_loss_fn(self):
        """Return loss function."""
        return None

    @staticmethod
    def print_network_info(net, device):
        num_params = sum(p.numel() for p in net.parameters())
        dist.print0(f"Number of parameters: {num_params}. Network: {net}")

    def validate(self, net=None):
        if net is None:
            net = self.net
        net.eval()

        for batch_num, batch in enumerate(self.valid_loader):
            if batch_num * self.batch_size >= self.valid_min_samples:
                break
            batch = self._stage_dict_batch(batch)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
                ):
                    self.test_step(plot_image=batch_num == 0, return_both=True, **batch)


@dataclasses.dataclass
class CLI:
    """Command-line interface config.

    Attributes
    ----------
    name : str
        Preset name (selects from ``LOOPS`` dict) or empty for custom config.
    output_dir : str
        Root directory for run outputs (checkpoints, logs).
    resume_dir : str
        Path to an existing run directory to resume training from. The code
        looks for a ``checkpoints_training_state/`` subdirectory and restores
        model weights, optimizer state, and training progress.
    finetune_from : str
        Path to a ``.mdlus`` file to initialize model weights from. Unlike
        ``resume_dir``, this starts a fresh optimizer and resets training
        progress -- only the model weights are loaded.
    """

    name: str = ""
    output_dir: str = config.CHECKPOINT_ROOT
    resume_dir: str = ""
    finetune_from: str = ""
    loop: TrainingLoop = dataclasses.field(
        default_factory=lambda: TrainingLoop(
            architecture="dit-l_reg_hpx6_per_sensor",
            batch_size=8,
            batch_gpu=1,
            lr=0.0005,
            lr_obs=0.0001,
            lr_rampup_img=50000,
            flat_imgs=0,
            decay_imgs=10000000,
            lr_min=0.0,
            gradient_clip_max_norm=1.0,
            steps_per_tick=2500,
            snapshot_ticks=100,
            state_dump_ticks=2,
            print_steps=50,
            loss_type="huber",
            loss_reduction="v1",
            huber_delta=0.1,
            dataloader_num_workers=5,
            dataloader_prefetch_factor=12,
            total_ticks=250,
            era5_chunk_size=24,
            weight_decay=0.05,
            drop_path=0.1,
            p_dropout=0.05,
            obs_config=ObsConfig(
                use_obs=True,
                context_start=-21,
                context_end=3,
                use_conv=True,
                conv_uv_in_situ_only=False,
                conv_gps_level1_only=False,
            ),
            dit_qk_rms_norm=True,
            embed_dim=32,
            fusion_dim=512,
        )
    )


warnings.filterwarnings(action="ignore", message="Cannot do a zero-copy NCHW to NHWC.")


LOOPS = {}

# HealDA v1 configuration: ERA5 observation-to-state training
LOOPS["era5-v2-dense-noInfill-10M-fusion512-lrObs1e-4"] = TrainingLoop(
    architecture="dit-l_reg_hpx6_per_sensor",
    batch_size=8,
    batch_gpu=1,
    lr=0.0005,
    lr_obs=0.0001,
    lr_rampup_img=50000,
    flat_imgs=0,
    decay_imgs=10000000,
    lr_min=0.0,
    gradient_clip_max_norm=1.0,
    steps_per_tick=2500,
    snapshot_ticks=100,
    state_dump_ticks=2,
    print_steps=1,
    loss_type="huber",
    loss_reduction="v1",
    huber_delta=0.1,
    dataloader_num_workers=5,
    dataloader_prefetch_factor=12,
    total_ticks=250,
    era5_chunk_size=24,
    weight_decay=0.05,
    drop_path=0.1,
    p_dropout=0.05,
    obs_config=ObsConfig(
        use_obs=True,
        context_start=-21,
        context_end=3,
        use_conv=True,
        conv_uv_in_situ_only=False,
        conv_gps_level1_only=False,
    ),
    dit_qk_rms_norm=True,
    embed_dim=32,
    fusion_dim=512,
)


def main():
    cli = parse_args(CLI, convert_underscore_to_hyphen=False)
    dist.init()

    if dist.get_rank() == 0:
        logging.basicConfig(level=logging.INFO)
        training.loop.logger.setLevel(level=logging.DEBUG)

    try:
        dist.print0(f"Using {cli.name=} preset.")
        loop = LOOPS[cli.name]
    except KeyError:
        dist.print0("Using --loop command line arguments")
        loop = cli.loop

    loop.run_dir = os.path.join(cli.output_dir, cli.name)
    loop.setup()
    dist.print0("Training with:", loop)

    if dist.get_rank() == 0:
        config.print_config()

    # Three mutually exclusive initialization paths:
    #
    # 1. Resume: load from a checkpoint *directory* (model weights + optimizer
    #    + training progress).  Used to continue an interrupted run.
    # 2. Finetune: load from a single .mdlus *file* (model weights only).
    #    Optimizer and nimg start fresh.
    # 3. From scratch: random initialization, nothing to load.
    resumed = False
    for rundir in [loop.run_dir, cli.resume_dir]:
        if not rundir:
            continue
        handler = training.loop.CheckpointHandler(rundir)
        result = handler.latest_checkpoint()
        if result is not None:
            checkpoint_dir, nimg, _ = result
            loop.cur_nimg = nimg
            loop.resume_from_state(checkpoint_dir, wandb=True)
            resumed = True
            break

    if not resumed:
        loop.wandb_id = None
        if cli.finetune_from:
            dist.print0(f'Loading pretrained weights from "{cli.finetune_from}"')
            loop.net.load(cli.finetune_from)
        dist.print0("Starting new training")

    loop.setup_wandb(name=cli.name)
    loop.train()


if __name__ == "__main__":
    main()
