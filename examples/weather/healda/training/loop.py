# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import abc
import contextlib
import dataclasses
import gc
import glob
import itertools
import json
import logging
import os
import re
import shutil
import signal
import time
import warnings
from functools import partial
from typing import Iterable, Union

import models
import numpy as np
import psutil
import torch
import torch.utils.tensorboard
import torchmetrics
from config.training import loop
from datasets.base import BatchInfo, SpatioTemporalDataset
from utils import checkpointing
from utils import distributed as dist
from utils.signals import QuitEarly, finish_before_quitting, handler

from physicsnemo.models.healda import ModelConfigV1, profiling

from . import training_stats

try:
    import wandb
except ImportError:
    wandb = None

DATASET_METADATA_FILENAME = "dataset-metadata.pth"
TRAINER_METADATA_FILENAME = "loop.json"


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.


@contextlib.contextmanager
def ddp_sync(module, sync):
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


# ----------------------------------------------------------------------------


def _to_batch(x, device, non_blocking=True):
    if isinstance(x, dict):
        return {
            k: _to_batch(v, device, non_blocking=non_blocking) for k, v in x.items()
        }
    elif isinstance(x, list):
        return [_to_batch(i, device, non_blocking=non_blocking) for i in x]
    elif torch.is_tensor(x):
        if torch.is_floating_point(x):
            x = x.float()
        return x.to(device, non_blocking=non_blocking)
    elif hasattr(x, "to") and callable(getattr(x, "to")):
        # custom object with a 'to' method
        return x.to(device, non_blocking=non_blocking)
    elif dataclasses.is_dataclass(x):
        return x.__class__(
            **{
                field.name: _to_batch(
                    getattr(x, field.name), device, non_blocking=non_blocking
                )
                for field in dataclasses.fields(x)
            }
        )
    else:
        raise NotImplementedError(x)


def _format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60
        )


class CheckpointHandler:
    """Manages checkpoint file naming and paths."""

    def __init__(self, run_dir, filename: str = "training-state-{}.checkpoint"):
        self.filename = filename
        self.run_dir = run_dir

    def get_filename(self, nimg):
        """Return checkpoint filename for given image count."""
        return self.filename.format("%09d" % nimg)

    def get_path(self, nimg):
        """Return full checkpoint path for given image count."""
        return os.path.join(self.run_dir, self.get_filename(nimg))

    def list_checkpoints(self, run_dir=None):
        run_dir = run_dir or self.run_dir
        files = glob.glob(self.filename.format("*"), root_dir=run_dir)
        pattern = self.filename.format(r"(\d{9})")
        files = sorted(files)
        for file in files:
            m = re.match(pattern, file)
            if m:
                nimg = int(m.group(1))
                yield os.path.join(run_dir, file), nimg


@dataclasses.dataclass
class TrainingLoopBase(loop.TrainingLoopBase, abc.ABC):
    """Abstract base class for diffusion trainings loops

    Implementations should define
    - get_data_loaders
    - get_network

    """

    device: torch.device | None = None

    def __post_init__(self):
        if self.steps_per_tick <= 0:
            ValueError(self.steps_per_tick)

        self._metrics_to_print = set()
        self.ema: torch.nn.Module | None = None
        self.iteration = 0
        self.do_wandb = False
        self._wandb_run = None

    @abc.abstractmethod
    def get_data_loaders(
        self, batch_gpu: int
    ) -> tuple[SpatioTemporalDataset, Iterable, Iterable]:
        """Returns dataset, training loader, and validation loader."""
        pass

    def get_network(self) -> torch.nn.Module:
        """Instantiates the model from config."""
        return models.get_model(self.model_config)

    @abc.abstractmethod
    def get_optimizer(self, parameters):
        """Returns network optimizer"""
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        """Returns the loss function."""
        pass

    @property
    def model_config(self) -> ModelConfigV1 | None:
        """Model configuration used for the network. This is used for checkpointing.

        If you are overriding get_network, then be sure to make this consistent.
        """
        return None

    def _setup_datasets(self):
        self.dataset_obj, self.train_loader, self.valid_loader = self.get_data_loaders(
            self.batch_gpu
        )
        if self.test_with_single_batch:
            self.train_loader = itertools.repeat(next(iter(self.train_loader)))
            self.valid_loader = [next(iter(self.valid_loader))]

    def _setup_networks(self):
        self.ddp = self.net = self.get_network()
        self.net.train().requires_grad_(True).to(self.device)
        if dist.get_world_size() > 1:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device],
                broadcast_buffers=False,
            )

    @profiling.nvtx
    def log_tick(
        self,
        maintenance_time,
        tick_start_time,
        tick_end_time,
        start_time,
        cur_tick,
        cur_nimg,
    ):
        # Print status line, accumulating the same information in training_stats.
        images_per_tick = self.steps_per_tick * self.batch_size
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {_format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / images_per_tick * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(self.device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

    def setup_logs(self):
        if dist.get_rank() != 0:
            logger.setLevel(logging.CRITICAL)

        self.writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)
        self._step_metrics = {}

    @property
    def batch_gpu_total(self) -> int:
        world_size: int = dist.get_world_size()
        return self.batch_size // world_size

    def setup_batching(self):
        # Select batch size per GPU.
        if self.batch_gpu is None or self.batch_gpu > self.batch_gpu_total:
            self.batch_gpu = self.batch_gpu_total

        if self.batch_gpu_total % self.batch_gpu != 0:
            raise ValueError()

        num_accumulation_rounds = self.batch_gpu_total // self.batch_gpu
        self.num_accumulation_rounds = num_accumulation_rounds

    @staticmethod
    def print_network_info(net, device):
        pass

    def _load_iterator_state(self, checkpoint):
        with checkpoint.open("iterator_state.json") as f:
            iterator_state = json.loads(f.read())
            if iterator_state:
                self.epoch_idx = iterator_state["epoch_idx"]
                self.samples_processed_this_epoch_per_rank = iterator_state[
                    "samples_processed_this_epoch_per_rank"
                ]

    def resume_from_state(
        self,
        resume_state_dump,
        optimizer=True,
        require_all=True,
        wandb=False,
        iterator_state=True,
    ):
        dist.print0(f'Loading training state from "{resume_state_dump}"...')

        with checkpointing.Checkpoint(resume_state_dump, "r") as checkpoint:
            self._load_net_state(checkpoint, require_all)
            gc.collect()
            if optimizer and self.optimizer is not None:
                self._load_optimizer_state(checkpoint)

            with checkpoint.open("loop.json") as f:
                old_loop = self.loads(f.read())

            # Restore iterator state if available (for backward compatibility)
            if iterator_state:
                try:
                    self._load_iterator_state(checkpoint)
                except FileNotFoundError as e:
                    logger.warning(
                        f"Iterator state not found in checkpoint (backward compatibility): {e}. "
                        "Using defaults (epoch_idx=0, samples_processed_this_epoch_per_rank=0)"
                    )

            # handle wandb
            if wandb:
                self.wandb_id = old_loop.wandb_id

    def _load_net_state(self, checkpoint, require_all):
        with checkpoint.open("net_state.pth", "r") as f:
            net_state = torch.load(f, weights_only=True, map_location="cpu")
            self.net.load_state_dict(net_state, strict=require_all)

    def _load_optimizer_state(self, checkpoint):
        with checkpoint.open("optimizer_state.pth", "r") as f:
            # load to cpu to avoid copies in gpu memory
            optimizer_state = torch.load(f, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)

    def train_step(
        self, *, condition=None, target, labels, augment_labels=None, **kwargs
    ):
        return self.loss_fn(
            net=partial(
                self.ddp,
                condition=condition,
                class_labels=labels,
                augment_labels=augment_labels,
                **kwargs,
            ),
            images=target,
        )

    def _stage_tuple_batch(self, batch):
        indict = {}
        images, labels, condition = batch[:3]
        if images.ndim != 4:
            raise ValueError(f"Expected images.ndim == 4, got {images.ndim}")
        indict["target"] = images.to(self.device)
        indict["condition"] = condition.to(self.device)
        indict["labels"] = labels.to(self.device)

        if len(batch) == 4:
            augment_labels = batch[3]
            if augment_labels is not None:
                augment_labels.to(self.device).float()
            indict["augment_labels"] = batch[3]
        return indict

    def _stage_dict_batch(self, batch):
        return _to_batch(batch, self.device)

    @profiling.nvtx
    def backward_batch(self, dataset_iterator):
        self.ddp.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        time_start = time.time()
        for round_idx in range(self.num_accumulation_rounds):
            with ddp_sync(self.ddp, (round_idx == self.num_accumulation_rounds - 1)):
                with profiling.nvtx_range("load data"):
                    batch = next(dataset_iterator)

                    if isinstance(batch, dict):
                        indict = self._stage_dict_batch(batch)
                    else:
                        warnings.warn(
                            DeprecationWarning(
                                "tuple based dataloaders will be removed soon. please refactor to use dicts."
                            )
                        )
                        indict = self._stage_tuple_batch(batch)

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
                ):
                    # print(f"indict size: {size_of(indict) / 1e9} GB")
                    loss = self.train_step(**indict)
                    self.log_metric("Loss/loss", loss, print=True)
                    time_length = loss.shape[2]  # (b, c, t, x)

                    if self.loss_reduction == "v1":
                        loss_mean = loss.sum().mul(
                            self.loss_scaling / (self.batch_gpu_total * time_length)
                        )
                    elif self.loss_reduction == "mean":
                        loss_mean = loss.mean() / self.num_accumulation_rounds
                    else:
                        raise NotImplementedError(self.loss_reduction)

                with profiling.nvtx_range("training_loop:backward"):
                    loss_mean.backward()

                total_loss += loss_mean.detach().cpu()
        time_end = time.time()
        self.log_debug(f"Final Loss: {total_loss.item()}")
        self.log_debug(
            f"Time taken for {self.num_accumulation_rounds} accumulation rounds: {time_end - time_start}"
        )

    def _log_parameter_and_gradient_norms(self):
        # Log parameter and gradient norms if enabled
        if self.log_parameter_norm or self.log_parameter_grad_norm:
            for name, param in self.net.named_parameters():
                if self.log_parameter_norm:
                    self.log_metric(
                        f"param_norm/{name}",
                        param.data.norm(2),
                        frequency="tick",
                        print=False,
                    )
                if self.log_parameter_grad_norm and param.grad is not None:
                    self.log_metric(f"grad_norm/{name}", param.grad.norm(2))

        grad_norm = torch.nn.utils.get_total_norm(
            [param.grad for param in self.net.parameters() if param.grad is not None]
        )
        self.log_metric("grad_norm", grad_norm, frequency="tick", print=True)
        self.log_metric("grad_norm", grad_norm, frequency="step")

    @profiling.nvtx
    @finish_before_quitting
    def step_optimizer(self, cur_nimg):
        torch.cuda.nvtx.range_push("training_loop:step")

        warmup_imgs = self.lr_rampup_img
        flat_imgs = self.flat_imgs
        decay_imgs = self.decay_imgs
        total_imgs = warmup_imgs + flat_imgs + decay_imgs

        def lr_lambda(cur_nimg):
            import math

            base_lr = self.lr
            min_lr = self.lr_min

            min_factor = min_lr / base_lr
            if cur_nimg < warmup_imgs:
                # linear ramp from 0 → 1
                return float(cur_nimg) / warmup_imgs
            elif cur_nimg < warmup_imgs + flat_imgs:
                return 1.0
            elif cur_nimg < total_imgs:
                # cosine decay from factor=1 → factor=min_factor
                progress = float(cur_nimg - warmup_imgs - flat_imgs) / decay_imgs
                # standard cosine schedule:
                return min_factor + 0.5 * (1.0 - min_factor) * (
                    1.0 + math.cos(math.pi * progress)
                )
            else:
                return min_factor

        def default_scale(cur_nimg):
            return min(cur_nimg / max(self.lr_rampup_img, 1e-8), 1)

        use_lr_lambda = True
        scale_fn = lr_lambda if use_lr_lambda else default_scale

        scale = scale_fn(self.cur_nimg)
        for g in self.optimizer.param_groups:
            if "base_lr" not in g:
                if "lr" in g:
                    g["base_lr"] = g["lr"]  # lazy init from existing LR
                else:
                    g["base_lr"] = self.optimizer.defaults["lr"]
            lr = g["base_lr"] * scale
            self.log_debug(
                f"Learning rate: {lr} from base: {g['base_lr']} with scale factor: {scale} (would normally be {default_scale(self.cur_nimg)})"
            )

            g["lr"] = lr
            self.writer.add_scalar("lr", lr, global_step=self.cur_nimg)

        self._log_parameter_and_gradient_norms()

        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        if self.gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=self.gradient_clip_max_norm
            )

        self._step_optimizer()
        # increment the number of images processed within the current epoch
        self.samples_processed_this_epoch_per_rank += self.batch_gpu or 1
        torch.cuda.nvtx.range_pop()

        self._flush_step_metrics()

    def on_tick(self):
        pass

    @profiling.nvtx
    def validate(self, net):
        loss_key = "Loss/test_loss"

        with torch.no_grad():
            for batch in self.valid_loader:
                if len(batch) == 4:
                    images, labels, condition, augment_labels = batch
                else:
                    images, labels, condition = batch
                    augment_labels = None

                if images.ndim != 4:
                    raise ValueError(f"Expected images.ndim == 4, got {images.ndim}")

                images = images.to(self.device).to(torch.float32)
                condition = condition.to(self.device).to(torch.float32)
                labels = labels.to(self.device)

                loss = self.train_step(
                    condition=condition,
                    target=images,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                training_stats.report(loss_key, loss)

    def log_metric(self, key, value, print=True, frequency="tick"):
        """Log a metric. Will be averaged over all calls within a tick

        Args:
            print: if True then print the metric to the console at the end of the tick

        """
        if frequency == "tick":
            training_stats.report(key, value)
        elif frequency == "step":
            if key not in self._step_metrics:
                self._step_metrics[key] = torchmetrics.MeanMetric().to(value)
            self._step_metrics[key].update(value)

        if print:
            self._metrics_to_print.add(key)

    def _flush_step_metrics(self):
        for key, metric in self._step_metrics.items():
            value = metric.compute()
            if dist.get_rank() == 0:
                self.writer.add_scalar(key, value, global_step=self.cur_nimg)
            metric.reset()

    def _flush_training_stats_to_wandb(self):
        if dist.get_rank() == 0:
            info = training_stats.default_collector.as_dict()
            metrics = {name: info[name]["mean"] for name in info}
            wandb.log(metrics, step=self.cur_nimg)

    @property
    def batch_info(self) -> None | BatchInfo:
        return None

    @profiling.nvtx
    @finish_before_quitting
    def _save_checkpoint(self, path, optimizer: bool):
        # ensure that file updates are atomic to avoid faulty
        # restart files
        tmppath = path + ".tmp" + str(os.getpid())
        with checkpointing.Checkpoint(tmppath, "w") as checkpoint:
            checkpoint.write_model(self.net)
            if self.batch_info is not None:
                checkpoint.write_batch_info(self.batch_info)

            if optimizer:
                with checkpoint.open("optimizer_state.pth", "w") as f:
                    torch.save(self.optimizer.state_dict(), f)

            with checkpoint.open("loop.json", "w") as f:
                f.write(self.dumps().encode())

            # Save iterator state for resuming
            with checkpoint.open("iterator_state.json", "w") as f:
                iterator_state = {
                    "epoch_idx": self.epoch_idx,
                    "samples_processed_this_epoch_per_rank": self.samples_processed_this_epoch_per_rank,
                }
                f.write(json.dumps(iterator_state).encode())

            if self.model_config is not None:
                checkpoint.write_model_config(self.model_config)
        shutil.move(tmppath, path)

    @profiling.nvtx
    def save_training_state(self, cur_nimg):
        if dist.get_rank() != 0:
            return

        state_filename = self._state_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving checkpoint to {state_filename}")
        self._save_checkpoint(state_filename, optimizer=True)
        dist.print0(f"Checkpoint saved to {state_filename}")

    @profiling.nvtx
    def save_network_snapshot(self, cur_nimg):
        if dist.get_rank() != 0:
            return

        filename = self._snapshot_checkpoint_handler.get_path(cur_nimg)
        dist.print0(f"Saving network snapshot to {filename}")
        self._save_checkpoint(filename, optimizer=False)

    def flush_training_stats(self):
        logger = logging.getLogger(__name__)
        logger.info("Begin. flushing training stats.")
        training_stats.default_collector.update()
        if self.do_wandb:
            self._flush_training_stats_to_wandb()

        if dist.get_rank() == 0:
            info = training_stats.default_collector.as_dict()
            try:
                nimg = info["Progress/kimg"]["mean"] * 1000
            except KeyError:
                nimg = self.cur_nimg

            for k, v in info.items():
                for moment in v:
                    self.writer.add_scalar(f"{k}/{moment}", v[moment], global_step=nimg)

            stats_path = os.path.join(self.run_dir, "stats.jsonl")
            with open(stats_path, "at") as f:
                stats = training_stats.default_collector.as_dict()
                for stat in stats:
                    mean = stats[stat]["mean"]
                    if stat in self._metrics_to_print:
                        print(f"{stat} = {mean:4g}")
                f.write(
                    json.dumps(
                        dict(
                            training_stats.default_collector.as_dict(),
                            timestamp=time.time(),
                        )
                    )
                    + "\n"
                )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.loads(f.read())

    @classmethod
    def from_rundir(cls, run_dir):
        path = os.path.join(run_dir, TRAINER_METADATA_FILENAME)
        loop = cls.from_json(path)
        loop.run_dir = run_dir
        return loop

    def dumps(self):
        fields = dataclasses.asdict(self)
        fields.pop("device", None)
        return json.dumps(fields)

    @classmethod
    def loads(cls, s):
        return cls(**json.loads(s))

    def save_metadata(self):
        with open(os.path.join(self.run_dir, TRAINER_METADATA_FILENAME), "w") as f:
            fields = dataclasses.asdict(self)
            fields.pop("device", None)
            f.write(self.dumps())

    def setup(self):
        self.setup_logs()
        self.save_metadata()
        self.device = self.device or torch.device("cuda", torch.cuda.current_device())
        self.loss_fn = self.get_loss_fn()

        # iterators
        # used to restore the sampler state during restarts
        self.cur_nimg = 0
        self.epoch_idx = 0
        self.samples_processed_this_epoch_per_rank = 0

        self._setup_datasets()
        self._setup_networks()
        self.print_network_info(self.net, self.device)
        self.setup_batching()
        self._setup_optimizer()
        self._state_checkpoint_handler = CheckpointHandler(self.run_dir)
        self._snapshot_checkpoint_handler = CheckpointHandler(
            self.run_dir, "network-snapshot-{}.checkpoint"
        )

    def _setup_optimizer(self):
        self.optimizer = self.get_optimizer(self.net.named_parameters())
        if self.compile_optimizer:
            self._step_optimizer = torch.compile(self.optimizer.step)
        else:
            self._step_optimizer = self.optimizer.step

    def setup_wandb(self, **kwargs):
        try:
            if wandb is not None and dist.get_rank() == 0:
                os.environ["WANDB_API_KEY"]
                run = wandb.init(
                    id=self.wandb_id,
                    config=json.loads(self.dumps()),
                    project="ufs-da",
                    entity="nv-research-climate",
                    **kwargs,
                )
                self.wandb_id = run.id
                self.do_wandb = True
                self._wandb_run = run
        except KeyError:
            # cannot init wandb
            dist.print0("WANDB_API_KEY not set. Cannot use wandb")
            pass

    def resume_from_rundir(self, run_dir=None, require_all=True):
        checkpoint_info = None
        for checkpoint_info in self._state_checkpoint_handler.list_checkpoints(run_dir):
            pass
        # now training_state and nimg are the final checkpoints
        if checkpoint_info is None:
            raise FileNotFoundError("No checkpoint file found.")

        path, nimg = checkpoint_info

        self.cur_nimg = nimg
        self.resume_from_state(path, require_all=require_all, wandb=True)

    def log_debug(self, msg):
        if dist.get_rank() != 0:
            return

        if self.iteration % self.print_steps != 0:
            return

        logger.debug(msg)

    def train(self):
        # signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        try:
            self._train()
        except QuitEarly as e:
            dist.print0(f"Caught {e}. Quitting early.")
            self.save_training_state(self.cur_nimg)
            try:
                del self.train_loader
                del self.valid_loader
            except AttributeError:
                pass

    def _batch_iterator(self):
        while True:
            for batch in self.train_loader:
                yield batch

            self.epoch_idx += 1
            self.samples_processed_this_epoch_per_rank = 0
            self.iteration = 0

    def _train(self):
        dist.print0("Loss function", self.loss_fn)
        start_time = time.time()
        np.random.seed(
            (self.seed * dist.get_world_size() + dist.get_rank() + self.cur_nimg)
            % (1 << 31)
        )
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            not self.tf32
        )

        # Train.
        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time
        dist.update_progress(0, self.total_ticks)
        dataset_iterator = self._batch_iterator()
        top_time = time.time()
        steps = 0
        for cur_tick in range(self.total_ticks):
            for _ in range(self.steps_per_tick):
                step_start = time.time()
                self.backward_batch(dataset_iterator)
                self.cur_nimg += self.batch_size
                self.step_optimizer(self.cur_nimg)

                step_end = time.time()
                self.log_debug(
                    f"Step {steps} time: {step_end - step_start}. Avg time: {(step_end - top_time) / (steps + 1)}"
                )
                self.log_debug(
                    f"CPU Memory: {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}GB"
                )
                steps += 1
                self.iteration = steps
            tick_end_time = time.time()
            self.log_tick(
                maintenance_time,
                tick_start_time,
                tick_end_time,
                start_time,
                cur_tick,
                self.cur_nimg,
            )

            # Save network snapshot.
            if (self.snapshot_ticks is not None) and (
                cur_tick % self.snapshot_ticks == 0
            ):
                self.save_network_snapshot(self.cur_nimg)
            if (self.state_dump_ticks is not None) and (
                cur_tick % self.state_dump_ticks == 0
            ):
                self.save_training_state(self.cur_nimg)

            self.net.eval()
            logger.info("Validating...")
            val_start_time = time.time()
            self.validate(self.net)
            val_time = time.time() - val_start_time
            logger.info(f"Validation time: {val_time:.2f}s.")
            self.net.train()

            # Update logs.
            self.flush_training_stats()
            dist.update_progress(cur_tick, self.total_ticks)

            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        # Done.
        self.save_training_state(self.cur_nimg)
        dist.print0("Exiting...")
