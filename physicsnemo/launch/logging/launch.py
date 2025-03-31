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

import re
import sys
import time
from os import getcwd, makedirs
from os.path import abspath, exists, join
from typing import Dict, Tuple, Union

import torch
import torch.cuda.profiler as profiler

from physicsnemo.distributed import DistributedManager, reduce_loss

from .console import PythonLogger


class LaunchLogger(object):
    """PhysicsNeMo Launch logger

    An abstracted logger class that takes care of several fundamental logging functions.
    This class should first be initialized and then used via a context manager. This will
    auto compute epoch metrics. This is the standard logger for PhysicsNeMo examples.

    Parameters
    ----------
    name_space : str
        Namespace of logger to use. This will define the loggers title in the console and
        the wandb group the metric is plotted
    epoch : int, optional
        Current epoch, by default 1
    num_mini_batch : Union[int, None], optional
        Number of mini-batches used to calculate the epochs progress, by default None
    profile : bool, optional
        Profile code using nvtx markers, by default False
    mini_batch_log_freq : int, optional
        Frequency to log mini-batch losses, by default 100
    epoch_alert_freq : Union[int, None], optional
        Epoch frequency to send training alert, by default None

    Example
    -------
    >>> from physicsnemo.launch.logging import LaunchLogger
    >>> LaunchLogger.initialize()
    >>> epochs = 3
    >>> for i in range(epochs):
    ...   with LaunchLogger("Train", epoch=i) as log:
    ...     # Log 3 mini-batches manually
    ...     log.log_minibatch({"loss": 1.0})
    ...     log.log_minibatch({"loss": 2.0})
    ...     log.log_minibatch({"loss": 3.0})
    """

    _instances = {}
    console_backend = True
    wandb_backend = False
    mlflow_backend = False
    tensorboard_backend = False
    enable_profiling = False

    mlflow_run = None
    mlflow_client = None

    def __new__(cls, name_space, *args, **kwargs):
        # If namespace already has an instance just return that
        if name_space in cls._instances:
            return cls._instances[name_space]

        # Otherwise create new singleton instance for this namespace
        self = super().__new__(cls)  # don't pass remaining parameters to object.__new__
        cls._instances[name_space] = self

        # Constructor set up to only be ran once by a logger
        self.pyLogger = PythonLogger(name_space)
        self.total_iteration_index = None
        # Distributed
        self.root = True
        if DistributedManager.is_initialized():
            self.root = DistributedManager().rank == 0
        # Profiler utils
        if torch.cuda.is_available():
            self.profiler = torch.autograd.profiler.emit_nvtx(
                enabled=cls.enable_profiling
            )
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.profiler = None

        return self

    def __init__(
        self,
        name_space: str,
        epoch: int = 1,
        num_mini_batch: Union[int, None] = None,
        profile: bool = False,
        mini_batch_log_freq: int = 100,
        epoch_alert_freq: Union[int, None] = None,
    ):
        self.name_space = name_space
        self.mini_batch_index = 0
        self.minibatch_losses = {}
        self.epoch_losses = {}

        self.mini_batch_log_freq = mini_batch_log_freq
        self.epoch_alert_freq = epoch_alert_freq
        self.epoch = epoch
        self.num_mini_batch = num_mini_batch
        self.profile = profile
        # Init initial iteration based on current epoch
        if self.total_iteration_index is None:
            if num_mini_batch is not None:
                self.total_iteration_index = (epoch - 1) * num_mini_batch
            else:
                self.total_iteration_index = 0

        # Set x axis metric to epoch for this namespace
        if self.wandb_backend:
            import wandb

            wandb.define_metric(name_space + "/mini_batch_*", step_metric="iter")
            wandb.define_metric(name_space + "/*", step_metric="epoch")

    def log_minibatch(self, losses: Dict[str, float]):
        """Logs metrics for a mini-batch epoch

        This function should be called every mini-batch iteration. It will accumulate
        loss values over a datapipe. At the end of a epoch the average of these losses
        from each mini-batch will get calculated.

        Parameters
        ----------
        losses : Dict[str, float]
            Dictionary of metrics/loss values to log
        """
        self.mini_batch_index += 1
        self.total_iteration_index += 1
        for name, value in losses.items():
            if name not in self.minibatch_losses:
                self.minibatch_losses[name] = 0
            self.minibatch_losses[name] += value

        # Log of mini-batch loss
        if self.mini_batch_index % self.mini_batch_log_freq == 0:
            # Backend Logging
            mini_batch_metrics = {}
            for name, value in losses.items():
                mini_batch_metrics[f"{self.name_space}/mini_batch_{name}"] = value
            self._log_backends(
                mini_batch_metrics, step=("iter", self.total_iteration_index)
            )

            # Console
            if self.root:
                message = "Mini-Batch Losses:"
                for name, value in losses.items():
                    message += f" {name} = {value:10.3e},"
                message = message[:-1]
                # If we have datapipe length we can get a percent complete
                if self.num_mini_batch:
                    mbp = 100 * (float(self.mini_batch_index) / self.num_mini_batch)
                    message = f"[{mbp:.02f}%] " + message

                self.pyLogger.log(message)

    def log_epoch(self, losses: Dict[str, float]):
        """Logs metrics for a single epoch

        Parameters
        ----------
        losses : Dict[str, float]
            Dictionary of metrics/loss values to log
        """
        for name, value in losses.items():
            self.epoch_losses[name] = value

    def __enter__(self):
        self.mini_batch_index = 0
        self.minibatch_losses = {}
        self.epoch_losses = {}

        # Trigger profiling
        if self.profile and self.profiler:
            self.logger.warning(f"Starting profile for epoch {self.epoch}")
            self.profiler.__enter__()
            profiler.start()

        # Timing stuff
        if torch.cuda.is_available():
            self.start_event.record()
        else:
            self.start_event = time.time()

        if self.mlflow_backend:
            self.mlflow_client.update_run(self.mlflow_run.info.run_id, "RUNNING")

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Abnormal exit dont log
        if exc_type is not None:
            if self.mlflow_backend:
                self.mlflow_client.set_terminated(
                    self.mlflow_run.info.run_id, status="KILLED"
                )
            return
        # Reduce mini-batch losses
        for name, value in self.minibatch_losses.items():
            process_loss = value / self.mini_batch_index
            self.epoch_losses[name] = process_loss
            # Compute global loss
            if DistributedManager.is_initialized() and DistributedManager().distributed:
                self.epoch_losses[name] = reduce_loss(process_loss)

        if self.root:
            # Console printing
            # TODO: add out of total epochs progress
            message = f"Epoch {self.epoch} Metrics:"
            for name, value in self.epoch_losses.items():
                message += f" {name} = {value:10.3e},"
            message = message[:-1]
            self.pyLogger.info(message)

        metrics = {
            f"{self.name_space}/{key}": value
            for key, value in self.epoch_losses.items()
        }

        # Exit profiling
        if self.profile and self.profiler:
            self.logger.warning("Ending profile")
            self.profiler.__exit__()
            profiler.end()

        # Timing stuff, TODO: histograms not line plots
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            # Returns milliseconds
            # https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time
            epoch_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            end_event = time.time()
            epoch_time = end_event - self.start_event

        # Return MS for time / iter
        time_per_iter = 1000 * epoch_time / max([1, self.mini_batch_index])

        if self.root:
            message = f"Epoch Execution Time: {epoch_time:10.3e}s"
            message += f", Time/Iter: {time_per_iter:10.3e}ms"
            self.pyLogger.info(message)

        metrics[f"{self.name_space}/Epoch Time (s)"] = epoch_time
        metrics[f"{self.name_space}/Time per iter (ms)"] = time_per_iter

        self._log_backends(metrics, step=("epoch", self.epoch))

        # TODO this should be in some on delete method / clean up
        if self.mlflow_backend:
            self.mlflow_client.set_terminated(
                self.mlflow_run.info.run_id, status="FINISHED"
            )

        # Alert
        if (
            self.epoch_alert_freq
            and self.root
            and self.epoch % self.epoch_alert_freq == 0
        ):
            if self.wandb_backend:
                import wandb

                from .wandb import alert

                # TODO: Make this a little more informative?
                alert(
                    title=f"{sys.argv[0]} training progress report",
                    text=f"Run {wandb.run.name} is at epoch {self.epoch}.",
                )

    def _log_backends(
        self,
        metric_dict: Dict[str, float],
        step: Tuple[str, int] = None,
    ):
        """Logs a dictionary of metrics to different supported backends

        Parameters
        ----------
        metric_dict : Dict[str, float]
            Metric dictionary
        step : Tuple[str, int], optional
            Tuple containing (step name, step index), by default None
        print : bool, optional
            Print metrics, by default False
        """

        # MLFlow Logging
        if self.mlflow_backend:
            for key, value in metric_dict.items():
                # If value is None just skip
                if value is None:
                    continue
                # Keys only allow alpha numeric, ., -, /, _ and spaces
                key = re.sub("[^a-zA-Z0-9\.\-\s\/\_]+", "", key)
                self.mlflow_client.log_metric(
                    self.mlflow_run.info.run_id, key, value, step=step[1]
                )

        # WandB Logging
        if self.wandb_backend:
            import wandb

            # For WandB send step in as a metric
            # Step argument in lod function does not work with multiple log calls at
            # different intervals
            metric_dict[step[0]] = step[1]
            wandb.log(metric_dict)

    def log_figure(
        self,
        figure,
        artifact_file: str = "artifact",
        plot_dir: str = "./",
        log_to_file: bool = False,
    ):
        """Logs figures on root process to wand or mlflow. Will store it to file in case neither are selected.

        Parameters
        ----------
        figure : Figure
            matplotlib or plotly figure to plot
        artifact_file : str, optional
            File name. CAUTION overrides old files of same name
        plot_dir : str, optional
            output directory for plot
        log_to_file : bool, optional
            set to true in case figure shall be stored to file in addition to logging it to mlflow/wandb
        """
        dist = DistributedManager()
        if dist.rank != 0:
            return

        if self.wandb_backend:
            import wandb

            wandb.log({artifact_file: figure})

        if self.mlflow_backend:
            self.mlflow_client.log_figure(
                figure=figure,
                artifact_file=artifact_file,
                run_id=self.mlflow_run.info.run_id,
            )

        if (not self.wandb_backend) and (not self.mlflow_backend):
            log_to_file = True

        if log_to_file:
            plot_dir = abspath(join(getcwd(), plot_dir))
            if not exists(plot_dir):
                makedirs(plot_dir)
            if not artifact_file.endswith(".png"):
                artifact_file += ".png"
            figure.savefig(join(plot_dir, artifact_file))

    @classmethod
    def toggle_wandb(cls, value: bool):
        """Toggle WandB logging

        Parameters
        ----------
        value : bool
            Use WandB logging
        """
        cls.wandb_backend = value

    @classmethod
    def toggle_mlflow(cls, value: bool):
        """Toggle MLFlow logging

        Parameters
        ----------
        value : bool
            Use MLFlow logging
        """
        cls.mlflow_backend = value

    @staticmethod
    def initialize(use_wandb: bool = False, use_mlflow: bool = False):
        """Initialize logging singleton

        Parameters
        ----------
        use_wandb : bool, optional
            Use WandB logging, by default False
        use_mlflow : bool, optional
            Use MLFlow logging, by default False
        """
        if use_wandb:
            import wandb

            if wandb.run is None:
                PythonLogger().warning("WandB not initialized, turning off")
                use_wandb = False

        if use_wandb:
            LaunchLogger.toggle_wandb(True)
            wandb.define_metric("epoch")
            wandb.define_metric("iter")

        # let only root process log to mlflow
        if DistributedManager.is_initialized():
            if DistributedManager().rank != 0:
                return

        if LaunchLogger.mlflow_run is None and use_mlflow:
            PythonLogger().warning("MLFlow not initialized, turning off")
            use_mlflow = False

        if use_mlflow:
            LaunchLogger.toggle_mlflow(True)
