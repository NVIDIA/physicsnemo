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


from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass


@dataclass(config={"extra": "forbid"})
class ModelConfig:
    """Model configuration: cfg.model"""

    model_name: Literal["regression", "diffusion"] = (
        "diffusion"  # model type, "regression" or "diffusion"
    )
    architecture: Literal["unet", "dit"] = "unet"  # model architecture, "unet" or "dit"
    regression_conditions: list[Literal["state", "background", "invariant"]] = Field(
        default=["state", "background", "invariant"]
    )
    diffusion_conditions: list[
        Literal["state", "regression", "background", "invariant"]
    ] = Field(default=["state", "regression", "invariant"])
    spatial_pos_embed: bool = False  # use spatial positional embedding for unet
    attn_resolutions: list[int] = Field(
        default=[]
    )  # unet resolutions where global self-attention if computed
    regression_weights: str | None = (
        None  # file with weights of pre-trained regression model, set to None when training regression or diffusion-only model
    )
    hyperparameters: dict[str, Any] = Field(
        default={}
    )  # additional parameters to be passed to model


@dataclass(config={"extra": "forbid"})
class PerfConfig:
    """Performance configuration: cfg.training.perf"""

    fp_optimizations: Literal["fp32", "amp-fp16", "amp-bf16"] = (
        "fp32"  # Floating point mode: "fp32", "amp-fp16", "amp-bf16"
    )
    torch_compile: bool = False  # Use torch.compile to compile model
    use_apex_gn: bool = (
        False  # Use Apex GroupNorm (enables channels_last memory format)
    )
    allow_tf32: bool = (
        False  # Allow TF32 for matmul and cuDNN (faster but less precise)
    )
    allow_fp16_reduced_precision: bool = (
        False  # Allow reduced precision reductions in fp16
    )


@dataclass(config={"extra": "forbid"})
class OptimizerConfig:
    """Optimizer configuration: cfg.training.optimizer"""

    name: Literal["adam", "adamw", "stableadamw"] | tuple[str, dict[str, Any]] = (
        "adam"  # Optimizer type: "adam", "adamw", "stableadamw"
    )
    lr: float = Field(default=4e-4, gt=0)  # Initial learning rate
    betas: tuple[float, float] = (0.9, 0.999)  # Adam beta parameters
    weight_decay: float = Field(default=0.0, ge=0.0)  # Weight decay (L2 regularization)
    eps: float = Field(default=1.0e-8, gt=0.0)  # Adam epsilon for numerical stability
    fused: bool = True  # Use fused CUDA kernel (faster)


@dataclass(config={"extra": "allow"})
class SchedulerConfig:
    """Model configuration: cfg.training.scheduler

    Additional parameters will be passed to the scheduler
    """

    name: str | None = (
        None  # name of scheduler class in torch.optim.lr_scheduler, of None for no scheduling
    )
    lr_rampup_steps: int = Field(
        default=1000, ge=1
    )  # Number of training steps over which to perform linear LR warmup


@dataclass(config={"extra": "forbid"})
class LossConfig:
    """Loss configuration: cfg.training.loss"""

    type: Literal["regression", "edm"] = "edm"
    sigma_distribution: Literal["lognormal", "loguniform"] = "lognormal"
    sigma_data: float | list[float] = 0.5
    P_mean: float = -1.2  # Center of the lognormal noise distribution
    P_std: float = Field(default=1.2, gt=0.0)  # Std of the lognormal noise distribution
    # Loguniform distribution parameters (used when sigma_distribution: "loguniform")
    sigma_min: float = Field(default=0.002, gt=0.0)  # Minimum noise level
    sigma_max: float = Field(default=80.0, gt=0.0)  # Maximum noise level

    @model_validator(mode="after")
    def _check_sigmas(self):
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")
        sigma_data = (
            self.sigma_data if isinstance(self.sigma_data, list) else [self.sigma_data]
        )
        if any(sigma <= 0 for sigma in sigma_data):
            raise ValueError("sigma_data must be > 0")
        return self


@dataclass(config={"extra": "forbid"})
class TrainingConfig:
    """Training configuration: cfg.training"""

    # General training config items
    rundir: str  # Path where experiement outputs will be saved
    outdir: str = "rundir"  # Root path under which to save training outputs
    experiment_name: str = "stormcast-training"  # Name for the training experiment
    run_id: str = "0"  # Unique ID to use for this training run
    num_data_workers: int = Field(
        default=4, ge=0
    )  # Number of dataloader worker threads per proc
    log_to_wandb: bool = (
        False  # Whether or not to log to Weights & Biases (requires wandb account)
    )
    wandb_mode: Literal["offline", "online"] = (
        "online"  # logging mode, "online" or "offline"
    )
    log_to_tensorboard: bool = (
        False  # Whether to log to TensorBoard (logs saved in rundir/tensorboard)
    )
    seed: int = Field(
        default=-1, ge=-1
    )  # Specify a random seed by setting this to an int > 0
    cudnn_benchmark: bool = True  # Enable/disable CuDNN benchmark mode
    resume_checkpoint: int | Literal["latest"] = (
        "latest"  # epoch number to continue training from, or "latest" for the latest checkpoint
    )
    initial_weights: str | None = (
        None  # if not None, a .mdlus checkpoint to load weights at the start of training; no effect if training continues from a checkpoint
    )
    max_run_steps: int | None = (
        None  # if not None, the maximum number of steps that will be trained on this run (disregarding restarts)
    )

    # Logging frequency
    print_progress_freq: int = Field(
        default=100, ge=1
    )  # How often to print progress, measured in number of training steps
    checkpoint_freq: int = Field(
        default=1000, ge=1
    )  # How often to save the checkpoints, measured in number of training steps
    validation_freq: int = Field(
        default=1000, ge=1
    )  # how often to record the validation loss, measured in number of training steps

    # Optimization hyperparameters
    batch_size: int = Field(
        default=64, ge=1
    )  # Total training batch size -- must be >= (and divisble by) number of GPUs being used
    batch_size_per_gpu: int | Literal["auto"] = (
        "auto"  # Batch size on each GPU, set to an int to force smaller local batch with gradient accumulation
    )
    total_train_steps: int = Field(
        default=16000, ge=1
    )  # Number of total training steps, 16000 with batch size 64 corresponds to StormCast paper regression
    clip_grad_norm: (
        float | Literal[-1]
    ) = -1  # Threshold for gradient clipping, set to -1 to disable

    domain_parallel_size: int = Field(
        default=1, ge=1
    )  # number of domain parallel subdivisions
    force_sharding: bool = (
        False  # use sharded tensors even with a single GPU (mostly useful for testing)
    )

    perf: PerfConfig = Field(default=PerfConfig())
    optimizer: OptimizerConfig = Field(default=OptimizerConfig())
    scheduler: SchedulerConfig = Field(default=SchedulerConfig())

    validation_steps: int = Field(
        default=1, ge=0
    )  # Number of batches to evaluate during validation
    validation_plot_variables: list[str] = Field(
        default=["u10m", "v10m", "t2m", "refc", "q1", "q5", "q10"]
    )
    validation_plot_background_channels: list[str] = Field(
        default=[]
    )  # Background channels to show in validation plots (list of names or indices)

    loss: LossConfig = Field(default=LossConfig())


@dataclass(config={"extra": "allow"})
class DatasetConfig:
    """Dataset configuration: cfg.dataset

    Additional parameters are passed to the dataset object.
    """

    name: str  # module and class of the dataset in the datasets directory, e.g. data_loader_hrrr_era5.HrrrEra5Dataset


@dataclass(config={"extra": "allow"})
class SamplerArgsConfig:
    """Sampler args configuration: cfg.sampler.args

    Additional parameters are passed to the sampler.
    """

    num_steps: int = Field(default=18, ge=1)  # number of diffusion steps


@dataclass(config={"extra": "forbid"})
class SamplerConfig:
    """Sampler configuration: cfg.sampler"""

    name: str
    args: SamplerArgsConfig = Field(default=SamplerArgsConfig())


@dataclass(config={"extra": "forbid"})
class MainConfig:
    """Main configuration: cfg"""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    sampler: SamplerConfig
