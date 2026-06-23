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

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(config={"extra": "allow"})
class DatasetConfig:
    name: str


@dataclass(config={"extra": "forbid"})
class ModelConfig:
    model_name: Literal["fgn"] = "fgn"
    history_frames: int = Field(default=2, ge=2)
    latent_dim: int = Field(default=32, ge=1)  # paper §2.3: z ~ N(0,I)^32
    background_channels: int | Literal["auto"] = "auto"
    invariant_channels: int | Literal["auto"] = "auto"
    # DiT backbone (physicsnemo.models.dit.DiT):
    patch_size: list[int] = Field(default_factory=lambda: [4, 4])
    hidden_size: int = Field(default=384, ge=16)
    depth: int = Field(default=12, ge=1)
    num_heads: int = Field(default=8, ge=1)


@dataclass(config={"extra": "forbid"})
class OptimizerConfig:
    # Paper Table A.2: lr=8e-5 (stages 2-4), weight_decay=0.1, AdamW.
    lr: float = Field(default=8e-5, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(default=0.1, ge=0.0)
    # Linear warmup for lr_warmup_steps then cosine decay to lr_min.
    # Paper Table A.2: 1000 warmup steps for all stages.
    lr_warmup_steps: int = Field(default=1000, ge=0)
    lr_min: float = Field(default=0.0, ge=0.0)


@dataclass(config={"extra": "forbid"})
class LossConfig:
    num_samples: int = Field(default=4, ge=2)
    mse_weight: float = Field(default=0.1, ge=0.0)
    # GraphCast-style per-variable weights with geopotential halved per
    # FGN §2.2.3. Independent of cos(lat) area weighting.
    use_channel_weights: bool = False
    # cos(lat) area weighting for the lat/lon grid.
    use_area_weights: bool = False


@dataclass(config={"extra": "forbid"})
class TrainingConfig:
    outdir: str = "rundir"
    experiment_name: str = "fgn"
    run_id: int | str = "0"
    rundir: str = "rundir/fgn/0"
    checkpoint_dir: str = "checkpoints"
    num_data_workers: int = Field(default=0, ge=0)
    seed: int = 7
    batch_size: int = Field(default=8, ge=1)
    total_train_steps: int = Field(default=100, ge=1)
    print_progress_freq: int = Field(default=10, ge=1)
    checkpoint_freq: int = Field(default=50, ge=1)
    validation_freq: int = Field(default=25, ge=1)
    resume_checkpoint: int | Literal["latest"] | None = "latest"
    clip_grad_norm: float = -1.0
    ar_steps: int = Field(default=1, ge=1)
    # Data + domain parallelism knobs. Mirrors StormCast's convention.
    # - domain_parallel_size=1 & force_sharding=False → pure single-process
    #   or plain DDP, no ShardTensor overhead (default for smoke tests).
    # - domain_parallel_size>1 → spatial sharding on the domain mesh axis.
    # - force_sharding=True → wrap tensors/model in ShardTensor even with a
    #   single domain rank (useful to test the sharding path end-to-end).
    domain_parallel_size: int = Field(default=1, ge=1)
    force_sharding: bool = False
    # bf16 autocast — enabled by default (H100 native). Set amp=false for
    # fp32 debugging. Mirrors graphcast/conf/config.yaml amp convention.
    amp: bool = True
    # torch.compile the model forward before FSDP wrapping for ~10-30%
    # throughput gain. Skipped automatically when ShardTensor is active
    # (DTensor ops cause graph breaks). Warm-up adds ~60 s on first run.
    torch_compile: bool = False
    # Experiment tracking via physicsnemo.utils.logging.LaunchLogger.
    # Set use_wandb=true to route metrics to W&B (requires wandb installed and
    # WANDB_API_KEY set). wandb_project is the W&B project name.
    use_wandb: bool = False
    wandb_project: str = "fgn"
    use_mlflow: bool = False
    # Validation diagnostic hooks. When enabled, the trainer runs a short
    # ensemble rollout on a single validation batch at each ``validation_freq``
    # step and writes per-variable CRPS / RMSE / spread-skill / rank-hist /
    # power-spectrum artifacts (Figures 2 + 3 of arXiv:2506.10772v1, minus
    # baseline-dependent scorecards and REV) to ``rundir/validation/``.
    # Cap on per-rank validation batches when running under ParallelHelper
    # (the rank-sharded sampler is infinite by design — StormCast convention).
    # None = sweep one local epoch.
    validation_steps: int | None = None
    validation_metrics: bool = False
    validation_ensemble_size: int = Field(default=4, ge=2)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)


@dataclass(config={"extra": "forbid"})
class InferenceConfig:
    # Single-checkpoint mode: set ``checkpoint`` ("latest" or a path).
    # Deep-ensemble mode: set ``checkpoints`` (list of paths); paper §2.2.1
    # uses J=4 independently-trained models with equal members each and a
    # fixed model identity per trajectory.
    checkpoint: str = "latest"
    checkpoints: list[str] | None = None
    dataset_index: int = Field(default=0, ge=0)
    num_steps: int = Field(default=3, ge=1)
    num_trajectories: int = Field(default=4, ge=1)
    seed: int = 17
    output_path: str = "rundir/fgn/0/forecast.pt"


@dataclass(config={"extra": "forbid"})
class EvalConfig:
    checkpoint: str = "latest"
    checkpoints: list[str] | None = None
    future_steps: int = Field(default=20, ge=1, le=60)
    ensemble_size: int = Field(default=8, ge=2)
    batch_size: int = Field(default=1, ge=1)
    num_workers: int = Field(default=0, ge=0)
    outdir: str = "rundir/fgn/0/eval"
    pool_sizes: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])


@dataclass(config={"extra": "forbid"})
class TrainMainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


@dataclass(config={"extra": "forbid"})
class InferenceMainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


@dataclass(config={"extra": "forbid"})
class EvalMainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig
