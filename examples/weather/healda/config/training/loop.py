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
import dataclasses
from typing import Optional


@dataclasses.dataclass
class TrainingLoopBase:
    """Base training config"""

    run_dir: str = "."  # Output directory.
    seed: int = 0  # Global random seed.
    batch_size: int = 512  # Total batch size for one training iteration.
    batch_gpu: Optional[int] = None  # Limit batch size per GPU, None = no limit.
    enable_ema: bool = False
    ema_halflife_kimg: int = (
        500  # Half-life of the exponential moving average (EMA) of model weights.
    )
    ema_rampup_ratio: float = 0.05  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_img: int = 10_000  # Learning rate ramp-up duration.
    flat_imgs: int = 1_500_000 - 10_000
    decay_imgs: int = 1_500_000
    lr_min: float = 1e-6
    lr: float = 1e-4

    loss_reduction: str = "v1"
    """
    Controls how the [b c t x] shaped loss is reduced, where 'b' is the

    Options:
    - v1 (default) - sum over c x, mean over b c
    - mean - mean over all dimensions
    """

    loss_scaling: float = 1.0  # Loss scaling factor for reducing FP16 under/overflows.
    gradient_clip_max_norm: Optional[float] = None
    total_ticks: int = 10
    print_steps: int = 50
    steps_per_tick: int = 1024
    snapshot_ticks: int | None = (
        50  # How often to save network snapshots, None = disable.
    )
    state_dump_ticks: int | None = (
        500  # How often to dump training state, None = disable.
    )

    test_with_single_batch: bool = False
    """Only load a single batch of data for testing and profiling purposes"""

    # Performance optimizations
    # Mixed precision and performance options
    cudnn_benchmark: bool = True  # Enable torch.backends.cudnn.benchmark?
    tf32: bool = True
    bf16: bool = True
    compile_optimizer: bool = False  # if true wrap the optimizer with torch compile

    # wandb
    wandb_id: str | None = None  # will be read from checkpoint if not provided

    # logging
    log_parameter_norm: bool = False
    log_parameter_grad_norm: bool = False
