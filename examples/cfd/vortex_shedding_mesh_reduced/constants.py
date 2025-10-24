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

from typing import Tuple

from pydantic import BaseModel


class Constants(BaseModel):
    """vortex shedding constants"""

    # data configs
    data_dir: str = "dataset/rawData.npy"
    pivotal_dir: str = "dataset/meshPosition_pivotal.txt"
    mesh_dir: str = "dataset/meshPosition_all.txt"
    sequence_len: int = 401

    # training configs for encoder-decoder model
    batch_size: int = 5  # GNN training batch
    epochs: int = 301
    num_training_samples: int = 400
    num_training_time_steps: int = 300
    lr: float = 0.00001  # 0.0001
    lr_decay_rate: float = 0.9999991
    num_input_features: int = 3
    num_output_features: int = 3
    num_edge_features: int = 3
    ckpt_path: str = "checkpoints/new_encoding"
    ckpt_name: str = "model.pt"

    # training configs for sequence model
    epochs_sequence: int = 200001
    batch_size_sequence: int = 10
    sequence_dim: int = 768
    sequence_context_dim: int = 6
    ckpt_sequence_path: str = "checkpoints/new_sequence"
    ckpt_sequence_name: str = "sequence_model.pt"
    sequence_batch_size: int = 1
    produce_latents: bool = False  # Set it as True when first produce latent representations from the encoder

    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    num_test_samples: int = 10
    num_test_time_steps: int = 300
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1

    # wb configs
    wandb_mode: str = "disabled"
    watch_model: bool = False
