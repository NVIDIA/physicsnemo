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

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark domain-parallel model performance"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "conv", "natten"],
        help=(
            "Model architecture: vit (ring attention), conv (halo "
            "convolutions), or natten (neighborhood attention; requires the "
            "optional natten package). All support 2D and 3D via "
            "--dimension. See model/__init__.py to register your own "
            "(default: vit)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Global Batch size for training (default: 1)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimension of the model: 2D or 3D (default: 2)",
    )
    parser.add_argument(
        "--image_size_start",
        type=int,
        default=1024,
        help="Starting image size (default: 256)",
    )
    parser.add_argument(
        "--image_size_stop",
        type=int,
        default=1024,
        help="Ending image size (default: 2048)",
    )
    parser.add_argument(
        "--image_size_step",
        type=int,
        default=128,
        help="Step size for image size progression (default: 128)",
    )
    parser.add_argument(
        "--ddp_size",
        type=int,
        default=-1,
        help=("DDP world size. -1 (default) infers it as world_size // domain_size"),
    )
    parser.add_argument(
        "--domain_size", type=int, default=1, help="Domain parallel size (default: 1)"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Enable mixed precision training (default: False)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run inference benchmarks only, skip training (default: False)",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help=(
            "Shard model parameters over the ddp mesh axis with FSDP2 "
            "(fully_shard) instead of replicating them. With ddp_size == 1 "
            "this is a degenerate size-1 shard: no communication, but "
            "parameters uniformly become distributed tensors (default: False)"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile (inductor backend). With domain_size > 1 the "
            "model is compiled regionally, leaving sharded attention eager "
            "(default: False)"
        ),
    )

    args = parser.parse_args()

    return args
