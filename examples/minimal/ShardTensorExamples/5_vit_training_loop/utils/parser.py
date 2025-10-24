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

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark HybridViT model performance"
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
        "--ddp_size", type=int, default=1, help="DDP world size (default: 1)"
    )
    parser.add_argument(
        "--domain_size", type=int, default=1, help="Domain parallel size (default: 1)"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Enable mixed precision training (default: False)",
    )

    args = parser.parse_args()

    return args
