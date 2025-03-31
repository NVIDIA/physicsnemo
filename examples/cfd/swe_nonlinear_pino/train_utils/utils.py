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

import yaml
import torch
import os
import numpy as np
from physicsnemo.sym.hydra import to_absolute_path


def load_config(file):
    with open(file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def get_grid3d(S, T, time_scale=1.0, device="cpu"):
    "Returns 3D grid of x, y, t"
    gridx = torch.tensor(
        np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device
    )
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(
        np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device
    )
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(
        np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device
    )
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def download_SWE_NL_dataset(dataset, outdir="datasets/"):
    "Tries to download dataset"

    outdir = to_absolute_path(outdir) + "/"

    # skip if already exists
    exists = True
    if not os.path.isfile(outdir + "swe_nl_dataset.pt"):
        exists = False
    if exists:
        print(
            "SWE NL dataset is detected, you need to delete previous dataset to download new one"
        )
        return
    print("SWE NL dataset not detected, downloading dataset")

    # get output directory
    os.makedirs(outdir, exist_ok=True)

    torch.save(dataset, outdir + "swe_nl_dataset.pt")
