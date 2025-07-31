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

import numpy as np
from torch.utils.data import DataLoader, Dataset


class DatasetTopoDiff(Dataset):
    def __init__(self, topologies, stress, strain, load_im, constraints):
        self.topologies = topologies
        self.constraints = constraints
        self.image_size = topologies.shape[1]

        self.stress = stress
        self.strain = strain
        self.load_im = load_im

    def __len__(self):
        return self.topologies.shape[0]

    def __getitem__(self, idx):
        cons = self.constraints[idx]

        vol_frac = cons["VOL_FRAC"]

        cons = np.zeros((5, self.image_size, self.image_size))

        cons[0] = self.stress[idx]
        cons[1] = self.strain[idx]
        cons[2] = self.load_im[idx][:, :, 0]
        cons[3] = self.load_im[idx][:, :, 1]
        cons[4] = np.ones((self.image_size, self.image_size)) * vol_frac

        return np.expand_dims(self.topologies[idx], 0) * 2 - 1, cons


def load_data_topodiff(
    topologies, constraints, stress, strain, load_img, batch_size, deterministic=False
):
    dataset = DatasetTopoDiff(topologies, stress, strain, load_img, constraints)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )

    while True:
        yield from loader
