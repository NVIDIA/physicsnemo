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

import torch
from .utils import get_grid3d


class DataLoader2D_swe(object):
    def __init__(self, data, nx=128, nt=100, sub=1, sub_t=1, tend=1.0, nin=1, nout=3):
        self.sub = sub
        self.sub_t = sub_t
        self.tend = tend

        self.nin = nin
        self.nout = nout

        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.S = s // sub
        self.T = nt // sub_t
        self.T += 1
        data = data[:, 0 : self.T : sub_t, 0:s:sub, 0:s:sub]
        self.data = data.permute(0, 2, 3, 1, 4)

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        a_data = self.data[start : start + n_sample, :, :, 0, : self.nin].reshape(
            n_sample, self.S, self.S, self.nin
        )
        u_data = self.data[start : start + n_sample].reshape(
            n_sample, self.S, self.S, self.T, self.nout
        )
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.tend)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, self.nin).repeat(
            [1, 1, 1, self.T, 1]
        )
        a_data = torch.cat(
            (
                gridx.repeat([n_sample, 1, 1, 1, 1]),
                gridy.repeat([n_sample, 1, 1, 1, 1]),
                gridt.repeat([n_sample, 1, 1, 1, 1]),
                a_data,
            ),
            dim=-1,
        )
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        if train:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )
        return loader
