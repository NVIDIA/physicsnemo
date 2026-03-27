# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""
Normalized dataset wrappers for flood prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class NormalizedDataset(Dataset):
    r"""
    Dataset wrapper that provides normalized data with query points.
    """

    def __init__(self, geometry, static, boundary, dynamic, target=None, query_res=None, cell_area=None):
        r"""
        Initialize normalized dataset.

        Parameters
        ----------
        geometry : torch.Tensor
            Normalized geometry tensor of shape :math:`(N, n_{cells}, 2)`.
        static : torch.Tensor
            Normalized static features tensor of shape :math:`(N, n_{cells}, C_{static})`.
        boundary : torch.Tensor
            Normalized boundary conditions tensor of shape :math:`(N, H, n_{cells}, C_{boundary})`.
        dynamic : torch.Tensor
            Normalized dynamic features tensor of shape :math:`(N, H, n_{cells}, C_{dynamic})`.
        target : torch.Tensor, optional
            Normalized target tensor of shape :math:`(N, n_{cells}, C_{target})`.
        query_res : List[int], optional, default=[64, 64]
            Query resolution [height, width]. If None, defaults to [64, 64].
        cell_area : torch.Tensor, optional
            Cell area tensor of shape :math:`(N, n_{cells})`.
        """
        self.geometry = geometry
        self.static = static
        self.boundary = boundary
        self.dynamic = dynamic
        self.target = target
        # Use immutable default to avoid mutable default argument issues
        self.query_res = query_res if query_res is not None else [64, 64]
        self.cell_area = cell_area

        if self.geometry is not None and self.geometry.shape[0] > 0:
            geom_sample = self.geometry[0].cpu().numpy()
            x_vals = geom_sample[:, 0]
            y_vals = geom_sample[:, 1]
            min_x, max_x = x_vals.min(), x_vals.max()
            min_y, max_y = y_vals.min(), y_vals.max()
            tx = np.linspace(min_x, max_x, self.query_res[0], dtype=np.float32)
            ty = np.linspace(min_y, max_y, self.query_res[1], dtype=np.float32)
            grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
            q_pts = np.stack([grid_x, grid_y], axis=-1)
            self.query_points = torch.tensor(q_pts, device='cpu')
        else:
            self.query_points = torch.zeros((self.query_res[0], self.query_res[1], 2), dtype=torch.float32)

    def __len__(self):
        return self.geometry.shape[0] if self.geometry is not None else 0

    def __getitem__(self, idx):
        r"""Get a single normalized sample."""
        sample = {
            "geometry": self.geometry[idx],
            "static": self.static[idx],
            "boundary": self.boundary[idx],
            "dynamic": self.dynamic[idx],
            "query_points": self.query_points
        }
        if self.target is not None:
            sample["target"] = self.target[idx]
        if self.cell_area is not None:
            sample["cell_area"] = self.cell_area[idx]
        return sample


class NormalizedRolloutTestDataset(Dataset):
    r"""
    Dataset wrapper for normalized rollout test samples.
    """

    def __init__(self, normalized_samples, query_res=None):
        r"""
        Initialize normalized rollout test dataset.

        Parameters
        ----------
        normalized_samples : List[Dict]
            List of normalized sample dictionaries.
        query_res : List[int], optional, default=[64, 64]
            Query resolution [height, width]. If None, defaults to [64, 64].
        """
        self.normalized_samples = normalized_samples
        # Use immutable default to avoid mutable default argument issues
        self.query_res = query_res if query_res is not None else [64, 64]

        if len(self.normalized_samples) > 0:
            geom_sample = self.normalized_samples[0]["geometry"].cpu().numpy()
            x_vals = geom_sample[:, 0]
            y_vals = geom_sample[:, 1]
            min_x, max_x = x_vals.min(), x_vals.max()
            min_y, max_y = y_vals.min(), y_vals.max()
            tx = np.linspace(min_x, max_x, self.query_res[0], dtype=np.float32)
            ty = np.linspace(min_y, max_y, self.query_res[1], dtype=np.float32)
            grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
            q_pts = np.stack([grid_x, grid_y], axis=-1)
            self.query_points = torch.tensor(q_pts, device='cpu')
        else:
            self.query_points = torch.zeros((self.query_res[0], self.query_res[1], 2), dtype=torch.float32)

    def __len__(self):
        return len(self.normalized_samples)

    def __getitem__(self, idx):
        r"""Get a single normalized rollout sample."""
        sample = self.normalized_samples[idx].copy()
        sample["query_points"] = self.query_points
        return sample

