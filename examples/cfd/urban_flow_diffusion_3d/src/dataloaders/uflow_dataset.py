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

from functools import cache
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataloaders.dataset_utils import load_h5, normalize, renormalize


class UflowDataset3D(Dataset):
    def __init__(
        self,
        data_path,
        in_memory=False,
        combined_channels=True,
        normalize=True,
        mins=(-1.1370700673207321, -1.0031025966526042, -1.3040300483418614),
        maxs=(1.041396074425084, 1.2183490678412636, 1.177769794747146),
        **kwargs,
    ):
        # Backwards
        self.ds_ratio = kwargs.get("ds_ratio")
        self.datatype = kwargs.get("datatype")

        self.h5_file = data_path
        self.in_memory = in_memory
        self.normalize = normalize
        self.mins = torch.tensor(mins)
        self.maxs = torch.tensor(maxs)

        if in_memory:
            print("Loading entire dataset into memory...")
            with h5py.File(self.h5_file, "r") as f:
                # Not every packaged file has been through the "-optimized"
                # repacking step that merges U/V/W into a single combined
                # `data` dataset -- fall back to the raw per-component
                # layout when `data` isn't present, instead of raising.
                self.combined_channels = combined_channels and "data" in f
                if self.combined_channels:
                    self.data = f["data"][:]  # type: ignore
                else:
                    channels = ["U", "V", "W"]
                    self.data = np.stack([f[c][:] for c in channels], axis=1)  # type: ignore
        else:
            print("Opening HDF5 file for lazy loading...")
            self.f = h5py.File(self.h5_file, "r")
            self.combined_channels = combined_channels and "data" in self.f
            if self.combined_channels:
                self.data = self.f["data"]
            else:
                self.data = [
                    self.f["U"],
                    self.f["V"],
                    self.f["W"],
                ]

    def close(self):
        """Close the HDF5 file if it was opened lazily."""
        if not self.in_memory:
            print("Closing HDF5 file...")
            self.f.close()
            self.data = None

    def __len__(self):
        if self.combined_channels:
            return self.data.shape[0]  # type: ignore
        else:
            # Assume same count for all channels
            return self.data[0].shape[0]  # type: ignore

    def __getitem__(self, idx):
        if self.combined_channels:
            field = torch.from_numpy(self.data[idx])  # type: ignore
        else:
            x = np.stack([self.data[i][idx] for i in range(3)], axis=0)  # type: ignore
            field = torch.from_numpy(x)
        if self.normalize:
            field = normalize(field, self.mins, self.maxs)
        return {"field": field}

    @cache
    def image_shape(self):
        return self[0]["field"].shape[1:]

    @cache
    def num_channels(self):
        return self[0]["field"].shape[0]


class UflowDataset2D(Dataset):
    def __init__(
        self,
        data_path: str,
        ds_ratio: int = 1,
        normalize: bool = True,
        transform: Optional[callable] = None,
    ):
        assert data_path is not None, "Dataset path cannot be None"
        self.normalize = normalize
        self.transform = transform
        self.ds_ratio = ds_ratio

        # --- Load data using shared utility ---
        u, v, x, y, _, t = load_h5(data_path, load_components="UV", verbose=True)

        # --- Preprocess shapes for uflow-downsampled dataset ---
        u, v, x, y = self._apply_downsampling(u, v, x, y)

        # --- Final downsampling ---
        u = u[:, ::ds_ratio, ::ds_ratio]
        v = v[:, ::ds_ratio, ::ds_ratio]

        self.data = np.stack((u, v), axis=1)  # [T, 2, H, W]
        self.x = x[::ds_ratio]
        self.y = y[::ds_ratio]
        self.t = t

        self.u_min, self.u_max = np.min(u), np.max(u)
        self.v_min, self.v_max = np.min(v), np.max(v)

        self.num, self.channels, self.nx, self.ny = self.data.shape

    def _apply_downsampling(self, u, v, x, y):
        if self.ds_ratio == 5:
            u = u[:, :-1, :-1]
            v = v[:, :-1, :-1]
            x = x[:-1]
            y = y[:-1]
        elif self.ds_ratio in [1, 2]:
            u = u[:, :-13, :-5]
            v = v[:, :-13, :-5]
            x = x[:-13]
            y = y[:-5]
        else:
            raise NotImplementedError(f"Unsupported ds_ratio: {self.ds_ratio}")
        return u, v, x, y

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        field = torch.tensor(self.data[idx], dtype=torch.float32)  # [2, H, W]

        if self.normalize:
            field = normalize(
                field,
                mins=torch.tensor([self.u_min, self.v_min]),
                maxs=torch.tensor([self.u_max, self.v_max]),
            )

        if self.transform:
            field = self.transform(field)

        return {
            "field": field,  # [2, H, W]
            "t": torch.tensor(self.t[idx]),  # [1] or scalar
            "x": torch.tensor(self.x),  # [W]
            "y": torch.tensor(self.y),  # [H]
            "index": idx,
        }

    def image_shape(self):
        return (self.nx, self.ny)

    def num_channels(self):
        return self.channels


class OldUflowDataset3D(Dataset):
    def __init__(
        self,
        data_path: str,
        ds_ratio: int = 1,
        normalize: bool = True,
        transform: Optional[callable] = None,
        mins=None,
        maxs=None,
        datatype=None,
        downsampled=True,
    ):
        assert data_path is not None, "Dataset path cannot be None"
        self.normalize = normalize
        self.transform = transform
        self.ds_ratio = ds_ratio
        self.datatype = datatype

        # --- Load U, V, W + coordinates ---
        u, v, w, x, y, z, t = load_h5(data_path, load_components="UVW", verbose=True)

        # --- Clip/pad to match known bounds if needed ---
        u, v, w, x, y, z = self._apply_downsampling(u, v, w, x, y, z)

        if not downsampled:
            # --- Final downsampling ---
            u = u[:, ::ds_ratio, ::ds_ratio, ::ds_ratio]
            v = v[:, ::ds_ratio, ::ds_ratio, ::ds_ratio]
            w = w[:, ::ds_ratio, ::ds_ratio, ::ds_ratio]
            x = x[::ds_ratio]
            y = y[::ds_ratio]
            z = z[::ds_ratio]

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.data = np.stack((u, v, w), axis=1)  # [T, 3, D, H, W]
        # print(self.data.shape)

        if mins is None or maxs is None:
            print(f"Mins and Maxs are not provided, using mins and maxs from data")
            self.u_min, self.u_max = np.min(u), np.max(u)
            self.v_min, self.v_max = np.min(v), np.max(v)
            self.w_min, self.w_max = np.min(w), np.max(w)
        else:
            self.u_min, self.u_max = mins[0], maxs[0]
            self.v_min, self.v_max = mins[1], maxs[1]
            self.w_min, self.w_max = mins[2], maxs[2]

        self.num, self.channels, self.nx, self.ny, self.nz = self.data.shape

    def _apply_downsampling(self, u, v, w, x, y, z):
        # Optional: If data has extra padding or needs to be trimmed
        # For now, it's already 4 convolution enabled
        return u, v, w, x, y, z

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        field = torch.tensor(self.data[idx], dtype=self.datatype)  # [3, D, H, W]

        if self.normalize:
            field = normalize(
                field,
                mins=torch.tensor([self.u_min, self.v_min, self.w_min]),
                maxs=torch.tensor([self.u_max, self.v_max, self.w_max]),
            )

        if self.transform:
            field = self.transform(field)

        field_padded = torch.zeros(
            (field.shape[0], field.shape[1], field.shape[2] + 2, field.shape[3] + 2),
            dtype=field.dtype,
            device=field.device,
        )
        field_padded[:, :, :-2, :-2] = field
        return {
            "field": field_padded,  # [3, D, H+2, W+2]
            "t": torch.tensor(self.t[idx]),
            "x": torch.tensor(self.x),
            "y": torch.tensor(self.y),
            "z": torch.tensor(self.z),
            "index": idx,
        }

    def image_shape(self):
        return (self.nx, self.ny + 2, self.nz + 2)

    def num_channels(self):
        return self.channels
