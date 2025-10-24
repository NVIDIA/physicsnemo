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

import os
import glob
import numpy as np
import pyvista as pv
from torch.utils.data import Dataset, DataLoader


class ProcessedVTPDataset(Dataset):
    """
    Dataset for processed VTP files.
    """

    def __init__(self, vtp_dir, norm_dir, normalize=True, return_mesh=False):
        self.files = sorted(glob.glob(os.path.join(vtp_dir, "processed_*.vtp")))
        self.normalize = normalize
        self.return_mesh = return_mesh

        # Load normalization stats
        self.p_mean = np.load(os.path.join(norm_dir, "pMeanTrim_mean.npy"))
        self.p_std = np.load(os.path.join(norm_dir, "pMeanTrim_std.npy"))
        self.shear_mean = np.load(
            os.path.join(norm_dir, "wallShearStressMeanTrim_mean.npy")
        )
        self.shear_std = np.load(
            os.path.join(norm_dir, "wallShearStressMeanTrim_std.npy")
        )
        self.area_mean = np.load(os.path.join(norm_dir, "Area_mean.npy"))
        self.area_std = np.load(os.path.join(norm_dir, "Area_std.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vtp_path = self.files[idx]
        mesh = pv.read(vtp_path)

        # True fields
        p = mesh.point_data["pMeanTrim"]
        shear = mesh.point_data["wallShearStressMeanTrim"]  # shape (N,3)
        area = mesh.cell_data["Area"]
        normals = mesh.point_data["Normals"]

        # Predicted fields
        p_pred_xmgn = mesh.point_data["pMeanTrimPred_xmgn"]
        p_pred_fignet = mesh.point_data["pMeanTrimPred_fignet"]
        p_pred_domino = mesh.point_data["pMeanTrimPred_domino"]

        shear_pred_xmgn = mesh.point_data["wallShearStressMeanTrimPred_xmgn"]
        shear_pred_fignet = mesh.point_data["wallShearStressMeanTrimPred_fignet"]
        shear_pred_domino = mesh.point_data["wallShearStressMeanTrimPred_domino"]

        # Add second dimension for single-dim arrays
        p = p.reshape(-1, 1)
        area = area.reshape(-1, 1)
        p_pred_xmgn = p_pred_xmgn.reshape(-1, 1)
        p_pred_fignet = p_pred_fignet.reshape(-1, 1)
        p_pred_domino = p_pred_domino.reshape(-1, 1)

        if self.normalize:
            # Normalize
            p = (p - self.p_mean) / self.p_std
            shear = (shear - self.shear_mean) / self.shear_std
            area = (area - self.area_mean) / self.area_std

            p_pred_xmgn = (p_pred_xmgn - self.p_mean) / self.p_std
            p_pred_fignet = (p_pred_fignet - self.p_mean) / self.p_std
            p_pred_domino = (p_pred_domino - self.p_mean) / self.p_std

            shear_pred_xmgn = (shear_pred_xmgn - self.shear_mean) / self.shear_std
            shear_pred_fignet = (shear_pred_fignet - self.shear_mean) / self.shear_std
            shear_pred_domino = (shear_pred_domino - self.shear_mean) / self.shear_std

        result = {
            "p": p,
            "shear": shear,
            "area": area,
            "normals": normals,
            "p_pred_xmgn": p_pred_xmgn,
            "p_pred_fignet": p_pred_fignet,
            "p_pred_domino": p_pred_domino,
            "shear_pred_xmgn": shear_pred_xmgn,
            "shear_pred_fignet": shear_pred_fignet,
            "shear_pred_domino": shear_pred_domino,
            "id": os.path.basename(vtp_path)
            .replace("processed_", "")
            .replace(".vtp", ""),
        }
        if self.return_mesh:
            result["mesh"] = mesh
        return result
