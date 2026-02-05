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

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
import yaml

import numpy as np
import pandas as pd
import pyvista as pv
import webdataset as wds

import torch
from torch.utils.data import Dataset

from src.data.base_datamodule import WebdatasetDataModule
from src.data.components.preprocessor_utils import (
    ComposePreprocessors,
    UnitGaussianNormalizer,
)
from src.data.components.webdataset_utils import from_numpy, split_by_node_equal
from src.data.mesh_utils import (
    compute_drag_coefficient,
)


# TODO: need to update/check these values.
# DrivAerNet dataset
# Air density = 1.205 kg/m^3
# Stream velocity = 30 m/s
DRIVAERNET_AIR_DENSITY = 1.205
DRIVAERNET_STREAM_VELOCITY = 30.0
DRIVAERNET_AIR_COEFF = 2 / (DRIVAERNET_AIR_DENSITY * DRIVAERNET_STREAM_VELOCITY**2)

# DrivAerNet pressure mean and std
DRIVAERNET_PRESSURE_MEAN = -94.5
DRIVAERNET_PRESSURE_STD = 117.25


class DrivAerNetPreprocessor:
    """DrivAerNet preprocessor"""

    KEY_MAPPING: dict[str, str] = {
        "Average Cd": "c_d",
        "Average Cl": "c_l",
        "Average Cl_f": "c_lf",
        "Average Cl_r": "c_lr",
        "pressure": "time_avg_pressure",
    }

    def __init__(self, num_points: int = 16384) -> None:
        self.num_points = num_points
        self.normalizer = UnitGaussianNormalizer(
            DRIVAERNET_PRESSURE_MEAN,
            DRIVAERNET_PRESSURE_STD,
        )

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        sample = {self.KEY_MAPPING.get(k, k): v for k, v in sample.items()}
        # Remove unnecessary keys
        if "design" in sample:
            sample.pop("design")

        if self.num_points > 0:
            # Randomly sample points
            vertices = sample["cell_centers"]
            point_sample_idx = np.random.choice(
                len(vertices), self.num_points, replace=True
            )

            for k, v in sample.items():
                if (isinstance(v, np.ndarray) and v.size > 1) or (
                    isinstance(v, torch.Tensor) and v.numel() > 1
                ):
                    sample[k] = v[point_sample_idx]

        # compute bbox center
        vertices = sample["cell_centers"]
        obj_min = np.min(vertices, axis=0)
        obj_max = np.max(vertices, axis=0)
        obj_center = (obj_min + obj_max) / 2.0
        vertices = vertices - obj_center
        sample["cell_centers"] = vertices
        sample["time_avg_pressure_whitened"] = self.normalizer.encode(
            sample["time_avg_pressure"]
        )

        return sample


class DrivAerNetDragPreprocessor:
    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        # Compute drag coefficient using area, normal, pressure and wall shear stress
        drag_coef = compute_drag_coefficient(
            sample["cell_normals"],
            sample["cell_areas"],
            DRIVAERNET_AIR_COEFF / sample["proj_area_x"],
            sample["time_avg_pressure"],
            sample["wall_shear_stress"],
        )
        # sample["c_d"] is computed on a finer mesh and the newly computed drag is on a coarser mesh so they are not equal
        sample["c_d_computed"] = drag_coef

        return sample


class DrivAerNetDataset(Dataset):
    """DrivAerNet dataset.

    For more information, see DrivAerNet code:
    https://github.com/Mohamedelrefaie/DrivAerNet
    """

    def __init__(
        self,
        data_path: str | Path,
        phase: str,
        coeff_filename: str = "coefficients.csv",
        preprocessors: Iterable[Callable] = None,
    ) -> None:
        """Initializes the dataset."""

        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError(
                f"Path {self.data_path} does not exist or is not a directory."
            )
        self.p_vtk_path = self.data_path / "SurfacePressureVTK"
        self.wss_vtk_path = self.data_path / "WallShearStressVTK"

        self.phase = phase.lower()
        phases = ["train", "val", "test"]
        if phase not in phases:
            raise ValueError(f"{phase = } is not supported, must be one of {phases}.")

        if preprocessors is None:
            preprocessors = []
        self.preprocessors = ComposePreprocessors(list(preprocessors))

        # Load phase ids used to select a corresponding data split.
        with open(self.data_path / f"{phase}_design_ids.txt", encoding="utf-8") as f:
            file_ids = set(l.rstrip() for l in f)

        # Read coefficients file which contains Cd, Cl etc.
        coeffs = pd.read_csv(self.data_path / coeff_filename, index_col="Design")
        coeffs = coeffs[coeffs.index.isin(file_ids)]

        # Read projected areas file which is in YAML-like format with entries that look like:
        # combined_DrivAer_F_D_WM_WW_1234.stl: 2.574603830871618
        with open(self.data_path / "projected_areas.txt", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        proj_areas = pd.DataFrame.from_dict(
            {k.removeprefix("combined_").removesuffix(".stl"): v for k, v in y.items()},
            orient="index",
            columns=["proj_area_x"],
        )
        # Merge projected areas into the coeffs dataframe.
        coeffs = coeffs.join(proj_areas)
        # There are 10 entries missing in projected_areas.txt.
        # Check them by running `coeffs[coeffs.proj_area.isna()]`
        # train: DrivAer_F_D_WM_WW_0132, 0797, 1118, 1421, 1556, 1891, 2353, 2459.
        # val: DrivAer_F_D_WM_WW_0603, 3199.
        coeffs = coeffs.dropna()
        num_missing = {"train": 8, "val": 2, "test": 0}
        assert len(file_ids) - len(coeffs) == num_missing[phase], (
            f"{len(file_ids)=} {len(coeffs)=}"
        )

        coeffs.sort_index(inplace=True)
        self.coeffs = coeffs

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, index) -> dict[str, Any]:
        if not (0 <= index < len(self)):
            raise IndexError(f"Invalid {index = } expected in [0, {len(self)})")

        coeffs = self.coeffs.iloc[index]

        key = coeffs.name

        # Read pressure and WSS data.
        p_vtk_file = self.p_vtk_path / (key + ".vtk")
        mesh = pv.read(p_vtk_file)

        # wss_vtk_file = self.wss_vtk_path / (key + ".vtk")
        # wss = pv.read(wss_vtk_file).point_data["wallShearStress"]

        # cchoy: Cell pressure not defined on DrivAerNet. Use point pressure.
        cell_centers = np.array(mesh.points)
        pressure = mesh.point_data["p"]

        sample = {
            "cell_centers": cell_centers,
            **coeffs.to_dict(),
            "pressure": pressure,
            "design": key,
        }

        return self.preprocessors(sample)


class DrivAerNetDataModule(WebdatasetDataModule):
    """DrivAerNet data module."""

    def __init__(
        self,
        data_path: str | Path,
        preprocessors: Iterable[Callable] = None,
        **kwargs,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.is_dir(), f"{data_path} is not a directory."
        assert (data_path / "train.tar").exists(), (
            f"{data_path} does not contain train.tar"
        )
        self.data_path = data_path
        if preprocessors is None:
            preprocessors = []
        self.preprocessors = ComposePreprocessors(preprocessors)
        self._train_dataset = self._create_dataset("train")
        self._val_dataset = self._create_dataset("val")
        self._test_dataset = self._create_dataset("test")
        for preproc in preprocessors:
            if hasattr(preproc, "normalizer"):
                self.normalizer = preproc.normalizer
        else:
            self.normalizer = UnitGaussianNormalizer(
                mean=DRIVAERNET_PRESSURE_MEAN, std=DRIVAERNET_PRESSURE_STD
            )

    def _create_dataset(self, prefix: str) -> wds.DataPipeline:
        # Create dataset with the processing pipeline.
        dataset = wds.DataPipeline(
            wds.SimpleShardList([str(self.data_path / f"{prefix}.tar")]),
            wds.tarfile_to_samples(),
            split_by_node_equal,
            wds.map(lambda x: from_numpy(x, "npz")),
            wds.map(self.preprocessors),
        )
        return dataset

    def encode(self, x):
        return self.normalizer.encode(x)

    def decode(self, x):
        return self.normalizer.decode(x)


def test_drivaernet_datamodule(data_path: str):
    preprocs = [DrivAerNetPreprocessor()]

    dm = DrivAerNetDataModule(data_path, preprocessors=preprocs)

    for x in dm.train_dataloader():
        print(x)
        break


def test_drivaernet_dataset(data_path: str, phase: str, size: int):
    dset = DrivAerNetDataset(data_path, phase)
    assert len(dset) == size

    x = dset[0]
    assert isinstance(x, dict)


if __name__ == "__main__":
    import fire

    fire.Fire(test_drivaernet_datamodule)
