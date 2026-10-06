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

"""Rotor37 cases encoded with training-only transforms."""

import json
from pathlib import Path

import numpy as np
import torch
from geometry import encode_geometry
from torch.utils.data import Dataset


class Rotor37Dataset(Dataset):
    """Encode one prepared split for GeoTransolver.

    Each case provides ``features`` with the two normalized operating
    conditions followed by the coordinate and normal PCA features, and a
    ``geometry`` context of ``geometry_points`` surface vertices with
    normalized coordinates, normals and standardized displacements from the
    training mean surface. Labeled splits add normalized ``fields`` and
    ``globals``, and the training split adds the target basis
    ``coefficients``. Encoded cases are cached in memory.
    """

    def __init__(
        self,
        data_dir,
        split,
        basis_path,
        stats_path,
        geometry_points=2048,
    ):
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        self.sample_ids = self.manifest["splits"][split]
        self.split = split
        self.stats = json.loads(Path(stats_path).read_text())
        with np.load(basis_path) as basis:
            self.basis = dict(basis)
        train_ids = self.manifest["splits"]["train"]
        if self.stats["sample_ids"] != train_ids or not np.array_equal(
            self.basis["training_sample_ids"], train_ids
        ):
            raise ValueError("Statistics and bases must come from the training split")
        vertices = len(self.basis["mean_points"])
        self.context_indices = np.linspace(0, vertices - 1, geometry_points).astype(
            np.int64
        )
        self.cache = {}

    def __len__(self):
        return len(self.sample_ids)

    def normalize(self, values, key):
        """Standardize values with the training statistics of ``key``."""
        stats = self.stats[key]
        return (values - np.asarray(stats["mean"])) / np.asarray(stats["std"])

    def denormalize_globals(self, values):
        """Return compressor outputs in source units."""
        stats = self.stats["globals"]
        return values * np.asarray(stats["std"]) + np.asarray(stats["mean"])

    def context(self, points, normals):
        """Return the sampled point context of one surface."""
        basis = self.basis
        displacement = (points - basis["mean_points"]) / basis["displacement_scale"]
        values = np.concatenate(
            (self.normalize(points, "points"), normals, displacement), axis=1
        )
        return torch.from_numpy(values[self.context_indices].astype(np.float32))

    def load(self, index):
        """Return the prepared arrays of a case."""
        path = self.manifest["samples"][str(self.sample_ids[index])]["path"]
        with np.load(self.data_dir / path) as sample:
            arrays = dict(sample)
        if not np.array_equal(arrays["quads"], self.basis["quads"]):
            raise ValueError("Every case must share the training mesh connectivity")
        return arrays

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.encode(index)
        return self.cache[index]

    def encode(self, index):
        """Encode a case with the fixed training transforms."""
        arrays = self.load(index)
        points = arrays["points"].astype(np.float64)
        normals = arrays["normals"].astype(np.float64)
        features = np.concatenate(
            (
                self.normalize(arrays["conditions"], "conditions"),
                encode_geometry(points, normals, self.basis),
            )
        )
        sample = {
            "sample_id": self.sample_ids[index],
            "features": torch.from_numpy(features.astype(np.float32)),
            "geometry": self.context(points, normals),
        }
        if "fields" in arrays:
            fields = self.normalize(arrays["fields"].astype(np.float64), "fields")
            globals_ = self.normalize(arrays["globals"].astype(np.float64), "globals")
            sample["fields"] = torch.from_numpy(fields.astype(np.float32))
            sample["globals"] = torch.from_numpy(globals_.astype(np.float32))
        if self.split == "train":
            coefficients = np.concatenate(
                (
                    self.basis["pressure_coefficients"][index],
                    self.basis["temperature_coefficients"][index],
                )
            )
            sample["coefficients"] = torch.from_numpy(coefficients.astype(np.float32))
        return sample
