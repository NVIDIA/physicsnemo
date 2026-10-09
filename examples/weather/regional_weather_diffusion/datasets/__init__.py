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

import importlib
import pkgutil

from .data_loader_hrrr_era5 import HrrrEra5Dataset
from .dataset import (
    LoaderSpec,
    StormCastDataset,
    StormCastDataSource,
    default_torch_loader,
    flatten_group_collate,
)
from .mock import MockDataset

# Find StormCastDataSource implementations (map-style StormCastDataset
# subclasses, or lower-level StormCastDataSource implementations) in any
# module under this package -- including a custom dataset a user drops in --
# and list them by "<module>.<ClassName>", matching the `dataset.name` config
# field. This is how a custom dataset is picked up with no registration step
# beyond adding the file; see the README's "Adding Custom Datasets" section.
dataset_modules = pkgutil.iter_modules(__path__)
dataset_modules = [mod.name for mod in dataset_modules if mod.name != "dataset"]
dataset_classes: dict[str, type[StormCastDataSource]] = {}
for mod_name in dataset_modules:
    module = importlib.import_module(f"datasets.{mod_name}")
    for name, member in module.__dict__.items():
        if (
            name not in ("StormCastDataset", "StormCastDataSource")
            and isinstance(member, type)
            and issubclass(member, StormCastDataSource)
        ):
            dataset_classes[f"{mod_name}.{name}"] = member

__all__ = [
    "HrrrEra5Dataset",
    "LoaderSpec",
    "MockDataset",
    "StormCastDataSource",
    "StormCastDataset",
    "dataset_classes",
    "default_torch_loader",
    "flatten_group_collate",
]
