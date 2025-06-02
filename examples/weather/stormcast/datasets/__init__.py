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

import importlib
import pkgutil

from .dataset import StormCastDataset


# Find StormCastDataset implementations found in files in the datasets directory
# and list them by module and name in the dataset_classes dict
dataset_modules = pkgutil.iter_modules(["datasets"])
dataset_modules = [mod.name for mod in dataset_modules if mod.name != "dataset"]
dataset_classes = {}
for mod_name in dataset_modules:
    module = importlib.import_module(f"datasets.{mod_name}")
    for (name, member) in module.__dict__.items():
        if (
            name != "StormCastDataset"
            and isinstance(member, type)
            and issubclass(member, StormCastDataset)
        ):
            dataset_classes[f"{mod_name}.{name}"] = member
