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

"""Auto-discovery registry of FGNDataset subclasses.

Mirrors ``examples/weather/stormcast/datasets/__init__.py``:
scan every module in this package and register any class that subclasses
``FGNDataset`` under the key ``"<module_name>.<ClassName>"``.

Usage::

    from datasets import dataset_classes
    cls = dataset_classes["arco.ArcoFGNDataset"]
"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil

from .dataset import FGNDataset

_pkg_dir = str(pathlib.Path(__file__).parent)
dataset_classes: dict[str, type[FGNDataset]] = {}

for _mod_info in pkgutil.iter_modules([_pkg_dir]):
    if _mod_info.name == "dataset":
        continue
    _module = importlib.import_module(f"datasets.{_mod_info.name}")
    for _name, _member in _module.__dict__.items():
        if (
            _name != "FGNDataset"
            and isinstance(_member, type)
            and issubclass(_member, FGNDataset)
        ):
            dataset_classes[f"{_mod_info.name}.{_name}"] = _member
