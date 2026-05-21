# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
