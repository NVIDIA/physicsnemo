# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File formats for checkpointing


# Version 1:
format = zip file
contains

```
net_state.pth
batch_info.json
model.json: str # json-serialized ModelConfigV1 (if version == 1)
metadata.json
    version: int
```

# other files are not interpreted by cbottle but can be used for other purposes
# (like in custom TrainingLoops)
"""

import json
import warnings
import zipfile
from typing import Literal

import datasets.base
import models
import torch

from physicsnemo.models.healda import ModelConfigV1

current_version = 1


class Checkpoint:
    """A checkpoint object

    This is similar to ZipFile, but with convenience methods for reading and
    writing models.
    """

    def __init__(self, f, mode: Literal["w", "r"] = "r"):
        # Set zip to None in case the zipfile fails to open
        # this will be used in __del__
        self._zip = None
        self._zip = zipfile.ZipFile(f, mode)

    def write_model(self, net: torch.nn.Module):
        with self._zip.open("net_state.pth", "w", force_zip64=True) as f:
            torch.save(net.state_dict(), f)

    def read_model(self, net=None, map_location=None) -> torch.nn.Module:
        """Read the model from the checkpoint

        Args:
            net: If provided, the state dict will be loaded into this net.
            Otherwise, a new model will be created.
        """
        try:
            metadata = json.loads(self._zip.read("metadata.json").decode())
        except KeyError:
            warnings.warn("Old checkpoint format detected. Falling back to old format.")
            with self._zip.open("net_state.pth", "r") as f:
                return torch.load(f, weights_only=False)["net"]

        # new checkpoint format
        if metadata["version"] != current_version:
            raise ValueError(f"Unsupported checkpoint version: {metadata['version']}")

        model_config = self.read_model_config()
        if net is None:
            net = models.get_model(model_config)

        with self._zip.open("net_state.pth", "r") as f:
            net.load_state_dict(
                torch.load(f, weights_only=True, map_location=map_location)
            )
        return net

    def read_model_config(self) -> ModelConfigV1:
        return ModelConfigV1.loads(self._zip.open("model.json").read())

    def read_model_state_dict(self) -> dict:
        with self._zip.open("net_state.pth", "r") as f:
            return torch.load(f, weights_only=True)

    def write_batch_info(self, batch_info: datasets.base.BatchInfo):
        d = {
            "channels": batch_info.channels,
            "time_step": batch_info.time_step,
            "time_unit": batch_info.time_unit.name,  # enums don't serialize nicely
        }
        if batch_info.scales is not None:
            d["scales"] = list(batch_info.scales)
        if batch_info.center is not None:
            d["center"] = list(batch_info.center)
        self._zip.writestr("batch_info.json", json.dumps(d))

    def read_batch_info(self) -> datasets.base.BatchInfo:
        with self._zip.open("batch_info.json", "r") as f:
            d = json.loads(f.read())
            scales = d.pop("scales", None)
            center = d.pop("center", None)
            if d["time_unit"] == "":  # backwards compatibility
                time_unit = datasets.base.TimeUnit.HOUR
            elif d["time_unit"] == "MINUTE":
                time_unit = datasets.base.TimeUnit.MINUTE
            elif d["time_unit"] == "HOUR":
                time_unit = datasets.base.TimeUnit.HOUR
            else:
                time_unit_dict = {v.value: v for v in datasets.base.TimeUnit}
                time_unit = time_unit_dict[d["time_unit"]]
            return datasets.base.BatchInfo(
                time_unit=time_unit,
                scales=scales,
                center=center,
                channels=d["channels"],
            )

    def write_model_config(self, model_config: ModelConfigV1):
        self._zip.writestr("model.json", model_config.dumps())

    def open(self, name, mode: Literal["w", "r"] = "r"):
        return self._zip.open(name, mode, force_zip64=True)

    def close(self):
        if self._zip.mode == "w":
            self._zip.writestr(
                "metadata.json", json.dumps({"version": current_version})
            )
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        if self._zip is not None and self._zip.fp:
            self.close()
