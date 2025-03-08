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

import json
import os
from typing import Any, Dict

import torch

try:
    import vtk
except ImportError:
    raise ImportError("vtk package is required. Install with pip install vtk.")


def read_vtp_file(file_path: str) -> Any:
    """
    Read a VTP file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the VTP file.

    Returns
    -------
    vtkPolyData
        The polydata read from the VTP file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtp extension
    if not file_path.endswith(".vtp"):
        raise ValueError(f"Expected a .vtp file, got {file_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata


def save_json(var: Dict[str, torch.Tensor], file: str) -> None:
    """
    Saves a dictionary of tensors to a JSON file.

    Parameters
    ----------
    var : Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    file : str
        Path to the output JSON file.
    """
    var_list = {k: v.numpy().tolist() for k, v in var.items()}
    with open(file, "w") as f:
        json.dump(var_list, f)


def load_json(file: str) -> Dict[str, torch.Tensor]:
    """
    Loads a JSON file into a dictionary of PyTorch tensors.

    Parameters
    ----------
    file : str
        Path to the JSON file.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    """
    with open(file, "r") as f:
        var_list = json.load(f)
    var = {k: torch.tensor(v, dtype=torch.float) for k, v in var_list.items()}
    return var
