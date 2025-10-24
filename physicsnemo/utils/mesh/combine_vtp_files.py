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


from typing import List

from vtk import (
    vtkAppendPolyData,
    vtkPolyData,
    vtkXMLPolyDataReader,
    vtkXMLPolyDataWriter,
)


def combine_vtp_files(input_files: List[str], output_file: str) -> None:
    """
    Combine multiple VTP files into a single VTP file.

    Args:
    - input_files (list[str]): List of paths to the input VTP files to be combined.
    - output_file (str): Path to save the combined VTP file.
    """
    reader = vtkXMLPolyDataReader()
    append = vtkAppendPolyData()

    for file in input_files:
        reader.SetFileName(file)
        reader.Update()
        polydata = vtkPolyData()
        polydata.ShallowCopy(reader.GetOutput())
        append.AddInputData(polydata)

    append.Update()

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(append.GetOutput())
    writer.Write()
