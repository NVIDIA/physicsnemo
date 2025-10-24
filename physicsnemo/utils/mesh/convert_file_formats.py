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

import vtk


def convert_obj_to_vtp(input_file: str, output_file: str) -> None:
    """
    Convert an OBJ file to a VTP file.

    Args:
    - input_file (str): Path to the input OBJ file.
    - output_file (str): Path to save the converted VTP file.
    """
    reader = vtk.vtkOBJReader()
    reader.SetFileName(input_file)
    reader.Update()

    polydata = reader.GetOutput()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()


def convert_vtp_to_stl(input_file: str, output_file: str) -> None:
    """
    Convert a VTP file to an STL file.
    Scope is limited to 2D manifolds. Volumetric data is not supported.

    Args:
    - input_file (str): Path to the input VTP file.
    - output_file (str): Path to save the converted STL file.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_file)
    if not reader.CanReadFile(input_file):
        raise ValueError(f"Error: Could not read file: {input_file}")
    reader.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_file)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()


def convert_tesselated_files_in_directory(conversion_type, input_dir, output_dir):
    """
    Convert all files in a directory to a desired tesselated file format.
    Supported conversions are OBJ to VTP and VTP to STL.
    Scope is limited to 2D manifolds. Volumetric data is not supported.

    Args:
    - conversion_type (str): Type of conversion to perform. Supported values are 'obj2vtp' and 'vtp2stl'.
    - input_dir (str): Path to the directory containing input files.
    - output_dir (str): Path to the directory to save the converted files.
    """

    if conversion_type == "obj2vtp":
        src_ext = ".obj"
        dst_ext = ".vtp"
        converter = convert_obj_to_vtp
    elif conversion_type == "vtp2stl":
        src_ext = ".vtp"
        dst_ext = ".stl"
        converter = convert_vtp_to_stl
    else:
        raise NotImplementedError(
            f"Conversion type {conversion_type} is not supported."
        )

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(src_ext):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(
                output_dir, os.path.splitext(filename)[0] + dst_ext
            )
            converter(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")
    print("Conversion complete.")
