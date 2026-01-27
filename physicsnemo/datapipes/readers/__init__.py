# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
Readers module - Data source interfaces for loading raw data.

Readers are responsible for:
- Loading data from various sources (HDF5, Zarr, NumPy, etc.)
- Converting to torch tensors
- Async CPU->GPU transfers with optional prefetching
- Returning Sample objects ready for the transform pipeline
"""

from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from physicsnemo.datapipes.readers.numpy import NumpyReader
from physicsnemo.datapipes.readers.tensorstore_zarr import TensorStoreZarrReader
from physicsnemo.datapipes.readers.vtk import VTKReader
from physicsnemo.datapipes.readers.zarr import ZarrReader

__all__ = [
    "Reader",
    "HDF5Reader",
    "ZarrReader",
    "NumpyReader",
    "VTKReader",
    "TensorStoreZarrReader",
]
