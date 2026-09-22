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

"""Helper functions for evaluating model predictions."""

import h5py


def load_predicted_data(data_path=None, filetype="h5"):
    """Load predicted data from a file.

    Parameters
    ----------
    data_path : str, optional
        Path to the file containing predicted data.
    filetype : str, optional
        Type of file to load. Currently supports "h5" (HDF5 format).
        Default is "h5".

    Returns
    -------
    numpy.ndarray
        Array containing the predicted data.
    """
    if filetype == "h5":
        hf = h5py.File(data_path, "r")
        pred_data = hf["pred"][:]

        print(f"Predicted data loaded : {pred_data.shape}!!")

        return pred_data
