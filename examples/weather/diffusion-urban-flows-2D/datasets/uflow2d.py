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

"""Module for loading and processing 2D optical flow data from HDF5 files.

This module provides utilities for loading 2D velocity field data (u and v components)
from HDF5 files, preprocessing them with downsampling and normalization, and exposing
them as a PyTorch Dataset for training and evaluation.
"""

import h5py
import numpy as np
from torch.utils.data import Dataset
from helpers.general_helpers import combine_fields


def read_hdf5(data_path):
    """Open an HDF5 file in read mode without file locking.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file to be opened.

    Returns
    -------
    h5py.File
        An opened HDF5 file object in read mode with locking disabled.
        The file contains keys such as 'u_fluc', 'v_fluc', 'x', 'y', 't', and 'means'
        for velocity components and spatial/temporal coordinates.
    """
    import h5py

    data_arr = h5py.File(data_path, "r", locking=False)
    return data_arr


class OneObs2D(Dataset):
    """PyTorch Dataset for 2D optical flow velocity field data.

    This dataset loads 2D velocity field data (u and v components) from an HDF5 file,
    applies spatial downsampling and optional normalization, and provides samples for
    training or evaluation. The data is expected to be structured with time-dependent
    velocity components on a 2D grid.

    Attributes
    ----------
    data : np.ndarray
        Stacked velocity components with shape (time, 2, height, width) where
        the channel dimension contains [u, v] components.
    x : np.ndarray
        X-axis spatial coordinates after downsampling.
    y : np.ndarray
        Y-axis spatial coordinates after downsampling.
    t : np.ndarray
        Time coordinates for each snapshot.
    means : np.ndarray
        Mean velocity field with shape (time, 2, height, width).
    num : int
        Number of temporal snapshots.
    channels : int
        Number of velocity components (always 2 for u, v).
    nx : int
        Number of grid points in x-direction after downsampling.
    ny : int
        Number of grid points in y-direction after downsampling.
    u_min, u_max : float
        Minimum and maximum values of u velocity component.
    v_min, v_max : float
        Minimum and maximum values of v velocity component.
    """

    def __init__(
        self,
        data_path=None,
        filetype="hdf5",
        transform=None,
        ds_ratio=1,
        normalize=True,
        image_size=None,
    ):
        """Initialize the 2D optical flow dataset.

        Parameters
        ----------
        data_path : str, optional
            Path to the HDF5 data file. Must be provided.
        filetype : str, default="hdf5"
            Format of the data file. Currently only "hdf5" is supported.
        transform : callable, optional
            Optional transformation to apply to each sample (e.g., torch transforms).
        ds_ratio : int, default=1
            Downsampling ratio. Supported values are 1, 2, and 5.
            - 1 or 2: Data is cropped to (288, 96) then downsampled by ds_ratio
            - 5: Data is cropped to (300, 100) then downsampled by ds_ratio
        normalize : bool, default=True
            Whether to normalize velocity values to [-1, 1] range.
        image_size : tuple, optional
            Not currently used. Kept for API compatibility.

        Raises
        ------
        AssertionError
            If data_path is None or if file dimensions do not match expected shapes.
        NotImplementedError
            If filetype is not "hdf5" or if ds_ratio is not in [1, 2, 5].
        """

        assert data_path is not None
        self.normalize = normalize
        self.transform = transform

        if filetype == "hdf5":
            # ------------------------------------------------------------------
            data_arr = read_hdf5(data_path)
            u = np.asarray(data_arr["u_fluc"][:], dtype=np.float32)  # time, nx, ny
            v = np.asarray(data_arr["v_fluc"][:], dtype=np.float32)  # time, nx, ny
            x = np.asarray(data_arr["x"][:])
            y = np.asarray(data_arr["y"][:])
            t = np.asarray(data_arr["t"][:])
            means = np.asarray(data_arr["means"][:], dtype=np.float32)

            nx, ny = 301, 101
            assert u.shape[1:] == (nx, ny) and v.shape[1:] == (nx, ny)
            assert (
                x.shape == (nx,)
                and y.shape == (ny,)
                and t.shape == (u.shape[0], 1)
                and t.shape == (v.shape[0], 1)
            )
            assert means.shape[1:] == (nx, ny)
            # ------------------------------------------------------------------
            # skip the wrongly interpolated data
            # skip_index = [185, 2490, 2718]
            # u = remove_indices_from_array(array=u, indices=skip_index)
            # v = remove_indices_from_array(array=v, indices=skip_index)

            if ds_ratio == 5:
                u = u[:, :-1, :-1]
                v = v[:, :-1, :-1]
                means = means[:, :-1, :-1]
                x = x[:-1]
                y = y[:-1]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (300, 100)
                # shape = (300,100) -> (60,20) # only use 2 layers

            elif ds_ratio == 2 or ds_ratio == 1:
                u = u[:, :-13, :-5]
                v = v[:, :-13, :-5]
                means = means[:, :-13, :-5]
                x = x[:-13]
                y = y[:-5]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (288, 96)
                # shape = (288,96) -> (144,48) # only use upto 4 layers

            else:
                print(f"ds_ratio {ds_ratio} not supported")
                raise NotImplementedError

            # ------------------------------------------------------------------

            # downsampling by ds_ratio
            u = u[:, ::ds_ratio, ::ds_ratio]
            v = v[:, ::ds_ratio, ::ds_ratio]

            assert u.shape[1:] == v.shape[1:]  # == image_size

            self.x = x[::ds_ratio]
            self.y = y[::ds_ratio]
            self.t = t
            self.means = means[:, ::ds_ratio, ::ds_ratio]
            self.data = np.stack((u, v), axis=1)  # time, nc, nx, ny

            self.num, self.channels, self.nx, self.ny = self.data.shape

            self.u_min, self.u_max = np.min(u), np.max(u)
            self.v_min, self.v_max = np.min(v), np.max(v)

            # ------------------------------------------------------------------------------
            assert (
                self.data.shape[1:] == (2, nx // ds_ratio, ny // ds_ratio)
                and self.data.dtype == np.float32
            )

        else:
            raise NotImplementedError

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of temporal snapshots in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single velocity field sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve. Must be in range [0, len(self)).

        Returns
        -------
        np.ndarray or torch.Tensor
            Velocity field at the requested index with shape (2, height, width) containing
            [u, v] components. If normalization is enabled, values are in [-1, 1] range.
            If a transform is provided, the output is transformed accordingly.
        """
        image = self.data[idx]
        if self.normalize:
            image = self.__normalize(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __normalize(self, x):
        """Normalize velocity field to [-1, 1] range using min-max scaling.

        Performs min-max normalization on the u and v components independently,
        then linearly maps the result from [0, 1] to [-1, 1] range.

        Parameters
        ----------
        x : np.ndarray
            Velocity field with shape (2, height, width) containing [u, v] components
            in their original data range.

        Returns
        -------
        np.ndarray
            Normalized velocity field with shape (2, height, width) where values
            are in the range [-1, 1]. Each component is normalized using its
            respective min and max values computed over the entire dataset.
        """

        # x shape = (2, h, w)
        eps = 1e-9
        center = np.array([self.u_min, self.v_min]).reshape((2, 1, 1))
        scale = np.array([self.u_max - self.u_min, self.v_max - self.v_min]).reshape(
            (2, 1, 1)
        )
        x_scaled = (x - center) / (scale + eps)
        return (2 * x_scaled) - 1

    def num_channels(self):
        """Number of channels in the datasets"""
        return self.channels

    def image_shape(self):
        """Shape of the 2D image"""
        return (self.nx, self.ny)


def get_data_for_evaluation(
    data_path=None, dim="2D", Train=False, Test=False, ds_ratio=1
):
    """Load flow field data for evaluation purposes.

    Parameters
    ----------
    data_path : str, optional
        Path to the HDF5 data file.
    dim : str, optional
        Dimensionality of the data. Default is "2D".
    Train : bool, optional
        If True, load training data. Default is False.
    Test : bool, optional
        If True, load test data. Default is False.
    ds_ratio : int, optional
        Downsampling ratio. Default is 1 (no downsampling).

    Returns
    -------
    tuple
        A tuple containing (data, x, y, t) where data is the combined
        velocity fields and x, y, t are the spatial and temporal coordinates.
    """
    if dim == "2D":
        if Train:
            OneObs = OneObs2D(data_path=data_path)
            data = OneObs.data
            x, y, t = OneObs.x, OneObs.y, OneObs.t
            print(f"Train data loaded: {data.shape} !!")

        if Test:
            data = h5py.File(data_path)
            u = data["u_fluc"][:]
            v = data["v_fluc"][:]
            x = data["x"][:]
            y = data["y"][:]
            t = data["t"][:]

            if ds_ratio == 1:
                u = u[:, :-13, :-5]
                v = v[:, :-13, :-5]
                x = x[:-13]
                y = y[:-5]

            assert u.shape[1:] == v.shape[1:] == (288, 96)
            # shape = (288,96) -> (144,48) # only use upto 4 layers

            data = combine_fields(u=u, v=v)
            print(f"Test data loaded: {data.shape} !!")

    return data, x, y, t


def get_data_for_evaluation_with_min_max(
    data_path=None, dim="2D", Train=False, Test=False, ds_ratio=1
):
    """Load flow field data for evaluation with normalization min/max values.

    Parameters
    ----------
    data_path : str, optional
        Path to the HDF5 data file.
    dim : str, optional
        Dimensionality of the data. Default is "2D".
    Train : bool, optional
        If True, load training data. Default is False.
    Test : bool, optional
        If True, load test data. Default is False.
    ds_ratio : int, optional
        Downsampling ratio. Default is 1 (no downsampling).

    Returns
    -------
    tuple
        A tuple containing (data, x, y, t, u_min, u_max, v_min, v_max) where
        data is the combined velocity fields, x, y, t are coordinates, and
        the min/max values are for denormalization.
    """
    if dim == "2D":
        OneObs = OneObs2D(data_path=data_path)
        u_min, u_max = OneObs.u_min, OneObs.u_max
        v_min, v_max = OneObs.v_min, OneObs.v_max

        print(f"u_min = {u_min}, u_max = {u_max}, u_min = {v_min}, u_max = {v_max}")

        if Train:
            data = OneObs.data
            x, y, t = OneObs.x, OneObs.y, OneObs.t
            print(f"Train data loaded: {data.shape} !!")

        if Test:
            data = h5py.File(data_path)
            u = data["u_fluc"][:]
            v = data["v_fluc"][:]
            t = data["t"][:]
            if ds_ratio == 1:
                u = u[:, :-13, :-5]
                v = v[:, :-13, :-5]

                nx, ny = u.shape[1:]
                assert u.shape[1:] == v.shape[1:] == (nx, ny) == (288, 96)
                # shape = (288,96) -> (144,48) # only use upto 4 layers

                data = combine_fields(u=u, v=v)
                print(f"Test data loaded: {data.shape} !!")

    return data, x, y, t, u_min, u_max, v_min, v_max
