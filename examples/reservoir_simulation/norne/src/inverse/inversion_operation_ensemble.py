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

"""
@Author : Clement Etienam
"""

import numpy as np
import numpy.matlib
from scipy import interpolate
import torch
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from torch.utils.data import TensorDataset
from inverse.inversion_operation_gather import Add_marker2
from utils.logging_utils import setup_logging
from utils.array_utils import (
    fit_operation,
)

from utils.opm_utils import (
    Get_fault,
    read_faults,
    assign_faults,
)

logger = setup_logging(__name__)


def replace_nan_with_zero(tensor):
    """Replace all NaN entries in a tensor with zero.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor that may contain NaN values.

    Returns
    -------
    torch.Tensor
        Tensor with NaN values replaced by 0.0.
    """
    nan_mask = torch.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def Add_marker(plt, XX, YY, locc):
    """Overlay well-type markers on a 2-D matplotlib plot.

    Parameters
    ----------
    plt : module
        The ``matplotlib.pyplot`` module used for scatter plotting.
    XX : np.ndarray
        2-D array of x-coordinates for the grid.
    YY : np.ndarray
        2-D array of y-coordinates for the grid.
    locc : np.ndarray
        Array of shape ``(n_wells, 3)`` with columns ``[x_idx, y_idx, type]``;
        type 2 plots an up-triangle, others plot a down-triangle.

    Returns
    -------
    None
    """
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def clip_ensemble_params(filcc, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P, effec):
    """Clip permeability and porosity fields in an ensemble dictionary to physical bounds.

    Works for both numpy arrays and torch tensors (CPU or CUDA).

    Parameters
    ----------
    filcc : dict
        Dictionary with keys ``'PERM'`` and ``'PORO'`` holding ``np.ndarray`` or
        ``torch.Tensor`` fields.
    nx, ny, nz : int
        Grid dimensions (unused internally; kept for API consistency).
    N_ens : int
        Number of ensemble members (unused internally; kept for API consistency).
    High_K, Low_K : float | np.ndarray | torch.Tensor
        Bounds for permeability values.
    High_P, Low_P : float | np.ndarray | torch.Tensor
        Bounds for porosity values.
    effec : np.ndarray
        Effective cell mask (unused internally; kept for API consistency).

    Returns
    -------
    dict
        The same ``filcc`` dictionary with clipped ``'PERM'`` and ``'PORO'`` arrays.
    """

    def _to_scalar(x):
        """Coerce a 0-d/1-element tensor or array to a Python float."""
        if isinstance(x, torch.Tensor):
            return x.item()
        if isinstance(x, np.ndarray):
            return float(x.flat[0])
        return float(x)

    lk, hk = _to_scalar(Low_K), _to_scalar(High_K)
    lp, hp = _to_scalar(Low_P), _to_scalar(High_P)

    def _clip(a, lo, hi):
        if isinstance(a, torch.Tensor):
            # torch.clamp keeps tensor on its current device — no GPU→CPU transfer
            return torch.clamp(a, min=lo, max=hi)
        return np.clip(a, lo, hi)

    filcc["PERM"] = _clip(filcc["PERM"], lk, hk)
    filcc["PORO"] = _clip(filcc["PORO"], lp, hp)
    return filcc


def compute_data_mismatch(sim_data, measurement):
    """Compute the mean and standard deviation of per-realization RMS data mismatch.

    Parameters
    ----------
    sim_data : np.ndarray or torch.Tensor
        Simulated data of shape ``(n_obs, n_ensemble)``.
    measurement : np.ndarray or torch.Tensor
        Observed measurement vector of shape ``(n_obs,)`` or ``(n_obs, 1)``.

    Returns
    -------
    obj : float
        Mean RMS mismatch across all ensemble members.
    obj_std : float
        Standard deviation of the per-member RMS mismatch.
    obj_real : np.ndarray or torch.Tensor
        Per-member RMS mismatch array of shape ``(n_ensemble, 1)``.
    """
    is_torch = isinstance(sim_data, torch.Tensor)
    if not is_torch:
        sim_data = np.asarray(sim_data)
        measurement = np.asarray(measurement)
        reshape_fn = np.reshape
        sqrt_fn = np.sqrt
        sum_fn = np.sum
        mean_fn = np.mean
        std_fn = np.std
        zeros_fn = np.zeros
    else:
        measurement = measurement.reshape(-1, 1)
        reshape_fn = torch.reshape
        sqrt_fn = torch.sqrt
        sum_fn = torch.sum
        mean_fn = torch.mean
        std_fn = torch.std

        def zeros_fn(shape, **kwargs):
            """Create a zero tensor on the same device and dtype as sim_data.

            Parameters
            ----------
            shape : tuple of int
                Desired output tensor shape.
            **kwargs : dict
                Additional keyword arguments (ignored; kept for numpy API compatibility).

            Returns
            -------
            torch.Tensor
                Zero tensor of the specified shape on `sim_data`'s device.
            """
            return torch.zeros(shape, device=sim_data.device, dtype=sim_data.dtype)

    ne = sim_data.shape[1]
    obj_real = zeros_fn((ne, 1))
    for j in range(ne):
        noww = reshape_fn(sim_data[:, j], (-1, 1))
        obj_real[j] = sqrt_fn(sum_fn((noww - measurement) ** 2)) / measurement.shape[0]
    obj = mean_fn(obj_real).item()
    obj_std = std_fn(obj_real).item()
    return obj, obj_std, obj_real


def pinvmatt(A, tol=0):
    """Compute the truncated Moore-Penrose pseudoinverse of a matrix using SVD.

    Parameters
    ----------
    A : torch.Tensor
        2-D input matrix to invert.
    tol : float, optional
        Singular-value truncation threshold; computed automatically when 0.

    Returns
    -------
    Vt_T : torch.Tensor
        Right singular vector matrix transposed (V), truncated to rank r.
    X : torch.Tensor
        Pseudoinverse of `A` of shape ``(A.shape[1], A.shape[0])``.
    U : torch.Tensor
        Left singular vector matrix, truncated to rank r.
    """
    device = A.device
    U, S1, Vt = torch.linalg.svd(A, full_matrices=False)
    if tol == 0:
        tol = torch.max(
            A.size(0) * torch.finfo(S1.dtype).eps * torch.linalg.norm(S1, float("inf"))
        )
    r1 = torch.sum(tol < S1).item()  # Don't add 1 here!
    U = U[:, :r1]
    Vt = Vt[:r1, :]
    S1 = S1[:r1]
    S_inv = torch.diag(1.0 / S1)  # Convert to diagonal matrix
    X = Vt.t() @ S_inv @ U.t()  # Correct multiplication order
    return Vt.t().to(device), X.to(device), U.to(device)


class MinMaxScalerVectorized:
    """Min Max Scaler Vectorized."""

    def __init__(self, **kwargs):
        """Initialize the scaler with arbitrary keyword arguments stored as attributes.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments; must include ``feature_range`` tuple ``(a, b)``.
        """
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Scale a list of tensors to the configured feature range.

        Parameters
        ----------
        tensor : list of torch.Tensor
            List of tensors to stack and scale together.

        Returns
        -------
        torch.Tensor
            Stacked and min-max scaled tensor in the range ``[a, b]``.
        """
        tensor = torch.stack(tensor)
        a, b = self.feature_range
        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)
        return tensor


def load_data_numpy_2(inn, out, ndata, batch_size):
    """Create a shuffled PyTorch DataLoader from numpy input/output arrays.

    Parameters
    ----------
    inn : np.ndarray
        Input feature array of shape ``(n_samples, n_features)``.
    out : np.ndarray
        Target output array of shape ``(n_samples, n_outputs)``.
    ndata : int
        Total number of data samples (logged but not used for slicing).
    batch_size : int
        Number of samples per mini-batch.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader yielding ``(FloatTensor_x, FloatTensor_y)`` batches.
    """
    x_data = inn
    y_data = out
    logger.info(f"xtrain_data: {x_data.shape}")
    logger.info(f"ytrain_data: {y_data.shape}")
    data_tuple = (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    return DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )


# rescale_linear, rescale_linear_numpy_pytorch, rescale_linear_pytorch_numpy
# are imported from utils.array_utils above.


def Equivalent_time(tim1, max_t1, tim2, max_t2):
    """Map time steps from one simulation schedule to equivalent fractions of another.

    Parameters
    ----------
    tim1 : float
        Time step size of the reference simulation.
    max_t1 : float
        Total simulation time of the reference schedule.
    tim2 : float
        Time step size of the target simulation schedule.
    max_t2 : float
        Total simulation time of the target schedule.

    Returns
    -------
    np.ndarray
        Normalized time fractions for each target time step, clipped to ``[0, 1]``.
    """
    tk2 = tim1 / max_t1
    tc2 = np.arange(0.0, 1 + tk2, tk2)
    tc2[tc2 >= 1] = 1
    tc2 = tc2.reshape(-1, 1)  # reference scaled to 1
    tc2r = np.arange(0.0, max_t1 + tim1, tim1)
    tc2r[tc2r >= max_t1] = max_t1
    tc2r = tc2r.reshape(-1, 1)  # reference original
    func = interpolate.interp1d(tc2r.ravel(), tc2.ravel())
    tc2rr = np.arange(0.0, max_t2 + tim2, tim2)
    tc2rr[tc2rr >= max_t2] = max_t2
    tc2rr = tc2rr.reshape(-1, 1)  # reference original
    return func(tc2rr.ravel())


be_verbose = False
# fit_operation is imported from utils.array_utils above.


def ensemble_pytorch(
    param,
    nx,
    ny,
    nz,
    Ne,
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
    input_variables,
    cfg,
):
    """Assemble and normalize an ensemble of reservoir input tensors for the neural surrogate.

    Parameters
    ----------
    param : dict
        Dictionary of ensemble arrays, e.g. ``{'PERM': np.ndarray, 'PORO': np.ndarray, ...}``.
    nx : int
        Number of grid cells in the x direction.
    ny : int
        Number of grid cells in the y direction.
    nz : int
        Number of grid cells in the z direction.
    Ne : int
        Number of ensemble members.
    effective : np.ndarray
        Effective cell mask array of shape ``(nx*ny*nz,)``.
    oldfolder : str
        Path to the working directory (used for fault file parsing).
    target_min : float
        Lower bound of the normalized output range.
    target_max : float
        Upper bound of the normalized output range.
    minK : float
        Minimum permeability value for normalization.
    maxK : float
        Maximum permeability value for normalization.
    minT : float
        Minimum time value for normalization.
    maxT : float
        Maximum time value for normalization.
    minP : float
        Minimum pressure value for normalization.
    maxP : float
        Maximum pressure value for normalization.
    minQ : float
        Minimum total flow rate for normalization.
    maxQ : float
        Maximum total flow rate for normalization.
    minQw : float
        Minimum water flow rate for normalization.
    maxQw : float
        Maximum water flow rate for normalization.
    minQg : float
        Minimum gas flow rate for normalization.
    maxQg : float
        Maximum gas flow rate for normalization.
    steppi : int
        Number of simulation time steps.
    device : torch.device
        Device (CPU or GPU) to place the output tensors on.
    steppi_indices : np.ndarray
        Indices of selected time steps.
    input_variables : list of str
        Names of variables to include, e.g. ``['PERM', 'PORO', 'PINI', ...]``.
    cfg : object
        Hydra configuration object containing custom reservoir properties.

    Returns
    -------
    dict
        Dictionary mapping variable names (lowercase) to normalized ``torch.Tensor``
        of shape ``(Ne, 1, nz, nx, ny)``.
    """
    param = {
        k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in param.items()
    }

    if "FAULT" in input_variables:
        Fault = np.ones((nx, ny, nz), dtype=np.float16)
        flt = []
        for k in range(Ne):
            floatts = param["FAULT"][:, k]
            fault_temp = Get_fault(cfg.custom.FAULT_INCLUDEE)
            fault_data = read_faults(
                to_absolute_path(cfg.custom.FAULT_DATA), fault_temp
            )  # OIl
            Fault = assign_faults(fault_data, nx, ny, nz, fault_temp, floatts)
            flt.append(Fault)
        flt = np.stack(flt, axis=0)[:, None, :, :, :]
        faultz = np.stack(flt, axis=0)
        faultz = faultz / 100
    ini_ensembles = {}
    if "PERM" in input_variables:
        ini_ensembles["perm"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PORO" in input_variables:
        ini_ensembles["poro"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "PINI" in input_variables:
        ini_ensembles["pini"] = cfg.custom.PROPS.initial_pressure * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "SINI" in input_variables:
        ini_ensembles["sini"] = cfg.custom.PROPS.initial_water_saturation * np.ones(
            (Ne, 1, nz, nx, ny), dtype=np.float32
        )  # * effea[None,None,:,:,:]
    if "FAULT" in input_variables:
        ini_ensembles["fault"] = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "SGINI" in input_variables:
        ini_ensembles["sgini"] = 1e-3 * np.ones((Ne, 1, nz, nx, ny), dtype=np.float32)
    if "SOINI" in input_variables:
        ini_ensembles["soini"] = 0.8 * np.ones((Ne, 1, nz, nx, ny), dtype=np.float32)
    for kk in range(Ne):
        if "PERM" in input_variables:
            a = np.reshape(param["PERM"][:, kk], (nx, ny, nz), "F")  # * effective
        if "PORO" in input_variables:
            a1 = np.reshape(param["PORO"][:, kk], (nx, ny, nz), "F")  # * effective
        for my in range(nz):
            if "PERM" in input_variables:
                ini_ensembles["perm"][kk, 0, my, :, :] = a[:, :, my]  # Permeability
            if "PORO" in input_variables:
                ini_ensembles["poro"][kk, 0, my, :, :] = a1[:, :, my]  # Porosity
            if "FAULT" in input_variables:
                ini_ensembles["fault"][kk, 0, my, :, :] = faultz[
                    kk, 0, :, :, my
                ]  # fault
    # Initial_pressure
    if "PINI" in input_variables:
        ini_ensembles["pini"] = fit_operation(
            ini_ensembles["pini"], target_min, target_max, minP, maxP
        )
    # Permeability
    if "PERM" in input_variables:
        ini_ensembles["perm"] = fit_operation(
            ini_ensembles["perm"], target_min, target_max, minK, maxK
        )
    # Prepare the dictionary dynamically
    inn = {}
    if "PERM" in input_variables:
        inn["perm"] = torch.from_numpy(ini_ensembles["perm"]).to(
            device, dtype=torch.float32
        )
    if "PORO" in input_variables:
        inn["poro"] = torch.from_numpy(ini_ensembles["poro"]).to(
            device, dtype=torch.float32
        )
    if "PINI" in input_variables:
        inn["pini"] = torch.from_numpy(ini_ensembles["pini"]).to(
            device, dtype=torch.float32
        )
    if "SINI" in input_variables:
        inn["sini"] = torch.from_numpy(ini_ensembles["sini"]).to(
            device, dtype=torch.float32
        )
    if "FAULT" in input_variables:
        inn["fault"] = torch.from_numpy(ini_ensembles["fault"]).to(
            device, dtype=torch.float32
        )
    if "SGINI" in input_variables:
        inn["sgini"] = torch.from_numpy(ini_ensembles["sgini"]).to(
            device, dtype=torch.float32
        )

    if "SOINI" in input_variables:
        inn["soini"] = torch.from_numpy(ini_ensembles["soini"]).to(
            device, dtype=torch.float32
        )

    return inn


def Plot_2D(
    XX,
    YY,
    plt,
    nx,
    ny,
    nz,
    Truee,
    N_injw,
    N_pr,
    N_injg,
    varii,
    injectors,
    producers,
    gas_injectors,
):
    """Render a 2-D pcolormesh of a reservoir field with well markers and colorbar.

    Parameters
    ----------
    XX : np.ndarray
        2-D array of x-coordinates for the plot grid.
    YY : np.ndarray
        2-D array of y-coordinates for the plot grid.
    plt : module
        The ``matplotlib.pyplot`` module used for plotting.
    nx : int
        Number of grid cells in the x direction.
    ny : int
        Number of grid cells in the y direction.
    nz : int
        Number of grid cells in the z direction.
    Truee : np.ndarray
        Field values; either 3-D ``(nx, ny, nz)`` (averaged) or flat 1-D/2-D array.
    N_injw : int
        Number of water injector wells.
    N_pr : int
        Number of producer wells.
    N_injg : int
        Number of gas injector wells.
    varii : str
        Field label string controlling colorbar label and plot title.
    injectors : list
        Injector well descriptor list passed to ``Add_marker2``.
    producers : list
        Producer well descriptor list passed to ``Add_marker2``.
    gas_injectors : list
        Gas injector well descriptor list passed to ``Add_marker2``.

    Returns
    -------
    None
    """
    avg_2d = (
        np.mean(Truee, axis=2) if Truee.ndim == 3 else np.reshape(Truee, (nx, ny), "F")
    )
    maxii = max(avg_2d.ravel())
    minii = min(avg_2d.ravel())
    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs

    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()

    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
        plt.title("Permeability Field with well locations", fontsize=11, weight="bold")
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation -PhysicsNeMo", fontsize=11, weight="bold")
    elif varii == "water FLOW":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - FLOW", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("water saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "oil FLOW":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Flow", fontsize=11, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("oil saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "gas FLOW":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -FLOW", fontsize=11, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("gas saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "pressure FLOW":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -FLOW", fontsize=11, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("Pressure - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)

    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
