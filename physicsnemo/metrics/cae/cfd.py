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

from typing import List, Tuple, Union

import numpy as np
import pyvista as pv
import torch
from numpy.fft import fft, fftfreq


def compute_frontal_area(mesh: pv.PolyData, direction: str = "x"):
    """
    Compute frontal area of a given mesh
    Ref: https://github.com/pyvista/pyvista/discussions/5211#discussioncomment-7794449

    Parameters:
    -----------
    mesh: pv.PolyData
        Input mesh
    direction: str, optional
        Direction to project the area. Defaults to "x".

    Raises:
    -------
    ValueError: Only supports x, y and z projection for computing frontal area.

    Returns:
    --------
    frontal area: float
        Frontal area of the mesh in the given direction
    """
    try:
        import shapely  # noqa: F401 for docs
    except ImportError:
        raise ImportError(
            "These metrics require shapely, install it using `pip install shapely`."
        )

    direction_map = {
        "x": ((1, 0, 0), [1, 2]),
        "y": ((0, 1, 0), [0, 2]),
        "z": ((0, 0, 1), [0, 1]),
    }

    if direction not in direction_map:
        raise ValueError("Direction must be x, y or z only")

    normal, indices = direction_map[direction]
    areas_proj = mesh.project_points_to_plane(origin=(0, 0, 0), normal=normal)
    merged = shapely.union_all(
        [
            shapely.Polygon(
                np.stack(
                    [
                        areas_proj.points[tri, indices[0]],
                        areas_proj.points[tri, indices[1]],
                    ],
                    axis=1,
                )
            )
            for tri in areas_proj.triangulate().regular_faces
        ]
    )

    return merged.area


def compute_force_coefficients(
    normals: np.ndarray,
    area: np.ndarray,
    coeff: float,
    p: np.ndarray,
    wss: np.ndarray,
    force_direction: np.ndarray = np.array([1, 0, 0]),
):
    """
    Computes force coefficients for a given mesh. Output includes the pressure and skin
    friction components. Can be used to compute lift and drag.
    For drag, use the `force_direction` as the direction of the motion,
    e.g. [1, 0, 0] for flow in x direction.
    For lift, use the `force_direction` as the direction perpendicular to the motion,
    e.g. [0, 1, 0] for flow in x direction and weight in y direction.

    Parameters:
    -----------
    normals: np.ndarray
        The surface normals on cells of the mesh
    area: np.ndarray
        The surface areas of each cell
    coeff: float
        Reciprocal of dynamic pressure times the frontal area, i.e. 2/(A * rho * U^2)
    p: np.ndarray
        Pressure distribution on the mesh (on each cell)
    wss: np.ndarray
        Wall shear stress distribution on the mesh (on each cell)
    force_direction: np.ndarray
        Direction to compute the force, default is np.array([1, 0, 0])

    Returns:
    --------
    c_total: float
        Computed total force coefficient
    c_p: float
        Computed pressure force coefficient
    c_f: float
        Computed skin friction coefficient
    """

    # Compute coefficients
    c_p = coeff * np.sum(np.dot(normals, force_direction) * area * p)
    c_f = -coeff * np.sum(np.dot(wss, force_direction) * area)

    # Compute total force coefficients
    c_total = c_p + c_f

    return c_total, c_p, c_f


def dominant_freq_calc(
    signal: List[Union[int, float]], sample_spacing: float = 1
) -> float:
    """
    Compute the dominant frequency in the signal

    Parameters:
    -----------
    signal : List[Union[int, float]]
        Signal
    sample_spacing : float
        Sample spacing, (inverse of sampling rate of the signal), by default 1

    Returns:
    --------
    dominant_freq : float
        Computed dominant frequency
    """
    N = len(signal)
    yf = fft(signal)

    mag = np.abs(yf)
    mag[0] = 0
    dom_idx = np.argmax(mag)
    dom_freq = fftfreq(N, sample_spacing)[dom_idx]

    return np.abs(dom_freq)


def compute_p_q_r(
    velocity_grad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the P, Q and R invariants of the velocity gradient tensor.
    The P, Q and R are normalized. Uses Finite Difference to compute the gradients.

    Parameters:
    -----------
    velocity_grad : torch.Tensor
        3D Velocity gradient tensor (N, 3, 3, nx, ny, nz).

    Reference:
    ----------
    https://ntrs.nasa.gov/api/citations/19960028952/downloads/19960028952.pdf

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Computed P, Q and R.
    """

    bs = velocity_grad.shape[0]
    J = velocity_grad.permute(0, 3, 4, 5, 1, 2).reshape(bs, -1, 3, 3)
    strain = 0.5 * (J + torch.permute(J, (0, 1, 3, 2)))

    # Combine the points across all batches
    J = J.reshape(-1, 3, 3)
    strain = strain.reshape(-1, 3, 3)

    # Compute J^2 and J^3 for each 3x3 matrix
    J2 = torch.bmm(J, J)
    J3 = torch.bmm(J, J2)
    strain2 = torch.bmm(strain, strain)

    # Reshape back to have batch dimension
    J = J.reshape(bs, -1, 3, 3)
    J2 = J2.reshape(bs, -1, 3, 3)
    J3 = J3.reshape(bs, -1, 3, 3)
    strain2 = strain2.reshape(bs, -1, 3, 3)

    # Compute traces
    trace_J1 = J.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_J2 = J2.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_J3 = J3.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_strain2 = strain2.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Compute P, Q and R invariants
    P = -1 * trace_J1
    Q = -0.5 * trace_J2
    R = -1 / 3 * trace_J3

    # Normalize P, Q and R
    P = P / torch.mean(trace_strain2**0.5, dim=-1, keepdim=True)
    Q = Q / torch.mean(trace_strain2, dim=-1, keepdim=True)
    R = R / torch.mean(trace_strain2**1.5, dim=-1, keepdim=True)

    return P, Q, R


def compute_tke_spectrum(
    field: np.ndarray, length: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the turbulent kinetic energy spectrum

    Parameters:
    -----------
    field : np.ndarray
        Velocity tensor (3, nx, ny, nz)
    length : float, optional
        Length of the domain. Defaults to None, in which case, the spacing is computed
        asuming bounds of 2*pi.

    Reference:
    ----------
    Pope, Stephen B. "Turbulent flows." Measurement Science and Technology 12.11 (2001): 2020-2021.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Wave numbers and TKE raw and binned.
    """

    if length is None:
        lx, ly, lz = 2 * np.pi, 2 * np.pi, 2 * np.pi

    u, v, w = field[0, :, :, :], field[1, :, :, :], field[2, :, :, :]

    nx = len(u[:, 0, 0])
    ny = len(v[0, :, 0])
    nz = len(w[0, 0, :])

    nt = nx * ny * nz

    # Compute FFT of the velocity components
    uhat = np.fft.fftn(u) / nt
    vhat = np.fft.fftn(v) / nt
    what = np.fft.fftn(w) / nt
    uhat_conj = np.conjugate(uhat)
    vhat_conj = np.conjugate(vhat)
    what_conj = np.conjugate(what)

    kx = np.fft.fftfreq(nx, lx / nx)
    ky = np.fft.fftfreq(ny, ly / ny)
    kz = np.fft.fftfreq(nz, lz / nz)

    kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing="ij")
    mk = np.sqrt(kx_g**2 + ky_g**2 + kz_g**2)
    E = 0.5 * (uhat * uhat_conj + vhat * vhat_conj + what * what_conj).real

    # Perform binning
    wave_numbers = np.arange(0, nx + 1) * 2 * np.pi
    tke_spectrum = np.zeros(wave_numbers.shape)

    # Filter out under-sampled regions
    # https://scicomp.stackexchange.com/questions/21360/computing-turbulent-energy-spectrum-from-isotropic-turbulence-flow-field-in-a-bo
    for rkx in range(nx):
        for rky in range(ny):
            for rkz in range(nz):
                rk = int(np.round(np.sqrt(rkx * rkx + rky * rky + rkz * rkz)))
                if rk < len(tke_spectrum):
                    tke_spectrum[rk] += E[rkx, rky, rkz]

    E = E.flatten()
    mk = mk.flatten()
    idx = mk.argsort()
    mk = mk[idx]
    E = E[idx]

    return mk, E, wave_numbers, tke_spectrum
