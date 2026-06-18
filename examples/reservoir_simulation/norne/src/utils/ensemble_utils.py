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
=====================================================================
               ENSEMBLE AND GEOSTATISTICAL UTILITIES
=====================================================================

Consolidated reservoir ensemble and geostatistical utility functions
shared across the forward, inverse, compare, and data_extract packages.

Canonical sources
-----------------
- ``compare/sequential/misc_operations.py`` — primary canonical source for
  most functions.
- ``forward/simulator.py`` — canonical for ``Add_marker``, ``Reinvent``.

Functions
---------
Gaussian field helpers:
    fast_gaussian, get_shape, gaussian_with_variable_parameters, add_gnoise

Norne ensemble / geostatistics:
    NorneGeostat, NorneInitialEnsemble, read_until_line

Ensemble generation:
    intial_ensemble, initial_ensemble_gaussian

DCT-based smoothing:
    smoothn, gcv, RobustWeights, InitialGuess, dct_nd

Spectral / random field helpers:
    peaks, Pkgen, distrib, test_points_gen

Variable clipping:
    adjust_variable_within_bounds

Visualisation helpers:
    Reinvent, Add_marker

Progress bar:
    ProgressBar, ShowBar

Rescaling helpers:
    round_array_to_4dp, rescale_linear,
    rescale_linear_numpy_pytorch, rescale_linear_pytorch_numpy

@Author : Clement Etienam
"""

import os
import sys
import warnings

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import scipy.linalg
import scipy.optimize.lbfgsb as lbfgsb
from scipy.fftpack import dct, idct
from scipy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
from hydra.utils import to_absolute_path
from imresize import imresize
from FyeldGenerator import generate_field
from glob import glob
from shutil import rmtree
import mpslib as mps

from utils.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Gaussian field helpers
# ---------------------------------------------------------------------------


def fast_gaussian(dimension, Sdev, Corr):
    """
    Generate a 2-D correlated Gaussian random field using Cholesky decomposition.

    Parameters
    ----------
    dimension : array-like
        Grid size; one element for square grid, two for (m, n).
    Sdev : float or np.ndarray
        Standard deviation scalar or element-wise std array of length m*n.
    Corr : array-like
        Correlation lengths; one element applies to both axes, two for (x, y).

    Returns
    -------
    x : np.ndarray
        Flattened 1-D Gaussian random field of length m*n.

    Raises
    ------
    ValueError
        If dimension has more than 2 elements or Sdev shape is inconsistent.
    """
    dimension = np.array(dimension).flatten()
    m = dimension[0]

    if len(dimension) == 1:
        n = m
    elif len(dimension) == 2:
        n = dimension[1]
    else:
        raise ValueError(
            "FastGaussian: Wrong input, dimension should have length at most 2"
        )
    # If Sdev is provided element-wise, the variance is folded in below; otherwise
    # use the scalar Sdev as the variance (works through the Kronecker product).
    variance = 1 if np.max(np.size(Sdev)) > 1 else Sdev
    if len(Corr) == 1:
        Corr = np.array([Corr[0], Corr[0]])
    elif len(Corr) > 2:
        raise ValueError("FastGaussian: Wrong input, Corr should have length at most 2")

    # Generate the covariance matrix for one layer
    dist = np.arange(0, m) / Corr[0]
    T = scipy.linalg.toeplitz(dist)

    T = variance * np.exp(-(T**2)) + 1e-10 * np.eye(m)

    # Cholesky decomposition for one layer:
    cholT = np.linalg.cholesky(T)
    if Corr[0] == Corr[1] and n == m:
        cholT2 = cholT
    else:
        # Same as for the first dimension:
        dist2 = np.arange(0, n) / Corr[1]
        T2 = scipy.linalg.toeplitz(dist2)
        T2 = variance * np.exp(-(T2**2)) + 1e-10 * np.eye(n)
        cholT2 = np.linalg.cholesky(T2)

    # Draw a random variable:
    x = np.random.randn(m * n)
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))
    x = x.flatten()

    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")

    return x


def get_shape(t):
    """
    Recursively determine the shape of nested tuples.

    Parameters
    ----------
    t : tuple
        Arbitrarily nested tuple structure.

    Returns
    -------
    shape : tuple of int
        Length of each nesting level from outermost to innermost.
    """
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def add_gnoise(Ytrue, SIGMA, SQ=None):
    """
    Add Gaussian noise to an array using a specified covariance matrix or scalar.

    Parameters
    ----------
    Ytrue : np.ndarray or float
        True values to which noise is added.
    SIGMA : float, np.ndarray
        Covariance specification; scalar, vector of std devs, or full covariance matrix.
    SQ : int or None, optional
        If 1, treat SIGMA as the square-root of the covariance (i.e. SIGMA@SIGMA^T). Default is None.

    Returns
    -------
    Y : np.ndarray
        Noisy observations (Ytrue + noise).
    RTSIGMA : float or np.ndarray
        Square-root of the covariance used for sampling.

    Raises
    ------
    Exception
        Re-raises any exception that occurs during noise generation.
    """
    try:
        if SQ is not None and SQ == 1:
            # Use SIGMA*SIGMA' as covariance matrix
            RTSIGMA = SIGMA
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is a scalar or vector
                error = RTSIGMA * np.random.randn(*Ytrue.shape)
            else:
                error = RTSIGMA @ np.random.randn(RTSIGMA.shape[1], 1)
        else:
            # Use SIGMA as covariance matrix
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is entered as a scalar or a vector
                RTSIGMA = np.sqrt(SIGMA)
                error = RTSIGMA * np.random.randn(*Ytrue.shape)
            else:
                # The matrix must be transposed.
                try:
                    RTSIGMA = np.linalg.cholesky(SIGMA).T
                except np.linalg.LinAlgError:
                    logger = setup_logging(__name__)
                    logger.warning("Problem with Cholesky factorization")
                    RTSIGMA = np.sqrtm(SIGMA).real
                    logger.info("Finally - we got a square root!")

                error = RTSIGMA @ np.random.randn(*Ytrue.shape)

        # Add the noise:
        Y = Ytrue + error.flatten()

    except Exception as e:
        logger = setup_logging(__name__)
        logger.error("Error in AddGnoise")
        raise e

    return Y, RTSIGMA


def gaussian_with_variable_parameters(
    field_dim, mean_value, sdev, mean_corr_length, std_corr_length
):
    """
    Generate a Gaussian random field with a stochastically varying correlation length.

    Parameters
    ----------
    field_dim : array-like
        Grid dimensions; 2-element (nx, ny) or 3-element (nx, ny, nz).
    mean_value : np.ndarray
        Background mean array of the same length as the field.
    sdev : float or np.ndarray
        Standard deviation; scalar or per-element array.
    mean_corr_length : float
        Mean value of the correlation length distribution.
    std_corr_length : float
        Standard deviation of the correlation length distribution.

    Returns
    -------
    x : np.ndarray
        Realised field values added to mean_value.
    corr_length : float
        Last sampled correlation length used in the final layer.
    """
    corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    if len(field_dim) < 3:
        x = mean_value + fast_gaussian(field_dim, sdev, corr_length)
    else:
        layer_dim = np.prod(field_dim[:2])
        x = np.copy(mean_value)

        if np.isscalar(sdev):
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev, corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    return x, corr_length


# ---------------------------------------------------------------------------
# Variable clipping
# ---------------------------------------------------------------------------


def adjust_variable_within_bounds(variable, lowerbound=None, upperbound=None):
    """
    Clip ensemble variable values to prescribed lower and/or upper bounds.

    Parameters
    ----------
    variable : np.ndarray
        2-D ensemble array of shape (n_params, n_members) to be clipped in-place.
    lowerbound : float or np.ndarray or None, optional
        Scalar or per-parameter lower bound. Default is None (no lower clipping).
    upperbound : float or np.ndarray or None, optional
        Scalar or per-parameter upper bound. Default is None (no upper clipping).

    Returns
    -------
    variable : np.ndarray
        Clipped ensemble array.
    n : int
        Total number of values that were clipped.

    Raises
    ------
    ValueError
        If both lowerbound and upperbound are None.
    """
    if lowerbound is None and upperbound is None:
        raise ValueError("At least one of lowerbound or upperbound must be provided.")

    n = 0
    ne = variable.shape[1]

    if lowerbound is not None:
        if np.isscalar(lowerbound):
            n += np.sum(variable < lowerbound)
            variable[variable < lowerbound] = lowerbound
        else:
            lowerbound_repeated = np.tile(lowerbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable < lowerbound_repeated)
            variable[variable < lowerbound_repeated] = lowerbound_repeated[
                variable < lowerbound_repeated
            ]

    if upperbound is not None:
        if np.isscalar(upperbound):
            n += np.sum(variable > upperbound)
            variable[variable > upperbound] = upperbound
        else:
            upperbound_repeated = np.tile(upperbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable > upperbound_repeated)
            variable[variable > upperbound_repeated] = upperbound_repeated[
                variable > upperbound_repeated
            ]

    return variable, n


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def read_until_line(file_path, line_num=None, skip=0, sep=r"\s+", header=None):
    """Read file until a specific line; if line_num is None, read entire file."""
    nrows_to_read = None if line_num is None else max(0, line_num - skip)
    df = pd.read_csv(
        file_path, skiprows=skip, nrows=nrows_to_read, sep=sep, header=header
    )
    return df.values


# ---------------------------------------------------------------------------
# Norne geostatistics and ensemble
# ---------------------------------------------------------------------------


def NorneGeostat(nx, ny, nz):
    """
    Load and return geostatistical parameter dictionary for the Norne field model.

    Parameters
    ----------
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.

    Returns
    -------
    norne : dict
        Dictionary of Norne geostatistical parameters including actnum, porosity,
        permeability, NTG, and fault multiplier statistics.
    """
    norne = {}

    dim = np.array([nx, ny, nz])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim

    # actnum
    act = read_until_line(to_absolute_path("../simulator_data/ACTNUM_0704.prop"))
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act

    # porosity
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    file_path = to_absolute_path("../simulator_data/porosity.dat")
    p = read_until_line(file_path)
    p = p[act != 0]

    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = p[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["poroMean"] = p
    norne["poroLayerMean"] = meanv
    norne["poroLayerStd"] = stdv
    norne["poroStd"] = 0.05
    norne["poroLB"] = 0.1
    norne["poroUB"] = 0.4
    norne["poroRange"] = 26
    k = read_until_line(to_absolute_path("../simulator_data/permx.dat"))
    k = np.log(k)
    k = k[act != 0]

    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])

    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = k[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["permxLogMean"] = k
    norne["permxLayerLnMean"] = meanv
    norne["permxLayerStd"] = stdv
    norne["permxStd"] = 1
    norne["permxLB"] = 0.1
    norne["permxUB"] = 10
    norne["permxRange"] = 26

    # Correlation between layers

    corr_with_next_layer = np.zeros(dim[2] - 1)

    for nr in range(dim[2] - 1):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)

        index2_start = ldim * (nr + 1)
        index2_end = ldim * (nr + 2)

        act_layer1 = act[index_start:index_end]
        act_layer2 = act[index2_start:index2_end]

        active = act_layer1 * act_layer2

        values1_range_start = int(np.sum(act[:index_start]))
        values1_range_end = int(np.sum(act[:index_end]))
        values1 = np.concatenate(
            (
                k[values1_range_start:values1_range_end],
                p[values1_range_start:values1_range_end],
            )
        )

        values2_range_start = int(np.sum(act[:index2_start]))
        values2_range_end = int(np.sum(act[:index2_end]))
        values2 = np.concatenate(
            (
                k[values2_range_start:values2_range_end],
                p[values2_range_start:values2_range_end],
            )
        )

        v1 = np.concatenate((act_layer1, act_layer1))
        v1[v1 == 1] = values1.flatten()

        v2 = np.concatenate((act_layer2, act_layer2))
        v2[v2 == 1] = values2.flatten()

        active_full = np.concatenate((active, active))
        co = np.corrcoef(v1[active_full == 1], v2[active_full == 1])

        corr_with_next_layer[nr] = co[0, 1]

    norne["corr_with_next_layer"] = corr_with_next_layer.T

    # Correlation between porosity and permeability
    norne["poroPermxCorr"] = 0.7

    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26

    # rel-perm end-point scaling
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1

    # oil-water contact
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10

    # region multipliers
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0

    # z-multipliers
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds, strict=False), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26

    # fault multipliers
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2

    return norne


def NorneInitialEnsemble(nx, ny, nz, ensembleSize=100, randomNumber=1.2345e5):
    """
    Generate initial ensemble realisations of permeability, porosity, and fault multipliers.

    Parameters
    ----------
    nx : int
        Grid dimension in the X direction.
    ny : int
        Grid dimension in the Y direction.
    nz : int
        Grid dimension in the Z direction.
    ensembleSize : int, optional
        Number of ensemble members to generate. Default is 100.
    randomNumber : float, optional
        Seed for numpy's random number generator. Default is 1.2345e5.

    Returns
    -------
    ensembleperm : np.ndarray
        Permeability ensemble of shape (nx*ny*nz, ensembleSize).
    ensembleporo : np.ndarray
        Porosity ensemble of shape (nx*ny*nz, ensembleSize).
    ensemblefault : np.ndarray
        Fault multiplier ensemble of shape (53, ensembleSize).
    """
    np.random.seed(int(randomNumber))
    N = ensembleSize
    norne = NorneGeostat(nx, ny, nz)
    A = norne["actnum"]
    D = norne["dim"]
    N_F = D[0] * D[1] * D[2]
    M = [norne["poroMean"], norne["permxLogMean"], 0.6]
    S = [norne["poroStd"], norne["permxStd"], norne["ntgStd"]]
    A_L = [A[i : i + D[1] * D[2]] for i in range(0, len(A), D[1] * D[2])]
    A_L = np.array(A_L)
    M_MF = 0.6
    S_MF = norne["multfltStd"]
    C = [norne["poroRange"], norne["permxRange"], norne["ntgRange"]]
    C_S = 2
    R1 = norne["poroPermxCorr"]
    ensembleperm = np.zeros((N_F, N))
    ensemblefault = np.zeros((53, N))
    ensembleporo = np.zeros((N_F, N))

    indices = np.where(A == 1)
    for i in range(N):
        # multz
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]  # Adjusted indexing to 0-based
        A_MZ = A_MZ.flatten()

        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X

        # poro
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)

        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()

        # permx
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2

        indices = np.where(A == 1)
        ensembleperm[indices, i] = (M[1] + S[1] * X[indices]).ravel()
    return ensembleperm, ensembleporo, ensemblefault


# ---------------------------------------------------------------------------
# Ensemble generation
# ---------------------------------------------------------------------------


def intial_ensemble(Nx, Ny, Nz, N, permx):
    """
    Generate an ensemble of permeability realisations using MPS simulation.

    Parameters
    ----------
    Nx : int
        Grid dimension in the X direction.
    Ny : int
        Grid dimension in the Y direction.
    Nz : int
        Grid dimension in the Z direction.
    N : int
        Number of ensemble members to generate.
    permx : np.ndarray
        Training image (TI) for the MPS simulation.

    Returns
    -------
    ensemble : np.ndarray
        Array of shape (Nx*Ny*Nz, N) with columnwise permeability realisations.
    """
    O_mps = mps.mpslib()

    # set the MPS method to 'mps_snesim_tree'
    O_mps = mps.mpslib(method="mps_snesim_tree")

    # set the number of realizations to N
    O_mps.par["n_real"] = N

    # set the permeability field TI
    k = permx
    kjenn = k
    O_mps.ti = kjenn

    # set the simulation grid size
    O_mps.par["simulation_grid_size"] = (Ny, Nx, Nz)

    # run MPS simulation in parallel
    O_mps.run_parallel()

    # get the ensemble of realizations
    ensemble = O_mps.sim

    # reformat the ensemble
    ens = []
    for kk in range(N):
        temp = np.reshape(ensemble[kk], (-1, 1), "F")
        ens.append(temp)
    ensemble = np.hstack(ens)

    # remove temporary files generated during MPS simulation

    for f3 in glob("thread*"):
        rmtree(f3)

    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)

    for f4 in glob("*ti_thread_*"):
        os.remove(f4)

    return ensemble


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx):
    """
    Generate an ensemble of Gaussian random fields scaled to [minn, maxx].

    Parameters
    ----------
    Nx : int
        Grid dimension in the X direction.
    Ny : int
        Grid dimension in the Y direction.
    Nz : int
        Grid dimension in the Z direction.
    N : int
        Number of ensemble members.
    minn : float
        Minimum of the desired output range.
    maxx : float
        Maximum of the desired output range.

    Returns
    -------
    fensemble : np.ndarray
        Array of shape (Nx*Ny*Nz, N) with scaled Gaussian field realisations.
    """
    shape = (Nx, Ny)
    distrib = "gaussian"

    fensemble = np.zeros((Nx * Ny * Nz, N))

    for k in range(N):
        fout = []

        # generate a 3D field
        for _j in range(Nz):
            field = generate_field(distrib, Pkgen(3), shape)
            field = imresize(field, output_shape=shape)
            foo = np.reshape(field, (-1, 1), "F")
            fout.append(foo)

        fout = np.vstack(fout)

        # scale the field to the desired range
        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(fout))
        fout = clfy.transform(fout)

        fensemble[:, k] = np.ravel(fout)

    return fensemble


# ---------------------------------------------------------------------------
# Spectral / random field helpers
# ---------------------------------------------------------------------------


def Pkgen(n):
    """
    Return a power-law spectral density function Pk(k) = k^(-n).

    Parameters
    ----------
    n : float
        Spectral exponent controlling the decay of power with wavenumber.

    Returns
    -------
    Pk : callable
        Function accepting wavenumber k and returning k**(-n).
    """

    def Pk(k):
        """Return ``k ** (-n)`` for scalar or array wavenumber *k*.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber value(s).

        Returns
        -------
        float or np.ndarray
            Power-law spectral weight ``k**(-n)``.
        """
        return np.power(k, -n)

    return Pk


def distrib(shape):
    """
    Draw a complex-valued sample from two independent standard normal distributions.

    Parameters
    ----------
    shape : tuple of int
        Shape of each of the two real and imaginary component arrays.

    Returns
    -------
    sample : np.ndarray
        Complex array of given shape: real + 1j*imaginary, both N(0,1).
    """
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib="random", **kwargs):
    """
    Generate test-point arrays using random uniform or Latin Hypercube sampling.

    Parameters
    ----------
    n_test : int
        Number of test points to generate.
    nder : int
        Number of input dimensions (variables).
    interval : tuple of float, optional
        (lower, upper) bounds for the output range. Default is (-1.0, 1.0).
    distrib : str, optional
        Sampling strategy; 'random' for uniform, 'lhs' for Latin Hypercube. Default is 'random'.
    **kwargs
        Additional keyword arguments forwarded to pyDOE.lhs when distrib='lhs'.

    Returns
    -------
    points : np.ndarray
        Array of shape (n_test, nder) scaled to interval.
    """
    return {
        "random": lambda n_test, nder: (interval[1] - interval[0])
        * np.random.rand(n_test, nder)
        + interval[0],
        "lhs": lambda n_test, nder: (interval[1] - interval[0])
        * lhs(nder, samples=n_test, **kwargs)
        + interval[0],
    }[distrib.lower()](n_test, nder)


# ---------------------------------------------------------------------------
# DCT-based smoothing
# ---------------------------------------------------------------------------


def dct_nd(data, f=dct):
    """
    Apply an N-dimensional discrete cosine (or inverse cosine) transform.

    Parameters
    ----------
    data : np.ndarray
        Input array; 1-D through 4-D are supported.
    f : callable, optional
        Transform function; scipy.fftpack.dct or idct. Default is dct.

    Returns
    -------
    transformed : np.ndarray
        Transformed array with the same shape as data.
    """
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    if nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    if nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )
    if nd == 4:
        return f(
            f(
                f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
                norm="ortho",
                type=2,
                axis=2,
            ),
            norm="ortho",
            type=2,
            axis=3,
        )
    raise ValueError(f"dct_nd only supports 1-D through 4-D arrays, got {nd}-D")


def peaks(n):
    """
    Mimic basic of matlab peaks fn
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for _i in range(n // 5):
        x0 = np.random.random() * n
        y0 = np.random.random() * n
        sdx = np.random.random() * n / 4.0
        sdy = sdx
        c = np.random.random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= np.random.random()
        z += f
    return z


def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    """
    Compute the Generalised Cross-Validation (GCV) score for a smoothness parameter.

    Parameters
    ----------
    p : float
        Log10 of the smoothing parameter s.
    Lambda : np.ndarray
        Eigenvalue tensor used in the penalised least-squares smoother.
    aow : float
        Amount-of-weights ratio (sum(Wtot) / n_elements).
    DCTy : np.ndarray
        Discrete cosine transform of the weighted residual field.
    IsFinite : np.ndarray
        Boolean mask of finite data locations.
    Wtot : np.ndarray
        Total weights array (same shape as y).
    y : np.ndarray
        Observed data array.
    nof : int
        Number of finite data elements.
    noe : int
        Total number of elements.
    smoothOrder : float
        Order of the smoothness penalty.

    Returns
    -------
    GCVscore : float
        GCV score for the given smoothing parameter.
    """
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10**p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    # --- RSS = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dct_nd(Gamma * DCTy, f=idct)
        RSS = norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    # ---
    TrH = sum(Gamma)
    return RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2


def RobustWeights(r, mask_valid, h, wstr):
    """
    Compute robust weights for iteratively re-weighted smoothing.

    Parameters
    ----------
    r : np.ndarray
        Residual array from the current smooth estimate.
    mask_valid : np.ndarray
        Boolean mask indicating finite/valid data locations.
    h : float
        Leverage value used to studentise the residuals.
    wstr : str
        Weighting scheme; one of 'bisquare', 'cauchy', or 'talworth'.

    Returns
    -------
    W : np.ndarray
        Array of robust weights, same shape as r, with NaN replaced by 0.
    """
    # weights for robust smoothing.
    MAD = np.median(
        abs(r[mask_valid] - np.median(r[mask_valid]))
    )  # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))  # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c  # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights

    if np.any(np.isnan(W)):
        warnings.warn("NaN values detected in W; replacing with 0.", stacklevel=2)
    W[np.isnan(W)] = 0
    return W


def InitialGuess(y, mask_valid):
    """
    Provide an initial smooth estimate by nearest-neighbour fill and coarse DCT smoothing.

    Parameters
    ----------
    y : np.ndarray
        Observed data array potentially containing missing values.
    mask_valid : np.ndarray
        Boolean mask; True where data are valid.

    Returns
    -------
    z : np.ndarray
        Initial guess array with missing values filled and coarsely smoothed.
    """
    # -- nearest neighbor interpolation (in case of missing values)
    if any(~mask_valid):
        try:
            from scipy.ndimage.morphology import distance_transform_edt

            # if license('test','image_toolbox')
            # [z,L] = bwdist(I);
            L = distance_transform_edt(1 - mask_valid)
            z = y
            z[~mask_valid] = y[L[~mask_valid]]
        except Exception:
            z = y
            z[~mask_valid] = np.mean(y[mask_valid])
    else:
        z = y
    # coarse fast smoothing
    z = dct_nd(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = [np.arange(m[i], k[i]) for i in range(len(k))]
    d = np.array(d).astype(int)
    z[d] = 0.0
    return dct_nd(z, f=idct)


def _smoothn_warning(s1, s2):
    """Log two warning/info strings using the module logger."""
    logger = setup_logging(__name__)
    logger.info("s1: %s", s1)
    logger.info("s2[0]: %s", s2[0])


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):
    """
    Smooth an N-dimensional array using a penalised least-squares DCT-based method.

    Parameters
    ----------
    y : np.ndarray or np.ma.MaskedArray
        Data array to smooth; missing values should be NaN or masked.
    nS0 : int, optional
        Number of smoothing-parameter candidates in the initial GCV grid. Default is 10.
    axis : tuple of int or None, optional
        Axes along which to smooth; None smooths all axes. Default is None.
    smoothOrder : float, optional
        Order of the smoothness penalty. Default is 2.0.
    sd : np.ndarray or None, optional
        Standard deviation array for weighted smoothing; produces weights = 1/sd**2. Default is None.
    verbose : bool, optional
        If True, log iteration progress. Default is False.
    s0 : float or None, optional
        Initial guess for log10(s); overrides GCV grid initialisation. Default is None.
    z0 : np.ndarray or None, optional
        Initial guess for the smooth estimate. Default is None.
    isrobust : bool, optional
        If True, apply iteratively re-weighted robust smoothing. Default is False.
    W : np.ndarray or None, optional
        Explicit weight array; overrides sd. Default is None.
    s : float or None, optional
        Fixed smoothing parameter; bypasses automatic GCV search. Default is None.
    MaxIter : int, optional
        Maximum number of smoothing iterations. Default is 100.
    TolZ : float, optional
        Convergence tolerance on the relative change in z. Default is 1e-3.
    weightstr : str, optional
        Robust weighting scheme: 'bisquare', 'cauchy', or 'talworth'. Default is 'bisquare'.

    Returns
    -------
    z : np.ndarray
        Smoothed array with the same shape as y.
    s : float
        Optimal smoothing parameter found (or supplied).
    exitflag : int
        1 if converged within MaxIter, 0 otherwise.
    Wtot : np.ndarray
        Final total weights used in the last smoothing iteration.
    """
    if isinstance(y, ma.core.MaskedArray):  # masked array
        # is_masked = True
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if W is not None and np.any(W):
            W = np.array(W)
            W[mask] = 0.0
        if sd is not None and np.any(sd):
            W = np.array(1.0 / sd**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if sd is not None and np.any(sd):
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if W is not None and np.any(W):
        W = W / W.max()

    sizy = y.shape
    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    # ---
    # Smoothness parameter and weights
    if W is None or not np.any(W):
        W = np.ones(sizy)

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.array(np.isfinite(y)).astype(bool)
    nof = IsFinite.sum()  # number of finite elements
    W = W * IsFinite
    if any(W < 0):
        raise RuntimeError("smoothn:NegativeWeights", "Weights must all be >=0")
    # ---
    # Weighted or missing data?
    isweighted = any(W != 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s
    # ---
    # DCTN and IDCTN are required
    try:
        from scipy.fftpack.realtransforms import dct, idct
    except Exception:
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    Lambda = np.zeros(sizy)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim))[0].astype(int)
        siz0[i] = sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

    N = sum(np.array(sizy) != 1)  # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    try:
        sMinBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
    except Exception:
        sMinBnd = None
        sMaxBnd = None
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,IsFinite);
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy)
    # ---
    z0 = z
    y[~IsFinite] = 0  # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    ## Main iterative process
    # ---
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except Exception:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = sum(Wtot) / noe  # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            if verbose:
                logger = setup_logging(__name__)
                logger.info("tol %s nit %s", tol, nit)
            nit = nit + 1
            DCTy = dct_nd(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                    xpost = [ss[g == g.min()]]
                else:
                    xpost = [s0]
                xpost, _f, _d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=1e7,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
            z = RF * dct_nd(Gamma * DCTy, f=idct) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * norm(z0 - z) / norm(z)
            z0 = z  # re-initialization
        exitflag = nit < MaxIter
        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h**N
            # --- take robust weights into account
            Wtot = W * RobustWeights(y - z, IsFinite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3  # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False  # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            _smoothn_warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    f"s = {s:.3f} "
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            _smoothn_warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    f"s = {s:.3f} "
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
    return z, s, exitflag, Wtot


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def Reinvent(matt):
    """
    Rearrange a (nz, nx, ny) array into an (nx, ny, nz) layout.

    Parameters
    ----------
    matt : np.ndarray
        3-D array of shape (nz, nx, ny).

    Returns
    -------
    dess : np.ndarray
        3-D array of shape (nx, ny, nz) with layers transposed.
    """
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


def Add_marker(plt, XX, YY, locc):
    """
    Overlay well-location scatter markers on a 2-D Matplotlib plot.

    Parameters
    ----------
    plt : module
        The matplotlib.pyplot module (or compatible axes object with scatter).
    XX : np.ndarray
        2-D mesh of X coordinates.
    YY : np.ndarray
        2-D mesh of Y coordinates.
    locc : np.ndarray
        Array of shape (n_wells, 3) with columns [x_index, y_index, well_type].
        well_type == 2 renders an upward triangle, otherwise a downward triangle.

    Returns
    -------
    None
    """
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])

        # if the location type is 2, add an upward pointing marker
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        # otherwise, add a downward pointing marker
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    """Return a textual progress bar string for console display."""
    try:
        # You can't have a progress bar with zero or negative length.
        if BarLength < 1:
            BarLength = 20
        # Use status variable for going to the next line after progress completion.
        Status = ""
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.0:
            Progress = 1
            Status = "\r\n"  # Going to the next line
        # Calculating how many places should be filled
        Block = round(BarLength * Progress)
        # Show this
        return f"[{ProgressIcon * Block + BarIcon * (BarLength - Block)}] {round(Progress * 100, 0):.0f}% {Status}"
    except Exception:
        return "ERROR"


def ShowBar(Bar):
    """Write a progress bar string to stdout without newline."""
    sys.stdout.write(Bar)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Rescaling helpers
# ---------------------------------------------------------------------------


def round_array_to_4dp(arr):
    """
    Round every element of an array to 4 decimal places.

    Parameters
    ----------
    arr : array-like
        Input data convertible to a numpy array.

    Returns
    -------
    rounded : np.ndarray or None
        Array rounded to 4 decimal places, or None if an error occurs.
    """
    try:
        arr = np.asarray(arr)  # Convert input to a numpy array if it's not already
        return np.around(arr, 4)
    except Exception as e:
        logger = setup_logging(__name__)
        logger.error(f"An error occurred: {e!s}")
        return None  # You can choose to return None or handle the error differently


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b
