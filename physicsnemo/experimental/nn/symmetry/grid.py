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
#
# This file contains code derived from `fairchem` found at
# https://github.com/facebookresearch/fairchem.
# Copyright (c) [2025] Meta, Inc. and its affiliates.
# Licensed under MIT License.

"""Grid layout utilities for spherical harmonic coefficients.

This module provides mask generation for the grid layout used in SO(2)
convolutions. The grid layout arranges spherical harmonic coefficients
in a regular 2D tensor indexed by (l, m) with masking for invalid positions.

Functions
---------
make_grid_mask
    Create a boolean mask for valid (l, m) pairs.

Examples
--------
>>> import torch
>>> from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
>>> lmax, mmax = 4, 2
>>> mask = make_grid_mask(lmax, mmax)
>>> mask.shape
torch.Size([5, 3])
>>> # mask[l, m] = True if m <= l
>>> mask[0, 0], mask[0, 1], mask[2, 2]  # Valid: (0,0), Invalid: (0,1), Valid: (2,2)
(tensor(True), tensor(False), tensor(True))
"""

from __future__ import annotations

import torch
from jaxtyping import Bool

__all__ = [
    "make_grid_mask",
]


def make_grid_mask(lmax: int, mmax: int) -> Bool[torch.Tensor, "lmax_p1 mmax_p1"]:
    """Create a boolean mask for valid (l, m) pairs.

    For spherical harmonics, a coefficient Y_l^m exists only when |m| <= l.
    This function creates a mask where mask[l, m] = True if m <= l.

    Parameters
    ----------
    lmax : int
        Maximum degree of spherical harmonics. Must be non-negative.
    mmax : int
        Maximum order of spherical harmonics. Must satisfy 0 <= mmax <= lmax.

    Returns
    -------
    Bool[torch.Tensor, "lmax_p1 mmax_p1"]
        Boolean mask of shape [lmax+1, mmax+1] where mask[l, m] = True
        indicates a valid (l, m) pair (i.e., m <= l).

    Raises
    ------
    ValueError
        If lmax < 0, mmax < 0, or mmax > lmax.

    Examples
    --------
    >>> mask = make_grid_mask(lmax=3, mmax=2)
    >>> mask
    tensor([[ True, False, False],
            [ True,  True, False],
            [ True,  True,  True],
            [ True,  True,  True]])
    >>> # Row l=0 has only m=0 valid
    >>> # Row l=1 has m=0,1 valid
    >>> # Row l=2,3 have all m=0,1,2 valid

    Notes
    -----
    The mask is constructed by comparing a grid of m-values against l-values:
    mask[l, m] = (m <= l). This is equivalent to checking that each
    spherical harmonic coefficient Y_l^m is defined.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if mmax < 0:
        raise ValueError(f"mmax must be non-negative, got {mmax}")
    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")

    # Create coordinate grids
    l_indices = torch.arange(lmax + 1).unsqueeze(1)  # [lmax+1, 1]
    m_indices = torch.arange(mmax + 1).unsqueeze(0)  # [1, mmax+1]

    # mask[l, m] = True if m <= l
    mask = m_indices <= l_indices  # [lmax+1, mmax+1]

    return mask
