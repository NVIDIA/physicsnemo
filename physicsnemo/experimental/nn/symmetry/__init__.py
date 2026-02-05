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

"""Symmetry-equivariant neural network layers.

This module provides layers for building SO(2) and SO(3) equivariant neural networks
for processing spherical harmonic representations.

Classes
-------
SO2Convolution
    SO(2) equivariant convolution layer using grid layout for efficient processing.
GateActivation
    Gated activation applying SiLU to l=0 and learned gating to l>0.
EdgeRotation
    Apply Wigner D-matrix rotations to spherical harmonic embeddings.
    Use ``get_wigner_matrices()`` to compute and cache D-matrices, then
    call the module to apply rotations.
SO3ConvolutionBlock
    SO(3) block-wise transformation using SO3Linear -> GateActivation -> SO3Linear.

Functions
---------
make_grid_mask
    Create a boolean mask for valid (l, m) pairs in the grid layout.
"""

from physicsnemo.experimental.nn.symmetry.activation import GateActivation
from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.experimental.nn.symmetry.so2_conv import SO2Convolution
from physicsnemo.experimental.nn.symmetry.so3_block import SO3ConvolutionBlock
from physicsnemo.experimental.nn.symmetry.wigner import EdgeRotation

__all__ = [
    "EdgeRotation",
    "GateActivation",
    "SO2Convolution",
    "SO3ConvolutionBlock",
    "make_grid_mask",
]
