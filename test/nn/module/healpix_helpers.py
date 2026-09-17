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

"""Shared helpers for HEALPix layer tests (``test/nn/module/test_healpix.py``
and ``test/models/dlwp_healpix/test_healpix_layers.py``)."""

import torch

Tensor = torch.Tensor


class MulX(torch.nn.Module):
    """Helper layer that just multiplies the values of an input tensor.

    Used as a minimal, non-convolutional stand-in for the ``layer`` argument
    of ``HEALPixLayer`` so wrapper behavior can be tested independently of
    any real convolution.
    """

    def __init__(self, multiplier: int = 1):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: Tensor) -> Tensor:
        return x * self.multiplier


def distinct_face_tensor(size: int, channels: int = 1, device: str = "cpu") -> Tensor:
    """Build a folded ``(12, C, H, W)`` tensor where every pixel encodes its
    own ``(face, row, col)``, so HEALPix face-stitching logic can be checked
    against independently-computed expected neighbor slices.

    Parameters
    ----------
    size : int
        Height/width of each (square) face.
    channels : int, optional
        Number of channels to replicate the pattern across, by default 1.
    device : str, optional
        Device to allocate the tensor on, by default "cpu".

    Returns
    -------
    Tensor
        Tensor of shape ``(12, channels, size, size)`` with values
        ``face * 1000 + row * 10 + col``.
    """
    faces = 12
    face_idx = torch.arange(faces, device=device).view(faces, 1, 1, 1)
    row_idx = torch.arange(size, device=device).view(1, 1, size, 1)
    col_idx = torch.arange(size, device=device).view(1, 1, 1, size)
    pattern = face_idx * 1000 + row_idx * 10 + col_idx
    return pattern.expand(faces, channels, size, size).float()
