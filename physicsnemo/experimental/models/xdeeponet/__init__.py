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

"""xDeepONet — the extended DeepONet family.

A single :class:`DeepONet` class assembles operator-learning
architectures spanning the DeepONet and FNO families:

- ``deeponet``, ``u_deeponet``, ``fourier_deeponet``, ``conv_deeponet``,
  ``hybrid_deeponet`` — single-branch + trunk variants.
- ``mionet``, ``fourier_mionet`` — two-branch multi-input + trunk variants.
- ``tno`` — Temporal Neural Operator (branch2 = previous solution) + trunk.
- ``ufno`` / xFNO-style trunkless operators — trunkless spatial branch
  with composable Fourier / UNet / Conv layers; the last spatial axis
  can be interpreted as time for autoregressive bundling via the
  :attr:`DeepONet.time_modes` parameter.

The :class:`DeepONet` class is dimension-generic (``dimension=2|3``
constructor argument; per-dimension primitives are dispatched
internally) and dispatches forward by two flags
(:attr:`auto_pad`, :attr:`trunk`-is-None) over six valid call
conventions: packed-input vs core-input × trunked vs trunkless,
plus the ``temporal_projection`` decoder variant.  See the
:class:`DeepONet` class docstring for the full matrix and worked
examples; see :class:`SpatialBranch` for the spatial-encoder
composition options (Fourier / UNet / Conv layers, multi-layer
pointwise lift, optional coordinate-feature channels).

A thin 4D-FNO convenience wrapper ships alongside :class:`DeepONet`:

- :class:`FNO4DWrapper` — wraps the library
  :class:`physicsnemo.models.fno.FNO` (``dimension=4``) and adds
  autoregressive time-axis extension over :math:`(B, X, Y, Z, T, C)` inputs.
  Use :class:`physicsnemo.models.fno.FNO` with ``dimension=4`` directly when
  the time-axis extension is not needed.

3D FNO / Conv-FNO / U-FNO operators are expressed as :class:`DeepONet` with
``trunk=None`` and a :class:`SpatialBranch` composed of Fourier / UNet / Conv
layers over the :math:`(H, W, T)` axes.
"""

from .branches import SpatialBranch
from .deeponet import DeepONet
from .fno4d import FNO4DWrapper

__all__ = [
    "DeepONet",
    "SpatialBranch",
    "FNO4DWrapper",
]
