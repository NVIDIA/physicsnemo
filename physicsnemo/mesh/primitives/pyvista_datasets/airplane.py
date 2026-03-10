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

"""Airplane surface mesh from PyVista examples.

Dimensional: 2D manifold in 3D space.
"""

import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.mesh.mesh import Mesh


@require_version_spec("pyvista")
def load(device: torch.device | str = "cpu") -> Mesh:
    """Load airplane surface mesh from PyVista examples.

    This is a classic test case for surface mesh algorithms.
    PyVista caches the downloaded file automatically.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    import importlib

    pv = importlib.import_module("pyvista")

    from physicsnemo.mesh.io.io_pyvista import from_pyvista

    pv_mesh = pv.examples.load_airplane()
    return from_pyvista(pv_mesh).to(device=device)
