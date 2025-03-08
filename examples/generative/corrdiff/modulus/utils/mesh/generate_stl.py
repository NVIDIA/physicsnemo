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

# ruff: noqa: F401

import numpy as np
import warp as wp
from numpy.typing import NDArray
from stl import mesh


def sdf_to_stl(
    field: NDArray[float],
    threshold: float = 0.0,
    backend: str = "warp",
    filename: str = "output_stl.stl",
):
    """
    Helper utility to create STL from input SDF using Marching Cube algorithm.
    Wrapper around Warp's algorithm: https://nvidia.github.io/warp/modules/runtime.html#marching-cubes
    and scikit-image's algorithm: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes

    Parameters
    ----------
    field : NDArray[float]
        SDF field array. Must be a 3D tensor of shape [nx, ny, nz].
    threshold : float, optional
        Target iso-surface value, by default 0.0
    backend : str, optional
        Backed to use. Options available warp and skimage, by default warp
    filename : str, optional
        Filename for output stl file, by default "output_stl.stl"
    """
    if backend == "warp":
        # Convert numpy array to warp array
        field = wp.array(field)

        mc = wp.MarchingCubes(
            field.shape[0],
            field.shape[1],
            field.shape[2],
            max_verts=int(1e6),
            max_tris=int(1e6),
        )

        # extract the surface
        mc.surface(field=field, threshold=threshold)

        # extract the vertices and faces
        verts = mc.verts.numpy()
        faces = mc.indices.numpy().reshape(-1, 3)

    elif backend == "skimage":
        try:
            import skimage  # noqa: F401 for docs
            from skimage import measure
        except ImportError:
            raise ImportError("Install `scikit-image` to use `skimage` backend.")

        verts, faces, _, _ = measure.marching_cubes(
            field, threshold, spacing=[field.shape[0], field.shape[1], field.shape[2]]
        )

    # save stl file
    mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)

    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data["vectors"][i][j] = verts[f[j], :]

    surface_mesh = mesh.Mesh(mesh_data)
    surface_mesh.save(filename)
