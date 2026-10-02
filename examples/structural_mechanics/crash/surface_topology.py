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

"""Dataset surface connectivity and opt-in, topology-only contact exclusions."""

import numpy as np
import torch
from torch_geometric.data import Data


class SurfaceContactData(Data):
    """PyG batching with node offsets for facet connectivity (not face IDs)."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "contact_faces":
            return self.num_nodes
        if key == "contact_surface_exclusions":
            return value.new_tensor([[self.num_nodes], [len(self.contact_faces)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "contact_surface_exclusions":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def surface_one_ring_exclusions(faces: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return sorted unique CPU [node, facet] exclusions for one material ring.

    A query node is excluded from a facet if it shares an original element with
    any vertex of that facet (including incidence). A quad's opposite vertices
    are material neighbors too; padded triangles do not add an extra neighbor.
    Connectivity, not spatial distance, defines this immutable neighborhood.
    Disconnected surfaces and distant regions of the same connected component
    remain eligible for contact, including after folding.

    Boolean sparse incidence B gives (B @ B.T) @ B. No dense N-by-N or N-by-F
    arrays are constructed. This optional one-ring ablation can suppress real
    contact within that ring; it does not reproduce unavailable solver sets.
    """
    from scipy.sparse import coo_matrix

    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if (
        faces.device.type != "cpu"
        or faces.dtype != torch.long
        or faces.ndim != 2
        or faces.shape[1] not in (3, 4)
    ):
        raise ValueError("one-ring exclusions require CPU int64 faces [F,3|4]")
    if faces.numel() and (faces.min() < 0 or faces.max() >= num_nodes):
        raise ValueError("surface node index out of bounds")
    if not len(faces):
        return torch.empty((2, 0), dtype=torch.long)
    values = faces.numpy()
    incidence = coo_matrix(
        (
            np.ones(values.size, dtype=bool),
            (values.ravel(), np.repeat(np.arange(len(faces)), faces.shape[1])),
        ),
        shape=(num_nodes, len(faces)),
    ).tocsr()
    exclusions = ((incidence @ incidence.T) @ incidence).tocsr()
    exclusions.sum_duplicates()
    exclusions.sort_indices()
    nodes, facets = exclusions.nonzero()
    return torch.from_numpy(np.stack((nodes, facets)).astype(np.int64))


def surface_faces_from_cells(mesh_cells, num_nodes):
    """Read packed VTK polygon connectivity into [F,4], with padded triangles.

    Repeated third/fourth ID encodes a triangle. Other degeneracies, unsupported
    polygons and duplicate facets raise rather than silently alter contact.
    """
    if mesh_cells is None:
        raise ValueError("surface contact requires element connectivity")
    raw = np.asarray(mesh_cells)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("packed surface connectivity must be a 1D integer array")
    faces = []
    seen = set()
    offset = 0
    while offset < len(raw):
        size = int(raw[offset])
        if size not in (3, 4) or offset + size >= len(raw):
            raise ValueError(
                "surface contact supports complete triangle/quad cells only"
            )
        cell = raw[offset + 1 : offset + 1 + size]
        # GM exports triangular shells as [a,b,c,c] in four-slot cells,
        # exactly the triangle encoding used by TYPE7. Preserve its facet ID.
        real_cell = cell[:3] if size == 4 and cell[2] == cell[3] else cell
        if (
            cell.min() < 0
            or cell.max() >= num_nodes
            or len(set(real_cell.tolist())) != len(real_cell)
        ):
            raise ValueError("invalid or repeated surface node index")
        canonical = tuple(sorted(real_cell.tolist()))
        if canonical in seen:
            raise ValueError("duplicate surface facets")
        seen.add(canonical)
        faces.append(np.pad(cell, (0, 4 - size), mode="edge"))
        offset += size + 1
    result = torch.from_numpy(np.asarray(faces, dtype=np.int64).reshape(-1, 4))
    return result
