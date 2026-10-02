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

"""Static material-neighborhood exclusions; never exclude entire components."""

from collections import defaultdict

import numpy as np
import torch


def material_contact_exclusions(edge_index, num_nodes, hops=2, mesh_cells=None):
    """Return sparse CPU exclusion edges for one/two hops plus shared elements.

    ``mesh_cells`` uses VTK packed polygon format: ``[size, ids..., size, ...]``.
    The graph square is sparse, not an N-by-N dense distance matrix. Polygon
    diagonals are excluded even for cells with more than four vertices. Distant
    material neighborhoods remain eligible when they fold back onto each other.
    """
    from scipy.sparse import coo_matrix

    if hops not in (1, 2) or isinstance(hops, bool):
        raise ValueError("contact_exclusion_hops must be 1 or 2")
    if edge_index.device.type != "cpu" or edge_index.shape[0] != 2:
        raise ValueError("material exclusions require CPU edge_index [2, E]")
    edges = edge_index.numpy()
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("material edge index out of bounds")
    adjacency = coo_matrix(
        (np.ones(edges.shape[1], dtype=bool), (edges[0], edges[1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    exclusion = adjacency.maximum(adjacency @ adjacency) if hops == 2 else adjacency
    if mesh_cells is not None:
        packed = np.asarray(mesh_cells, dtype=np.int64).reshape(-1)
        grouped = defaultdict(list)
        offset = 0
        while offset < packed.size:
            size = int(packed[offset])
            if size < 3 or offset + size >= packed.size:
                raise ValueError("invalid packed polygon connectivity")
            cell = packed[offset + 1 : offset + 1 + size]
            if cell.min() < 0 or cell.max() >= num_nodes:
                raise ValueError("polygon node index out of bounds")
            grouped[size].append(cell)
            offset += size + 1
        for size, cells in grouped.items():
            cells = np.stack(cells)
            a, b = np.triu_indices(size, k=1)
            src, dst = cells[:, a].ravel(), cells[:, b].ravel()
            local = coo_matrix(
                (np.ones(src.size, dtype=bool), (src, dst)),
                shape=adjacency.shape,
            ).tocsr()
            exclusion = exclusion.maximum(local).maximum(local.T)
    exclusion.setdiag(False)
    exclusion.eliminate_zeros()
    row, col = exclusion.nonzero()
    return torch.from_numpy(np.stack((row, col)).astype(np.int64))
