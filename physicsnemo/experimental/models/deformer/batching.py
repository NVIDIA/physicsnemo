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
"""PyG batching of graphs that carry static neighbourhoods for :class:`DeFormer`.

:class:`DeFormer` reads ``knn_index``, ``knn_mask`` and ``knn_edge_attr`` from
its graph, all ``[N, K, ...]`` per-node tensors. PyG's default collation treats
any key containing ``index`` as a ``[2, E]`` edge list (concatenated along the
last axis and offset by the node count), so a plain
:class:`torch_geometric.data.Data` batches ``knn_index`` wrongly.
:class:`LocalNeighborhoodData` fixes the concatenation axis and offsets the
neighbour IDs per graph.
"""

from physicsnemo.nn.module.gnn_layers import graph_types as _graph_types

_KNN_KEYS = ("knn_index", "knn_mask", "knn_edge_attr")

if getattr(_graph_types, "PYG_AVAILABLE", False):
    from torch_geometric.data import Data as _Data

    class LocalNeighborhoodData(_Data):
        """PyG ``Data`` whose ``knn_*`` tensors batch as per-node tensors.

        ``knn_index`` is offset by the node count of the preceding graphs so a
        batched neighbourhood still points into its own graph; padded slots
        repeat the node itself and are offset with it.
        """

        def __inc__(self, key, value, *args, **kwargs):
            if key == "knn_index":
                return self.num_nodes
            return super().__inc__(key, value, *args, **kwargs)

        def __cat_dim__(self, key, value, *args, **kwargs):
            if key in _KNN_KEYS:
                return 0
            return super().__cat_dim__(key, value, *args, **kwargs)

else:

    class LocalNeighborhoodData:  # pragma: no cover - exercised without PyG only
        """Placeholder raising at construction when PyTorch Geometric is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LocalNeighborhoodData requires torch_geometric; install PyG to "
                "batch graphs with static neighbourhoods"
            )


__all__ = ["LocalNeighborhoodData"]
