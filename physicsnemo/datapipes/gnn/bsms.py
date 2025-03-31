# ignore_header_test
# ruff: noqa: E402

""""""
"""
BSMS-GNN model. This code was modified from,
https://github.com/Eydcao/BSMS-GNN

The following license is provided from their source,


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse
import torch
from dgl import DGLGraph
from torch.utils.data import Dataset

try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    import warnings

    warnings.warn(
        "sparse_dot_mkl is not installed, install using: pip install sparse_dot_mkl"
    )


_INF = 1 + 1e10


class BistrideMultiLayerGraphDataset(Dataset):
    """Wrapper over graph dataset that enables multi-layer graphs."""

    def __init__(
        self,
        dataset: Dataset,
        num_layers: int = 1,
        cache_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.num_layers = num_layers
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = Path(cache_dir) / self.dataset.split
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, idx):
        graph = self.dataset[idx]
        # Check if MS graph is already in the cache.
        edges_and_ids = None
        if self.cache_dir is not None:
            edges_and_ids = self._load_from_cache(idx)
        if edges_and_ids is None:
            ms_graph = BistrideMultiLayerGraph(graph, self.num_layers)
            _, *edges_and_ids = ms_graph.get_multi_layer_graphs()

            if self.cache_dir is not None:
                self._save_to_cache(idx, edges_and_ids)
        ms_edges, ms_ids = edges_and_ids

        return {
            "graph": graph,
            "ms_edges": [torch.tensor(e, dtype=torch.long) for e in ms_edges],
            "ms_ids": [torch.tensor(ids, dtype=torch.long) for ids in ms_ids],
        }

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _get_cache_filename(self, idx: int) -> str:
        return f"{idx:03}.cache"

    def _load_from_cache(self, idx: int) -> tuple[list, list]:
        if self.cache_dir is None or not self.cache_dir.is_dir():
            raise ValueError("Cache directory is not set or does not exist.")

        filename = self.cache_dir / (self._get_cache_filename(idx) + ".npz")
        if not filename.exists():
            return None
        return np.load(filename, allow_pickle=True)["edges_and_ids"]

    def _save_to_cache(self, idx: int, edges_and_ids: tuple[list, list]) -> None:
        if self.cache_dir is None or not self.cache_dir.is_dir():
            raise ValueError("Cache directory is not set or does not exist.")

        filename = self.cache_dir / self._get_cache_filename(idx)
        return np.savez(
            filename,
            edges_and_ids=np.asanyarray(edges_and_ids, dtype=object),
        )


class BistrideMultiLayerGraph:
    """Multi-layer graph."""

    def __init__(self, graph: DGLGraph, num_layers: int):
        """
        Initializes the BistrideMultiLayerGraph object.

        Parameters
        ----------
        graph: DGLGraph
            The source graph.
        num_layers: int:
            The number of layers to generate.
        """
        self.num_nodes = graph.num_nodes()
        self.num_layers = num_layers
        self.pos_mesh = graph.ndata["pos"].numpy()

        # Initialize the first layer graph
        # Flatten edges to [2, num_edges].
        edges = graph.edges()
        flattened_edges = torch.cat(
            (edges[0].view(1, -1), edges[1].view(1, -1)), dim=0
        ).numpy()
        self.m_gs = [Graph(flattened_edges, GraphType.FLAT_EDGE, self.num_nodes)]
        self.m_flat_es = [self.m_gs[0].get_flat_edge()]
        self.m_ids = []

        self.generate_multi_layer_graphs()

    def generate_multi_layer_graphs(self):
        """
        Generate multiple layers of graphs with pooling.
        """
        g_l = self.m_gs[0]
        pos_l = self.pos_mesh

        index_to_keep = []
        for layer in range(self.num_layers):
            n_l = self.num_nodes if layer == 0 else len(index_to_keep)
            index_to_keep, g_l = self.bstride_selection(g_l, pos_l, n_l)
            pos_l = pos_l[index_to_keep]
            self.m_gs.append(g_l)
            self.m_flat_es.append(g_l.get_flat_edge())
            self.m_ids.append(index_to_keep)

    def get_multi_layer_graphs(self):
        """
        Get the multi-layer graph structures.

        Returns:
        tuple: A tuple containing three lists:
            - m_gs (list): List of graph wrappers for each layer.
            - m_flat_es (list): List of flat edges for each layer.
            - m_ids (list): List of node indices to be pooled at each layer.
        """
        return self.m_gs, self.m_flat_es, self.m_ids

    @staticmethod
    def bstride_selection(g, pos_mesh, n):
        """
        Perform bstride selection to pool nodes and edges.

        Parameters:
        g (Graph): The graph wrapper object.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        n (int): The number of nodes.

        Returns:
        tuple: A tuple containing:
            - combined_idx_kept (list): List of node indices to be pooled.
            - new_g (Graph): The new graph wrapper object after pooling.
        """
        combined_idx_kept = set()
        adj_mat = g.get_sparse_adj_mat()
        adj_mat.setdiag(1)
        clusters = g.clusters

        seeds = BistrideMultiLayerGraph.nearest_center_seed(pos_mesh, clusters)

        for seed, c in zip(seeds, clusters):
            even, odd = set(), set()
            dist_from_central_node = g.bfs_dist(seed)

            for i, dist in enumerate(dist_from_central_node):
                if dist % 2 == 0 and dist != _INF:
                    even.add(i)
                elif dist % 2 == 1 and dist != _INF:
                    odd.add(i)

            if len(even) <= len(odd) or not odd:
                index_kept, index_rmvd = even, odd  # noqa: F841 for clarity
            else:
                index_kept, index_rmvd = odd, even  # noqa: F841 for clarity

            combined_idx_kept = combined_idx_kept.union(index_kept)

        combined_idx_kept = list(combined_idx_kept)
        combined_idx_kept.sort()
        adj_mat = adj_mat.tocsr().astype(float)
        adj_mat = dot_product_mkl(adj_mat, adj_mat)
        adj_mat.setdiag(0)
        new_g = BistrideMultiLayerGraph.pool_edge(adj_mat, n, combined_idx_kept)

        return combined_idx_kept, new_g

    @staticmethod
    def nearest_center_seed(pos_mesh, clusters):
        """
        Find the nearest center seed for each cluster.

        Parameters:
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        clusters (list): List of clusters, each cluster is a list of node indices.

        Returns:
        list: List of seeds per cluster.
        """
        seeds = []
        for c in clusters:
            center = np.mean(pos_mesh[c], axis=0)
            delta_to_center = pos_mesh[c] - center[None, :]
            dist_to_center = np.linalg.norm(delta_to_center, 2, axis=-1)
            min_node = c[np.argmin(dist_to_center)]
            seeds.append(min_node)

        return seeds

    @staticmethod
    def pool_edge(adj_mat, num_nodes, idx):
        """
        Pool the edges based on the provided node indices.

        Parameters:
        adj_mat (scipy.sparse.csr_matrix): The adjacency matrix in CSR format.
        num_nodes (int): The number of nodes in the input graph.
        idx (list): List of node indices to be kept.

        Returns:
        Graph: The new graph wrapper object after pooling.
        """
        flat_e = Graph.adj_mat_to_flat_edge(adj_mat)
        idx = np.array(idx, dtype=np.int64)
        idx_new_valid = np.arange(len(idx)).astype(np.int64)
        idx_new_all = -1 * np.ones(num_nodes).astype(np.int64)
        idx_new_all[idx] = idx_new_valid
        new_flat_e = -1 * np.ones_like(flat_e).astype(np.int64)
        new_flat_e[0] = idx_new_all[flat_e[0]]
        new_flat_e[1] = idx_new_all[flat_e[1]]
        both_valid = np.logical_and(new_flat_e[0] >= 0, new_flat_e[1] >= 0)
        e_idx = np.where(both_valid)[0]
        new_flat_e = new_flat_e[:, e_idx]
        new_g = Graph(new_flat_e, GraphType.FLAT_EDGE, len(idx))

        return new_g


class GraphType(Enum):
    """
    Enumeration to define the types of graph representations.
    """

    FLAT_EDGE = 1
    ADJ_LIST = 2
    ADJ_MAT = 3


class Graph:
    """Convenience graph class."""

    def __init__(self, g, g_type, num_nodes):
        """
        Initialize the Graph object.

        Parameters:
        g (np.ndarray, list, or scipy.sparse.coo_matrix): The graph data in the specified format.
        g_type (GraphType): The type of the input graph representation.
        num_nodes (int): The number of nodes in the graph.
        """
        self.num_nodes = num_nodes

        if g_type == GraphType.FLAT_EDGE:
            self.flat_edges = g
        elif g_type == GraphType.ADJ_LIST:
            self.flat_edges = self.adj_list_to_flat_edge(g)
        elif g_type == GraphType.ADJ_MAT:
            self.flat_edges = self.adj_mat_to_flat_edge(g)
        else:
            raise ValueError(f"Unknown graph type: {g_type}")

        self.adj_list = self.get_adj_list()
        self.clusters = self.find_clusters()

    def get_flat_edge(self):
        """
        Get the flat edge representation of the graph.

        Returns:
        np.ndarray: The flat edge representation of the graph.
        """
        return self.flat_edges

    def get_adj_list(self):
        """
        Get the adjacency list representation of the graph.

        Returns:
        list: The adjacency list representation of the graph.
        """
        return self.flat_edge_to_adj_list(self.flat_edges, self.num_nodes)

    def get_sparse_adj_mat(self):
        """
        Get the sparse adjacency matrix representation of the graph.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix representation of the graph.
        """
        return self.flat_edge_to_adj_mat(self.flat_edges, self.num_nodes)

    def bfs_dist(self, seed):
        """
        Perform a Breadth-First Search (BFS) to find the shortest distance from the seed to all other nodes.

        Parameters:
        seed (int or list): The starting node(s) for BFS.

        Returns:
        np.ndarray: The shortest distance from the seed to all other nodes.
        """
        _INF = 1 + 1e10
        res = np.ones(self.num_nodes) * _INF
        visited = [False for _ in range(self.num_nodes)]
        if isinstance(seed, list):
            for s in seed:
                res[s] = 0
                visited[s] = True
            frontier = seed
        else:
            res[seed] = 0
            visited[seed] = True
            frontier = [seed]

        depth = 0
        track = [frontier]
        while frontier:
            this_level = frontier
            depth += 1
            frontier = []
            while this_level:
                f = this_level.pop(0)
                for n in self.adj_list[f]:
                    if not visited[n]:
                        visited[n] = True
                        frontier.append(n)
                        res[n] = depth
            track.append(frontier)

        return res

    def find_clusters(self):
        """
        Find connected clusters in the graph using BFS.

        Returns:
        list: A list of clusters, each cluster is a list of node indices.
        """
        _INF = 1 + 1e10
        remaining_nodes = list(range(self.num_nodes))
        clusters = []
        while remaining_nodes:
            if len(remaining_nodes) > 1:
                seed = remaining_nodes[0]
                dist = self.bfs_dist(seed)
                tmp = []
                new_remaining = []
                for n in remaining_nodes:
                    if dist[n] != _INF:
                        tmp.append(n)
                    else:
                        new_remaining.append(n)
                clusters.append(tmp)
                remaining_nodes = new_remaining
            else:
                clusters.append([remaining_nodes[0]])
                break

        return clusters

    @staticmethod
    def flat_edge_to_adj_mat(edge_list, n):
        """
        Convert a flat edge list to a sparse adjacency matrix.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix.
        """
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n)
        )
        return adj_mat

    @staticmethod
    def flat_edge_to_adj_list(edge_list, n):
        """
        Convert a flat edge list to an adjacency list.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        list: The adjacency list.
        """
        adj_list = [[] for _ in range(n)]
        for i in range(len(edge_list[0])):
            adj_list[edge_list[0, i]].append(edge_list[1, i])
        return adj_list

    @staticmethod
    def adj_list_to_flat_edge(adj_list):
        """
        Convert an adjacency list to a flat edge list.

        Parameters:
        adj_list (list): The adjacency list.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        edge_list = []
        for i in range(len(adj_list)):
            for n in adj_list[i]:
                edge_list.append(  # noqa: PERF401 list comprehension makes the code less clear.
                    [i, n]
                )
        return np.array(edge_list).transpose()

    @staticmethod
    def adj_mat_to_flat_edge(adj_mat):
        """
        Convert a sparse adjacency matrix to a flat edge list.

        Parameters:
        adj_mat (np.ndarray or scipy.sparse.spmatrix): The sparse adjacency matrix.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        if isinstance(adj_mat, np.ndarray):
            s, r = np.where(adj_mat.astype(bool))
        elif isinstance(adj_mat, scipy.sparse.coo_matrix):
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        elif isinstance(adj_mat, scipy.sparse.csr_matrix):
            adj_mat = scipy.sparse.coo_matrix(adj_mat)
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        else:
            raise ValueError(
                "Unsupported adjacency matrix type in adj_mat_to_flat_edge. Now only support numpy.ndarray, scipy.sparse.coo_matrix, scipy.sparse.csr_matrix."
            )
        return np.array([s, r])
