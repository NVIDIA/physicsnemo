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


import json
import os

import numpy as np
import torch

from tfrecord.torch.dataset import TFRecordDataset


try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )
from torch.nn import functional as F

from physicsnemo.datapipes.gnn.utils import load_json, save_json


class DeformingPlateDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        - A single adj matrix is used for each transient simulation.
            Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    num_samples : int, optional
        Number of samples, by default 1000
    num_steps : int, optional
        Number of time steps in each sample, by default 400
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.003
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self,
        name="dataset",
        data_dir=None,
        split="train",
        num_samples=1000,
        num_steps=400,
        noise_std=0.003,
        force_reload=False,
        verbose=False,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = data_dir
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.length = num_samples * (num_steps - 1)

        print(f"Preparing the {split} dataset...")
        # create the graphs with edge features
        # Build TFRecordDataset from .tfrecord file
        tfrecord = os.path.join(data_dir, f"{split}.tfrecord")
        index = None  # or path to .index if you generated it
        # Define the schema per meta.json
        meta = json.load(open(os.path.join(data_dir, "meta.json")))
        description = {k: "byte" for k in meta["field_names"]}  # raw bytes
        self.torch_ds = TFRecordDataset(
            tfrecord,
            index,
            description,
            transform=lambda rec: self._decode_record(rec, meta),
        )
        self.graphs, self.cells, self.node_type = [], [], []
        (
            noise_mask,
            self.moving_points_mask,
            self.object_points_mask,
            self.clamped_points_mask,
        ) = [], [], [], []
        self.mesh_pos = []
        for i, rec in enumerate(self.torch_ds):
            if i >= num_samples:
                break
            data_np = {k: v[:num_steps] for k, v in rec.items()}
            src, dst = self.cell_to_adj(data_np["cells"][0])  # assuming stationary mesh
            graph = self.create_graph(src, dst, dtype=torch.int32)
            graph = self.add_edge_features(graph, data_np["mesh_pos"][0])
            self.graphs.append(graph)
            node_type = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
            self.node_type.append(self._one_hot_encode(node_type))
            noise_mask.append(torch.eq(node_type, torch.zeros_like(node_type)))

            if self.split != "train":
                self.mesh_pos.append(torch.tensor(data_np["mesh_pos"][0]))
                self.cells.append(data_np["cells"][0])
                moving_points_mask, object_points_mask, clamped_points_mask = (
                    self._get_rollout_mask(node_type)
                )
                self.moving_points_mask.append(moving_points_mask)
                self.object_points_mask.append(object_points_mask)
                self.clamped_points_mask.append(clamped_points_mask)

        # compute or load edge data stats
        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("edge_stats.json")

        # normalize edge features
        for i in range(num_samples):
            self.graphs[i].edata["x"] = self.normalize_edge(
                self.graphs[i],
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

        # create the node features
        self.node_features, self.node_targets = [], []
        for i, rec in enumerate(self.torch_ds):
            if i >= num_samples:
                break
            data_np = {k: v[:num_steps] for k, v in rec.items()}
            features, targets = {}, {}
            features["world_pos"] = self._drop_last(
                data_np["world_pos"]
            )  # Shape: (num_steps-1, num_nodes, num_features)
            targets["velocity"] = self._push_forward_diff(
                data_np["world_pos"]
            )  # Shape: (num_steps-1, num_nodes, num_features)
            targets["stress"] = self._push_forward(
                data_np["stress"]
            )  # Shape: (num_steps-1, num_nodes, num_features)

            # add noise
            if (
                split == "train"
            ):  # TODO: noise has to be added at each iteration during training
                features["world_pos"], targets["velocity"] = self._add_noise(
                    features["world_pos"],
                    targets["velocity"],
                    self.noise_std,
                    noise_mask[i],
                )
            self.node_features.append(features)
            self.node_targets.append(targets)

        # compute or load node data stats
        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("node_stats.json")

        # normalize node features
        for i in range(num_samples):
            self.node_targets[i]["velocity"] = self.normalize_node(
                self.node_targets[i]["velocity"],
                self.node_stats["velocity_mean"],
                self.node_stats["velocity_std"],
            )
            self.node_targets[i]["stress"] = self.normalize_node(
                self.node_targets[i]["stress"],
                self.node_stats["stress_mean"],
                self.node_stats["stress_std"],
            )

    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)  # graph index
        tidx = idx % (self.num_steps - 1)  # time step index
        graph = self.graphs[gidx].clone()
        node_features = self.node_type[gidx].float()
        node_targets = torch.cat(
            (
                self.node_targets[gidx]["velocity"][tidx],
                self.node_targets[gidx]["stress"][tidx],
            ),
            dim=-1,
        )
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.ndata["world_pos"] = self.node_features[gidx]["world_pos"][tidx]
        if self.split == "train":
            return graph
        else:
            graph.ndata["mesh_pos"] = self.mesh_pos[gidx]
            cells = self.cells[gidx]
            moving_points_mask = self.moving_points_mask[gidx]
            object_points_mask = self.object_points_mask[gidx]
            clamped_points_mask = self.clamped_points_mask[gidx]

            return (
                graph,
                cells,
                moving_points_mask,
                object_points_mask,
                clamped_points_mask,
            )

    def __len__(self):
        return self.length

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.num_samples
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0)
                / self.num_samples
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "velocity_mean": 0,
            "velocity_meansqr": 0,
            "stress_mean": 0,
            "stress_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += (
                torch.mean(self.node_targets[i]["velocity"], dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["velocity"]), dim=(0, 1))
                / self.num_samples
            )
            stats["stress_mean"] += (
                torch.mean(self.node_targets[i]["stress"], dim=(0, 1))
                / self.num_samples
            )
            stats["stress_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["stress"]), dim=(0, 1))
                / self.num_samples
            )
        stats["velocity_std"] = torch.sqrt(
            stats["velocity_meansqr"] - torch.square(stats["velocity_mean"])
        )
        stats["stress_std"] = torch.sqrt(
            stats["stress_meansqr"] - torch.square(stats["stress_mean"])
        )
        stats.pop("velocity_meansqr")
        stats.pop("stress_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    @staticmethod
    def cell_to_adj(cells):
        """creates adjacency matrix in COO format from mesh cells (tetrahedra)"""
        num_cells = np.shape(cells)[0]
        # For each tetrahedron, generate all 6 edges
        edge_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        src = [cells[i][a] for i in range(num_cells) for a, b in edge_indices]
        dst = [cells[i][b] for i in range(num_cells) for a, b in edge_indices]
        return src, dst

    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        """
        creates a DGL graph from an adj matrix in COO format.
        torch.int32 can handle graphs with up to 2**31-1 nodes or edges.
        """
        graph = dgl.to_bidirected(dgl.graph((src, dst), idtype=dtype))
        graph = dgl.to_simple(graph)
        return graph

    @staticmethod
    def add_edge_features(graph, pos):
        """
        adds relative displacement & displacement norm as edge features
        """
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def normalize_node(invar, mu, std):
        """normalizes a tensor"""
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def normalize_edge(graph, mu, std):
        """normalizes a tensor"""
        if (
            graph.edata["x"].size()[-1] != mu.size()[-1]
            or graph.edata["x"].size()[-1] != std.size()[-1]
        ):
            raise AssertionError("Graph edge data must be same size as stats.")
        return (graph.edata["x"] - mu) / std

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

    @staticmethod
    def _one_hot_encode(node_type):
        # node_type: tensor of shape (...), values in {0, 1, 3}
        node_type = torch.squeeze(node_type, dim=-1)
        # Map 0 -> 0, 1 -> 1, 3 -> 2
        mapping = {0: 0, 1: 1, 3: 2}
        mapped = torch.full_like(node_type, fill_value=-1)
        for k, v in mapping.items():
            mapped[node_type == k] = v
        if (mapped == -1).any():
            raise ValueError("node_type contains values outside of {0, 1, 3}")
        node_type = F.one_hot(mapped.long(), num_classes=3)
        return node_type

    @staticmethod
    def _drop_last(invar):
        return torch.tensor(invar[0:-1], dtype=torch.float)

    @staticmethod
    def _push_forward(invar):
        return torch.tensor(invar[1:], dtype=torch.float)

    @staticmethod
    def _push_forward_diff(invar):
        return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _get_rollout_mask(node_type):
        moving_points_mask = torch.eq(node_type, torch.zeros_like(node_type))
        object_points_mask = torch.eq(node_type, torch.zeros_like(node_type) + 1)
        clamped_points_mask = torch.eq(node_type, torch.zeros_like(node_type) + 3)
        return moving_points_mask, object_points_mask, clamped_points_mask

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 3)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets

    def _decode_record(self, rec_bytes, meta):
        out = {}
        for k, v in rec_bytes.items():
            dtype = meta["features"][k]["dtype"]
            shape = meta["features"][k]["shape"]
            arr = np.frombuffer(v, dtype=getattr(np, dtype))
            arr = arr.reshape(shape)
            if meta["features"][k]["type"] == "static":
                arr = np.tile(arr, (meta["trajectory_length"], 1, 1))
            out[k] = arr
        return out
