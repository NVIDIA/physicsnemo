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


import functools
import json
import os

import numpy as np
import torch

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )
from torch.nn import functional as F

from .utils import load_json, save_json

# Hide GPU from visible devices for TF
tf.config.set_visible_devices([], "GPU")


class VortexSheddingDataset(DGLDataset):
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
        Number of time steps in each sample, by default 600
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.02
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
        num_steps=600,
        noise_std=0.02,
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
        dataset_iterator = self._load_tf_data(self.data_dir, self.split)
        self.graphs, self.cells, self.node_type = [], [], []
        noise_mask, self.rollout_mask = [], []
        self.mesh_pos = []
        for i in range(self.num_samples):
            data_np = dataset_iterator.get_next()
            data_np = {key: arr[:num_steps].numpy() for key, arr in data_np.items()}
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
                self.rollout_mask.append(self._get_rollout_mask(node_type))

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
        dataset_iterator = self._load_tf_data(self.data_dir, self.split)
        self.node_features, self.node_targets = [], []
        for i in range(self.num_samples):
            data_np = dataset_iterator.get_next()
            data_np = {key: arr[:num_steps].numpy() for key, arr in data_np.items()}
            features, targets = {}, {}
            features["velocity"] = self._drop_last(data_np["velocity"])
            targets["velocity"] = self._push_forward_diff(data_np["velocity"])
            targets["pressure"] = self._push_forward(data_np["pressure"])

            # add noise
            if split == "train":
                features["velocity"], targets["velocity"] = self._add_noise(
                    features["velocity"],
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
            self.node_features[i]["velocity"] = self.normalize_node(
                self.node_features[i]["velocity"],
                self.node_stats["velocity_mean"],
                self.node_stats["velocity_std"],
            )
            self.node_targets[i]["velocity"] = self.normalize_node(
                self.node_targets[i]["velocity"],
                self.node_stats["velocity_diff_mean"],
                self.node_stats["velocity_diff_std"],
            )
            self.node_targets[i]["pressure"] = self.normalize_node(
                self.node_targets[i]["pressure"],
                self.node_stats["pressure_mean"],
                self.node_stats["pressure_std"],
            )

    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)  # graph index
        tidx = idx % (self.num_steps - 1)  # time step index
        graph = self.graphs[gidx]
        node_features = torch.cat(
            (self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1
        )
        node_targets = torch.cat(
            (
                self.node_targets[gidx]["velocity"][tidx],
                self.node_targets[gidx]["pressure"][tidx],
            ),
            dim=-1,
        )
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        if self.split == "train":
            return graph
        else:
            graph.ndata["mesh_pos"] = self.mesh_pos[gidx]
            cells = self.cells[gidx]
            rollout_mask = self.rollout_mask[gidx]
            return graph, cells, rollout_mask

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
            "velocity_diff_mean": 0,
            "velocity_diff_meansqr": 0,
            "pressure_mean": 0,
            "pressure_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += (
                torch.mean(self.node_features[i]["velocity"], dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_meansqr"] += (
                torch.mean(torch.square(self.node_features[i]["velocity"]), dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_mean"] += (
                torch.mean(self.node_targets[i]["pressure"], dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["pressure"]), dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_diff_mean"] += (
                torch.mean(
                    self.node_targets[i]["velocity"],
                    dim=(0, 1),
                )
                / self.num_samples
            )
            stats["velocity_diff_meansqr"] += (
                torch.mean(
                    torch.square(self.node_targets[i]["velocity"]),
                    dim=(0, 1),
                )
                / self.num_samples
            )
        stats["velocity_std"] = torch.sqrt(
            stats["velocity_meansqr"] - torch.square(stats["velocity_mean"])
        )
        stats["pressure_std"] = torch.sqrt(
            stats["pressure_meansqr"] - torch.square(stats["pressure_mean"])
        )
        stats["velocity_diff_std"] = torch.sqrt(
            stats["velocity_diff_meansqr"] - torch.square(stats["velocity_diff_mean"])
        )
        stats.pop("velocity_meansqr")
        stats.pop("pressure_meansqr")
        stats.pop("velocity_diff_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    def _load_tf_data(self, path, split):
        """
        Utility for loading the .tfrecord dataset in DeepMind's MeshGraphNet repo:
        https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        Follow the instructions provided in that repo to download the .tfrecord files.
        """
        dataset = self._load_dataset(path, split)
        dataset_iterator = tf.data.make_one_shot_iterator(dataset)
        return dataset_iterator

    def _load_dataset(self, path, split):
        with open(os.path.join(path, "meta.json"), "r") as fp:
            meta = json.loads(fp.read())
        dataset = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
        return dataset.map(
            functools.partial(self._parse_data, meta=meta), num_parallel_calls=8
        ).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def cell_to_adj(cells):
        """creates adjancy matrix in COO format from mesh cells"""
        num_cells = np.shape(cells)[0]
        src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
        dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
        return src, dst

    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        """
        creates a DGL graph from an adj matrix in COO format.
        torch.int32 can handle graphs with up to 2**31-1 nodes or edges.
        """
        graph = dgl.to_bidirected(dgl.graph((src, dst), idtype=dtype))
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
    def _one_hot_encode(node_type):  # TODO generalize
        node_type = torch.squeeze(node_type, dim=-1)
        node_type = torch.where(
            node_type == 0,
            torch.zeros_like(node_type),
            node_type - 3,
        )
        node_type = F.one_hot(node_type.long(), num_classes=4)
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
        mask = torch.logical_or(
            torch.eq(node_type, torch.zeros_like(node_type)),
            torch.eq(
                node_type,
                torch.zeros_like(node_type) + 5,
            ),
        )
        return mask

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 2)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets

    @staticmethod
    def _parse_data(p, meta):
        outvar = {}
        feature_dict = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
        features = tf.io.parse_single_example(p, feature_dict)
        for k, v in meta["features"].items():
            data = tf.reshape(
                tf.io.decode_raw(features[k].values, getattr(tf, v["dtype"])),
                v["shape"],
            )
            if v["type"] == "static":
                data = tf.tile(data, [meta["trajectory_length"], 1, 1])
            elif v["type"] == "dynamic_varlen":
                row_len = tf.reshape(
                    tf.io.decode_raw(features["length_" + k].values, tf.int32), [-1]
                )
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=row_len)
            outvar[k] = data
        return outvar
