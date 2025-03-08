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

# ruff: noqa: S101
import functools
import json
import logging
import os
from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. "
        'Install: pip install "tensorflow<=2.17.1"'
    )

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )

from .lagrangian_reading_utils import parse_serialized_simulation_example

# Hide GPU from visible devices for TF
tf.config.set_visible_devices([], "GPU")

logger = logging.getLogger("lmgn")


def compute_edge_index(pos, radius):
    # compute the graph connectivity using pairwise distance
    distances = torch.cdist(pos, pos, p=2)
    mask = distances < radius  # & (distances > 0) # include self-edge
    edge_index = torch.nonzero(mask).t().contiguous()
    return edge_index


def compute_edge_attr(graph, radius=0.015):
    # compute the displacement and distance per edge
    edge_index = graph.edges()
    displacement = graph.ndata["pos"][edge_index[1]] - graph.ndata["pos"][edge_index[0]]
    distance = torch.pairwise_distance(
        graph.ndata["pos"][edge_index[0]],
        graph.ndata["pos"][edge_index[1]],
        keepdim=True,
    )
    # direction = displacement / distance
    distance = torch.exp(-(distance**2) / radius**2)
    graph.edata["x"] = torch.cat((displacement, distance), dim=-1)
    return


def graph_update(graph, radius):
    """Updates the graph structure.

    Removes all previous edges and re-constructs
    the graph using pair-wise distance.

    """
    # TODO: use more efficient graph construction method
    num_edges = graph.num_edges()
    if num_edges > 0:
        graph.remove_edges(torch.arange(num_edges, device=graph.device))
    pos = graph.ndata["pos"]
    edge_index = compute_edge_index(pos, radius)
    graph.add_edges(edge_index[0], edge_index[1])
    compute_edge_attr(graph)
    return graph


class LagrangianDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for Lagrangian mesh.
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "valid", "test"], by default "train"
    num_sequences : int, optional
        Number of sequences, by default 1000
    num_history : int, optional.
        Number of velocities, including the current, to include in the history, by default 5.
    num_steps : int, optional
        Number of time steps in each sequence, by default is set from the dataset metadata.
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.0003
    radius : float, optional
        Connectivity radius, by default is set from the dataset metadata.
    dt : float, optional
        Time step increment, by default is set from the dataset metadata.
    bounds :
        Domain bounds, by default is set from the dataset metadata.
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """

    KINEMATIC_PARTICLE_ID = 3  # See train.py in DeepMind code.

    def __init__(
        self,
        name: str = "dataset",
        data_dir: Optional[str] = None,
        split: str = "train",
        num_sequences: int = 1000,
        num_history: int = 5,
        num_steps: Optional[int] = None,
        noise_std: float = 0.0003,
        radius: Optional[float] = None,
        dt: Optional[float] = None,
        bounds: Optional[Sequence[tuple[float, float]]] = None,
        num_node_types: int = 6,
        force_reload: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = data_dir
        self.split = split
        self.num_sequences = num_sequences
        self.num_history = num_history
        self.noise_std = noise_std
        self.num_node_types = num_node_types

        path_metadata = os.path.join(data_dir, "metadata.json")
        with open(path_metadata, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        # Note: DeepMind datasets contain sequence_length + 1 time steps for each sequence.
        self.num_steps = (
            (metadata["sequence_length"] + 1) if num_steps is None else num_steps
        )
        self.dt = metadata["dt"] if dt is None else dt
        self.radius = (
            metadata["default_connectivity_radius"] if radius is None else radius
        )
        # Assuming bounds are the same for all dimensions.
        self.bounds = metadata["bounds"][0] if bounds is None else bounds[0]
        self.dim = metadata["dim"]

        self.vel_mean = torch.tensor(metadata["vel_mean"]).reshape(1, self.dim)
        self.vel_std = torch.tensor(metadata["vel_std"]).reshape(1, self.dim)
        self.acc_mean = torch.tensor(metadata["acc_mean"]).reshape(1, self.dim)
        self.acc_std = torch.tensor(metadata["acc_std"]).reshape(1, self.dim)

        # Create the node features.
        logger.info(f"Preparing the {split} dataset...")
        dataset_iterator = self._load_tf_data(self.data_dir, self.split)
        self.node_type = []
        self.rollout_mask = []
        self.node_features = []
        for i in range(self.num_sequences):
            data_np = dataset_iterator.get_next()

            position = torch.from_numpy(
                data_np[1]["position"][: self.num_steps].numpy()
            )  # (num_steps, num_particles, 2)
            assert position.shape[0] == self.num_steps, f"{self.num_steps=}, {i=}"

            node_type = torch.from_numpy(
                data_np[0]["particle_type"].numpy()
            )  # (num_particles,)
            assert node_type.shape[0] == position.shape[1], f"{i=}"

            features = {}
            features["position"] = position[: self.num_steps]

            self.node_type.append(F.one_hot(node_type, num_classes=self.num_node_types))
            self.node_features.append(features)

        # For each sequence, there are (num_steps - num_history - 1) values
        # with velocity and acceleration.
        self.num_samples_per_sequence = self.num_steps - self.num_history - 1
        self.length = num_sequences * self.num_samples_per_sequence

        logger.info("Finished dataset preparation.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not (0 <= idx < self.length):
            raise IndexError(f"Invalid index {idx}, must be in [0, {self.length})")

        # graph and time step indices.
        gidx, tidx = divmod(idx, self.num_samples_per_sequence)

        # Current time step.
        t = tidx + self.num_history
        pos = self.node_features[gidx]["position"][tidx : t + 2]
        assert len(pos) == self.num_history + 2
        # Current position at t.
        pos_t = pos[-2]

        # Mask for material particles (i.e. non-kinematic).
        mask = ~self.get_kinematic_mask(gidx)
        # Add noise.
        if self.split == "train":
            pos_noise = self.random_walk_noise(*pos.shape[:2])
            # Do not apply noise to kinematic particles.
            pos_noise *= mask.unsqueeze(-1)
            # Add noise to positions.
            pos += pos_noise

        # Velocities.
        vel = self.time_diff(pos)
        # Target acceleration.
        acc = self.time_diff(vel[-2:])

        # Boundary features for the current position.
        boundary_features = self.compute_boundary_feature(
            pos_t, self.radius, bounds=self.bounds
        )

        # Normalize velocity and acceleration.
        vel = self.normalize_velocity(vel)
        acc = self.normalize_acceleration(acc)

        # Create graph node features.
        # (num_history, num_particles, dimension) -> (num_particles, num_history * dimension)
        vel_history = vel[:-1].permute(1, 0, 2).flatten(start_dim=1)

        node_features = torch.cat(
            (pos_t, vel_history, boundary_features, self.node_type[gidx]), dim=-1
        )

        # Target position and velocity are for time t + 1, acceleration - for t.
        target_pos = pos[-1]
        target_vel = vel[-1]
        target_acc = acc[-1]

        node_targets = torch.cat((target_pos, target_vel, target_acc), dim=-1)

        graph = dgl.graph(([], []), num_nodes=node_features.shape[0])
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.ndata["pos"] = pos_t
        graph.ndata["mask"] = mask
        graph.ndata["t"] = torch.tensor([tidx]).repeat(
            node_features.shape[0]
        )  # just to track the start
        graph_update(graph, radius=self.radius)

        return graph

    def normalize_velocity(self, velocity):
        velocity = velocity - self.vel_mean.to(velocity.device)
        velocity = velocity / self.vel_std.to(velocity.device)
        return velocity

    def denormalize_velocity(self, velocity):
        velocity = velocity * self.vel_std.to(velocity.device)
        velocity = velocity + self.vel_mean.to(velocity.device)
        return velocity

    def normalize_acceleration(self, acceleration):
        acceleration = acceleration - self.acc_mean.to(acceleration.device)
        acceleration = acceleration / self.acc_std.to(acceleration.device)
        return acceleration

    def denormalize_acceleration(self, acceleration):
        acceleration = acceleration * self.acc_std.to(acceleration.device)
        acceleration = acceleration + self.acc_mean.to(acceleration.device)
        return acceleration

    def time_integrator(self, position, velocity, acceleration, dt, denormalize=True):
        """Semi-implicit Euler integration.

        Given the position x(t), velocity v(t), and acceleration a(t)
        computes next step position and velocity.

        Returns:
        --------
        Tuple
            position, velocity for t + 1
        """

        if denormalize:
            velocity = self.denormalize_velocity(velocity)
            acceleration = self.denormalize_acceleration(acceleration)

        velocity_next = velocity + acceleration  # * dt
        position_next = position + velocity_next  # * dt
        return position_next, velocity_next

    def unpack_inputs(self, graph: dgl.DGLGraph):
        """Unpacks the graph inputs into position and velocity.

        Returns:
        --------
        Tuple
            position, velocity inputs. Velocity is normalized.
        """
        ndata = graph.ndata["x"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : self.dim + self.dim * self.num_history]
        # (num_particles, t * dimension) -> (t, num_particles, dimension)
        vel = vel.reshape(-1, self.num_history, self.dim).permute(1, 0, 2)
        return pos, vel

    def unpack_targets(self, graph: dgl.DGLGraph):
        """Unpacks the graph targets into position, velocity and acceleration.

        Returns:
        --------
        Tuple
            position, velocity, acceleration targets. Velocity and acceleration are normalized.
        """
        ndata = graph.ndata["y"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : 2 * self.dim]
        acc = ndata[..., 2 * self.dim : 3 * self.dim]
        return pos, vel, acc

    def random_walk_noise(self, num_steps: int, num_particles: int):
        """Creates random walk noise for positions."""

        num_velocities = num_steps - 1
        # See comments in get_random_walk_noise_for_position_sequence in DeepMind code.
        std_each_step = self.noise_std / num_velocities**0.5
        vel_noise = std_each_step * torch.randn(num_velocities, num_particles, self.dim)

        # Apply the random walk to velocities.
        vel_noise = vel_noise.cumsum(dim=0)

        # Integrate to get position noise with no noise at the first step.
        pos_noise = torch.cat(
            (torch.zeros(1, *vel_noise.shape[1:]), vel_noise.cumsum(dim=0))
        )

        # Set the target position noise the same as the current so it cancels out
        # during velocity calculation.
        # See get_predicted_and_target_normalized_accelerations in DeepMind code.
        pos_noise[-1] = pos_noise[-2]

        return pos_noise

    @staticmethod
    def time_diff(x: Tensor):
        return x[1:] - x[:-1]

    @staticmethod
    def compute_boundary_feature(position, radius=0.015, bounds=[0.1, 0.9]):
        distance = torch.cat([position - bounds[0], bounds[1] - position], dim=-1)
        features = torch.exp(-(distance**2) / radius**2)
        features[distance > radius] = 0
        return features

    @staticmethod
    def boundary_clamp(position, bounds=[0.1, 0.9], eps=0.001):
        return torch.clamp(position, min=bounds[0] + eps, max=bounds[1] - eps)

    def _load_tf_data(self, path, split):
        """
        Utility for loading the .tfrecord dataset in DeepMind's MeshGraphNet repo:
        https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate
        Follow the instructions provided in that repo to download the .tfrecord files.
        """
        dataset = self._load_dataset(path, split)
        dataset_iterator = tf.data.make_one_shot_iterator(dataset)
        return dataset_iterator

    def _load_dataset(self, path, split):
        with open(os.path.join(path, "metadata.json"), "r") as fp:
            meta = json.loads(fp.read())
        dataset = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
        return dataset.map(
            functools.partial(parse_serialized_simulation_example, metadata=meta),
            num_parallel_calls=8,
        ).prefetch(tf.data.AUTOTUNE)

    def get_kinematic_mask(self, graph_idx: int) -> Tensor:
        """Returns kinematic particles mask for a graph."""
        return self.node_type[graph_idx][:, self.KINEMATIC_PARTICLE_ID] != 0
