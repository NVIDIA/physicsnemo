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

# ruff: noqa: S101
import json
import logging
import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

logger = logging.getLogger("lmgn")


def compute_edge_index(pos, radius):
    """Computes graph connectivity based on pairwise distance.

    Parameters
    ----------
    pos : Tensor
        Node positions
    radius : float
        Connectivity radius

    Returns
    -------
    Tensor
        Edge indices
    """
    distances = torch.cdist(pos, pos, p=2)
    mask = distances < radius  # & (distances > 0) # include self-edge
    edge_index = torch.nonzero(mask).t().contiguous()
    return edge_index


def compute_edge_attr(graph: PyGData, radius: float = 0.015) -> PyGData:
    """Computes edge attributes (displacement and distance).

    Parameters
    ----------
    graph : PyGData
        Input graph
    radius : float, optional
        Radius for distance calculation, by default 0.015
    """
    edge_index = graph.edge_index
    displacement = graph.pos[edge_index[1]] - graph.pos[edge_index[0]]
    distance = torch.pairwise_distance(
        graph.pos[edge_index[0]],
        graph.pos[edge_index[1]],
        keepdim=True,
    )
    # direction = displacement / distance
    distance = torch.exp(-(distance**2) / radius**2)
    graph.edge_attr = torch.cat((displacement, distance), dim=-1)
    return graph


def graph_update(graph: PyGData, radius) -> PyGData:
    """Updates graph structure by reconstructing edges based on positions.

    Parameters
    ----------
    graph : PyGData
        Input graph
    radius : float
        Connectivity radius

    Returns
    -------
    PyGData
        Updated graph
    """
    graph.edge_index = compute_edge_index(graph.pos, radius)
    return compute_edge_attr(graph)


class LagrangianDataset(Dataset):
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
        The standard deviation of the noise added to the "train" split, by default 0.0003.
    radius : float, optional
        Connectivity radius, by default is set from the dataset metadata.
    dt : float, optional
        Time step increment, by default is set from the dataset metadata.
    bounds :
        Domain bounds, by default is set from the dataset metadata.
    num_node_types : int, optional
        Number of node types, by default 6.
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
    ):
        self.name = name
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
        tfrecord_dataset = self._load_tfrecord_dataset(self.data_dir, self.split)
        self.node_type = []
        self.rollout_mask = []
        self.node_features = []
        for i, data_np in enumerate(tfrecord_dataset):
            if i >= self.num_sequences:
                break

            position = torch.from_numpy(
                data_np["position"][: self.num_steps]
            )  # (num_steps, num_particles, dim)
            assert position.shape[0] == self.num_steps, f"{self.num_steps=}, {i=}"

            node_type = torch.from_numpy(data_np["particle_type"])  # (num_particles,)
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

        # Normalize velocity and acceleration.
        vel = self.normalize_velocity(vel)
        acc = self.normalize_acceleration(acc)

        # Create graph node features.
        node_features = self.pack_inputs(pos_t, vel[:-1], self.node_type[gidx])

        # Target position and velocity are for time t + 1, acceleration - for t.
        target_pos = pos[-1]
        target_vel = vel[-1]
        target_acc = acc[-1]

        node_targets = torch.cat((target_pos, target_vel, target_acc), dim=-1)

        graph = PyGData(num_nodes=node_features.shape[0])
        graph.x = node_features
        graph.y = node_targets
        graph.pos = pos_t
        graph.mask = mask
        graph.t = torch.tensor([tidx]).repeat(
            node_features.shape[0]
        )  # just to track the start
        graph_update(graph, radius=self.radius)

        return graph

    def normalize_velocity(self, velocity):
        """Normalizes velocity using dataset statistics.

        Parameters
        ----------
        velocity : Tensor
            Input velocity

        Returns
        -------
        Tensor
            Normalized velocity
        """
        velocity = velocity - self.vel_mean.to(velocity.device)
        velocity = velocity / self.vel_std.to(velocity.device)
        return velocity

    def denormalize_velocity(self, velocity):
        """Denormalizes velocity using dataset statistics.

        Parameters
        ----------
        velocity : Tensor
            Normalized velocity

        Returns
        -------
        Tensor
            Denormalized velocity
        """
        velocity = velocity * self.vel_std.to(velocity.device)
        velocity = velocity + self.vel_mean.to(velocity.device)
        return velocity

    def normalize_acceleration(self, acceleration):
        """Normalizes acceleration using dataset statistics.

        Parameters
        ----------
        acceleration : Tensor
            Input acceleration

        Returns
        -------
        Tensor
            Normalized acceleration
        """
        acceleration = acceleration - self.acc_mean.to(acceleration.device)
        acceleration = acceleration / self.acc_std.to(acceleration.device)
        return acceleration

    def denormalize_acceleration(self, acceleration):
        """Denormalizes acceleration using dataset statistics.

        Parameters
        ----------
        acceleration : Tensor
            Normalized acceleration

        Returns
        -------
        Tensor
            Denormalized acceleration
        """
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

    def pack_inputs(
        self, position: Tensor, vel_history: Tensor, node_type: Tensor
    ) -> Tensor:
        """Pack position, velocity history and node type into a single input tensor.

        Parameters
        ----------
        position : Tensor
            Current particle positions of shape (num_particles, dimension)
        vel_history : Tensor
            Velocity history of shape (num_history, num_particles, dimension)
        node_type : Tensor
            Node type features of shape (num_particles, num_node_types)

        Returns
        -------
        Tensor
            Concatenated input features of shape (num_particles, input_dimension)
            where input_dimension = dimension + num_history * dimension + num_boundary_features + num_node_types
        """
        # Boundary features for the current position.
        boundary_features = self.compute_boundary_feature(
            position, self.radius, bounds=self.bounds
        )

        # (num_history, num_particles, dimension) -> (num_particles, num_history * dimension)
        vel_history = vel_history.permute(1, 0, 2).flatten(start_dim=1)

        return torch.cat((position, vel_history, boundary_features, node_type), dim=-1)

    def unpack_inputs(self, graph: PyGData):
        """Unpacks the graph inputs into position, velocity and node type.

        Returns:
        --------
        Tuple
            position, velocity and node type inputs. Velocity is normalized.
        """
        ndata = graph.x
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : self.dim + self.dim * self.num_history]
        # (num_particles, t * dimension) -> (t, num_particles, dimension)
        vel = vel.reshape(-1, self.num_history, self.dim).permute(1, 0, 2)
        # (num_particles, num_node_types)
        node_type = ndata[..., -self.num_node_types :]
        return pos, vel, node_type

    def unpack_targets(self, graph: PyGData):
        """Unpacks the graph targets into position, velocity and acceleration.

        Returns:
        --------
        Tuple
            position, velocity, acceleration targets. Velocity and acceleration are normalized.
        """
        ndata = graph.y
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : 2 * self.dim]
        acc = ndata[..., 2 * self.dim : 3 * self.dim]
        return pos, vel, acc

    def random_walk_noise(self, num_steps: int, num_particles: int):
        """Generates random walk noise for particle positions.

        Parameters
        ----------
        num_steps : int
            Number of time steps
        num_particles : int
            Number of particles

        Returns
        -------
        Tensor
            Position noise
        """

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
        """Computes time differences between consecutive steps.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Time differences
        """
        return x[1:] - x[:-1]

    @staticmethod
    def compute_boundary_feature(position, radius=0.015, bounds=[0.1, 0.9]):
        """Computes boundary features based on distance to domain bounds.

        Parameters
        ----------
        position : Tensor
            Particle positions
        radius : float, optional
            Feature radius, by default 0.015
        bounds : list, optional
            Domain bounds, by default [0.1, 0.9]

        Returns
        -------
        Tensor
            Boundary features
        """
        distance = torch.cat([position - bounds[0], bounds[1] - position], dim=-1)
        features = torch.exp(-(distance**2) / radius**2)
        features[distance > radius] = 0
        return features

    @staticmethod
    def boundary_clamp(position, bounds=[0.1, 0.9], eps=0.001):
        """Clamps positions to stay within domain bounds.

        Parameters
        ----------
        position : Tensor
            Particle positions
        bounds : list, optional
            Domain bounds, by default [0.1, 0.9]
        eps : float, optional
            Boundary offset, by default 0.001

        Returns
        -------
        Tensor
            Clamped positions
        """
        return torch.clamp(position, min=bounds[0] + eps, max=bounds[1] - eps)

    def _load_tfrecord_dataset(self, path, split):
        """Load TFRecord dataset using the tfrecord package.

        Utility for loading the .tfrecord dataset from DeepMind's Learning to Simulate:
        https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

        Parameters
        ----------
        path : str
            Path to the directory containing TFRecord files and metadata.json.
        split : str
            Dataset split name (e.g., "train", "valid", "test").

        Returns
        -------
        TFRecordDataset
            An iterable dataset that yields decoded records.
        """
        with open(os.path.join(path, "metadata.json"), "r") as fp:
            meta = json.loads(fp.read())

        tfrecord_path = os.path.join(path, split + ".tfrecord")
        # Check for index file (enables multi-worker DataLoader).
        index_path = os.path.join(path, split + ".tfindex")
        if not os.path.exists(index_path):
            index_path = None

        # Define context (static) feature description.
        description = {
            "key": "int",
            "particle_type": "byte",
        }

        # Define sequence feature description for SequenceExample format.
        sequence_description = {
            "position": "byte",
        }
        if "context_mean" in meta:
            sequence_description["step_context"] = "byte"

        # Create dataset with transform to decode records.
        dataset = TFRecordDataset(
            tfrecord_path,
            index_path,
            description,
            transform=lambda rec: self._decode_record(rec, meta),
            sequence_description=sequence_description,
        )
        return dataset

    @staticmethod
    def _decode_record(rec: tuple, meta: dict) -> dict:
        """Decode raw bytes from TFRecord SequenceExample into numpy arrays.

        The tfrecord package parses the TFRecord and provides raw bytes
        for each feature, which are decoded using numpy.

        Parameters
        ----------
        rec : tuple
            Tuple of (context_dict, sequence_dict) from tfrecord package.
        meta : dict
            Metadata dictionary containing sequence_length and dim.

        Returns
        -------
        dict
            Dictionary with 'position' and 'particle_type' arrays.
        """
        context, sequence = rec

        # Decode particle_type from context (int64 encoded as bytes).
        # Use .copy() to make array writable (np.frombuffer returns read-only view).
        particle_type = np.frombuffer(context["particle_type"], dtype=np.int64).copy()

        # Decode position from sequence features.
        # Each element in sequence["position"] is bytes for one timestep.
        position_list = []
        for pos_bytes in sequence["position"]:
            pos = np.frombuffer(pos_bytes, dtype=np.float32)
            position_list.append(pos)

        # Stack positions: shape (num_steps, num_particles * dim).
        # np.stack creates a new writable array from the read-only views.
        position = np.stack(position_list, axis=0)
        # Reshape to (num_steps, num_particles, dim).
        num_steps = position.shape[0]
        dim = meta["dim"]
        num_particles = position.shape[1] // dim
        position = position.reshape(num_steps, num_particles, dim)

        result = {
            "position": np.ascontiguousarray(position),
            "particle_type": particle_type,
        }

        # Handle optional step_context.
        if "step_context" in sequence:
            context_list = []
            for ctx_bytes in sequence["step_context"]:
                ctx = np.frombuffer(ctx_bytes, dtype=np.float32)
                context_list.append(ctx)
            result["step_context"] = np.ascontiguousarray(
                np.stack(context_list, axis=0)
            )

        return result

    def get_kinematic_mask(self, graph_idx: int) -> Tensor:
        """Returns mask for kinematic particles in a graph.

        Parameters
        ----------
        graph_idx : int
            Graph index

        Returns
        -------
        Tensor
            Boolean mask for kinematic particles
        """
        return self.node_type[graph_idx][:, self.KINEMATIC_PARTICLE_ID] != 0
