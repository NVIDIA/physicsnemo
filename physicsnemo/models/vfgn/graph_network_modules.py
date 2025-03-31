# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU

try:
    from torch_scatter import scatter
except ImportError:
    raise ImportError(
        "VFGN pipeline requires the PyTorch_Geometric library. Install the "
        + "package at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )

from dataclasses import dataclass

from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..meta import ModelMetaData
from ..module import Module

STD_EPSILON = 1e-8


@dataclass
class MetaData(ModelMetaData):
    name: str = "VFGN"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class MLPNet(Module):
    """
    A Multilayer Perceptron (MLP) network implemented in PyTorch, configurable with
    a variable number of hidden layers and layer normalization.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    output_size : int
        Number of output channels
    layer_norm : boolean
        If to apply layer normalization in the output layer, default True

    Example
    -------
    # # Use MLPNet to encode the features
    # >>> model = physicsnemo.models.graph_network.MLPNet(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... output_size=128)
    # >>> input = torch.randn([5193, 128]) #(N, C)
    # >>> output = model(input)
    # >>> output.size()
    # torch.Size([5193, 128])
    ----
    """

    def __init__(
        self,
        mlp_hidden_size: int = 128,
        mlp_num_hidden_layers: int = 2,
        output_size: int = 128,
        layer_norm: bool = True,
    ):
        if not (
            mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and output_size >= 0
        ):
            raise ValueError("Invalid arch params")
        super().__init__(meta=MetaData(name="vfgn_mlpnet"))

        # create cnt = hidden_layer layers
        self.mlp_hidden_size = mlp_hidden_size
        self.lins = []
        if mlp_num_hidden_layers > 1:
            for i in range(mlp_num_hidden_layers - 1):
                self.lins.append(Linear(mlp_hidden_size, mlp_hidden_size))
        self.lins = torch.nn.ModuleList(self.lins)

        # create output layer
        self.lin_e = Linear(mlp_hidden_size, output_size)
        self.layer_norm = layer_norm
        self.relu = ReLU()

    def dynamic(self, name: str, module_class, *args, **kwargs):
        """Use dynamic layer to create 1st layer according to the input node number"""
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)

    def forward(self, x):
        origin_device = x.device
        lin_s = self.dynamic("lin_s", Linear, x.shape[-1], self.mlp_hidden_size)
        lin_s = lin_s.to(origin_device)

        x = lin_s(x)
        x = self.relu(x)

        for lin_i in self.lins:
            x = lin_i(x)
            x = self.relu(x)

        x = self.lin_e(x)
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[1:])
        return x


class EncoderNet(Module):
    """
    Construct EncoderNet based on the NLPNet architecture.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    """

    def __init__(
        self,
        mlp_hidden_size: int = 128,
        mlp_num_hidden_layers: int = 2,
        latent_size: int = 128,
    ):
        if not (
            mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and latent_size >= 0
        ):
            raise ValueError("Invalid arch params - EncoderNet")

        super().__init__(meta=MetaData(name="vfgn_encoder"))

        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers

        self.edge_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
        self.node_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, node_attr, edge_attr):
        # encode node attributes
        node_attr = self.node_mlp(node_attr)
        # encode edge attributes
        edge_attr = self.edge_mlp(edge_attr)

        return node_attr, edge_attr


class EdgeBlock(Module):
    """
    Update the edge attributes by collecting the sender and/or receiver-nodes'
    edge attributes, pass through the edge-MLP network.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    use_receiver_nodes : bool, optional, default = True
        whether to take the receiver-node's edges atrributes into compute
    use_sender_nodes : bool, optional, default = True
        whether to take the sender-node's edges atrributes into compute

    Example
    -------
    # >>> #2D convolutional encoder decoder
    # >>> model = physicsnemo.models.graph_network.EdgeBlock(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... latent_size=128,
    # ... node_dim=0)
    # >>> input = (node_attr, edge_attr, receiver_list, sender_list)
    # >>> output = node_attr, updated_edge_attr, receiver_list, sender_list
    # >>> output.size()

    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        node_dim=0,
        use_receiver_nodes=True,
        use_sender_nodes=True,
    ):
        super().__init__(meta=MetaData(name="vfgn_edgeblock"))
        self.node_dim = node_dim
        self._edge_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

        self.use_receiver_nodes = use_receiver_nodes
        self.use_sender_nodes = use_sender_nodes

    def forward(self, node_attr, edge_attr, receivers, senders):
        edges_to_collect = []
        edges_to_collect.append(edge_attr)

        if self.use_receiver_nodes:
            receivers_edge = node_attr[receivers, :]
            edges_to_collect.append(receivers_edge)

        if self.use_sender_nodes:
            senders_edge = node_attr[senders, :]
            edges_to_collect.append(senders_edge)

        collected_edges = torch.cat(edges_to_collect, axis=-1)

        updated_edges = self._edge_model(collected_edges)

        return node_attr, updated_edges, receivers, senders


class NodeBlock(Module):
    """
    Update the nodes attributes by collecting the sender and/or receiver-nodes'
    edge attributes, pass through the node-MLP network.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    use_receiver_nodes : bool, optional, default = True
        whether to take the receiver-node's edges atrributes into compute
    use_sender_nodes : bool, optional, default = True
        whether to take the sender-node's edges atrributes into compute

    # Example
    # -------
    # >>> #2D convolutional encoder decoder
    # >>> model = physicsnemo.models.graph_network.NodeBlock(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... latent_size=128,
    # ... node_dim=0)
    # >>> input = (node_attr, edge_attr, receiver_list, sender_list)
    # >>> output = updated_node_attr, edge_attr, receiver_list, sender_list
    # >>> output.size()

    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
        use_received_edges=True,
        use_sent_edges=False,
    ):
        super().__init__(meta=MetaData(name="vfgn_nodeblock"))
        self.aggr = aggr
        self.node_dim = node_dim

        self.use_received_edges = use_received_edges
        self.use_sent_edges = use_sent_edges

        self._node_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, x, edge_attr, receivers, senders):
        nodes_to_collect = []
        nodes_to_collect.append(x)

        dim_size = x.shape[self.node_dim]

        # aggregate received edges
        if self.use_received_edges:
            receivers_edge = scatter(
                dim=self.node_dim,
                dim_size=dim_size,
                index=receivers,
                src=edge_attr,
                reduce=self.aggr,
            )
            nodes_to_collect.append(receivers_edge)

        # aggregate sent edges
        if self.use_sent_edges:
            senders_edge = scatter(
                dim=self.node_dim,
                dim_size=dim_size,
                index=senders,
                src=edge_attr,
                reduce=self.aggr,
            )
            nodes_to_collect.append(senders_edge)

        collected_nodes = torch.cat(nodes_to_collect, axis=-1)

        updated_nodes = self._node_model(collected_nodes)

        return updated_nodes, edge_attr, receivers, senders


class InteractionNet(torch.nn.Module):
    """
    Iterate to compute the edge attributes, then node attributes

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
    ):
        super(InteractionNet, self).__init__()
        self._edge_block = EdgeBlock(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )
        self._node_block = NodeBlock(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )

    def forward(self, x, edge_attr, receivers, senders):
        if not (x.shape[-1] == edge_attr.shape[-1]):
            raise ValueError("node feature size should equal to edge feature size")

        return self._node_block(*self._edge_block(x, edge_attr, receivers, senders))


class ResInteractionNet(torch.nn.Module):
    """
    Update the edge attributes and node attributes

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
    ):
        super(ResInteractionNet, self).__init__()
        self.itn = InteractionNet(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )

    def forward(self, x, edge_attr, receivers, senders):
        x_res, edge_attr_res, receivers, senders = self.itn(
            x, edge_attr, receivers, senders
        )

        x_new = x + x_res
        edge_attr_new = edge_attr + edge_attr_res

        return x_new, edge_attr_new, receivers, senders


class DecoderNet(Module):
    """
    Construct DecoderNet based on the NLPNet architecture. Used for
    decoding the predicted features with multi-layer perceptron network module

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    output_size : int
        Number of output channels
    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, output_size):
        if not (
            mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and output_size >= 0
        ):
            raise ValueError("Invalid arch params - DecoderNet")
        super().__init__(meta=MetaData(name="vfgn_decoder"))
        self.mlp = MLPNet(
            mlp_hidden_size, mlp_num_hidden_layers, output_size, layer_norm=False
        )

    def forward(self, x):
        # number of layer is important, or the network will overfit
        x = self.mlp(x)
        return x


class EncodeProcessDecode(Module):
    """
    Construct the network architecture that consists of Encoder - Processor - Decoder modules

    Parameters
    ----------
    latent_size : int
        Number of latent channels
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    num_message_passing_steps : int, default = 10
        Number of message passing steps
    output_size : int
        Number of output channels
    device_list : list[str], optional
        device to execute the computation

    # Example
    # -------
    # >>> #Use EncodeProcessDecode to update the node, edge features
    # >>> model = physicsnemo.models.graph_network.EncodeProcessDecode(
    # ... latent_size=128,
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... num_message_passing_steps=10,
    # ... output_size=3)
    # >>> node_attr = torch.randn([1394, 61]) #(node_cnt, node_feat_sizes)
    # >>> edge_attr = torch.randn([5193, 4]) #(edge_cnt, edge_feat_sizes)
    # >>> invar_receivers = torch.Size([5193]) : int # node index list
    # >>> invar_senders = torch.Size([5193]) : int # node index list
    # >>> invar = (node_attr, edge_attr, invar_receivers, invar_senders)
    # >>> output = model(*invar, )
    # >>> output.size()
    # torch.Size([1394, 3])    #(node_cnt, output_size)
    """

    def __init__(
        self,
        latent_size,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        num_message_passing_steps,
        output_size,
        device_list=None,
    ):
        if not (latent_size > 0 and mlp_hidden_size > 0 and mlp_num_hidden_layers > 0):
            raise ValueError("Invalid arch params - EncodeProcessDecode")
        if not (num_message_passing_steps > 0):
            raise ValueError("Invalid arch params - EncodeProcessDecode")
        super().__init__(meta=MetaData(name="vfgn_encoderprocess_decode"))

        if device_list is None:
            self.device_list = ["cpu"]
        else:
            self.device_list = device_list

        self._encoder_network = EncoderNet(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size
        )

        self._processor_networks = []
        for _ in range(num_message_passing_steps):
            self._processor_networks.append(
                InteractionNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
            )
        self._processor_networks = torch.nn.ModuleList(self._processor_networks)

        self._decoder_network = DecoderNet(
            mlp_hidden_size, mlp_num_hidden_layers, output_size
        )

    def set_device(self, device_list):
        """device list"""
        self.device_list = device_list

    def forward(self, x, edge_attr, receivers, senders):
        """
        x:
            Torch tensor of node attributes, shape: (batch_size, node_number, feature_size)
        edge_attr:
            Torch tensor of edge_attr attributes, shape: (batch_size, edge_number, feature_size)
        receivers/ senders:
            Torch tensor, list of node indexes, shape: (batch_size,  edge_list_size:[list of node indexes])
        """
        # todo: uncomment
        # self.device_list = x.device.type  # decide the device type
        x, edge_attr = self._encoder_network(x, edge_attr)

        pre_x = x
        pre_edge_attr = edge_attr

        n_steps = len(self._processor_networks)
        # n_inter = int(n_steps)  # prevent divide by zero
        # todo: check the multi-gpus
        n_inter = int(n_steps / len(self.device_list))

        i = 0
        j = 0

        origin_device = x.device

        for processor_network_k in self._processor_networks:
            # todo: device_list
            # p_device = self.device_list  # [j]
            p_device = self.device_list[j]
            processor_network_k = processor_network_k.to(p_device)
            pre_x = pre_x.to(p_device)
            pre_edge_attr = pre_edge_attr.to(p_device)
            receivers = receivers.to(p_device)
            senders = senders.to(p_device)

            diff_x, diff_edge_attr, receivers, senders = checkpoint(
                processor_network_k, pre_x, pre_edge_attr, receivers, senders
            )

            pre_x = x.to(p_device) + diff_x
            pre_edge_attr = edge_attr.to(p_device) + diff_edge_attr
            i += 1
            if i % n_inter == 0:
                j += 1

        x = self._decoder_network(pre_x.to(origin_device))

        return x


class LearnedSimulator(Module):
    """
    Construct the Simulator model architecture

    Parameters
    ----------
    num_dimensions : int
        Number of dimensions to make the prediction
    num_seq : int
        Number of sintering steps
    boundaries : list[list[float]]
        boundary value that the object is placed/ normalized in,
        i.e.[[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    num_particle_types : int
        Number of types to differentiate the different nodes, i.e. fixed/ moving nodes
    particle_type_embedding_size: int
        positional embedding dimension with different particle types,
        in torch.nn.Embedding()
    normalization_stats: dict{list[float]}
        Stored in metadata.json
        {'acceleration': acceleration_stats, 'velocity': velocity_stats, 'context': context_stats}
    graph_mode : str, optional
    connectivity_param: float
        Distance to normalize the displacement between nodes

    Example
    -------
    # >>> model = physicsnemo.models.graph_network.LearnedSimulator(
    # ... num_dimensions=3*5, # metadata['dim'] * PREDICT_LENGTH
    # ... num_seq=2,
    # ... boundaries=128)

    # >>> input = torch.randn([5193, 128]) #(N, C)
    # >>> output = model(input)
    # >>> output.size()
    # torch.Size([5193, 128])
    ----
    """

    def __init__(
        self,
        num_dimensions: int = 3,
        num_seq: int = 5,
        boundaries: list[list[float]] = None,
        num_particle_types: int = 3,
        particle_type_embedding_size: int = 16,
        normalization_stats: map = None,
        graph_mode: str = "radius",
        connectivity_param: float = 0.015,
    ):
        if not (num_dimensions >= 0 and num_seq >= 3):
            raise ValueError("Invalid arch params - LearnedSimulator")
        super().__init__(meta=MetaData(name="vfgn_simulator"))

        # network parameters
        self._latent_size = 128
        self._mlp_hidden_size = 128
        self._mlp_num_hidden_layers = 2
        self._num_message_passing_steps = 10
        self._num_dimensions = num_dimensions
        self._num_seq = num_seq

        # graph parameters
        self._connectivity_param = connectivity_param  # either knn or radius
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats

        self.graph_mode = graph_mode

        self._graph_network = EncodeProcessDecode(
            self._latent_size,
            self._mlp_hidden_size,
            self._mlp_num_hidden_layers,
            self._num_message_passing_steps,
            self._num_dimensions,
        )

        # positional embedding with different particle types
        self._num_particle_types = num_particle_types
        self.embedding = Embedding(
            self._num_particle_types + 1, particle_type_embedding_size
        )
        self.message_passing_devices = []

    def setMessagePassingDevices(self, devices):
        """
        setts the devices to be used for message passing in the neural network model.
        """
        self.message_passing_devices = devices

    def to(self, device):
        """Device transfer"""
        new_self = super(LearnedSimulator, self).to(device)
        new_self._boundaries = self._boundaries.to(device)
        for key in self._normalization_stats:
            new_self._normalization_stats[key].to(device)
        if device != "cpu":
            self._graph_network.set_device(self.message_passing_devices)
        return new_self

    def time_diff(self, input_seq):
        """
        Calculates the difference between consecutive elements in a sequence, effectively computing the discrete time derivative.
        """
        return input_seq[:, 1:] - input_seq[:, :-1]

    def _compute_connectivity_for_batch(
        self, senders_list, receivers_list, n_node, n_edge
    ):
        """
        Dynamically update the edge features with random dropout
        For each graph, randomly select whether apply edge drop-out to this node
        If applying random drop-out, a default drop_out_rate = 0.6 is applied to the edges
        """
        senders_per_graph_list = np.split(senders_list, np.cumsum(n_edge[:-1]), axis=0)
        receivers_per_graph_list = np.split(
            receivers_list, np.cumsum(n_edge[:-1]), axis=0
        )

        receivers_list = []
        senders_list = []
        n_edge_list = []
        num_nodes_in_previous_graphs = 0

        n = n_node.shape[0]

        drop_out_rate = 0.6

        # Compute connectivity for each graph in the batch.
        for i in range(n):
            total_num_edges_graph_i = len(senders_per_graph_list[i])

            random_num = False  # random.choice([True, False])

            if random_num:
                choiced_indices = random.choices(
                    [j for j in range(total_num_edges_graph_i)],
                    k=int(total_num_edges_graph_i * drop_out_rate),
                )
                choiced_indices = sorted(choiced_indices)

                senders_graph_i = senders_per_graph_list[i][choiced_indices]
                receivers_graph_i = receivers_per_graph_list[i][choiced_indices]
            else:
                senders_graph_i = senders_per_graph_list[i]
                receivers_graph_i = receivers_per_graph_list[i]

            num_edges_graph_i = len(senders_graph_i)
            n_edge_list.append(num_edges_graph_i)

            # Because the inputs will be concatenated, we need to add offsets to the
            # sender and receiver indices according to the number of nodes in previous
            # graphs in the same batch.
            receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
            senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

            num_nodes_graph_i = n_node[i]
            num_nodes_in_previous_graphs += num_nodes_graph_i

        # Concatenate all of the results.
        senders = np.concatenate(senders_list, axis=0).astype(np.int32)
        receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)

        return senders, receivers

    def get_random_walk_noise_for_position_sequence(
        self, position_sequence, noise_std_last_step
    ):
        """Returns random-walk noise in the velocity applied to the position."""

        velocity_sequence = self.time_diff(position_sequence)

        # We want the noise scale in the velocity at the last step to be fixed.
        # Because we are going to compose noise at each step using a random_walk:
        # std_last_step**2 = num_velocities * std_each_step**2
        # so to keep `std_last_step` fixed, we apply at each step:
        # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
        # TODO(alvarosg): Make sure this is consistent with the value and
        # description provided in the paper.
        num_velocities = velocity_sequence.shape[1]
        velocity_sequence_noise = torch.empty(
            velocity_sequence.shape, dtype=velocity_sequence.dtype
        ).normal_(
            mean=0, std=noise_std_last_step / num_velocities**0.5
        )  # float

        # Apply the random walk
        velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

        # Integrate the noise in the velocity to the positions, assuming
        # an Euler intergrator and a dt = 1, and adding no noise to the very first
        # position (since that will only be used to calculate the first position
        # change).
        position_sequence_noise = torch.cat(
            [
                torch.zeros(
                    velocity_sequence_noise[:, 0:1].shape, dtype=velocity_sequence.dtype
                ),
                torch.cumsum(velocity_sequence_noise, axis=1),
            ],
            axis=1,
        )

        return position_sequence_noise

    def EncodingFeature(
        self,
        position_sequence,
        n_node,
        n_edge,
        senders_list,
        receivers_list,
        global_context,
        particle_types,
    ):
        """
        Feature encoder contains 3 parts:
            - Adding the node features that includes: position, velocity, sequence of accelerations
            - Adding the edge features with random dropout applied
            - Adding the global features to the node features, in this case, sintering temperature is includes
        """
        # aggregate all features
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = self.time_diff(position_sequence)
        acceleration_sequence = self.time_diff(velocity_sequence)

        # dynamically updage the graph
        senders, receivers = self._compute_connectivity_for_batch(
            senders_list.cpu().detach().numpy(),
            receivers_list.cpu().detach().numpy(),
            n_node.cpu().detach().numpy(),
            n_edge.cpu().detach().numpy(),
        )
        senders = torch.LongTensor(senders).to(position_sequence.device)
        receivers = torch.LongTensor(receivers).to(position_sequence.device)

        # 1. Node features
        node_features = []
        velocity_stats = self._normalization_stats["velocity"]

        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats.mean
        ) / velocity_stats.std
        normalized_velocity_sequence = normalized_velocity_sequence[:, -1]

        flat_velocity_sequence = normalized_velocity_sequence.reshape(
            [normalized_velocity_sequence.shape[0], -1]
        )
        node_features.append(flat_velocity_sequence)

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration_sequence = (
            acceleration_sequence - acceleration_stats.mean
        ) / acceleration_stats.std

        flat_acceleration_sequence = normalized_acceleration_sequence.reshape(
            [normalized_acceleration_sequence.shape[0], -1]
        )
        node_features.append(flat_acceleration_sequence)

        if self._num_particle_types > 1:
            particle_type_embedding = self.embedding(particle_types)
            node_features.append(particle_type_embedding)

        # 2. Edge features
        edge_features = []
        # Relative displacement and distances normalized to radius
        # normalized_relative_displacements = (most_recent_position.index_select(0, senders) -
        #                                      most_recent_position.index_select(0, receivers)) / self._connectivity_param

        normalized_relative_displacements = (
            most_recent_position.index_select(0, senders.squeeze())
            - most_recent_position.index_select(0, receivers.squeeze())
        ) / self._connectivity_param
        edge_features.append(normalized_relative_displacements)
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        # 3. Normalized the global context.
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            # Context in some datasets are all zero, so add an epsilon for numerical
            global_context = (global_context - context_stats.mean) / torch.maximum(
                context_stats.std,
                torch.FloatTensor([STD_EPSILON]).to(context_stats.std.device),
            )

            global_features = []
            # print("repeat_interleave n_node: ", n_node)
            # print("global_context: ", global_context.shape)
            for i in range(global_context.shape[0]):
                global_context_ = torch.unsqueeze(global_context[i], 0)
                context_i = torch.repeat_interleave(
                    global_context_, n_node[i].to(torch.long), dim=0
                )

                global_features.append(context_i)

            global_features = torch.cat(global_features, 0)
            global_features = global_features.reshape(global_features.shape[0], -1)

            node_features.append(global_features)

        x = torch.cat(node_features, -1)
        edge_attr = torch.cat(edge_features, -1)

        #  cast from double to float as the input of network
        x = x.float()
        edge_attr = edge_attr.float()

        return x, edge_attr, senders, receivers

    def DecodingFeature(
        self, normalized_accelerations, position_sequence, predict_length
    ):
        """Feature decoder"""
        #  cast from float to double as the output of network
        normalized_accelerations = normalized_accelerations.double()

        # model works on the normal space - need to invert it to the original space
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_accelerations = normalized_accelerations.reshape(
            [-1, predict_length, 3]
        )

        accelerations = (
            normalized_accelerations * acceleration_stats.std
        ) + acceleration_stats.mean
        velocity_changes = torch.cumsum(
            accelerations, axis=1, dtype=accelerations.dtype
        )

        most_recent_velocity = position_sequence[:, -1] - position_sequence[:, -2]
        most_recent_velocity = torch.unsqueeze(most_recent_velocity, axis=1)
        most_recent_velocities = torch.tile(
            most_recent_velocity, [1, predict_length, 1]
        )
        velocities = most_recent_velocities + velocity_changes

        position_changes = torch.cumsum(velocities, axis=1, dtype=velocities.dtype)

        most_recent_position = position_sequence[:, -1]
        most_recent_position = torch.unsqueeze(most_recent_position, axis=1)
        most_recent_positions = torch.tile(most_recent_position, [1, predict_length, 1])

        new_positions = most_recent_positions + position_changes

        return new_positions

    def _inverse_decoder_postprocessor(self, next_positions, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        most_recent_positions = position_sequence[:, -2:]
        previous_positions = torch.cat(
            [most_recent_positions, next_positions[:, :-1]], axis=1
        )

        positions = torch.cat(
            [torch.unsqueeze(position_sequence[:, -1], axis=1), next_positions], axis=1
        )

        velocities = positions - previous_positions
        accelerations = velocities[:, 1:] - velocities[:, :-1]

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_accelerations = (
            accelerations - acceleration_stats.mean
        ) / acceleration_stats.std

        normalized_accelerations = normalized_accelerations.reshape(
            [-1, self._num_dimensions]
        )

        #  cast from double to float as the input of network
        normalized_accelerations = normalized_accelerations.float()
        return normalized_accelerations

    def inference(
        self,
        position_sequence: Tensor,
        n_particles_per_example,
        n_edges_per_example,
        senders,
        receivers,
        predict_length,
        global_context=None,
        particle_types=None,
    ) -> Tensor:
        """
        Inference with the LearnedSimulator network

        Args:
        position_sequence: Model inference input tensor
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
            i.e. torch.Size([1394, 5, 3])

        n_particles_per_example: torch.Size([1]), [tf.shape(pos)[0]]
            torch.Tensor([node_cnt], dtype=torch.int32)
            i.e. tensor([1394])
        n_edges_per_example: torch.Size([1]), [tf.shape(context['senders'])[0]]
            torch.Tensor([edge_cnt], dtype=torch.int32)
            i.e. tensor([8656])

        senders: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        receivers: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        predict_length: prediction steps, int
            i.e. 1
        particle_types: torch.Tensor([node_cnt], dtype=torch.int32)
            torch.Size([1394])
        global_context: torch.Tensor([sim_step, feat_dim], dtype=torch.float)
            i.e. torch.Size([34, 1])
        """

        input_graph = self.EncodingFeature(
            position_sequence,
            n_particles_per_example,
            n_edges_per_example,
            senders,
            receivers,
            global_context,
            particle_types,
        )

        predicted_normalized_accelerations = self._graph_network(*input_graph)

        next_position = self.DecodingFeature(
            predicted_normalized_accelerations, position_sequence, predict_length
        )

        return next_position

    def forward(
        self,
        next_positions: Tensor,
        position_sequence_noise: Tensor,
        position_sequence: Tensor,
        n_particles_per_example,
        n_edges_per_example,
        senders: Tensor,
        receivers: Tensor,
        predict_length,
        global_context=None,
        particle_types=None,
    ) -> Tensor:
        """
        Training step with the LearnedSimulator network,
        Produces normalized and predicted nodal acceleration.

        Args:
        next_position: Model prediction target tensor
            torch.Tensor([node_cnt, pred_dim] ,)
            i.e. torch.Size([1394, 3])
        position_sequence_noise: Tensor of the same shape as `position_sequence`
            with the noise to apply to each particle.
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
        position_sequence: Model inference input tensor
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
            i.e. torch.Size([1394, 5, 3])

        n_particles_per_example: torch.Size([1]), [tf.shape(pos)[0]]
            i.e. tensor([1394])
        n_edges_per_example: torch.Size([1]), [tf.shape(context['senders'])[0]]
            i.e. tensor([8656])
        senders: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        receivers: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        predict_length: prediction steps, int
            i.e. 1
        particle_types: torch.Tensor([node_cnt], dtype=torch.int32)
            torch.Size([1394])
        global_context: torch.Tensor([sim_step, feat_dim], dtype=torch.float)
            i.e. torch.Size([34, 1])

        Returns:
            Tensors of shape [num_particles_in_batch, num_dimensions] with the
            predicted and target normalized accelerations.
        """

        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        # print("forward global_context: ", global_context.shape)

        input_graph = self.EncodingFeature(
            noisy_position_sequence,
            n_particles_per_example,
            n_edges_per_example,
            senders,
            receivers,
            global_context,
            particle_types,
        )

        predicted_normalized_accelerations = self._graph_network(*input_graph)

        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        most_recent_noise = position_sequence_noise[:, -1]

        most_recent_noise = torch.unsqueeze(most_recent_noise, axis=1)

        most_recent_noises = torch.tile(most_recent_noise, [1, predict_length, 1])

        next_position_adjusted = next_positions + most_recent_noises

        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        # As a result the inverted Euler update in the `_inverse_decoder` produces:
        # * A target acceleration that does not explicitly correct for the noise in
        #   the input positions, as the `next_position_adjusted` is different
        #   from the true `next_position`.
        # * A target acceleration that exactly corrects noise in the input velocity
        #   since the target next velocity calculated by the inverse Euler update
        #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
        #   matches the ground truth next velocity (noise cancels out).
        # print("predicted_normalized_accelerations: ", predicted_normalized_accelerations, predicted_normalized_accelerations.shape)
        # print("target_normalized_acceleration: ", target_normalized_acceleration, target_normalized_acceleration.shape)
        # #for both:  torch.Size([71424, 3])
        return predicted_normalized_accelerations, target_normalized_acceleration

    def get_normalized_acceleration(self, acceleration, predict_length):
        """
        Normalizes the acceleration data using predefined statistics and
        replicates it across a specified prediction length.
        """
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats.mean
        ) / acceleration_stats.std
        normalized_acceleration = torch.tile(normalized_acceleration, [predict_length])
        return normalized_acceleration
