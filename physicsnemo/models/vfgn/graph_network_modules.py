# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ReLU
from torch.utils.checkpoint import checkpoint

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.version_check import OptionalImport, require_version_spec

_torch_scatter = OptionalImport("torch_scatter")

STD_EPSILON = 1e-8


@dataclass
class MetaData(ModelMetaData):
    name: str = "VFGNLearnedSimulator"
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
    r"""Configurable multilayer perceptron (MLP).

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    output_size : int
        Number of output features.
    layer_norm : bool, optional, default=True
        Apply layer normalization on the output.

    Forward
    -------
    x : torch.Tensor
        Input features of shape :math:`(N, C_{in})`.

    Returns
    -------
    torch.Tensor
        Output features of shape :math:`(N, C_{out})`.
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
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input tensor (N, C_in) but got shape {tuple(x.shape)}"
                )
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
    r"""Feature encoders for nodes and edges using MLPs.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    latent_size : int
        Latent feature size.

    Forward
    -------
    node_attr : torch.Tensor
        Node attributes of shape :math:`(N_{nodes}, D_{node})`.
    edge_attr : torch.Tensor
        Edge attributes of shape :math:`(N_{edges}, D_{edge})`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Encoded ``(node_attr, edge_attr)`` with last dimension mapped to ``latent_size``.
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
        if not torch.compiler.is_compiling():
            if node_attr.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected 2D tensors for node_attr and edge_attr, got "
                    f"{tuple(node_attr.shape)} and {tuple(edge_attr.shape)}"
                )
        # encode node attributes
        node_attr = self.node_mlp(node_attr)
        # encode edge attributes
        edge_attr = self.edge_mlp(edge_attr)

        return node_attr, edge_attr


class EdgeBlock(Module):
    r"""Edge update block aggregating sender/receiver node features.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    latent_size : int
        Latent feature size.
    node_dim : int, optional, default=0
        Feature dimension corresponding to nodes.
    use_receiver_nodes : bool, optional, default=True
        Include receiver node features in edge update.
    use_sender_nodes : bool, optional, default=True
        Include sender node features in edge update.

    Forward
    -------
    node_attr : torch.Tensor
        Node attributes :math:`(N_{nodes}, D)`.
    edge_attr : torch.Tensor
        Edge attributes :math:`(N_{edges}, D)`.
    receivers : torch.Tensor
        Receiver indices :math:`(N_{edges},)`.
    senders : torch.Tensor
        Sender indices :math:`(N_{edges},)`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(node_attr, updated_edge_attr, receivers, senders)``.
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
        if not torch.compiler.is_compiling():
            if node_attr.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected node_attr and edge_attr tensors to be 2D, got "
                    f"{tuple(node_attr.shape)} and {tuple(edge_attr.shape)}"
                )
            if receivers.ndim != 1 or senders.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )
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
    r"""Node update block aggregating incident edge features.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    latent_size : int
        Latent feature size.
    aggr : Literal["add", "sum", "mean", "min", "max", "mul"], optional, default="add"
        Aggregation operation for edges per node.
    node_dim : int, optional, default=0
        Feature dimension corresponding to nodes.
    use_received_edges : bool, optional, default=True
        Include received edges in aggregation.
    use_sent_edges : bool, optional, default=False
        Include sent edges in aggregation.

    Forward
    -------
    x : torch.Tensor
        Node features :math:`(N_{nodes}, D)`.
    edge_attr : torch.Tensor
        Edge features :math:`(N_{edges}, D)`.
    receivers : torch.Tensor
        Receiver indices :math:`(N_{edges},)`.
    senders : torch.Tensor
        Sender indices :math:`(N_{edges},)`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(updated_nodes, edge_attr, receivers, senders)``.
    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr: Literal["add", "sum", "mean", "min", "max", "mul"] = "add",
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

    @require_version_spec("torch_scatter")
    def forward(self, x, edge_attr, receivers, senders):
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected x and edge_attr as 2D tensors, got "
                    f"{tuple(x.shape)} and {tuple(edge_attr.shape)}"
                )
            if receivers.ndim != 1 or senders.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )
        nodes_to_collect = []
        nodes_to_collect.append(x)

        dim_size = x.shape[self.node_dim]

        scatter_fn = _torch_scatter.scatter

        # aggregate received edges
        if self.use_received_edges:
            receivers_edge = scatter_fn(
                dim=self.node_dim,
                dim_size=dim_size,
                index=receivers,
                src=edge_attr,
                reduce=self.aggr,
            )
            nodes_to_collect.append(receivers_edge)

        # aggregate sent edges
        if self.use_sent_edges:
            senders_edge = scatter_fn(
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
    r"""Alternating edge and node updates (one interaction step).

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    latent_size : int
        Latent feature size.
    aggr : Literal["add", "sum", "mean", "min", "max", "mul"], optional, default="add"
        Aggregation operation for edges per node.

    Forward
    -------
    x : torch.Tensor
        Node features :math:`(N_{nodes}, D)`.
    edge_attr : torch.Tensor
        Edge features :math:`(N_{edges}, D)`.
    receivers : torch.Tensor
        Receiver node indices :math:`(N_{edges},)`.
    senders : torch.Tensor
        Sender node indices :math:`(N_{edges},)`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Updated ``(x, edge_attr, receivers, senders)``.
    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr: Literal["add", "sum", "mean", "min", "max", "mul"] = "add",
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
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected x and edge_attr as 2D tensors, got "
                    f"{tuple(x.shape)} and {tuple(edge_attr.shape)}"
                )
            if receivers.ndim != 1 or senders.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )
        if not (x.shape[-1] == edge_attr.shape[-1]):
            raise ValueError("node feature size should equal to edge feature size")

        return self._node_block(*self._edge_block(x, edge_attr, receivers, senders))


class ResInteractionNet(torch.nn.Module):
    r"""Residual interaction block that wraps :class:`InteractionNet`.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    latent_size : int
        Latent feature size.
    aggr : Literal["add", "sum", "mean", "min", "max", "mul"], optional, default="add"
        Aggregation operation for edges per node.
    node_dim : int, optional, default=0
        Feature dimension corresponding to nodes.

    Forward
    -------
    x : torch.Tensor
        Node features :math:`(N_{nodes}, D)`.
    edge_attr : torch.Tensor
        Edge features :math:`(N_{edges}, D)`.
    receivers : torch.Tensor
        Receiver indices :math:`(N_{edges},)`.
    senders : torch.Tensor
        Sender indices :math:`(N_{edges},)`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x_new, edge_attr_new, receivers, senders)`` after residual update.
    """

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr: Literal["add", "sum", "mean", "min", "max", "mul"] = "add",
        node_dim=0,
    ):
        super(ResInteractionNet, self).__init__()
        self.itn = InteractionNet(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )

    def forward(self, x, edge_attr, receivers, senders):
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected x and edge_attr as 2D tensors, got "
                    f"{tuple(x.shape)} and {tuple(edge_attr.shape)}"
                )
            if receivers.ndim != 1 or senders.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )
        x_res, edge_attr_res, receivers, senders = self.itn(
            x, edge_attr, receivers, senders
        )

        x_new = x + x_res
        edge_attr_new = edge_attr + edge_attr_res

        return x_new, edge_attr_new, receivers, senders


class DecoderNet(Module):
    r"""MLP-based decoder for node features.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    output_size : int
        Number of output features.

    Forward
    -------
    x : torch.Tensor
        Node features :math:`(N_{nodes}, D_{in})`.

    Returns
    -------
    torch.Tensor
        Decoded node features :math:`(N_{nodes}, D_{out})`.
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
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input tensor (N, D) but got shape {tuple(x.shape)}"
                )
        # number of layer is important, or the network will overfit
        x = self.mlp(x)
        return x


class EncodeProcessDecode(Module):
    r"""Encoder → Processor → Decoder architecture.

    Parameters
    ----------
    latent_size : int
        Latent feature size.
    mlp_hidden_size : int
        Number of hidden features.
    mlp_num_hidden_layers : int
        Number of hidden layers.
    num_message_passing_steps : int
        Number of interaction steps in the processor.
    output_size : int
        Number of output features.
    device_list : list[str], optional
        Devices to execute message passing.

    Forward
    -------
    x : torch.Tensor
        Node features :math:`(N_{nodes}, D_{in})`.
    edge_attr : torch.Tensor
        Edge features :math:`(N_{edges}, D_{in})`.
    receivers : torch.Tensor
        Receiver indices :math:`(N_{edges},)`.
    senders : torch.Tensor
        Sender indices :math:`(N_{edges},)`.

    Returns
    -------
    torch.Tensor
        Decoded node features :math:`(N_{nodes}, D_{out})`.
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
        """Set devices for message passing execution."""
        self.device_list = device_list

    def forward(self, x, edge_attr, receivers, senders):
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or edge_attr.ndim != 2:
                raise ValueError(
                    f"Expected x and edge_attr as 2D tensors, got "
                    f"{tuple(x.shape)} and {tuple(edge_attr.shape)}"
                )
            if receivers.ndim != 1 or senders.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )
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


class VFGNLearnedSimulator(Module):
    r"""Simulator model using graph-based encode-process-decode.

    Parameters
    ----------
    num_dimensions : int, optional, default=3
        Output dimensions per node (e.g., coordinates).
    num_seq : int, optional, default=5
        Number of input time steps in the sequence.
    boundaries : list[list[float]] or None, optional, default=None
        Bounding box used for normalization ``[[x_min, x_max], ...]``.
    num_particle_types : int, optional, default=3
        Number of particle types.
    particle_type_embedding_size : int, optional, default=16
        Embedding size for particle types.
    normalization_stats : dict or None, optional, default=None
        Dict with keys ``'acceleration'``, ``'velocity'``, ``'context'`` each
        containing objects with ``mean`` and ``std`` tensors.
    graph_mode : str, optional, default="radius"
        Connectivity construction mode.
    connectivity_param : float, optional, default=0.015
        Normalization distance for displacements.

    Forward
    -------
    next_positions : torch.Tensor
        Target next positions :math:`(N_{nodes}, D)`.
    position_sequence_noise : torch.Tensor
        Noise to apply to input positions :math:`(N_{nodes}, T, D)`.
    position_sequence : torch.Tensor
        Input position sequence :math:`(N_{nodes}, T, D)`.
    n_particles_per_example : torch.Tensor
        Number of particles per graph :math:`(B,)`.
    n_edges_per_example : torch.Tensor
        Number of edges per graph :math:`(B,)`.
    senders : torch.Tensor
        Sender indices :math:`(N_{edges},)`.
    receivers : torch.Tensor
        Receiver indices :math:`(N_{edges},)`.
    predict_length : int
        Number of future steps to predict.
    global_context : torch.Tensor or None, optional
        Global context per example.
    particle_types : torch.Tensor or None, optional
        Per-node particle type indices.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(predicted_normalized_accelerations, target_normalized_acceleration)``.
    """

    def __init__(
        self,
        num_dimensions: int = 3,
        num_seq: int = 5,
        boundaries: list[list[float]] = None,
        num_particle_types: int = 3,
        particle_type_embedding_size: int = 16,
        normalization_stats: map = None,
        graph_mode: Literal["radius", "knn"] = "radius",
        connectivity_param: float = 0.015,
    ):
        if not (num_dimensions >= 0 and num_seq >= 3):
            raise ValueError("Invalid arch params - VFGNLearnedSimulator")
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
        """Set devices to be used for message passing in the model."""
        self.message_passing_devices = devices

    def to(self, device):
        """Transfer model and relevant buffers to ``device``."""
        new_self = super(VFGNLearnedSimulator, self).to(device)
        new_self._boundaries = self._boundaries.to(device)
        for key in self._normalization_stats:
            new_self._normalization_stats[key].to(device)
        if device != "cpu":
            self._graph_network.set_device(self.message_passing_devices)
        return new_self

    def time_diff(self, input_seq):
        r"""Discrete time derivative along the sequence axis.

        Parameters
        ----------
        input_seq : torch.Tensor
            Sequence tensor :math:`(N, T, D)`.

        Returns
        -------
        torch.Tensor
            Differences ``input_seq[:, 1:] - input_seq[:, :-1]`` of shape :math:`(N, T-1, D)`.
        """
        return input_seq[:, 1:] - input_seq[:, :-1]

    def _compute_connectivity_for_batch(
        self, senders_list, receivers_list, n_node, n_edge
    ):
        r"""Compute connectivity indices with optional random edge dropout per-graph."""
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
        r"""Generate random-walk velocity noise and integrate to position noise."""

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
        ).normal_(mean=0, std=noise_std_last_step / num_velocities**0.5)  # float

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
        r"""Build encoded node and edge features.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(x, edge_attr, senders, receivers)`` ready for processor.
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
        r"""Decode normalized accelerations back to predicted positions."""
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
        r"""Inverse of the decoder postprocessor (computes normalized accelerations)."""
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
        r"""Inference with the simulator.

        Parameters
        ----------
        position_sequence : torch.Tensor
            Input position sequence :math:`(N_{nodes}, T, D)`.
        n_particles_per_example : torch.Tensor
            Number of nodes per graph :math:`(B,)`.
        n_edges_per_example : torch.Tensor
            Number of edges per graph :math:`(B,)`.
        senders : torch.Tensor
            Sender indices :math:`(N_{edges},)`.
        receivers : torch.Tensor
            Receiver indices :math:`(N_{edges},)`.
        predict_length : int
            Number of future steps to predict.
        global_context : torch.Tensor, optional
            Global context per example.
        particle_types : torch.Tensor, optional
            Per-node particle type indices.

        Returns
        -------
        torch.Tensor
            Predicted positions :math:`(N_{nodes}, \mathrm{predict\_length}, D)`.
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
        if not torch.compiler.is_compiling():
            if position_sequence.ndim != 3 or position_sequence_noise.ndim != 3:
                raise ValueError(
                    "Expected position_sequence and position_sequence_noise to be 3D "
                    f"(N_nodes, T, D), got {tuple(position_sequence.shape)} and "
                    f"{tuple(position_sequence_noise.shape)}"
                )
            if next_positions.ndim != 2:
                raise ValueError(
                    f"Expected next_positions to be 2D (N_nodes, D), got {tuple(next_positions.shape)}"
                )
            if senders.ndim != 1 or receivers.ndim != 1:
                raise ValueError(
                    f"Expected 1D index tensors for receivers/senders, got "
                    f"{tuple(receivers.shape)} and {tuple(senders.shape)}"
                )

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
        r"""Normalize acceleration and tile across ``predict_length``."""
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats.mean
        ) / acceleration_stats.std
        normalized_acceleration = torch.tile(normalized_acceleration, [predict_length])
        return normalized_acceleration
