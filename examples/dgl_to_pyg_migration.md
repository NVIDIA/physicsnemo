# Switching from DGL to PyTorch Geometric

<!-- markdownlint-disable MD013-->

Graph Neural Networks (GNNs) are a popular and widely used type of model with
extensive support in PhysicsNeMo.
PhysicsNeMo currently supports two popular GNN backends: [DGL](https://www.dgl.ai/)
and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (PyG).

In the past, PhysicsNeMo supported only DGL as the GNN backend. PyG has been
added in the `25.08` (August) release and is the recommended backend for all
existing and new GNN-based models. PyG is an active open-source project that
supports a broad range of features and enables certain performance optimizations
that can improve GNN model performance up to 30% compared to similar DGL
implementation.

**Note**: In the future PhysicsNeMo release, support for the DGL backend will be removed.

## Switching to PyG Backend

Starting with the `25.11` (November) release of PhysicsNeMo, all examples and
models that use the DGL backend will switch to using the PyG backend by default.

### How it Works

The backend selection is done automatically by the PhysicsNeMo GNN implementation
based on the type of the input graph:

- DGL backend is used when the input graph is of `dgl.DGLGraph` type.
- PyG backend is used when the input graph is of `torch_geometric.data.Data` type.

In this scenario, you retain full control over how the source graph is created.
This approach is backward compatible by default: if no changes are made to your
code, PhysicsNeMo will use the DGL backend.

To change the backend to PyG, create a graph using the PyG API.

### Models and Checkpoints

In most cases, existing checkpoints created during training a model with the DGL
backend should work with the PyG backend without any changes or retraining.
The key is to ensure that the input data to the model is the same for both DGL
and PyG data loaders.

Some possible exceptions:

- The output of the DGL data loader is not the same as PyG.
- Other components, besides the data loader and the model, such as data augmentation,
exist in the DGL version but not in PyG.

### Data and Dataset Loading Code

Existing data does not need to be modified unless it was created and stored using
the DGL API, such as `dgl.graph()` and `dgl.save_graphs()`. PyG does not support
this format, so you need to convert the data to a format supported by the
PyTorch `torch.load()` method.

Dataset loading and processing code may need to be modified to use and return PyG
graph objects. Compare the `VortexSheddingDataset` DGL and PyG implementations
located in `physicsnemo/datapipes/gnn/vortex_shedding_dataset_dgl.py`
and `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py`, respectively.

### PhysicsNeMo Examples

Existing DGL-based examples have been copied to examples with the same name
and a `_dgl` suffix. For example, `examples/cfd/vortex_shedding_mgn` has been
renamed to `examples/cfd/vortex_shedding_mgn_dgl`. A new example, `vortex_shedding_mgn`,
has been created based on the previous one, but now it uses the PyG implementation.
In most examples, the changes are minimal and easy to compare.

**Note**: In a future PhysicsNeMo release, `_dgl` examples will be removed.

### How to Switch to PyG

To switch to PyG:

1. Update your dataset and dataloader code to use the PyG API instead of DGL.
2. See one of the examples, such as `examples/cfd/vortex_shedding_mgn`, for
implementation details.
3. Compare the example with its `_dgl` version to understand the necessary changes.

The changes are usually relatively straightforward.

## Comparison of DGL and PyG API

A short comparison of the DGL and PyG APIs can help in migrating code from DGL to PyG.
The DGL API may seem a bit more high-level, though PyG often provides more flexibility.

Arguably, one of the most important operations is graph construction. The code snippet
below provides a comparison of DGL and PyG APIs.

```py
import torch
import dgl
from torch_geometric.data import Data

# Node indices that define a simple, 3-node, 2-edge directed graph:
src = [0, 1]
dst = [1, 2]
node_features = torch.tensor([[0.], [1.], [2.]])

# DGL:
graph_dgl = dgl.graph((src, dst))
graph_dgl.ndata["x"] = node_features

# PyG:
edge_index = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)
graph_pyg = Data(x=node_features, edge_index=edge_index)

# Alternative approach:
# graph_pyg = Data(edge_index=edge_index)
# graph_pyg.x = node_features

print(graph_dgl)
print(graph_pyg)

```

The following table shows other popular operations:

| DGL | PyG | Notes |
|-----|-----|-------|
| `dgl.save_graphs()` | `torch.save()` | Save graph to disk |
| `dgl.load_graphs()` | `torch.load()` | Load graph from disk |
| `dgl.to_bidirected()` | `torch_geometric.utils.to_undirected()` | Convert to bidirectional graph |
| `dgl.add_self_loop()` | `torch_geometric.utils.add_self_loops()` | Add self-loops to graph |
| `dgl.remove_self_loop()` | `torch_geometric.utils.remove_self_loops()` | Remove self-loops |
| `dgl.to_simple()` | `torch_geometric.utils.coalesce()` | Remove duplicate edges |
| `dgl.metis_partition()` | METIS: `loader.ClusterData` Halo: `utils.k_hop_subgraph` | Graph partitioning |
| `dgl.heterograph()` | `torch_geometric.data.HeteroData` | Create heterogeneous graph |
| `dgl.DGLDataset` | `torch_geometric.data.Dataset` or `torch.utils.data.Dataset` | Base dataset class |
| `dgl.dataloading.GraphDataLoader` | `torch_geometric.loader.DataLoader` | Data loading |
| `dgl.add_edges` | Re-create `edge_index`, no in-place option | Add edges |

See [DGL](https://www.dgl.ai/dgl_docs/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html) documentation for more information.

**Note**: for saving and loading graphs operations, respective DGL and PyG versions produce data in different formats. That is, data written using DGL `dgl.save_graphs()` cannot be read using PyG `torch.load()`.
