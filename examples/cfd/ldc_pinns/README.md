# Lid Driven Cavity Flow using Purely Physics Driven Neural Networks (PINNs)

This example demonstrates how to set up a purely physics-driven model for solving a Lid
Driven Cavity (LDC) flow using PINNs. The goal of this example is to demonstrate the
interoperability of PhysicsNeMo, `physicsnemo.sym` and PyTorch. This example adopts a workflow
where appropriate utilities are imported from `physicsnemo`, `physicsnemo.sym`
and `torch` to define the training pipeline.

Specifically, this example demonstrates how the geometry and physics utilities from
`physicsnemo.sym` can be used in custom training pipelines to handle geometry objects
(typically found in Computer Aided Engineering (CAE)) workflows and introduce physics
residual and boundary condition losses.

This example takes a non-abstracted way to define the problem. The
boundary condition constraints, residual constraints, and the subsequent physics loss
computation are defined explicitly. If you previously used the (now archived)
[`physicsnemo-sym`](https://github.com/NVIDIA/physicsnemo-sym) repository,
where the `Solver` / `Domain` / `Constraint` abstractions handled these steps
implicitly, see the
[PhysicsNeMo v2.0 Migration Guide](../../../v2.0-MIGRATION-GUIDE.md#physicsnemo-sym--physicsnemosym)
for how the equivalent pieces look in this newer, explicit style.

## Getting Started

### Prerequisites

If you are running this example outside of the PhysicsNeMo container, install
PhysicsNeMo with the sym extra: `pip install "nvidia-physicsnemo[sym]"`

### Training

To train the model, run

```bash
python train.py
```

This should start training the model. Since this is training in a purely Physics based
fashion, there is no dataset required.

Instead, we generate the geometry using the `physicsnemo.mesh` module and sample
the point cloud using the `GeometryDatapipe` utility. The
[PhysicsNeMo v2.0 Migration Guide](../../../v2.0-MIGRATION-GUIDE.md#physicsnemo-sym--physicsnemosym)
shows how this maps from the older `physicsnemo-sym` geometry primitives.

For computing the physics losses, we will use the `PhysicsInformer` utility from
`physicsnemo.sym`. The
[PhysicsNeMo v2.0 Migration Guide](../../../v2.0-MIGRATION-GUIDE.md#physicsnemo-sym--physicsnemosym)
shows how this maps from the older `physicsnemo-sym` PDE / `make_nodes` workflow.

The results would get saved in the `./outputs/` directory.

## Additional Reading

This example demonstrates computing physics losses on point clouds. For more examples
on physics informing different type of models and model outputs, refer to:

* Point clouds: [Darcy Flow (DeepONet)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/darcy_physics_informed/README.html),
[Stokes Flow (MLP)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/stokes_mgn/README.html)
* Regular grid: [Darcy Flow (FNO)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/darcy_physics_informed/README.html)
* Unstructured meshes: [Stokes Flow (MeshGraphNet)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/stokes_mgn/README.html)
