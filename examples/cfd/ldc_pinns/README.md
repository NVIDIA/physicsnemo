# Lid Driven Cavity Flow using Purely Physics Driven Neural Networks (PINNs)

This example demonstrates how to set up a purely physics-driven model for solving a Lid
Driven Cavity (LDC) flow using PINNs. The goal of this example is to demonstrate the
interoperability of PhysicsNeMo, PhysicsNeMo-Sym and PyTorch. This example adopts a workflow
where appropriate utilities are imported from `physicsnemo`, `physicsnemo.sym`
and `torch` to define the training pipeline.

Specifically, this example demonstrates how the geometry and physics utilites from
PhysicsNeMo-Sym can be used in custom training pipelines to handle geometry objects
(typically found in Computer Aided Engineering (CAE)) workflows and introduce physics
residual and boundary condition losses.

This example takes a non-abstracted way to define the problem. The
boundary condition constraints, residual constraints, and the subsequent physics loss
computation are defined explicitly. For a more abstracted version of this workflow,
where some of these steps are automated and abstracted, we recommend the
[Introductory example tutorial from PhysicsNeMo-Sym](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/basics/lid_driven_cavity_flow.html).

## Getting Started

### Prerequisites

If you are running this example outside of the PhysicsNeMo container, install
PhysicsNeMo Sym using the instructions from [here](https://github.com/NVIDIA/physicsnemo-sym?tab=readme-ov-file#pypi)

### Training

To train the model, run

```bash
python train.py
```

This should start training the model. Since this is training in a purely Physics based
fashion, there is no dataset required.

Instead, we generate the geometry using the PhysicsNeMo Sym's geometry module and sample
point cloud using `GeometryDatapipe` utility. For more details refer documentation
[here](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.geometry.html#physicsnemo.sym.geometry.geometry_dataloader.GeometryDatapipe)

For computing the physics losses, we will use the `PhysicsInformer` utility from
PhysicsNeMo-Sym. For more details, refer documentation
[here](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.phy_informer.PhysicsInformer)

The results would get saved in the `./outputs/` directory.

## Additional Reading

This example demonstrates computing physics losses on point clouds. For more examples
on physics informing different type of models and model outputs, refer:

* Point clouds: [Darcy Flow (DeepONet)](../darcy_physics_informed/darcy_physics_informed_deeponet.py),
[Stokes Flow (MLP)](../stokes_mgn/pi_fine_tuning.py)
* Regular grid: [Darcy Flow (FNO)](../darcy_physics_informed/darcy_physics_informed_fno.py)
* Unstructured meshes: [Stokes Flow (MeshGraphNet)](../stokes_mgn/pi_fine_tuning_gnn.py)
