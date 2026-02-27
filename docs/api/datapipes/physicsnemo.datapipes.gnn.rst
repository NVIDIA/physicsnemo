Graph Neural Network Datapipes
==============================

The VortexSheddingDataset processes flow field data around bluff bodies,
capturing vortex shedding patterns and flow structures for graph-based learning.
The VortexSheddingDataset is used in the VortexShedding CFD examples.

.. automodule:: physicsnemo.datapipes.gnn.vortex_shedding_dataset
    :members:
    :show-inheritance:

The AhmedBodyDataset manages flow field data around Ahmed bodies, supporting aerodynamic
analysis and drag prediction tasks.  The AhmedBodyDataset is used in the 
AeroGraphNet CFD External Aerodynamics example.

.. automodule:: physicsnemo.datapipes.gnn.ahmed_body_dataset
    :members:
    :show-inheritance:

The DrivAerNetDataset handles automotive aerodynamics surface data, providing access to
surface pressure and wall shear stress distributions.  The DrivAerNetDataset is used in the
AeroGraphNet and FIGConvNet CFD External Aerodynamics examples.

.. automodule:: physicsnemo.datapipes.gnn.drivaernet_dataset
    :members:
    :show-inheritance:

The StokesDataset processes Stokes flow simulations in pipe domains obstructed by random
polygons, supporting various boundary conditions and geometry configurations.  The
StokesDataset is used in the Stokes MeshGraphNet CFD example.

.. automodule:: physicsnemo.datapipes.gnn.stokes_dataset
    :members:
    :show-inheritance:

The GNN utilities provide helper functions for reading VTP files and saving/loading
JSON-serialized statistics used by the GNN datapipes.  The GNN utilities are used by
the GNN dataset classes and in the structural mechanics examples.

.. automodule:: physicsnemo.datapipes.gnn.utils
    :members:
    :show-inheritance:
