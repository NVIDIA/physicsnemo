CAE Datapipes
=============

The CAE Datapipes are v1 datapipes for specific datasets for external
aerodynamics datasets.  These are maintained but not under active development
in every case.  

The MeshDataPipe uses VTK to read CFD mesh data and simulations, and DALI
for data loading and preprocessing.  The MeshDataPipe is used in the DataCenter
example.

.. automodule:: physicsnemo.datapipes.cae.mesh_datapipe
    :members:
    :show-inheritance:

The DoMINO DataPipe reads the DrivearML dataset, and other datasets, for 
the DoMINO model for external aerodynamics.  The expected format of inputs can
be achieved using PhysicsNeMo-Curator.

.. automodule:: physicsnemo.datapipes.cae.domino_datapipe
    :members:
    :show-inheritance:

The Transolver DataPipe reads the same inputs as the DoMINO DataPipe, but
produces outputs for the Transolver and GeoTransolver models for external
aerodynamics.

.. automodule:: physicsnemo.datapipes.cae.transolver_datapipe
    :members:
    :show-inheritance:
