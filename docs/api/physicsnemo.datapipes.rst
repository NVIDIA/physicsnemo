PhysicsNeMo Datapipes
======================

.. automodule:: physicsnemo.datapipes
.. currentmodule:: physicsnemo.datapipes

PhysicsNeMo Datapipes provides a collection of data loading and processing utilities designed
to handle various types of physics-based datasets. The datapipes are organized into several
categories to support different types of physics simulations and machine learning tasks.

The datapipes in PhysicsNeMo are built on top of PhysicsNeMo's DataPipe class and provide
specialized implementations for handling physics simulation data, climate data,
graph-based data, and mesh-based data. Each category of datapipes is designed to efficiently
load and preprocess specific types of physics datasets.

The example below shows how to use a typical datapipe in PhysicsNeMo:

.. code:: python

    import torch
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    def main():
        # Create a datapipe for Darcy flow simulation data
        datapipe = Darcy2D(
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Iterate through the datapipe
        for batch in datapipe:
            # batch contains input features and target values
            input_features = batch["permeability"]
            target_values = batch["darcy"]

            # Use the data for training or inference
            ...

    if __name__ == "__main__":
        main()

Here's another example showing how to use the ERA5HDF5Datapipe for weather data processing:

.. code:: python

    import torch
    from physicsnemo.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe

    def main():
        # Create a datapipe for ERA5 weather data in HDF5 format
        datapipe = ERA5HDF5Datapipe(
            data_dir="path/to/era5/data",
            stats_dir="path/to/era5/stats",
            channels=[0, 1],
            latlon_resolution=(721, 1440),
            shuffle=True,
        )

        # Iterate through the datapipe
        for batch in datapipe:
            invar = batch[0]["invar"]
            outvar = batch[0]["outvar"]

            # Use the data for weather prediction or analysis
            ...

    if __name__ == "__main__":
        main()

Available Datapipes
"""""""""""""""""""

PhysicsNeMo provides several categories of datapipes:

1. Benchmark Datapipes
   - Designed for standard physics benchmark problems
   - Include implementations for Darcy flow and Kelvin-Helmholtz instability

2. Weather and Climate Datapipes
   - Specialized for handling climate and weather data
   - Support ERA5 HDF5 format and synthetic climate data
   - Include utilities for HEALPix time series data

3. Graph Datapipes
   - Handle graph-based physics data
   - Support vortex shedding, Ahmed body, DrivaerNet, and Stokes flow datasets
   - Include utility functions for graph data processing

4. CAE (Computer-Aided Engineering) Datapipes
   - Specialized for mesh-based data
   - Support various mesh formats and configurations

Each category of datapipes is designed to handle specific data formats and preprocessing
requirements. The datapipes automatically handle data loading, preprocessing,
and device placement, making it easy to integrate them into training or inference pipelines.

.. autosummary::
   :toctree: generated

Benchmark datapipes
-------------------

.. automodule:: physicsnemo.datapipes.benchmarks.darcy
    :members:
    :show-inheritance:

The Darcy2D provides data loading and preprocessing utilities for 2D Darcy
flow simulations. It handles permeability fields and pressure solutions, supporting
various boundary conditions and mesh resolutions.

.. automodule:: physicsnemo.datapipes.benchmarks.kelvin_helmholtz
    :members:
    :show-inheritance:

The KelvinHelmholtz2D manages data for Kelvin-Helmholtz instability simulations,
including velocity fields and density distributions. It supports both 2D and 3D simulation
data with various initial conditions.

Weather and climate datapipes
-----------------------------

.. automodule:: physicsnemo.datapipes.climate.era5_hdf5
    :members:
    :show-inheritance:

The ERA5HDF5Datapipe handles ERA5 reanalysis data stored in HDF5 format, providing access to
atmospheric variables like temperature, pressure, and wind fields at various pressure levels.

.. automodule:: physicsnemo.datapipes.climate.climate
    :members:
    :show-inheritance:

The ClimateDataPipe provides a general interface for climate data processing, supporting
various climate datasets and variables with standardized preprocessing and normalization.

.. automodule:: physicsnemo.datapipes.climate.synthetic
    :members:
    :show-inheritance:

The SyntheticWeatherDataset generates synthetic climate data for testing and development
purposes, supporting various climate patterns and noise models.

.. automodule:: physicsnemo.datapipes.healpix.timeseries_dataset
    :members:
    :show-inheritance:

The TimeSeriesDataset handles spherical harmonic data in HEALPix format,
supporting time series analysis of global climate variables.

Graph datapipes
---------------

.. automodule:: physicsnemo.datapipes.gnn.vortex_shedding_dataset
    :members:
    :show-inheritance:

The VortexSheddingDataset processes flow field data around bluff bodies,
capturing vortex shedding patterns and flow structures for graph-based learning.

.. automodule:: physicsnemo.datapipes.gnn.ahmed_body_dataset
    :members:
    :show-inheritance:

The AhmedBodyDataset manages flow field data around Ahmed bodies, supporting aerodynamic
analysis and drag prediction tasks.

.. automodule:: physicsnemo.datapipes.gnn.drivaernet_dataset
    :members:
    :show-inheritance:

The DrivaerNetDataset handles automotive aerodynamics data, providing access to flow field
measurements and surface pressure distributions.

.. automodule:: physicsnemo.datapipes.gnn.stokes_dataset
    :members:
    :show-inheritance:

The StokesDataset processes Stokes flow simulations, supporting various boundary conditions
and geometry configurations for microfluidic applications.

.. automodule:: physicsnemo.datapipes.gnn.utils
    :members:
    :show-inheritance:

The GNN utilities provide helper functions for graph construction, feature extraction,
and data preprocessing in graph-based physics learning tasks.

CAE datapipes
-------------

.. automodule:: physicsnemo.datapipes.cae.mesh_datapipe
    :members:
    :show-inheritance:

The MeshDataPipe handles mesh data for physics simulations,
supporting various mesh formats and providing utilities for mesh preprocessing
and feature extraction.

.. automodule:: physicsnemo.datapipes.cae.domino_datapipe
    :members:
    :show-inheritance:
