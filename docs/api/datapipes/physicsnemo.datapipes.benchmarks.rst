Benchmark Datapipes
===================

Benchmark datapipes are a unique class of datapipes: they are generating
data on the fly, rather than reading from disk, and are therefore highly portable,
great for testing new applications against known datapipes without worrying
about IO, and overall useful for development.

The Benchmark Datapipes are targeted v1 datapipes for specific datasets.  These
are largely maintained but not actively developed.

.. automodule:: physicsnemo.datapipes.benchmarks.darcy
    :members:
    :show-inheritance:

The Darcy2D provides data loading and preprocessing utilities for 2D Darcy
flow simulations. It handles permeability fields and pressure solutions, supporting
various boundary conditions and mesh resolutions.

.. code-block:: python

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

.. automodule:: physicsnemo.datapipes.benchmarks.kelvin_helmholtz
    :members:
    :show-inheritance:

The KelvinHelmholtz2D manages data for Kelvin-Helmholtz instability simulations,
including velocity fields and density distributions. It supports both 2D and 3D simulation
data with various initial conditions.
