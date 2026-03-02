Weather and Climate Datapipes
=============================

The ERA5HDF5Datapipe handles ERA5 reanalysis data stored in HDF5 format, providing access to
atmospheric variables like temperature, pressure, and wind fields at various pressure levels.
The ERA5HDF5Datapipe is used in the DLWP Example.

.. code-block:: python

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

.. automodule:: physicsnemo.datapipes.climate.era5_hdf5
    :members:
    :show-inheritance:

The ClimateDataPipe provides a general interface for climate data processing, supporting
various climate datasets and variables with standardized preprocessing and normalization.

The ClimateDataPipe is used in the Weather Diagnostic example and the Temporal
Interpolation example.

.. automodule:: physicsnemo.datapipes.climate.climate
    :members:
    :show-inheritance:


The SyntheticWeatherDataset generates synthetic climate data for testing and development
purposes, supporting various climate patterns and noise models.

.. automodule:: physicsnemo.datapipes.climate.synthetic
    :members:
    :show-inheritance:

The TimeSeriesDataset handles spherical harmonic data in HEALPix format,
supporting time series analysis of global climate variables.

.. automodule:: physicsnemo.datapipes.healpix.timeseries_dataset
    :members:
    :show-inheritance:


