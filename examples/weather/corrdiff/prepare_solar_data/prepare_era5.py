

import netCDF4 as nc
import numpy as np
import xarray as xr
import os
from datetime import datetime
from tqdm import tqdm


def rename_variable_on_load(ds, filename, filename_to_varname_map):
    
    base_filename = os.path.basename(filename)
    
    target_var_name = filename_to_varname_map.get(base_filename)

    if 'time' in ds.coords and 'valid_time' not in ds.coords:
        ds = ds.rename({'time': 'valid_time'})
    if 'pressure_level' in ds.dims and ds.dims['pressure_level'] == 1:
        ds = ds.squeeze('pressure_level', drop=True)
    
    if target_var_name:
        original_var_name = list(ds.data_vars)[0]
        if original_var_name != target_var_name:
            return ds.rename({original_var_name: target_var_name})
    return ds


def main(year):
    path = "path/to/ERA5/data/{}".format(year)

    filename_to_varname_map = {
        f'{year}_2m_temperature.nc': 't2m',
        f'{year}_surface_pressure.nc': 'sp',
        f'{year}_ssrd.nc': 'ssrd',
        f'{year}_total_column_water_vapour.nc': 'tcwv'
    }
    
    for level in [1000, 925, 500, 300, 100, 50]:
        filename_to_varname_map[f'{year}_q_{level}.nc'] = f'q{level}'
        filename_to_varname_map[f'{year}_t_{level}.nc'] = f't{level}'
        filename_to_varname_map[f'{year}_z_{level}.nc'] = f'z{level}'

    input_files = [os.path.join(path, fname) for fname in filename_to_varname_map.keys()]
    
    print(input_files)

    from functools import partial
    preprocess_func = partial(rename_variable_on_load, filename_to_varname_map=filename_to_varname_map)

    ds = xr.open_mfdataset(
        input_files, 
        preprocess=lambda ds: preprocess_func(ds, ds.encoding["source"]),
        combine='by_coords'
    )
    
    
    lat_bounds = [15, 55]
    lon_bounds = [80, 135]
    ds_cropped = ds.sel(
        latitude=slice(max(lat_bounds), min(lat_bounds)),
        longitude=slice(min(lon_bounds), max(lon_bounds))
    )
    ds_sorted = ds_cropped.sortby('latitude')

    order = [
        "sp", "t2m", "tcwv", "ssrd", "q1000", "q925", "q500", "q300", "q100", "q50",
        "t1000", "t925", "t500", "t300", "t100", "t50",
        "z1000", "z925", "z500", "z300", "z100", "z50"
    ]
    ds_sorted = ds_sorted[order]

    print(ds['valid_time'].values[0],ds['valid_time'].values[-1])
    output_filename = "output/path/era5_{}_opt.zarr".format(year)
    ds_final = ds_sorted.chunk({
        'valid_time': 1,
        'latitude': "auto", 
        'longitude': "auto",
    })
    print(ds_final)
    print(ds_final.data_vars)
    ds_final = ds_final.fillna(0)
    ds_final.to_zarr(output_filename, mode='w', consolidated=True)
    

if __name__ == "__main__":
    years = [2016,2017,2018,2019,2020]
    for year in years:
        main(year)