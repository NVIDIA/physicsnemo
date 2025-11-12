import subprocess
import sys
import os

import xarray as xr
try:
    import rioxarray
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rioxarray"])
        print("rioxarray installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install rioxarray. Error: {e}")
        sys.exit(1)

import numpy as np

import glob
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


tiff_folder_path = './tif'

tiff_files = sorted(glob.glob(os.path.join(tiff_folder_path, 'cut_*.tif')))

if not tiff_files:
    print(f"no files under '{tiff_folder_path}'")
else:
    lat_keys = sorted(list(set(re.search(r'(n\d+|s\d+)', f).group(1) for f in tiff_files)), reverse=True)
    lon_keys = sorted(list(set(re.search(r'(e\d+|w\d+)', f).group(1) for f in tiff_files)))

    nested_files = []
    for lat_key in lat_keys:
        row = []
        for lon_key in lon_keys:
            
            matching_file = next((f for f in tiff_files if lat_key in f and lon_key in f), None)
            if matching_file:
                row.append(matching_file)
        if row:
            nested_files.append(row)

    
    for r in nested_files:
        print([os.path.basename(p) for p in r]) 
    
    try:
        merged_dataset = xr.open_mfdataset(
            nested_files,
            engine="rasterio",
            combine='nested',
            concat_dim=['y', 'x'],
            chunks={}
        )

        print("\nSucess")
        print(merged_dataset)

    except Exception as e:
        print(f"\nFail:{e}")


data_subset = merged_dataset['band_data'].squeeze('band', drop=True).rename({'y': 'latitude', 'x': 'longitude'})

print(data_subset)

lon_num_points = int(round((135-80) / 0.05)) + 1
lat_num_points = int(round((55-15) / 0.05)) + 1
target_lon = np.linspace(80, 135, lon_num_points)
target_lat = np.linspace(15, 55, lat_num_points)


interpolated_ds = data_subset.interp(
    latitude=target_lat, 
    longitude=target_lon, 
    method="nearest",
    kwargs={"fill_value": None}
)
interpolated_ds = interpolated_ds.fillna(0)

print('Interpolated data structure')
interpolated_ds = interpolated_ds.to_dataset(name='dem')
print(interpolated_ds)


print("\nTake a view...")

data_to_plot_dask = interpolated_ds['dem']

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

im = ax.pcolormesh(
    data_to_plot_dask['longitude'], 
    data_to_plot_dask['latitude'], 
    data_to_plot_dask.compute(), 
    transform=ccrs.PlateCarree(), 
    cmap='terrain',  
)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAKES)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.colorbar(im, ax=ax, shrink=0.7, label='Elevation (m)')
plt.title('Interpolated Terrain Data (0.05-degree Resolution)')
plt.savefig('dem_005degree.png', dpi=150)
OUTPUT_NC_PATH = 'dem.nc'
print(f"\nStoring to: {OUTPUT_NC_PATH}...")
interpolated_ds.to_netcdf(OUTPUT_NC_PATH)
