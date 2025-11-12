import xarray as xr
import pandas as pd
import os
from pathlib import Path

def process_year_data(base_path, year):
    """
    Args:
        base_path (str or Path): the path to data
        year (int)
    """
    base_path = Path(base_path)
    output_dir = base_path / f"H08_{year}_hourly"
    output_dir.mkdir(exist_ok=True)

    #creat the times range, with 10mins interval
    time_range = pd.date_range(start=f'{year}-01-01 00:00', end=f'{year}-12-31 23:50', freq='10min')

    for group_name, group_df in time_range.to_frame().resample('H'):
        if len(group_df) == 6:  # Make sure there are 6 files in an hour
            file_paths = []
            all_files_exist = True
            for dt in group_df.index:
                #your original SWDR nc files
                file_path = base_path / str(dt.year) / f"{dt.year}{dt.month:02d}" / f"{dt.day:02d}" / f"H08_{dt.strftime('%Y%m%d_%H%M')}_SWDR.nc"

                if file_path.exists():
                    file_paths.append(file_path)
                else:
                    print(f"Files Missing: {group_name.strftime('%Y-%m-%d %H')}")
                    all_files_exist = False
                    break

            if all_files_exist:
                try:
                    datasets = [xr.open_dataset(fp) for fp in file_paths]
                    
                    lat_slice = slice(55, 15)
                    lon_slice = slice(80, 135)
                    
                    datasets = [ds.sel(lat=lat_slice, lon=lon_slice) for ds in datasets]

                    is_night_data = False
                    for ds in datasets:
                        if ds['SWDR'].max() == 0:
                            print(f"This is night (SWDR全为0). Skip:{group_name.strftime('%Y-%m-%d %H')}")
                            is_night_data = True
                            break
                    
                    if not is_night_data:
                        valid_times = [pd.to_datetime(ds.encoding['source'].split('_')[1] + ds.encoding['source'].split('_')[2], format='%Y%m%d%H%M') for ds in datasets]

                        combined_ds = xr.concat(
                            [ds['SWDR'] for ds in datasets],
                            dim=pd.Index(valid_times, name='valid_time')
                        )
                    
                        combined_ds = combined_ds.fillna(0)
                    
                        combined_ds = combined_ds.astype('float32')

                        combined_ds = combined_ds.sortby('lat')
                        combined_ds = combined_ds.sortby('valid_time')
                    
                        combined_ds = combined_ds.rename({'lat': 'latitude', 'lon': 'longitude'})

                        output_filename = output_dir / f"H08_{group_name.strftime('%Y%m%d_%H%M')}_hourly.nc"
                    
                        combined_ds.to_netcdf(output_filename)
                        print(f"Saved H08_{group_name.strftime('%Y%m%d_%H%M')}_hourly")
                except Exception as e:
                    print(f"Errors happen at {group_name.strftime('%Y-%m-%d %H')}: {e}")


data_directory = "path/to/original/data" 

year_to_process = [2016,2017,2018,2019]
for year in year_to_process:
    process_year_data(data_directory, year)