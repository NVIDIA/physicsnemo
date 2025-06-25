#!/usr/bin/env python3
import os
import argparse
import json

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

def trim_and_stats(nc_file, output_dir):
    # 1) Load the root (time & coord) and grab all dim defs
    ds_root = xr.open_dataset(nc_file, engine='netcdf4', chunks=None)
    times   = pd.to_datetime(ds_root["time"].values)

    # 2) Load each group and re-attach the same time index where needed
    ds_input  = xr.open_dataset(nc_file, group="input",     engine='netcdf4', chunks=None).assign_coords(time=ds_root["time"])
    ds_output = xr.open_dataset(nc_file, group="output",    engine='netcdf4', chunks=None).assign_coords(time=ds_root["time"])
    ds_inv    = xr.open_dataset(nc_file, group="invariant", engine='netcdf4', chunks=None)

    os.makedirs(output_dir, exist_ok=True)

    # 3) Loop over each calendar day
    dates       = times.date
    unique_days = np.unique(dates)

    for day in tqdm(unique_days, desc="Processing days"):
        day_str = pd.to_datetime(str(day)).strftime("%Y%m%d")
        idx     = np.where(dates == day)[0]
        out_nc  = os.path.join(output_dir, f"{day_str}.nc")

        # 4) Slice each dataset
        root_day   = ds_root.isel(sample=idx)
        input_day  = ds_input.isel(sample=idx)
        output_day = ds_output.isel(sample=idx)
        inv_day    = ds_inv  # static, no sample dim

        # 5) Write root group fresh (mode='w')
        root_day.to_netcdf(out_nc, mode='w', format='NETCDF4')

        # 6) Append each subgroup
        input_day.to_netcdf(out_nc, mode='a', group='input')
        output_day.to_netcdf(out_nc, mode='a', group='output')
        inv_day.to_netcdf(out_nc, mode='a', group='invariant')

        # 7) Compute and dump stats JSON if you still need it
        def stats_of(da):
            a = da.astype(float)
            return {"mean": float(a.mean().values), "std": float(a.std().values)}

        stats = {"input": {}, "output": {}, "invariant": {}}
        for v in input_day.data_vars:  stats["input"][v]    = stats_of(input_day[v])
        for v in output_day.data_vars: stats["output"][v]   = stats_of(output_day[v])
        for v in inv_day.data_vars:    stats["invariant"][v] = stats_of(inv_day[v])
        for c in inv_day.coords:       stats["invariant"][c] = stats_of(inv_day.coords[c])

        with open(os.path.join(output_dir, f"stats_{day_str}.json"), "w") as f:
            json.dump(stats, f, indent=4)

def main():
    p = argparse.ArgumentParser(description="Split CorrDiff .nc into daily .nc (all groups) + stats")
    p.add_argument("nc_file",    help="Path to original hrrr_mini_train.nc")
    p.add_argument("output_dir", help="Where to write daily files + stats")
    args = p.parse_args()

    print(f"→ Processing {args.nc_file} → {args.output_dir}")
    trim_and_stats(args.nc_file, args.output_dir)
    print("✅ Done!")

if __name__ == "__main__":
    main()