
## Overview

1. **Slicing & Stats**  
   `daily_split_and_stats.py`  
   - Reads the combined `hrrr_mini_train.nc` (with `/`, `/input`, `/output`, `/invariant` groups).  
   - Splits it into one file per day, preserving all four groups.  
   - Computes per-variable means & standard deviations and writes `stats_YYYYMMDD.json`. 
   ```bash
   Execution: python3 hrrr_mini_daily_split_and_stats.py nc_file output_dir
2. **Inspection**  
   `nc_group_inspector.py`  
   - Recursively lists all NetCDF-4 groups, dimensions, variables (and their dims) inside any `.nc` file.  
    ```bash
    Execution: python3 nc_group_inspector.py nc_file
---

## Requirements

- Python 3.8+  
- [xarray](https://xarray.pydata.org)  
- [netCDF4-python](https://unidata.github.io/netcdf4-python/)  
- [pandas](https://pandas.pydata.org)  
- [numpy](https://numpy.org)  
- [tqdm](https://github.com/tqdm/tqdm)  

Install via pip:

```bash
pip install xarray netCDF4 pandas numpy tqdm