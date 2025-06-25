#!/usr/bin/env python3
import argparse
from netCDF4 import Dataset

def walk_group(grp, prefix="", indent=0):
    """
    Recursively print group contents:
      • Group path
      • Dimensions in that group (name = size)
      • Variables in that group (with their dims)
      • Recurse into subgroups
    """
    spacer = "    " * indent
    path = prefix or "/"
    print(f"{spacer}Group: {path}")

    # 1) Dimensions
    if grp.dimensions:
        print(f"{spacer}  Dimensions:")
        for dim_name, dim in grp.dimensions.items():
            size = dim.size if not dim.isunlimited() else "UNLIMITED"
            print(f"{spacer}    - {dim_name} = {size}")
    else:
        print(f"{spacer}  (no dimensions)")

    # 2) Variables
    if grp.variables:
        print(f"{spacer}  Variables:")
        for name, var in grp.variables.items():
            dims = ", ".join(var.dimensions)
            print(f"{spacer}    - {name}  (dims: {dims})")
    else:
        print(f"{spacer}  (no variables)")

    # 3) Recurse into subgroups
    for sub_name, sub_grp in grp.groups.items():
        sub_path = f"{path}/{sub_name}" if path != "/" else f"/{sub_name}"
        walk_group(sub_grp, prefix=sub_path, indent=indent+1)

def list_nc_tree_with_dims(nc_file):
    """
    Open a NetCDF-4 file and print every group along with its
    dimensions and variables in a tree view.
    """
    ds = Dataset(nc_file, mode='r')
    print("Contents of", nc_file)
    walk_group(ds)  # start at root group
    ds.close()

def main():
    parser = argparse.ArgumentParser(
        description="Recursively list all groups, dimensions, and variables in a NetCDF-4 file"
    )
    parser.add_argument("nc_file", help="Path to NetCDF-4 file (.nc)")
    args = parser.parse_args()

    list_nc_tree_with_dims(args.nc_file)

if __name__ == "__main__":
    main()