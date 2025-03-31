# Training from Object Storage using Multi-Storage Client

## What is Multi-Storage Client (MSC)?

[Multi-Storage Client](https://github.com/NVIDIA/multi-storage-client) is a Python
library that provides a unified interface for accessing various object stores and
file systems. It makes it easy for ML workloads to use object stores by providing
a familiar file-like interface without sacrificing performance. The library adds
new functionality, such as caching, client-side observability, and leverages the native
SDKs specific to each object store for optimal performance.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

Or install different extra dependencies based on your object storage backend:

```bash
# POSIX file systems.
pip install multi-storage-client

# NVIDIA AIStore.
pip install "multi-storage-client[aistore]"

# Azure Blob Storage.
pip install "multi-storage-client[azure-storage-blob]"

# AWS S3 and S3-compatible object stores.
pip install "multi-storage-client[boto3]"

# Google Cloud Storage (GCS).
pip install "multi-storage-client[google-cloud-storage]"

# Oracle Cloud Infrastructure (OCI) Object Storage.
pip install "multi-storage-client[oci]"
```

### Configuration File

The MSC configuration file defines profiles which include storage provider configurations.
An example MSC configuration file can be found at [msc_config.yaml](./msc_config.yaml).
In this example, we're pointing to the [CMIP6 archive on AWS](https://registry.opendata.aws/cmip6/).

## Usage Example

MSC supports fsspec and integrates with frameworks such as Zarr and Xarray via
the fsspec interface. The following example demonstrates how to use Zarr to
access the CMIP6 dataset stored in AWS S3:

```bash
export MSC_CONFIG=./msc_config.yaml
python
>>> import zarr
>>> zarr_group = zarr.open("msc://cmip6-pds/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp119/r1i1p1f1/day/tas/gr1/v20180701")
>>> zarr_group.tree()
/
 ├── bnds (2,) float64
 ├── height () float64
 ├── lat (180,) float64
 ├── lat_bnds (180, 2) float64
 ├── lon (288,) float64
 ├── lon_bnds (288, 2) float64
 ├── tas (31390, 180, 288) float32
 ├── time (31390,) int64
 └── time_bnds (31390, 2) float64
```

## Update Existing Code Path with MSC

For other PhysicsNeMo’s examples, where Zarr is commonly used in training workflows,
migrating to MSC is a straightforward process involving only configuration changes.
For example, in the [Corrdiff](../generative/corrdiff/) training example, data
currently accessed from the file system can be updated to MSC by modifying the
input path from `/code/2023-01-24-cwb-4years.zarr` to `msc://cwb-diffusions/2023-01-24-cwb-4years.zarr`,
assuming the data stored in local has been moved to a S3 bucket `cwb-diffusions`,
and MSC has a profile `cwb-diffusions` pointing to this S3 bucket.

### Current code path (Training from File System)

```bash
input_path = "/code/2023-01-24-cwb-4years.zarr"
zarr.open_consolidated(input_path)
```

### Updated code path (Training from Object Store using MSC)

```bash
input_path = "msc://cwb-diffusions/2023-01-24-cwb-4years.zarr"
zarr.open_consolidated(input_path)
```

## Additional Information

- [Multi-Storage Client Documentation](https://nvidia.github.io/multi-storage-client/)
