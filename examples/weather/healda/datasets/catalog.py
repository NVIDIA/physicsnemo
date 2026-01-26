# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import urllib.parse
from dataclasses import dataclass

import xarray
import zarr
import zarr.storage
from config import environment
from utils import storage


@dataclass
class _Zarr:
    path: str
    profile: str

    @property
    def storage_options(self):
        return storage.get_storage_options(self.profile)

    def to_store(self, obstore=True) -> zarr.storage.StoreLike:
        if self.profile == "":
            return self.path

        url = urllib.parse.urlparse(self.path)
        if obstore:
            bucket = url.netloc
            store = storage.get_obstore(self.profile, bucket=bucket, prefix=url.path)
            zarr_store = zarr.storage.ObjectStore(store)
        else:
            import fsspec

            fs = fsspec.filesystem(
                url.scheme, storage_options=self.storage_options, asyn=True
            )
            zarr_store = zarr.storage.FsspecStore(fs)

        return zarr_store

    def to_zarr(self, obstore=True, use_consolidated=True) -> zarr.Group:
        store = self.to_store(obstore=True)
        return zarr.open_group(store, use_consolidated=use_consolidated)

    def to_xarray(self, obstore: bool = True, **kwargs) -> xarray.Dataset:
        return xarray.open_zarr(
            self.to_store(obstore=obstore),
            **kwargs,
        )

    def consolidate_metadata(self):
        store = zarr.storage.FsspecStore.from_url(
            self.path, storage_options=storage.get_storage_options(self.profile)
        )
        zarr.consolidate_metadata(store)


@dataclass
class _Parquet:
    path: str
    profile: str

    @property
    def storage_options(self):
        return storage.get_storage_options(self.profile)

    @property
    def polars_storage_options(self):
        return storage.get_polars_storage_options(self.profile)

    def files(self):
        import fsspec

        fs = fsspec.filesystem("s3", **self.storage_options)
        return ["s3://" + f for f in fs.glob(self.path)]

    def to_pandas(self, year, month, day):
        import pandas as pd

        path = f"{self.path}/{year:04d}{month:02d}{day:02d}.parquet"
        return pd.read_parquet(path, storage_options=self.storage_options)

    def to_polars(self):
        import polars

        return polars.scan_parquet(
            self.path + "/*.parquet",
            storage_options=storage.get_polars_storage_options(self.profile),
        )


def era5_hpx6():
    return _Zarr(environment.V6_ERA5_ZARR, environment.V6_ERA5_ZARR_PROFILE)


def ufs_obs_parquet():
    return _Parquet(environment.UFS_OBS_PATH + "amsua", "pbss")


def ufs_obs():
    return _Parquet(environment.UFS_OBS_PATH + "amsua", "pbss")


def ufs():
    return _Zarr(path=environment.UFS_HPX6_ZARR, profile=environment.UFS_ZARR_PROFILE)
