# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
import configparser
import os
import shutil
import sys
import tempfile

import fsspec

DEFAULT_PATH = os.path.expanduser("~/.config/rclone/rclone.conf")


class StorageConfigError(Exception):
    """Exception raised when storage configuration is invalid or missing."""

    pass


def get_remote_config(remote_name, config_path=DEFAULT_PATH):
    """Parse rclone config and return the section for the given remote."""
    if not remote_name:
        return None
    # Parse the rclone config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure the remote exists in the config
    if remote_name not in config:
        raise StorageConfigError(f"Remote '{remote_name}' not found in rclone config.")

    # Extract credentials from the config
    remote_config = config[remote_name]
    return remote_config


def get_storage_options(remote_name, config_path=DEFAULT_PATH):
    """Return S3 storage options dict for fsspec from rclone remote config."""
    remote_config = get_remote_config(remote_name, config_path)

    if remote_config is None:
        return None

    if remote_config.get("type") != "s3":
        raise StorageConfigError(f"Remote '{remote_name}' is not an S3 remote.")

    access_key = remote_config.get("access_key_id")
    secret_key = remote_config.get("secret_access_key")
    endpoint_url = remote_config.get("endpoint", None)  # Optional endpoint

    if not access_key or not secret_key:
        raise StorageConfigError(
            f"Access key or secret key missing for remote '{remote_name}'."
        )

    # Instantiate and return the S3FileSystem object
    return dict(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else None,
    )


def get_polars_storage_options(profile):
    """Return S3 storage options dict for Polars from rclone remote config."""
    opts = get_storage_options(profile)
    key = opts["key"]
    secret = opts["secret"]
    endpoint = opts["client_kwargs"]["endpoint_url"]
    return {
        "aws_access_key_id": key,
        "aws_secret_access_key": secret,
        "aws_endpoint_url": endpoint,
    }


def get_duckdb_connection(profile):
    """Return a DuckDB connection configured with S3 credentials from rclone."""
    import duckdb

    opts = get_storage_options(profile)
    con = duckdb.connect()
    key = opts["key"]
    secret = opts["secret"]
    endpoint = opts["client_kwargs"]["endpoint_url"]
    if endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]
    con.execute(f"""
    CREATE SECRET (
        TYPE s3,
        PROVIDER config,
        ENDPOINT '{endpoint}',
        KEY_ID '{key}',
        SECRET '{secret}'
    );
    """)

    return con


def ensure_downloaded(url, local):
    if os.path.exists(local):
        return

    fs = fsspec.filesystem("http")
    print(f"Downloading from {url} to {local}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as d:
        tmpfile = os.path.join(d, "file")
        fs.get(url, tmpfile)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.move(tmpfile, local)


def _get_endpoint(opts):
    return opts["client_kwargs"]["endpoint_url"]


def get_pyarrow_filesystem(profile: str, **kwargs):
    """Return a PyArrow S3FileSystem configured from rclone remote profile."""
    import pyarrow.fs

    opts = get_storage_options(profile)
    if opts is None:
        return None

    return pyarrow.fs.S3FileSystem(
        access_key=opts.get("key"),
        secret_key=opts.get("secret"),
        region=opts.get("region", ""),
        endpoint_override=_get_endpoint(opts),
        **kwargs,
    )


def get_obstore(profile: str, bucket=None, **kwargs):
    """Return an obstore S3Store configured from rclone remote profile."""
    from obstore.store import S3Store

    opts = get_storage_options(profile)
    if opts is None:
        return None

    return S3Store(
        bucket=bucket,
        access_key_id=opts.get("key"),
        secret_access_key=opts.get("secret"),
        endpoint=_get_endpoint(opts),
        **kwargs,
    )
