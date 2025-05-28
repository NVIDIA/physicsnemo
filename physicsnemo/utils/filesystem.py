# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import logging
import os
import re
import urllib.request
import zipfile

import fsspec
import fsspec.implementations.cached
import fsspec.utils
import requests
import s3fs
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    LOCAL_CACHE = os.environ["LOCAL_CACHE"]
except KeyError:
    LOCAL_CACHE = os.environ["HOME"] + "/.cache/physicsnemo"


def _cache_fs(fs):
    return fsspec.implementations.cached.CachingFileSystem(
        fs=fs, cache_storage=LOCAL_CACHE
    )


def _get_fs(path):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url="https://pbss.s8k.io"))
    else:
        return fsspec.filesystem(fsspec.utils.get_protocol(path))


def _download_ngc_model_file(path: str, out_path: str, timeout: int = 300) -> str:
    """Pulls files from model registry on NGC. Supports private registries when NGC
    API key is set the the `NGC_API_KEY` environment variable. If download file is a zip
    folder it will get unzipped.

    Args:
        path (str): NGC model file path of form:
            `ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>`
            or if no team
            `ngc://models/<org_id/model_id>@<version>/<path/in/repo>`
        out_path (str): Output path to save file / folder as
        timeout (int): Time out of requests, default 5 minutes

    Raises:
        ValueError: Invlaid url

    Returns:
        str: output file / folder path
    """
    # Strip ngc model url prefix
    suffix = "ngc://models/"
    # The regex check
    pattern = re.compile(f"{suffix}[\w-]+(/[\w-]+)?/[\w-]+@[A-Za-z0-9.]+/[\w/](.*)")
    if not pattern.match(path):
        raise ValueError(
            "Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>"
        )

    path = path.replace(suffix, "")
    if len(path.split("@")[0].split("/")) == 3:
        (org, team, model_version, filename) = path.split("/", 3)
        (model, version) = model_version.split("@", 1)
    else:
        (org, model_version, filename) = path.split("/", 2)
        (model, version) = model_version.split("@", 1)
        team = None

    token = ""
    # If API key environment variable
    if "NGC_API_KEY" in os.environ:
        try:
            # SSA tokens
            if os.environ["NGC_API_KEY"].startswith("nvapi-"):
                raise NotImplementedError("New personal keys not supported yet")
            # Legacy tokens
            # https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#download-models-via-wget-authenticated-access
            else:
                session = requests.Session()
                session.auth = ("$oauthtoken", os.environ["NGC_API_KEY"])
                headers = {"Accept": "application/json"}
                authn_url = f"https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{org}&group/ngc:{org}/{team}"
                r = session.get(authn_url, headers=headers, timeout=5)
                r.raise_for_status()
                token = json.loads(r.content)["token"]
        except requests.exceptions.RequestException:
            logger.warning(
                "Failed to get JWT using the API set in NGC_API_KEY environment variable"
            )
            raise  # Re-raise the exception

    # Download file, apparently the URL for private registries is different than the public?
    if len(token) > 0:
        # Sloppy but works
        if team:
            file_url = f"https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model}/versions/{version}/files/{filename}"
        else:
            file_url = f"https://api.ngc.nvidia.com/v2/org/{org}/models/{model}/versions/{version}/files/{filename}"
    else:
        if team:
            file_url = f"https://api.ngc.nvidia.com/v2/models/{org}/{team}/{model}/versions/{version}/files/{filename}"
        else:
            file_url = f"https://api.ngc.nvidia.com/v2/models/{org}/{model}/versions/{version}/files/{filename}"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # Streaming here for larger files
    with requests.get(file_url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        chunk_size = 1024  # 1 kb
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        progress_bar.set_description(f"Fetching {filename}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    # Unzip contents if zip file (most model files are)
    if zipfile.is_zipfile(out_path) and path.endswith(".zip"):
        temp_path = out_path + ".zip"
        os.rename(out_path, temp_path)
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(out_path)
        # Clean up zip
        os.remove(temp_path)

    return out_path


def _download_cached(
    path: str, recursive: bool = False, local_cache_path: str = LOCAL_CACHE
) -> str:
    sha = hashlib.sha256(path.encode())
    filename = sha.hexdigest()
    try:
        os.makedirs(local_cache_path, exist_ok=True)
    except PermissionError as error:
        logger.error(
            "Failed to create cache folder, check permissions or set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error
    except OSError as error:
        logger.error(
            "Failed to create cache folder, set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error

    cache_path = os.path.join(local_cache_path, filename)

    url = urllib.parse.urlparse(path)

    # TODO watch for race condition here
    if not os.path.exists(cache_path):
        logger.debug("Downloading %s to cache: %s", path, cache_path)
        if url.scheme in ("s3", "msc"):
            fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
            fs.get(path, cache_path, recursive=recursive)
        elif path.startswith("ngc://models/"):
            path = _download_ngc_model_file(path, cache_path)
            return path
        elif url.scheme == "http":
            # urllib.request.urlretrieve(path, cache_path)
            # TODO: Check if this supports directory fetches
            response = requests.get(path, stream=True, timeout=5)
            with open(cache_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
        elif url.scheme == "file":
            path = os.path.join(url.netloc, url.path)
            return path
        else:
            return path

    else:
        logger.debug("Opening from cache: %s", cache_path)

    return cache_path


class Package:
    """A generic file system abstraction. Can be used to represent local and remote
    file systems. Remote files are automatically fetched and stored in the
    $LOCAL_CACHE or $HOME/.cache/physicsnemo folder. The `get` method can then be used
    to access files present.

    Presently one can use Package with the following directories:
    - Package("/path/to/local/directory") = local file system
    - Package("s3://bucket/path/to/directory") = object store file system
    - Package("http://url/path/to/directory") = http file system
    - Package("ngc://model/<org_id/team_id/model_id>@<version>") = ngc model file system

    Args:
        root (str): Root directory for file system
        seperator (str, optional): directory seperator. Defaults to "/".
    """

    def __init__(self, root: str, seperator: str = "/"):
        self.root = root
        self.seperator = seperator

    def get(self, path: str, recursive: bool = False) -> str:
        """Get a local path to the item at ``path``

        ``path`` might be a remote file, in which case it is downloaded to a
        local cache at $LOCAL_CACHE or $HOME/.cache/physicsnemo first.
        """
        return _download_cached(self._fullpath(path), recursive=recursive)

    def _fullpath(self, path):
        return self.root + self.seperator + path
