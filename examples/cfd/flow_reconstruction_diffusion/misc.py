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


"""Miscellaneous utility classes and functions."""

import glob
import hashlib
import html
import tempfile
import urllib
import urllib.request
import uuid
from typing import Any, List, Tuple, Union
import requests


# Cache directories
# -------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:  # pragma: no cover
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:  # pragma: no cover
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if "DNNLIB_CACHE_DIR" in os.environ:
        return os.path.join(os.environ["DNNLIB_CACHE_DIR"], *paths)
    if "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".cache", "dnnlib", *paths)
    if "USERPROFILE" in os.environ:
        return os.path.join(os.environ["USERPROFILE"], ".cache", "dnnlib", *paths)
    return os.path.join(tempfile.gettempdir(), ".cache", "dnnlib", *paths)


# URL helpers
# ------------------------------------------------------------------------------------------


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:  # pragma: no cover
    """
    Determine whether the given object is a valid URL string.
    """
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith("file://"):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(
    url: str,
    cache_dir: str = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
) -> Any:  # pragma: no cover
    """
    Download the given URL and return a binary-mode file object to access the data.
    This code handles unusual file:// patterns that
    arise on Windows:

    file:///c:/foo.txt

    which would translate to a local '/c:/foo.txt' filename that's
    invalid.  Drop the forward slash for such pathnames.

    If you touch this code path, you should test it on both Linux and
    Windows.

    Some internet resources suggest using urllib.request.url2pathname() but
    but that converts forward slashes to backslashes and this causes
    its own set of problems.
    """
    if not num_attempts >= 1:
        raise ValueError("num_attempts must be at least 1")
    if return_filename and (not cache):
        raise ValueError("return_filename requires cache=True")

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    if not is_url(url):
        raise IOError("Not a URL: " + url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path("downloads")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[: min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(
            cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name
        )
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    if return_filename:
        raise ValueError("return_filename requires cache=True")
    return io.BytesIO(url_data)
