# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
import os
from pathlib import Path

import pytest

from physicsnemo.core import filesystem


def calculate_checksum(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            sha256.update(data)

    calculated_checksum = sha256.hexdigest()
    return calculated_checksum


def test_package(tmp_path: Path):
    string = "hello"
    afile = tmp_path / "a.txt"
    afile.write_text(string)

    path = "file://" + tmp_path.as_posix()
    package = filesystem.Package(path, seperator="/")
    path = package.get("a.txt")
    with open(path) as f:
        ans = f.read()

    assert ans == string


def test_local_package_checksum(tmp_path: Path):
    content = b"local package content"
    file_path = tmp_path / "model.pt"
    file_path.write_bytes(content)
    package = filesystem.Package(str(tmp_path), seperator=os.sep)

    path = package.get("model.pt", checksum=hashlib.sha256(content).hexdigest())
    assert path == str(file_path)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        package.get("model.pt", checksum="0" * 64)


def test_https_package(monkeypatch, tmp_path: Path):
    content = b"test package content"
    known_checksum = hashlib.sha256(content).hexdigest()
    status_checked = False

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            nonlocal status_checked
            status_checked = True

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield content

    monkeypatch.setattr(filesystem.requests, "get", lambda *args, **kwargs: Response())

    path = filesystem._download_cached(
        "https://example.com/assets/model.pt",
        local_cache_path=tmp_path,
        checksum=known_checksum,
    )

    assert status_checked
    assert calculate_checksum(path) == known_checksum


def test_plain_http_package_warns(monkeypatch, tmp_path: Path):
    content = b"test package content"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield content

    monkeypatch.setattr(filesystem.requests, "get", lambda *args, **kwargs: Response())

    with pytest.warns(UserWarning, match="plain HTTP"):
        path = filesystem._download_cached(
            "http://example.com/assets/model.pt",
            local_cache_path=tmp_path,
        )

    assert Path(path).read_bytes() == content


def test_https_package_rejects_bad_checksum(monkeypatch, tmp_path: Path):
    content = b"unexpected content"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield content

    monkeypatch.setattr(filesystem.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="Checksum mismatch"):
        filesystem._download_cached(
            "https://example.com/assets/model.pt",
            local_cache_path=tmp_path,
            checksum="0" * 64,
        )

    assert list(tmp_path.iterdir()) == []

    cached_url = "https://example.com/assets/cached-model.pt"
    cache_path = Path(
        filesystem._download_cached(
            cached_url,
            local_cache_path=tmp_path,
            checksum=hashlib.sha256(content).hexdigest(),
        )
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        filesystem._download_cached(
            cached_url, local_cache_path=tmp_path, checksum="0" * 64
        )

    assert cache_path.read_bytes() == content
    assert [entry.name for entry in tmp_path.iterdir()] == [cache_path.name]


def test_https_package_does_not_cache_failed_response(monkeypatch, tmp_path: Path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            raise filesystem.requests.HTTPError("not found")

    monkeypatch.setattr(filesystem.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(filesystem.requests.HTTPError):
        filesystem._download_cached(
            "https://example.com/assets/missing.pt",
            local_cache_path=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_https_package_rejects_plaintext_redirect(monkeypatch, tmp_path: Path):
    class Response:
        url = "http://example.com/model.pt"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"unexpected content"

    monkeypatch.setattr(filesystem.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="redirected to a non-HTTPS URL"):
        filesystem._download_cached(
            "https://example.com/assets/model.pt",
            local_cache_path=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_ngc_bad_checksum_does_not_evict_valid_cache_entry(monkeypatch, tmp_path: Path):
    """A failed NGC refetch cannot remove an existing valid cache entry."""
    content = b"shared NGC artifact"
    url = "ngc://models/org/model@v1/model.pt"

    def download(path, out_path):
        Path(out_path).write_bytes(content)
        return out_path

    monkeypatch.setattr(filesystem, "_download_ngc_model_file", download)

    cache_path = Path(
        filesystem._download_cached(
            url,
            local_cache_path=tmp_path,
            checksum=hashlib.sha256(content).hexdigest(),
        )
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        filesystem._download_cached(url, local_cache_path=tmp_path, checksum="0" * 64)

    assert cache_path.read_bytes() == content
    assert [entry.name for entry in tmp_path.iterdir()] == [cache_path.name]


@pytest.mark.skip("Skipping because slow, need better test solution")
def test_ngc_model_file():
    test_url = "ngc://models/nvidia/modulus/modulus_dlwp_cubesphere@v0.2"
    package = filesystem.Package(test_url, seperator="/")
    path = package.get("dlwp_cubesphere.zip")

    path = Path(path)
    folders = [f for f in path.iterdir()]
    assert len(folders) == 1 and folders[0].name == "dlwp"

    files = [f for f in folders[0].iterdir()]
    assert len(files) == 11


@pytest.mark.skipif(
    "NGC_API_KEY" not in os.environ, reason="Skipping because no NGC API key"
)
def test_ngc_model_file_private():
    test_url = "ngc://models/nvstaging/simnet/modulus_ci@v0.1"
    package = filesystem.Package(test_url, seperator="/")
    path = package.get("test.txt")

    known_checksum = "d2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26"
    assert calculate_checksum(path) == known_checksum


@pytest.mark.skip("Need no-org file to test")
@pytest.mark.skipif(
    "NGC_API_KEY" not in os.environ, reason="Skipping because no NGC API key"
)
def test_ngc_model_file_private_no_team():
    test_url = ""
    package = filesystem.Package(test_url, seperator="/")
    path = package.get("model/layers.py")

    known_checksum = "177eb43feecf3b4ebdb6cb59e7d445bb5878a26cd9015962b8c9ddd13a648638"
    assert calculate_checksum(path) == known_checksum


def test_ngc_model_file_invalid():
    test_url = "ngc://models/nvidia/modulus/modulus_dlwp_cubesphere/v0.2"
    package = filesystem.Package(test_url, seperator="/")
    with pytest.raises(ValueError):
        package.get("dlwp_cubesphere.zip")

    test_url = "ngc://models/modulus_dlwp_cubesphere@v0.2"
    package = filesystem.Package(test_url, seperator="/")
    with pytest.raises(ValueError):
        package.get("dlwp_cubesphere.zip")

    test_url = "ngc://models/nvidia/modulus/other/modulus_dlwp_cubesphere@v0.2"
    package = filesystem.Package(test_url, seperator="/")
    with pytest.raises(ValueError):
        package.get("dlwp_cubesphere.zip")
