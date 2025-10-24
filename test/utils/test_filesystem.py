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
import os
from pathlib import Path

import pytest

from physicsnemo.utils import filesystem


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


def test_http_package():
    test_url = "http://raw.githubusercontent.com/NVIDIA/modulus/main/docs/img"
    package = filesystem.Package(test_url, seperator="/")
    path = package.get("modulus-pipes.jpg")

    known_checksum = "e075b2836d03f7971f754354807dcdca51a7875c8297cb161557946736d1f7fc"
    assert calculate_checksum(path) == known_checksum


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
