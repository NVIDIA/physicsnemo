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

r"""Download and extract the parts of DeepJEB and SimJEB that this example reads.

- DeepJEB (Hugging Face ``KAIST-SmartDesignLab/DeepJEB``, ODC-By 1.0): the solver decks, the field
  tables and the split metadata, 77 GB.
- SimJEB v4.0 (Harvard Dataverse ``doi:10.7910/DVN/XFUWJG``, CC0 1.0): the volume meshes, the
  field tables and the solver decks, 7.1 GB.

Every archive is checked against its published checksum before extraction into
``<output>/<dataset>/extracted``.

    python download_data.py --output <data> [--datasets deepjeb simjeb]
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DEEPJEB = {
    "repo_id": "KAIST-SmartDesignLab/DeepJEB",
    "revision": "160424ca2a5095315ff9ead8a5fc36d2667fd3c6",
    "files": {
        "Metadata.tar.gz": "08d594565aed01183df8de7693e4f6699f17af0c16e4990cfa7c29565bb5ed73",
        "SolverDeck.tar.gz": "ecec03c8867dba5b8643489c8f8d2c3639ec03259c32b680a1fff321aeab4087",
        "Field.tar.gz": "570a9d6db968fd37a27d82820e07dae17fd06455bb9c88a7075fb7a70d1d8b7f",
    },
}
SIMJEB = {
    "url": "https://dataverse.harvard.edu/api/access/datafile/{id}",
    "files": {
        "SimJEB_volmesh_(vtk).zip": (4640734, "1e2f9dde53053ebdc2d5d478fc1eb8d7"),
        "SimJEB_simresults_(csv)_firsthalf.zip": (
            4640686,
            "6208cd1dc0a24face00c7999615f71c4",
        ),
        "SimJEB_simresults_(csv)_secondhalf.zip": (
            4640716,
            "9f843ed84843eb5014549dd5e84d0156",
        ),
        "SimJEB_fea_(fem)_firsthalf.zip": (4640545, "bcbdb8e781fb1e879c45ff195150d310"),
        "SimJEB_fea_(fem)_secondhalf.zip": (
            4640683,
            "d9f8fe8b736fad7f58a23ab62c6e91d6",
        ),
    },
}


def digest(path: Path, algorithm: str) -> str:
    r"""Hex digest of a file."""
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def deepjeb(root: Path) -> None:
    r"""Download and extract the DeepJEB files that the example uses."""
    from huggingface_hub import hf_hub_download

    for name, sha256 in DEEPJEB["files"].items():
        path = Path(
            hf_hub_download(
                DEEPJEB["repo_id"],
                name,
                repo_type="dataset",
                revision=DEEPJEB["revision"],
                local_dir=root / "raw",
            )
        )
        if digest(path, "sha256") != sha256:
            raise SystemExit(f"Checksum mismatch: {path}")
        print(f"extracting {name}", flush=True)
        with tarfile.open(path) as archive:
            archive.extractall(root / "extracted", filter="data")  # noqa: S202 - the "data" filter rejects unsafe members


def simjeb(root: Path) -> None:
    r"""Download and extract the SimJEB files that the example uses."""
    (root / "raw").mkdir(parents=True, exist_ok=True)
    for name, (file_id, md5) in SIMJEB["files"].items():
        path = root / "raw" / name
        if not path.exists() or digest(path, "md5") != md5:
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(SIMJEB["url"].format(id=file_id), path)  # noqa: S310 - fixed https URL
        if digest(path, "md5") != md5:
            raise SystemExit(f"Checksum mismatch: {path}")
        print(f"extracting {name}", flush=True)
        with zipfile.ZipFile(path) as archive:
            if any(
                Path(n).is_absolute() or ".." in Path(n).parts
                for n in archive.namelist()
            ):
                raise SystemExit(f"Unsafe member path in {path}")
            archive.extractall(root / "extracted" / path.stem)  # noqa: S202 - member paths checked above


def main() -> None:
    r"""Download and extract both datasets (command line)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["deepjeb", "simjeb"],
        choices=["deepjeb", "simjeb"],
    )
    args = parser.parse_args()
    for name in args.datasets:
        {"deepjeb": deepjeb, "simjeb": simjeb}[name](args.output / name)


if __name__ == "__main__":
    main()
