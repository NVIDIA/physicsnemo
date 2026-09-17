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

r"""Download a configurable subset of the SuperWing dataset.

Pulls files from the public Hugging Face dataset
`yunplus/SuperWing <https://huggingface.co/datasets/yunplus/SuperWing>`_
using :func:`huggingface_hub.snapshot_download`. The default ``--include``
covers the four files needed by the tutorial recipe; the optional
``data_vol.*`` volumetric shards are not part of the default selection.

Run as a script:

.. code-block:: bash

    python -m src.datapipes.download_superwing \
        --output-dir /path/to/SuperWing_Dataset \
        --include configs.dat data.npy index.npy origingeom.npy geom0.npy
"""

from __future__ import annotations

import argparse


SUPERWING_REPO_ID: str = "yunplus/SuperWing"
SUPERWING_DEFAULT_INCLUDE: list[str] = [
    "configs.dat",
    "data.npy",
    "index.npy",
    "origingeom.npy",
    "geom0.npy",
]


def download_superwing(
    *,
    output_dir: str,
    include: list[str] | None = None,
    repo_id: str = SUPERWING_REPO_ID,
) -> str:
    r"""Download the requested SuperWing files into ``output_dir``.

    Parameters
    ----------
    output_dir : str
        Local directory to populate. Created if missing.
    include : list[str] or None, optional
        Hugging Face ``allow_patterns`` for the snapshot. Defaults to
        :data:`SUPERWING_DEFAULT_INCLUDE`.
    repo_id : str, optional
        Hugging Face dataset repo id. Default ``"yunplus/SuperWing"``.

    Returns
    -------
    str
        The local directory that ``snapshot_download`` populated.
    """
    from huggingface_hub import snapshot_download

    patterns = list(include) if include else list(SUPERWING_DEFAULT_INCLUDE)
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=output_dir,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to download SuperWing files into.",
    )
    p.add_argument(
        "--include",
        nargs="+",
        default=SUPERWING_DEFAULT_INCLUDE,
        help=(
            "Hugging Face allow_patterns for which files to fetch. "
            "Default covers configs.dat, data.npy, index.npy, "
            "origingeom.npy, geom0.npy."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=SUPERWING_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    return p.parse_args()


def main() -> None:
    """Command-line entry point — see module docstring."""
    args = _parse_args()
    print(f"Downloading {args.include} from {args.repo_id} -> {args.output_dir} ...")
    local_dir = download_superwing(
        output_dir=args.output_dir,
        include=args.include,
        repo_id=args.repo_id,
    )
    print(f"Download complete. Files saved to: {local_dir}")


if __name__ == "__main__":
    main()
