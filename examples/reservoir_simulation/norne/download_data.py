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

"""Download the Norne example's simulation data from a GitHub release.

The bulk reservoir data (grids, decks, summaries, results) is published as a
release asset rather than committed to the repository. Run this once before
using the example:

    python download_data.py

Setup (one-time, by the data maintainer):
  1. Create a release on the repo and attach ``norne_data_backup.zip`` to it.
  2. Copy the asset's download URL; it looks like:
     https://github.com/<owner>/<repo>/releases/download/<tag>/norne_data_backup.zip
  3. Paste that URL into RELEASE_URL below.
"""

import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("norne-data")

# --- set this to your release asset URL --------------------------------------
RELEASE_URL = "https://github.com/clementetienam/physicsnemo/releases/download/norne-data-v1/norne_data_backup.zip"
# -----------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
ZIP_PATH = EXAMPLE_DIR / "norne_data_backup.zip"
ALREADY_PRESENT = EXAMPLE_DIR / "simulator_data"


def main() -> None:
    """Download and unpack the Norne data unless it is already present."""
    if ALREADY_PRESENT.exists():
        logger.info("Data already present (%s). Nothing to do.", ALREADY_PRESENT.name)
        return

    if "PASTE_YOUR" in RELEASE_URL:
        logger.error("Set RELEASE_URL at the top of this script first.")
        sys.exit(1)

    logger.info("Downloading data ...")
    req = urllib.request.Request(RELEASE_URL, headers={"User-Agent": "norne-example"})
    try:
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)
        logger.error("Check that RELEASE_URL is correct and the repo is public.")
        sys.exit(1)

    logger.info("Extracting into %s ...", EXAMPLE_DIR)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXAMPLE_DIR)
    ZIP_PATH.unlink()
    logger.info("Done. Data restored under simulator_data/ and RESULTS/.")


if __name__ == "__main__":
    main()