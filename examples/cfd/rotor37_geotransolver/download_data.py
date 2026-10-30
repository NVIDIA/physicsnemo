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

"""Download a pinned revision of the PLAID Rotor37 dataset from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "PLAID-datasets/Rotor37"
REVISION = "bac06c0caa7254120eecc6711a5fb85c58dfbdbc"


def main():
    """Download the dataset card and Parquet files into ``--raw-dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rotor37/raw"))
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.raw_dir,
    )
    print(f"Downloaded {DATASET_ID} at revision {args.revision} to {args.raw_dir}")


if __name__ == "__main__":
    main()
