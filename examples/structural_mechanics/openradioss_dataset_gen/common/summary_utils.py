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

"""Shared dataset-summary helpers (JSON + CSV writers)."""

from __future__ import annotations

import csv
import json
import os
from typing import Callable


def save_summaries(
    dataset_dir: str,
    all_runs_data: list,
    flatten_row: Callable[[dict], dict],
) -> None:
    """Write `summary.json` (full hierarchical log) and `summary.csv`
    (flat, ML-loader-friendly) for a dataset directory.

    `flatten_row` receives one metadata entry and returns the CSV row dict.
    """
    json_path = os.path.join(dataset_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(all_runs_data, f, indent=4)
    print(f"Summary JSON saved to: {json_path}")

    if not all_runs_data:
        return

    csv_path = os.path.join(dataset_dir, "summary.csv")
    flattened = [flatten_row(r) for r in all_runs_data]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
        writer.writeheader()
        writer.writerows(flattened)
    print(f"Summary CSV saved to: {csv_path}")
