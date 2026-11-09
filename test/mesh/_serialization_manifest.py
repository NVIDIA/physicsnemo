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

"""Compact snapshots of ``tensordict`` memmap directory layouts."""

import json
from pathlib import Path

# tensordict 0.15.3 (pytorch/tensordict#3114) writes these format markers into each
# meta.json. They describe tensordict's file format, not the saved layout, so the
# snapshot leaves them out and stays valid across tensordict releases.
_TENSORDICT_FORMAT_KEYS = ("_format_version", "_class")


def serialization_manifest(root: Path) -> dict[str, object]:
    """Describe file structure and metadata without copying tensor bytes."""
    described: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name == "meta.json":
            meta = json.loads(path.read_text())
            for key in _TENSORDICT_FORMAT_KEYS:
                meta.pop(key, None)
            described[relative] = meta
        else:
            described[relative] = path.stat().st_size
    return described
