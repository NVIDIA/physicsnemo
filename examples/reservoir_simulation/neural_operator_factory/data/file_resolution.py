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

"""Resolve input/output file paths for reservoir simulation datasets."""

from pathlib import Path
from typing import Optional, Tuple


def resolve_data_files(
    data_path: Path,
    mode: str,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    variable: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Resolve input and output file paths with flexible naming.

    Supports three resolution strategies in priority order:

    1. **Explicit files** — ``input_file`` and ``output_file`` are both
       provided (may contain a ``{mode}`` placeholder).
    2. **Variable-based** — ``variable`` maps to CO2 naming convention
       (``dP_*`` for pressure, ``sg_*`` for saturation).
    3. **Auto-detect** — scans the directory for common patterns.

    Parameters
    ----------
    data_path : Path
        Root data directory.
    mode : str
        Dataset split: ``'train'``, ``'val'``, or ``'test'``.
    input_file : str, optional
        Input filename or pattern with ``{mode}`` placeholder.
    output_file : str, optional
        Output filename or pattern with ``{mode}`` placeholder.
    variable : str, optional
        ``'pressure'`` or ``'saturation'`` for CO2 convention.

    Returns
    -------
    Tuple[Path, Path]
        ``(input_path, output_path)``

    Raises
    ------
    FileNotFoundError
        If auto-detection fails (no known pattern matches).
    ValueError
        If *variable* is not a recognised name.
    """
    if input_file is not None and output_file is not None:
        return (
            data_path / input_file.format(mode=mode),
            data_path / output_file.format(mode=mode),
        )

    if variable is not None:
        var_map = {"pressure": "dP", "saturation": "sg", "dP": "dP", "sg": "sg"}
        if variable.lower() not in var_map:
            raise ValueError(
                f"Variable must be 'pressure' or 'saturation', got {variable}"
            )
        prefix = var_map[variable.lower()]
        return (
            data_path / f"{prefix}_{mode}_a.pt",
            data_path / f"{prefix}_{mode}_u.pt",
        )

    return _auto_detect_files(data_path, mode)


def _auto_detect_files(data_path: Path, mode: str) -> Tuple[Path, Path]:
    """Scan *data_path* for known filename patterns."""
    patterns = [
        (f"{mode}_input.pt", f"{mode}_output.pt"),
        (f"input_{mode}.pt", f"output_{mode}.pt"),
        (f"{mode}_x.pt", f"{mode}_y.pt"),
        (f"x_{mode}.pt", f"y_{mode}.pt"),
        (f"dP_{mode}_a.pt", f"dP_{mode}_u.pt"),
        (f"sg_{mode}_a.pt", f"sg_{mode}_u.pt"),
    ]
    for inp_name, out_name in patterns:
        inp_path = data_path / inp_name
        out_path = data_path / out_name
        if inp_path.exists() and out_path.exists():
            return inp_path, out_path

    pt_files = sorted(f.name for f in data_path.glob("*.pt"))
    raise FileNotFoundError(
        f"Could not auto-detect data files in {data_path}\n"
        f"Available .pt files: {pt_files}\n"
        f"Please specify input_file and output_file explicitly."
    )
