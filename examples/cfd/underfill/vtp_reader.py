# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
VTP File Reader for Transient CFD Simulation Data.

Reads VTP files containing:
    - Static mesh coordinates (points)
    - Time-dependent scalar fields stored as numbered point-data arrays

Naming conventions supported (auto-detected per field):
    - <field>_step00, <field>_step01, ...   (e.g. epoxy_vof_step00)
    - <field>_00, <field>_01, ...           (e.g. epoxy_vof_00)
Each VTP file represents one simulation case (e.g., G1, G2, ...) with
multiple time steps.
"""

import os
import re
import numpy as np
import pyvista as pv
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Time-series field discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns ordered from most specific to most permissive.
# Each pattern must expose a named group ``idx`` used for sorting.
_TIME_SERIES_PATTERNS: list[re.Pattern] = [
    # <field>_step00, <field>_step01, ...
    re.compile(r"^(?P<field>.+?)_step(?P<idx>\d+)$"),
    # <field>_t0.000, <field>_t0.005, ...  (float time label)
    re.compile(r"^(?P<field>.+?)_t(?P<idx>\d+\.\d+)$"),
    # <field>_00, <field>_01, ...  (bare numeric suffix)
    re.compile(r"^(?P<field>.+?)_(?P<idx>\d+)$"),
]


def _discover_time_series(
    keys: list[str],
    field_name: Optional[str] = None,
) -> dict[str, list[tuple[float, str]]]:
    """
    Scan VTP point-data keys and group them into time-series fields.

    Args:
        keys: All point-data key names in the VTP file.
        field_name: If provided, only return series whose base name matches
                    this value exactly (case-sensitive). If ``None``, return
                    every detected series.

    Returns:
        ``{field_name: [(sort_key, original_key), ...]}`` sorted by time index.
        ``sort_key`` is a float so that both integer step indices and
        floating-point time labels are handled uniformly.
    """
    # field_name -> list of (sort_key, original_key)
    groups: dict[str, list[tuple[float, str]]] = {}

    for key in keys:
        for pattern in _TIME_SERIES_PATTERNS:
            m = pattern.match(key)
            if m is None:
                continue
            name = m.group("field")
            idx = float(m.group("idx"))

            # Filter early when a specific field is requested.
            if field_name is not None and name != field_name:
                break  # no point trying other patterns for this key

            groups.setdefault(name, []).append((idx, key))
            break  # matched – skip remaining patterns for this key

    # Sort each series by time index
    for name in groups:
        groups[name].sort(key=lambda x: x[0])

    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# File-level helpers
# ═══════════════════════════════════════════════════════════════════════════════


def find_vtp_files(base_data_dir: str) -> list[str]:
    """
    Find all VTP files in directory, sorted naturally.

    Args:
        base_data_dir: Directory to search

    Returns:
        List of absolute paths to VTP files, sorted
    """
    if not os.path.isdir(base_data_dir):
        return []

    vtps = [
        os.path.join(base_data_dir, f)
        for f in os.listdir(base_data_dir)
        if f.lower().endswith(".vtp")
    ]

    def natural_key(name):
        """Sort key for natural ordering (handles numbers properly)."""
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(name))
        ]

    return sorted(vtps, key=natural_key)


def load_vtp_file(
    vtp_path: str,
    field_name: Optional[str] = None,
    debug: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load a single VTP file and extract time-series scalar fields.

    Args:
        vtp_path: Path to VTP file.
        field_name: Base name of the time-series to extract (e.g.
                    ``"epoxy_vof"``).  If ``None``, *every* detected
                    time-series field is returned.
        debug: Enable verbose output.

    Returns:
        Tuple of:
            - coords: ``[N, 3]`` mesh coordinates
            - fields: ``{name: [T, N, 1]}`` for each discovered series
    """
    mesh = pv.read(vtp_path)

    if not hasattr(mesh, "points"):
        raise ValueError(f"Cannot extract points from {vtp_path}")

    coords = np.array(mesh.points, dtype=np.float64)  # [N, 3]
    N_nodes = coords.shape[0]

    if debug:
        print(f"\n    [VTP] {os.path.basename(vtp_path)}")
        print(f"      Nodes: {N_nodes}")
        print(
            f"      Coords range: "
            f"x[{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]  "
            f"y[{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]  "
            f"z[{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]"
        )
        print(f"      Available keys: {list(mesh.point_data.keys())}")

    # ── Discover time-series fields ───────────────────────────────────────
    all_keys = list(mesh.point_data.keys())
    series = _discover_time_series(all_keys, field_name=field_name)

    if not series:
        hint = f" matching '{field_name}'" if field_name else ""
        raise ValueError(
            f"No time-series fields{hint} found in {vtp_path}. "
            f"Available keys: {all_keys}"
        )

    # ── Stack each series into [T, N, 1] ─────────────────────────────────
    fields: dict[str, np.ndarray] = {}

    for name, entries in series.items():
        if debug:
            first_idx = entries[0][0]
            last_idx = entries[-1][0]
            print(
                f"      Field '{name}': {len(entries)} steps "
                f"(idx {first_idx} → {last_idx})"
            )

        arrays: list[np.ndarray] = []
        for _idx, key in entries:
            arr = np.asarray(mesh.point_data[key], dtype=np.float64)
            if arr.ndim != 1:
                arr = arr.flatten()
            if arr.shape[0] != N_nodes:
                raise ValueError(
                    f"Node count mismatch in '{key}': "
                    f"got {arr.shape[0]}, expected {N_nodes}"
                )
            arrays.append(arr)

        stacked = np.stack(arrays, axis=0)[:, :, np.newaxis]  # [T, N, 1]
        fields[name] = stacked

        if debug:
            print(
                f"        shape: {stacked.shape}, "
                f"range [{stacked.min():.4f}, {stacked.max():.4f}]"
            )

    return coords, fields


# ═══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════════════


def process_vtp_data(
    data_dir: str,
    field_name: Optional[str] = None,
    num_samples: Optional[int] = None,
    logger=None,
    debug: bool = False,
) -> list[dict]:
    """
    Process all VTP files in a directory.

    Args:
        data_dir: Directory containing VTP files.
        field_name: Base name of the time-series to extract (passed to
                    ``load_vtp_file``). ``None`` → auto-detect all.
        num_samples: Maximum number of files to process.
        logger: Logger instance.
        debug: Enable verbose output.

    Returns:
        List of dictionaries, each containing:
            - ``"coords"``: ``[N, 3]`` coordinates
            - One key per discovered field: ``[T, N, 1]``
    """
    vtp_files = find_vtp_files(data_dir)

    if not vtp_files:
        msg = f"No VTP files found in: {data_dir}"
        if logger:
            logger.error(msg)
        print(f"ERROR: {msg}")
        return []

    if debug:
        print(f"\n{'=' * 60}")
        print("VTP Data Processing")
        print(f"{'=' * 60}")
        print(f"  Directory:  {data_dir}")
        print(f"  Found:      {len(vtp_files)} VTP files")
        print(f"  Requested:  {num_samples or 'all'}")
        print(f"  Field:      {field_name or '(auto-detect all)'}")

    data_records: list[dict] = []

    for i, vtp_path in enumerate(vtp_files):
        if num_samples is not None and i >= num_samples:
            break

        if logger:
            logger.info(f"Processing: {os.path.basename(vtp_path)}")

        try:
            coords, fields = load_vtp_file(
                vtp_path, field_name=field_name, debug=debug
            )
            record = {"coords": coords, **fields}
            data_records.append(record)

        except Exception as e:
            msg = f"Error processing {vtp_path}: {e}"
            if logger:
                logger.error(msg)
            print(f"ERROR: {msg}")
            continue

    if debug:
        print(f"\n  Successfully processed {len(data_records)} files")
        if data_records:
            rec = data_records[0]
            print("  First record:")
            for k, v in rec.items():
                print(f"    {k}: {v.shape}")
        print(f"{'=' * 60}\n")

    return data_records


# ═══════════════════════════════════════════════════════════════════════════════
# Hydra-compatible Reader class
# ═══════════════════════════════════════════════════════════════════════════════


class Reader:
    """
    VTP Reader class for integration with datapipe.

    The ``field_name`` parameter controls which time-series is extracted:

        - ``"epoxy_vof"`` → extracts ``epoxy_vof_step00``, ``epoxy_vof_step01``, …
        - ``None`` → auto-detects and returns *all* time-series fields

    Usage::

        reader = Reader(field_name="epoxy_vof", debug=True)
        records = reader(data_dir="./data", num_samples=10, split="train")
    """

    def __init__(
        self,
        field_name: Optional[str] = "epoxy_vof",
        debug: bool = False,
    ):
        """
        Initialize reader.

        Args:
            field_name: Base name of the time-series field to extract.
                        Set to ``None`` to auto-detect all fields.
            debug: Enable verbose output.
        """
        self.field_name = field_name
        self.debug = debug

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: Optional[str] = None,
        logger=None,
        **kwargs,
    ) -> list[dict]:
        """
        Load VTP data.

        Args:
            data_dir: Directory containing VTP files.
            num_samples: Maximum number of samples to load.
            split: Data split name (for logging).
            logger: Logger instance.

        Returns:
            List of data record dictionaries.
        """
        return process_vtp_data(
            data_dir=data_dir,
            field_name=self.field_name,
            num_samples=num_samples,
            logger=logger,
            debug=self.debug,
        )
