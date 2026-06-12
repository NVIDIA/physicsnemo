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

"""Reader for the MIT/Harvard bumper-beam crash dataset (VTKHDF format).

Each ``sim_<id>/sim_<id>.vtkhdf`` file is a VTKHDF v2.2 ``PartitionedDataSetCollection``:
a multi-part assembly of ~23 transient ``UnstructuredGrid`` parts. This reader keeps the
"structural" parts (those carrying ``Von Mises_2D`` / ``Plastic Strain_2D`` cell fields),
merges them into a single mesh, and emits the 4-tuple contract expected by ``datapipe.py``,
matching ``vtp_reader.Reader``:

    srcs, dsts, point_data_all, global_features_all = Reader()(data_dir, num_samples, ...)

per sample:
    srcs[i], dsts[i]          : int64 undirected mesh-edge endpoints in [0, N)
    point_data_all[i]         : {"coords": [T,N,3],
                                 "point_data": {"effective_plastic_strain_t<j>": [N],
                                                "stress_vm_t<j>": [N], ...}}
    global_features_all[i]    : {"velocity_x": ..., "thickness_scale": ..., "rwall_origin_y": ...}

Cell-centered targets are averaged to points. Per-run global scalars come from a JSON
(``global_features_filepath``) keyed by run id, exactly like ``vtp_reader`` (built from
``bumper_beam_master_with_split.csv`` via ``make_global_features.py``). The same master CSV
supplies the train/val/test split.
"""

import csv
import os
import re

import numpy as np
import h5py

try:  # the example's helper; replicated below so the reader also runs standalone
    from utils import get_global_features_for_run, load_global_features
except Exception:  # pragma: no cover - fallback when run outside the example package
    import json

    def load_global_features(json_path):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Global features file not found: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError("Global features JSON must be a dict keyed by run_id")
        return data

    def get_global_features_for_run(all_global_features, run_id):
        try:
            return all_global_features[run_id]
        except KeyError:
            raise KeyError(f"run_id '{run_id}' not found in global features file")


# CellData field name in the file  ->  point_data base key the datapipe expects.
TARGET_MAP = {
    "Plastic Strain_2D": "effective_plastic_strain",
    "Von Mises_2D": "stress_vm",
}


def _natural_key(name):
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.findall(r"\d+|\D+", os.path.basename(str(name)))
    ]


def load_split_map(master_csv):
    """Return {sim_name -> split} from the master CSV ('val' is used for 'validation')."""
    if master_csv is None or not os.path.isfile(master_csv):
        return {}
    out = {}
    with open(master_csv, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("sim_name")
            if name:
                out[name] = (row.get("split") or "").strip().lower()
    return out


def find_sim_files(data_dir, split=None, split_map=None):
    """List ``<data_dir>/<sim>/<sim>.vtkhdf`` files, optionally filtered by CSV split.

    Falls back to every ``*.vtkhdf`` found when no split map / split is given, so the
    reader is usable standalone.
    """
    if not os.path.isdir(data_dir):
        return []
    files = []
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if not os.path.isdir(sub):
            continue
        cand = os.path.join(sub, f"{entry}.vtkhdf")
        if not os.path.isfile(cand):
            hits = [g for g in os.listdir(sub) if g.lower().endswith(".vtkhdf")]
            if not hits:
                continue
            cand = os.path.join(sub, hits[0])
        files.append(cand)

    if split and split_map:
        want = "val" if split in ("val", "validation") else split
        files = [
            p
            for p in files
            if split_map.get(os.path.splitext(os.path.basename(p))[0]) == want
        ]
    return sorted(files, key=_natural_key)


def _is_structural(part_group):
    cd = part_group.get("CellData")
    if cd is None:
        return False
    return all(name in cd for name in TARGET_MAP)


def structural_part_names(root, exclude_parts=()):
    """Part names (assembly members) that carry both stress and strain cell fields."""
    names = []
    for name, obj in root.items():
        if name in ("Assembly", "Steps") or not isinstance(obj, h5py.Group):
            continue
        if name in exclude_parts:
            continue
        if _is_structural(obj):
            names.append(name)
    return sorted(names, key=_natural_key)


def _step_slices(part, dataset_len_key, offsets_key):
    """Yield (t, start, count) for a per-step concatenated array."""
    counts = np.asarray(part[dataset_len_key])
    offsets = np.asarray(part[f"Steps/{offsets_key}"])
    for t in range(len(counts)):
        yield t, int(offsets[t]), int(counts[t])


def read_part(part):
    """Read one structural part.

    Returns:
        pos:        [T, n, 3] absolute point positions per step
        cells:      list[list[int]] connectivity (local node ids), constant over time
        targets:    {base_key: [T, n]} point-averaged target series
    """
    n_steps = len(part["NumberOfPoints"])
    n_pts = int(part["NumberOfPoints"][0])

    # --- geometry: Points are stored per-step, indexed by Steps/PointOffsets ---
    points = np.asarray(part["Points"], dtype=np.float32)
    pos = np.empty((n_steps, n_pts, 3), dtype=np.float32)
    for t, start, count in _step_slices(part, "NumberOfPoints", "PointOffsets"):
        assert count == n_pts, "non-constant point count across steps not supported"
        pos[t] = points[start : start + count]

    # --- connectivity: stored once (CellOffsets/ConnectivityIdOffsets are all zero) ---
    n_cells = int(part["NumberOfCells"][0])
    offsets = np.asarray(part["Offsets"][: n_cells + 1])
    conn = np.asarray(part["Connectivity"][: int(part["NumberOfConnectivityIds"][0])])
    cells = [conn[offsets[i] : offsets[i + 1]].tolist() for i in range(n_cells)]

    # flat (cell-corner -> node) incidence for cell->point averaging (any cell size)
    cell_sizes = np.diff(offsets)
    node_ids = conn  # already the flattened corner->node map
    cell_ids = np.repeat(np.arange(n_cells), cell_sizes)
    counts = np.zeros(n_pts, dtype=np.float64)
    np.add.at(counts, node_ids, 1.0)
    counts[counts == 0] = 1.0  # isolated points: avoid divide-by-zero

    # --- targets: cell-centered, per-step; average onto points ---
    targets = {}
    for field_name, base_key in TARGET_MAP.items():
        cell_series = np.asarray(part[f"CellData/{field_name}"], dtype=np.float64)
        cell_off = np.asarray(part[f"Steps/CellDataOffsets/{field_name}"])
        cell_vals = np.empty((n_steps, n_cells), dtype=np.float64)
        for t in range(n_steps):
            s = int(cell_off[t])
            cell_vals[t] = cell_series[s : s + n_cells]
        point_sum = np.zeros((n_steps, n_pts), dtype=np.float64)
        np.add.at(point_sum, (slice(None), node_ids), cell_vals[:, cell_ids])
        targets[base_key] = (point_sum / counts[None, :]).astype(np.float32)

    return pos, cells, targets


def build_edges(cells):
    """Unique undirected edges (consecutive node pairs around each cell)."""
    edges = set()
    for cell in cells:
        m = len(cell)
        for i in range(m):
            a, b = cell[i], cell[(i + 1) % m]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    return edges


def load_vtkhdf_file(path, exclude_parts=()):
    """Merge the structural parts of one VTKHDF file into the flat reader record.

    Returns:
        coords: [T, N, 3]
        src, dst: [E] int64 edge endpoints in [0, N)
        targets: {base_key: [T, N]}
    """
    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        part_names = structural_part_names(root, exclude_parts=exclude_parts)
        if not part_names:
            raise ValueError(f"No structural parts with target fields in {path}")

        pos_list, target_acc, edge_set = [], None, set()
        offset = 0
        n_steps_ref = None
        for name in part_names:
            pos, cells, targets = read_part(root[name])
            if n_steps_ref is None:
                n_steps_ref = pos.shape[0]
                target_acc = {k: [] for k in targets}
            elif pos.shape[0] != n_steps_ref:
                raise ValueError(
                    f"Part '{name}' has {pos.shape[0]} steps, expected {n_steps_ref}"
                )
            pos_list.append(pos)
            for k, v in targets.items():
                target_acc[k].append(v)
            for a, b in build_edges(cells):
                edge_set.add((a + offset, b + offset))
            offset += pos.shape[1]

    coords = np.concatenate(pos_list, axis=1)  # [T, N, 3]
    merged_targets = {k: np.concatenate(v, axis=1) for k, v in target_acc.items()}
    if edge_set:
        edge_arr = np.array(sorted(edge_set), dtype=np.int64)
        src, dst = edge_arr[:, 0], edge_arr[:, 1]
    else:
        src = dst = np.zeros((0,), dtype=np.int64)
    assert coords.shape[1] == offset
    assert src.size == 0 or (src.min() >= 0 and dst.max() < offset)
    return coords, src, dst, merged_targets


def process_vtkhdf_data(
    data_dir,
    num_samples,
    split=None,
    global_features_filepath=None,
    master_csv=None,
    exclude_parts=(),
    logger=None,
):
    """Build the per-sample lists expected by the datapipe."""
    split_map = load_split_map(master_csv)
    files = find_sim_files(data_dir, split=split, split_map=split_map)
    if not files:
        msg = f"No VTKHDF samples found in {data_dir} for split={split!r}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    all_global = (
        load_global_features(global_features_filepath)
        if global_features_filepath
        else None
    )

    srcs, dsts, point_data_all, global_features_all = [], [], [], []
    for path in files[:num_samples]:
        run_id = os.path.splitext(os.path.basename(path))[0]
        if logger:
            logger.info(f"Processing {run_id} ...")
        coords, src, dst, targets = load_vtkhdf_file(path, exclude_parts=exclude_parts)

        # Per-step point_data keys: "<base>_t<j>" so the datapipe groups them by prefix.
        point_data = {}
        for base_key, series in targets.items():  # series: [T, N]
            for t in range(series.shape[0]):
                point_data[f"{base_key}_t{t}"] = series[t]

        record = {"coords": coords, "point_data": point_data}
        global_features = (
            get_global_features_for_run(all_global, run_id)
            if all_global is not None
            else {}
        )

        srcs.append(src)
        dsts.append(dst)
        point_data_all.append(record)
        global_features_all.append(global_features)

    return srcs, dsts, point_data_all, global_features_all


class Reader:
    """VTKHDF reader for the bumper crash dataset.

    Args (from ``conf/reader/vtkhdf.yaml`` via Hydra):
        master_csv: path to ``bumper_beam_master_with_split.csv`` (for the split filter).
                    If None, the reader auto-discovers it next to ``data_dir`` and, failing
                    that, loads every ``*.vtkhdf`` regardless of split.
        exclude_parts: optional list of part names to drop (e.g. ["RWALL_1"]).
    """

    def __init__(self, master_csv=None, exclude_parts=None):
        self.master_csv = master_csv
        self.exclude_parts = tuple(exclude_parts or ())

    def _resolve_master_csv(self, data_dir):
        if self.master_csv and os.path.isfile(self.master_csv):
            return self.master_csv
        for cand in (
            os.path.join(os.path.dirname(os.path.abspath(data_dir)),
                         "bumper_beam_master_with_split.csv"),
            os.path.join(data_dir, "bumper_beam_master_with_split.csv"),
        ):
            if os.path.isfile(cand):
                return cand
        return None

    def __call__(
        self,
        data_dir,
        num_samples,
        split=None,
        global_features_filepath=None,
        logger=None,
        **kwargs,
    ):
        return process_vtkhdf_data(
            data_dir=data_dir,
            num_samples=num_samples,
            split=split,
            global_features_filepath=global_features_filepath,
            master_csv=self._resolve_master_csv(data_dir),
            exclude_parts=self.exclude_parts,
            logger=logger,
        )
