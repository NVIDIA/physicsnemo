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

"""Fit the training-only statistics, geometry encodings and field bases of Rotor37."""

import json
from pathlib import Path

import numpy as np
from geometry import fit_geometry_encodings

EOS_TOLERANCE = 1e-6


def quad_edges(quads, num_points):
    """Return the sorted unique undirected edges of quadrilateral faces."""
    quads = np.asarray(quads)
    if quads.ndim != 2 or quads.shape[1] != 4 or not len(quads):
        raise ValueError("Expected nonempty quadrilateral connectivity")
    if not np.issubdtype(quads.dtype, np.integer):
        raise ValueError("Quadrilateral connectivity must contain integer vertex IDs")
    if quads.min() < 0 or quads.max() >= num_points:
        raise ValueError("Quadrilateral connectivity references an invalid vertex")
    if np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0):
        raise ValueError("Quadrilaterals must have four distinct vertices")
    pairs = np.stack((quads, np.roll(quads, -1, axis=1)), axis=-1).reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int64)


def fit_log_basis(log_fields, weight=None, edges=None, energy_fraction=0.999):
    """Fit standardized POD coefficients and modes of training log fields.

    The POD metric is the mean square of ``weight`` times the centered log
    field. With ``edges``, it adds the mean square of the weighted differences
    across those edges, scaled so both terms carry equal training energy.
    Training coefficients have unit variance and multiply the returned modes,
    so the centered log fields are approximated by ``coefficients @ modes``.
    The retained modes hold at least ``energy_fraction`` of the training
    energy in this metric.
    """
    cases = len(log_fields)
    mean = log_fields.mean(axis=0)
    residual = log_fields - mean
    weighted = residual if weight is None else residual * weight
    gram = weighted @ weighted.T / weighted.shape[1]
    edge_weight = 0.0
    if edges is not None:
        jumps = weighted[:, edges[:, 0]] - weighted[:, edges[:, 1]]
        edge_weight = float(np.mean(weighted**2) / np.mean(jumps**2))
        gram += edge_weight * (jumps @ jumps.T) / jumps.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(gram / cases)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    energy = np.cumsum(eigenvalues) / eigenvalues.sum()
    rank = int(np.searchsorted(energy, energy_fraction)) + 1
    vectors = eigenvectors[:, :rank]
    modes = vectors.T @ residual / np.sqrt(cases)
    pivots = np.abs(modes).argmax(axis=1)
    signs = np.sign(modes[np.arange(rank), pivots])
    arrays = {
        "log_mean": mean,
        "modes": modes * signs[:, None],
        "coefficients": vectors * signs * np.sqrt(cases),
        "eigenvalues": eigenvalues[:rank],
    }
    metadata = {
        "rank": rank,
        "retained_energy": float(energy[rank - 1]),
        "edge_weight": edge_weight,
    }
    return arrays, metadata


def fit_field_bases(fields, edges, pressure_scale, energy_fraction=0.999):
    """Fit pressure and temperature bases and the pressure jump normalization.

    Pressure uses log-pressure residuals multiplied by the training geometric
    mean pressure divided by ``pressure_scale``, which linearizes them into
    normalized pressure perturbations, together with their edge differences.
    Temperature uses centered log temperature. The jump normalization is the
    mean squared normalized edge error of the training geometric mean pressure.
    """
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 3 or fields.shape[2] != 3 or len(fields) < 2:
        raise ValueError("Expected aligned [cases, vertices, 3] training fields")
    if not np.isfinite(fields).all() or np.any(fields <= 0):
        raise ValueError("Training fields must be finite and positive")
    log_pressure = np.log(fields[:, :, 1])
    pressure_mean = np.exp(log_pressure.mean(axis=0))
    pressure, pressure_metadata = fit_log_basis(
        log_pressure, pressure_mean / pressure_scale, edges, energy_fraction
    )
    temperature, temperature_metadata = fit_log_basis(
        np.log(fields[:, :, 2]), energy_fraction=energy_fraction
    )
    error = pressure_mean - fields[:, :, 1]
    jump = (error[:, edges[:, 0]] - error[:, edges[:, 1]]) / pressure_scale
    arrays = {f"pressure_{key}": value for key, value in pressure.items()}
    arrays.update({f"temperature_{key}": value for key, value in temperature.items()})
    arrays["pressure_jump_scale"] = np.asarray(np.mean(jump**2))
    return arrays, {"pressure": pressure_metadata, "temperature": temperature_metadata}


def fit_statistics(arrays, sample_ids):
    """Return normalization statistics and the gas constant of the training cases.

    ``arrays`` maps each prepared array name to its stacked training values.
    The gas constant is the mean of pressure over density times temperature,
    and preparation stops if it varies by more than ``EOS_TOLERANCE``.
    """
    stats = {}
    for key in ("points", "conditions", "fields", "globals"):
        values = arrays[key].astype(np.float64)
        values = values.reshape(-1, values.shape[-1])
        std = values.std(axis=0)
        if not np.all(std > 0):
            raise ValueError(f"Cannot normalize a constant training channel in {key}")
        stats[key] = {"mean": values.mean(axis=0).tolist(), "std": std.tolist()}
    fields = arrays["fields"].astype(np.float64)
    if not np.all(fields > 0):
        raise ValueError("Training fields must be finite and positive")
    gas = fields[..., 1] / (fields[..., 0] * fields[..., 2])
    gas_constant = float(gas.mean())
    deviation = float(np.abs(1 - gas_constant / gas).max())
    if deviation > EOS_TOLERANCE:
        raise ValueError(
            "Training fields do not satisfy a constant ideal-gas relation "
            f"within relative tolerance {EOS_TOLERANCE:g}"
        )
    stats["gas_constant"] = gas_constant
    stats["gas_constant_max_relative_deviation"] = deviation
    stats["sample_ids"] = list(sample_ids)
    return stats


def fit_transforms(data_dir, geometry_rank=32, energy_fraction=0.999):
    """Fit the statistics, geometry encodings and field bases on training cases.

    Writes ``stats.json``, ``basis.npz`` and ``basis.json`` next to the
    prepared samples.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_ids = manifest["splits"]["train"]
    names = ("points", "normals", "conditions", "fields", "globals")
    arrays, quads = {name: [] for name in names}, None
    for sample_id in train_ids:
        path = data_dir / manifest["samples"][str(sample_id)]["path"]
        with np.load(path) as sample:
            if quads is None:
                quads = sample["quads"]
            if not np.array_equal(sample["quads"], quads):
                raise ValueError("Training meshes must share ordered connectivity")
            for name in names:
                arrays[name].append(sample[name])
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    stats = fit_statistics(arrays, train_ids)
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    edges = quad_edges(quads, arrays["points"].shape[1])
    geometry, geometry_metadata = fit_geometry_encodings(
        arrays["points"], arrays["normals"], geometry_rank
    )
    bases, field_metadata = fit_field_bases(
        arrays["fields"], edges, stats["fields"]["std"][1], energy_fraction
    )
    np.savez(
        data_dir / "basis.npz",
        training_sample_ids=np.asarray(train_ids),
        quads=quads,
        edges=edges,
        **geometry,
        **bases,
    )
    metadata = {
        "training_sample_ids": train_ids,
        "geometry": geometry_metadata,
        "fields": field_metadata,
        "energy_fraction": energy_fraction,
        "pressure_jump_scale": float(bases["pressure_jump_scale"]),
    }
    (data_dir / "basis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
