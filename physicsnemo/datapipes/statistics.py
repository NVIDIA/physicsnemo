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

"""
Per-field statistics collection with parquet caching.

Provides :class:`FieldStatisticsCollector`, a utility that iterates a
dataset, computes summary statistics for every tensor field, and writes
the results to a parquet file.  A lightweight caching layer avoids
recomputation when the dataset hasn't changed.

Requires ``pyarrow`` (not a core physicsnemo dependency).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes.registry import register

pa = OptionalImport("pyarrow", package_hint="pip install pyarrow")
pq = OptionalImport("pyarrow.parquet", package_hint="pip install pyarrow")

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """Online accumulator for mean, variance, skewness, kurtosis, min, max.

    Uses Welford's algorithm for numerically stable single-pass computation
    of central moments up to 4th order.
    """

    __slots__ = ("n", "mean", "m2", "m3", "m4", "vmin", "vmax", "abs_sum")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.m3: float = 0.0
        self.m4: float = 0.0
        self.vmin: float = float("inf")
        self.vmax: float = float("-inf")
        self.abs_sum: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Incorporate a 1-D tensor of values."""
        for x in values.tolist():
            self.n += 1
            n = self.n
            delta = x - self.mean
            delta_n = delta / n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (n - 1)

            self.mean += delta_n
            self.m4 += (
                term1 * delta_n2 * (n * n - 3 * n + 3)
                + 6 * delta_n2 * self.m2
                - 4 * delta_n * self.m3
            )
            self.m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.m2
            self.m2 += term1

            if x < self.vmin:
                self.vmin = x
            if x > self.vmax:
                self.vmax = x
            self.abs_sum += abs(x)

    def update_bulk(self, values: torch.Tensor) -> None:
        """Incorporate a 1-D tensor using vectorized operations for speed."""
        v = values.double()
        count = int(v.numel())
        if count == 0:
            return

        batch_mean = v.mean().item()
        batch_min = v.min().item()
        batch_max = v.max().item()
        batch_abs_sum = v.abs().sum().item()

        # Centered moments for the batch
        centered = v - batch_mean
        batch_m2 = (centered**2).sum().item()
        batch_m3 = (centered**3).sum().item()
        batch_m4 = (centered**4).sum().item()

        if self.n == 0:
            self.n = count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.m3 = batch_m3
            self.m4 = batch_m4
            self.vmin = batch_min
            self.vmax = batch_max
            self.abs_sum = batch_abs_sum
            return

        # Parallel/combined Welford merge
        n_a = self.n
        n_b = count
        n_ab = n_a + n_b
        delta = batch_mean - self.mean
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2

        self.m4 = (
            self.m4
            + batch_m4
            + delta4 * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n_ab**3)
            + 6 * delta2 * (n_a * n_a * batch_m2 + n_b * n_b * self.m2) / (n_ab**2)
            + 4 * delta * (n_a * batch_m3 - n_b * self.m3) / n_ab
        )
        self.m3 = (
            self.m3
            + batch_m3
            + delta3 * n_a * n_b * (n_a - n_b) / (n_ab**2)
            + 3 * delta * (n_a * batch_m2 - n_b * self.m2) / n_ab
        )
        self.m2 = self.m2 + batch_m2 + delta2 * n_a * n_b / n_ab
        self.mean = (n_a * self.mean + n_b * batch_mean) / n_ab
        self.n = n_ab

        if batch_min < self.vmin:
            self.vmin = batch_min
        if batch_max > self.vmax:
            self.vmax = batch_max
        self.abs_sum += batch_abs_sum

    def result(self) -> dict[str, float]:
        """Return computed statistics."""
        n = self.n
        if n == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "var": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "min": 0.0,
                "max": 0.0,
                "abs_mean": 0.0,
            }
        var = self.m2 / n if n > 1 else 0.0
        std = var**0.5
        skewness = (self.m3 / n) / (var**1.5) if var > 0 else 0.0
        # Excess kurtosis (normal = 0)
        kurtosis = (self.m4 / n) / (var**2) - 3.0 if var > 0 else 0.0
        return {
            "count": n,
            "mean": self.mean,
            "std": std,
            "var": var,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": self.vmin,
            "max": self.vmax,
            "abs_mean": self.abs_sum / n,
        }


def _field_stats(tensor: torch.Tensor) -> list[dict[str, float | int]]:
    """Compute per-component summary statistics for a field tensor.

    Returns one dict per component. Scalars get a single row with
    ``component=-1``. Vectors get one row per component (0, 1, 2, ...).
    """
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    n_spatial = int(tensor.shape[0])
    n_components = int(tensor.shape[1]) if tensor.ndim > 1 else 1

    if n_components == 1:
        col = tensor.float().reshape(-1)
        return [
            {
                "component": -1,
                "n_spatial": n_spatial,
                "n_components": n_components,
                "mean": col.mean().item(),
                "std": col.std().item(),
                "min": col.min().item(),
                "max": col.max().item(),
                "abs_mean": col.abs().mean().item(),
                "abs_max": col.abs().max().item(),
            }
        ]

    rows = []
    for c in range(n_components):
        col = tensor[:, c].float()
        rows.append(
            {
                "component": c,
                "n_spatial": n_spatial,
                "n_components": n_components,
                "mean": col.mean().item(),
                "std": col.std().item(),
                "min": col.min().item(),
                "max": col.max().item(),
                "abs_mean": col.abs().mean().item(),
                "abs_max": col.abs().max().item(),
            }
        )
    return rows


def _update_accumulators(
    accumulators: dict[tuple[str, int], WelfordAccumulator],
    field_key: str,
    tensor: torch.Tensor,
) -> None:
    """Feed a field tensor into the appropriate Welford accumulators."""
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    n_components = int(tensor.shape[1]) if tensor.ndim > 1 else 1
    if n_components == 1:
        key = (field_key, -1)
        if key not in accumulators:
            accumulators[key] = WelfordAccumulator()
        accumulators[key].update_bulk(tensor.float().reshape(-1))
    else:
        for c in range(n_components):
            key = (field_key, c)
            if key not in accumulators:
                accumulators[key] = WelfordAccumulator()
            accumulators[key].update_bulk(tensor[:, c].float())


def _extract_fields_from_mesh(mesh, sections: Sequence[str] | None):
    """Yield (dotted_key, tensor) pairs from a Mesh object."""
    if hasattr(mesh, "points") and mesh.points is not None:
        yield "points", mesh.points

    if sections is None:
        sections = ["point_data", "cell_data", "global_data"]
    for section_name in sections:
        section = getattr(mesh, section_name, None)
        if section is None:
            continue
        for key in sorted(section.keys()):
            val = section[key]
            if isinstance(val, torch.Tensor):
                yield f"{section_name}.{key}", val


def _extract_fields_from_tensordict(td, sections: Sequence[str] | None):
    """Yield (dotted_key, tensor) pairs from a TensorDict."""
    from tensordict import TensorDict

    if sections is not None:
        for section_name in sections:
            if section_name in td.keys():
                sub = td[section_name]
                if isinstance(sub, TensorDict):
                    for key in sorted(sub.keys()):
                        val = sub[key]
                        if isinstance(val, torch.Tensor):
                            yield f"{section_name}.{key}", val
    else:
        for key in sorted(td.keys()):
            val = td[key]
            if isinstance(val, TensorDict):
                for sub_key in sorted(val.keys()):
                    sub_val = val[sub_key]
                    if isinstance(sub_val, torch.Tensor):
                        yield f"{key}.{sub_key}", sub_val
            elif isinstance(val, torch.Tensor):
                yield key, val


def _extract_run_id(source_path: str) -> str:
    """Extract a run identifier from the source path."""
    parts = Path(source_path).parts
    for part in reversed(parts):
        if part.startswith("run"):
            return part
    return Path(source_path).parent.name


@register()
class FieldStatisticsCollector:
    """Compute per-field, per-sample statistics from a dataset and cache to parquet.

    The collector iterates the dataset, computes summary statistics (mean, std,
    min, max, abs_mean, abs_max) for every tensor field, and writes the results
    to a parquet file.  Arrow file-level metadata stores the dataset size and
    tracked keys, enabling cache validity checks that avoid recomputation when
    the dataset hasn't changed.

    Works with datasets that return ``(Mesh, metadata)`` or
    ``(TensorDict, metadata)`` tuples from ``__getitem__``.

    Parameters
    ----------
    output_path : str or Path
        Where to write the parquet file.
    keys : list of str, optional
        Dotted field keys to track (e.g. ``["point_data.pressure"]``).
        ``None`` means all leaf tensors found in the first sample.
    sections : list of str, optional
        Which data sections to inspect (e.g. ``["point_data", "cell_data"]``).
        ``None`` means all available sections.
    force : bool
        If ``True``, always recompute even if a valid cache exists.
    """

    def __init__(
        self,
        output_path: str | Path,
        keys: list[str] | None = None,
        sections: list[str] | None = None,
        force: bool = False,
    ) -> None:
        self._output_path = Path(output_path)
        self._keys = keys
        self._sections = sections
        self._force = force

    def _extract_fields(self, data) -> list[tuple[str, torch.Tensor]]:
        """Extract (dotted_key, tensor) pairs from a data sample."""
        from tensordict import TensorDict

        if hasattr(data, "point_data"):
            fields = list(_extract_fields_from_mesh(data, self._sections))
        elif isinstance(data, TensorDict):
            fields = list(_extract_fields_from_tensordict(data, self._sections))
        else:
            raise TypeError(
                f"Unsupported data type {type(data).__name__}. "
                "Expected Mesh or TensorDict."
            )

        if self._keys is not None:
            key_set = set(self._keys)
            fields = [(k, v) for k, v in fields if k in key_set]

        return fields

    def _build_cache_metadata(
        self, n_samples: int, tracked_keys: list[str]
    ) -> dict[str, str]:
        return {
            b"n_samples": str(n_samples).encode(),
            b"keys": json.dumps(sorted(tracked_keys)).encode(),
            b"timestamp": datetime.now(timezone.utc).isoformat().encode(),
        }

    def _read_cache_metadata(self) -> dict[str, Any] | None:
        """Read cache metadata from an existing parquet file."""
        if not self._output_path.exists():
            return None
        try:
            pf = pq.ParquetFile(str(self._output_path))
            meta = pf.schema_arrow.metadata
            if meta is None:
                return None
            return {
                "n_samples": int(meta.get(b"n_samples", b"-1")),
                "keys": json.loads(meta.get(b"keys", b"[]")),
                "timestamp": meta.get(b"timestamp", b"").decode(),
            }
        except Exception:
            return None

    def is_cached(self, dataset) -> bool:
        """Check whether a valid cached stats file exists for this dataset.

        Parameters
        ----------
        dataset : DatasetBase
            Dataset to check against (uses ``len(dataset)`` and tracked keys).

        Returns
        -------
        bool
        """
        if self._force:
            return False
        meta = self._read_cache_metadata()
        if meta is None:
            return False
        if meta["n_samples"] != len(dataset):
            logger.info(
                "Cache stale: n_samples changed (%d -> %d)",
                meta["n_samples"],
                len(dataset),
            )
            return False

        if self._keys is not None and meta["keys"] != sorted(self._keys):
            logger.info("Cache stale: tracked keys changed")
            return False

        return True

    def load_cached(self):
        """Load previously cached statistics from parquet.

        Returns
        -------
        pyarrow.Table

        Raises
        ------
        FileNotFoundError
            If no cached file exists at ``output_path``.
        """
        if not self._output_path.exists():
            raise FileNotFoundError(f"No cached stats at {self._output_path}")
        return pq.read_table(str(self._output_path))

    @property
    def aggregate_path(self) -> Path:
        """Path to the companion aggregate-statistics parquet file."""
        stem = self._output_path.stem
        return self._output_path.with_name(f"{stem}_aggregate.parquet")

    def load_aggregate(self):
        """Load the aggregate (Welford) statistics from parquet.

        Returns
        -------
        pyarrow.Table

        Raises
        ------
        FileNotFoundError
            If no aggregate file exists.
        """
        p = self.aggregate_path
        if not p.exists():
            raise FileNotFoundError(f"No aggregate stats at {p}")
        return pq.read_table(str(p))

    def collect(self, dataset) -> "pa.Table":
        """Iterate the dataset, compute per-field stats, and write to parquet.

        Computes both per-sample statistics (written to ``output_path``) and
        global aggregate statistics using Welford's online algorithm (written
        to ``output_path`` with ``_aggregate`` suffix).  The aggregates include
        exact global mean, std, variance, skewness, excess kurtosis, min, max,
        and abs_mean computed over **every data point** across all samples --
        not approximated from per-sample summaries.

        If a valid cache exists and ``force=False``, returns the cached table
        without recomputing.

        Parameters
        ----------
        dataset : DatasetBase
            Must support ``len(dataset)`` and ``dataset[i]`` returning
            ``(data, metadata)`` where ``data`` is a ``Mesh`` or ``TensorDict``.

        Returns
        -------
        pyarrow.Table
            The per-sample statistics table.
        """
        if self.is_cached(dataset):
            logger.info("Using cached statistics from %s", self._output_path)
            return self.load_cached()

        n_samples = len(dataset)
        rows: list[dict[str, Any]] = []
        tracked_keys: set[str] = set()
        accumulators: dict[tuple[str, int], WelfordAccumulator] = {}

        for i in range(n_samples):
            data, metadata = dataset[i]
            source_path = metadata.get("source_path", "")
            run_id = _extract_run_id(source_path)

            fields = self._extract_fields(data)
            for field_key, tensor in fields:
                tracked_keys.add(field_key)
                _update_accumulators(accumulators, field_key, tensor)
                for stats in _field_stats(tensor):
                    stats["sample_index"] = i
                    stats["source_path"] = source_path
                    stats["run_id"] = run_id
                    stats["field_key"] = field_key
                    rows.append(stats)

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                logger.info("  [%d/%d]", i + 1, n_samples)

        # -- Per-sample table (unchanged schema) --
        per_sample_schema = pa.schema(
            [
                ("sample_index", pa.int32()),
                ("source_path", pa.string()),
                ("run_id", pa.string()),
                ("field_key", pa.string()),
                ("component", pa.int32()),
                ("n_spatial", pa.int64()),
                ("n_components", pa.int32()),
                ("mean", pa.float64()),
                ("std", pa.float64()),
                ("min", pa.float64()),
                ("max", pa.float64()),
                ("abs_mean", pa.float64()),
                ("abs_max", pa.float64()),
            ]
        )

        if not rows:
            table = pa.table(
                {field.name: [] for field in per_sample_schema},
                schema=per_sample_schema,
            )
        else:
            arrays = {
                field.name: [row[field.name] for row in rows]
                for field in per_sample_schema
            }
            table = pa.table(arrays, schema=per_sample_schema)

        cache_meta = self._build_cache_metadata(n_samples, sorted(tracked_keys))
        existing_meta = table.schema.metadata or {}
        existing_meta.update(cache_meta)
        table = table.replace_schema_metadata(existing_meta)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(self._output_path))
        logger.info(
            "Wrote %d per-sample rows (%d samples, %d fields) to %s",
            len(rows),
            n_samples,
            len(tracked_keys),
            self._output_path,
        )

        # -- Aggregate table from Welford accumulators --
        agg_schema = pa.schema(
            [
                ("field_key", pa.string()),
                ("component", pa.int32()),
                ("count", pa.int64()),
                ("mean", pa.float64()),
                ("std", pa.float64()),
                ("var", pa.float64()),
                ("skewness", pa.float64()),
                ("kurtosis", pa.float64()),
                ("min", pa.float64()),
                ("max", pa.float64()),
                ("abs_mean", pa.float64()),
            ]
        )
        agg_rows: list[dict[str, Any]] = []
        for (field_key, comp), acc in sorted(accumulators.items()):
            result = acc.result()
            agg_rows.append(
                {
                    "field_key": field_key,
                    "component": comp,
                    "count": result["count"],
                    "mean": result["mean"],
                    "std": result["std"],
                    "var": result["var"],
                    "skewness": result["skewness"],
                    "kurtosis": result["kurtosis"],
                    "min": result["min"],
                    "max": result["max"],
                    "abs_mean": result["abs_mean"],
                }
            )

        if not agg_rows:
            agg_table = pa.table(
                {field.name: [] for field in agg_schema}, schema=agg_schema
            )
        else:
            agg_arrays = {
                field.name: [row[field.name] for row in agg_rows]
                for field in agg_schema
            }
            agg_table = pa.table(agg_arrays, schema=agg_schema)

        agg_table = agg_table.replace_schema_metadata(
            {**cache_meta, b"type": b"welford_aggregate"}
        )
        agg_path = self.aggregate_path
        pq.write_table(agg_table, str(agg_path))
        logger.info("Wrote %d aggregate rows to %s", len(agg_rows), agg_path)

        return table

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_path={str(self._output_path)!r}, "
            f"keys={self._keys}, "
            f"sections={self._sections}, "
            f"force={self._force})"
        )
