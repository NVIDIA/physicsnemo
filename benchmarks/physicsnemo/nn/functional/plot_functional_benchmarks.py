#!/usr/bin/env python3
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

"""Generate functional benchmark bar plots from ASV JSON outputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import torch

from benchmarks.physicsnemo.nn.functional._spec_utils import (
    PHASE_ORDER,
    BenchmarkKey,
    build_benchmark_plan,
    case_labels,
)
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec

# Name of the ASV benchmark function to extract from result payloads.
_BENCHMARK_SUFFIX = "FunctionalBenchmarks.time_functional"

# Keep implementation order and colors stable across plots.
_IMPL_ORDER = ("warp", "cuml", "scipy", "torch")
_IMPL_COLORS = {
    "warp": "#76B900",
    "cuml": "#2E2E2E",
    "scipy": "#5A5A5A",
    "torch": "#111111",
    "unknown": "#8A8A8A",
}

# Type aliases used throughout the parsing/plotting pipeline.
CaseMap: TypeAlias = dict[str, dict[str, float]]
SpecPhaseMap: TypeAlias = dict[str, dict[str, CaseMap]]


@dataclass(frozen=True)
class BenchmarkSpecData:
    """Plottable benchmark metadata for one FunctionSpec."""

    slug: str
    implementations: tuple[str, ...]
    labels_by_phase: dict[str, list[str]]


def _camel_to_snake(name: str) -> str:
    """Convert a class-style name (e.g. ``MeshToVoxelFraction``) to snake_case."""

    # Step 1: split transitions like "PointCloud" -> "Point_Cloud".
    stage_1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Step 2: split acronym-to-word boundaries like "IRFFT2" -> "IRFFT_2".
    stage_2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stage_1)
    return stage_2.replace("__", "_").lower()


def _spec_category(spec: type[FunctionSpec]) -> str:
    """Infer functional category from a FunctionSpec module path."""

    module = spec.__module__
    prefix = "physicsnemo.nn.functional."
    if module.startswith(prefix):
        relative = module[len(prefix) :]
        category = relative.split(".", maxsplit=1)[0]
        if category:
            return category
    return "misc"


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Walk nested dict/list containers and yield dictionary nodes."""

    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _latest_result_file(results_dir: Path) -> Path:
    """Return the most recent ASV result JSON file in ``results_dir``."""

    if not results_dir.exists():
        raise FileNotFoundError(f"ASV results directory not found: {results_dir}")

    # Exclude ASV metadata files that do not contain benchmark timing vectors.
    candidates = [
        path
        for path in results_dir.rglob("*.json")
        if path.name not in {"benchmarks.json", "machine.json"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No ASV result JSON files found under: {results_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _benchmark_entry(data: dict[str, Any]) -> Any:
    """Extract the functional benchmark entry from a loaded ASV payload."""

    for mapping in _walk_dicts(data):
        for key, value in mapping.items():
            if isinstance(key, str) and _BENCHMARK_SUFFIX in key:
                return value
    raise KeyError(f"Unable to find benchmark entry for {_BENCHMARK_SUFFIX}")


def _entry_vectors(
    entry: Any, fallback_params: list[BenchmarkKey]
) -> tuple[list[float | None], list[str]]:
    """Normalize the ASV entry payload into ``(values, labels)`` vectors."""

    # ASV may store benchmark vectors directly or under "result"/"results".
    if isinstance(entry, dict):
        entry = entry.get("result", entry.get("results"))

    if not isinstance(entry, list) or not entry:
        raise ValueError("Unexpected ASV benchmark entry format.")

    values = entry[0]
    labels = (
        entry[1]
        if len(entry) > 1
        else [str(param) for param in fallback_params[: len(values)]]
    )
    if labels and isinstance(labels[0], list):
        labels = labels[0]
    return values, labels


def _parse_benchmark_label(label: str) -> BenchmarkKey:
    """Parse ASV benchmark labels from legacy and phase-aware tuple formats."""

    parsed = ast.literal_eval(label)

    # New format: (phase, spec_name, impl_name, case_index).
    if isinstance(parsed, tuple) and len(parsed) == 4:
        phase, spec_name, impl_name, case_index = parsed
        return str(phase), str(spec_name), str(impl_name), int(case_index)

    # Legacy format: (spec_name, impl_name, case_index) => forward-only.
    if isinstance(parsed, tuple) and len(parsed) == 3:
        spec_name, impl_name, case_index = parsed
        return "forward", str(spec_name), str(impl_name), int(case_index)

    raise ValueError(f"Unsupported benchmark label format: {label}")


def _build_spec_data(device: torch.device | str) -> dict[str, BenchmarkSpecData]:
    """Materialize plottable metadata for each FunctionSpec in the registry."""

    spec_data: dict[str, BenchmarkSpecData] = {}

    for spec in FUNCTIONAL_SPECS:
        implementations = tuple(spec.available_implementations())
        # Skip specs with a single backend: these are less informative as bar charts.
        if len(implementations) < 2:
            continue

        labels_by_phase: dict[str, list[str]] = {}
        for phase in PHASE_ORDER:
            labels = case_labels(spec=spec, phase=phase, device=device)
            if labels:
                labels_by_phase[phase] = labels
        if not labels_by_phase:
            continue

        snake_name = _camel_to_snake(spec.__name__)
        category = _spec_category(spec)
        spec_data[spec.__name__] = BenchmarkSpecData(
            slug=f"{category}/{snake_name}",
            implementations=implementations,
            labels_by_phase=labels_by_phase,
        )

    return spec_data


def _build_fallback_params(
    *,
    device: torch.device | str,
    phases: Sequence[str] = PHASE_ORDER,
    selected_specs: Iterable[type[FunctionSpec]] = FUNCTIONAL_SPECS,
) -> list[BenchmarkKey]:
    """Reconstruct ASV key ordering for result payloads missing explicit labels."""

    params, _ = build_benchmark_plan(
        device=device,
        phases=phases,
        selected_specs=selected_specs,
    )
    return params


def _collect_grouped_data(
    values: list[float | None],
    labels: list[str],
    spec_data: dict[str, BenchmarkSpecData],
) -> SpecPhaseMap:
    """Build phase/spec/case/implementation timing maps from ASV vectors."""

    # Initialize all phases so downstream plotting can iterate predictably.
    grouped: SpecPhaseMap = {phase: {} for phase in PHASE_ORDER}

    for label, value in zip(labels, values):
        if value is None:
            continue

        phase, spec_name, implementation, case_index = _parse_benchmark_label(label)
        if phase not in PHASE_ORDER or spec_name not in spec_data:
            continue

        phase_labels = spec_data[spec_name].labels_by_phase.get(phase, [])
        if case_index < 0 or case_index >= len(phase_labels):
            continue

        case_label = phase_labels[case_index]
        grouped[phase].setdefault(spec_name, {}).setdefault(case_label, {})[
            implementation
        ] = float(value)

    return grouped


def _ordered_implementations(case_map: CaseMap) -> list[str]:
    """Return implementation names sorted by canonical display order."""

    implementations = {impl for impl_map in case_map.values() for impl in impl_map}
    return sorted(
        implementations,
        key=lambda name: (_IMPL_ORDER.index(name) if name in _IMPL_ORDER else 99, name),
    )


def _plot_phase_spec(
    phase: str,
    spec_name: str,
    case_map: CaseMap,
    metadata: BenchmarkSpecData,
    output_root: Path,
) -> None:
    """Render one grouped bar plot for one (phase, spec) pair."""

    import matplotlib.pyplot as plt

    # Preserve label ordering from FunctionSpec metadata/generator output.
    case_labels_in_order = [
        label for label in metadata.labels_by_phase.get(phase, []) if label in case_map
    ]
    if not case_labels_in_order:
        return

    # Plot only backend comparisons (at least two implementations present).
    implementations = _ordered_implementations(case_map)
    if len(implementations) < 2:
        return

    # Build output directory under <output_root>/<category>/<functional_name>/.
    output_dir = output_root / metadata.slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grouped-bar figure.
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_width = 0.8 / len(implementations)
    x_positions = list(range(len(case_labels_in_order)))

    # Draw one bar group per case, one color per implementation.
    for impl_index, implementation in enumerate(implementations):
        offsets = [x + impl_index * bar_width for x in x_positions]
        y_values = [
            case_map[label].get(implementation, float("nan"))
            for label in case_labels_in_order
        ]
        ax.bar(
            offsets,
            y_values,
            width=bar_width,
            color=_IMPL_COLORS.get(implementation, _IMPL_COLORS["unknown"]),
            label=implementation,
        )

    # Configure axes and legend.
    tick_positions = [
        x + bar_width * (len(implementations) - 1) / 2 for x in x_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(case_labels_in_order, rotation=20, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{spec_name} {phase.title()} Benchmark", color="#111111")
    ax.grid(axis="y", linestyle=":", color="#E0E0E0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", colors="#111111")
    ax.tick_params(axis="y", colors="#111111")
    ax.legend(
        frameon=False,
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    # Save to canonical benchmark file name for the phase.
    fig.tight_layout()
    output_name = (
        "benchmark_forward.png" if phase == "forward" else f"benchmark_{phase}.png"
    )
    fig.savefig(output_dir / output_name)

    plt.close(fig)


def _plot_all(
    grouped: SpecPhaseMap,
    spec_data: dict[str, BenchmarkSpecData],
    output_root: Path,
) -> None:
    """Render all available benchmark plots from grouped benchmark data."""

    for phase in PHASE_ORDER:
        for spec_name, case_map in grouped[phase].items():
            _plot_phase_spec(
                phase=phase,
                spec_name=spec_name,
                case_map=case_map,
                metadata=spec_data[spec_name],
                output_root=output_root,
            )


def main() -> int:
    """CLI entrypoint for generating benchmark plots from ASV outputs."""

    parser = argparse.ArgumentParser(
        description="Generate functional benchmark bar plots from ASV results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path(".asv/results"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/img/nn/functional"),
        help="Root docs directory for benchmark images.",
    )
    parser.add_argument(
        "--label-device",
        default="cpu",
        help="Device used to resolve make_inputs labels (default: cpu).",
    )
    args = parser.parse_args()

    # Build spec metadata and fallback ASV key ordering.
    spec_data = _build_spec_data(device=args.label_device)
    fallback_params = _build_fallback_params(device=args.label_device)

    # Load the newest ASV result payload and extract benchmark vectors.
    result_file = _latest_result_file(args.results_dir)
    data = json.loads(result_file.read_text())
    entry = _benchmark_entry(data)
    values, labels = _entry_vectors(entry=entry, fallback_params=fallback_params)

    # Group raw vectors by phase/spec/case/implementation, then render plots.
    grouped = _collect_grouped_data(values=values, labels=labels, spec_data=spec_data)
    _plot_all(
        grouped=grouped,
        spec_data=spec_data,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
