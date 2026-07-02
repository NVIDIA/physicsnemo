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

"""Metrics and optional visual outputs for FP-DDM Schwarz iteration."""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .domain import Fields, Subdomain, assemble_avg


def visualize_array(
    array: np.ndarray | torch.Tensor,
    output_path: str | Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    clean: bool = False,
) -> None:
    """Render a scalar array to a PNG file."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = (
        array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(values, origin="upper", cmap="jet", vmin=vmin, vmax=vmax)
    if clean:
        axis.set_axis_off()
    else:
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=clean,
    )
    plt.close(figure)


def plot_heat_flux(
    conductivity: torch.Tensor,
    temperature: torch.Tensor,
    output_path: str | Path,
    *,
    trim: int = 1,
) -> None:
    """Render the log-magnitude of heat flux for a solved thermal field."""

    conductivity_np = conductivity.detach().cpu().numpy()
    temperature_np = temperature.detach().cpu().numpy()
    if trim:
        conductivity_np = conductivity_np[trim:-trim, trim:-trim]
        temperature_np = temperature_np[trim:-trim, trim:-trim]
    grad_y, grad_x = np.gradient(temperature_np, edge_order=2)
    magnitude = np.hypot(
        -conductivity_np * grad_x,
        -conductivity_np * grad_y,
    )
    low, high = np.percentile(magnitude, [1, 99])
    magnitude = np.clip(magnitude, max(low, 1.0e-12), high)
    visualize_array(np.log(magnitude), output_path, clean=True)


class AttributeVisualizer:
    """Record a global assembled field as PNG, NumPy series, and MP4."""

    def __init__(
        self,
        field: Fields,
        save_dir: str | Path,
        *,
        save_name: str,
        fps: int = 20,
        vmin: float | None = None,
        vmax: float | None = None,
        max_size: tuple[int, int] = (608, 512),
    ) -> None:
        """Configure field extraction and output rendering."""

        self.field = field
        self.file_path = Path(save_dir) / save_name
        self.fps = fps
        self.vmin = vmin
        self.vmax = vmax
        self.max_size = max_size
        self.frames: list[np.ndarray] = []
        self.series: list[np.ndarray] = []
        self.iteration = 0
        self.global_output: torch.Tensor | None = None

    def __call__(self, grid: list[list[Subdomain]]) -> None:
        """Capture the configured field for one Schwarz iteration."""

        import imageio.v3
        import matplotlib.pyplot as plt

        self.global_output = assemble_avg(grid, self.field).detach().cpu().float()
        self.series.append(self.global_output.numpy())
        if self.iteration == 0:
            visualize_array(
                self.global_output,
                self.file_path.with_name(self.file_path.name + "_init.png"),
                vmin=self.vmin,
                vmax=self.vmax,
            )
        visualize_array(
            self.global_output,
            self.file_path.with_name(self.file_path.name + "_current.png"),
            vmin=self.vmin,
            vmax=self.vmax,
        )
        np.save(
            self.file_path.with_name(self.file_path.name + "_current.npy"),
            self.global_output.numpy(),
        )

        figure, axis = plt.subplots(figsize=(6.08, 5.12))
        image = axis.imshow(
            self.global_output,
            origin="upper",
            vmin=self.vmin,
            vmax=self.vmax,
            cmap="jet",
        )
        axis.set_title(str(self.iteration))
        figure.colorbar(image, ax=axis)
        buffer = io.BytesIO()
        width, height = figure.get_size_inches()
        dpi = int(
            100
            * min(
                1.0,
                self.max_size[0] / (width * 100),
                self.max_size[1] / (height * 100),
            )
        )
        figure.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        self.frames.append(imageio.v3.imread(buffer))
        plt.close(figure)
        self.iteration += 1

    def finalize(self) -> None:
        """Write final images, the full series, and an MP4 animation."""

        if self.global_output is None:
            return
        import imageio

        with imageio.get_writer(
            self.file_path.with_suffix(".mp4"),
            fps=self.fps,
            codec="libx264",
            quality=9,
        ) as writer:
            for frame in self.frames:
                writer.append_data(frame)
        visualize_array(
            self.global_output,
            self.file_path.with_name(self.file_path.name + "_post.png"),
            vmin=self.vmin,
            vmax=self.vmax,
        )
        np.save(
            self.file_path.with_name(self.file_path.name + "_series.npy"),
            np.stack(self.series, axis=-1),
        )


class MetricsLogger:
    """Collect handler metrics and append one CSV row per Schwarz iteration."""

    def __init__(self, csv_path: str | Path) -> None:
        """Initialize the output path and elapsed-time origin."""

        self.csv_path = Path(csv_path)
        self.metric_sources: list[Callable[[], tuple[str, float]]] = []
        self.started = time.perf_counter()
        self.iteration = 0
        self.rows: list[dict[str, float | int]] = []

    def register_metric(self, source) -> bool:
        """Register an object exposing a callable ``get_metric`` method."""

        getter = getattr(source, "get_metric", None)
        if not callable(getter):
            return False
        self.metric_sources.append(getter)
        return True

    def step(self) -> dict[str, float | int]:
        """Collect current metrics, persist them, and return the new row."""

        self.iteration += 1
        row: dict[str, float | int] = {
            "iteration": self.iteration,
            "elapsed": time.perf_counter() - self.started,
        }
        for getter in self.metric_sources:
            name, value = getter()
            row[name] = float(value)
        self.rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.rows[-1]))
            writer.writeheader()
            writer.writerows(self.rows)
        return row

    def finalize(self) -> list[dict[str, float | int]]:
        """Return all recorded metric rows."""

        return self.rows


class R2Metric:
    """Measure coefficient of determination against a reference solution."""

    def __init__(self, reference: torch.Tensor, field: Fields) -> None:
        """Store the reference solution and assembled field identifier."""

        self.reference = reference.detach().cpu().flatten()
        self.field = field
        self.value = float("nan")

    def __call__(self, grid: list[list[Subdomain]]) -> float:
        """Update and return the current R-squared value."""

        prediction = assemble_avg(grid, self.field).detach().cpu().flatten()
        residual = (self.reference - prediction).square().sum()
        total = (self.reference - self.reference.mean()).square().sum()
        self.value = float(1.0 - residual / total.clamp_min(1.0e-12))
        return self.value

    def get_metric(self) -> tuple[str, float]:
        """Return the latest R-squared metric."""

        return "loss_true_R2", self.value


class NRMAMetric:
    """Measure range-normalized mean absolute error against a reference."""

    def __init__(self, reference: torch.Tensor, field: Fields) -> None:
        """Store the reference solution and assembled field identifier."""

        self.reference = reference.detach().cpu().flatten()
        self.field = field
        self.value = float("nan")

    def __call__(self, grid: list[list[Subdomain]]) -> float:
        """Update and return the current normalized absolute error."""

        prediction = assemble_avg(grid, self.field).detach().cpu().flatten()
        error = (prediction - self.reference).abs().mean()
        data_range = (self.reference.max() - self.reference.min()).abs()
        self.value = float(error / data_range.clamp_min(1.0e-8))
        return self.value

    def get_metric(self) -> tuple[str, float]:
        """Return the latest normalized mean absolute error."""

        return "loss_true_NRMAE", self.value
