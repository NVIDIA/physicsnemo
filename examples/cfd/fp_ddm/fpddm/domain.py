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

"""Domain, field, and overlap data structures for thermal FP-DDM."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto

import torch
import torch.nn.functional as F

from .data import make_layout_fields


class MaskKind(Enum):
    """Boundary-condition mask categories attached to domain fields."""

    DIRICHLET = auto()
    NEUMANN = auto()


class Direction(Enum):
    """Cardinal direction used to identify neighboring subdomains."""

    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

    def opposite(self) -> Direction:
        """Return the direction opposite this one."""

        return {
            Direction.TOP: Direction.BOTTOM,
            Direction.BOTTOM: Direction.TOP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


DIRECTIONS = tuple(Direction)


class Fields(Enum):
    """Thermal fields stored on the global domain and each subdomain."""

    TEMPERATURE = auto()
    CONDUCTIVITY = auto()
    HEAT_SOURCE = auto()
    TEMPERATURE_BC = auto()


INPUT_FIELDS = (Fields.CONDUCTIVITY, Fields.TEMPERATURE_BC, Fields.HEAT_SOURCE)
BC_FIELDS = (Fields.TEMPERATURE_BC,)
OUTPUT_FIELDS = (Fields.TEMPERATURE,)


def _tile_slices(
    row: int, column: int, width: int, height: int, overlap: int
) -> tuple[slice, slice]:
    stride_x = width - overlap
    stride_y = height - overlap
    x0 = column * stride_x
    y0 = row * stride_y
    return slice(y0, y0 + height), slice(x0, x0 + width)


def _crop(value, y_slice: slice, x_slice: slice):
    if torch.is_tensor(value):
        return value[y_slice, x_slice].clone()
    if isinstance(value, dict):
        return {key: _crop(item, y_slice, x_slice) for key, item in value.items()}
    return value


class Subdomain:
    """One overlapping rectangular patch in a decomposed thermal domain."""

    def __init__(
        self, row: int, column: int, width: int, height: int, overlap: int
    ) -> None:
        """Create an empty subdomain and its overlap indexing metadata."""

        self.row = row
        self.column = column
        self.width = width
        self.height = height
        self.overlap = overlap
        self.neighbors: dict[Direction, Subdomain | None] = {
            direction: None for direction in DIRECTIONS
        }
        self.optimizer: torch.optim.Optimizer | None = None
        self.fields: dict[Fields, torch.Tensor] = {}
        self.masks: dict[Fields, dict[MaskKind, torch.Tensor]] = {}

        full = slice(None)
        self.boundary_slices = {
            Direction.LEFT: (full, 0),
            Direction.RIGHT: (full, -1),
            Direction.TOP: (0, full),
            Direction.BOTTOM: (-1, full),
        }
        self.overlap_slices = {
            Direction.LEFT: (full, overlap - 1),
            Direction.RIGHT: (full, -overlap),
            Direction.TOP: (overlap - 1, full),
            Direction.BOTTOM: (-overlap, full),
        }

    def __repr__(self) -> str:
        """Return a compact grid-coordinate representation."""

        return f"Subdomain({self.row},{self.column})"


class Domain:
    """Global thermal domain partitioned into overlapping rectangular patches."""

    def __init__(
        self, rows: int, columns: int, width: int, height: int, overlap: int
    ) -> None:
        """Initialize an empty global domain with the requested decomposition."""

        if overlap < 1 or overlap >= min(width, height):
            raise ValueError("overlap must be positive and smaller than each patch")
        self.rows = rows
        self.columns = columns
        self.width = width
        self.height = height
        self.overlap = overlap
        self.total_width = (columns - 1) * (width - overlap) + width
        self.total_height = (rows - 1) * (height - overlap) + height
        self.fields: dict[Fields, torch.Tensor] = {}
        self.masks: dict[Fields, dict[MaskKind, torch.Tensor]] = {}

    def build_subdomains(self) -> list[list[Subdomain]]:
        """Crop global fields into an interconnected subdomain grid."""

        grid = [
            [
                Subdomain(row, column, self.width, self.height, self.overlap)
                for column in range(self.columns)
            ]
            for row in range(self.rows)
        ]
        for row in range(self.rows):
            for column in range(self.columns):
                subdomain = grid[row][column]
                y_slice, x_slice = _tile_slices(
                    row, column, self.width, self.height, self.overlap
                )
                subdomain.fields = {
                    field: _crop(value, y_slice, x_slice)
                    for field, value in self.fields.items()
                }
                subdomain.masks = {
                    field: _crop(value, y_slice, x_slice)
                    for field, value in self.masks.items()
                }
                if Fields.TEMPERATURE_BC in subdomain.fields:
                    subdomain.fields[Fields.TEMPERATURE_BC] = torch.nn.Parameter(
                        subdomain.fields[Fields.TEMPERATURE_BC]
                    )

                if column > 0:
                    subdomain.neighbors[Direction.LEFT] = grid[row][column - 1]
                if column + 1 < self.columns:
                    subdomain.neighbors[Direction.RIGHT] = grid[row][column + 1]
                if row > 0:
                    subdomain.neighbors[Direction.TOP] = grid[row - 1][column]
                if row + 1 < self.rows:
                    subdomain.neighbors[Direction.BOTTOM] = grid[row + 1][column]
        return grid


def assemble_avg(grid: list[list[Subdomain]], field: Fields) -> torch.Tensor:
    """Assemble a global field by averaging values in overlapping cells."""

    rows, columns = len(grid), len(grid[0])
    first = grid[0][0]
    total_height = (rows - 1) * (first.height - first.overlap) + first.height
    total_width = (columns - 1) * (first.width - first.overlap) + first.width
    sample = first.fields[field]
    accumulated = torch.zeros(
        (total_height, total_width), device=sample.device, dtype=sample.dtype
    )
    counts = torch.zeros(
        (total_height, total_width), device=sample.device, dtype=torch.int32
    )
    for row in range(rows):
        for column in range(columns):
            y_slice, x_slice = _tile_slices(
                row, column, first.width, first.height, first.overlap
            )
            accumulated[y_slice, x_slice] += grid[row][column].fields[field]
            counts[y_slice, x_slice] += 1
    return accumulated / counts.clamp_min_(1)


def _temperature_from_boundary(
    boundary: Mapping[str, Mapping[str, object]],
    height: int,
    width: int,
    *,
    fill: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    top = float(boundary["top"]["value"])
    bottom = float(boundary["bottom"]["value"])
    left = float(boundary["left"]["value"])
    right = float(boundary["right"]["value"])
    if fill is None:
        values = torch.empty(1, 1, height, width, device=device).uniform_(
            min(top, bottom, left, right), max(top, bottom, left, right)
        )
        values = F.avg_pool2d(values, kernel_size=3, stride=1, padding=1)[0, 0]
    else:
        values = fill.clone().to(device)
    values[0] = top
    values[-1] = bottom
    values[:, 0] = left
    values[:, -1] = right
    return values


def initialize_thermal_fields(
    domain: Domain,
    boundary: Mapping[str, Mapping[str, object]],
    layout_config: Mapping[str, object],
    *,
    interior_temperature: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> None:
    """Populate a domain with temperature, conductivity, and source fields."""

    temperature = _temperature_from_boundary(
        boundary,
        domain.total_height,
        domain.total_width,
        fill=interior_temperature,
        device=device,
    )
    domain.fields[Fields.TEMPERATURE_BC] = temperature.clone()
    domain.fields[Fields.TEMPERATURE] = temperature.clone()

    if interior_temperature is not None and Fields.CONDUCTIVITY in domain.fields:
        return

    field_config = dict(layout_config)
    field_config.update(grid_size=domain.total_height, k_min=0.5)
    conductivity, heat_source, _ = make_layout_fields(field_config)
    if conductivity.shape != (domain.total_height, domain.total_width):
        raise ValueError("FP-DDM currently requires a square global domain")

    domain.fields[Fields.CONDUCTIVITY] = conductivity.to(device)
    domain.fields[Fields.HEAT_SOURCE] = heat_source.to(device)

    dirichlet = torch.ones(
        (domain.total_height, domain.total_width), dtype=torch.bool, device=device
    )
    dirichlet[1:-1, 1:-1] = False
    domain.masks[Fields.TEMPERATURE_BC] = {MaskKind.DIRICHLET: dirichlet}
