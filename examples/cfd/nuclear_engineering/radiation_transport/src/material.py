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

"""Material/physics-domain module for radiation-transport surrogates.

Consolidates two concerns into one file:

1. Pure-numpy *material mappers* that, given an (x, y) point cloud, assign a
   discrete material label and the corresponding cross-section properties
   (sigma_a, sigma_s, sigma_t, Q) for the two benchmark geometries:

   - ``LatticeMaterialMapper``: 7x7 grid of square blocks in
     [-3.5, 3.5] x [-3.5, 3.5] (blue / red / white).
   - ``HohlraumMaterialMapper``: complex hohlraum geometry in
     [-0.65, 0.65] x [-0.65, 0.65] (black / red / green / blue / white).

2. The ``MaterialPropertyExtractor`` *transform* that runs as part of the
   per-sample pipeline. It uses precomputed sigma fields when the Zarr store
   provides them, otherwise it falls back to invoking the mappers above on
   the integer ``material_properties`` labels stored in the sample.

A separate stats-computation utility (``compute_material_statistics``) is
used by ``compute_normalizations.py`` for offline normalization stats; it
is not a transform and lives outside this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict

from transforms import Transform, td_get, to_numpy


# =========================================================================
# Lattice material mapper
# =========================================================================


class LatticeMaterialMapper:
    """Maps spatial points to material properties for the lattice dataset.

    Domain: 7x7 grid of square blocks in [-3.5, 3.5] x [-3.5, 3.5]
    - Blue blocks (11): pure absorption (sigma_a only)
    - Red block (1):    scattering source (sigma_s + Q=1)
    - White blocks (37): pure scattering (sigma_s only)
    """

    # Domain parameters
    DOMAIN_BOUNDS = (-3.5, 3.5)
    BLOCK_SIZE = 1.0
    NUM_BLOCKS = 7

    # Material region definitions
    BLUE_BLOCKS = [
        (2, 2),
        (4, 2),
        (6, 2),
        (3, 3),
        (5, 3),
        (2, 4),
        (6, 4),
        (3, 5),
        (5, 5),
        (2, 6),
        (6, 6),
    ]
    RED_BLOCKS = [(4, 4)]

    # Material labels
    MATERIAL_LABELS = {"blue": 0, "red": 1, "white": 2}

    # Default material properties
    DEFAULT_MATERIAL_PROPERTIES = {
        0: {"sigma_t": None, "sigma_s": 0.0, "sigma_a": None, "Q": 0},  # blue
        1: {"sigma_t": None, "sigma_s": None, "sigma_a": 0.0, "Q": 1},  # red
        2: {"sigma_t": None, "sigma_s": None, "sigma_a": 0.0, "Q": 0},  # white
        -1: {"sigma_t": 0, "sigma_s": 0, "sigma_a": 0, "Q": 0},  # outside
    }

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        simulation_parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the lattice material mapper."""
        self.logger = logger or logging.getLogger(__name__)
        self.simulation_parameters = simulation_parameters or {}
        self._material_properties = self._calculate_material_properties()

    def _calculate_material_properties(self) -> Dict[int, Dict[str, float]]:
        """Calculate material properties based on simulation parameters."""
        absorption_coeff = self.simulation_parameters.get("absorption_coeff", np.nan)
        scattering_coeff = self.simulation_parameters.get("scattering_coeff", np.nan)

        # Blue: pure absorption
        blue_props = {
            "sigma_a": absorption_coeff,
            "sigma_s": 0.0,
            "sigma_t": absorption_coeff,
            "Q": 0,
        }

        # Red: scattering source - TODO: should be pure scattering.
        red_props = {
            "sigma_a": 0.0,
            "sigma_s": scattering_coeff,
            "sigma_t": 1.0,
            "Q": 1,
        }

        # White: pure scattering
        white_props = {
            "sigma_a": 0.0,
            "sigma_s": scattering_coeff,
            "sigma_t": scattering_coeff,
            "Q": 0,
        }

        # Outside domain
        outside_props = {"sigma_a": 0, "sigma_s": 0, "sigma_t": 0, "Q": 0}

        return {0: blue_props, 1: red_props, 2: white_props, -1: outside_props}

    def get_block_index(self, x: float, y: float) -> tuple[int, int]:
        """Convert x,y coordinates to block indices (1-indexed)."""
        # Shift coordinates to [0, 7] range
        x_shifted = x - self.DOMAIN_BOUNDS[0]
        y_shifted = y - self.DOMAIN_BOUNDS[0]

        # Get block indices (1-indexed)
        i = int(np.floor(x_shifted / self.BLOCK_SIZE)) + 1
        j = int(np.floor(y_shifted / self.BLOCK_SIZE)) + 1

        # Clamp to valid range [1, 7]
        i = max(1, min(self.NUM_BLOCKS, i))
        j = max(1, min(self.NUM_BLOCKS, j))

        return i, j

    def map_coordinates_to_materials(self, coordinates: np.ndarray) -> np.ndarray:
        """Map coordinates to material labels."""
        n_points = coordinates.shape[0]
        material_labels = np.full(n_points, 2, dtype=np.int32)  # Default: white

        for idx in range(n_points):
            x, y = coordinates[idx, 0], coordinates[idx, 1]
            i, j = self.get_block_index(x, y)
            block = (i, j)

            if block in self.BLUE_BLOCKS:
                material_labels[idx] = 0
            elif block in self.RED_BLOCKS:
                material_labels[idx] = 1
            # else: stays white (2)

        return material_labels

    def get_material_properties(self, coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """Get material property arrays for coordinates."""
        material_labels = self.map_coordinates_to_materials(coordinates)
        n_points = len(coordinates)

        # Initialize arrays
        sigma_t = np.zeros(n_points, dtype=np.float32)
        sigma_s = np.zeros(n_points, dtype=np.float32)
        sigma_a = np.zeros(n_points, dtype=np.float32)
        Q = np.zeros(n_points, dtype=np.float32)

        # Fill arrays based on material labels
        for label in [0, 1, 2]:
            mask = material_labels == label
            props = self._material_properties[label]
            sigma_t[mask] = props["sigma_t"]
            sigma_s[mask] = props["sigma_s"]
            sigma_a[mask] = props["sigma_a"]
            Q[mask] = props["Q"]

        return {"sigma_t": sigma_t, "sigma_s": sigma_s, "sigma_a": sigma_a, "Q": Q}


# =========================================================================
# Hohlraum material mapper
# =========================================================================


class HohlraumMaterialMapper:
    """Maps spatial points to material properties for the hohlraum dataset.

    Domain: [-0.65, 0.65] x [-0.65, 0.65] with complex geometric regions
    - Black (0): top/bottom horizontal strips
    - Red   (1): left/right vertical strips
    - Green (2): capsule frame
    - Blue  (3): central capsule interior
    - White (4): background region
    """

    # Domain parameters
    DOMAIN_BOUNDS = (-0.65, 0.65)

    # Material labels
    MATERIAL_LABELS = {"black": 0, "red": 1, "green": 2, "blue": 3, "white": 4}

    # Material properties (fixed values)
    MATERIAL_PROPERTIES = {
        0: {"sigma_t": 100, "sigma_s": 50, "sigma_a": 50, "Q": 0},  # black
        1: {"sigma_t": 100, "sigma_s": 95, "sigma_a": 5, "Q": 0},  # red
        2: {"sigma_t": 100, "sigma_s": 90, "sigma_a": 10, "Q": 0},  # green
        3: {"sigma_t": 100, "sigma_s": 0, "sigma_a": 100, "Q": 0},  # blue
        4: {"sigma_t": 0.1, "sigma_s": 0.1, "sigma_a": 0, "Q": 0},  # white
        -1: {"sigma_t": 0, "sigma_s": 0, "sigma_a": 0, "Q": 0},  # outside
    }

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        boundary_thickness: float = 0.05,  # Green frame thickness
        simulation_parameters: Optional[Dict[str, Any]] = None,
        capsule_half_width: float = 0.15,  # Blue inner capsule half-width
        capsule_half_height: float = 0.35,  # Blue inner capsule half-height
    ):
        """Initialize the hohlraum material mapper."""
        self.logger = logger or logging.getLogger(__name__)
        self.boundary_thickness = boundary_thickness
        self.simulation_parameters = simulation_parameters or {}
        self.capsule_half_width = capsule_half_width
        self.capsule_half_height = capsule_half_height
        self._material_regions = self._calculate_material_regions()

    def _calculate_material_regions(self) -> Dict:
        """Calculate material regions based on simulation parameters."""
        # Get design parameters
        ulr = self.simulation_parameters.get("ulr", 0.4)
        llr = self.simulation_parameters.get("llr", -0.4)
        urr = self.simulation_parameters.get("urr", 0.4)
        lrr = self.simulation_parameters.get("lrr", -0.4)
        hlr = self.simulation_parameters.get("hlr", -0.5)
        hrr = self.simulation_parameters.get("hrr", 0.5)
        cx = self.simulation_parameters.get("cx", 0.0)
        cy = self.simulation_parameters.get("cy", 0.0)

        regions = {}

        # Black: top/bottom horizontal strips (wide regions, not thin borders)
        # Black regions are defined as y > 0.6 or y < -0.6
        # This matches the KiT-RT QoI calculation logic (line 619)
        regions["black"] = [
            {
                "name": "K1",
                "bounds": (
                    self.DOMAIN_BOUNDS[0],
                    self.DOMAIN_BOUNDS[1],
                    self.DOMAIN_BOUNDS[0],
                    -0.6,  # Bottom: y < -0.6
                ),
            },
            {
                "name": "K2",
                "bounds": (
                    self.DOMAIN_BOUNDS[0],
                    self.DOMAIN_BOUNDS[1],
                    0.6,  # Top: y > 0.6
                    self.DOMAIN_BOUNDS[1],
                ),
            },
        ]

        # Red: left/right vertical strips (wide regions, not thin borders)
        # Red regions are defined as everything left of hlr and right of hrr
        # This matches the KiT-RT QoI calculation logic and the physical hohlraum geometry
        regions["red"] = [
            {
                "name": "R1",
                "bounds": (self.DOMAIN_BOUNDS[0], hlr, llr, ulr),
            },  # Left: x < hlr
            {
                "name": "R2",
                "bounds": (hrr, self.DOMAIN_BOUNDS[1], lrr, urr),
            },  # Right: x > hrr
        ]

        # Green: capsule outer box (KiT-RT assigns green to entire outer box, then overwrites with blue)
        # This matches KiT-RT lines 79-83: x in [-0.2+cx, 0.2+cx] && y in [-0.4+cy, 0.4+cy]
        # The frame thickness is implicit: outer_width/2 - inner_width/2 = 0.2 - 0.15 = 0.05
        x_min_outer = cx - 0.2
        x_max_outer = cx + 0.2
        y_min_outer = cy - 0.4
        y_max_outer = cy + 0.4

        regions["green"] = [
            {
                "name": "G_outer",
                "bounds": (x_min_outer, x_max_outer, y_min_outer, y_max_outer),
            },
        ]

        # Blue: central capsule interior (checkered area)
        # KiT-RT lines 84-88: x in [-0.15+cx, 0.15+cx] && y in [-0.35+cy, 0.35+cy]
        x_min_blue = cx - 0.15
        x_max_blue = cx + 0.15
        y_min_blue = cy - 0.35
        y_max_blue = cy + 0.35

        regions["blue"] = [
            {
                "name": "B",
                "bounds": (x_min_blue, x_max_blue, y_min_blue, y_max_blue),
            },
        ]

        return regions

    def _is_in_blue_region(self, x: float, y: float) -> bool:
        """Check if point is in blue capsule region."""
        for region in self._material_regions.get("blue", []):
            x_min, x_max, y_min, y_max = region["bounds"]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _in_rect(self, x: float, y: float, bounds: tuple) -> bool:
        """Check if point is in rectangle."""
        x_min, x_max, y_min, y_max = bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def get_material_property(self, x: float, y: float) -> int:
        """Get material label for a single point.

        Uses KiT-RT's exact material assignment logic:
        1. Check black (top/bottom boundary regions)
        2. Check red (left/right vertical regions)
        3. Check green outer box (entire capsule region)
        4. Overwrite with blue if in inner region
        5. Default to white (background)
        """
        # Priority order: black > red > green+blue > white

        # Check black (top/bottom strips) - highest priority
        for region in self._material_regions.get("black", []):
            if self._in_rect(x, y, region["bounds"]):
                return 0

        # Check red (left/right strips)
        for region in self._material_regions.get("red", []):
            if self._in_rect(x, y, region["bounds"]):
                return 1

        # Check green outer box (entire capsule region including corners)
        # This assigns green to the FULL outer box first
        for region in self._material_regions.get("green", []):
            if self._in_rect(x, y, region["bounds"]):
                # Now check if this point should be overwritten with blue (inner region)
                if self._is_in_blue_region(x, y):
                    return 3  # blue overwrites green
                return 2  # green

        # Default: white (background)
        return 4

    def map_coordinates_to_materials(self, coordinates: np.ndarray) -> np.ndarray:
        """Map coordinates to material labels."""
        n_points = coordinates.shape[0]
        material_labels = np.zeros(n_points, dtype=np.int32)

        for idx in range(n_points):
            x, y = coordinates[idx, 0], coordinates[idx, 1]
            material_labels[idx] = self.get_material_property(x, y)

        return material_labels

    def get_material_properties(self, coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """Get material property arrays for coordinates."""
        material_labels = self.map_coordinates_to_materials(coordinates)
        n_points = len(coordinates)

        # Initialize arrays
        sigma_t = np.zeros(n_points, dtype=np.float32)
        sigma_s = np.zeros(n_points, dtype=np.float32)
        sigma_a = np.zeros(n_points, dtype=np.float32)
        Q = np.zeros(n_points, dtype=np.float32)

        # Fill arrays based on material labels
        for label in range(5):
            mask = material_labels == label
            props = self.MATERIAL_PROPERTIES[label]
            sigma_t[mask] = props["sigma_t"]
            sigma_s[mask] = props["sigma_s"]
            sigma_a[mask] = props["sigma_a"]
            Q[mask] = props["Q"]

        return {"sigma_t": sigma_t, "sigma_s": sigma_s, "sigma_a": sigma_a, "Q": Q}


# =========================================================================
# Material transforms
# =========================================================================


class MaterialPropertyExtractor(Transform):
    """Extract physical material properties for radiation transport.

    Uses precomputed sigma fields (``sigma_a``, ``sigma_s``, ``sigma_t``, ``Q``)
    when present in the sample; otherwise falls back to computing them from the
    integer ``material_properties`` labels via the lattice/hohlraum mappers
    defined above. The extracted properties are stored as
    ``physical_properties`` with shape ``(N, 4)``: ``[sigma_a, sigma_s, sigma_t, Q]``.
    """

    def __init__(self, case_type: Optional[str] = None, add_to_sample: bool = True):
        super().__init__()
        self.case_type = case_type
        self.add_to_sample = add_to_sample

    def __call__(self, data: TensorDict) -> TensorDict:
        has_sigma_a = "sigma_a" in data
        has_sigma_s = "sigma_s" in data
        has_sigma_t = "sigma_t" in data
        has_Q = "Q" in data

        if has_sigma_a and has_sigma_s and has_sigma_t:
            if not has_Q:
                raise KeyError(
                    "Zarr store has precomputed sigma fields but is missing 'Q'. "
                    "All four fields (sigma_a, sigma_s, sigma_t, Q) are required."
                )
            physical_props = torch.stack(
                [data["sigma_a"], data["sigma_s"], data["sigma_t"], data["Q"]],
                dim=-1,
            ).to(dtype=torch.float32)
        else:
            if "material_properties" not in data:
                raise KeyError(
                    "Sample must contain either precomputed sigma fields "
                    "(sigma_a, sigma_s, sigma_t) or 'material_properties' labels."
                )

            case_type = self.case_type
            if case_type is None:
                metadata = td_get(data, "metadata", default={}) or {}
                case_type = (
                    metadata.get("case_type", "") if isinstance(metadata, dict) else ""
                )
            case_type = case_type.lower()

            material_labels = to_numpy(data["material_properties"])
            n_points = len(material_labels)

            if case_type == "lattice":
                metadata = td_get(data, "metadata", default={}) or {}
                sim_params = (
                    metadata.get("simulation_params", {})
                    if isinstance(metadata, dict)
                    else {}
                )
                params = (
                    sim_params.get("parameters", {})
                    if isinstance(sim_params, dict)
                    else {}
                )
                absorption_coeff = params.get("absorption_coeff")
                scattering_coeff = params.get("scattering_coeff")
                if absorption_coeff is None or scattering_coeff is None:
                    raise ValueError(
                        "Lattice case requires 'absorption_coeff' and 'scattering_coeff' "
                        "in metadata.simulation_params.parameters"
                    )

                mapper = LatticeMaterialMapper(
                    simulation_parameters={
                        "absorption_coeff": absorption_coeff,
                        "scattering_coeff": scattering_coeff,
                    }
                )
                material_props = mapper._material_properties

                props_np = np.zeros((n_points, 4), dtype=np.float32)
                for label in (0, 1, 2):
                    mask = material_labels == label
                    props = material_props[label]
                    props_np[mask, 0] = props["sigma_a"]
                    props_np[mask, 1] = props["sigma_s"]
                    props_np[mask, 2] = props["sigma_t"]
                    props_np[mask, 3] = props["Q"]

            elif case_type == "hohlraum":
                material_props = HohlraumMaterialMapper.MATERIAL_PROPERTIES
                props_np = np.zeros((n_points, 4), dtype=np.float32)
                for label in (0, 1, 2, 3, 4):
                    mask = material_labels == label
                    props = material_props[label]
                    props_np[mask, 0] = props["sigma_a"]
                    props_np[mask, 1] = props["sigma_s"]
                    props_np[mask, 2] = props["sigma_t"]
                    props_np[mask, 3] = props["Q"]
            else:
                raise ValueError(
                    f"Unknown case_type: {case_type}. Must be 'lattice' or 'hohlraum'"
                )

            physical_props = torch.from_numpy(props_np)

        if self.add_to_sample:
            data["physical_properties"] = physical_props

        return data

    def extra_repr(self) -> str:
        return f"case_type={self.case_type}, add_to_sample={self.add_to_sample}"


__all__ = [
    "LatticeMaterialMapper",
    "HohlraumMaterialMapper",
    "MaterialPropertyExtractor",
]
