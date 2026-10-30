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

"""AB-UPT adapter and bounded, independently sampled DomainMesh reader.

Only the paper and PhysicsNeMo APIs were used for this integration. Sampling
streams are independent of query counts, access order, worker count, and rank.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from output_normalize import split_concat_by_target
from tensordict import TensorDict

from physicsnemo import Module
from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.datapipes.readers import DomainMeshReader
from physicsnemo.experimental.models.ab_upt import ABUPT
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures

SURFACE_TARGETS = {"surface.pressure": "scalar", "surface.wss": "vector"}
VOLUME_TARGETS = {
    "volume.velocity": "vector",
    "volume.pressure": "scalar",
    "volume.nut": "scalar",
}


def _modes(mode: str) -> tuple[bool, bool]:
    """Validate which branches are present and supervised."""
    if mode not in ("surface", "volume", "joint"):
        raise ValueError("mode must be 'surface', 'volume', or 'joint'")
    return mode != "volume", mode != "surface"


class ABUPTRecipe(Module):
    """Map the independent AB-UPT operator to named recipe targets.

    A context is built once per forward. Decoder chunks share it, including
    during training; chunking does not detach gradients or resample anchors.
    Smaller chunks bound individual attention operations, not the full
    training autograd graph. This DrivAerML adapter predicts velocity,
    pressure, and eddy viscosity in the volume, rather than the paper's
    velocity/pressure/vorticity channel layout.

    ``train_on_anchors`` uses the paper's anchor-supervised training path.
    Training queries must then equal the anchors, in the same order, so the
    returned predictions remain aligned with recipe targets. Evaluation
    always supports independent queries. ``parameter_scale`` divides the
    freestream velocity supplied to the operator without changing metadata
    used for nondimensionalization or force calculations.
    ``pooling_message_fp32`` computes geometry messages with FP32 model weights
    outside autocast; attention continues to follow the recipe precision.
    """

    def __init__(
        self,
        mode: str = "joint",
        query_chunk_size: int = 4096,
        coordinate_scale: float = 1000.0,
        hidden_dim: int = 192,
        num_heads: int = 3,
        num_interaction_blocks: int = 2,
        num_shared_layers: int = 1,
        num_decoder_layers: int = 6,
        pooling_radius: float = 0.25,
        pooling_chunk_size: int = 4096,
        radius_search_backend: str | None = None,
        train_on_anchors: bool = False,
        parameter_scale: float = 1.0,
        pooling_message_fp32: bool = False,
    ):
        super().__init__()
        surface, volume = _modes(mode)
        if (
            isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        if not isinstance(train_on_anchors, bool):
            raise ValueError("train_on_anchors must be a boolean")
        if (
            isinstance(parameter_scale, bool)
            or not math.isfinite(parameter_scale)
            or parameter_scale <= 0
        ):
            raise ValueError("parameter_scale must be finite and positive")
        self.coordinate_scale = coordinate_scale
        self.train_on_anchors = train_on_anchors
        self.parameter_scale = parameter_scale
        self.mode = mode
        self.query_chunk_size = query_chunk_size
        self.operator = ABUPT(
            surface_channels=4 if surface else 0,
            volume_channels=5 if volume else 0,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_interaction_blocks=num_interaction_blocks,
            num_shared_layers=num_shared_layers,
            num_decoder_layers=num_decoder_layers,
            pooling_radius=pooling_radius * coordinate_scale,
            pooling_chunk_size=pooling_chunk_size,
            pooling_message_fp32=pooling_message_fp32,
            parameter_channels=3,
            radius_search_backend=radius_search_backend,
        )
        self.meta = replace(self.operator.meta, name="ABUPTRecipe")

    def get_extra_state(self):
        """Persist settings that change predictions but not weight shapes."""
        return {
            "version": 2,
            "coordinate_scale": self.coordinate_scale,
            "train_on_anchors": self.train_on_anchors,
            "parameter_scale": self.parameter_scale,
            "pooling_radius": self.operator.pooling_radius,
            "num_heads": self.operator.num_heads,
            "cross_branch": list(self.operator.cross_branch),
        }

    def set_extra_state(self, state):
        """Reject numerically incompatible resume/inference configurations."""
        expected = self.get_extra_state()
        if isinstance(state, dict) and state.get("version") == 1:
            # Older checkpoints used only independent-query training and raw
            # freestream velocity. Retain those semantics when loading them.
            state = dict(state)
            state.update(version=2, train_on_anchors=False, parameter_scale=1.0)
        if not isinstance(state, dict) or state != expected:
            raise ValueError(
                "AB-UPT checkpoint settings do not match the current model: "
                f"checkpoint={state!r}, current={expected!r}. "
                "Use the model settings from the training run's resolved_config.yaml. "
                "Query/pooling chunk sizes, pooling message precision, and search "
                "backend may be changed."
            )

    def forward(
        self,
        geometry,
        supernodes,
        surface_anchors=None,
        volume_anchors=None,
        parameters=None,
        surface_queries=None,
        volume_queries=None,
    ):
        """Return named batched fields; surface and volume lengths may differ."""
        branches = (
            (
                "surface",
                self.operator.surface_channels,
                surface_queries,
                SURFACE_TARGETS,
            ),
            ("volume", self.operator.volume_channels, volume_queries, VOLUME_TARGETS),
        )
        # Reject invalid branch contracts before building the geometry graph.
        anchors = {"surface": surface_anchors, "volume": volume_anchors}
        for field, enabled, queries, _ in branches:
            if enabled and queries is None:
                raise ValueError(f"Explicit {field}_queries are required by the recipe")
            if not enabled and queries is not None:
                raise ValueError(f"Queries supplied for disabled {field} head")
            if enabled and self.training and self.train_on_anchors:
                positions = anchors[field]
                if (
                    positions is None
                    or queries.shape != positions.shape
                    or not torch.equal(queries, positions)
                ):
                    raise ValueError(
                        f"Anchor-supervised training requires {field}_queries to "
                        f"equal {field}_anchors in shape, coordinates, and order"
                    )
        if parameters is None:
            raise ValueError("Freestream parameters are required by the recipe")
        if parameters.ndim == 3 and parameters.shape[1] == 1:
            parameters = parameters[:, 0]
        # Native mesh coordinates and metadata can use different floating
        # dtypes. Cast model inputs without mutating the meshes used for export;
        # Tensor.to preserves gradients through any required conversion.
        model_dtype = next(self.operator.parameters()).dtype
        # Velocity magnitude remains available to distinguish flow conditions.
        context = self.operator.encode(
            geometry.to(dtype=model_dtype) * self.coordinate_scale,
            supernodes.to(dtype=model_dtype) * self.coordinate_scale,
            surface_anchors.to(dtype=model_dtype) * self.coordinate_scale
            if surface_anchors is not None
            else None,
            volume_anchors.to(dtype=model_dtype) * self.coordinate_scale
            if volume_anchors is not None
            else None,
            parameters.to(dtype=model_dtype) / self.parameter_scale,
        )
        output = TensorDict({}, batch_size=[geometry.shape[0]], device=geometry.device)
        for field, enabled, queries, targets in branches:
            if not enabled:
                continue
            if self.training and self.train_on_anchors:
                prediction = getattr(context, f"{field}_predictions")
            else:
                chunks = queries.split(self.query_chunk_size, dim=1)
                prediction = torch.cat(
                    [
                        self.operator.decode_queries(
                            q.to(dtype=model_dtype) * self.coordinate_scale,
                            context,
                            field,
                        )
                        for q in chunks
                    ],
                    dim=1,
                )
            branch = split_concat_by_target(prediction, targets)
            for name in targets:
                key = as_nested_key(name)
                output.set(key, branch[key])
        return output


class ABUPTDomainReader(DomainMeshReader):
    """Sample geometry, anchors, and targets independently from native domains.

    This reader gathers only selected fields and coordinates from memory-mapped
    TensorDict stores before device transfer. Zarr loading uses the base
    reader's full-domain path, so its host-memory use can be larger. Geometry
    uses surface vertices, supernodes are a subset of that cloud, and surface
    anchors/queries use cell centroids. Only enabled branches have anchors;
    surface mode accepts a domain with an empty interior mesh.
    Uniform sampling without replacement is performed on CPU with independent
    NumPy SeedSequence streams. Validation must use a distinct reader at epoch 0.

    ``surface_queries`` and ``volume_queries`` may be None for all locations.
    Counts above the available size use all locations without replacement.
    Surface integration measures use inverse inclusion probabilities; sampled
    force integrals are Monte Carlo estimates, not full-surface integrals.
    With ``queries_at_anchors=True``, query positions and labels use the exact
    anchor indices; the query counts are ignored in favor of the anchor counts.
    Set this option to False for independent-query or full-mesh evaluation.
    """

    def __init__(
        self,
        path,
        *,
        pattern="run_*/domain_*.pdmsh",
        mode="joint",
        geometry_points=65536,
        num_supernodes=16384,
        num_surface_anchors=16384,
        num_volume_anchors=16384,
        surface_queries=16384,
        volume_queries=16384,
        pin_memory=False,
        queries_at_anchors=False,
    ):
        super().__init__(path, pattern=pattern, pin_memory=pin_memory)
        self.surface, self.volume = _modes(mode)
        if not isinstance(queries_at_anchors, bool):
            raise ValueError("queries_at_anchors must be a boolean")
        self.queries_at_anchors = queries_at_anchors
        for name, count in (
            ("geometry_points", geometry_points),
            ("num_supernodes", num_supernodes),
            ("num_surface_anchors", num_surface_anchors),
            ("num_volume_anchors", num_volume_anchors),
            ("surface_queries", surface_queries),
            ("volume_queries", volume_queries),
        ):
            if count is None and name.endswith("queries"):
                continue
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_supernodes > geometry_points:
            raise ValueError("num_supernodes must not exceed geometry_points")
        self.geometry_points, self.num_supernodes = geometry_points, num_supernodes
        self.num_surface_anchors, self.num_volume_anchors = (
            num_surface_anchors,
            num_volume_anchors,
        )
        self.surface_queries, self.volume_queries = surface_queries, volume_queries

    def close(self):
        """Release reader resources (loads are scoped to each sample)."""

    def _indices(self, total, count, index, stream):
        """Select sorted indices with a stream independent of other draws."""
        if total == 0:
            raise ValueError("AB-UPT requires nonempty meshes for enabled branches")
        if count is None or count >= total:
            return torch.arange(total)
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed_base or 0, self._epoch, index, stream])
        )
        return torch.from_numpy(np.sort(rng.choice(total, count, replace=False)))

    def __getitem__(self, index):
        """Load one case and return small geometry/anchor/query meshes."""
        domain = self._load_sample(index)
        if "vehicle" not in domain.boundary_names:
            raise ValueError("AB-UPT requires a 'vehicle' surface boundary")
        vehicle = domain.boundaries["vehicle"]
        interior = domain.interior if self.volume else None
        if vehicle.cells.shape[-1] != 3:
            raise ValueError("AB-UPT recipe requires a triangulated vehicle surface")
        gi = self._indices(vehicle.n_points, self.geometry_points, index, 0)
        geometry = vehicle.points[gi].clone()
        si = self._indices(len(gi), self.num_supernodes, index, 1)
        boundaries = {
            "abupt_geometry": Mesh(points=geometry),
            "abupt_supernodes": Mesh(points=geometry[si].clone()),
        }
        if self.surface:
            sa = self._indices(vehicle.n_cells, self.num_surface_anchors, index, 2)
            sq = (
                sa
                if self.queries_at_anchors
                else self._indices(vehicle.n_cells, self.surface_queries, index, 4)
            )

            def centroids(ids):
                return vehicle.points[vehicle.cells[ids].long()].mean(dim=1)

            surface_data = TensorDict({}, [len(sq)])
            for src, dst in (
                ("pMeanTrim", "surface.pressure"),
                ("wallShearStressMeanTrim", "surface.wss"),
            ):
                surface_data.set(as_nested_key(dst), vehicle.cell_data[src][sq].clone())
            boundaries["abupt_surface_anchors"] = Mesh(points=centroids(sa))
            boundaries["abupt_surface"] = Mesh(
                points=centroids(sq), point_data=surface_data
            )
            # Remap only selected triangles, without a full-mesh point-index map.
            vertices, cells = torch.unique(
                vehicle.cells[sq].long(), sorted=True, return_inverse=True
            )
            sampled_vehicle = Mesh(
                points=vehicle.points[vertices].clone(), cells=cells.reshape(-1, 3)
            )
            if "_effective_measure" in vehicle.cell_data:
                measures = vehicle.cell_data["_effective_measure"][sq].clone()
            else:
                measures = cell_measures(sampled_vehicle)
            set_cell_measures(sampled_vehicle, measures * (vehicle.n_cells / len(sq)))
            sampled_vehicle.cell_data["normals"] = sampled_vehicle.cell_normals
            boundaries["vehicle"] = sampled_vehicle
        else:
            boundaries["vehicle"] = Mesh(points=geometry.clone())
        sampled_interior = Mesh(points=geometry.new_empty((0, 3)))
        if self.volume:
            va = self._indices(interior.n_points, self.num_volume_anchors, index, 3)
            vq = (
                va
                if self.queries_at_anchors
                else self._indices(interior.n_points, self.volume_queries, index, 5)
            )
            volume_data = TensorDict({}, [len(vq)])
            for src, dst in (
                ("UMeanTrim", "volume.velocity"),
                ("pMeanTrim", "volume.pressure"),
                ("nutMeanTrim", "volume.nut"),
            ):
                volume_data.set(
                    as_nested_key(dst), interior.point_data[src][vq].clone()
                )
            boundaries["abupt_volume_anchors"] = Mesh(
                points=interior.points[va].clone()
            )
            sampled_interior = Mesh(
                points=interior.points[vq].clone(), point_data=volume_data
            )
        result = DomainMesh(
            interior=sampled_interior,
            boundaries=boundaries,
            global_data=domain.global_data.clone(),
        )
        if self.pin_memory:
            result = result.pin_memory()
        return result, {"source_path": str(self._paths[index]), "index": index}
