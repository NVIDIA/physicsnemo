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

import random
from typing import NamedTuple

import torch
from contact_graph import BumperCylinderContactEncoder
from datapipe import SimSample
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.experimental.models.meshtransolver import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ContactGraph,
    ContactSearchTopology,
    FunctionalContactGraphBuilder,
    MeshGeoFLARE,
    MeshGeoTransolver,
    MeshTransolver,
    SparseContactGraphBuilder,
    SurfaceContactGraphBuilder,
    merge_contact_graphs,
)
from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.transolver import Transolver

EPS = 1e-8
_FO_MIN = 3  # position-only; with dynamic_targets can be larger
_POS_DIM = 3  # position (x,y,z)


class AutoregressiveRolloutOutput(NamedTuple):
    """Trajectory prediction and transition-level acceleration supervision."""

    trajectory: torch.Tensor
    normalized_acceleration: torch.Tensor
    target_normalized_acceleration: torch.Tensor
    acceleration_supervision_mask: torch.Tensor


# =============================================================================
# One-shot rollout models
# =============================================================================


def _oneshot_init(kwargs: dict, out_key: str) -> int:
    """Validate and set rollout_steps. Returns rollout_steps."""
    num_time_steps = kwargs.pop("num_time_steps")
    rollout_steps = num_time_steps - 1
    out_dim = kwargs.get(out_key)
    required_min = rollout_steps * _FO_MIN
    if out_dim is not None and out_dim < required_min:
        raise ValueError(
            f"{out_key}={out_dim} is too small for num_time_steps={num_time_steps} "
            f"(rollout_steps={rollout_steps}). Need {out_key} >= {required_min}."
        )
    return rollout_steps


def _oneshot_inputs(sample: SimSample, rollout_steps: int):
    """Extract coords, features, N, T, Fo. Returns (coords, features, N, T, Fo)."""
    inputs = sample.node_features
    coords = inputs["coords"]  # [N,3]
    features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
    N, T = coords.size(0), rollout_steps
    Fo = sample.node_target.shape[2]
    return coords, features, N, T, Fo


def _cat_global(
    coords: torch.Tensor, features: torch.Tensor, sample: SimSample
) -> torch.Tensor:
    """Concatenate coords, features, and global (broadcast). Returns [N, C]."""
    out = torch.cat([coords, features], dim=-1)
    if sample.global_features is not None:
        g = torch.stack(
            [sample.global_features[k] for k in sample.global_features], dim=0
        )
        out = torch.cat([out, g.unsqueeze(0).expand(coords.size(0), -1)], dim=-1)
    return out


def _global_tokens(sample: SimSample) -> torch.Tensor | None:
    """Return selected global features as one token, shaped ``[1, 1, G]``."""
    if sample.global_features is None:
        return None
    values = [sample.global_features[k] for k in sample.global_features]
    return torch.stack(values, dim=0).view(1, 1, -1)


def _oneshot_output(pred_flat: torch.Tensor, N: int, T: int, Fo: int) -> torch.Tensor:
    """Validate and reshape to [N, T, Fo]."""
    if pred_flat.shape[-1] < T * Fo:
        raise ValueError(
            f"Model output dim {pred_flat.shape[-1]} smaller than T*Fo={T * Fo}"
        )
    return pred_flat[:, : T * Fo].view(N, T, Fo)


def _oneshot_add_coords(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Add initial coords to position slice only. pred [N,T,Fo], coords [N,3]. Fo >= 3."""
    pred = pred.clone()
    pred[:, :, :_POS_DIM] += coords.unsqueeze(1)  # [N,1,3] broadcasts to [N,T,3]
    return pred


class GeoTransolverOneShot(GeoTransolver):
    """GeoTransolver model with one-shot training."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        fx = torch.cat([coords, features], dim=-1)
        global_emb = None
        if sample.global_features is not None:
            g = torch.stack(
                [sample.global_features[k] for k in sample.global_features], dim=0
            )
            global_emb = g.unsqueeze(0).unsqueeze(0)  # [1, 1, G]
        raw = (
            super()
            .forward(
                local_embedding=fx.unsqueeze(0),
                geometry=coords.unsqueeze(0),
                local_positions=coords.unsqueeze(0),
                global_embedding=global_emb,
            )
            .squeeze(0)
        )
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class TransolverOneShot(Transolver):
    """Transolver model with one-shot training."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        fx = _cat_global(coords, features, sample).unsqueeze(0)
        raw = super().forward(fx=fx, embedding=coords.unsqueeze(0)).squeeze(0)
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class MeshGraphNetOneShot(MeshGraphNet):
    """MeshGraphNet model with one-shot training."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "output_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        node_feat = _cat_global(coords, features, sample)
        raw = super().forward(
            node_features=node_feat,
            edge_features=sample.graph.edge_attr,
            graph=sample.graph,
        )
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class MeshTransolverOneShot(MeshTransolver):
    """MeshTransolver with direct full-trajectory prediction."""

    def __init__(self, num_time_steps: int, **kwargs):
        kwargs["num_time_steps"] = num_time_steps
        self.rollout_steps = _oneshot_init(kwargs, "output_dim")
        super().__init__(**kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        raw = super().forward(
            node_features=_cat_global(coords, features, sample),
            edge_features=sample.graph.edge_attr,
            graph=sample.graph,
        )
        return _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)


class MeshGeoTransolverOneShot(MeshGeoTransolver):
    """MeshGeoTransolver with direct full-trajectory prediction."""

    def __init__(self, num_time_steps: int, **kwargs):
        kwargs["num_time_steps"] = num_time_steps
        self.rollout_steps = _oneshot_init(kwargs, "output_dim")
        super().__init__(**kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        raw = super().forward(
            node_features=_cat_global(coords, features, sample),
            edge_features=sample.graph.edge_attr,
            graph=sample.graph,
            geometry=coords,
            global_embedding=_global_tokens(sample),
        )
        return _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)


class MeshGeoFLAREOneShot(MeshGeoFLARE):
    """MeshGeoFLARE with direct full-trajectory prediction."""

    def __init__(self, num_time_steps: int, **kwargs):
        kwargs["num_time_steps"] = num_time_steps
        out_key = "out_dim" if "out_dim" in kwargs else "output_dim"
        self.rollout_steps = _oneshot_init(kwargs, out_key)
        super().__init__(**kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        local_features = torch.cat([coords, features], dim=-1)
        raw = super().forward(
            node_features=local_features,
            edge_features=sample.graph.edge_attr,
            graph=sample.graph,
            geometry=coords,
            local_positions=coords,
            global_embedding=_global_tokens(sample),
        )
        return _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)


class FIGConvUNetOneShot(FIGConvUNet):
    """FIGConvUNet model with one-shot training."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_channels")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        feat = _cat_global(coords, features, sample).unsqueeze(0)  # [1, N, C]
        raw, _ = super().forward(vertices=coords.unsqueeze(0), features=feat)
        pred = _oneshot_add_coords(_oneshot_output(raw.squeeze(0), N, T, Fo), coords)
        return pred


# =============================================================================
# Autoregressive rollout models
# =============================================================================


def _geo_global_emb(sample: SimSample):
    """Build global embedding for GeoTransolver from sample."""
    return _global_tokens(sample)


def _raw_global_feature(
    sample: SimSample, data_stats: dict, feature_name: str
) -> torch.Tensor:
    if sample.global_features is None or feature_name not in sample.global_features:
        raise KeyError(f"Missing required global feature {feature_name!r}")
    value = sample.global_features[feature_name]
    global_stats = data_stats.get("global_features")
    if global_stats is None:
        return value
    keys = list(global_stats.get("keys", []))
    if feature_name not in keys:
        raise KeyError(
            f"Global statistics do not contain {feature_name!r}; available: {keys}"
        )
    index = keys.index(feature_name)
    return value * global_stats["std"][index] + global_stats["mean"][index]


def _initial_velocity_from_global(
    coords: torch.Tensor,
    sample: SimSample,
    data_stats: dict,
    velocity_feature: str,
    velocity_axis: int,
    velocity_unit_scale: float,
) -> torch.Tensor:
    """Return the normalized initial velocity encoded by a global feature."""

    raw_speed = _raw_global_feature(sample, data_stats, velocity_feature)
    physical_velocity = coords.new_zeros(3)
    physical_velocity[velocity_axis] = raw_speed * velocity_unit_scale
    pos_std = data_stats["node"]["pos_std"].reshape(-1)
    return physical_velocity / pos_std


def _configure_contact_core(kwargs: dict, enable_contact: bool) -> None:
    """Keep the contact block for fair ablations when explicitly requested."""

    use_contact = bool(kwargs.pop("use_contact", enable_contact))
    if enable_contact and not use_contact:
        raise ValueError("enable_contact=True requires use_contact=True")
    kwargs["use_contact"] = use_contact


class _MeshAttentionAutoregressiveMixin:
    """Closed-loop acceleration integration shared by the hybrid graph models."""

    def _configure_autoregressive(
        self,
        *,
        num_time_steps: int,
        dt: float,
        velocity_feature: str,
        velocity_axis: int,
        velocity_unit_scale: float,
        initial_velocity_mode: str,
        rollout_steps_from_target: bool,
        checkpoint_rollout: bool,
        teacher_forcing: bool,
        node_input_mode: str,
        enable_contact: bool,
        enable_node_contact: bool,
        node_contact_radius: float,
        contact_max_neighbors: int,
        contact_candidate_neighbors: int,
        exclude_same_component: bool,
        base_shell_thickness: float,
        enable_cylinder_contact: bool,
        cylinder_center_x: float,
        cylinder_center_z: float,
        cylinder_radius: float,
        cylinder_search_distance: float,
        cylinder_center_y_feature: str,
        contact_graph_backend: str = "legacy",
        contact_search_implementation: str | None = None,
        contact_include_velocity: bool = False,
        contact_velocity_scale: float = 1000.0,
        contact_smooth_cutoff: bool = False,
        contact_selection_taper: bool = False,
        contact_normal_epsilon: float | None = None,
        contact_activation_distance: float | None = None,
        contact_prune_zero_weight: bool = False,
        contact_surface_max_pairs: int = 2_000_000,
        contact_surface_predictive: bool = False,
        contact_surface_material_fan: bool = False,
    ) -> None:
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least two")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if velocity_axis not in (0, 1, 2):
            raise ValueError("velocity_axis must be 0, 1, or 2")
        if velocity_unit_scale <= 0.0:
            raise ValueError("velocity_unit_scale must be positive")
        if initial_velocity_mode not in {"global", "previous_coords"}:
            raise ValueError(
                "initial_velocity_mode must be 'global' or 'previous_coords'"
            )
        if base_shell_thickness < 0.0:
            raise ValueError("base_shell_thickness cannot be negative")
        valid_node_input_modes = {
            "velocity",
            "position_velocity",
            "position_velocity_globals",
        }
        if node_input_mode not in valid_node_input_modes:
            raise ValueError(
                "node_input_mode must be one of "
                f"{sorted(valid_node_input_modes)}; got {node_input_mode!r}"
            )
        self.rollout_steps = num_time_steps - 1
        self.dt = float(dt)
        self.velocity_feature = velocity_feature
        self.velocity_axis = velocity_axis
        self.velocity_unit_scale = float(velocity_unit_scale)
        self.initial_velocity_mode = initial_velocity_mode
        self.rollout_steps_from_target = bool(rollout_steps_from_target)
        self.checkpoint_rollout = checkpoint_rollout
        self.teacher_forcing = bool(teacher_forcing)
        self.node_input_mode = node_input_mode
        self.enable_contact = enable_contact
        self.enable_node_contact = enable_node_contact
        self.base_shell_thickness = float(base_shell_thickness)
        self.enable_cylinder_contact = enable_cylinder_contact
        self.cylinder_center_y_feature = cylinder_center_y_feature
        self.contact_graph_backend = contact_graph_backend
        if contact_surface_material_fan and contact_graph_backend != "surface":
            raise ValueError("material fan contact requires the surface backend")
        if contact_graph_backend == "surface":
            if exclude_same_component or contact_selection_taper:
                raise ValueError(
                    "surface contact does not use component exclusions or nearest-k selection taper"
                )
            if (
                not contact_smooth_cutoff
                or contact_activation_distance is None
                or contact_normal_epsilon is None
            ):
                raise ValueError(
                    "surface contact requires smooth cutoff, activation distance and normal regularization"
                )
            self.node_contact_builder = SurfaceContactGraphBuilder(
                activation_distance=contact_activation_distance,
                feature_scale=node_contact_radius,
                velocity_scale=contact_velocity_scale,
                normal_epsilon=contact_normal_epsilon,
                include_velocity=contact_include_velocity,
                implementation=contact_search_implementation or "warp",
                max_pairs=contact_surface_max_pairs,
                prediction_horizon=self.dt if contact_surface_predictive else 0.0,
                material_fan=contact_surface_material_fan,
            )
        elif contact_graph_backend == "nearest_k":
            if contact_surface_predictive:
                raise ValueError(
                    "predictive surface contact requires the surface backend"
                )
            if exclude_same_component:
                raise ValueError(
                    "nearest_k contact allows self-contact within a component; use explicit topology exclusions instead"
                )
            self.node_contact_builder = FunctionalContactGraphBuilder(
                search_radius=node_contact_radius,
                max_neighbors=contact_max_neighbors,
                implementation=contact_search_implementation,
                include_velocity=contact_include_velocity,
                velocity_scale=contact_velocity_scale,
                smooth_cutoff=contact_smooth_cutoff,
                selection_taper=contact_selection_taper,
                normal_epsilon=contact_normal_epsilon,
                activation_distance=contact_activation_distance,
                prune_zero_weight=contact_prune_zero_weight,
            )
        elif contact_graph_backend == "legacy":
            if contact_surface_predictive:
                raise ValueError(
                    "predictive surface contact requires the surface backend"
                )
            if (
                contact_include_velocity
                or contact_smooth_cutoff
                or contact_search_implementation is not None
                or contact_selection_taper
                or contact_normal_epsilon is not None
                or contact_activation_distance is not None
                or contact_prune_zero_weight
            ):
                raise ValueError(
                    "Contact velocity, smooth cutoff and functional implementation require contact_graph_backend='nearest_k'"
                )
            self.node_contact_builder = SparseContactGraphBuilder(
                search_radius=node_contact_radius,
                max_neighbors=contact_max_neighbors,
                candidate_neighbors=contact_candidate_neighbors,
                exclude_structural_edges=True,
                exclude_same_component=exclude_same_component,
            )
        else:
            raise ValueError(
                "contact_graph_backend must be 'legacy', 'nearest_k', or 'surface'"
            )
        self.cylinder_contact_encoder = BumperCylinderContactEncoder(
            center_x=cylinder_center_x,
            center_z=cylinder_center_z,
            radius=cylinder_radius,
            search_distance=cylinder_search_distance,
            include_velocity=contact_include_velocity,
            velocity_scale=contact_velocity_scale,
            smooth_cutoff=contact_smooth_cutoff,
        )

    def _initial_velocity(
        self, coords: torch.Tensor, sample: SimSample, data_stats: dict
    ) -> torch.Tensor:
        if self.initial_velocity_mode == "previous_coords":
            previous_coords = sample.node_features.get("previous_coords")
            if previous_coords is None:
                raise ValueError(
                    "initial_velocity_mode='previous_coords' requires "
                    "sample.node_features['previous_coords']"
                )
            if previous_coords.shape != coords.shape:
                raise ValueError(
                    "previous_coords must have the same shape as coords; got "
                    f"{tuple(previous_coords.shape)} and {tuple(coords.shape)}"
                )
            return (coords - previous_coords) / self.dt
        global_velocity = _initial_velocity_from_global(
            coords,
            sample,
            data_stats,
            self.velocity_feature,
            self.velocity_axis,
            self.velocity_unit_scale,
        )
        return global_velocity.unsqueeze(0).expand_as(coords)

    def _physical_shell_thickness(
        self, sample: SimSample, data_stats: dict, num_nodes: int
    ) -> torch.Tensor:
        coords = sample.node_features["coords"]
        scale = coords.new_ones(())
        if (
            sample.global_features is not None
            and "thickness_scale" in sample.global_features
        ):
            scale = _raw_global_feature(sample, data_stats, "thickness_scale")
        fallback = scale.expand(num_nodes) * self.base_shell_thickness
        supplied = getattr(sample.graph, "shell_thickness", None)
        if supplied is None:
            if self.contact_graph_backend == "surface":
                raise ValueError(
                    "surface contact requires actual graph.shell_thickness; no fallback"
                )
            return fallback
        supplied = supplied.to(device=coords.device, dtype=coords.dtype).reshape(-1)
        if supplied.shape != (num_nodes,):
            raise ValueError("graph.shell_thickness must have one value per node")
        if not torch.isfinite(supplied).all() or torch.any(supplied < 0.0):
            raise ValueError("graph.shell_thickness must be finite and nonnegative")
        if self.contact_graph_backend == "surface":
            return supplied
        return torch.where(supplied > 0.0, supplied, fallback)

    def _contact_graph(
        self,
        physical_positions: torch.Tensor,
        sample: SimSample,
        data_stats: dict,
        shell_thickness: torch.Tensor,
        physical_velocities: torch.Tensor | None = None,
        topology: ContactSearchTopology | None = None,
    ) -> ContactGraph:
        batch = getattr(sample.graph, "batch", None)
        graphs: list[ContactGraph] = []
        if self.enable_node_contact:
            builder_kwargs = (
                {"velocities": physical_velocities, "topology": topology}
                if self.contact_graph_backend == "nearest_k"
                else {"component_ids": getattr(sample.graph, "component_id", None)}
            )
            if self.contact_graph_backend == "surface":
                if getattr(sample.graph, "shell_thickness", None) is None:
                    raise ValueError(
                        "surface contact requires actual graph.shell_thickness; no fallback"
                    )
                builder_kwargs = {
                    "velocities": physical_velocities,
                    "faces": getattr(sample.graph, "contact_faces", None),
                    # The observed start of this window, never a future target
                    # or the evolving predicted configuration. It anchors the
                    # material fan's reference-quality check across the rollout.
                    "reference_positions": (
                        sample.node_features["coords"] * data_stats["node"]["pos_std"]
                        + data_stats["node"]["pos_mean"]
                    ),
                    "excluded_pairs": getattr(
                        sample.graph, "contact_surface_exclusions", None
                    ),
                }
            graphs.append(
                self.node_contact_builder(
                    physical_positions,
                    structural_edge_index=sample.graph.edge_index,
                    batch=batch,
                    shell_thickness=shell_thickness,
                    **builder_kwargs,
                )
            )
        if self.enable_cylinder_contact:
            center_y = _raw_global_feature(
                sample, data_stats, self.cylinder_center_y_feature
            )
            graphs.append(
                self.cylinder_contact_encoder(
                    physical_positions,
                    center_y=center_y,
                    batch=batch,
                    shell_thickness=shell_thickness,
                    velocities=physical_velocities,
                )
            )
        if not graphs:
            return ContactGraph.empty(
                physical_positions.device,
                physical_positions.dtype,
                getattr(self, "contact_dim", CONTACT_FEATURE_DIM),
            )
        return merge_contact_graphs(*graphs)

    def _checkpointed_core_step(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        geometry: torch.Tensor,
        global_embedding: torch.Tensor | None,
        contact_graph: ContactGraph | None,
        graph,
    ) -> torch.Tensor:
        if global_embedding is None:
            global_embedding = node_features.new_empty((0, 0, 0))
        if contact_graph is None:
            contact_graph = ContactGraph.empty(
                node_features.device,
                node_features.dtype,
                getattr(self, "contact_dim", CONTACT_FEATURE_DIM),
            )

        def step_fn(
            nodes,
            edges,
            positions,
            globals_,
            contact_edge_index,
            contact_features,
            obstacle_mask,
            contact_weights,
            source_nodes,
            source_weights,
        ):
            step_contact = None
            if self.use_contact:
                step_contact = ContactGraph(
                    edge_index=contact_edge_index,
                    edge_features=contact_features,
                    obstacle_mask=obstacle_mask,
                    edge_weights=contact_weights if contact_weights.numel() else None,
                    source_nodes=source_nodes if source_nodes.numel() else None,
                    source_weights=source_weights if source_weights.numel() else None,
                )
            step_globals = globals_ if globals_.numel() else None
            return self._call_hybrid_core(
                nodes,
                edges,
                graph,
                positions,
                step_globals,
                step_contact,
            )

        args = (
            node_features,
            edge_features,
            geometry,
            global_embedding,
            contact_graph.edge_index,
            contact_graph.edge_features,
            contact_graph.obstacle_mask,
            contact_graph.edge_weights
            if contact_graph.edge_weights is not None
            else node_features.new_empty((0,)),
            contact_graph.source_nodes
            if contact_graph.source_nodes is not None
            else contact_graph.edge_index.new_empty((0, 0)),
            contact_graph.source_weights
            if contact_graph.source_weights is not None
            else node_features.new_empty((0, 0)),
        )
        if self.training and self.checkpoint_rollout:
            return ckpt(step_fn, *args, use_reentrant=False)
        return step_fn(*args)

    def _target_normalized_acceleration(
        self,
        sample: SimSample,
        coords: torch.Tensor,
        initial_velocity: torch.Tensor,
        data_stats: dict,
        rollout_steps: int,
    ) -> torch.Tensor:
        """Derive normalized transition accelerations from the target trajectory."""

        target_positions = sample.node_target[..., :_POS_DIM].detach()
        if target_positions.shape[1] < rollout_steps:
            raise ValueError(
                "Acceleration supervision requires one target position for every "
                f"rollout step; got {target_positions.shape[1]} targets for "
                f"{rollout_steps} steps"
            )
        target_positions = target_positions[:, :rollout_steps]
        positions = torch.cat((coords.unsqueeze(1), target_positions), dim=1)
        previous_initial = coords - initial_velocity * self.dt
        previous = torch.cat((previous_initial.unsqueeze(1), positions[:, :-2]), dim=1)
        current = positions[:, :-1]
        following = positions[:, 1:]
        acceleration = (following - 2.0 * current + previous) / (self.dt * self.dt)
        return (acceleration - data_stats["node"]["norm_acc_mean"]) / (
            data_stats["node"]["norm_acc_std"] + EPS
        )

    def _autoregressive_forward(
        self,
        sample: SimSample,
        data_stats: dict,
        *,
        teacher_forcing_probability: float | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | AutoregressiveRolloutOutput:
        coords = sample.node_features["coords"]
        features = sample.node_features.get(
            "features", coords.new_zeros((coords.shape[0], 0))
        )
        num_nodes = coords.shape[0]
        target_features = sample.node_target.shape[-1]
        rollout_steps = (
            int(sample.node_target.shape[1])
            if self.rollout_steps_from_target
            else self.rollout_steps
        )
        if rollout_steps < 1:
            raise ValueError("Autoregressive rollout requires at least one target step")
        if rollout_steps > self.rollout_steps:
            raise ValueError(
                f"Target requests {rollout_steps} steps, exceeding configured maximum "
                f"of {self.rollout_steps}"
            )
        pos_mean = data_stats["node"]["pos_mean"].reshape(-1)
        pos_std = data_stats["node"]["pos_std"].reshape(-1)
        initial_velocity = self._initial_velocity(coords, sample, data_stats)
        previous_coords = coords - initial_velocity * self.dt
        current_coords = coords
        global_embedding = _global_tokens(sample)
        normalized_globals = None
        if sample.global_features is not None:
            normalized_globals = torch.stack(
                [sample.global_features[key] for key in sample.global_features], dim=0
            )
        shell_thickness = None
        contact_topology = None
        if self.enable_contact:
            shell_thickness = self._physical_shell_thickness(
                sample, data_stats, num_nodes
            )
            if self.enable_node_contact and self.contact_graph_backend == "nearest_k":
                contact_topology = self.node_contact_builder.prepare_topology(
                    num_nodes,
                    sample.graph.edge_index,
                    getattr(sample.graph, "batch", None),
                    device=coords.device,
                    extra_exclusion_edges=getattr(
                        sample.graph, "contact_exclusion_edges", None
                    ),
                )

        if teacher_forcing_probability is None:
            teacher_forcing_probability = 1.0 if self.teacher_forcing else 0.0
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0, 1]")
        if not self.training:
            teacher_forcing_probability = 0.0

        teacher_positions = None
        if self.training and teacher_forcing_probability > 0.0:
            teacher_positions = sample.node_target[..., :3].detach()
            if teacher_positions.shape[1] < rollout_steps:
                raise ValueError(
                    "Teacher forcing requires one target position for every "
                    f"rollout step; got {teacher_positions.shape[1]} targets for "
                    f"{rollout_steps} steps"
                )
        acceleration_supervision_mask = torch.zeros(
            rollout_steps, dtype=torch.bool, device=coords.device
        )
        # The initial state and velocity are observed, so the first transition is
        # always eligible for direct acceleration supervision.
        acceleration_supervision_mask[0] = True
        outputs: list[torch.Tensor] = []
        normalized_accelerations: list[torch.Tensor] = []
        for step in range(rollout_steps):
            # Scheduled teacher forcing operates on a consistent position pair,
            # which is required because velocity is a second-order state. Eval
            # remains fully closed-loop regardless of the configured probability.
            use_teacher_state = False
            if teacher_positions is not None and step > 0:
                use_teacher_state = random.random() < teacher_forcing_probability
            if use_teacher_state:
                previous_coords = (
                    coords if step == 1 else teacher_positions[:, step - 2]
                )
                current_coords = teacher_positions[:, step - 1]
                acceleration_supervision_mask[step] = True
            velocity = (current_coords - previous_coords) / self.dt
            normalized_velocity = (velocity - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            if self.node_input_mode == "velocity":
                node_parts = [normalized_velocity, features]
            else:
                node_parts = [current_coords, normalized_velocity, features]
            if (
                self.node_input_mode == "position_velocity_globals"
                and normalized_globals is not None
            ):
                node_parts.append(normalized_globals.unsqueeze(0).expand(num_nodes, -1))
            node_features = torch.cat(node_parts, dim=-1)
            # The finite-element connectivity and its structural edge
            # attributes describe the reference mesh. They are material data,
            # not rollout state, and therefore stay fixed as nodes deform.
            edge_features = sample.graph.edge_attr

            contact_graph = None
            if self.enable_contact:
                if shell_thickness is None:
                    raise RuntimeError("Contact rollout requires shell thickness")
                physical_positions = current_coords * pos_std + pos_mean
                contact_graph = self._contact_graph(
                    physical_positions,
                    sample,
                    data_stats,
                    shell_thickness,
                    physical_velocities=velocity * pos_std,
                    topology=contact_topology,
                )
            raw_output = self._checkpointed_core_step(
                node_features,
                edge_features,
                current_coords,
                global_embedding,
                contact_graph,
                sample.graph,
            )
            if raw_output.shape[-1] < target_features:
                raise ValueError(
                    f"Model output has {raw_output.shape[-1]} channels but the "
                    f"rollout target requires {target_features}"
                )
            acceleration = (
                raw_output[:, :3] * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            normalized_accelerations.append(raw_output[:, :3])
            next_velocity = velocity + acceleration * self.dt
            next_coords = current_coords + next_velocity * self.dt
            frame = next_coords
            if target_features > 3:
                frame = torch.cat(
                    (next_coords, raw_output[:, 3:target_features]), dim=-1
                )
            outputs.append(frame)
            previous_coords, current_coords = current_coords, next_coords
        trajectory = torch.stack(outputs, dim=1)
        if not return_auxiliary:
            return trajectory
        return AutoregressiveRolloutOutput(
            trajectory=trajectory,
            normalized_acceleration=torch.stack(normalized_accelerations, dim=1),
            target_normalized_acceleration=self._target_normalized_acceleration(
                sample, coords, initial_velocity, data_stats, rollout_steps
            ),
            acceleration_supervision_mask=acceleration_supervision_mask,
        )


class MeshTransolverAutoregressive(MeshTransolver, _MeshAttentionAutoregressiveMixin):
    """MeshTransolver with closed-loop acceleration rollout and optional contact."""

    def __init__(
        self,
        num_time_steps: int,
        dt: float = 5.0e-3,
        velocity_feature: str = "velocity_x",
        velocity_axis: int = 0,
        velocity_unit_scale: float = 1000.0,
        initial_velocity_mode: str = "global",
        rollout_steps_from_target: bool = False,
        checkpoint_rollout: bool = True,
        teacher_forcing: bool = False,
        node_input_mode: str = "position_velocity_globals",
        enable_contact: bool = True,
        enable_node_contact: bool = True,
        node_contact_radius: float = 10.0,
        contact_max_neighbors: int = 32,
        contact_candidate_neighbors: int = 128,
        exclude_same_component: bool = False,
        base_shell_thickness: float = 2.0,
        enable_cylinder_contact: bool = True,
        cylinder_center_x: float = -170.0,
        cylinder_center_z: float = 0.0,
        cylinder_radius: float = 127.0,
        cylinder_search_distance: float = 200.0,
        cylinder_center_y_feature: str = "rwall_origin_y",
        **kwargs,
    ) -> None:
        _configure_contact_core(kwargs, enable_contact)
        super().__init__(**kwargs)
        self._configure_autoregressive(
            num_time_steps=num_time_steps,
            dt=dt,
            velocity_feature=velocity_feature,
            velocity_axis=velocity_axis,
            velocity_unit_scale=velocity_unit_scale,
            initial_velocity_mode=initial_velocity_mode,
            rollout_steps_from_target=rollout_steps_from_target,
            checkpoint_rollout=checkpoint_rollout,
            teacher_forcing=teacher_forcing,
            node_input_mode=node_input_mode,
            enable_contact=enable_contact,
            enable_node_contact=enable_node_contact,
            node_contact_radius=node_contact_radius,
            contact_max_neighbors=contact_max_neighbors,
            contact_candidate_neighbors=contact_candidate_neighbors,
            exclude_same_component=exclude_same_component,
            base_shell_thickness=base_shell_thickness,
            enable_cylinder_contact=enable_cylinder_contact,
            cylinder_center_x=cylinder_center_x,
            cylinder_center_z=cylinder_center_z,
            cylinder_radius=cylinder_radius,
            cylinder_search_distance=cylinder_search_distance,
            cylinder_center_y_feature=cylinder_center_y_feature,
        )

    def _call_hybrid_core(
        self, nodes, edges, graph, geometry, global_embedding, contact_graph
    ):
        return MeshTransolver.forward(
            self,
            node_features=nodes,
            edge_features=edges,
            graph=graph,
            contact_graph=contact_graph,
        )

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
        *,
        teacher_forcing_probability: float | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | AutoregressiveRolloutOutput:
        return self._autoregressive_forward(
            sample,
            data_stats,
            teacher_forcing_probability=teacher_forcing_probability,
            return_auxiliary=return_auxiliary,
        )


class MeshGeoTransolverAutoregressive(
    MeshGeoTransolver, _MeshAttentionAutoregressiveMixin
):
    """MeshGeoTransolver with closed-loop acceleration rollout and contact."""

    def __init__(
        self,
        num_time_steps: int,
        dt: float = 5.0e-3,
        velocity_feature: str = "velocity_x",
        velocity_axis: int = 0,
        velocity_unit_scale: float = 1000.0,
        initial_velocity_mode: str = "global",
        rollout_steps_from_target: bool = False,
        checkpoint_rollout: bool = True,
        teacher_forcing: bool = False,
        node_input_mode: str = "position_velocity_globals",
        enable_contact: bool = True,
        enable_node_contact: bool = True,
        node_contact_radius: float = 10.0,
        contact_max_neighbors: int = 16,
        contact_candidate_neighbors: int = 64,
        exclude_same_component: bool = False,
        base_shell_thickness: float = 2.0,
        enable_cylinder_contact: bool = True,
        cylinder_center_x: float = -170.0,
        cylinder_center_z: float = 0.0,
        cylinder_radius: float = 127.0,
        cylinder_search_distance: float = 200.0,
        cylinder_center_y_feature: str = "rwall_origin_y",
        **kwargs,
    ) -> None:
        _configure_contact_core(kwargs, enable_contact)
        super().__init__(**kwargs)
        self._configure_autoregressive(
            num_time_steps=num_time_steps,
            dt=dt,
            velocity_feature=velocity_feature,
            velocity_axis=velocity_axis,
            velocity_unit_scale=velocity_unit_scale,
            initial_velocity_mode=initial_velocity_mode,
            rollout_steps_from_target=rollout_steps_from_target,
            checkpoint_rollout=checkpoint_rollout,
            teacher_forcing=teacher_forcing,
            node_input_mode=node_input_mode,
            enable_contact=enable_contact,
            enable_node_contact=enable_node_contact,
            node_contact_radius=node_contact_radius,
            contact_max_neighbors=contact_max_neighbors,
            contact_candidate_neighbors=contact_candidate_neighbors,
            exclude_same_component=exclude_same_component,
            base_shell_thickness=base_shell_thickness,
            enable_cylinder_contact=enable_cylinder_contact,
            cylinder_center_x=cylinder_center_x,
            cylinder_center_z=cylinder_center_z,
            cylinder_radius=cylinder_radius,
            cylinder_search_distance=cylinder_search_distance,
            cylinder_center_y_feature=cylinder_center_y_feature,
        )

    def _call_hybrid_core(
        self, nodes, edges, graph, geometry, global_embedding, contact_graph
    ):
        return MeshGeoTransolver.forward(
            self,
            node_features=nodes,
            edge_features=edges,
            graph=graph,
            geometry=geometry,
            global_embedding=global_embedding,
            contact_graph=contact_graph,
        )

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
        *,
        teacher_forcing_probability: float | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | AutoregressiveRolloutOutput:
        return self._autoregressive_forward(
            sample,
            data_stats,
            teacher_forcing_probability=teacher_forcing_probability,
            return_auxiliary=return_auxiliary,
        )


class MeshGeoFLAREAutoregressive(MeshGeoFLARE, _MeshAttentionAutoregressiveMixin):
    """DeFormer with closed-loop acceleration integration and optional contact.

    Keep the historical class name for saved checkpoints and Hydra targets.
    The supported contact experiment selects the predictive ``surface`` backend;
    legacy node/obstacle options remain for older configurations only.
    """

    def __init__(
        self,
        num_time_steps: int,
        dt: float = 5.0e-3,
        velocity_feature: str = "velocity_x",
        velocity_axis: int = 0,
        velocity_unit_scale: float = 1000.0,
        initial_velocity_mode: str = "global",
        rollout_steps_from_target: bool = False,
        checkpoint_rollout: bool = True,
        teacher_forcing: bool = False,
        node_input_mode: str = "position_velocity_globals",
        enable_contact: bool = True,
        enable_node_contact: bool = True,
        node_contact_radius: float = 10.0,
        contact_max_neighbors: int = 16,
        contact_candidate_neighbors: int = 64,
        exclude_same_component: bool = False,
        base_shell_thickness: float = 2.0,
        enable_cylinder_contact: bool = True,
        cylinder_center_x: float = -170.0,
        cylinder_center_z: float = 0.0,
        cylinder_radius: float = 127.0,
        cylinder_search_distance: float = 200.0,
        cylinder_center_y_feature: str = "rwall_origin_y",
        contact_graph_backend: str = "legacy",
        contact_search_implementation: str | None = None,
        contact_include_velocity: bool = False,
        contact_velocity_scale: float = 1000.0,
        contact_smooth_cutoff: bool = False,
        contact_selection_taper: bool = False,
        contact_normal_epsilon: float | None = None,
        contact_activation_distance: float | None = None,
        contact_prune_zero_weight: bool = False,
        **kwargs,
    ) -> None:
        _configure_contact_core(kwargs, enable_contact)
        contact_surface_max_pairs = kwargs.pop("contact_surface_max_pairs", 2_000_000)
        contact_surface_predictive = kwargs.pop("contact_surface_predictive", False)
        contact_surface_material_fan = kwargs.pop("contact_surface_material_fan", False)
        feature_dim = (
            KINEMATIC_CONTACT_FEATURE_DIM
            if contact_include_velocity
            else CONTACT_FEATURE_DIM
        )
        if kwargs.setdefault("contact_dim", feature_dim) != feature_dim:
            raise ValueError(
                f"contact_dim must be {feature_dim} for this recipe's contact features"
            )
        super().__init__(**kwargs)
        self._configure_autoregressive(
            num_time_steps=num_time_steps,
            dt=dt,
            velocity_feature=velocity_feature,
            velocity_axis=velocity_axis,
            velocity_unit_scale=velocity_unit_scale,
            initial_velocity_mode=initial_velocity_mode,
            rollout_steps_from_target=rollout_steps_from_target,
            checkpoint_rollout=checkpoint_rollout,
            teacher_forcing=teacher_forcing,
            node_input_mode=node_input_mode,
            enable_contact=enable_contact,
            enable_node_contact=enable_node_contact,
            node_contact_radius=node_contact_radius,
            contact_max_neighbors=contact_max_neighbors,
            contact_candidate_neighbors=contact_candidate_neighbors,
            exclude_same_component=exclude_same_component,
            base_shell_thickness=base_shell_thickness,
            enable_cylinder_contact=enable_cylinder_contact,
            cylinder_center_x=cylinder_center_x,
            cylinder_center_z=cylinder_center_z,
            cylinder_radius=cylinder_radius,
            cylinder_search_distance=cylinder_search_distance,
            cylinder_center_y_feature=cylinder_center_y_feature,
            contact_graph_backend=contact_graph_backend,
            contact_search_implementation=contact_search_implementation,
            contact_include_velocity=contact_include_velocity,
            contact_velocity_scale=contact_velocity_scale,
            contact_smooth_cutoff=contact_smooth_cutoff,
            contact_selection_taper=contact_selection_taper,
            contact_normal_epsilon=contact_normal_epsilon,
            contact_activation_distance=contact_activation_distance,
            contact_prune_zero_weight=contact_prune_zero_weight,
            contact_surface_max_pairs=contact_surface_max_pairs,
            contact_surface_predictive=contact_surface_predictive,
            contact_surface_material_fan=contact_surface_material_fan,
        )

    def _call_hybrid_core(
        self, nodes, edges, graph, geometry, global_embedding, contact_graph
    ):
        return MeshGeoFLARE.forward(
            self,
            node_features=nodes,
            edge_features=edges,
            graph=graph,
            geometry=geometry,
            global_embedding=global_embedding,
            contact_graph=contact_graph,
        )

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
        *,
        teacher_forcing_probability: float | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | AutoregressiveRolloutOutput:
        return self._autoregressive_forward(
            sample,
            data_stats,
            teacher_forcing_probability=teacher_forcing_probability,
            return_auxiliary=return_auxiliary,
        )


class GeoTransolverAutoregressive(GeoTransolver):
    """Point-cloud GeoTransolver with teacher-forced or closed-loop rollout."""

    def __init__(
        self,
        num_time_steps: int,
        dt: float = 5.0e-3,
        velocity_feature: str = "velocity_x",
        velocity_axis: int = 0,
        velocity_unit_scale: float = 1000.0,
        initial_velocity_mode: str = "global",
        rollout_steps_from_target: bool = False,
        checkpoint_rollout: bool = True,
        teacher_forcing: bool = False,
        **kwargs,
    ) -> None:
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least two")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if velocity_axis not in (0, 1, 2):
            raise ValueError("velocity_axis must be 0, 1, or 2")
        if velocity_unit_scale <= 0.0:
            raise ValueError("velocity_unit_scale must be positive")
        if initial_velocity_mode not in {"global", "previous_coords"}:
            raise ValueError(
                "initial_velocity_mode must be 'global' or 'previous_coords'"
            )
        self.rollout_steps = num_time_steps - 1
        self.dt = float(dt)
        self.velocity_feature = velocity_feature
        self.velocity_axis = velocity_axis
        self.velocity_unit_scale = float(velocity_unit_scale)
        self.initial_velocity_mode = initial_velocity_mode
        self.rollout_steps_from_target = bool(rollout_steps_from_target)
        self.checkpoint_rollout = bool(checkpoint_rollout)
        self.teacher_forcing = bool(teacher_forcing)
        super().__init__(**kwargs)

    def _core_step(
        self,
        local_embedding: torch.Tensor,
        geometry: torch.Tensor,
        global_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if global_embedding is None:
            global_embedding = local_embedding.new_empty((0, 0, 0))

        def step_fn(local, positions, globals_):
            step_globals = globals_ if globals_.numel() else None
            return (
                super(GeoTransolverAutoregressive, self)
                .forward(
                    local_embedding=local.unsqueeze(0),
                    geometry=positions.unsqueeze(0),
                    local_positions=positions.unsqueeze(0),
                    global_embedding=step_globals,
                )
                .squeeze(0)
            )

        args = (local_embedding, geometry, global_embedding)
        if self.training and self.checkpoint_rollout:
            return ckpt(step_fn, *args, use_reentrant=False)
        return step_fn(*args)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords = sample.node_features["coords"]
        features = sample.node_features.get(
            "features", coords.new_zeros((coords.shape[0], 0))
        )
        target_features = sample.node_target.shape[-1]
        rollout_steps = (
            int(sample.node_target.shape[1])
            if self.rollout_steps_from_target
            else self.rollout_steps
        )
        if rollout_steps < 1:
            raise ValueError("Autoregressive rollout requires at least one target step")
        if rollout_steps > self.rollout_steps:
            raise ValueError(
                f"Target requests {rollout_steps} steps, exceeding configured maximum "
                f"of {self.rollout_steps}"
            )
        if self.initial_velocity_mode == "previous_coords":
            previous_coords = sample.node_features.get("previous_coords")
            if previous_coords is None:
                raise ValueError(
                    "initial_velocity_mode='previous_coords' requires "
                    "sample.node_features['previous_coords']"
                )
            if previous_coords.shape != coords.shape:
                raise ValueError(
                    "previous_coords must have the same shape as coords; got "
                    f"{tuple(previous_coords.shape)} and {tuple(coords.shape)}"
                )
        else:
            initial_velocity = _initial_velocity_from_global(
                coords,
                sample,
                data_stats,
                self.velocity_feature,
                self.velocity_axis,
                self.velocity_unit_scale,
            )
            previous_coords = coords - initial_velocity.unsqueeze(0) * self.dt
        current_coords = coords
        global_embedding = _global_tokens(sample)

        teacher_positions = None
        if self.training and self.teacher_forcing:
            teacher_positions = sample.node_target[..., :3].detach()
            if teacher_positions.shape[1] < rollout_steps:
                raise ValueError(
                    "Teacher forcing requires one target position for every "
                    f"rollout step; got {teacher_positions.shape[1]} targets for "
                    f"{rollout_steps} steps"
                )

        outputs: list[torch.Tensor] = []
        for step in range(rollout_steps):
            if teacher_positions is not None and step > 0:
                previous_coords = (
                    coords if step == 1 else teacher_positions[:, step - 2]
                )
                current_coords = teacher_positions[:, step - 1]
            velocity = (current_coords - previous_coords) / self.dt
            normalized_velocity = (velocity - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            local_embedding = torch.cat((normalized_velocity, features), dim=-1)
            raw_output = self._core_step(
                local_embedding, current_coords, global_embedding
            )
            if raw_output.shape[-1] < target_features:
                raise ValueError(
                    f"Model output has {raw_output.shape[-1]} channels but the "
                    f"rollout target requires {target_features}"
                )
            acceleration = (
                raw_output[:, :3] * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            next_velocity = velocity + acceleration * self.dt
            next_coords = current_coords + next_velocity * self.dt
            frame = next_coords
            if target_features > 3:
                frame = torch.cat(
                    (next_coords, raw_output[:, 3:target_features]), dim=-1
                )
            outputs.append(frame)
            previous_coords, current_coords = current_coords, next_coords
        return torch.stack(outputs, dim=1)


class GeoTransolverAutoregressiveRolloutTraining(GeoTransolver):
    """
    GeoTransolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, T, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)
        global_emb = _geo_global_emb(sample)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input: vel_norm + features
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F]

            def step_fn(local_emb, geometry, local_pos):
                return super(GeoTransolverAutoregressiveRolloutTraining, self).forward(
                    local_embedding=local_emb,
                    geometry=geometry,
                    local_positions=local_pos,
                    global_embedding=global_emb,
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                ).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]


# =============================================================================
# Time-conditional rollout models
# =============================================================================


class GeoTransolverTimeConditional(GeoTransolver):
    """
    GeoTransolver model with time-conditional rollout training.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        if self.training:
            return self._forward(sample, data_stats)
        else:
            return self._rollout(sample, data_stats)

    def _forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, Fo] prediction at time t
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        global_embedding = None
        if sample.global_features is not None:
            global_embedding = (
                torch.stack(
                    [sample.global_features[k] for k in sample.global_features], dim=0
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # [1, 1, num_global]

        N, T = coords.size(0), self.rollout_steps
        Fo = sample.node_target.shape[-1]  # 3 + sum(C_k)

        fx_t = torch.cat(
            [coords, features, inputs["time"].unsqueeze(0).repeat(N, 1)], dim=-1
        )  # [N, 3+F+1]
        pred = (
            super(GeoTransolverTimeConditional, self)
            .forward(
                local_embedding=fx_t.unsqueeze(0),
                geometry=coords.unsqueeze(0),
                local_positions=coords.unsqueeze(0),
                global_embedding=global_embedding,
            )
            .squeeze(0)
        )  # [N, Fo]

        outputs = coords + pred[:, :3]
        outputs = torch.cat([outputs, pred[:, 3:]], dim=-1)

        return outputs  # [N,3]

    def _rollout(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, T, Fo] rollout of predicted positions
        """
        device = sample.node_features["coords"].device
        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            time = torch.tensor(t / self.rollout_steps, device=device)
            sample.node_features["time"] = time
            y_t2 = self._forward(sample, data_stats)
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]


# =============================================================================
# One-step rollout models
# =============================================================================


class GeoTransolverOneStepRollout(GeoTransolver):
    """
    One-step rollout:
      - Training: teacher forcing (uses GT for each step, but first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))
        global_emb = _geo_global_emb(sample)

        # Ground truth sequence [T+1, N, 3] (t0 + rollout steps)
        N = coords0.size(0)
        gt_seq = torch.cat(
            [
                coords0.unsqueeze(0),
                sample.node_target.transpose(0, 1),
            ],  # [N,T,3] -> [T,N,3]
            dim=0,
        )

        outputs: list[torch.Tensor] = []

        # First step: backstep to create y_-1
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # teacher forcing uses GT pairs
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([vel_norm, features], dim=-1)

            def step_fn(local_emb, geometry, local_pos):
                return super(GeoTransolverOneStepRollout, self).forward(
                    local_embedding=local_emb,
                    geometry=geometry,
                    local_positions=local_pos,
                    global_embedding=global_emb,
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                ).squeeze(0)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # autoregressive update for inference
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]
