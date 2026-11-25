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
"""Target-relative penetration metrics for the exact Yaris validation.

The ground truth of a shell crash simulation is not penetration-free under a
geometric test: the solver's penalty contact tolerates small overlaps, folded
regions pack shells closer than their nominal thickness, and a node can be on
the far side of a facet it never passed through. Absolute counts therefore do
not separate models well (the LS-DYNA frames themselves register about 0.2% of
nodes as penetrating). The statistics here are defined per node and frame
against the ground-truth frame scored with the same machinery, so that only
penetration the truth does not have counts against a model.

Definitions (physical millimetres, per held-out case and future frame):

* Candidate pairs: node-facet pairs of the dataset's shell surface discovered
  by :func:`physicsnemo.experimental.nn.shell_contact.shell_contact_pairs` on the frame's positions inside the
  ``activation_distance_mm`` shell-clearance band, minus the one-ring
  exclusions, minus pairs already within ``reference_clearance_mm`` of each
  other in the initial configuration (flanges, in-plane neighbours).
* Signed gap of a pair: ``g = s * delta - (h_node + h_facet) / 2`` with
  ``delta`` the distance from the node to the facet's material centre fan,
  ``h`` the nodal shell thickness (barycentric on the facet) and ``s = -1``
  when the node has crossed the facet plane since the initial configuration,
  ``+1`` otherwise. A crossing requires the node to project onto the facet's
  interior in BOTH the initial and the current configuration, so a node that
  was beside the facet and slid under its edge is not a crossing (the records
  carry this as ``sidedness: symmetric``, ``metric_version`` 2). The rule is
  deliberately stricter than the one-sided side test of the corrector's
  contact features: on the reference trajectories most deep one-sided
  "crossings" are nodes sliding past a facet edge, not penetrations, so the
  metric counts only certified crossings and under-counts a node that slides
  by more than about one element before it crosses.
* Envelope depth of a node: ``d_env = max over its pairs of relu(-g)``, the
  deepest violation of the thickness envelope; ``crossing depth``
  ``d_x = max over its pairs of relu(-s * delta)``, the distance it has
  travelled through a shell's mid-surface (zero unless crossed).
* Rigid wall (optional): the plane ``axis = position``; ``d_env`` is the
  envelope violation ``relu(h/2 - clearance)`` and ``d_x`` the distance beyond
  the plane itself.
* Detection ceiling of the surface families: a pair exists only while the
  node's unsigned distance to the facet lies inside the activation band, so a
  node that has travelled further through a shell than
  ``activation_distance_mm + (h_node + h_facet) / 2`` (about 6 mm with the
  defaults and 1 mm shells) has no pair and both surface depths read zero for
  it. The summary records this ceiling (``crossing_depth_ceiling_mm``, with
  the mean thickness of the validated designs). ``crossing_radius_mm`` lifts it
  to that radius plus the half thicknesses by sweeping candidates beyond the
  band that the side test certifies as crossed (about thirty times the search
  cost at full-vehicle scale), which is why it is off by default.
* A node *penetrates* when its depth exceeds ``tolerance_mm``. Scored against
  the target frame: a node is **spurious** when it penetrates in the prediction
  and not in the target, **missed** when the reverse holds, and its **excess**
  is ``relu(d_pred - d_target)``. Spurious depths are the prediction's depths
  of spurious nodes; their distribution is reported through fixed thresholds
  and a histogram (percentiles are bin upper edges).
* Everything is accumulated per future frame and per reference-``x`` region
  (``region_bounds_mm`` splits the mesh; ``all`` is the whole mesh), summed
  over the held-out cases. The absolute ``prediction``/``target`` blocks
  (crossed pairs, penetrating fraction, mean depth over all nodes with the
  tolerance subtracted, maximum depth) come from the same per-node depths.

Validation diagnostic only: a failure on any rank (pair budget, out of
memory, a non-finite frame) disables the metrics and never fails training
validation. Every rank logs its own failure; the reasons are gathered in
:meth:`PenetrationMetrics.reduce` and the summary then carries ``complete:
false``, ``failure_reasons`` and ``null`` in place of every count and depth,
so a disabled probe can never read as a perfect score. Results are written
under ``penetration`` in the validation JSON by
:class:`trainers.px_yaris_trainer.YarisFullTrainer`; ``evaluate_px_yaris.py`` fails
when the probe ends disabled.
"""

from __future__ import annotations

import logging
import math

import torch

from physicsnemo.experimental.nn.shell_contact import (
    ShellContactSearch,
    shell_contact_pairs,
    shell_penetration_depths,
    wall_penetration_depths,
)

_LOGGER = logging.getLogger(__name__)

SURFACE_FAMILIES = ("surface_envelope", "surface_crossing")
WALL_FAMILIES = ("wall_envelope", "wall_crossing")
FAMILIES = SURFACE_FAMILIES + WALL_FAMILIES
BASE_COUNTS = (
    "nodes",
    "prediction_penetrating",
    "target_penetrating",
    "spurious",
    "missed",
    "spurious_depth_sum",
    "excess_sum",
    "absolute_prediction_depth_sum",
    "absolute_target_depth_sum",
)
MAXIMA = (
    "prediction_max",
    "target_max",
    "spurious_max",
    "excess_max",
    "absolute_prediction_max",
    "absolute_target_max",
)
PAIRS = ("crossed_pairs", "active_pairs")


def region_names(bounds):
    """Names of the reference-axis regions plus ``all``, as the trainer spells them."""
    edges = [-float("inf"), *bounds, float("inf")]
    return [f"x_{lo:g}_to_{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])] + ["all"]


def histogram_percentiles(hist, bin_mm, quantiles):
    """Upper bin edges at the given quantiles of a histogram with an overflow bin."""
    total = float(hist.sum())
    if total <= 0:
        return {f"p{q:g}": 0.0 for q in quantiles}
    cumulative = torch.cumsum(hist.double(), 0)
    out = {}
    for q in quantiles:
        index = int(
            torch.searchsorted(cumulative, cumulative.new_tensor(q / 100 * total))
        )
        index = min(index, hist.numel() - 1)
        out[f"p{q:g}"] = float((index + 1) * bin_mm)
    return out


class PenetrationMetrics:
    """Accumulate target-relative penetration statistics over validated frames.

    ``cfg`` keys (all optional): ``tolerance_mm`` 0.5, ``activation_distance_mm``
    5, ``reference_clearance_mm`` 10, ``crossing_radius_mm`` (default ``None``:
    no crossing sweep, see below), ``implementation``
    ``warp``, ``max_pairs`` 60e6, ``depth_thresholds_mm`` [1, 2, 5],
    ``histogram_bin_mm`` 0.1, ``histogram_max_mm`` 50, ``region_axis`` 0,
    ``region_bounds_mm`` (default: the ``region_bounds`` argument),
    ``wall_position_mm`` (default: ``graph.rigid_wall`` when the datapipe sets
    one, else no wall statistics), ``wall_axis`` 0, ``wall_sign`` 1.

    The surface crossing depth has a detection ceiling: without
    ``crossing_radius_mm`` a node deeper through a shell than
    ``activation_distance_mm`` plus the half thicknesses of the pair has no
    candidate pair and reads as clean. The summary records the ceiling
    (``crossing_depth_ceiling_mm``) next to the crossing counts; a crossing
    radius lifts it to that radius plus the half thicknesses.
    """

    def __init__(self, cfg, rollout_steps, device, logger=None, region_bounds=()):
        cfg = dict(cfg or {})
        self.tolerance = float(cfg.get("tolerance_mm", 0.5))
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("validation_penetration.tolerance_mm must be >= 0")
        # Recorded with every summary so readers can tell the definition apart
        # from earlier ones; the depths are read from the pairs' signed gaps.
        self.sidedness = "symmetric"
        self.metric_version = 2
        crossing_radius = cfg.get("crossing_radius_mm")
        self.search = ShellContactSearch(
            activation_distance=float(cfg.get("activation_distance_mm", 5.0)),
            implementation=str(cfg.get("implementation", "warp")),
            max_pairs=int(cfg.get("max_pairs", 60_000_000)),
            material_fan=True,
            side_from_reference=True,
            reference_clearance=float(cfg.get("reference_clearance_mm", 10.0)),
            side_symmetric=True,
            crossing_radius=None if crossing_radius is None else float(crossing_radius),
        )
        self.thresholds = [float(v) for v in cfg.get("depth_thresholds_mm", [1, 2, 5])]
        if any(not math.isfinite(v) or v < 0 for v in self.thresholds):
            raise ValueError("depth_thresholds_mm must be finite and nonnegative")
        self.bin_mm = float(cfg.get("histogram_bin_mm", 0.1))
        self.hist_max_mm = float(cfg.get("histogram_max_mm", 50.0))
        if self.bin_mm <= 0 or self.hist_max_mm <= self.bin_mm:
            raise ValueError(
                "histogram_bin_mm must be positive and below histogram_max_mm"
            )
        self.bins = int(round(self.hist_max_mm / self.bin_mm))
        self.region_axis = int(cfg.get("region_axis", 0))
        # A declared ``region_bounds_mm: null`` defers to the trainer's bounds.
        bounds = cfg.get("region_bounds_mm")
        if bounds is None:
            bounds = region_bounds
        self.region_bounds = sorted(float(v) for v in (bounds or ()))
        self.regions = region_names(self.region_bounds)
        self.wall_position = cfg.get("wall_position_mm")
        self.wall_position = (
            None if self.wall_position is None else float(self.wall_position)
        )
        self.wall_axis = int(cfg.get("wall_axis", 0))
        self.wall_sign = float(cfg.get("wall_sign", 1.0))
        if self.wall_axis not in (0, 1, 2) or self.wall_sign not in (1.0, -1.0):
            raise ValueError("wall_axis must be 0, 1 or 2 and wall_sign +1 or -1")
        self.columns = BASE_COUNTS + tuple(
            f"spurious_above_{t:g}" for t in self.thresholds
        )
        steps, regions, families = int(rollout_steps), len(self.regions), len(FAMILIES)
        self.counts = torch.zeros(
            (steps, regions, families, len(self.columns)),
            dtype=torch.float64,
            device=device,
        )
        self.maxima = torch.zeros(
            (steps, regions, families, len(MAXIMA)), dtype=torch.float64, device=device
        )
        self.pairs = torch.zeros(
            (steps, 2, len(PAIRS)), dtype=torch.float64, device=device
        )
        self.hist = torch.zeros(
            (regions, families, self.bins + 1), dtype=torch.float64, device=device
        )
        # summed nodal thickness and node count over the validated designs, for
        # the crossing-depth ceiling recorded with the summary
        self.thickness_stats = torch.zeros(2, dtype=torch.float64, device=device)
        self.wall_seen = False
        self.logger = logger
        self.disabled = False
        self.failure_reasons = []

    # ----- settings -----
    def settings(self):
        """Return the metric settings recorded with every summary."""
        return dict(
            tolerance_mm=self.tolerance,
            activation_distance_mm=self.search.activation_distance,
            reference_clearance_mm=self.search.reference_clearance,
            crossing_radius_mm=self.search.crossing_radius,
            side_from_reference=True,
            sidedness=self.sidedness,
            metric_version=self.metric_version,
            depth_thresholds_mm=list(self.thresholds),
            histogram_bin_mm=self.bin_mm,
            histogram_max_mm=self.hist_max_mm,
            region_axis=self.region_axis,
            region_bounds_mm=list(self.region_bounds),
            wall_position_mm=self.wall_position,
            wall_axis=self.wall_axis,
            wall_sign=self.wall_sign,
        )

    # ----- per-frame geometry -----
    def surface_depths(self, positions, reference, thickness, graph):
        """Envelope and crossing depth per node [N], crossed and active pair counts."""
        pairs = shell_contact_pairs(
            positions,
            graph.contact_faces,
            thickness,
            reference_positions=reference,
            excluded_pairs=graph.contact_surface_exclusions,
            batch=getattr(graph, "batch", None),
            search=self.search,
        )
        envelope, crossing = shell_penetration_depths(pairs, positions.shape[0])
        return (
            envelope,
            crossing,
            float(pairs.stats["crossed_pairs"]),
            float(pairs.stats["active_pairs"]),
        )

    def wall_depths(self, positions, thickness, wall_position):
        """Envelope and crossing depth per node against the rigid wall plane."""
        return wall_penetration_depths(
            positions,
            thickness,
            wall_position,
            axis=self.wall_axis,
            sign=self.wall_sign,
        )

    def region_masks(self, reference):
        """[R, N] bool: the reference-axis regions and the whole mesh."""
        x = reference[:, self.region_axis]
        edges = [-float("inf"), *self.region_bounds, float("inf")]
        return torch.stack(
            [(x >= lo) & (x < hi) for lo, hi in zip(edges[:-1], edges[1:])]
            + [torch.ones_like(x, dtype=torch.bool)]
        )

    def _wall_position(self, graph):
        if self.wall_position is not None:
            return self.wall_position
        wall = getattr(graph, "rigid_wall", None)
        return None if wall is None else float(wall.reshape(-1)[0])

    # ----- accumulation -----
    def _family(self, step, family, d_pred, d_true, masks):
        tol = self.tolerance
        pred_pen = d_pred > tol
        true_pen = d_true > tol
        spurious = pred_pen & ~true_pen
        missed = true_pen & ~pred_pen
        excess = (d_pred - d_true).clamp_min(0.0)
        spurious_depth = torch.where(spurious, d_pred, torch.zeros_like(d_pred))
        absolute_pred = (d_pred - tol).clamp_min(0.0)
        absolute_true = (d_true - tol).clamp_min(0.0)
        columns = [
            torch.ones_like(d_pred),
            pred_pen.to(d_pred.dtype),
            true_pen.to(d_pred.dtype),
            spurious.to(d_pred.dtype),
            missed.to(d_pred.dtype),
            spurious_depth,
            excess,
            absolute_pred,
            absolute_true,
        ] + [(spurious & (d_pred > t)).to(d_pred.dtype) for t in self.thresholds]
        table = torch.stack(columns, dim=1).double()
        self.counts[step, :, family] += masks.double() @ table
        for r in range(masks.shape[0]):
            mask = masks[r]
            if not bool(mask.any()):
                continue
            self.maxima[step, r, family] = torch.maximum(
                self.maxima[step, r, family],
                torch.stack(
                    [
                        torch.where(mask, d_pred, torch.zeros_like(d_pred)).max(),
                        torch.where(mask, d_true, torch.zeros_like(d_true)).max(),
                        torch.where(
                            mask, spurious_depth, torch.zeros_like(d_pred)
                        ).max(),
                        torch.where(mask, excess, torch.zeros_like(excess)).max(),
                        torch.where(
                            mask, absolute_pred, torch.zeros_like(d_pred)
                        ).max(),
                        torch.where(
                            mask, absolute_true, torch.zeros_like(d_true)
                        ).max(),
                    ]
                ).double(),
            )
            selected = spurious_depth[mask & spurious]
            if selected.numel():
                inside = selected[selected < self.hist_max_mm]
                self.hist[r, family, : self.bins] += torch.histc(
                    inside.float(), bins=self.bins, min=0.0, max=self.hist_max_mm
                ).double()
                self.hist[r, family, self.bins] += float(
                    (selected >= self.hist_max_mm).sum()
                )

    def _disable(self, reason):
        """Disable the probe on this rank, keeping ``reason`` for the summary."""
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        message = f"rank {rank}: {reason}"
        self.disabled = True
        self.failure_reasons.append(message)
        self.counts.zero_()
        self.maxima.zero_()
        self.pairs.zero_()
        self.hist.zero_()
        # Every rank reports its own failure: the trainer's logger is a rank-zero
        # wrapper that would drop the message of any other rank.
        logger = self.logger if rank == 0 and self.logger is not None else _LOGGER
        logger.warning(f"Penetration metrics disabled: {message}")

    def accumulate(self, prediction, target, sample, data_stats):
        """Add every step of one case; ``prediction``/``target`` are [N, T, 3]."""
        if self.disabled:
            return
        graph = sample.graph
        if any(
            getattr(graph, name, None) is None
            for name in (
                "contact_faces",
                "contact_surface_exclusions",
                "shell_thickness",
            )
        ):
            self._disable("graph has no contact faces (datapipe.contact_surface=false)")
            return
        try:
            with torch.autocast(device_type=prediction.device.type, enabled=False):
                device = prediction.device
                std = data_stats["node"]["pos_std"].reshape(1, 3).to(device).float()
                mean = data_stats["node"]["pos_mean"].reshape(1, 3).to(device).float()
                reference = sample.node_features["coords"].float() * std + mean
                thickness = graph.shell_thickness.to(
                    device=device, dtype=torch.float32
                ).reshape(-1)
                self.thickness_stats += torch.stack(
                    (thickness.double().sum(), thickness.new_tensor(thickness.numel()))
                ).double()
                masks = self.region_masks(reference)
                wall = self._wall_position(graph)
                for step in range(int(prediction.shape[1])):
                    frames = [
                        prediction[:, step].float() * std + mean,
                        target[:, step].float() * std + mean,
                    ]
                    depths = [
                        self.surface_depths(frame, reference, thickness, graph)
                        for frame in frames
                    ]
                    for which in range(2):
                        self.pairs[step, which, 0] += depths[which][2]
                        self.pairs[step, which, 1] += depths[which][3]
                    self._family(step, 0, depths[0][0], depths[1][0], masks)
                    self._family(step, 1, depths[0][1], depths[1][1], masks)
                    if wall is not None:
                        self.wall_seen = True
                        walls = [
                            self.wall_depths(frame, thickness, wall) for frame in frames
                        ]
                        self._family(step, 2, walls[0][0], walls[1][0], masks)
                        self._family(step, 3, walls[0][1], walls[1][1], masks)
        except Exception as error:  # noqa: BLE001 - diagnostic only
            self._disable(f"{type(error).__name__}: {error}")

    def reduce(self, world_size):
        """Reduce the counters, histograms and flags across ranks."""
        if world_size <= 1:
            return
        flags = torch.tensor(
            [1.0 if self.disabled else 0.0, 1.0 if self.wall_seen else 0.0],
            device=self.counts.device,
        )
        for tensor in (self.counts, self.pairs, self.hist, self.thickness_stats):
            torch.distributed.all_reduce(tensor)
        torch.distributed.all_reduce(self.maxima, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, list(self.failure_reasons))
        self.failure_reasons = [reason for reasons in gathered for reason in reasons]
        self.wall_seen = bool(flags[1].item() > 0)
        if bool(flags[0].item() > 0):
            # Zero the counters everywhere so no rank reports partial numbers;
            # the gathered reasons already name the failing ranks.
            self.disabled = True
            self.counts.zero_()
            self.maxima.zero_()
            self.pairs.zero_()
            self.hist.zero_()
            if self.logger is not None:
                self.logger.warning(
                    "Penetration metrics disabled: " + "; ".join(self.failure_reasons)
                )

    # ----- summaries -----
    def _column(self, name):
        return self.columns.index(name)

    def _family_summary(self, horizon, family):
        counts = self.counts[:horizon, :, family].cpu()
        maxima = self.maxima[:horizon, :, family].cpu()
        hist = self.hist[:, family].cpu()
        c = self._column
        out = {"per_region": {}, "per_time": {}}
        for r, name in enumerate(self.regions):
            t = counts[:, r]
            node_frames = t[:, c("nodes")].sum().clamp_min(1.0)
            spurious = t[:, c("spurious")].sum()
            target_pen = t[:, c("target_penetrating")].sum()
            m = maxima[:, r].max(dim=0).values if horizon else torch.zeros(len(MAXIMA))
            out["per_region"][name] = dict(
                node_frames=float(t[:, c("nodes")].sum()),
                penetrating_fraction=dict(
                    prediction=float(
                        t[:, c("prediction_penetrating")].sum() / node_frames
                    ),
                    target=float(target_pen / node_frames),
                ),
                spurious_count=float(spurious),
                spurious_fraction=float(spurious / node_frames),
                missed_count=float(t[:, c("missed")].sum()),
                missed_fraction=float(
                    t[:, c("missed")].sum() / target_pen.clamp_min(1.0)
                ),
                excess_mean_mm=float(t[:, c("excess_sum")].sum() / node_frames),
                excess_max_mm=float(m[3]),
                spurious_depth_mean_mm=float(
                    t[:, c("spurious_depth_sum")].sum() / spurious.clamp_min(1.0)
                ),
                spurious_depth_max_mm=float(m[2]),
                spurious_depth_percentiles_mm=histogram_percentiles(
                    hist[r], self.bin_mm, (50, 90, 99, 99.9)
                ),
                spurious_above_mm={
                    f"{th:g}": dict(
                        count=float(t[:, c(f"spurious_above_{th:g}")].sum()),
                        fraction=float(
                            t[:, c(f"spurious_above_{th:g}")].sum() / node_frames
                        ),
                    )
                    for th in self.thresholds
                },
                max_depth_mm=dict(prediction=float(m[0]), target=float(m[1])),
            )
        t = counts[:, self.regions.index("all")]
        nodes = t[:, c("nodes")].clamp_min(1.0)
        out["per_time"] = dict(
            spurious_fraction=(t[:, c("spurious")] / nodes).tolist(),
            missed_fraction=(
                t[:, c("missed")] / t[:, c("target_penetrating")].clamp_min(1.0)
            ).tolist(),
            excess_mean_mm=(t[:, c("excess_sum")] / nodes).tolist(),
            penetrating_fraction_prediction=(
                t[:, c("prediction_penetrating")] / nodes
            ).tolist(),
            penetrating_fraction_target=(
                t[:, c("target_penetrating")] / nodes
            ).tolist(),
            spurious_depth_mean_mm=(
                t[:, c("spurious_depth_sum")] / t[:, c("spurious")].clamp_min(1.0)
            ).tolist(),
            spurious_depth_max_mm=maxima[:, self.regions.index("all"), 2].tolist(),
            **{
                f"spurious_above_{th:g}_fraction": (
                    t[:, c(f"spurious_above_{th:g}")] / nodes
                ).tolist()
                for th in self.thresholds
            },
        )
        return out

    def _absolute(self, horizon, which):
        """The absolute block for the prediction (0) or the target (1)."""
        r = self.regions.index("all")
        c = self._column
        t = self.counts[:horizon, r, 0].cpu()
        nodes = t[:, c("nodes")].clamp_min(1.0)
        pen = c("prediction_penetrating") if which == 0 else c("target_penetrating")
        depth = (
            c("absolute_prediction_depth_sum")
            if which == 0
            else c("absolute_target_depth_sum")
        )
        maxima = self.maxima[:horizon, r, 0, 4 + which].cpu()
        pairs = self.pairs[:horizon, which, 0].cpu()
        return dict(
            crossed_pairs_per_time=pairs.tolist(),
            penetrating_nodes_per_time=t[:, pen].tolist(),
            mean_depth_mm_per_time=(t[:, depth] / nodes).tolist(),
            max_depth_mm_per_time=maxima.tolist(),
            crossed_pairs=float(pairs.sum()),
            penetrating_node_fraction=float(t[:, pen].sum() / nodes.sum()),
            mean_depth_mm=float(t[:, depth].sum() / nodes.sum()),
            max_depth_mm=float(maxima.max()) if horizon else 0.0,
        )

    def crossing_depth_ceiling(self):
        """Deepest surface crossing the search can still see, in mm, plus a note.

        A node is paired with a facet only while its unsigned distance to it is
        inside the band (or the crossing radius); deeper crossings have no pair
        and read zero. The half thicknesses of the pair add to the ceiling, so
        the mean thickness of the validated designs stands in for them.
        """
        count = float(self.thickness_stats[1])
        mean_thickness = (
            float(self.thickness_stats[0]) / count if count > 0 else float("nan")
        )
        radius = self.search.crossing_radius
        reach = self.search.activation_distance if radius is None else radius
        return dict(
            crossing_depth_ceiling_mm=(
                reach + mean_thickness if math.isfinite(mean_thickness) else None
            ),
            crossing_depth_ceiling_note=(
                ("activation_distance_mm" if radius is None else "crossing_radius_mm")
                + " plus the mean shell thickness of the validated designs "
                + f"({mean_thickness:.3f} mm); a node deeper through a shell than "
                "this has no candidate pair and its surface depths read zero"
                + (
                    "; set crossing_radius_mm to lift the ceiling"
                    if radius is None
                    else ""
                )
            ),
        )

    @staticmethod
    def _nullify(value):
        """Replace every number of a JSON-ready block by ``None`` (booleans kept)."""
        if isinstance(value, dict):
            return {k: PenetrationMetrics._nullify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PenetrationMetrics._nullify(v) for v in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return None
        return value

    def summary(self, nodes_w_time):
        """JSON-ready statistics: the absolute blocks plus the target-relative families.

        When the probe ended disabled on any rank, every count and depth is
        ``null`` and ``failure_reasons`` lists the ranks' reasons.
        """
        horizon = int(nodes_w_time.numel())
        out = dict(
            self.settings(),
            complete=not self.disabled,
            failure_reasons=list(self.failure_reasons),
            **self.crossing_depth_ceiling(),
        )
        out["prediction"] = self._absolute(horizon, 0)
        out["target"] = self._absolute(horizon, 1)
        out["pairs_per_time"] = {
            f"{PAIRS[k]}_{name}": self.pairs[:horizon, which, k].cpu().tolist()
            for which, name in ((0, "prediction"), (1, "target"))
            for k in range(len(PAIRS))
        }
        out["surface"] = {
            "envelope": self._family_summary(horizon, 0),
            "crossing": self._family_summary(horizon, 1),
        }
        out["wall"] = (
            {
                "envelope": self._family_summary(horizon, 2),
                "crossing": self._family_summary(horizon, 3),
            }
            if self.wall_seen
            else None
        )
        if self.disabled:
            for key in ("prediction", "target", "pairs_per_time", "surface", "wall"):
                out[key] = self._nullify(out[key])
        return out

    def log_line(self, summary):
        """Format the summary as one log line."""
        if not summary["complete"]:
            return "DISABLED (" + "; ".join(summary.get("failure_reasons", [])) + ")"
        parts = []
        for name in ("prediction", "target"):
            s = summary[name]
            parts.append(
                f"{name}: crossed {s['crossed_pairs']:.0f}, penetrating "
                f"{100 * s['penetrating_node_fraction']:.3f}% of nodes, mean depth "
                f"{s['mean_depth_mm']:.4f} mm, max {s['max_depth_mm']:.2f} mm"
            )
        for label, block in (
            ("surface", summary.get("surface")),
            ("wall", summary.get("wall")),
        ):
            if not block:
                continue
            for kind in ("envelope", "crossing"):
                s = block[kind]["per_region"]["all"]
                above = ", ".join(
                    f">{th}mm {100 * v['fraction']:.4f}%"
                    for th, v in s["spurious_above_mm"].items()
                )
                parts.append(
                    f"{label} {kind}: spurious {100 * s['spurious_fraction']:.4f}% of nodes "
                    f"(p99 {s['spurious_depth_percentiles_mm']['p99']:.2f} mm, {above}), "
                    f"missed {100 * s['missed_fraction']:.1f}% of target, "
                    f"excess mean {s['excess_mean_mm']:.5f} mm max {s['excess_max_mm']:.2f} mm"
                )
        return " | ".join(parts)
