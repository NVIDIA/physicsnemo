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
"""Temporal smoothness and per-part kinematics of validated trajectories.

The pooled position RMSE is dominated by the late frames and the crushed
front, so two models can tie on it while differing in what an engineer reads
first. This validation diagnostic (``training.validation_kinematics``) adds
two views, both from the predicted and the true frames of each held-out case:

* **Smoothness in time.** The acceleration of every node as the second
  difference of its position over the frame interval, with the reference
  frame as the point before the first future frame. Accumulated per step and
  region: the squared acceleration error, the true and the predicted squared
  acceleration (a model that predicts frames independently can be noisier in
  time than the truth, which the position error does not show).
* **Per-part kinematics.** The least-squares rigid motion of every part fitted
  to the predicted and to the true frame (:func:`fit_part_rigid_motion`, Kabsch):
  the node-weighted error of the part translations per step and region, the
  deceleration pulse per part (second difference of the part translation) and
  its error, and the whole-vehicle centroid error. The per-case part
  trajectories are saved so individual parts can be plotted.

Units: millimetres, seconds, accelerations in g.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

G_MM_S2 = 9806.65
"""Standard gravity in mm/s^2 (accelerations are reported in g)."""


def part_counts(part_index: torch.Tensor, num_parts: int) -> torch.Tensor:
    """Nodes per part, ``[P]`` float64, never below one."""
    return (
        torch.bincount(part_index.reshape(-1), minlength=num_parts)
        .to(torch.float64)
        .clamp_min(1.0)
    )


def part_means(
    values: torch.Tensor,
    part_index: torch.Tensor,
    num_parts: int,
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of ``values`` ``[N, C]`` over the nodes of every part, ``[P, C]``."""
    if counts is None:
        counts = part_counts(part_index, num_parts)
    sums = torch.zeros(
        (num_parts, values.shape[1]), device=values.device, dtype=values.dtype
    ).index_add_(0, part_index, values)
    return sums / counts.to(values.dtype)[:, None]


def fit_part_rigid_motion(
    reference: torch.Tensor,
    target: torch.Tensor,
    part_index: torch.Tensor,
    num_parts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares rigid motion of every part from ``reference`` to ``target``.

    Parameters
    ----------
    reference, target : Tensor ``[N, 3]``
        Node positions of the two configurations in the same (physical) units.
    part_index : LongTensor ``[N]``
        Contiguous part index per node (``0 .. P-1``; any grouping works).
    num_parts : int
        Number of parts ``P``.

    Returns
    -------
    rotation : Tensor ``[P, 3, 3]`` (float64)
        Proper rotation minimising ``sum |R (x - c) - (y - c')|^2`` over the
        part's nodes (Kabsch, SVD of the per-part cross-covariance). A part
        with one node or collinear nodes still gets a valid rotation.
    translation : Tensor ``[P, 3]`` (float64)
        Displacement of the part centroid ``c' - c``.
    centroid : Tensor ``[P, 3]`` (float64)
        Reference centroid ``c`` of every part, the rotation's pivot.
    """
    part_index = part_index.reshape(-1)
    if reference.shape != target.shape or reference.shape[-1] != 3:
        raise ValueError("reference and target must both be [N, 3]")
    if part_index.numel() != reference.shape[0]:
        raise ValueError("part_index must have one entry per node")
    ref = reference.to(torch.float64)
    tgt = target.to(torch.float64)
    counts = part_counts(part_index, num_parts)
    centroid = part_means(ref, part_index, num_parts, counts)
    target_centroid = part_means(tgt, part_index, num_parts, counts)
    a = ref - centroid[part_index]
    b = tgt - target_centroid[part_index]
    covariance = torch.zeros(
        (num_parts, 3, 3), device=ref.device, dtype=torch.float64
    ).index_add_(0, part_index, a[:, :, None] * b[:, None, :])
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    # Reflections are not rigid: flip the least significant axis when needed.
    sign = torch.linalg.det(v @ u.transpose(-1, -2)).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    correction = torch.ones((num_parts, 3), device=ref.device, dtype=torch.float64)
    correction[:, 2] = sign
    rotation = (v * correction[:, None, :]) @ u.transpose(-1, -2)
    return rotation, target_centroid - centroid, centroid


class KinematicsReadout:
    """Accumulate the temporal-smoothness and per-part kinematics statistics of
    validated trajectories (see the module docstring).

    Parameters
    ----------
    part_index : torch.Tensor
        ``[N]`` contiguous part index per node.
    part_ids : array-like
        Label of every part (``[P]``), stored with the saved trajectories.
    reference_mm : torch.Tensor
        ``[N, 3]`` reference positions in millimetres.
    dt : float
        Frame interval in seconds.
    region_bounds : sequence of float
        Reference-x bounds splitting the mesh into regions.
    rollout_steps : int
        Maximum number of future frames.
    device : torch.device
        Accumulation device.
    logger : logging.Logger, optional
        Logger for messages.
    """

    def __init__(
        self,
        part_index: torch.Tensor,
        part_ids,
        reference_mm: torch.Tensor,
        dt: float,
        region_bounds,
        rollout_steps: int,
        device,
        logger=None,
    ):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.dt = float(dt)
        if not self.dt > 0:
            raise ValueError("dt must be positive")
        self.part_index = part_index.reshape(-1).long().to(device)
        self.part_ids = np.asarray(part_ids)
        self.num_parts = int(len(self.part_ids))
        if int(self.part_index.max()) >= self.num_parts:
            raise ValueError("part_index exceeds the number of parts")
        self.reference = reference_mm.to(device=device, dtype=torch.float64)
        self.counts = torch.bincount(self.part_index, minlength=self.num_parts).to(
            torch.float64
        )
        self.centroid = part_means(
            self.reference, self.part_index, self.num_parts, self.counts
        )
        bounds = [-float("inf"), *[float(b) for b in region_bounds], float("inf")]
        self.region_names = [
            f"x_{lo:g}_to_{hi:g}" for lo, hi in zip(bounds[:-1], bounds[1:])
        ] + ["all"]
        x = self.reference[:, 0]
        node_masks = [(x >= lo) & (x < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
        self.node_masks = torch.stack(
            node_masks + [torch.ones_like(x, dtype=torch.bool)]
        )  # [R, N]
        cx = self.centroid[:, 0]
        part_masks = [(cx >= lo) & (cx < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
        self.part_masks = torch.stack(
            part_masks + [torch.ones_like(cx, dtype=torch.bool)]
        )  # [R, P]
        self.part_weight = self.counts / self.counts.sum()
        T, R = int(rollout_steps), self.node_masks.shape[0]
        f64 = dict(dtype=torch.float64, device=device)
        # Node acceleration per step and region: error^2, true^2, predicted^2 sums and node-frame counts.
        self.acc = torch.zeros((T, R, 4), **f64)
        # Part kinematics per step and region: node-weighted translation error^2, pulse error^2,
        # true pulse^2, predicted pulse^2, the translation weight sum and the pulse weight
        # sum (the pulse has one entry fewer than the translation: its stencil needs a
        # frame on each side).
        self.parts = torch.zeros((T, R, 6), **f64)
        # Whole-vehicle centroid error^2 per step and frame count.
        self.vehicle = torch.zeros((T, 2), **f64)
        self.cases = []  # (name, predicted translation [T_c, P, 3], true translation [T_c, P, 3]) as float32 numpy

    @torch.no_grad()
    def accumulate(self, prediction, target, sample, data_stats):
        """Accumulate one case's kinematic errors from predicted and true positions."""
        std = data_stats["node"]["pos_std"].to(self.device).double().reshape(1, 3)
        mean = data_stats["node"]["pos_mean"].to(self.device).double().reshape(1, 3)
        steps = int(target.shape[1])
        name = getattr(sample, "yaris_case_name", str(len(self.cases)))
        # Positions in mm with the reference first: [T+1] frames.
        pred_prev2, pred_prev1 = None, self.reference
        true_prev2, true_prev1 = None, self.reference
        pt = torch.zeros(
            (steps, self.num_parts, 3), dtype=torch.float64, device=self.device
        )
        tt = torch.zeros_like(pt)
        dt2 = self.dt**2
        for k in range(steps):
            pred = prediction[:, k].double() * std + mean
            true = target[:, k].double() * std + mean
            # Acceleration centred on the previous frame needs two neighbours.
            if pred_prev2 is not None:
                a_pred = (pred - 2.0 * pred_prev1 + pred_prev2) / dt2 / G_MM_S2
                a_true = (true - 2.0 * true_prev1 + true_prev2) / dt2 / G_MM_S2
                err = (a_pred - a_true).square().sum(-1)
                tr = a_true.square().sum(-1)
                pr = a_pred.square().sum(-1)
                for r in range(self.node_masks.shape[0]):
                    m = self.node_masks[r]
                    self.acc[k - 1, r, 0] += err[m].sum()
                    self.acc[k - 1, r, 1] += tr[m].sum()
                    self.acc[k - 1, r, 2] += pr[m].sum()
                    self.acc[k - 1, r, 3] += float(m.sum())
            pred_prev2, pred_prev1 = pred_prev1, pred
            true_prev2, true_prev1 = true_prev1, true
            _, pt[k], _ = fit_part_rigid_motion(
                self.reference, pred, self.part_index, self.num_parts
            )
            _, tt[k], _ = fit_part_rigid_motion(
                self.reference, true, self.part_index, self.num_parts
            )
        # Translation error per part and step, node-weighted per region.
        terr = (pt - tt).square().sum(-1)  # [T_c, P]
        for r in range(self.part_masks.shape[0]):
            w = self.part_weight * self.part_masks[r]
            self.parts[:steps, r, 0] += terr @ w
            self.parts[:steps, r, 4] += w.sum()
        # Pulse: second difference of the part translation centred on predicted
        # frame j, with the reference as the single frame before frame 0 (the
        # stencil of the node acceleration), so that entry j aligns with acc[j].
        zero = torch.zeros(
            (1, self.num_parts, 3), dtype=torch.float64, device=self.device
        )
        p_seq = torch.cat((zero, pt), 0)
        t_seq = torch.cat((zero, tt), 0)
        p_acc = (
            (p_seq[2:] - 2 * p_seq[1:-1] + p_seq[:-2]) / dt2 / G_MM_S2
        )  # [T_c - 1, P, 3]
        t_acc = (t_seq[2:] - 2 * t_seq[1:-1] + t_seq[:-2]) / dt2 / G_MM_S2
        perr = (p_acc - t_acc).square().sum(-1)
        ptrue = t_acc.square().sum(-1)
        ppred = p_acc.square().sum(-1)
        for r in range(self.part_masks.shape[0]):
            w = self.part_weight * self.part_masks[r]
            self.parts[: steps - 1, r, 1] += perr @ w
            self.parts[: steps - 1, r, 2] += ptrue @ w
            self.parts[: steps - 1, r, 3] += ppred @ w
            self.parts[: steps - 1, r, 5] += w.sum()
        # Whole vehicle: node-weighted mean translation of all parts.
        v_err = (
            ((pt - tt) * self.part_weight[None, :, None]).sum(1).square().sum(-1)
        )  # [T_c]
        self.vehicle[:steps, 0] += v_err
        self.vehicle[:steps, 1] += 1.0
        self.cases.append((name, pt.float().cpu().numpy(), tt.float().cpu().numpy()))

    def reduce(self, world_size: int) -> None:
        """Sum the accumulators and gather the case names across ranks."""
        if world_size <= 1:
            return
        for tensor in (self.acc, self.parts, self.vehicle):
            torch.distributed.all_reduce(tensor)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, self.cases)
        self.cases = [case for cases in gathered for case in cases]

    def summary(self) -> dict:
        """Return the accumulated kinematic metrics as a nested dictionary."""
        acc = self.acc.cpu()
        parts = self.parts.cpu()
        vehicle = self.vehicle.cpu()
        out = {
            "dt_s": self.dt,
            "regions": self.region_names,
            "part_count": self.num_parts,
            "units": "mm, s, g",
        }

        def rms(s, n):
            return float(torch.sqrt(s.sum() / max(float(n.sum()), 1.0)))

        def per_time(s, n):
            # Steps nobody reached serialise as null, never NaN.
            return [
                math.sqrt(a / b) if b > 0 else None
                for a, b in zip(s.tolist(), n.tolist())
            ]

        smooth = {}
        for r, region in enumerate(self.region_names):
            n = acc[:, r, 3]
            smooth[region] = dict(
                acceleration_error_g=rms(acc[:, r, 0], n),
                true_acceleration_g=rms(acc[:, r, 1], n),
                predicted_acceleration_g=rms(acc[:, r, 2], n),
                acceleration_error_g_per_time=per_time(acc[:, r, 0], n),
                true_acceleration_g_per_time=per_time(acc[:, r, 1], n),
                predicted_acceleration_g_per_time=per_time(acc[:, r, 2], n),
            )
            t, p = (
                smooth[region]["true_acceleration_g"],
                smooth[region]["predicted_acceleration_g"],
            )
            smooth[region]["jitter_ratio"] = float(p / t) if t > 0 else None
        out["smoothness"] = smooth
        kin = {}
        for r, region in enumerate(self.region_names):
            w, wp = parts[:, r, 4], parts[:, r, 5]
            kin[region] = dict(
                part_translation_error_mm=rms(parts[:, r, 0], w),
                part_translation_error_mm_per_time=per_time(parts[:, r, 0], w),
                pulse_error_g=rms(parts[:, r, 1], wp),
                true_pulse_g=rms(parts[:, r, 2], wp),
                predicted_pulse_g=rms(parts[:, r, 3], wp),
                pulse_error_g_per_time=per_time(parts[:, r, 1], wp),
                true_pulse_g_per_time=per_time(parts[:, r, 2], wp),
            )
        out["parts"] = kin
        out["vehicle_centroid_error_mm"] = rms(vehicle[:, 0], vehicle[:, 1])
        out["vehicle_centroid_error_mm_per_time"] = per_time(
            vehicle[:, 0], vehicle[:, 1]
        )
        out["cases"] = sorted(name for name, _, _ in self.cases)
        return out

    def log_line(self, summary: dict) -> str:
        """Format the summary as one log line."""
        s = summary["smoothness"]["all"]
        k = summary["parts"]["all"]
        return (
            f"accel err {s['acceleration_error_g']:.2f} g (true {s['true_acceleration_g']:.2f}, "
            f"pred {s['predicted_acceleration_g']:.2f}) | part translation err {k['part_translation_error_mm']:.2f} mm | "
            f"pulse err {k['pulse_error_g']:.2f} g (true {k['true_pulse_g']:.2f}) | "
            f"vehicle centroid err {summary['vehicle_centroid_error_mm']:.2f} mm"
        )

    def save(self, path) -> None:
        """Per-case per-part translation trajectories (predicted and true), padded with NaN."""
        if not self.cases:
            return
        cases = sorted(self.cases, key=lambda c: c[0])
        horizon = max(c[1].shape[0] for c in cases)
        pred = np.full(
            (len(cases), horizon, self.num_parts, 3), np.nan, dtype=np.float32
        )
        true = np.full_like(pred, np.nan)
        for i, (_, p, t) in enumerate(cases):
            pred[i, : p.shape[0]], true[i, : t.shape[0]] = p, t
        np.savez(
            path,
            cases=np.array([c[0] for c in cases]),
            part_ids=self.part_ids.astype(np.int64),
            part_counts=self.counts.cpu().numpy().astype(np.int64),
            part_centroid=self.centroid.float().cpu().numpy(),
            predicted_translation=pred,
            true_translation=true,
            dt_s=np.array(self.dt),
        )
