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

"""Train PINN variants for the steel bridge case and benchmark them against CAE.

Experiments
-----------
``fea``
    CAE reference only: mesh convergence, beam-theory verification, field plots.
``pinn_physics``
    Purely physics-driven PINN (total potential energy, no measurements).
``pinn_warm``
    The same PINN warm started from a short supervised fit on a handful of CAE
    measurements, then trained on energy + data.
``supervised``
    Data-only surrogate trained on the same measurements (no physics): the ablation
    that shows what the governing equations add.
``compare``
    Run the CAE model and all three learning variants, plus the data-efficiency
    sweep, and write tables and figures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")

import compare as comparison  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from steel_bridge_cae import (  # noqa: E402
    PlaneStressSolution,
    QuadMesh,
    SteelBridge,
    convergence_study,
    solve_plane_stress,
)
from steel_bridge_pinn import (  # noqa: E402
    BridgeEnergyLoss,
    BridgePINN,
    BridgeSampler,
    LossScales,
    analytic_scales,
)
from torch.optim import Adam  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

__all__ = ["main"]

VARIANTS = ("pinn_physics", "pinn_warm", "supervised")


@dataclass
class TrainResult:
    """Outcome of one training run."""

    name: str
    model: BridgePINN
    history: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def build_bridge(cfg: DictConfig) -> SteelBridge:
    """Instantiate the bridge description from the hydra config."""
    return SteelBridge(**OmegaConf.to_container(cfg.bridge, resolve=True))


def build_model(
    cfg: DictConfig, bridge: SteelBridge, scales: LossScales, device: torch.device
) -> BridgePINN:
    model = BridgePINN(
        bridge=bridge,
        displacement_scale=scales.displacement,
        num_layers=cfg.model.num_layers,
        layer_size=cfg.model.layer_size,
        activation=cfg.model.activation,
        skip_connections=cfg.model.skip_connections,
        bearing_halo=cfg.model.bearing_halo,
    )
    return model.to(device)


def decay_scheduler(
    optimizer: torch.optim.Optimizer, initial_lr: float, final_lr: float, steps: int
) -> LambdaLR:
    """Exponential decay from ``initial_lr`` to ``final_lr`` over ``steps``."""
    ratio = (final_lr / initial_lr) ** (1.0 / max(steps - 1, 1))
    return LambdaLR(optimizer, lr_lambda=lambda step: ratio**step)


def sensor_batch(
    solution: PlaneStressSolution, n_sensors: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sparse CAE readings used as data-consistency targets."""
    if n_sensors <= 0:
        return None
    points, displacement = solution.sensor_measurements(n_sensors, seed=seed)
    return BridgeSampler.data_points(points, displacement)


def sensor_metrics(
    model: BridgePINN, data: tuple[torch.Tensor, torch.Tensor] | None
) -> dict[str, float]:
    """How well the network reproduces the sparse measurements it was trained on."""
    if data is None:
        return {"sensor_rms_error_mm": float("nan")}
    with torch.no_grad():
        error = (model(data[0]) - data[1]).norm(dim=-1)
    return {
        "sensor_rms_error_mm": float(error.pow(2).mean().sqrt() * 1e3),
        "sensor_max_error_mm": float(error.max() * 1e3),
    }


def train_variant(
    name: str,
    cfg: DictConfig,
    bridge: SteelBridge,
    solution: PlaneStressSolution,
    scales: LossScales,
    device: torch.device,
    *,
    n_sensors: int,
    log: PythonLogger | None = None,
    overrides: dict | None = None,
) -> TrainResult:
    """Train one PINN variant and evaluate it against the CAE reference."""
    spec = OmegaConf.to_container(cfg.variants[name], resolve=True)
    if overrides:
        spec.update(overrides)
    steps = int(spec["steps"])
    n_interior = int(spec["interior_points"])
    initial_lr = float(spec["initial_lr"])
    clip = float(spec.get("clip_grad_norm", 0.0)) or None
    use_data = bool(spec["use_data"]) and n_sensors > 0
    data_only = bool(spec.get("data_only", False))

    sampler = BridgeSampler(bridge, seed=cfg.seed)
    model = build_model(cfg, bridge, scales, device)
    energy_loss = BridgeEnergyLoss(bridge=bridge, scales=scales, device=device)
    data = sensor_batch(solution, n_sensors, cfg.cae.sensor_seed) if use_data else None
    data_weight = float(cfg.training.data_weight)
    start = time.perf_counter()

    # ------------------------------------------------------------ supervised warm start
    if data is not None and spec.get("warm_start", False):
        warm_steps = int(cfg.training.warmstart_steps)
        optimizer = Adam(model.parameters(), lr=float(cfg.training.warmstart_lr))
        for _ in range(warm_steps):
            optimizer.zero_grad()
            mismatch = energy_loss.data(model, *data)
            loss = mismatch.pow(2).mean()
            loss.backward()
            optimizer.step()
        if log is not None:
            log.info(
                f"[{name}] supervised warm start over {warm_steps} steps, "
                f"data loss {float(loss.detach()):.3e}"
            )

    # ------------------------------------------------------------------- main loop
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = decay_scheduler(optimizer, initial_lr, cfg.training.final_lr, steps)
    result = TrainResult(name=name, model=model)
    progress_grid = (cfg.evaluation.progress_nx, cfg.evaluation.progress_ny)

    for step in range(steps):
        optimizer.zero_grad()
        if data_only:
            loss = data_weight * energy_loss.data(model, *data).pow(2).mean()
            contributions = {"data": float(loss.detach())}
        else:
            energy, contributions = energy_loss.energy(
                model, sampler, n_interior, int(cfg.training.patch_points)
            )
            # Ramp the energy term in over the first few hundred steps.  A
            # warm-started field fits the measurements but is usually too stiff, so
            # switching the physics on abruptly would discard the warm start.
            ramp = int(cfg.training.energy_ramp_steps)
            scale = 1.0 if ramp <= 0 else min(1.0, (step + 1) / ramp)
            loss = scale * energy
            if data is not None:
                mismatch = energy_loss.data(model, *data)
                data_term = data_weight * mismatch.pow(2).mean()
                loss = loss + data_term
                contributions["data"] = float(data_term.detach())

        if not torch.isfinite(loss):
            if log is not None:
                log.warning(f"[{name}] non-finite loss at step {step}, stopping")
            break

        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.training.log_every == 0 or step == steps - 1:
            model.eval()
            metrics = comparison.evaluate_model(
                model, bridge, energy_loss, solution, *progress_grid
            )
            model.train()
            row = {
                "step": float(step),
                "loss": float(loss.detach()),
                **contributions,
                "rel_l2_displacement": metrics["rel_l2_displacement"],
                "midspan_deflection_mm": metrics["midspan_deflection_mm"],
            }
            result.history.append(row)
            if log is not None:
                log.info(
                    f"[{name}] step {step:6d} loss {row['loss']:+.3e} "
                    f"rel_L2 {row['rel_l2_displacement']:.3e} "
                    f"midspan {row['midspan_deflection_mm']:.3f} mm"
                )

    result.wall_time = time.perf_counter() - start

    # ------------------------------------------------------------- final evaluation
    model.eval()
    result.metrics = comparison.evaluate_model(
        model, bridge, energy_loss, solution, cfg.evaluation.nx, cfg.evaluation.ny
    )
    balance = (
        {"energy_balance": float("nan")}
        if data_only
        else energy_loss.energy_balance(
            model, sampler, n_interior, int(cfg.training.patch_points)
        )
    )
    result.metrics.update(
        balance
        | sensor_metrics(model, data)
        | {
            "variant": name,
            "sensors": float(0 if data is None else n_sensors),
            "steps": float(steps),
            "parameters": float(result.parameters),
            "wall_time_s": result.wall_time,
        }
    )
    return result


def plot_cae_overview(solution: PlaneStressSolution, output_dir: Path) -> None:
    """CAE reference fields and the deflection profile along the span."""
    bridge = solution.bridge
    magnitude, stress, _ = solution.grid_fields(241, 61)
    x = np.linspace(0.0, bridge.span, 241)
    bottom = solution.displacement_at(np.stack([x, np.zeros_like(x)], axis=-1))[:, 1]
    deck = solution.displacement_at(
        np.stack([x, np.full_like(x, bridge.depth)], axis=-1)
    )[:, 1]
    theory = bridge.beam_theory()

    extent = [0.0, bridge.span, 0.0, bridge.depth]
    fig, axes = plt.subplots(1, 3, figsize=(13, 2.9), constrained_layout=True)
    image = axes[0].imshow(
        (magnitude * 1e3).T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_title("CAE: |u| [mm]", fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=0.9)
    image = axes[1].imshow(
        (stress / 1e6).T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(stress, 99.5)) / 1e6,
    )
    axes[1].set_title("CAE: von Mises [MPa]", fontsize=10)
    fig.colorbar(image, ax=axes[1], shrink=0.9)
    axes[2].plot(x, -bottom * 1e3, label="bottom flange")
    axes[2].plot(x, -deck * 1e3, label="deck")
    axes[2].axhline(
        theory.midspan_deflection * 1e3,
        color="k",
        linestyle=":",
        label="beam theory (midspan)",
    )
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("deflection [mm]")
    axes[2].set_title("Deflection along the span", fontsize=10)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    fig.savefig(output_dir / "cae_reference.png", dpi=150)
    plt.close(fig)


def run_fea(
    cfg: DictConfig, bridge: SteelBridge, output_dir: Path, log: PythonLogger
) -> PlaneStressSolution:
    """Solve the CAE reference and verify it against beam theory."""
    theory = bridge.beam_theory()
    log.info(
        "beam theory: M_max %.1f kNm, sigma_b %.1f MPa, midspan deflection %.2f mm"
        % (
            theory.max_moment / 1e3,
            theory.max_bending_stress / 1e6,
            theory.midspan_deflection * 1e3,
        )
    )

    study = convergence_study(bridge, [tuple(item) for item in cfg.cae.convergence])
    for row in study:
        log.info(
            "mesh %4dx%-3d elements %6d : midspan %.3f mm (error %+.2f%%)"
            % (
                row["nx"],
                row["ny"],
                row["elements"],
                row["midspan_deflection"] * 1e3,
                row["deflection_error"] * 100.0,
            )
        )

    start = time.perf_counter()
    solution = solve_plane_stress(
        bridge,
        QuadMesh(nx=cfg.cae.nx, ny=cfg.cae.ny, span=bridge.span, depth=bridge.depth),
    )
    log.info(
        "reference solve: %d elements, midspan %.3f mm, wall time %.2f s"
        % (
            cfg.cae.nx * cfg.cae.ny,
            solution.midspan_deflection() * 1e3,
            time.perf_counter() - start,
        )
    )
    plot_cae_overview(solution, output_dir)

    applied = (
        bridge.total_load
        + bridge.density
        * bridge.gravity
        * bridge.span
        * bridge.depth
        * bridge.thickness
    )
    summary = {
        "beam_theory": {
            "max_moment_knm": theory.max_moment / 1e3,
            "max_bending_stress_mpa": theory.max_bending_stress / 1e6,
            "midspan_deflection_mm": theory.midspan_deflection * 1e3,
        },
        "mesh_convergence": study,
        "reference": {
            "elements": cfg.cae.nx * cfg.cae.ny,
            "midspan_deflection_mm": solution.midspan_deflection() * 1e3,
            "deflection_error_percent": abs(
                solution.midspan_deflection() / theory.midspan_deflection - 1.0
            )
            * 100.0,
            "total_reaction_kn": float(solution.reactions[:, 1].sum() / 1e3),
            "applied_load_kn": float(applied / 1e3),
        },
    }
    with open(output_dir / "cae_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return solution


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    torch.set_num_threads(int(cfg.threads))
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log = PythonLogger(name="steel_bridge_pinns")
    log.info(f"experiment: {cfg.experiment}")

    bridge = build_bridge(cfg)
    scales = analytic_scales(bridge)
    solution = run_fea(cfg, bridge, output_dir, log)

    if cfg.experiment == "fea":
        log.info("CAE reference complete, no training requested.")
        return
    if cfg.experiment not in VARIANTS and cfg.experiment != "compare":
        raise ValueError(f"Unknown experiment '{cfg.experiment}'")

    names = [cfg.experiment] if cfg.experiment in VARIANTS else list(VARIANTS)
    results: dict[str, TrainResult] = {}
    for name in names:
        results[name] = train_variant(
            name,
            cfg,
            bridge,
            solution,
            scales,
            dist.device,
            n_sensors=int(cfg.cae.sensor_count),
            log=log,
        )

    sweep_rows: list[dict[str, float]] = []
    if cfg.experiment == "compare" and cfg.sweep.enabled:
        for n_sensors in cfg.sweep.sensor_counts:
            for name in ("pinn_warm", "supervised"):
                sweep = train_variant(
                    name,
                    cfg,
                    bridge,
                    solution,
                    scales,
                    dist.device,
                    n_sensors=int(n_sensors),
                    log=log,
                    overrides={"steps": int(cfg.sweep.steps)},
                )
                sweep_rows.append(sweep.metrics)

    # -------------------------------------------------------------------- reporting
    fields_x = np.linspace(0.0, bridge.span, cfg.evaluation.nx)
    fields_y = np.linspace(0.0, bridge.depth, cfg.evaluation.ny)
    rows = []
    for name, result in results.items():
        energy_loss = BridgeEnergyLoss(bridge=bridge, scales=scales, device=dist.device)
        prediction = comparison.predict_fields(
            result.model, bridge, energy_loss, fields_x, fields_y
        )
        comparison.plot_field_comparison(
            solution,
            prediction,
            output_dir / f"fields_{name}.png",
            title=f"{name}: CAE vs PINN",
        )
        comparison.plot_error_maps(
            solution, prediction, output_dir / f"error_{name}.png"
        )
        rows.append(result.metrics)
        log.info(
            f"{name}: rel L2 displacement "
            f"{result.metrics['rel_l2_displacement']:.3e}, midspan "
            f"{result.metrics['midspan_deflection_mm']:.3f} mm vs "
            f"{result.metrics['midspan_deflection_reference_mm']:.3f} mm, "
            f"wall time {result.wall_time:.1f} s"
        )

    comparison.plot_convergence(
        {name: result.history for name, result in results.items()},
        output_dir / "convergence.png",
        reference_deflection_mm=float(solution.midspan_deflection() * 1e3),
    )
    if sweep_rows:
        comparison.plot_data_efficiency(sweep_rows, output_dir / "data_efficiency.png")

    columns = [
        "variant",
        "sensors",
        "rel_l2_displacement",
        "rel_l2_von_mises",
        "midspan_deflection_mm",
        "bottom_flange_stress_mpa",
        "sensor_rms_error_mm",
        "energy_balance",
        "wall_time_s",
        "parameters",
    ]
    table = comparison.format_table(rows, columns)
    (output_dir / "comparison.md").write_text(table + "\n")
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(
            {
                "variants": rows,
                "data_efficiency": sweep_rows,
                "history": {name: result.history for name, result in results.items()},
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            handle,
            indent=2,
        )
    log.info("summary\n" + table)

    for name, result in results.items():
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "displacement_scale": scales.displacement,
                "metrics": result.metrics,
            },
            output_dir / f"model_{name}.pt",
        )


if __name__ == "__main__":
    main()
