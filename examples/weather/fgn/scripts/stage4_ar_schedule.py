# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper Stage-4 AR-finetune scheduler (arXiv:2506.10772v1 Table A.2).

Chains multiple training stages at increasing ``ar_steps``. Table A.2
Stage 4 is:

    8000 steps at 1AR, then
    4000 steps at 2AR, then
    1000 steps each at 3AR, 4AR, 5AR, 6AR, 7AR, 8AR

(LR decays ``8e-5 → 8e-6 → 8e-7``.)

This helper runs the sequence **in-process**: for each stage we call
``train.run_training(cfg)``, then copy the final checkpoint of that stage
into the next stage's rundir so the existing ``resume_checkpoint: latest``
logic in ``Trainer._resume_if_needed`` picks it up without any trainer
code changes. Operators are expected to wrap a single invocation of this
script in their own ``sbatch`` (one GPU job, sequential stages).

Usage
-----
    python scripts/stage4_ar_schedule.py \\
        --config-name fgn_arco_dev \\
        --rundir /mnt/data/.../fgn_stage4 \\
        --stats-path /mnt/data/.../stats.npz \\
        [--dry-run]

The ``--config-name`` is the Hydra base config; everything else is
supplied via command-line overrides so the orchestration layer is thin
and the per-stage configuration stays faithful to Table A.2.

Per-stage knobs can be overridden via ``--stages`` which takes a JSON
list of ``{"ar_steps": int, "total_train_steps": int, "lr": float}``
dicts — useful for dev runs that don't want the full 18 000-step
schedule.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

# Ensure ``train``/``utils``/``datasets`` resolve when invoking this script
# from outside the example directory, e.g. via sbatch --chdir elsewhere.
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from hydra import compose, initialize  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

# Stage 4 of paper Table A.2.
PAPER_STAGES: list[dict] = [
    {"ar_steps": 1, "total_train_steps": 8000, "lr": 8e-5},
    {"ar_steps": 2, "total_train_steps": 4000, "lr": 8e-5},
    {"ar_steps": 3, "total_train_steps": 1000, "lr": 8e-5},
    {"ar_steps": 4, "total_train_steps": 1000, "lr": 8e-6},
    {"ar_steps": 5, "total_train_steps": 1000, "lr": 8e-6},
    {"ar_steps": 6, "total_train_steps": 1000, "lr": 8e-7},
    {"ar_steps": 7, "total_train_steps": 1000, "lr": 8e-7},
    {"ar_steps": 8, "total_train_steps": 1000, "lr": 8e-7},
]

# Small-footprint dev schedule for quick smoke testing on ARCO.
DEV_STAGES: list[dict] = [
    {"ar_steps": 1, "total_train_steps": 20, "lr": 3e-4},
    {"ar_steps": 2, "total_train_steps": 20, "lr": 1e-4},
    {"ar_steps": 4, "total_train_steps": 10, "lr": 1e-4},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-name",
        default="fgn",
        help="Hydra config name. Default: fgn. Use fgn_arco_dev for dev runs.",
    )
    p.add_argument(
        "--config-path",
        default="../config",
        help="Hydra config_path relative to this script. Default: ../config.",
    )
    p.add_argument(
        "--rundir",
        required=True,
        type=Path,
        help="Base run directory; each stage writes to <rundir>/stage<N>/.",
    )
    p.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="Normalization stats .npz (propagated to dataset.stats_path).",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=None,
        help="JSON list of stage dicts to override the paper schedule.",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Use the small-footprint DEV_STAGES schedule instead of paper.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each stage's resolved config without invoking the trainer.",
    )
    p.add_argument("--extra", nargs="*", default=(), help="Hydra overrides.")
    return p.parse_args()


def _last_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("*.mdlus"))
    return candidates[-1] if candidates else None


def _seed_from_prev_stage(
    prev_checkpoint_dir: Path, stage_checkpoint_dir: Path
) -> None:
    """Copy the previous stage's final ``.mdlus`` + ``.pt`` to the new stage.

    ``Trainer._resume_if_needed`` uses ``physicsnemo.utils.load_checkpoint``
    which looks in the configured ``checkpoint_dir``. Copying the final
    files over is enough for ``resume_checkpoint: latest`` to pick them up,
    and keeps the trainer completely unchanged.
    """
    last_mdlus = _last_checkpoint(prev_checkpoint_dir)
    if last_mdlus is None:
        raise FileNotFoundError(
            f"No .mdlus checkpoint found in {prev_checkpoint_dir} — previous stage didn't save?"
        )
    stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last_mdlus, stage_checkpoint_dir / last_mdlus.name)
    # Copy optimizer/scheduler state if present so the resume is exact.
    # physicsnemo names these ``checkpoint.{mp_rank}.{epoch}.pt``; find the
    # matching epoch by filename suffix (``.<epoch>.mdlus``).
    epoch_suffix = last_mdlus.stem.split(".")[-1]
    for cand in prev_checkpoint_dir.glob(f"checkpoint.*.{epoch_suffix}.pt"):
        shutil.copy2(cand, stage_checkpoint_dir / cand.name)


def build_stage_cfg(
    base_cfg: DictConfig,
    stage: dict,
    stage_rundir: Path,
    stats_path: Path | None,
) -> DictConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.training.outdir = str(stage_rundir.parent)
    cfg.training.experiment_name = stage_rundir.parent.name
    cfg.training.run_id = stage_rundir.name
    cfg.training.rundir = str(stage_rundir)
    cfg.training.ar_steps = int(stage["ar_steps"])
    cfg.training.total_train_steps = int(stage["total_train_steps"])
    cfg.training.optimizer.lr = float(stage["lr"])
    cfg.training.resume_checkpoint = "latest"
    if stats_path is not None:
        cfg.dataset.stats_path = str(stats_path)
    return cfg


def main() -> int:
    args = parse_args()

    if args.stages is not None:
        stages = json.loads(args.stages)
    elif args.dev:
        stages = DEV_STAGES
    else:
        stages = PAPER_STAGES

    args.rundir.mkdir(parents=True, exist_ok=True)

    # Hydra's ``initialize`` resolves ``config_path`` relative to THIS
    # file, not the caller's cwd. Default ``../config`` points at the
    # example's config tree regardless of where the user runs the script.
    with initialize(version_base=None, config_path=args.config_path, job_name="stage4"):
        base_cfg = compose(config_name=args.config_name, overrides=list(args.extra))

    prev_checkpoint_dir: Path | None = None

    from train import run_training  # noqa: E402 — imported after Hydra setup

    for i, stage in enumerate(stages):
        stage_rundir = args.rundir / f"stage{i}_ar{stage['ar_steps']}"
        stage_rundir.mkdir(parents=True, exist_ok=True)

        stage_cfg = build_stage_cfg(base_cfg, stage, stage_rundir, args.stats_path)
        print(
            f"\n[stage {i}] ar_steps={stage['ar_steps']} "
            f"steps={stage['total_train_steps']} lr={stage['lr']:g} "
            f"rundir={stage_rundir}"
        )
        if prev_checkpoint_dir is not None and not args.dry_run:
            target = stage_rundir / stage_cfg.training.checkpoint_dir
            _seed_from_prev_stage(prev_checkpoint_dir, target)
            print(f"[stage {i}] seeded from {prev_checkpoint_dir} → {target}")

        if args.dry_run:
            print(OmegaConf.to_yaml(stage_cfg))
        else:
            run_training(stage_cfg)

        prev_checkpoint_dir = stage_rundir / stage_cfg.training.checkpoint_dir

    return 0


if __name__ == "__main__":
    sys.exit(main())
