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

"""Run autoregressive stochastic inference for the FGN recipe.

Supports both single-model stochastic rollout and the paper's deep-ensemble
inference path (§2.2.1 of arXiv:2506.10772v1: J=4 independently-trained
models, equal number of members per model, model identity fixed over all
timesteps of a given trajectory; aleatoric noise ``z_t`` resampled every
step).
"""

from pathlib import Path

import hydra
import torch
from datasets import dataset_classes
from omegaconf import DictConfig
from utils.trainer import find_latest_model_checkpoint

from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager


def _resolve_checkpoints(cfg: DictConfig) -> list[str]:
    """Resolve the inference config to an ordered list of checkpoint paths.

    Priority: ``inference.checkpoints`` (list) wins if set. Otherwise fall
    back to ``inference.checkpoint`` (single path or ``"latest"``). Single-
    model inference is just the length-1 deep-ensemble case.
    """
    checkpoints = (
        cfg.inference.get("checkpoints", None)
        if hasattr(cfg.inference, "get")
        else getattr(cfg.inference, "checkpoints", None)
    )
    if checkpoints:
        return [str(c) for c in checkpoints]

    checkpoint = cfg.inference.checkpoint
    if checkpoint == "latest":
        return [
            str(
                find_latest_model_checkpoint(
                    Path(cfg.training.rundir) / cfg.training.checkpoint_dir
                )
            )
        ]
    return [str(checkpoint)]


def _allocate_members(num_trajectories: int, num_models: int) -> list[int]:
    """Distribute ``num_trajectories`` members across ``num_models`` models.

    Paper §2.2.1: "we generate an equal number of ensemble member
    trajectories from each model". When ``num_trajectories`` is not
    divisible by ``num_models``, put the remainder on the earlier models.
    """
    if num_models <= 0:
        raise ValueError("num_models must be positive")
    base = num_trajectories // num_models
    rem = num_trajectories % num_models
    return [base + (1 if i < rem else 0) for i in range(num_models)]


def _rollout(
    model: torch.nn.Module,
    history: torch.Tensor,
    background: torch.Tensor,
    invariants: torch.Tensor | None,
    num_steps: int,
    latent_dim: int,
    num_trajectories: int,
    device: torch.device,
    output_only_channels: list[int] | None = None,
) -> torch.Tensor:
    """Run ``num_trajectories`` independent autoregressive rollouts.

    Returns a tensor of shape ``(num_trajectories, num_steps, C, H, W)``.
    The model identity is fixed for the lifetime of each trajectory
    (paper §2.2.1); ``z_t`` is resampled every step (paper §2.2.2).
    Paper §3: predicted-only channels (``tp06``) are zeroed before being
    fed back as input for the next rollout step.
    """
    output_only_channels = output_only_channels or []
    trajectories: list[torch.Tensor] = []
    for _ in range(num_trajectories):
        rollout_history = history.clone()
        states: list[torch.Tensor] = []
        for _ in range(num_steps):
            latent = torch.randn(
                history.shape[0],
                latent_dim,
                device=device,
                dtype=torch.float32,
            )
            pred = model(
                history=rollout_history,
                latent=latent,
                background=background,
                invariants=invariants,
            )
            states.append(pred)
            next_frame = pred
            if output_only_channels:
                next_frame = next_frame.clone()
                for ci in output_only_channels:
                    next_frame[:, ci].zero_()
            rollout_history = torch.cat(
                [rollout_history[:, 1:], next_frame.unsqueeze(1)],
                dim=1,
            )
        trajectories.append(torch.stack(states, dim=1))
    return torch.cat(trajectories, dim=0)


def run_inference(cfg: DictConfig) -> dict[str, float | str | int | list[int]]:
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        raise NotImplementedError(
            "The FGN inference scaffold currently supports a single process only."
        )

    device = dist.device
    torch.manual_seed(int(cfg.inference.seed))

    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=False)
    sample = dataset[int(cfg.inference.dataset_index)]

    checkpoint_paths = _resolve_checkpoints(cfg)
    num_trajectories = int(cfg.inference.num_trajectories)
    members_per_model = _allocate_members(num_trajectories, len(checkpoint_paths))

    history = sample["history"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    # Datasets may emit target as (K, C, H, W) for AR training; inference MAE
    # only uses the first step, so collapse to (B, C, H, W).
    if target.ndim == 5:
        target = target[:, 0]
    background = (
        sample["background"].unsqueeze(0).to(device=device, dtype=torch.float32)
    )

    invariants = dataset.get_invariants()
    if invariants is not None:
        invariants = (
            torch.from_numpy(invariants)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

    all_trajectories: list[torch.Tensor] = []
    num_steps = int(cfg.inference.num_steps)
    output_only = dataset.output_only_channels()
    with torch.no_grad():
        for ckpt_path, n_members in zip(
            checkpoint_paths, members_per_model, strict=True
        ):
            if n_members <= 0:
                continue
            model = Module.from_checkpoint(ckpt_path).to(device).eval()
            latent_dim = int(getattr(model, "latent_dim", cfg.model.latent_dim))
            traj = _rollout(
                model=model,
                history=history,
                background=background,
                invariants=invariants,
                num_steps=num_steps,
                latent_dim=latent_dim,
                num_trajectories=n_members,
                device=device,
                output_only_channels=output_only,
            )
            all_trajectories.append(traj)
            del model  # free VRAM before loading next checkpoint

    trajectory_tensor = torch.cat(all_trajectories, dim=0).cpu()
    ensemble_mean = trajectory_tensor[:, 0].mean(dim=0, keepdim=True)
    first_step_mae = float((ensemble_mean - target.cpu()).abs().mean())

    output_path = Path(cfg.inference.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "history": history.cpu(),
            "target": target.cpu(),
            "trajectories": trajectory_tensor,
            "first_step_mae": first_step_mae,
            "num_models": len(checkpoint_paths),
            "members_per_model": members_per_model,
            "checkpoint_paths": checkpoint_paths,
        },
        output_path,
    )

    return {
        "output_path": str(output_path),
        "first_step_mae": first_step_mae,
        "num_models": len(checkpoint_paths),
        "members_per_model": members_per_model,
    }


@hydra.main(version_base=None, config_path="config", config_name="inference_fgn")
def main(cfg: DictConfig) -> None:
    result = run_inference(cfg)
    print(f"Saved inference outputs to {result['output_path']}")
    print(f"First-step ensemble-mean MAE: {result['first_step_mae']:.6f}")


if __name__ == "__main__":
    main()
