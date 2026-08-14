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

import argparse
import glob
import logging
import os
import random
import re

import numpy as np
import torch
import zarr
from data import HRRRSurfaceDataset  # noqa: F401  imported for dataset registration
from nn import HRRRUnconditionalUNet  # noqa: F401  imported so from_checkpoint can resolve the class
from tensordict import TensorDict
from torch.utils.data import DataLoader

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionDataConsistencyDPSGuidance,
    MultiDiffusionDPSScorePredictor,
    MultiDiffusionModel2D,
    MultiDiffusionPredictor,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner  # noqa: F401  needed by from_checkpoint
from physicsnemo.diffusion.samplers import sample
from physicsnemo.utils.logging import PythonLogger


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="HRRR surface diffusion SDA inference")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--output-path", type=str, default="./outputs/samples.pt")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument(
        "--solver",
        type=str,
        default="heun",
        choices=["euler", "heun", "edm_stochastic_euler", "edm_stochastic_heun"],
    )
    p.add_argument("--overlap-pix", type=int, default=32)
    p.add_argument("--boundary-pix", type=int, default=0)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Patches per model call; defaults to 4 when --use-dps is set",
    )
    p.add_argument(
        "--obs-fraction",
        type=float,
        default=0.01,
        help="Fraction of spatial pixels used as observations",
    )
    p.add_argument(
        "--obs-channels",
        type=int,
        nargs="+",
        default=[0, 1, 4],
        help="Channel indices to observe: 0=u10m, 1=v10m, 4=t2m",
    )
    p.add_argument("--std-y", type=float, default=0.1, help="Observation noise std dev")
    p.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="SDA covariance scaling in DPS (>0 enables sigma_fn/alpha_fn injection)",
    )
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--use-dps", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_obs_mask(H, W, obs_channels, obs_fraction, seed):
    """Build a reproducible sparse observation mask, shape (16, H, W)."""
    mask = torch.zeros(16, H, W, dtype=torch.bool)
    rng = torch.Generator()
    rng.manual_seed(seed)
    num_obs = max(1, int(obs_fraction * H * W))
    flat_idx = torch.randperm(H * W, generator=rng)[:num_obs]
    rows, cols = flat_idx // W, flat_idx % W
    for ch in obs_channels:
        mask[ch, rows, cols] = True
    return mask


def main():
    """Run inference on a checkpoint and save samples to a `.pt` file."""
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = PythonLogger("generate")
    logger.logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_resolution = [1059, 1799]
    img_channels = 16
    patch_shape = (448, 448)

    ckpt_files = glob.glob(
        os.path.join(args.checkpoint_dir, "MultiDiffusionModel2D.*.mdlus")
    )
    if not ckpt_files:
        raise FileNotFoundError(
            f"No MultiDiffusionModel2D.*.mdlus in {args.checkpoint_dir}"
        )
    latest_ckpt = max(
        ckpt_files,
        key=lambda f: int(
            re.search(r"MultiDiffusionModel2D\.\d+\.(\d+)\.mdlus$", f).group(1)
        ),
    )
    # strict=False: training checkpoints predate the _patch_shape persistent buffer
    md_model = MultiDiffusionModel2D.from_checkpoint(latest_ckpt, strict=False)
    md_model = md_model.to(device).to(memory_format=torch.channels_last)
    md_model.eval()
    logger.info(f"Loaded checkpoint from {latest_ckpt}")

    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v2",
        mode="r",
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo",
        },
    )
    time_coord = root["time"][:]
    sidx = np.where(time_coord == np.datetime64("2025-01-01T00:00:00"))[0][0]
    eidx = np.where(time_coord == np.datetime64("2025-12-31T00:00:00"))[0][0]
    time_idx = np.arange(sidx, eidx, 25)
    dataset = HRRRSurfaceDataset(
        "s3://hrrr-surface-sda/zarr-v2",
        time_idx,
        storage_options={
            "endpoint_url": "https://pdx.s8k.io",
            "profile": "physicsnemo",
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    H, W = img_resolution
    x_gt, cond_spatial, cond_time = next(iter(loader))
    x_gt = x_gt.to(device)
    cond_spatial = cond_spatial.to(device).to(memory_format=torch.channels_last)
    cond_time = cond_time.to(device).float()
    B = x_gt.shape[0]

    condition = TensorDict(
        {"cond_concat": cond_spatial, "cond_time": cond_time},
        batch_size=[B],
    )

    chunk_size = args.chunk_size if args.chunk_size is not None else 4
    logger.info(f"chunk_size={chunk_size}")

    predictor = MultiDiffusionPredictor(
        md_model,
        condition=condition,
        chunk_size=chunk_size,
        use_checkpointing=True,
    )
    predictor.set_patching(args.overlap_pix, args.boundary_pix, patch_shape=patch_shape)
    logger.info(f"Grid patching: {predictor._P} patches, overlap={args.overlap_pix}px")

    mask = (
        build_obs_mask(H, W, args.obs_channels, args.obs_fraction, args.seed)
        .unsqueeze(0)
        .expand(B, -1, -1, -1)
        .contiguous()
        .to(device)
    )
    y_obs = x_gt * mask

    noise_scheduler = EDMNoiseScheduler(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        sigma_data=1.0,
    )

    if args.use_dps:
        guidance_kwargs = {}
        if args.gamma > 0.0:
            guidance_kwargs["sigma_fn"] = noise_scheduler.sigma
            guidance_kwargs["alpha_fn"] = noise_scheduler.alpha
        guidance = MultiDiffusionDataConsistencyDPSGuidance(
            predictor=predictor,
            mask=mask,
            y=y_obs,
            std_y=args.std_y,
            gamma=args.gamma,
            **guidance_kwargs,
        )
        dps_predictor = MultiDiffusionDPSScorePredictor(
            x0_predictor=predictor,
            x0_to_score_fn=noise_scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = noise_scheduler.get_denoiser(score_predictor=dps_predictor)
    else:
        denoiser = noise_scheduler.get_denoiser(x0_predictor=predictor)

    time_steps = noise_scheduler.timesteps(
        args.num_steps, device=device, dtype=torch.float32
    )
    tN = time_steps[0].expand(B)
    xN = noise_scheduler.init_latents(
        (img_channels, H, W), tN, device=device, dtype=torch.float32
    ).to(memory_format=torch.channels_last)

    logger.info(
        f"Sampling {B} samples: solver={args.solver}, steps={args.num_steps}, "
        f"overlap={args.overlap_pix}px, use_dps={args.use_dps}"
    )
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = sample(
                denoiser,
                xN,
                noise_scheduler,
                num_steps=args.num_steps,
                solver=args.solver,
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(
        {
            "x0": x0.cpu().float(),
            "ground_truth": x_gt.cpu().float(),
            "mask": mask.cpu(),
            "y_obs": y_obs.cpu().float(),
            "cond_spatial": cond_spatial.cpu().float().contiguous(),
            "cond_time": cond_time.cpu(),
            "use_dps": args.use_dps,
        },
        args.output_path,
    )
    logger.info(f"Saved {B} samples to {args.output_path}")


if __name__ == "__main__":
    main()
