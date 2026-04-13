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

from datetime import datetime
from pathlib import Path

import deepwave
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets.dataset import EFWIDatapipe
from datasets.transforms import Interpolate, ZscoreNormalize
from einops import rearrange, repeat
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from utils.plot import plot_prediction

from physicsnemo import Module
from physicsnemo.diffusion.guidance import (
    DPSScorePredictor,
    ModelConsistencyDPSGuidance,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.utils import StackedRandomGenerator
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb


class ClippedGuidance:
    """Thin wrapper that clips DPS guidance output to a given range."""

    def __init__(self, inner, clip_min, clip_max):
        self.inner = inner
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, x, t, x_0):
        return torch.clamp(
            self.inner(x, t, x_0),
            min=self.clip_min,
            max=self.clip_max,
        )


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    Generate predictions using the trained diffusion FWI model.
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = PythonLogger("generate")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    # Initialize wandb: resume from training run if possible
    wandb_id = getattr(cfg.wandb, "wandb_id", None)
    if wandb_id is not None:
        rank_zero_logger.info(f"Connecting to existing wandb run: {wandb_id}")
    initialize_wandb(
        project=f"DiffusionFWI-{'Training' if wandb_id is not None else 'Generation'}",
        entity=(cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "PhysicsNeMo"),
        mode=cfg.wandb.mode,
        results_dir=cfg.io.output_dir,
        wandb_id=wandb_id,
        resume="must" if wandb_id is not None else None,
        name=f"generate-{timestamp}",
    )

    device = dist.device
    rank_zero_logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    global_seed: int = cfg.generation.global_seed
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)

    # Define random seeds and split them across ranks
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = list(all_batches[dist.rank :: dist.world_size])

    # Initialize the validation dataset
    val_dataset = EFWIDatapipe(
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="test",
        batch_size_per_device=1,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        seed=global_seed,
        use_sharding=False,
    )

    # Define dataset transforms
    stats_mean = val_dataset.get_stats("mean")
    stats_std = val_dataset.get_stats("std")
    val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    img_H, img_W = list(cfg.dataset.x_resolution)

    interp_size = {var: (img_H, img_W) for var in cfg.dataset.x_vars}
    interp_size.update({var: (img_W,) for var in cfg.dataset.y_vars})
    interp_dim = {var: (-2, -1) for var in cfg.dataset.x_vars}
    interp_dim.update({var: (-1,) for var in cfg.dataset.y_vars})
    interp_mode = {var: "bilinear" for var in cfg.dataset.x_vars}
    interp_mode.update({var: "bilinear" for var in cfg.dataset.y_vars})
    val_dataset = Interpolate(
        val_dataset,
        size=interp_size,
        dim=interp_dim,
        mode=interp_mode,
    )

    # Load model from checkpoint
    checkpoint_path = to_absolute_path(cfg.model.checkpoint_path)
    rank_zero_logger.info(f"Loading model from {checkpoint_path}")
    try:
        model = Module.from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        rank_zero_logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
    except Exception as e:
        rank_zero_logger.error(f"Error loading checkpoint: {e}")
        return
    rank_zero_logger.info("Diffusion model loaded successfully.")
    model = model.eval().to(device)
    rank_zero_logger.info(
        f"Using model {model.__class__.__name__} "
        f"with {model.num_parameters()} parameters."
    )

    # Noise scheduler
    noise_scheduler = EDMNoiseScheduler(
        sigma_min=cfg.generation.sampler.sigma_min,
        sigma_max=cfg.generation.sampler.sigma_max,
        sigma_data=cfg.noise_schedule.sigma_data,
    )

    # Wave operator for diffusion posterior sampling (DPS) based on PDE
    # constraint.  The wave equation is solved on the original PDE grid
    # (``pde_resolution``) rather than on the model grid so that the
    # observation operator is consistent with the training data pipeline:
    #   training:  vel_orig -> PDE(vel_orig) -> seismic_orig -> norm -> interp
    #   guidance:  vel_model -> interp_to_orig -> PDE -> seismic_orig -> norm -> interp
    def wave_operator(x: torch.Tensor) -> torch.Tensor:
        # Unpack velocity model from latent state x
        B = x.shape[0]
        x_vars = torch.split(x, 1, dim=1)
        vars_names = list(cfg.dataset.x_vars)
        vp = x_vars[vars_names.index("vp")].squeeze(1)  # (B, H, W)
        vs = x_vars[vars_names.index("vs")].squeeze(1)  # (B, H, W)
        rho = x_vars[vars_names.index("rho")].squeeze(1)  # (B, H, W)

        # Denormalize velocity model
        vp = stats_mean["vp"] + stats_std["vp"] * vp  # (B, H, W)
        vs = stats_mean["vs"] + stats_std["vs"] * vs  # (B, H, W)
        rho = stats_mean["rho"] + stats_std["rho"] * rho  # (B, H, W)

        # Clamp denormalized values to physical ranges if specified
        guidance_cfg = cfg.generation.sampler.physics_informed_guidance
        vp_range = getattr(guidance_cfg, "vp_range", None)
        if vp_range is not None:
            vp = torch.clamp(vp, min=vp_range[0], max=vp_range[1])

        vs_range = getattr(guidance_cfg, "vs_range", None)
        if vs_range is not None:
            vs = torch.clamp(vs, min=vs_range[0], max=vs_range[1])

        rho_range = getattr(guidance_cfg, "rho_range", None)
        if rho_range is not None:
            rho = torch.clamp(rho, min=rho_range[0], max=rho_range[1])

        # Interpolate velocity model from model resolution to original PDE
        # resolution so that the wave equation is solved on the same grid
        # that was used to generate the training data.
        pde_H, pde_W = list(guidance_cfg.pde_resolution)
        pde_dx = guidance_cfg.pde_dx
        vp = F.interpolate(
            vp.unsqueeze(1), size=(pde_H, pde_W), mode="bilinear"
        ).squeeze(1)
        vs = F.interpolate(
            vs.unsqueeze(1), size=(pde_H, pde_W), mode="bilinear"
        ).squeeze(1)
        rho = F.interpolate(
            rho.unsqueeze(1), size=(pde_H, pde_W), mode="bilinear"
        ).squeeze(1)

        # Define geometry, sources and receivers on the original PDE grid
        nt = cfg.dataset.y_resolution[0]
        dt = 0.001
        freq = guidance_cfg.source_frequency
        peak_time = 1.5 / freq
        n_shots = cfg.dataset.nb_shots
        source_depth = 1
        receiver_depth = 1
        n_receivers_per_shot = pde_W - 1

        # Set sources evenly spaced on the original grid
        source_locations = torch.zeros(
            n_shots, 1, 2, dtype=torch.long, device=x.device
        )  # (Ns, 1, 2)
        source_locations[..., 0] = source_depth
        source_spacing = (pde_W - 2) // (n_shots - 1)
        source_locations[:, 0, 1] = torch.arange(n_shots) * source_spacing

        # Set receivers on the original grid
        receiver_locations = torch.zeros(
            n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=x.device
        )  # (Ns, Nr, 2)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = torch.arange(n_receivers_per_shot).repeat(
            n_shots, 1
        )
        source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, 1, 1)
            .to(x.device)
            * 100000.0
        )  # (Ns, 1, Nt)

        # Re-batch the sources, receivers, and velocity models
        source_locations = repeat(source_locations, "Ns u v -> (B Ns) u v", B=B)
        receiver_locations = repeat(receiver_locations, "Ns Nr v -> (B Ns) Nr v", B=B)
        source_amplitudes = repeat(source_amplitudes, "Ns u Nt -> (B Ns) u Nt", B=B)
        vp = repeat(vp, "B H W -> (B Ns) H W", Ns=n_shots)
        vs = repeat(vs, "B H W -> (B Ns) H W", Ns=n_shots)
        rho = repeat(rho, "B H W -> (B Ns) H W", Ns=n_shots)

        # Run the forward wave PDE at original resolution
        out = {}
        out["vz"], out["vx"] = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            grid_spacing=pde_dx,
            dt=dt,
            source_amplitudes_y=source_amplitudes,
            source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations,
            source_locations_x=source_locations,
            receiver_locations_y=receiver_locations,
            receiver_locations_x=receiver_locations,
            pml_freq=freq,
            pml_width=[20, 20, 20, 20],
        )[-2:]  # (B * Ns, Nr_orig, Nt)

        y: torch.Tensor = torch.cat(
            [
                rearrange(out[var], "(B Ns) H W -> B Ns H W", B=B, Ns=n_shots)
                for var in list(cfg.dataset.y_vars)
            ],
            dim=1,
        ).transpose(3, 2)  # (B, C, Nt, Nr_orig)

        # Z-score normalize to match the normalized conditioning data
        y_vars = list(cfg.dataset.y_vars)
        for idx, var in enumerate(y_vars):
            ch_start = idx * n_shots
            ch_end = (idx + 1) * n_shots
            y[:, ch_start:ch_end] = (
                y[:, ch_start:ch_end] - stats_mean[var]
            ) / stats_std[var]

        # Interpolate receiver dimension to model resolution, matching the
        # bilinear interpolation applied to the training data
        y = F.interpolate(
            y,
            size=(y.shape[2], cfg.dataset.y_resolution[1]),
            mode="bilinear",
        )

        return y

    # Precompute timesteps for sampling
    num_steps = cfg.generation.sampler.num_steps
    t_steps = noise_scheduler.timesteps(num_steps, device=device)
    spatial_shape = (len(cfg.dataset.x_vars), img_H, img_W)

    output_dir = Path(to_absolute_path(cfg.io.output_dir))
    rank_zero_logger.info(f"Starting generation, saving results to {output_dir}...")
    for i, data in enumerate(val_dataset):
        # Stop generation after num_samples
        if i >= cfg.generation.num_samples:
            break

        y = torch.cat(
            [data.get(var, None) for var in list(cfg.dataset.y_vars) if var in data],
            dim=1,
        )  # (1, C_y, T, W)
        y = y.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(
            memory_format=torch.channels_last
        )  # (B, C_y, T, W)

        # x0-predictor closed over the conditioning y
        def x0_predictor(x, t, _y=y):
            return model(x, t, condition=_y[: x.shape[0]])

        # Build denoiser: either plain or with DPS guidance
        if cfg.generation.sampler.physics_informed:
            guidance_cfg = cfg.generation.sampler.physics_informed_guidance
            guidance = ModelConsistencyDPSGuidance(
                observation_operator=wave_operator,
                y=y,
                std_y=guidance_cfg.std_y,
                norm=guidance_cfg.norm,
                gamma=guidance_cfg.gamma,
                sigma_fn=noise_scheduler.sigma,
                alpha_fn=noise_scheduler.alpha,
            )

            # Apply score clipping if specified
            clip_range = getattr(guidance_cfg, "score_clip_range", None)
            if clip_range is not None:
                clip_range = list(clip_range)
                rank_zero_logger.info(f"Using score clipping with range {clip_range}")
                guidance = ClippedGuidance(guidance, clip_range[0], clip_range[1])

            dps_score_predictor = DPSScorePredictor(
                x0_predictor=x0_predictor,
                x0_to_score_fn=noise_scheduler.x0_to_score,
                guidances=guidance,
            )
            denoiser = noise_scheduler.get_denoiser(
                score_predictor=dps_score_predictor,
                denoising_type="ode",
            )
        else:
            denoiser = noise_scheduler.get_denoiser(
                x0_predictor=x0_predictor,
                denoising_type="ode",
            )

        # Guidance requires intermediate grad computation; inference mode
        # does not allow this
        torch_grad_ctx = (
            torch.no_grad
            if cfg.generation.sampler.physics_informed
            else torch.inference_mode
        )

        with torch_grad_ctx():
            x_generated = []
            for batch_seeds in rank_batches:
                B = len(batch_seeds)
                if B == 0:
                    continue
                seeds_list = (
                    batch_seeds.tolist()
                    if isinstance(batch_seeds, torch.Tensor)
                    else list(batch_seeds)
                )
                rnd = StackedRandomGenerator(device, seeds_list)
                x_T = (
                    noise_scheduler.sigma(t_steps[0])
                    * rnd.randn((B,) + spatial_shape, device=device)
                ).to(memory_format=torch.channels_last)

                x_0 = sample(
                    denoiser=denoiser,
                    xN=x_T,
                    noise_scheduler=noise_scheduler,
                    num_steps=num_steps,
                    solver="heun",
                    time_steps=t_steps,
                )
                x_generated.append(x_0)
            if not x_generated:
                continue
            x_pred_rank = torch.cat(x_generated)

        # Gather predictions to rank 0
        x_pred = gather_tensors(x_pred_rank, dist)

        # Compute statistics and metrics on rank 0
        if dist.rank == 0:
            data_pred = {
                var: x_pred[:, i : i + 1]
                for i, var in enumerate(cfg.dataset.x_vars)
                if var in data
            }
            data_true, x_mean_pred, x_std_pred = {}, {}, {}
            rmse, mae = {}, {}
            for var in data_pred.keys():
                data_true[var] = data[var] * stats_std[var] + stats_mean[var]
                data_pred[var] = data_pred[var] * stats_std[var] + stats_mean[var]
                x_mean_pred[var] = data_pred[var].mean(dim=0, keepdim=True)
                x_std_pred[var] = data_pred[var].std(dim=0, keepdim=True)
                rmse[var] = RMSE(data_pred[var], data_true[var])
                mae[var] = MAE(data_pred[var], data_true[var])
            data_input = {
                var: data[var] * stats_std[var] + stats_mean[var]
                for var in list(cfg.dataset.y_vars)
                if var in data
            }

            # Log metrics
            rank_zero_logger.info(f"Sample {i}:")
            metrics = {}
            for var in data_pred.keys():
                rank_zero_logger.info(
                    f"{var} - RMSE: {rmse[var]:.6f}, MAE: {mae[var]:.6f}"
                )
                metrics.update(
                    {
                        f"sample_{i}/{var}_rmse": rmse[var],
                        f"sample_{i}/{var}_mae": mae[var],
                    }
                )
            wandb.log(metrics)

            # Plot results
            output_path = output_dir / f"sample_{i}"
            output_path.mkdir(parents=True, exist_ok=True)
            plot_prediction(
                sample_idx=i,
                inputs=data_input,
                targets=data_true,
                predictions=data_pred,
                statistics={"mean": x_mean_pred, "std": x_std_pred},
                metrics={"rmse": rmse, "mae": mae},
                save_dir=output_path,
                sources_to_plot=3,
            )

            # Save raw numpy arrays
            output_path = output_dir / f"sample_{i}" / "numpy"
            output_path.mkdir(parents=True, exist_ok=True)
            save_data = {}
            for var in data_pred.keys():
                save_data[f"{var}_pred"] = data_pred[var].cpu().numpy()
                save_data[f"{var}_true"] = data_true[var].cpu().numpy()
                save_data[f"{var}_mean"] = x_mean_pred[var].cpu().numpy()
                save_data[f"{var}_std"] = x_std_pred[var].cpu().numpy()
                save_data[f"{var}_ensemble"] = data_pred[var].cpu().numpy()
            for var in list(cfg.dataset.y_vars):
                if var in data:
                    data_input = data[var] * stats_std[var] + stats_mean[var]
                    save_data[f"{var}"] = data_input.cpu().numpy()
            np.savez_compressed(output_path / "data.npz", **save_data)

    rank_zero_logger.success("Generation completed!")
    wandb.finish()
    return


def gather_tensors(tensor, dist):
    """
    Gather tensors from all ranks to rank 0.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to gather
    dist : DistributedManager
        The distributed manager instance

    Returns
    -------
    torch.Tensor or None
        Concatenated tensor on rank 0, None on other ranks
    """
    if dist.world_size > 1:
        if dist.rank == 0:
            gathered_tensors = [
                torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device)
                for _ in range(dist.world_size)
            ]
        else:
            gathered_tensors = None

        torch.distributed.barrier()
        torch.distributed.gather(
            tensor,
            gather_list=gathered_tensors if dist.rank == 0 else None,
            dst=0,
        )

        if dist.rank == 0:
            return torch.cat(gathered_tensors, dim=0)
        else:
            return None
    else:
        return tensor


if __name__ == "__main__":
    main()
