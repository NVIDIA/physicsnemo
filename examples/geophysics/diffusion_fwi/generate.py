# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
from functools import partial

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import wandb
from einops import repeat, rearrange

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from physicsnemo.launch.logging.wandb import initialize_wandb

from datasets.dataset import EFWIDatapipe
from utils.preconditioning import edm_precond
from utils.diffusion import (
    DiffusionAdapter,
    ModelBasedGuidance,
    generate,
    EDMStochasticSampler,
)
from datasets.transforms import ZscoreNormalize, Interpolate
from utils.plot import plot_prediction
import deepwave


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
    rank_batches = all_batches[dist.rank :: dist.world_size]

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
    # Zscore normalization
    stats_mean = val_dataset.get_stats("mean")
    stats_std = val_dataset.get_stats("std")
    val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    img_H, img_W = list(cfg.dataset.x_resolution)

    # Interpolation to the UNet model accepted resolution
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

    # Load diffusion model
    checkpoint_path = to_absolute_path(cfg.model.checkpoint_path)
    rank_zero_logger.info(f"Loading diffusion model from {checkpoint_path}")
    try:
        diffusion_net = Module.from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        rank_zero_logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
    except Exception as e:
        rank_zero_logger.error(f"Error loading checkpoint: {e}")
        return
    diffusion_net = diffusion_net.eval().to(device)
    rank_zero_logger.info("Diffusion model loaded successfully.")
    rank_zero_logger.info(
        f"Using model {diffusion_net.__class__.__name__} "
        f"with {diffusion_net.num_parameters()} parameters."
    )
    model = DiffusionAdapter(
        model=diffusion_net,
        args_map=("x", "t", {"y": "y"}),
    )
    # EDM preconditioning wrapper
    model_fn = partial(edm_precond, model, sigma_data=0.5)

    # Sampler
    sampler = EDMStochasticSampler(
        model=model_fn,
        num_steps=cfg.generation.sampler.num_steps,
        sigma_min=cfg.generation.sampler.sigma_min,
        sigma_max=cfg.generation.sampler.sigma_max,
    )

    # Wave operator for diffusion posterior sampling (DPS) based on PDE
    # constraint
    def wave_operator(x: torch.Tensor) -> torch.Tensor:
        def smooth_clamp(
            x: torch.Tensor,
            min_val: float,
            max_val: float,
        ) -> torch.Tensor:
            x_scaled = torch.sigmoid(x)
            return min_val + (max_val - min_val) * x_scaled

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

        # Apply smooth clamping to denormalized values if ranges are specified
        guidance_cfg = cfg.generation.sampler.physics_informed_guidance
        vp_range = getattr(guidance_cfg, "vp_range", None)
        if vp_range is not None:
            vp_min, vp_max = list(vp_range)
            vp = smooth_clamp(vp, vp_min, vp_max)

        vs_range = getattr(guidance_cfg, "vs_range", None)
        if vs_range is not None:
            vs_min, vs_max = list(vs_range)
            vs = smooth_clamp(vs, vs_min, vs_max)

        rho_range = getattr(guidance_cfg, "rho_range", None)
        if rho_range is not None:
            rho_min, rho_max = list(rho_range)
            rho = smooth_clamp(rho, rho_min, rho_max)

        # Define geometry, sources and receivers
        # NOTE: hard-coded resolution change from 70 to 80.
        dx = 5.0 * 7 / 8
        nt = cfg.dataset.y_resolution[0]
        dt = 0.001
        freq = cfg.generation.sampler.physics_informed_guidance.source_frequency
        peak_time = 1.5 / freq
        n_shots = cfg.dataset.nb_shots
        source_depth = 1
        receiver_depth = 1
        n_receivers_per_shot = cfg.dataset.y_resolution[1] - 1

        # Set sources and receivers
        source_locations = torch.zeros(
            n_shots, 1, 2, dtype=torch.long, device=x.device
        )  # (Ns, 1, 2)
        source_locations[..., 0] = source_depth
        # NOTE: hard-coded to go from 5 sources at 0, 17, 34, 51, 68 on a mesh
        # of width 70, to 5 sources on a mesh of width 80.
        source_locations[:, 0, 1] = torch.arange(n_shots) * 17 * 8 // 7
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

        # Run the forward wave PDE
        out = {}
        out["vz"], out["vx"] = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            grid_spacing=dx,
            dt=dt,
            source_amplitudes_y=source_amplitudes,
            source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations,
            source_locations_x=source_locations,
            receiver_locations_y=receiver_locations,
            receiver_locations_x=receiver_locations,
            pml_freq=freq,
            pml_width=[20, 20, 20, 20],
        )[-2:]  # (B * Ns, Nr, Nt)

        y: torch.Tensor = torch.cat(
            [
                rearrange(out[var], "(B Ns) H W -> B Ns H W", B=B, Ns=n_shots)
                for var in list(cfg.dataset.y_vars)
            ],
            dim=1,
        ).transpose(3, 2)  # (B, 2 * Ns, Nt, Nr)

        # Pad to match target resolution
        if y.shape[-1] != cfg.dataset.y_resolution[1]:
            pad_r = cfg.dataset.y_resolution[1] - y.shape[-1]
            y = torch.nn.functional.pad(y, pad=(0, pad_r))

        return y

    # DPS guidance based on the wave operator
    physics_informed_guidance = ModelBasedGuidance(
        guide_model=wave_operator,
        std=cfg.generation.sampler.physics_informed_guidance.std,
        gamma=cfg.generation.sampler.physics_informed_guidance.gamma,
        scale=cfg.generation.sampler.physics_informed_guidance.scale,
        power=cfg.generation.sampler.physics_informed_guidance.power,
        norm_ord=cfg.generation.sampler.physics_informed_guidance.norm_ord,
        magnitude_scaling=cfg.generation.sampler.physics_informed_guidance.magnitude_scaling,
    )

    # Add hook to perform score clipping if specified
    if cfg.generation.sampler.physics_informed:
        clip_range = getattr(
            cfg.generation.sampler.physics_informed_guidance, "score_clip_range", None
        )
        if clip_range is not None:
            clip_range = list(clip_range)

            def score_clipping_hook(guidance, x, x_0_hat, sigma, y, log_p):
                """Post-hook that applies clipping to the log-likelihood score."""
                clip_min, clip_max = clip_range
                return torch.clamp(log_p, min=clip_min, max=clip_max)

            rank_zero_logger.info(
                f"Registering score clipping hook with range {clip_range}"
            )
            physics_informed_guidance.register_score_post_hook(score_clipping_hook)

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

        # Generate ensemble predictions
        if cfg.generation.sampler.physics_informed:
            sampler_kwargs = {
                "guidance": physics_informed_guidance,
                "guidance_args": (y,),
            }
        else:
            sampler_kwargs = {}

        # NOTE: need intermediate grad computation when using physics-informed
        # guidance, inference mode does not allow this
        torch_grad_ctx = (
            torch.no_grad
            if cfg.generation.sampler.physics_informed
            else torch.inference_mode
        )

        with torch_grad_ctx():
            x_pred_rank = generate(
                sampler_fn=sampler,
                x_channels=len(cfg.dataset.x_vars),
                x_resolution=(img_H, img_W),
                rank_batches=rank_batches,
                cond={"y": y},
                device=device,
                sampler_kwargs=sampler_kwargs,
            )

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
