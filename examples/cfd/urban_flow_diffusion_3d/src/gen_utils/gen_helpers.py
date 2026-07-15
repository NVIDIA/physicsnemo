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

import os
from functools import partial

import h5py
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.utils import StackedRandomGenerator
from src.dataloaders.dataset_utils import rescale_and_crop_ds4


def save_uncond_samples(filename, preds_np, batch_seeds):
    """Append a batch of samples and their seeds to a resizable HDF5 file.

    Creates the ``uncond_preds`` / ``uncond_seeds`` datasets on the first call
    and extends them on subsequent calls, so samples can be streamed to disk
    batch by batch.
    """
    with h5py.File(filename, "a") as f:
        if "uncond_preds" not in f:
            f.create_dataset(
                "uncond_preds",
                data=preds_np,
                maxshape=(None,) + preds_np.shape[1:],
                chunks=True,
            )
        else:
            preds_ds = f["uncond_preds"]
            current_size = preds_ds.shape[0]
            preds_ds.resize((current_size + preds_np.shape[0],) + preds_ds.shape[1:])
            preds_ds[current_size:] = preds_np

        if "uncond_seeds" not in f:
            f.create_dataset(
                "uncond_seeds",
                data=batch_seeds,
                maxshape=(None,) + batch_seeds.shape[1:],
                chunks=True,
            )
        else:
            seeds_ds = f["uncond_seeds"]
            current_size = seeds_ds.shape[0]
            seeds_ds.resize((current_size + batch_seeds.shape[0],) + seeds_ds.shape[1:])
            seeds_ds[current_size:] = batch_seeds


def generate_samples(
    cfg: DictConfig,
    model: torch.nn.Module,
    noise_scheduler,
    dist,
    logger,
    logger0,
    eval_mode: bool = True,
    val_mode: bool = False,
) -> None:
    """Generate unconditional samples using ``physicsnemo.diffusion.samplers.sample``.

    ``model`` must already be an EDM-preconditioned model (e.g.
    :class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner`), whose
    output is directly an x0-prediction, so it doubles as an x0-predictor.
    """
    if eval_mode:
        model.eval().to(dist.device)

    sample_shape = cfg.generate.io.sample_shape
    assert len(sample_shape) == 4
    dataset_channels = sample_shape[0]
    img_shape = list(sample_shape[1:])

    batch_size_per_gpu = cfg.generate.batch_size_per_gpu
    img_outdir = cfg.generate.io.gen_dir
    filename_h5 = os.path.join(img_outdir, cfg.generate.io.filename)

    if val_mode:
        filename_h5 = filename_h5.replace(".h5", f"_rank{dist.rank}.h5")
        if os.path.exists(filename_h5):
            os.remove(filename_h5)
    else:
        epoch = cfg.generate.io.inf_ckpt
        filename_h5 = filename_h5.replace(
            ".h5", f"_epoch{epoch:04d}_rank{dist.rank}.h5"
        )

    total_images = cfg.generate.total_images
    gen_seeds = np.arange(total_images)
    rank_seeds = gen_seeds[dist.rank :: dist.world_size]
    rank_batches = torch.as_tensor(rank_seeds, device=dist.device).split(
        batch_size_per_gpu
    )

    num_steps = cfg.generate.sampler.num_steps
    solver = cfg.generate.sampler.solver

    if dist.rank == 0:
        os.makedirs(img_outdir, exist_ok=True)
    if dist.world_size > 1:
        torch.distributed.barrier()

    logger.info(f'Generating {len(rank_seeds)} images to "{img_outdir}"...')

    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(dist.device, batch_seeds + cfg.generate.seed)

        # Reproduce NoiseScheduler.init_latents' x_N = sigma(tN) * noise formula
        # (sigma(t) = t for EDM) but with per-seed StackedRandomGenerator noise,
        # instead of calling init_latents (which uses the global torch RNG and
        # would not be reproducible per-seed across ranks/restarts).
        noise = rnd.randn(
            [batch_size, dataset_channels, *img_shape], device=dist.device
        )
        t_steps = noise_scheduler.timesteps(num_steps, device=dist.device)
        tN = t_steps[0].expand(batch_size)
        latents = tN.view(-1, *([1] * (noise.ndim - 1))) * noise

        x0_predictor = partial(model, condition=None)
        denoiser = noise_scheduler.get_denoiser(x0_predictor=x0_predictor)

        preds = sample(
            denoiser,
            latents,
            noise_scheduler,
            num_steps=num_steps,
            solver=solver,
        ).detach()

        min_scaler = np.array(
            [cfg.dataset.u_comp.min, cfg.dataset.v_comp.min, cfg.dataset.w_comp.min]
        ).reshape(1, 3, 1, 1, 1)
        max_scaler = np.array(
            [cfg.dataset.u_comp.max, cfg.dataset.v_comp.max, cfg.dataset.w_comp.max]
        ).reshape(1, 3, 1, 1, 1)
        preds_np = rescale_and_crop_ds4(preds, min_scaler, max_scaler, sample_shape[2:])

        save_uncond_samples(filename_h5, preds_np, batch_seeds.cpu().numpy())

    if dist.world_size > 1:
        torch.distributed.barrier()
    logger0.info("Generation complete.")


def pack_uncond_preds(base_filename_h5=None, dist=None):
    """Concatenate the per-rank sample files into a single array.

    Reads ``uncond_preds`` from each rank's ``*_rank{r}.h5`` shard and stacks
    them along the batch axis. Returns the combined NumPy array.
    """
    all_preds = []

    for rank in range(dist.world_size):
        filename_rank = base_filename_h5.replace(".h5", f"_rank{rank}.h5")
        if not os.path.isfile(filename_rank):
            raise FileNotFoundError(f"File not found: {filename_rank}")

        with h5py.File(filename_rank, "r") as f:
            preds = f["uncond_preds"][:]
            all_preds.append(preds)

    concatenated_preds = np.concatenate(all_preds, axis=0)
    return concatenated_preds
