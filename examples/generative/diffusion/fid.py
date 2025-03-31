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

# TODO (mnabian) refactor, generalize

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import pickle

import hydra
import numpy as np
import torch
import tqdm
from dataset import ImageFolderDataset
from omegaconf import DictConfig
from misc import open_url

from physicsnemo.metrics.diffusion import calculate_fid_from_inception_stats
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper


def calculate_inception_stats(
    image_path,
    dist,
    logger0,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
):
    device = dist.device
    # Rank 0 goes first.
    if dist.world_size > 1 and dist.rank != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    logger0.info("Loading Inception-v3 model...")
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    logger0.info(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(
        path=image_path, max_size=num_expected, random_seed=seed
    )
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise ValueError(
            f"Found {len(dataset_obj)} images, but expected at least {num_expected}"
        )
    if len(dataset_obj) < 2:
        raise ValueError(
            f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics"
        )

    # Other ranks follow.
    if dist.world_size > 1 and dist.rank == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = (
        (len(dataset_obj) - 1) // (max_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_sampler=rank_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Accumulate statistics.
    logger0.info(f"Calculating statistics for {len(dataset_obj)} images...")
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _ in tqdm.tqdm(
        data_loader, unit="batch", disable=(dist.get_rank() != 0)
    ):
        if dist.world_size > 1:
            torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    if dist.world_size > 1:
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu, sigma


def calc(image_path, ref_path, num_expected, seed, batch, dist, logger, logger0):
    """Calculate FID for a given set of images."""

    logger0.info(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.rank == 0:
        with open_url(ref_path) as f:
            ref = dict(np.load(f))
            mu_ref = torch.as_tensor(ref["mu"], device=dist.device)
            sigma_ref = torch.as_tensor(ref["sigma"], device=dist.device)

    mu, sigma = calculate_inception_stats(
        image_path=image_path,
        dist=dist,
        logger0=logger0,
        num_expected=num_expected,
        seed=seed,
        max_batch_size=batch,
    )
    logger0.info("Calculating FID...")
    if dist.rank == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)
        logger.info(f"{fid:g}")
    if dist.world_size > 1:
        torch.distributed.barrier()


def ref(dataset_path, dest_path, batch, dist, logger0):
    """Calculate dataset reference statistics needed by 'calc'."""

    mu, sigma = calculate_inception_stats(
        image_path=dataset_path, dist=dist, logger0=logger0, max_batch_size=batch
    )
    logger0.info(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.rank == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    if dist.world_size > 1:
        torch.distributed.barrier()
    logger0.info("Done.")


# ----------------------------------------------------------------------------


@hydra.main(version_base="1.2", config_path="conf", config_name="config_fid")
def main(cfg: DictConfig) -> None:

    """Calculate Frechet Inception Distance (FID)."""

    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging()

    if cfg.mode == "calc":
        calc(
            cfg.image_path,
            cfg.ref_path,
            cfg.num_expected,
            cfg.seed,
            cfg.batch,
            dist,
            logger,
            logger0,
        )
    elif cfg.mode == "ref":
        ref(cfg.dataset_path, cfg.dest_path, cfg.batch, dist, logger0)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
