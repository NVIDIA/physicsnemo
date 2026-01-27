# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import functools

import numpy as np
import torch
import zarr
from datasets import samplers
from datasets.prefetch_map import prefetch_map
from datasets.transform import TransformV2, collate
from inference_helpers import (
    DAConfig,
    DAModel,
    post_process_to_fcn3,
    scoring_times,
    setup_zarr_output,
    write_to_zarr,
)
from tqdm import tqdm
from utils import distributed as dist
from utils.dataclass_parser import parse_args


def _device_transform(batch, transform, device):
    return transform.device_transform(batch, device=device)


def main():
    args = parse_args(DAConfig, convert_underscore_to_hyphen=False)

    if args.post_process_to_fcn3 and args.dataset != "era5":
        raise ValueError("can only post process to fcn3 if ussing ERA5 data")

    dist.init(timeout_infinite=True)
    dist.print0("Inference configuration:")
    dist.print0(f"  Dataset: {args.dataset}")
    dist.print0(f"  Innovation type: {args.innovation_type.value}")
    dist.print0(f"  Number of samples: {args.num_samples}")
    dist.print0(f"  Output: {args.output_path}")
    dist.print0(f"  Save mode: {args.save_mode.value}")

    # Load the checkpoint
    dist.print0(f"Loading checkpoint from {args.checkpoint_path}")
    da_model = DAModel(args)
    dataset = da_model.get_dataset(split=args.split)

    # full config
    times = scoring_times(args.z06_18_inits, args.time_frequency, args.split)

    if args.num_samples != -1:
        min_samples = max(args.num_samples, dist.get_world_size())
        tasks = samplers.subsample(times, min_samples=min_samples)
    else:
        tasks = list(range(len(times)))

    subsampled_times = times[tasks]
    subsampled_dataset_idx = dataset.times.get_indexer(subsampled_times)

    gpu_tasks = samplers.distributed_split(subsampled_dataset_idx)
    gpu_times = dataset.times[gpu_tasks]
    print(
        f"Tasks: Length: {len(gpu_tasks)} on rank {dist.get_rank()}. Min time: {gpu_times[0]}. Max time: {gpu_times[-1]}"
    )
    batch_size = min(args.batch_gpu, len(gpu_tasks))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=gpu_tasks,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=5,
        prefetch_factor=12,
        multiprocessing_context="spawn",
    )

    transform = TransformV2(variable_config=da_model.variable_config)
    dataloader = prefetch_map(
        dataloader,
        functools.partial(
            _device_transform, transform=transform, device=da_model.device
        ),
        queue_size=2,
    )

    channels = da_model.batch_info.channels

    if dist.get_rank() == 0:
        group = setup_zarr_output(
            args.output_path,
            channels=channels,
            num_times=len(subsampled_times),
            batch_size=batch_size,
            subsampled_times=subsampled_times,
        )

    if dist.get_world_size() > 1:
        torch.distributed.barrier()

    # reopen with consolidated metadata to avoid reading extra metadata when writing
    group = zarr.open_group(args.output_path, mode="r+")

    dist.print0("Setup output zarr file. Starting inference...")

    with torch.no_grad():
        # denormalize
        scale = torch.tensor(da_model.batch_info.scales)[:, None, None]
        mean = torch.tensor(da_model.batch_info.center)[:, None, None]

        for k, batch in enumerate(
            tqdm(dataloader, disable=dist.get_rank() != 0, desc="Inference")
        ):
            if args.use_analysis:
                analysis = batch["target"]
            else:
                analysis = da_model.get_state(batch)["target"]

            analysis_scaled = analysis.cpu() * scale + mean
            target_scaled = batch["target"].cpu() * scale + mean
            mse = torch.mean((analysis_scaled - target_scaled) ** 2, dim=(0, 2, 3))
            rmse = mse.sqrt()

            for field in ["Z500", "T850", "uas"]:
                cz500 = da_model.batch_info.channels.index(field)
                value = rmse[cz500].item()
                print(f"RMSE {field} {value}")

            batch_times = batch["timestamp"][:, -1].cpu()
            batch_times = batch_times.numpy().astype("datetime64[s]")
            output_index = subsampled_times.get_indexer(batch_times)

            if np.any(output_index == -1):
                raise KeyError(output_index, batch_times)

            write_to_zarr(group, channels, output_index, analysis_scaled.numpy())

    if args.post_process_to_fcn3:
        post_process_to_fcn3(args.output_path, da_model.batch_info)

    if dist.get_world_size() > 1:
        torch.distributed.barrier()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    dist.print0("Inference completed.")


if __name__ == "__main__":
    main()
