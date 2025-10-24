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

import argparse
import json
import math
from pathlib import Path
from typing import Dict
import datetime
import sys

import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataset import EFWIDatapipe  # noqa: E402

_VARIABLES: set[str] = set()


def main() -> None:
    # Parse command line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Compute dataset statistics (mean/std/min/max)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the dataset directory (containing the 'samples' folder)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers per device",
    )
    args = parser.parse_args()

    # Validate dataset directory
    data_dir: Path = Path(args.dir).expanduser().resolve()
    if not (data_dir / "samples").is_dir():
        raise FileNotFoundError(f"{data_dir}/samples not found.")

    # Initialize distributed manager
    DistributedManager.initialize()
    dist: DistributedManager = DistributedManager()

    # General python logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger: PythonLogger = PythonLogger("main")
    logger0: RankZeroLoggingWrapper = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    # Build datapipes (train & test)
    train_dp: EFWIDatapipe = EFWIDatapipe(
        data_dir=data_dir,
        phase="train",
        batch_size_per_device=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    test_dp: EFWIDatapipe = EFWIDatapipe(
        data_dir=data_dir,
        phase="test",
        batch_size_per_device=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # Define statistics accumulators
    def _init_stats(var_name: str, d: Dict[str, float]) -> None:
        d[f"sum_{var_name}"] = torch.tensor(0.0, device=dist.device)
        d[f"sum_{var_name}2"] = torch.tensor(0.0, device=dist.device)
        d[f"min_{var_name}"] = torch.tensor(float("inf"), device=dist.device)
        d[f"max_{var_name}"] = torch.tensor(float("-inf"), device=dist.device)
        return

    train_stats: Dict[str, torch.Tensor] = {}
    test_stats: Dict[str, torch.Tensor] = {}
    ns_train_local: int = 0  # number of samples processed on *this* rank
    ns_test_local: int = 0

    # Accumulate statistics for train dataset
    logger0.info("Accumulating statistics for train dataset...")
    for i, batch in enumerate(train_dp):
        B: int = next(iter(batch.values())).shape[0]
        ns_train_local += B
        for var_name, v in batch.items():
            if f"sum_{var_name}" not in train_stats:
                _init_stats(var_name, train_stats)
                _VARIABLES.add(var_name)
            nb_points: int = math.prod(v.shape[-2:])
            train_stats[f"sum_{var_name}"] += v.sum() / nb_points
            train_stats[f"sum_{var_name}2"] += (v**2).sum() / nb_points
            train_stats[f"min_{var_name}"] = torch.minimum(
                train_stats[f"min_{var_name}"],
                torch.amin(v),
            )
            train_stats[f"max_{var_name}"] = torch.maximum(
                train_stats[f"max_{var_name}"],
                torch.amax(v),
            )

    logger0.info(f"Discovered variables: {', '.join(_VARIABLES)}")

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Accumulate statistics for test dataset
    logger0.info("Accumulating statistics for test dataset...")
    for i, batch in enumerate(test_dp):
        B: int = next(iter(batch.values())).shape[0]
        ns_test_local += B
        for var_name, v in batch.items():
            if f"sum_{var_name}" not in test_stats:
                _init_stats(var_name, test_stats)
            nb_points: int = math.prod(v.shape[-2:])
            test_stats[f"sum_{var_name}"] += v.sum() / nb_points
            test_stats[f"sum_{var_name}2"] += (v**2).sum() / nb_points
            test_stats[f"min_{var_name}"] = torch.minimum(
                test_stats[f"min_{var_name}"],
                torch.amin(v),
            )
            test_stats[f"max_{var_name}"] = torch.maximum(
                test_stats[f"max_{var_name}"],
                torch.amax(v),
            )

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Reduce across ranks (SUM for sums, MIN/MAX for extrema)
    logger0.info("Reducing across ranks...")
    if dist.world_size > 1:
        ns: torch.Tensor = torch.tensor(
            [ns_train_local, ns_test_local], device=dist.device
        )
        torch.distributed.all_reduce(ns, op=torch.distributed.ReduceOp.SUM)
        ns_train: int = ns[0].item()
        ns_test: int = ns[1].item()

        # Define reduction operations
        opSUM = torch.distributed.ReduceOp.SUM
        opMIN = torch.distributed.ReduceOp.MIN
        opMAX = torch.distributed.ReduceOp.MAX

        # Define buffers for reduction
        vars_sorted = sorted(_VARIABLES)
        buffer_sum = torch.cat(
            [train_stats[f"sum_{v}"].unsqueeze(0) for v in vars_sorted]
            + [train_stats[f"sum_{v}2"].unsqueeze(0) for v in vars_sorted]
            + [test_stats[f"sum_{v}"].unsqueeze(0) for v in vars_sorted]
            + [test_stats[f"sum_{v}2"].unsqueeze(0) for v in vars_sorted],
            dim=0,
        )
        buffer_min = torch.cat(
            [train_stats[f"min_{v}"].unsqueeze(0) for v in vars_sorted]
            + [test_stats[f"min_{v}"].unsqueeze(0) for v in vars_sorted],
            dim=0,
        )
        buffer_max = torch.cat(
            [train_stats[f"max_{v}"].unsqueeze(0) for v in vars_sorted]
            + [test_stats[f"max_{v}"].unsqueeze(0) for v in vars_sorted],
            dim=0,
        )

        torch.distributed.all_reduce(buffer_sum, op=opSUM)
        torch.distributed.all_reduce(buffer_min, op=opMIN)
        torch.distributed.all_reduce(buffer_max, op=opMAX)

        for i, v in enumerate(vars_sorted):
            train_stats[f"sum_{v}"] = buffer_sum[i]
            train_stats[f"sum_{v}2"] = buffer_sum[i + len(vars_sorted)]
            test_stats[f"sum_{v}"] = buffer_sum[i + 2 * len(vars_sorted)]
            test_stats[f"sum_{v}2"] = buffer_sum[i + 3 * len(vars_sorted)]
            train_stats[f"min_{v}"] = buffer_min[i]
            train_stats[f"max_{v}"] = buffer_max[i]
            test_stats[f"min_{v}"] = buffer_min[i + len(vars_sorted)]
            test_stats[f"max_{v}"] = buffer_max[i + len(vars_sorted)]
    else:
        ns_train: int = ns_train_local
        ns_test: int = ns_test_local

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Final computation of mean/std/min/max (on rank 0 only)
    logger0.info("Computing final statistics...")
    if dist.rank == 0:
        all_samples: int = ns_train + ns_test
        final_stats: Dict[str, Dict] = {}
        for v in _VARIABLES:
            train_mean: float = train_stats[f"sum_{v}"].item() / ns_train
            train_std: float = math.sqrt(
                train_stats[f"sum_{v}2"].item() / ns_train - train_mean**2
            )
            test_mean: float = test_stats[f"sum_{v}"].item() / ns_test
            test_std: float = math.sqrt(
                test_stats[f"sum_{v}2"].item() / ns_test - test_mean**2
            )
            all_mean: float = (
                train_stats[f"sum_{v}"].item() + test_stats[f"sum_{v}"].item()
            ) / all_samples
            all_std: float = math.sqrt(
                (train_stats[f"sum_{v}2"].item() + test_stats[f"sum_{v}2"].item())
                / all_samples
                - all_mean**2
            )

            final_stats[v] = {
                "train": {
                    "min": train_stats[f"min_{v}"].item(),
                    "max": train_stats[f"max_{v}"].item(),
                    "mean": train_mean,
                    "std": train_std,
                },
                "test": {
                    "min": test_stats[f"min_{v}"].item(),
                    "max": test_stats[f"max_{v}"].item(),
                    "mean": test_mean,
                    "std": test_std,
                },
                "all": {
                    "min": min(
                        train_stats[f"min_{v}"].item(), test_stats[f"min_{v}"].item()
                    ),
                    "max": max(
                        train_stats[f"max_{v}"].item(), test_stats[f"max_{v}"].item()
                    ),
                    "mean": all_mean,
                    "std": all_std,
                },
            }

        # Save json file
        out_file: Path = data_dir / "stats.json"
        with open(out_file, "w") as f:
            json.dump(final_stats, f, indent=4)
        logger0.success(f"Statistics written to {out_file}")

    # Make sure all ranks wait for I/O completion
    if dist.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
