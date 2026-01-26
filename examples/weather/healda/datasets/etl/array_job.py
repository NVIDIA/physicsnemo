# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LaunchMode(Enum):
    """Distributed launch backend detection mode."""

    SLURM = "slurm"  # SLURM job arrays
    TORCH = "torch"  # torch.distributed
    MANUAL = "manual"  # Explicit rank/world_size for single node, multi-process
    AUTO = "auto"  # Auto-detect, defaulting to SLURM when slurm and torch are present


class WorkManager:
    def __init__(
        self,
        path: str = "",
        *,
        launch_mode: LaunchMode = LaunchMode.AUTO,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Args:
            path: Directory for task completion logs
            launch_mode: Backend detection mode
            rank: Manual rank (required for MANUAL mode)
            world_size: Manual world size (required for MANUAL mode)
        """
        self.job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "0")

        if launch_mode == LaunchMode.MANUAL:
            if rank is None or world_size is None:
                raise ValueError("MANUAL mode requires rank and world_size")
            self.rank, self.world_size = rank, world_size

        elif launch_mode == LaunchMode.TORCH:
            self.rank, self.world_size = self._init_torch()

        elif launch_mode == LaunchMode.SLURM:
            self.rank, self.world_size = self._init_slurm()

        elif launch_mode == LaunchMode.AUTO:
            self.rank, self.world_size = self._auto_detect()

        else:
            raise ValueError(f"Unknown launch mode: {launch_mode}")

        logger.info(
            f"WorkManager(mode={launch_mode}): rank={self.rank}, world_size={self.world_size}"
        )

        if path:
            os.makedirs(path, exist_ok=True)
        self.path = path

    def _init_torch(self) -> tuple[int, int]:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )

    def _init_slurm(self) -> tuple[int, int]:
        slurm_rank = int(os.environ.get("SLURM_PROCID", 0))
        slurm_world = int(os.environ.get("SLURM_NPROCS", 1))

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 1))
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", 1))

        array_rank = task_id - task_min
        array_world = task_max - task_min + 1

        rank = array_rank * slurm_world + slurm_rank
        world = array_world * slurm_world
        return rank, world

    def _auto_detect(self) -> tuple[int, int]:
        has_torch = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        has_slurm = "SLURM_PROCID" in os.environ
        has_slurm_array = "SLURM_ARRAY_TASK_ID" in os.environ

        # Priority order:
        # 1. SLURM array jobs: Must use SLURM logic to compute global rank across array
        #    (RANK/WORLD_SIZE from dist.init() are only per-task, not global)
        # 2. torch.distributed: Use when set by torchrun or explicit dist.init()
        #    (single-task srun + dist.init() sets RANK correctly for that task)
        # 3. SLURM non-array: Fallback for pure SLURM without torch

        if has_slurm_array and has_slurm:
            if has_torch:
                logger.info(
                    "SLURM array job detected. Using SLURM to compute global rank "
                    "across array tasks (RANK/WORLD_SIZE are per-task only)."
                )
            return self._init_slurm()

        if has_torch:
            return self._init_torch()

        if has_slurm:
            return self._init_slurm()

        # Single process fallback
        return 0, 1

    def split(self, tasks: list[int], seed: int = 0):
        # seed the random number generator with the job id
        # so that tasks are reshuffled across the array when a new job is
        # started
        rng = random.Random(self.job_id)
        tasks = tasks.copy()
        rng.shuffle(tasks)
        return tasks[self.rank :: self.world_size]

    def task_done(self, task: int):
        if not self.path:
            return

        filename = f"{self.rank}.log"
        logger.info(f"marking {task} as done")
        with open(os.path.join(self.path, filename), "a") as f:
            print(task, file=f)

    def _get_completed_tasks(self):
        log_files = glob.glob(os.path.join(self.path, "*.log"))
        tasks = set()
        for file in log_files:
            with open(file) as f:
                tasks_in_file = [int(line.strip()) for line in f.readlines()]
            tasks.update(tasks_in_file)
        return tasks

    def map(self, func: Callable[int, None], tasks: list[int]):
        completed_tasks = self._get_completed_tasks()
        my_tasks = self.split(tasks)
        logger.info(
            f"Starting tasks: {len(completed_tasks)} total tasks already completed globally of {len(tasks)}. Have {len(my_tasks)} tasks to run on rank {self.rank} of {self.world_size}."
        )
        for task in my_tasks:
            if task in completed_tasks:
                logger.info(f"{task} already done. skipping.")
                continue

            func(task)
            self.task_done(task)
