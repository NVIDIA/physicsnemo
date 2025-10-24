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

from typing import Callable, Optional, Union

from pathlib import Path
import signal
import os
import logging.config

import torch
import torch.distributed


# Set the logger to print INFO with format and stream handler
logger = logging.getLogger("SignalHandler")


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "SignalHandler": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}


def default_handler(signum, frame):
    """
    Default handler to print the signal received.
    """
    logger.info(f"Received signal: {signum}.")


class SignalHandler:
    """
    Signal handler class to handle SIGUSR1 and SIGUSR2 signals.

    To run it with SLURM, use the following sbatch script that sends SIGUSR1 300
    seconds before the job ends:

    .. code-block:: bash

        #!/bin/bash
        #SBATCH --time=10:00
        #SBATCH --signal=USR1@300
        ...

    """

    def __init__(
        self,
        handler: Optional[Callable] = None,
        handler_rank: Optional[int] = None,
        status_path: Optional[Union[str, Path]] = None,
    ):
        """
        Signal handler class to handle SIGUSR1 and SIGUSR2 signals.

        Args:
            handler (Callable, optional): The function to call when the signal is
            received. The function should take two arguments: signum and frame.
            Also, the handler should not call any distributed torch call as it
            might not be called on all processes.

            handler_rank (Optional[int], optional): Rank to run the handler on.
            Run on all ranks if None. Defaults to None.

            status_path (Union[str, Path], optional): Path to save the status of
            the current run.

        """
        if handler is None:
            handler = default_handler

        self.handler = handler
        self.handler_rank = handler_rank

        # Get distributed backend and move the _SIG_RECEIVED to the device
        # Global flag for signaling for all DDP processes
        self._DISTRIBUTED = torch.distributed.is_initialized()
        self._RANK = torch.distributed.get_rank() if self._DISTRIBUTED else 0
        logger.debug(f"Setting rank to: {self._RANK}")
        self._STATUS = "RUNNING"
        self._ENTERED = False

        # Get the device
        backend = torch.distributed.get_backend() if self._DISTRIBUTED else "cpu"
        self.device = torch.device("cuda" if "nccl" in backend else "cpu")
        self._SIG_RECEIVED = torch.tensor([0], dtype=torch.int32).to(self.device)

        # Register signal handlers
        self._register_sigusr_handler()

        # Logger configuration
        logging.config.dictConfig(logging_config)

        # Save the status of the current run
        self._set_status_path(status_path)

    @property
    def STATUS(self):
        """
        Get the status of the current run.
        """
        return self._STATUS

    @STATUS.setter
    def STATUS(self, status: str):
        """
        Set the status of the current run.
        """
        self._write_status(status)
        self._STATUS = status

    def __enter__(self):
        self._ENTERED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Exception: {exc_type} - {exc_value}")
            self.STATUS = "ERROR"
            # bubble up the exception
            return False

        if self.is_stopped():
            self.STATUS = "STOPPED"
        else:
            self.STATUS = "FINISHED"

    def __del__(self):
        """
        Destructor to cleanup resources.

        If this is called before the signal is received, it means the job is finished successfully.
        If this is called after the signal is received, it means the job is stopped.
        Write the status to a file accordingly.
        """
        if (not self._ENTERED) and (self.STATUS != "ERROR"):
            if self.is_stopped():
                self.STATUS = "STOPPED"
            else:
                self.STATUS = "FINISHED"

    def stop(self):
        """
        Manually send a signal to stop.
        """
        self._SIG_RECEIVED += 1

    def is_stopped(self):
        """
        Check if the signal to stop has been received. Must be called on all processes.
        """
        if self._DISTRIBUTED:
            # All reduce to check if any process has received the signal
            torch.distributed.all_reduce(
                self._SIG_RECEIVED, op=torch.distributed.ReduceOp.SUM
            )

        # Return if any process has received the signal
        is_stopped = self._SIG_RECEIVED.item() > 0

        # Write the status to a file
        if is_stopped:
            self._write_status("STOPPED")

        return is_stopped

    def barrier(self):
        """
        Barrier to synchronize all processes.
        """
        if self._DISTRIBUTED:
            torch.distributed.barrier()

    def signal_handler(self, signum, frame):
        """
        Signal handler function to handle SIGUSR1 and SIGUSR2 signals.

        Some processes may not received the signal. So for the handler should be
        used for stopping only not for running important tasks such as saving
        checkpoints or cleanup.
        """
        # Broadcast the signal to all processes
        self.stop()
        logger.debug(
            f"Received signal: {signum}, set flag to stop in rank: {self._RANK}, PID: {os.getpid()}."
        )

        # Run handler for all rank if handler_rank is None
        if self.handler_rank is None:
            self.handler(signum, frame)
        elif self._RANK == self.handler_rank:
            logger.debug(f"Running handler for signal: {signum} on rank {self._RANK}.")
            self.handler(signum, frame)

    def _register_sigusr_handler(self):
        # Register signal handlers
        signal.signal(signal.SIGUSR1, self.signal_handler)
        signal.signal(signal.SIGUSR2, self.signal_handler)

    def _set_status_path(self, status_path: Union[str, Path]):
        # Run only on rank 0
        if self._RANK != 0:
            return

        if status_path is None:
            # Get the JOB_ID from the environment. If not found, set it to "status.txt" in the current directory
            ID = os.environ.get("SLURM_JOB_ID", "status")
            status_path = Path(ID + ".txt")

        if isinstance(status_path, str):
            status_path = Path(status_path)

        # Create a directory if it does not exist
        if status_path is not None:
            status_path.parent.mkdir(parents=True, exist_ok=True)

        self.status_path = status_path
        self._write_status("RUNNING")

    def _write_status(self, status: str):
        # Run only on rank 0
        if self._RANK != 0:
            return

        if self.status_path is not None:
            logger.debug(f"Writing status: {status} to file: {self.status_path}")
            with open(self.status_path, "w") as f:
                f.write(status)


def test_signal_handler():
    """
    Test signal handler with SIGUSR1 and SIGUSR2 signals.
    """
    import time

    local_rank = torch.distributed.get_rank()

    def handler(signum, frame):
        # The handler should not call any distributed torch call as it might not be called on all processes
        print(
            f"\n=======================\nHandler called with signal: {signum}.\n=======================\n"
        )

    # Main training loop
    with SignalHandler(handler) as sighandler:
        for i in range(5):
            # Training
            time.sleep(1)

            # check if the signal is received
            if sighandler.is_stopped():  # this is synced
                break

            # Simulate SIGUSR1 on rank 1
            if local_rank == 1:
                print("Sending SIGUSR1 signal")
                os.kill(os.getpid(), signal.SIGUSR1)

            # Simulate error on rank 3
            if local_rank == 3:
                raise RuntimeError("Simulate error")

    # Run cleanup code
    logger.info(
        f"Cleaning up resources and potentially save checkpoint on rank: {local_rank}"
    )
    time.sleep(1)

    # Check the status file
    if local_rank == 0:
        status_file_name = os.environ.get("SLURM_JOB_ID", "status") + ".txt"
        with open(status_file_name, "r") as f:
            status = f.read()
            logger.info(f"Status: {status}")


if __name__ == "__main__":
    # Print all stdout and stderr messages
    # torchrun --nproc-per-node=4 signal_handlers.py 2>&1
    torch.distributed.init_process_group(backend="gloo")
    test_signal_handler()
