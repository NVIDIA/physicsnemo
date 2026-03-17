# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
import pytest
import torch
from datasets.base import BatchInfo, TimeUnit
from training import loop

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class MockLoop(loop.TrainingLoopBase):
    def get_data_loaders(self, batch_gpu: int):
        class MockDataset:
            def __init__(self):
                self.batch_info = BatchInfo(
                    channels=["channel_0", "channel_1"],
                    time_step=1,
                    time_unit=TimeUnit.HOUR,
                    scales=[1.0, 1.0],
                    center=[0.0, 0.0],
                )

        dataset = MockDataset()
        return dataset, None, None

    def get_network(self) -> torch.nn.Module:
        return torch.nn.Linear(10, 10)

    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters)

    def get_loss_fn(self):
        return torch.nn.MSELoss()


@requires_cuda
def test_loop_save_load(tmp_path):
    rundir = tmp_path
    loop_a = MockLoop(rundir.as_posix(), batch_gpu=1)
    loop_a.setup()
    loop_a.save_training_state()

    handler = loop.CheckpointHandler(rundir.as_posix())
    result = handler.latest_checkpoint()
    assert result is not None
    checkpoint_dir, nimg, metadata = result

    loop_b = MockLoop(rundir.as_posix(), batch_gpu=1)
    loop_b.setup()
    loop_b.cur_nimg = nimg
    loop_b.resume_from_state(checkpoint_dir)
