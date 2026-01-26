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
from unittest.mock import patch

from training.loop import CheckpointHandler


def test_checkpoint_handler_basic_functionality(tmp_path):
    """Test core CheckpointHandler functionality."""
    handler = CheckpointHandler(str(tmp_path))

    # Test get_path
    assert handler.get_path(123) == f"{tmp_path}/training-state-000000123.checkpoint"

    # Test list_checkpoints with mock files
    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = ["training-state-000000001.checkpoint", "invalid.txt"]
        checkpoints = list(handler.list_checkpoints())
        assert checkpoints == [(f"{tmp_path}/training-state-000000001.checkpoint", 1)]
