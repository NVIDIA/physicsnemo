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

import logging

import physicsnemo

logger = logging.getLogger("__name__")


def check_datapipe_iterable(
    datapipe: "physicsnemo.Datapipe", nr_iterations: int = 3
) -> bool:
    """Checks if datapipe is iterable

    Parameters
    ----------
    datapipe : physicsnemo.Datapipe
        datapipe to check if iterable
    nr_iterations : int
        number of iterations to check datapipe iterable

    Returns
    -------
    bool
        Test passed
    """
    # Check if datapipe is iterable
    try:
        for i, data in enumerate(datapipe):
            if i >= nr_iterations:
                break
            pass
        assert len(datapipe) > 0  # even if infinite, len should return a int
        return True
    except Exception as e:
        logger.warning(f"Datapipe is not iterable: {e}")
        return False
