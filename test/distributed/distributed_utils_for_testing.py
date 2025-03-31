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


import contextlib
import os


@contextlib.contextmanager
def modify_environment(*remove, **update):
    """
    Context manager to allow modification of the environment variables.

    Based on the implementation here:
    https://stackoverflow.com/questions/2059482/temporarily-modify-the-current-processs-environment

    """

    env = os.environ

    update = update or {}
    remove = remove or []

    # Make sure all update values are strings:
    update = {k: str(v) for k, v in update.items()}

    # Find out which environment variables are updated OR removed
    # This compares the keys in both the remove list and update list
    # and returns the overlap with current env.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())

    # Cache everything getting changed from the default env:
    restore_after = {k: env[k] for k in stomped}

    # Keep a list of things that need to be purged after:
    purge_after = tuple(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(restore_after)
        [env.pop(k, None) for k in purge_after]
