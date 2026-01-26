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
"""Utilities for catching unix signals and gracefully exiting

Usage:
```
from utils import signals
import signal

# now can catch signals with exceptions
signal.signal(signal.SIGTERM, signals.handler)


def do_stuff():
    do_optional_stuff()
    with signals.finish_before_quitting():
        do_stuff_i_need_to_finish()

try:
    do_stuff()
except signals.QuitEarly:
    cleanup()

```

"""


class QuitEarly(Exception):
    depth = 0
    quit_requested = False


def finish_before_quitting(func):
    """If signal caught defer quitting until the wrapped line of code completes

    Used to handle sensitive code blocks
    """

    def newfunc(*args, **kwargs):
        QuitEarly.depth += 1
        func(*args, **kwargs)
        QuitEarly.depth -= 1

        if QuitEarly.quit_requested and QuitEarly.depth == 0:
            QuitEarly.quit_requested = False
            raise QuitEarly()

    return newfunc


def handler(signum, frame):
    if QuitEarly.depth == 0:
        raise QuitEarly(signum, frame)
    else:
        QuitEarly.quit_requested = True
