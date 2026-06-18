# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""
Centralised logging helper used by all sub-modules.
"""

import logging
import sys


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[0;36m",  # cyan
        logging.INFO: "\033[0;32m",  # green
        logging.WARNING: "\033[1;33m",  # yellow
        logging.ERROR: "\033[0;31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format."""
        msg = super().format(record)
        if sys.stderr.isatty():
            return f"{self.COLORS.get(record.levelno, '')}{msg}{self.RESET}"
        return msg


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure and return a module logger.
    Creates a StreamHandler with a standard timestamp format the first time it
    is called for *name*; subsequent calls for the same name are no-ops because
    the handler is already attached.
    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            _ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
