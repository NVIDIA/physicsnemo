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

r"""Training modules for flood prediction."""

# Lazy import pattern to handle circular dependencies between domain_adaptation and pretraining
# domain_adaptation imports create_scheduler from pretraining, but __init__ imports both
# Using lazy imports ensures modules are fully loaded before accessing functions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking, import normally
    from .domain_adaptation import adapt_model
    from .pretraining import pretrain_model
else:
    # At runtime, use lazy imports to avoid circular dependency issues
    # Functions are imported on first access via __getattr__
    adapt_model = None
    pretrain_model = None


def __getattr__(name: str):
    """Lazy import function to resolve circular dependencies at runtime."""
    if name == "adapt_model":
        from .domain_adaptation import adapt_model as _adapt_model
        globals()["adapt_model"] = _adapt_model
        return _adapt_model
    elif name == "pretrain_model":
        from .pretraining import pretrain_model as _pretrain_model
        globals()["pretrain_model"] = _pretrain_model
        return _pretrain_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["pretrain_model", "adapt_model"]

