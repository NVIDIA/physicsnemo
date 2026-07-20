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

"""Nonlinear deformation operations for simplicial meshes."""

from functools import wraps

from physicsnemo.mesh.transformations.deform.displace import displace
from physicsnemo.mesh.transformations.deform.ffd import ffd as _ffd
from physicsnemo.mesh.transformations.deform.morph import morph


@wraps(_ffd, assigned=("__doc__", "__annotations__"))
def free_form_deform(*args, **kwargs):
    """Expose free-form deformation under its public API name."""

    return _ffd(*args, **kwargs)


__all__ = ["displace", "free_form_deform", "morph"]
