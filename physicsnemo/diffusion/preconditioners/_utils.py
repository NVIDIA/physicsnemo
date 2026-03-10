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

from typing import Any


def _wrapped_property(prop_name: str, wrapped_obj_name: str, doc: str) -> property:
    """
    Property factory to define a property on a Module ``self`` that is
    wraps another Module in an attribute ``self.<wrapped_obj_name>``. The
    property delegates the setter and getter to the wrapped object's.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    wrapped_obj_name : str
        The name of the attribute that wraps the other Module.
    doc : str
        The documentation string for the property.

    Returns
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        wrapped_obj = getattr(self, wrapped_obj_name)
        if hasattr(wrapped_obj, prop_name):
            setattr(wrapped_obj, prop_name, value)
        else:
            raise AttributeError(f"{prop_name} is not supported by the wrapped model.")

    def _getter(self):
        wrapped_obj = getattr(self, wrapped_obj_name)
        return getattr(wrapped_obj, prop_name)

    return property(_getter, _setter, doc=doc)
