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

from pytest_utils import import_or_fail


# Test tuple_product function
@import_or_fail("cftime")
def test_tuple_product(pytestconfig):

    from physicsnemo.utils.diffusion import tuple_product

    # Test with an empty tuple
    assert tuple_product(()) == 1

    # Test with a tuple containing one element
    assert tuple_product((5,)) == 5

    # Test with a tuple containing multiple elements
    assert tuple_product((2, 3, 4)) == 24
    assert tuple_product((1, 2, 3, 4, 5)) == 120
    assert tuple_product((10, 20, 30, 40)) == 240000
