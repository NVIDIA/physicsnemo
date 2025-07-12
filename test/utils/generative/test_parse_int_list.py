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


@import_or_fail("cftime")
def test_parse_int_list(pytestconfig):

    from physicsnemo.utils.diffusion import parse_int_list

    # Test parsing a simple comma-separated list
    input_str = "1,2,5,7,10"
    expected_result = [1, 2, 5, 7, 10]
    assert parse_int_list(input_str) == expected_result

    # Test parsing a range
    input_str = "1-5"
    expected_result = [1, 2, 3, 4, 5]
    assert parse_int_list(input_str) == expected_result

    # Test parsing a combination of ranges and numbers
    input_str = "1,3-6,10"
    expected_result = [1, 3, 4, 5, 6, 10]
    assert parse_int_list(input_str) == expected_result

    # Test parsing a single number
    input_str = "42"
    expected_result = [42]
    assert parse_int_list(input_str) == expected_result
