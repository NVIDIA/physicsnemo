#!/bin/bash
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

# do the coverage checks
coverage run \
--rcfile='coverage.pytest.rc' \
-m pytest 

coverage run \
--rcfile='coverage.docstring.rc' \
-m pytest \
--doctest-modules ../physicsnemo/

coverage combine --data-file=.coverage
coverage report -i --omit=*test*

# if you wish to view the report in HTML format uncomment below
# coverage html -i --omit=*test*

# cleanup
rm .coverage
