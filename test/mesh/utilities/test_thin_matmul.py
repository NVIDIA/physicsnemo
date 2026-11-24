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

import pytest
import torch

from physicsnemo.mesh.utilities._thin_matmul import thin_matmul


@pytest.mark.parametrize("rows_shape", [(3,), (7, 3), (2, 7, 3)])
@pytest.mark.parametrize("n_out", [2, 3, 4])
def test_thin_matmul_equals_matmul(rows_shape, n_out):
    """The column products equal ``rows @ matrix`` for vectors, rows and batches of
    rows, and for matrices that change the row length."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(rows_shape, generator=generator, dtype=torch.float64)
    matrix = torch.randn(3, n_out, generator=generator, dtype=torch.float64)

    torch.testing.assert_close(thin_matmul(rows, matrix), rows @ matrix)


def test_thin_matmul_refuses_mismatched_sizes():
    """Rows shorter or longer than the matrix has rows raise, as ``@`` does."""
    with pytest.raises(ValueError, match="cannot multiply"):
        thin_matmul(torch.ones(5, 3), torch.ones(4, 3))
    with pytest.raises(ValueError, match="cannot multiply"):
        thin_matmul(torch.ones(5, 3), torch.ones(2, 3))
