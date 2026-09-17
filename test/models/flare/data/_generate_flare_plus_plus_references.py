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

"""Regenerate the committed FLARE++ forward and checkpoint references.

Run from the repository root::

    python test/models/flare/data/_generate_flare_plus_plus_references.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from physicsnemo.models.flare import FLAREPlusPlus  # noqa: E402

DATA_DIR = Path(__file__).parent


def _make_model(n_hidden: int, n_head: int, slice_num: int, out_dim: int):
    return FLAREPlusPlus(
        functional_dim=2,
        out_dim=out_dim,
        embedding_dim=3,
        n_layers=2,
        n_hidden=n_hidden,
        n_head=n_head,
        mlp_ratio=2 if out_dim == 2 else 1,
        slice_num=slice_num,
        structured_shape=None,
    )


def main() -> None:
    for n_hidden, n_head, slice_num, file_name in (
        (16, 4, 4, "flare_plus_plus_small_output.pth"),
        (24, 3, 5, "flare_plus_plus_custom_output.pth"),
    ):
        torch.manual_seed(1234)
        model = _make_model(n_hidden, n_head, slice_num, out_dim=2)
        functional_input = torch.randn(2, 17, 2)
        embedding = torch.randn(2, 17, 3)
        with torch.no_grad():
            output = model(functional_input, embedding)
        torch.save({0: output.detach().contiguous()}, DATA_DIR / file_name)

    torch.manual_seed(0)
    checkpoint_model = _make_model(16, 4, 4, out_dim=1)
    checkpoint_model.save(DATA_DIR / "flare_plus_plus_v1.mdlus")


if __name__ == "__main__":
    main()
