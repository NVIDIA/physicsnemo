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

"""Regenerate AB-UPT fixtures: python -m test.experimental.models.ab_upt.generate_data."""

import torch

from physicsnemo.experimental.models.ab_upt import ABUPT
from test.common.fwdaccuracy import save_output
from test.experimental.models.ab_upt.test_ab_upt import CONFIGS, DATA, inputs


@torch.no_grad()
def main():
    """Write untrained CPU float32 references with independent anchor sets."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    for index, config in enumerate(CONFIGS):
        torch.manual_seed(1729 + index)
        model = ABUPT(**config).cpu().float().eval()
        geometry, supernodes, surface, volume = inputs()
        surface_queries = torch.rand(2, 9, 3)
        volume_queries = torch.rand(2, 11, 3)
        parameters = (
            torch.randn(2, model.parameter_channels)
            if model.parameter_channels
            else None
        )
        args = (
            geometry,
            supernodes,
            surface,
            volume,
            surface_queries,
            volume_queries,
            parameters,
        )
        context = model.encode(*args[:4], parameters=parameters)
        prefix = DATA / f"ab_upt_{index}"
        model.save(str(prefix.with_suffix(".mdlus")))
        save_output(model(*args), DATA / f"ab_upt_{index}_output.pth")
        torch.save(
            {
                "inputs": args,
                "geometry": model.encode_geometry(geometry, supernodes),
                "surface": model.decode_queries(surface_queries, context, "surface"),
                "volume": model.decode_queries(volume_queries, context, "volume"),
                "surface_context": context.surface,
                "volume_context": context.volume,
                "vorticity": model.decode_vorticity(volume_queries, context, step=0.01),
            },
            DATA / f"ab_upt_{index}_methods.pth",
        )
    print(f"Generated CPU float32 AB-UPT fixtures with PyTorch {torch.__version__}")


if __name__ == "__main__":
    main()
