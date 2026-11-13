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
"""CPU-only preparation for the full-mesh Yaris recipe, before allocating GPUs.

Builds the TRAIN dataset once with ``stats_mode="compute"``: frozen
normalization statistics on kept nodes, the static kNN neighborhood cache and
the kNN edge statistics. Every later rank loads these files.
"""

import os
import time
from pathlib import Path

import hydra
from datapipes.px_yaris_dataset import STATS_FILE
from hydra.utils import instantiate
from metrics.validation_suite import atomic_json
from readers.px_yaris_reader import file_sha256


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="px_yaris_deformer",
)
def main(cfg):
    """Build the statistics and caches of the Yaris datapipe ahead of training."""
    started = time.monotonic()
    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe,
        reader=reader,
        split="train",
        stats_mode="compute",
        stats_workers=int(os.environ.get("YARIS_STATS_WORKERS", "8")),
    )
    sample = dataset[0]
    stats_path = Path(cfg.datapipe.stats_dir) / STATS_FILE
    atomic_json(
        stats_path.parent / "PREPARED.json",
        dict(
            stats_sha256=file_sha256(stats_path),
            split_manifest_signature=reader.manifest_signature,
            outlier_signature=reader.outlier_signature,
            total_nodes=reader.total_nodes,
            kept_nodes=dataset.num_nodes,
            stride=reader.stride,
            dt=reader.dt,
            cases=len(dataset),
            local_knn_k=dataset.local_knn_k,
            local_knn_metric=dataset.local_knn_metric,
            knn_valid_slots=int(dataset.graph.knn_mask.sum()),
            sample=repr(sample),
            elapsed_s=round(time.monotonic() - started, 1),
        ),
    )
    print(
        f"prepared {dataset.num_nodes} kept nodes in {time.monotonic() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
