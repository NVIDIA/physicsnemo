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

"""Deploy a trained LoRA adapter (companion to ``finetune.py``).

Two modes:
  * Adapter-swap: keep the frozen base + small adapter, ``load_adapter`` at
    serve time (one base + N adapters, swappable per request).
  * Merge: fold the adapter into the base for zero inference overhead and save
    a plain ``.mdlus``. (Fused ``te.LayerNormMLP`` residual adapters are not
    mergeable and are left in place; deploy those via adapter-swap instead.)

Run from the example root::

    python src/finetune/deploy.py init_from=<base.mdlus>            # adapter-swap
    python src/finetune/deploy.py init_from=<base.mdlus> merge=true  # fold in
"""

import logging

import hydra
from omegaconf import DictConfig

from physicsnemo.experimental.peft import is_lora_layer, load_adapter, merge_lora

logger = logging.getLogger("finetune_lora.deploy")


@hydra.main(version_base=None, config_path="../conf", config_name="finetune_lora")
def main(cfg: DictConfig) -> None:
    """Load a trained adapter onto the base for serving (adapter-swap), optionally
    merging it into the base weights (``merge=true``) for zero-overhead inference."""
    logging.basicConfig(level=logging.INFO)

    # Reconstruct the SAME base architecture, then load its pretrained weights.
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    if cfg.get("init_from"):
        model.load(str(cfg.init_from))

    adapter_path = f"{cfg.output_dir}/{cfg.run_id}.lora"
    # load_adapter verifies kind + base fingerprint, re-applies LoRA, loads weights.
    load_adapter(model, adapter_path)
    logger.info("loaded adapter %s onto base", adapter_path)

    if cfg.get("merge", False):
        merge_lora(model)  # fold mergeable adapters into base weights
        remaining = [n for n, m in model.named_modules() if is_lora_layer(m)]
        if remaining:
            # e.g. te.LayerNormMLP residuals are non-mergeable; saving now would
            # write wrapper-prefixed keys that won't reload as the base model.
            logger.warning(
                "merge requested but %d non-mergeable adapter(s) remain "
                "(e.g. te.LayerNormMLP residuals); NOT writing a merged "
                "checkpoint. Serve with the adapter via load_adapter instead.",
                len(remaining),
            )
        else:
            merged_path = f"{cfg.output_dir}/{cfg.run_id}_merged.mdlus"
            model.save(merged_path)  # plain full-model .mdlus, no adapter overhead
            logger.info("merged and saved %s", merged_path)

    model.eval()
    logger.info("model ready for inference")


if __name__ == "__main__":
    main()
