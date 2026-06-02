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

"""Adapter save/load — a ``.mdlus`` ZIP archive holding only adapter state.

Only the trainable adapter tensors are stored (not the frozen base), so an
adapter is small and reloads onto any architecturally-compatible base. Layout::

    adapter.mdlus (zip)
    ├── adapter_config.json   # loadable LoRAConfig (rank, alpha, target_modules=wrapped, ...)
    ├── adapter_model.pt      # state_dict slice: lora_A/lora_B + extras_trainable params
    └── metadata.json         # {format_version, kind: "lora_adapter", versions, base_fingerprint, ...}

Adapter archives reuse the ``.mdlus`` extension but are disambiguated from full
model checkpoints by ``metadata.kind == "lora_adapter"``. The ``base_fingerprint``
(a hash of the base model's structure, not its weights) lets ``load_adapter``
reject an incompatible base.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

from physicsnemo.experimental.peft.apply import apply_lora
from physicsnemo.experimental.peft.config import LoRAConfig
from physicsnemo.experimental.peft.lora import is_lora_layer
from physicsnemo.experimental.peft.utils import compute_base_fingerprint

logger = logging.getLogger("experimental.peft")

_FORMAT_VERSION = 1
_KIND = "lora_adapter"
_FILES = ("adapter_config.json", "adapter_model.pt", "metadata.json")


def _adapter_state_dict(model: nn.Module) -> dict:
    """The trainable slice: lora_A/lora_B + any extras_trainable params (after
    apply_lora, exactly the params with requires_grad=True)."""
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _wrapped_module_names(model: nn.Module) -> list[str]:
    return [name for name, m in model.named_modules() if is_lora_layer(m)]


def save_adapter(model: nn.Module, path: str | Path) -> None:
    """Save adapter-only state for a LoRA-wrapped ``model`` to ``path`` (a
    ``.mdlus`` archive). The model must have been processed by ``apply_lora``.

    The stored config uses an explicit ``target_modules`` list of the
    actually-wrapped names, so it reloads identically regardless of how the
    layers were originally selected (including a non-serializable
    ``target_filter``).
    """
    path = str(path)
    if not path.endswith(".mdlus"):
        raise ValueError(f"adapter path must end with .mdlus, got {path!r}")

    config = getattr(model, "_lora_config", None)
    if config is None:
        raise ValueError(
            "model has no stashed LoRA config; call apply_lora(model, config) "
            "before save_adapter."
        )
    fingerprint = getattr(model, "_lora_base_fingerprint", "")
    wrapped = _wrapped_module_names(model)

    adapter_config = {
        "rank": config.rank,
        "alpha": config.effective_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": wrapped,  # exact wrapped names → robust reload
        "extras_trainable": list(config.extras_trainable),
        "init": config.init,
    }

    import physicsnemo  # lazy: avoid any import-time cycle

    metadata = {
        "format_version": _FORMAT_VERSION,
        "kind": _KIND,
        "physicsnemo_version": getattr(physicsnemo, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "base_fingerprint": fingerprint,
        "n_wrapped": len(wrapped),
        "rank": config.rank,
        "alpha": config.effective_alpha,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    state_buffer = io.BytesIO()
    torch.save(_adapter_state_dict(model), state_buffer)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("adapter_model.pt", state_buffer.getvalue())
        archive.writestr("adapter_config.json", json.dumps(adapter_config, indent=2))
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))

    logger.info("save_adapter: wrote %d wrapped layers to %s", len(wrapped), path)


def load_adapter(model: nn.Module, path: str | Path, strict: bool = True) -> None:
    """Load an adapter into a compatible base ``model`` (mutated in place):
    verify it is a LoRA adapter, check the base fingerprint, re-apply LoRA to
    the same modules, then load the adapter weights.

    Parameters
    ----------
    strict : bool
        If True (default), a base-fingerprint mismatch raises. If False, it
        only logs a warning (you assert the base is compatible).
    """
    path = str(path)
    with zipfile.ZipFile(path, "r") as archive:
        present = set(archive.namelist())
        missing = [f for f in _FILES if f not in present]
        if missing:
            raise IOError(f"{path} is missing adapter files {missing}.")
        metadata = json.loads(archive.read("metadata.json"))
        adapter_config = json.loads(archive.read("adapter_config.json"))
        state_bytes = archive.read("adapter_model.pt")

    if metadata.get("kind") != _KIND:
        raise ValueError(
            f"{path} is not a LoRA adapter (kind={metadata.get('kind')!r}). "
            "If this is a full model checkpoint, load it with "
            "physicsnemo.Module.load / from_checkpoint instead."
        )

    current_fp = compute_base_fingerprint(model)
    saved_fp = metadata.get("base_fingerprint", "")
    if not saved_fp:
        logger.warning(
            "adapter %s has no base_fingerprint; skipping architecture "
            "compatibility check (load relies on name/shape matching only).",
            path,
        )
    elif saved_fp != current_fp:
        msg = (
            f"base fingerprint mismatch: adapter was trained on a different "
            f"base model (adapter={saved_fp}, this model={current_fp}). The "
            "architectures likely differ."
        )
        if strict:
            raise ValueError(msg + " Pass strict=False to load anyway.")
        logger.warning(msg)

    config = LoRAConfig(
        rank=adapter_config["rank"],
        alpha=adapter_config["alpha"],
        lora_dropout=adapter_config.get("lora_dropout", 0.0),
        target_modules=adapter_config["target_modules"],
        extras_trainable=adapter_config.get("extras_trainable", []),
        init=adapter_config.get("init", "default"),
    )
    apply_lora(model, config)

    state = torch.load(io.BytesIO(state_bytes), map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    # load_state_dict(strict=False) reports the (expected) frozen base keys as
    # "missing"; what must be empty is "unexpected" — adapter keys not in model.
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(
            f"adapter contains keys not present after re-applying LoRA: "
            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}. The "
            "adapter and base model are incompatible."
        )
    logger.info("load_adapter: loaded %d adapter tensors from %s", len(state), path)
