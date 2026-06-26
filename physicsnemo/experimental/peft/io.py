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

"""Adapter save/load — a plain ZIP archive holding only adapter state.

Only the trainable adapter tensors are stored (not the frozen base), so an
adapter is small and reloads onto any architecturally-compatible base. Works on
any ``torch.nn.Module`` — no dependency on ``physicsnemo.Module`` or the
``.mdlus`` checkpoint format. Layout::

    adapter (zip; any file extension)
    ├── adapter_config.json   # loadable LoRAConfig (rank, alpha, target_modules=wrapped, ...)
    ├── adapter_model.pt      # state_dict slice: lora_A/lora_B + extras_trainable params
    └── metadata.json         # {format_version, kind: "lora_adapter", versions, base_fingerprint, ...}

The file extension is unconstrained, but a dedicated one such as ``.lora`` is
recommended: this archive is read only by :func:`load_adapter`. Naming it
``.pt`` would wrongly imply ``torch.load`` and ``.mdlus`` would wrongly imply
``physicsnemo.Module.load`` — neither can read it (they expect different
contents and error out). The ``metadata.kind`` field marks it as an adapter, and
``base_fingerprint`` (a hash of the base model's structure, not its weights)
lets ``load_adapter`` reject an incompatible base.
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
    """Fully-qualified names of all LoRA-wrapped submodules in ``model``."""
    return [name for name, m in model.named_modules() if is_lora_layer(m)]


def save_adapter(model: nn.Module, path: str | Path) -> None:
    """Save adapter-only state for a LoRA-wrapped ``model`` to ``path``.

    The archive is a plain multi-file ZIP (contents below) — load it with
    :func:`load_adapter`, never ``torch.load`` or ``physicsnemo.Module.load``.
    Any file extension is accepted, but a dedicated one such as ``.lora`` is
    recommended: ``.pt`` implies ``torch.load`` and ``.mdlus`` implies
    ``Module.load``, and neither can read this archive. The model must have been
    processed by ``apply_lora``.

    Archive contents:
      - ``adapter_config.json`` — the adapter config (rank, alpha, dropout, init,
        and an explicit ``target_modules`` list of the actually-wrapped names, so it
        reloads identically regardless of the original selector, including a
        non-serializable ``target_filter``).
      - ``adapter_model.pt`` — the trainable tensors only: ``lora_A``/``lora_B``
        and any ``extras_trainable`` params (the frozen base is NOT stored).
      - ``metadata.json`` — ``kind="lora_adapter"``, format/library versions, the
        base fingerprint, and a summary (n_wrapped, rank, alpha, timestamp).
    """
    path = str(path)

    meta = getattr(model, "_lora_adapter_config", None)
    if meta is None:
        raise ValueError(
            "model has no stashed LoRA config; call apply_lora(model, config) "
            "before save_adapter."
        )
    fingerprint = getattr(model, "_lora_base_fingerprint", "")
    wrapped = _wrapped_module_names(model)
    if not wrapped:
        raise ValueError(
            "no LoRA layers found in model; cannot save an adapter. If "
            "merge_lora was already called, the adapter has been folded into "
            "the base weights — save a full model checkpoint (e.g. model.save() "
            "for a physicsnemo.Module) instead."
        )

    adapter_config = {
        "rank": meta["rank"],
        "alpha": meta["alpha"],
        "lora_dropout": meta["lora_dropout"],
        "target_modules": wrapped,  # exact wrapped names → robust reload
        "extras_trainable": list(meta["extras_trainable"]),
        # "custom" if a callable init was used (not recoverable); see apply_lora.
        "init": meta["init"],
    }

    import physicsnemo  # lazy: avoid any import-time cycle

    metadata = {
        "format_version": _FORMAT_VERSION,
        "kind": _KIND,
        "physicsnemo_version": getattr(physicsnemo, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "base_fingerprint": fingerprint,
        "n_wrapped": len(wrapped),
        "rank": meta["rank"],
        "alpha": meta["alpha"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    state_buffer = io.BytesIO()
    torch.save(_adapter_state_dict(model), state_buffer)

    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)

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

    # init only seeds lora_A at apply time and is then overwritten by the loaded
    # adapter weights, so the saved label (which may be "custom" for a callable, not
    # re-runnable) is irrelevant on reload. Always use "default".
    config = LoRAConfig(
        rank=adapter_config["rank"],
        alpha=adapter_config["alpha"],
        lora_dropout=adapter_config.get("lora_dropout", 0.0),
        target_modules=adapter_config["target_modules"],
        extras_trainable=adapter_config.get("extras_trainable", []),
        init="default",
    )
    apply_lora(model, config)

    # weights_only=True restricts unpickling to safe tensor types: adapters may
    # be distributed independently of the base model (community / NIM), so a
    # malicious adapter_model.pt must not be able to execute arbitrary code.
    state = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=True)
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
