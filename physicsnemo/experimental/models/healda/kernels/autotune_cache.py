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
"""Persist the winning Triton ``@autotune`` tile config per shape key (warps,
stages, block sizes) to JSON (not the compiled kernels themselves, whose caching is controlled
by Triton).

HealDA pixel attention uses this only when ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR``
is set (opt-in). Unset means Triton will autotune on first-use per shape key.

Reusable across any triton.autotune'd kernel: pass {name: autotuner} where each
autotuner is a kernel decorated with @triton.autotune (a triton Autotuner with a
`.cache` dict {key_tuple: Config}).

Workflow:
  * load_caches(autotuners, path)  -> repopulate .cache from JSON (skips benchmark)
  * <run / prewarm the kernels>    -> autotune fills .cache for any missing keys
  * save_caches(autotuners, path)  -> write .cache back to JSON

The JSON is keyed by GPU name (configs differ per arch -> regenerated on a new
cluster automatically). Key elements are ints/strings plus a triton dtype; the
dtype round-trips via repr() and is decoded through a fixed whitelist (never
eval()) so a tampered/corrupt cache file cannot execute arbitrary code.
"""

import json
import os
import warnings

import torch

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
tl = OptionalImport("triton.language")


def _key_dtypes():
    # The only non-primitive autotune-key element is COMPUTE_DTYPE (a triton
    # dtype). Whitelist by repr() so decoding is an exact-match lookup, not eval().
    return {repr(d): d for d in (tl.float32, tl.float16, tl.bfloat16)}


def gpu_tag() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "-")
    return "cpu"


def _enc_key_elem(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return {"__eval__": repr(x)}  # e.g. triton dtype -> "triton.language.bfloat16"


def _dec_key_elem(x):
    if isinstance(x, dict) and "__eval__" in x:
        s = x["__eval__"]
        key_dtypes = _key_dtypes()
        if s not in key_dtypes:
            raise ValueError(f"unexpected non-primitive autotune key element: {s!r}")
        return key_dtypes[s]
    return x


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(
            f"ignoring unreadable triton autotune cache {path}: {e}", stacklevel=2
        )
        return {}


def _enc_config(c) -> dict:
    return {
        "kwargs": dict(c.kwargs),
        "num_warps": c.num_warps,
        "num_stages": c.num_stages,
        "num_ctas": getattr(c, "num_ctas", 1),
        "maxnreg": getattr(c, "maxnreg", None),
    }


def _dec_config(d):
    return triton.Config(
        d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg", None),
    )


def load_caches(autotuners: dict, path: str) -> int:
    """Repopulate each autotuner's .cache from the JSON file (for this GPU).
    Returns the number of (kernel,key) entries loaded."""
    if not path or not os.path.exists(path):
        return 0
    data = _read_json(path)
    g = data.get(gpu_tag(), {})
    n = 0
    for name, tuner in autotuners.items():
        cache = getattr(tuner, "cache", None)
        if cache is None:
            continue
        for entry in g.get(name, []):
            key = tuple(_dec_key_elem(e) for e in entry["key"])
            cache[key] = _dec_config(entry["config"])
            n += 1
    return n


def save_caches(autotuners: dict, path: str) -> int:
    """Write each autotuner's .cache to the JSON file (merged, per GPU).
    Returns the number of entries written for this GPU."""
    data = _read_json(path) if os.path.exists(path) else {}
    g = data.setdefault(gpu_tag(), {})
    n = 0
    for name, tuner in autotuners.items():
        entries = []
        for key, cfg in getattr(tuner, "cache", {}).items():
            entries.append(
                {"key": [_enc_key_elem(e) for e in key], "config": _enc_config(cfg)}
            )
            n += 1
        g[name] = entries
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return n
