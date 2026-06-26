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

"""apply_lora / resolve_targets / utils tests (CPU-only).

Uses a toy model mirroring the GeoTransolver ``blocks.{i}.Attn.*`` naming so
the targeting tests exercise realistic fully-qualified-name patterns.
"""

import pytest
import torch
import torch.nn as nn

from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    is_lora_layer,
    print_trainable_parameters,
    set_adapter_enabled,
    split_params_for_optimizer,
)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv_project = nn.Linear(d, 3 * d, bias=False)
        self.out_linear = nn.Linear(d, d)


class _Ffn(nn.Module):
    """Mimics physicsnemo.nn.Mlp: Linears live in a `layers` Sequential."""

    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        return self.layers(x)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Attn = _Attn(d)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)
        # GALE_block-style FFN naming (non-TE path): Sequential(LayerNorm, Mlp)
        self.ln_mlp1 = nn.Sequential(nn.LayerNorm(d), _Ffn(d))


class _Toy(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])
        self.head = nn.Linear(d, 4)

    def forward(self, x):
        for b in self.blocks:
            x = b.Attn.out_linear(b.norm(x)) + b.mlp(x)
        return self.head(x)


def _model():
    torch.manual_seed(0)
    return _Toy()


_ATTN_PATTERN = r"blocks\.\d+\.Attn\."


@pytest.mark.parametrize(
    "selector, expected, wrapped, unwrapped",
    [
        # regex over fully-qualified names → qkv_project + out_linear per block
        (
            dict(target_pattern=_ATTN_PATTERN),
            3 * 2,
            "blocks.0.Attn.qkv_project",
            "head",
        ),
        # exact module name
        (dict(target_modules=["head"]), 1, "head", "blocks.0.Attn.qkv_project"),
        # callable predicate (the per-block `mlp` Linear)
        (
            dict(target_filter=lambda name, mod: name.endswith("mlp")),
            3,
            "blocks.0.mlp",
            "head",
        ),
    ],
)
def test_selector_targeting(selector, expected, wrapped, unwrapped):
    m = _model()
    res = apply_lora(m, LoRAConfig(rank=2, **selector))
    assert res.n_wrapped == expected
    mods = dict(m.named_modules())
    assert is_lora_layer(mods[wrapped])
    assert not is_lora_layer(mods[unwrapped])


def test_zero_match_raises():
    m = _model()
    with pytest.raises(ValueError, match="0 wrappable"):
        apply_lora(m, LoRAConfig(rank=2, target_pattern="does_not_exist"))


def test_double_apply_raises():
    m = _model()
    apply_lora(m, LoRAConfig(rank=2, target_modules=["head"]))
    with pytest.raises(ValueError, match="already contains LoRA"):
        apply_lora(m, LoRAConfig(rank=2, target_modules=["head"]))


def test_freezing_and_extras():
    m = _model()
    cfg = LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN, extras_trainable=["head"])
    apply_lora(m, cfg)
    assert m.blocks[0].Attn.qkv_project.lora_A.requires_grad
    assert not m.blocks[0].Attn.qkv_project.base_layer.weight.requires_grad
    assert all(p.requires_grad for p in m.head.parameters())  # extras
    assert not m.blocks[0].mlp.weight.requires_grad  # untargeted, frozen


def test_extras_trainable_does_not_unfreeze_nested_lora_base():
    # extras_trainable naming a CONTAINER that also holds LoRA-wrapped layers must
    # not re-enable grad on their frozen base weights (would bloat the adapter).
    m = _model()
    apply_lora(
        m,
        LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN, extras_trainable=["blocks.0"]),
    )
    blk = m.blocks[0]
    assert not blk.Attn.qkv_project.base_layer.weight.requires_grad  # stays frozen
    assert not blk.Attn.out_linear.base_layer.weight.requires_grad
    assert blk.norm.weight.requires_grad  # block's own non-wrapped params trainable
    assert blk.mlp.weight.requires_grad
    assert blk.Attn.qkv_project.lora_A.requires_grad  # lora trainable as usual


def test_split_params_routes_lora_separately():
    m = _model()
    apply_lora(
        m, LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN, extras_trainable=["head"])
    )
    groups = split_params_for_optimizer(m)
    assert len(groups["lora"]) == 3 * 2 * 2  # lora_A + lora_B per wrapped layer
    assert len(groups["extras"]) == 2  # head weight + bias
    assert len(groups["frozen"]) > 0


def test_result_counts_and_fingerprint_stashed():
    m = _model()
    res = apply_lora(m, LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN))
    assert res.base_fingerprint == m._lora_base_fingerprint
    assert res.n_trainable > 0 and res.n_frozen > 0
    assert len(res.trainable_names) == 3 * 2 * 2


def test_set_adapter_enabled_roundtrip():
    m = _model()
    apply_lora(m, LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN))
    for mod in m.modules():
        if is_lora_layer(mod):
            with torch.no_grad():
                mod.lora_B.copy_(torch.randn_like(mod.lora_B))
    x = torch.randn(2, 5, 16)
    set_adapter_enabled(m, False)
    base_out = m(x)
    set_adapter_enabled(m, True)
    assert not torch.allclose(base_out, m(x))


def test_wrap_mlp_includes_ffn_linears():
    m = _model()
    # attention pattern alone → 2 per block; + wrap_mlp adds the FFN Linears
    # (ln_mlp1.1.layers.0 and .2 — the GELU at .1 is filtered out by type).
    res = apply_lora(m, LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN, wrap_mlp=True))
    assert res.n_wrapped == 3 * (2 + 2)  # (qkv, out_linear) + (ffn fc1, fc2) per block
    assert is_lora_layer(m.blocks[0].ln_mlp1[1].layers[0])
    assert is_lora_layer(m.blocks[0].ln_mlp1[1].layers[2])
    assert not is_lora_layer(m.blocks[0].ln_mlp1[0])  # the LayerNorm, untouched


def test_print_trainable_parameters():
    m = _model()
    apply_lora(m, LoRAConfig(rank=4, target_pattern=_ATTN_PATTERN))
    assert "trainable params" in print_trainable_parameters(m)


def test_warns_on_selector_match_without_wrapper(caplog):
    # A selector that matches a layer with no registered wrapper (e.g. an
    # embedding) should warn and skip it, not silently no-op.
    import logging

    class _NetWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 8)  # no registered LoRA wrapper
            self.fc = nn.Linear(8, 8)

    m = _NetWithEmbedding()
    with caplog.at_level(logging.WARNING, logger="experimental.peft"):
        res = apply_lora(m, LoRAConfig(rank=2, target_pattern=r"emb|fc"))

    assert res.n_wrapped == 1  # only fc wrapped
    assert is_lora_layer(m.fc)
    assert not is_lora_layer(m.emb)  # embedding skipped
    assert "emb" in caplog.text and "no wrapper is registered" in caplog.text


def test_apply_honors_config_init():
    # config.init flows through apply_lora into the wrapper's lora_A init.
    m = _model()
    apply_lora(
        m,
        LoRAConfig(
            rank=4,
            target_pattern=_ATTN_PATTERN,
            init=lambda t: nn.init.constant_(t, 0.25),
        ),
    )
    qkv = m.blocks[0].Attn.qkv_project
    assert torch.allclose(qkv.lora_A, torch.full_like(qkv.lora_A, 0.25))


def test_wrapped_model_is_picklable_with_callable_selectors():
    # apply_lora stashes a plain metadata dict (no callables), so a wrapped model
    # stays picklable even when the user passed a callable target_filter and init.
    import pickle

    m = _model()
    apply_lora(
        m,
        LoRAConfig(
            rank=4,
            target_filter=lambda name, mod: "Attn" in name,
            init=lambda t: nn.init.normal_(t, std=0.01),
        ),
    )
    # the stash holds only serializable metadata; a callable init is labelled.
    assert isinstance(m._lora_adapter_config, dict)
    assert m._lora_adapter_config["init"] == "custom"

    # Round-trip a locally-constructed, trusted model (NOT untrusted data): the
    # dumps would raise if a lambda were retained, and loads confirms the wrappers
    # survive. S301 (pickle-of-untrusted-data) does not apply here.
    blob = pickle.dumps(m)
    restored = pickle.loads(blob)  # noqa: S301
    assert any(is_lora_layer(mod) for mod in restored.modules())


def test_is_compatible_vetoes_incompatible_instance(caplog):
    # A wrapper can veto a specific instance via is_compatible; resolve_targets
    # then skips it (with a warning) instead of wrapping it.
    import logging

    from physicsnemo.experimental.peft import register_lora_wrapper
    from physicsnemo.experimental.peft.lora import _LORA_WRAPPERS, LoRALinear

    class _PickyLinear(nn.Linear):
        pass

    class _PickyLoRA(LoRALinear):
        @classmethod
        def is_compatible(cls, base_layer):
            return False  # never adaptable

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.picky = _PickyLinear(8, 8)
            self.fc = nn.Linear(8, 8)

    register_lora_wrapper(_PickyLinear, _PickyLoRA)
    try:
        m = _Net()
        with caplog.at_level(logging.WARNING, logger="experimental.peft"):
            res = apply_lora(m, LoRAConfig(rank=2, target_pattern=r"picky|fc"))
        assert res.n_wrapped == 1  # only fc; picky vetoed
        assert is_lora_layer(m.fc)
        assert not is_lora_layer(m.picky)
        assert "picky" in caplog.text and "is_compatible" in caplog.text
    finally:
        _LORA_WRAPPERS.pop(_PickyLinear, None)  # don't leak into other tests


def test_apply_rejects_wrapper_not_subclassing_loralayer():
    # register_lora_wrapper contract: the factory must return a LoRALayer
    # subclass. apply_lora rejects one that does not — otherwise freeze/save/merge
    # (which key on isinstance(module, LoRALayer)) would silently skip it.
    from physicsnemo.experimental.peft import register_lora_wrapper
    from physicsnemo.experimental.peft.lora import _LORA_WRAPPERS

    class _Custom(nn.Linear):
        pass

    class _BadWrapper(nn.Module):  # accepts the ctor args but is NOT a LoRALayer
        def __init__(self, base_layer, *args, **kwargs):
            super().__init__()
            self.base_layer = base_layer

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = _Custom(8, 8)

    register_lora_wrapper(_Custom, _BadWrapper)
    try:
        with pytest.raises(TypeError, match="does not subclass LoRALayer"):
            apply_lora(_Net(), LoRAConfig(rank=2, target_modules=["fc"]))
    finally:
        _LORA_WRAPPERS.pop(_Custom, None)  # don't leak into other tests
