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
"""Tests for the experimental S2Transformer (:class:`S2Transformer`).

Most tests build a tiny CPU-friendly model (global attention) and check the
forward pass, encoder/decoder activation, positional embeddings, grid types,
input validation and checkpoint round-trips. The reference tests compare two
frozen configurations against outputs stored in ``data/``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

# The model requires torch_harmonics; skip the whole module if it is missing.
pytest.importorskip("torch_harmonics")

from physicsnemo import Module  # noqa: E402
from physicsnemo.experimental.models.s2transformer import (  # noqa: E402
    S2PosEmbedConfig,
    S2ProcessorConfig,
    S2ResamplingConfig,
    S2Transformer,
)

DATA_DIR = Path(__file__).parent / "data"

# Frozen configurations for the reference tests (MOD-008b/c). The files in
# ``data/`` were generated from these with ``torch.manual_seed(0)`` on CPU.
REFERENCE_CASES = {
    "processor_qk": dict(
        model_grid_type="equiangular",
        sht_grid_type="equiangular",
        inp_shape=(16, 32),
        out_shape=(16, 32),
        in_channels=8,
        out_channels=8,
        scale_factor=1,
        big_skip=True,
        processor=dict(
            embed_dim=8,
            num_layers=2,
            num_heads=4,
            attention_mode="global",
            attn_optimized_kernel=False,
            qk_norm=True,
            normalization="layer_norm",
        ),
        pos_embed=dict(kind="spectral", application="qk"),
    ),
    "encoder_decoder": dict(
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        inp_grid_type="legendre-gauss",
        out_grid_type="equiangular",
        inp_shape=(16, 32),
        out_shape=(17, 32),
        latent_shape=(8, 16),
        in_channels=4,
        out_channels=2,
        processor=dict(
            embed_dim=8,
            num_layers=2,
            num_heads=2,
            attention_mode="global",
            attn_optimized_kernel=False,
            normalization="layer_norm",
        ),
        pos_embed=dict(kind="spectral", application="input"),
    ),
}


@pytest.fixture(autouse=True)
def full_precision_matmul():
    """Keep reference tolerances independent of the host's TF32 defaults."""
    previous = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(previous)
    torch.backends.cudnn.allow_tf32 = previous_cudnn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _tiny_model(**overrides) -> S2Transformer:
    """A small CPU-friendly model: global attention, no encoder/decoder.

    With ``in_channels == out_channels == embed_dim`` and ``scale_factor=1`` the
    grid/channels don't change between input, latent, and output, so the
    encoder/decoder auto-deactivate (nn.Identity) and the processor runs directly
    on the input. ``attention_mode="global"`` + ``attn_optimized_kernel=False``
    avoids the neighborhood-attention CUDA kernel, so the model runs on CPU.
    """
    kwargs = dict(
        model_grid_type="equiangular",
        sht_grid_type="equiangular",
        inp_shape=(16, 32),
        out_shape=(16, 32),
        in_channels=8,
        out_channels=8,
        scale_factor=1,
        processor=S2ProcessorConfig(
            embed_dim=8,
            num_layers=2,
            num_heads=4,
            attention_mode="global",
            attn_optimized_kernel=False,
            normalization="layer_norm",
        ),
        pos_embed=S2PosEmbedConfig(kind="spectral"),
    )
    kwargs.update(overrides)
    return S2Transformer(**kwargs)


# -----------------------------------------------------------------------------
# Constructor and attributes (MOD-008a)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["default", "custom"])
def test_constructor_attributes(config):
    """Default and custom constructor arguments end up in the public attributes."""
    if config == "default":
        model = S2Transformer()
        expected = dict(
            inp_shape=(721, 1440),
            out_shape=(721, 1440),
            in_channels=27,
            out_channels=1,
            h=90,
            w=180,
            embed_dim=128,
            num_layers=4,
            big_skip=False,
            checkpointing_level=0,
            model_grid_type="equiangular",
            sht_grid_type="legendre-gauss",
            inp_grid_type="equiangular",
            out_grid_type="equiangular",
            pos_embed_application="input",
            use_encoder=True,
            use_decoder=True,
        )
        assert model.resampling.decoder_theta_cutoff_factor is None
    else:
        model = S2Transformer(
            model_grid_type="legendre-gauss",
            sht_grid_type="equiangular",
            inp_shape=(16, 32),
            out_shape=(17, 32),
            in_channels=4,
            out_channels=2,
            latent_shape=(8, 16),
            big_skip=True,
            bias=False,
            checkpointing_level=1,
            processor=S2ProcessorConfig(
                embed_dim=8,
                num_layers=3,
                num_heads=2,
                attention_mode="global",
                attn_optimized_kernel=False,
            ),
            resampling=S2ResamplingConfig(decoder_theta_cutoff_factor=3.0),
            pos_embed=S2PosEmbedConfig(kind="spectral", application="qk"),
            out_grid_type="equiangular",
        )
        expected = dict(
            inp_shape=(16, 32),
            out_shape=(17, 32),
            in_channels=4,
            out_channels=2,
            h=8,
            w=16,
            embed_dim=8,
            num_layers=3,
            big_skip=True,
            checkpointing_level=1,
            model_grid_type="legendre-gauss",
            sht_grid_type="equiangular",
            inp_grid_type="legendre-gauss",
            out_grid_type="equiangular",
            pos_embed_application="qk",
            use_encoder=True,
            use_decoder=True,
        )
        assert model.resampling.decoder_theta_cutoff_factor == 3.0
    for name, value in expected.items():
        assert getattr(model, name) == value, name
    assert len(model.blocks) == expected["num_layers"]
    assert isinstance(model, Module)


# -----------------------------------------------------------------------------
# Forward + checkpoint round-trip
# -----------------------------------------------------------------------------


def test_forward_shape():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(2, 8, 16, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 8, 16, 32)
    assert torch.isfinite(y).all()


def test_encoder_decoder_auto_activation():
    # equal grid + channels -> encoder/decoder collapse to Identity
    m = _tiny_model()
    assert isinstance(m.encoder, torch.nn.Identity)
    assert isinstance(m.decoder, torch.nn.Identity)
    assert m.use_encoder is False and m.use_decoder is False
    # a channel change auto-activates both (construction only; no CPU forward)
    m2 = _tiny_model(in_channels=4, out_channels=1)
    assert m2.use_encoder and m2.use_decoder


class _RecordQKV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = None
        self.k = None
        self.v = None

    def forward(self, query, key=None, value=None):
        self.q = query.detach().clone()
        self.k = key.detach().clone()
        self.v = value.detach().clone()
        return torch.zeros_like(value)


def test_qk_positional_embedding_is_applied_in_every_attention_only():
    model = _tiny_model(
        pos_embed=S2PosEmbedConfig(kind="spectral", application="qk")
    ).eval()
    recorders = []
    for block in model.blocks:
        recorder = _RecordQKV()
        block.self_attn = recorder
        recorders.append(recorder)

    x = torch.randn(1, 8, 16, 32)
    with torch.no_grad():
        model(x)

    expected_pe = model.pos_embed.position_embeddings
    for recorder in recorders:
        assert recorder.q is not None
        assert torch.equal(recorder.q, recorder.k)
        assert torch.allclose(recorder.q - recorder.v, expected_pe)


def test_grid_type_defaults_preserve_legacy_behavior():
    torch.manual_seed(0)
    legacy = _tiny_model().eval()
    torch.manual_seed(0)
    explicit = _tiny_model(
        inp_grid_type="equiangular",
        out_grid_type="equiangular",
    ).eval()

    assert legacy.inp_grid_type == legacy.model_grid_type == "equiangular"
    assert legacy.out_grid_type == legacy.model_grid_type == "equiangular"
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    for legacy_value, explicit_value in zip(
        legacy.state_dict().values(), explicit.state_dict().values()
    ):
        assert torch.equal(legacy_value, explicit_value)

    x = torch.randn(1, 8, 16, 32)
    with torch.no_grad():
        assert torch.equal(legacy(x), explicit(x))


def test_legendre_gauss_input_to_equiangular_output():
    """Small analogue of a 360x720 LG input decoded to 361x720 equiangular."""
    model = _tiny_model(
        inp_grid_type="legendre-gauss",
        out_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        inp_shape=(8, 16),
        out_shape=(9, 16),
        latent_shape=(4, 8),
        in_channels=4,
        out_channels=2,
        processor=S2ProcessorConfig(
            embed_dim=4,
            num_layers=1,
            num_heads=1,
            attention_mode="global",
            attn_optimized_kernel=False,
            normalization="layer_norm",
        ),
        pos_embed=S2PosEmbedConfig(kind="none"),
    ).eval()

    x = torch.randn(1, 4, 8, 16)
    with torch.no_grad():
        y = model(x)

    assert model.inp_grid_type == "legendre-gauss"
    assert model.out_grid_type == "equiangular"
    assert y.shape == (1, 2, 9, 16)
    assert torch.isfinite(y).all()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(1, 8, 16, 32)
    with torch.no_grad():
        y_ref = model(x)

    ckpt = tmp_path / "s2transformer.mdlus"
    model.save(str(ckpt))
    reloaded = S2Transformer.from_checkpoint(str(ckpt)).eval()
    with torch.no_grad():
        y_new = reloaded(x)

    assert torch.allclose(y_ref, y_new, atol=1e-6, rtol=0)


def test_asymmetric_grid_types_checkpoint_roundtrip(tmp_path):
    model = _tiny_model(
        inp_grid_type="legendre-gauss",
        out_grid_type="equiangular",
    )
    assert model._args["__args__"]["inp_grid_type"] == "legendre-gauss"
    assert model._args["__args__"]["out_grid_type"] == "equiangular"

    ckpt = tmp_path / "s2transformer_asymmetric.mdlus"
    model.save(str(ckpt))
    reloaded = S2Transformer.from_checkpoint(str(ckpt))

    assert reloaded.inp_grid_type == "legendre-gauss"
    assert reloaded.out_grid_type == "equiangular"
    assert reloaded.state_dict().keys() == model.state_dict().keys()
    for original, restored in zip(
        model.state_dict().values(), reloaded.state_dict().values()
    ):
        assert torch.equal(original, restored)


def test_qk_positional_embedding_checkpoint_roundtrip(tmp_path):
    model = _tiny_model(pos_embed=S2PosEmbedConfig(kind="spectral", application="qk"))
    assert model._args["__args__"]["pos_embed"]["application"] == "qk"

    ckpt = tmp_path / "s2transformer_qk_pos_embed.mdlus"
    model.save(str(ckpt))
    reloaded = S2Transformer.from_checkpoint(str(ckpt))

    assert reloaded.pos_embed_application == "qk"
    assert hasattr(reloaded.pos_embed, "position_embeddings")


# -----------------------------------------------------------------------------
# Input validation and checkpoint compatibility
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(2, 7, 16, 32), (2, 8, 15, 32), (8, 16, 32)],
    ids=["channels", "grid", "ndim"],
)
def test_forward_rejects_wrong_input_shape(shape):
    model = _tiny_model()
    with pytest.raises(ValueError, match="Expected tensor x of shape"):
        model(torch.randn(*shape))


def _write_legacy_checkpoint(model: S2Transformer, path: Path, noise: dict) -> None:
    """Save ``model`` as a 0.1.0 checkpoint that still carries a ``noise`` argument."""
    current = path.with_name("current.mdlus")
    model.save(str(current))
    with zipfile.ZipFile(current) as src, zipfile.ZipFile(path, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item == "args.json":
                args = json.loads(data)
                args["__args__"]["noise"] = noise
                data = json.dumps(args).encode()
            elif item == "metadata.json":
                metadata = json.loads(data)
                metadata["mdlus_file_version"] = "0.1.0"
                data = json.dumps(metadata).encode()
            dst.writestr(item, data)


def test_legacy_checkpoint_without_noise_conditioning_loads(tmp_path):
    model = _tiny_model().eval()
    path = tmp_path / "legacy.mdlus"
    _write_legacy_checkpoint(model, path, {"mode": None, "latent_dim": 64, "hidden": 0})
    with pytest.warns(UserWarning, match="0.1.0"):
        restored = Module.from_checkpoint(str(path)).eval()
    x = torch.randn(2, 8, 16, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_legacy_checkpoint_with_noise_conditioning_is_rejected(tmp_path):
    path = tmp_path / "legacy_film.mdlus"
    _write_legacy_checkpoint(_tiny_model(), path, {"mode": "film", "latent_dim": 8})
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="noise.mode"):
        Module.from_checkpoint(str(path))


# -----------------------------------------------------------------------------
# Reference outputs (MOD-008b) and checkpoint loading (MOD-008c)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["constructor", "checkpoint"])
@pytest.mark.parametrize("case", sorted(REFERENCE_CASES))
def test_reference_outputs(case, source, device):
    """Frozen weights reproduce the stored outputs, built either way."""
    data = torch.load(DATA_DIR / f"{case}.pth", map_location=device, weights_only=True)
    checkpoint = DATA_DIR / f"{case}.mdlus"
    if source == "checkpoint":
        model = Module.from_checkpoint(str(checkpoint))
    else:
        model = S2Transformer(**REFERENCE_CASES[case])
        model.load(str(checkpoint))
    model = model.to(device).eval()

    config = REFERENCE_CASES[case]
    assert model.inp_shape == config["inp_shape"]
    assert model.out_shape == config["out_shape"]
    assert model.in_channels == config["in_channels"]
    assert model.out_channels == config["out_channels"]
    assert model.embed_dim == config["processor"]["embed_dim"]
    assert model.pos_embed_application == config["pos_embed"]["application"]

    with torch.no_grad():
        out = model(data["x"])
    torch.testing.assert_close(out, data["out"], rtol=1e-4, atol=1e-5)
