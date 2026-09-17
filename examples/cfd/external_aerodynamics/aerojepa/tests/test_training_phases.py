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

"""Training-objective tests: latent-loss gradient flow, phases, context SIGReg."""

import torch
from omegaconf import OmegaConf
from physicsnemo.experimental.models.aerojepa import (
    AeroJEPA,
    AeroJEPATrunk,
    ContextTransformer,
    PrototypeTokenJEPAHead,
    QueryTokenDecoder,
    TargetTransformer,
)

from src.losses import build_recon_loss_from_config, build_sigreg_from_config
from train import (
    _apply_phase,
    _compute_term_weights,
    _compute_total_loss,
    _forward_sample,
    _resolve_phase,
    _run_epoch,
)

_ENC = dict(
    point_input_dim=3,
    token_dim=16,
    max_point_tokens=8,
    tokenizer_strategy="fps",
    tokenizer_knn_chunk_size=16,
    point_pos_pe_bands=2,
    num_heads=2,
    num_layers=1,
    neighbor_k=3,
    mlp_ratio=2,
    dropout=0.0,
)


def _build_model() -> AeroJEPA:
    trunk = AeroJEPATrunk(
        context_encoder=ContextTransformer(**_ENC),
        target_encoder=TargetTransformer(**_ENC),
        decoder=QueryTokenDecoder(
            token_dim=16,
            hidden_dim=32,
            num_layers=1,
            out_dim=4,
            use_sdf=True,
            cond_dim=4,
            pe_num_bands=2,
            cross_attention_heads=2,
            cross_attention_layers=1,
            cross_attention_k=3,
            query_chunk_size=64,
        ),
        include_geometry_global_in_decoder_cond=False,
    )
    predictor = PrototypeTokenJEPAHead(
        token_dim=16,
        cond_dim=4,
        depth=1,
        num_heads=2,
        neighbor_k=3,
        query_pe_bands=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    return AeroJEPA(trunk=trunk, predictor=predictor)


def _sample(device):
    return {
        "context_pos": torch.randn(30, 3, device=device),
        "context_feat": torch.zeros(30, 0, device=device),
        "target_surface_pos": torch.randn(24, 3, device=device),
        "target_surface_main_feat": torch.randn(24, 3, device=device),
        "target_volume_pos": torch.randn(20, 3, device=device),
        "target_volume_feat": torch.randn(20, 3, device=device),
        "gen_params": torch.randn(4, device=device),
        "query_pos": torch.randn(16, 3, device=device),
        "query_sdf": torch.randn(16, 1, device=device),
        "query_target": torch.randn(16, 4, device=device),
    }


def _loss_cfg(*, sigreg_context_weight: float = 0.0):
    return OmegaConf.create(
        {
            "recon": {"kind": "mse", "weight": 1.0, "warmup_epochs": 0},
            "latent": {
                "weight": 1.0,
                "warmup_epochs": 0,
                "mse_weight": 0.5,
                "cosine_weight": 0.5,
            },
            "sigreg": {"weight": 0.1, "warmup_epochs": 0},
            "sigreg_context": {"weight": sigreg_context_weight, "warmup_epochs": 0},
        }
    )


def _losses(device, *, sigreg_context_weight: float = 0.0):
    recon = build_recon_loss_from_config({"kind": "mse"}).to(device)
    sigreg = build_sigreg_from_config({"knots": 9, "num_proj": 16}).to(device)
    sigreg_ctx = build_sigreg_from_config({"knots": 9, "num_proj": 16}).to(device)
    return recon, sigreg, sigreg_ctx


def _grad_sum(module) -> float:
    return float(
        sum(
            p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None
        )
    )


def _has_no_grad(module) -> bool:
    return all(p.grad is None for p in module.parameters())


def _step(model, device, *, phase=None, sigreg_context_weight=0.0):
    recon_fn, sigreg_fn, sigreg_ctx_fn = _losses(device)
    loss_cfg = _loss_cfg(sigreg_context_weight=sigreg_context_weight)
    model.train()
    _apply_phase(model, phase, is_train=True)
    term_weights = _compute_term_weights(1, loss_cfg, phase)
    run_reconstruction = float(term_weights["recon"]) != 0.0
    pred_field, pred_features, context_tokens, target_tokens, _, _ = _forward_sample(
        model, _sample(device), run_reconstruction=run_reconstruction
    )
    loss, _ = _compute_total_loss(
        pred_field=pred_field,
        query_target=_sample(device)["query_target"],
        pred_features=pred_features,
        context_tokens=context_tokens,
        target_tokens=target_tokens,
        recon_loss_fn=recon_fn,
        sigreg_loss_fn=sigreg_fn,
        sigreg_context_loss_fn=sigreg_ctx_fn,
        loss_cfg=loss_cfg,
        term_weights=term_weights,
    )
    loss.backward()
    return loss


def test_latent_loss_trains_target_encoder(device):
    """The latent loss must give the (grad-bearing) target encoder a gradient."""
    torch.manual_seed(0)
    model = _build_model().to(device)
    _step(model, device)
    # The core fix: without detaching the latent target, the target encoder
    # receives gradient (previously it was trained only by SIGReg).
    assert _grad_sum(model.target_encoder) > 0.0
    assert _grad_sum(model.context_encoder) > 0.0
    assert _grad_sum(model.predictor) > 0.0
    assert _grad_sum(model.decoder) > 0.0


def test_phase1_trains_encoders_and_predictor_not_decoder(device):
    """Phase 1 trains the encoders + predictor and leaves the decoder frozen."""
    torch.manual_seed(0)
    model = _build_model().to(device)
    phase = {
        "name": "phase1_latent",
        "epoch_in_phase": 1,
        "config": OmegaConf.create({}),
    }
    _step(model, device, phase=phase)
    assert _grad_sum(model.context_encoder) > 0.0
    assert _grad_sum(model.target_encoder) > 0.0
    assert _grad_sum(model.predictor) > 0.0
    assert _has_no_grad(model.decoder)  # frozen


def test_phase2_trains_only_decoder(device):
    """Phase 2 freezes the encoders + predictor and trains only the decoder."""
    torch.manual_seed(0)
    model = _build_model().to(device)
    phase = {
        "name": "phase2_reconstruction",
        "epoch_in_phase": 1,
        "config": OmegaConf.create({}),
    }
    _step(model, device, phase=phase)
    assert _grad_sum(model.decoder) > 0.0
    assert _has_no_grad(model.context_encoder)
    assert _has_no_grad(model.target_encoder)
    assert _has_no_grad(model.predictor)


def test_compute_term_weights_phase_gating():
    """Phase weights: P1 = latent+sigreg on, recon off; P2 = recon only."""
    loss_cfg = _loss_cfg()
    none_w = _compute_term_weights(1, loss_cfg, None)
    assert none_w["latent"] > 0 and none_w["sigreg"] > 0 and none_w["recon"] > 0

    p1 = {"name": "phase1_latent", "epoch_in_phase": 1, "config": OmegaConf.create({})}
    w1 = _compute_term_weights(1, loss_cfg, p1)
    assert w1["latent"] > 0 and w1["sigreg"] > 0
    assert w1["recon"] == 0.0

    p2 = {
        "name": "phase2_reconstruction",
        "epoch_in_phase": 1,
        "config": OmegaConf.create({}),
    }
    w2 = _compute_term_weights(1, loss_cfg, p2)
    assert w2["recon"] > 0
    assert w2["latent"] == 0.0 and w2["sigreg"] == 0.0 and w2["sigreg_context"] == 0.0


def test_resolve_phase_boundary():
    """Two-phase resolution splits on phase1_epochs (1-indexed)."""
    cfg = OmegaConf.create({"enabled": True, "phase1_epochs": 3})
    assert _resolve_phase(1, cfg)["name"] == "phase1_latent"
    assert _resolve_phase(3, cfg)["name"] == "phase1_latent"
    assert _resolve_phase(4, cfg)["name"] == "phase2_reconstruction"
    assert _resolve_phase(4, cfg)["epoch_in_phase"] == 1
    # Disabled or absent -> single phase (None).
    assert _resolve_phase(1, OmegaConf.create({"enabled": False})) is None
    assert _resolve_phase(1, None) is None


def test_run_epoch_train_step_updates_params(device):
    """One `_run_epoch` train step runs end-to-end (with a GradScaler) and steps."""
    torch.manual_seed(0)
    model = _build_model().to(device)
    recon_fn, sigreg_fn, sigreg_ctx_fn = _losses(device)
    # Build a one-sample batch (leading batch axis) for the loader.
    batch = {k: v.unsqueeze(0) for k, v in _sample(device).items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # fp32 -> no-op
    before = [p.detach().clone() for p in model.parameters()]
    metrics = _run_epoch(
        model=model,
        loader=[batch],
        recon_loss_fn=recon_fn,
        sigreg_loss_fn=sigreg_fn,
        sigreg_context_loss_fn=sigreg_ctx_fn,
        optimizer=optimizer,
        lr_scheduler=None,
        ema=None,
        device=torch.device(device),
        precision="fp32",
        grad_clip_norm=1.0,
        loss_cfg=_loss_cfg(),
        epoch=0,
        max_batches=None,
        phase=None,
        scaler=scaler,
    )
    assert all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values())
    after = list(model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_context_sigreg_default_off_but_wirable(device):
    """Context SIGReg is off by default (weight 0) and adds grad when enabled."""
    loss_cfg = _loss_cfg()
    assert _compute_term_weights(1, loss_cfg, None)["sigreg_context"] == 0.0

    # With only the context-SIGReg term active, the context encoder still gets
    # a gradient -- confirming the term is wired to the context latents.
    torch.manual_seed(0)
    model = _build_model().to(device)
    recon_fn, sigreg_fn, sigreg_ctx_fn = _losses(device)
    model.train()
    term_weights = {
        "latent": 0.0,
        "sigreg": 0.0,
        "sigreg_context": 1.0,
        "recon": 0.0,
    }
    _, pred_features, context_tokens, target_tokens, _, _ = _forward_sample(
        model, _sample(device), run_reconstruction=False
    )
    loss, parts = _compute_total_loss(
        pred_field=None,
        query_target=_sample(device)["query_target"],
        pred_features=pred_features,
        context_tokens=context_tokens,
        target_tokens=target_tokens,
        recon_loss_fn=recon_fn,
        sigreg_loss_fn=sigreg_fn,
        sigreg_context_loss_fn=sigreg_ctx_fn,
        loss_cfg=loss_cfg,
        term_weights=term_weights,
    )
    loss.backward()
    assert _grad_sum(model.context_encoder) > 0.0
    assert "sigreg_context" in parts
