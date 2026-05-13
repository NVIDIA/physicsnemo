r"""
examples/structural_mechanics/ensemble_uq/ensemble_uq_beam.py

Ensemble uncertainty quantification for a 1D beam deflection surrogate
=======================================================================

Demonstrates ``EnsembleWrapper`` on a structural mechanics problem:
predicting the deflection of a simply-supported Euler-Bernoulli beam
under a distributed load, as a function of load magnitude and beam
stiffness (EI).

This example deliberately keeps the training data small so that the
ensemble spread is visible — illustrating that ``std`` grows in regions
where the surrogate is uncertain (e.g. near the edges of the training
distribution).

Teacher
-------
Analytical solution for simply-supported beam deflection under uniform load:

.. math::

    w(x) = \\frac{q}{24 EI} \\left( x^4 - 2Lx^3 + L^3 x \\right)

where :math:`L = 1\\,\\mathrm{m}`, :math:`x \\in [0, L]`.

The surrogate maps :math:`(q, EI)` to the maximum deflection
:math:`w_{\\max} = w(L/2)`.

Ensemble
--------
5 ``FullyConnected`` members, each trained from a different random seed
using a standard PyTorch loop. Wrapped with ``EnsembleWrapper`` for
uncertainty-aware inference.

Dependencies
------------
See ``requirements.txt`` in this directory::

    pip install -r examples/structural_mechanics/ensemble_uq/requirements.txt

Run
---
From the repo root::

    python examples/structural_mechanics/ensemble_uq/ensemble_uq_beam.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from physicsnemo.nn import FullyConnected
from physicsnemo.experimental.models.ensemble_wrapper import EnsembleWrapper

# ---------------------------------------------------------------------------
# Teacher: analytical beam deflection
# ---------------------------------------------------------------------------

L = 1.0   # beam length [m]


def beam_max_deflection(q: float, EI: float) -> float:
    r"""
    Maximum mid-span deflection of a simply-supported beam.

    .. math::

        w_{\max} = \frac{5 q L^4}{384 EI}

    Parameters
    ----------
    q : float
        Uniform distributed load :math:`[\mathrm{N/m}]`.
    EI : float
        Flexural rigidity :math:`[\mathrm{N \cdot m^2}]`.

    Returns
    -------
    float
        Maximum deflection :math:`[\mathrm{m}]`.
    """
    return (5 * q * L**4) / (384 * EI)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

N_TRAIN = 80
N_TEST  = 400

Q_BOUNDS  = (100.0, 5000.0)   # N/m
EI_BOUNDS = (1e4,   1e6)      # N·m²

rng = np.random.default_rng(0)


def make_dataset(n: int, seed: int) -> tuple:
    rng_ = np.random.default_rng(seed)
    q  = rng_.uniform(*Q_BOUNDS,  n)
    EI = rng_.uniform(*EI_BOUNDS, n)
    X  = np.stack([q, EI], axis=1).astype(np.float32)
    y  = np.array([beam_max_deflection(q[i], EI[i]) for i in range(n)],
                  dtype=np.float32)[:, None]
    return X, y


X_train_np, y_train_np = make_dataset(N_TRAIN, seed=1)
X_test_np,  y_test_np  = make_dataset(N_TEST,  seed=2)

# Z-score normalisation
X_mean, X_std = X_train_np.mean(0), X_train_np.std(0) + 1e-8
y_mean, y_std = y_train_np.mean(),  y_train_np.std()  + 1e-8

X_train = torch.tensor((X_train_np - X_mean) / X_std)
y_train = torch.tensor((y_train_np - y_mean) / y_std)
X_test  = torch.tensor((X_test_np  - X_mean) / X_std)
y_test  = torch.tensor((y_test_np  - y_mean) / y_std)


# ---------------------------------------------------------------------------
# Train N ensemble members
# ---------------------------------------------------------------------------

N_MEMBERS = 5
EPOCHS    = 500
LR        = 5e-3


def train_member(seed: int) -> FullyConnected:
    r"""Train one ``FullyConnected`` member from a given random seed."""
    torch.manual_seed(seed)
    model = FullyConnected(in_features=2, out_features=1, num_layers=3, layer_size=32)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        sched.step()

    model.eval()
    return model


print(f"Training {N_MEMBERS} ensemble members...")
members = [train_member(seed=i) for i in range(N_MEMBERS)]
print("  Done.")

# ---------------------------------------------------------------------------
# Wrap with EnsembleWrapper
# ---------------------------------------------------------------------------

ensemble = EnsembleWrapper(members)
ensemble.eval()

with torch.no_grad():
    result = ensemble.predict_with_uncertainty(X_test)

# Denormalise
mean_np = (result.mean.numpy() * y_std + y_mean).squeeze()
std_np  = (result.std.numpy()  * y_std).squeeze()          # std in physical units
true_np = y_test_np.squeeze()

rel_err = float(np.mean(np.abs(mean_np - true_np) / (np.abs(true_np) + 1e-10)))
print(f"\nEnsemble mean relative error : {rel_err:.2%}")
print(f"Mean epistemic std            : {std_np.mean():.4e} m")

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

# Sort by true deflection for clean 1-D plots
order = np.argsort(true_np)
t = true_np[order]
m = mean_np[order]
s = std_np[order]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Beam Deflection Surrogate — Ensemble UQ", fontsize=13)

# ── Panel 1: parity plot ──────────────────────────────────────────────────
ax = axes[0]
ax.scatter(t, m, s=10, alpha=0.3, label="Test samples")
lim = [t.min() * 0.95, t.max() * 1.05]
ax.plot(lim, lim, "r--", lw=1, label="Ideal")
ax.set_xlabel("Analytical deflection [m]")
ax.set_ylabel("Ensemble mean [m]")
ax.set_title(f"Parity Plot  (err = {rel_err:.1%})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 2: uncertainty ribbon ───────────────────────────────────────────
ax = axes[1]
ax.plot(t, m, lw=1, label="Ensemble mean")
ax.fill_between(t, m - 2 * s, m + 2 * s, alpha=0.3, label="±2σ (epistemic)")
ax.plot(t, t,  "r--", lw=1, label="Ground truth")
ax.set_xlabel("True deflection (sorted) [m]")
ax.set_ylabel("Predicted deflection [m]")
ax.set_title("Uncertainty Ribbon (±2σ)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 3: std vs prediction magnitude ─────────────────────────────────
ax = axes[2]
ax.scatter(t, s, s=10, alpha=0.3, color="steelblue")
ax.set_xlabel("True deflection [m]")
ax.set_ylabel("Epistemic std [m]")
ax.set_title("Uncertainty vs Prediction Magnitude")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path("ensemble_uq_beam.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path.resolve()}")
plt.show()
