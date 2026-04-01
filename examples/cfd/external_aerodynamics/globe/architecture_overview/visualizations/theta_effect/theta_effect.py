"""Visualize the effect of the Barnes-Hut theta parameter on GLOBE's field kernel.

Produces six figures:
1a. ClusterTree spatial hierarchy (2D nested AABBs)
1b. ClusterTree spatial hierarchy (3D stacked view)
2.  Near/far source classification at different theta values
3.  Kernel scalar field comparison (exact vs approximate)
4.  Approximation error fields
5.  Error convergence and computational cost vs theta
"""

from pathlib import Path

import aerosandbox.tools.pretty_plots as p
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tensordict import TensorDict

from physicsnemo.experimental.models.globe import BarnesHutKernel, Kernel
from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent

### [Configuration]
N_SOURCE_POINTS = 20
SEED = 39
THETA_VALUES = [0.1, 0.25, 0.5, 1.0]
THETA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
GRID_RES = 128
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = -6.0, 6.0
# Target point for the near/far classification visualization (above the source curve)
TARGET_VIZ_POINT = torch.tensor([[0.3, 0.8]], device=device)


### [Source points: curve with normals, matching kernel_visualizations.py figure 5]
t = torch.linspace(0, 1, N_SOURCE_POINTS, device=device)
source_points = torch.stack(
    [2 * (t - 0.5), torch.sin((t - 0.5) * 5) / 5 - t / 3], dim=1
)
source_strengths = (
    torch.ones(N_SOURCE_POINTS, device=device) * 1e2 / N_SOURCE_POINTS
)
source_data = TensorDict(
    {
        "normal": torch.stack(
            [torch.sin((t - 0.5 + 1) / 2), torch.cos((t - 0.5 + 1) / 2)], dim=1
        ),
        "other": torch.zeros_like(source_points),
    },
    batch_size=torch.Size([N_SOURCE_POINTS]),
    device=device,
)
src_np = source_points.cpu().numpy()


### [Kernels: create exact and BH kernels with identical weights]
torch.manual_seed(SEED)
np.random.seed(SEED)
kernel_kwargs = dict(
    n_spatial_dims=2,
    output_field_ranks={"phi": 0, "u": 1},
    source_data_ranks={"normal": 1, "other": 1},
    hidden_layer_sizes=[64, 64],
    n_spherical_harmonics=4,
    network_type="pade",
    spectral_norm=False,
    use_gradient_checkpointing=False,
)
bh_kernel = BarnesHutKernel(**kernel_kwargs, leaf_size=1).to(device)
exact_kernel = Kernel(**kernel_kwargs).to(device)
exact_kernel.load_state_dict(bh_kernel.state_dict(), strict=False)
bh_kernel.eval()
exact_kernel.eval()


### [Evaluation grid and prebuilt trees]
x_1d = np.linspace(X_MIN, X_MAX, GRID_RES)
y_1d = np.linspace(Y_MIN, Y_MAX, GRID_RES)
X_grid, Y_grid = np.meshgrid(x_1d, y_1d, indexing="xy")
grid_targets = torch.stack(
    [
        torch.as_tensor(X_grid, device=device, dtype=torch.float32).flatten(),
        torch.as_tensor(Y_grid, device=device, dtype=torch.float32).flatten(),
    ],
    dim=1,
)

source_tree = ClusterTree.from_points(source_points, leaf_size=1)
grid_target_tree = ClusterTree.from_points(grid_targets, leaf_size=1)


# =====================================================================
# Helpers
# =====================================================================


def evaluate_on_grid(kernel_obj, **extra_kwargs) -> dict[str, np.ndarray]:
    """Evaluate a kernel on the 2D meshgrid, returning reshaped numpy arrays."""
    with torch.no_grad():
        result = kernel_obj(
            reference_length=torch.tensor(1.0, device=device),
            source_points=source_points,
            target_points=grid_targets,
            source_strengths=source_strengths,
            source_data=source_data,
            **extra_kwargs,
        )
    return {
        k: v.detach()
        .reshape([GRID_RES, GRID_RES, *v.shape[1:]])
        .cpu()
        .numpy()
        .astype(np.float64)
        for k, v in result.items()
    }


def save_figure(fig: plt.Figure, *, stem: str) -> None:
    """Save figure as both PDF and PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"Saved {path}")


def walk_tree(tree: ClusterTree) -> list[dict]:
    """Depth-first traversal collecting per-node AABB, depth, and leaf status."""
    nodes = []
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        is_leaf = tree.leaf_count[node_id].item() > 0
        nodes.append(
            {
                "id": node_id,
                "depth": depth,
                "aabb_min": tree.node_aabb_min[node_id].cpu().numpy(),
                "aabb_max": tree.node_aabb_max[node_id].cpu().numpy(),
                "is_leaf": is_leaf,
            }
        )
        if not is_leaf:
            # Push right then left so left is popped first (DFS left-to-right)
            for child_id in (
                tree.node_right_child[node_id].item(),
                tree.node_left_child[node_id].item(),
            ):
                if child_id >= 0:
                    stack.append((child_id, depth + 1))
    return nodes


def classify_sources_for_target(
    tree: ClusterTree,
    target_pt: torch.Tensor,
    theta: float,
) -> tuple[set[int], set[int], set[int]]:
    """Classify each source as near (exact) or far (approximate) for one target.

    Returns (near_source_ids, far_source_ids, far_node_ids).

    With leaf_size=1 every single-point leaf that enters the monopole
    code path is technically exact (centroid = point).  We still label
    it "far" here because the *algorithm's decision* was to approximate,
    and the visualization aims to show that decision boundary.
    """
    tgt_tree = ClusterTree.from_points(target_pt, leaf_size=1)
    plan = tree.find_dual_interaction_pairs(target_tree=tgt_tree, theta=theta)

    # Exact: (near,near) individual pairs + (far,near) where the source
    # is evaluated at its true position.
    near: set[int] = set(plan.near_source_ids.cpu().tolist())
    near.update(plan.fn_source_ids.cpu().tolist())

    # Approximate: (near,far) and (far,far) - source represented by its
    # node centroid.  Expand node IDs to individual source indices.
    far: set[int] = set()
    far_nodes: set[int] = set()
    for nids in (plan.nf_source_node_ids, plan.far_source_node_ids):
        for nid in nids.cpu().tolist():
            far_nodes.add(nid)
            start = tree.node_range_start[nid].item()
            count = tree.node_range_count[nid].item()
            far.update(
                tree.sorted_source_order[start : start + count].cpu().tolist()
            )

    return near, far, far_nodes


# =====================================================================
# Figure 1a: ClusterTree AABB hierarchy (2D)
# =====================================================================

tree_nodes = walk_tree(source_tree)
max_depth = max(n["depth"] for n in tree_nodes)
cmap_tree = plt.cm.viridis

fig, ax = plt.subplots(figsize=(8, 6))
# Draw from shallowest to deepest so deeper (smaller) nodes paint on top.
for node in sorted(tree_nodes, key=lambda n: n["depth"]):
    frac = node["depth"] / max(max_depth, 1)
    color = cmap_tree(frac)
    xy = node["aabb_min"]
    size = node["aabb_max"] - node["aabb_min"]
    rect = mpatches.FancyBboxPatch(
        xy,
        size[0],
        size[1],
        boxstyle="round,pad=0.02",
        linewidth=max(2.5 - 0.3 * node["depth"], 0.5),
        edgecolor=color,
        facecolor=(*color[:3], 0.05),
    )
    ax.add_patch(rect)

ax.scatter(*src_np.T, color="black", s=25, zorder=5)
ax.set_xlabel(r"$x/\ell$")
ax.set_ylabel(r"$y/\ell$")
ax.set_title("ClusterTree AABB hierarchy")
ax.set_aspect("equal")
ax.autoscale()
sm = plt.cm.ScalarMappable(cmap=cmap_tree, norm=plt.Normalize(0, max_depth))
plt.colorbar(sm, ax=ax, label="Tree depth")
p.show_plot(show=False)
save_figure(fig, stem="tree_hierarchy_2d")
plt.show()


# =====================================================================
# Figure 1b: ClusterTree AABB hierarchy (3D stacked)
# =====================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
Z_SPACING = 1.5

for node in sorted(tree_nodes, key=lambda n: n["depth"]):
    frac = node["depth"] / max(max_depth, 1)
    color = cmap_tree(frac)
    # Root at top, leaves at z=0 (same level as source points)
    z = (max_depth - node["depth"]) * Z_SPACING
    x0, y0 = node["aabb_min"]
    x1, y1 = node["aabb_max"]
    verts = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
    ax.add_collection3d(
        Poly3DCollection(
            [verts],
            alpha=0.12,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
        )
    )

ax.scatter(
    *src_np.T, zs=0, zdir="z", color="black", s=20, depthshade=False, zorder=5
)
ax.set_xlabel(r"$x/\ell$")
ax.set_ylabel(r"$y/\ell$")
ax.set_zlabel("Tree level")
ax.set_title("ClusterTree AABB hierarchy (3D)")
ax.set_zticks([i * Z_SPACING for i in range(max_depth + 1)])
ax.set_zticklabels([str(max_depth - i) for i in range(max_depth + 1)])
p.show_plot(show=False)
save_figure(fig, stem="tree_hierarchy_3d")
plt.show()


# =====================================================================
# Figure 2: Near/far source classification at different theta
# =====================================================================

tgt_np = TARGET_VIZ_POINT.cpu().numpy().squeeze()
fig, axes = plt.subplots(
    1, len(THETA_VALUES), figsize=(4 * len(THETA_VALUES), 4.5)
)

for ax, theta in zip(axes, THETA_VALUES):
    near_ids, far_ids, far_node_ids = classify_sources_for_target(
        source_tree, TARGET_VIZ_POINT, theta
    )

    ### Draw AABBs of far-field source nodes
    for nid in far_node_ids:
        bb_min = source_tree.node_aabb_min[nid].cpu().numpy()
        bb_max = source_tree.node_aabb_max[nid].cpu().numpy()
        s = bb_max - bb_min
        ax.add_patch(
            mpatches.Rectangle(
                bb_min,
                s[0],
                s[1],
                linewidth=0.8,
                edgecolor="steelblue",
                facecolor="steelblue",
                alpha=0.15,
            )
        )

    ### Source points colored by classification
    for idx in range(N_SOURCE_POINTS):
        color = "tab:red" if idx in near_ids else "tab:blue"
        ax.scatter(
            src_np[idx, 0],
            src_np[idx, 1],
            color=color,
            s=45,
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )

    ### Target point
    ax.scatter(
        tgt_np[0],
        tgt_np[1],
        color="gold",
        marker="*",
        s=250,
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )

    ax.set_title(
        rf"$\theta = {theta}$" + f"\n{len(near_ids)} near, {len(far_ids)} far"
    )
    ax.set_xlim(src_np[:, 0].min() - 0.5, src_np[:, 0].max() + 0.5)
    ax.set_ylim(src_np[:, 1].min() - 0.8, tgt_np[1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\ell$")
    if ax is axes[0]:
        ax.set_ylabel(r"$y/\ell$")

legend_handles = [
    mpatches.Patch(color="tab:red", label="Near (exact)"),
    mpatches.Patch(color="tab:blue", label="Far (approximate)"),
    plt.Line2D(
        [0],
        [0],
        marker="*",
        color="gold",
        markersize=14,
        markeredgecolor="black",
        linestyle="None",
        label="Target",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.05),
)
fig.suptitle(
    r"Near/far source classification vs. Barnes-Hut $\theta$", y=1.02
)
plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="near_far_classification")
plt.show()


# =====================================================================
# Precompute kernel results for Figures 3 and 4
# =====================================================================

print("Evaluating exact kernel on grid...")
exact_result = evaluate_on_grid(exact_kernel)

bh_results: dict[float, dict[str, np.ndarray]] = {}
for theta in THETA_VALUES:
    print(f"Evaluating BH kernel on grid (theta={theta})...")
    bh_results[theta] = evaluate_on_grid(
        bh_kernel,
        theta=theta,
        cluster_tree=source_tree,
        target_tree=grid_target_tree,
    )


# =====================================================================
# Figure 3: Kernel scalar field comparison
# =====================================================================

labels = ["Exact"] + [rf"$\theta = {th}$" for th in THETA_VALUES]
phi_fields = [exact_result["phi"]] + [
    bh_results[th]["phi"] for th in THETA_VALUES
]
n_cols = len(labels)

phi_scale = float(
    np.max(np.abs(np.percentile(exact_result["phi"], [0.1, 99.9])))
)
levels = np.linspace(-phi_scale, phi_scale, 31)

fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
for col, (ax, label, phi) in enumerate(zip(axes, labels, phi_fields)):
    plt.sca(ax)
    p.contour(
        X_grid,
        Y_grid,
        phi,
        levels=levels,
        extend="both",
        cmap="RdBu_r",
        colorbar=False,
        linelabels=False,
    )
    plt.clim(-phi_scale, phi_scale)
    ax.scatter(*src_np.T, color="black", s=5, zorder=5)
    ax.set_title(label)
    ax.set_aspect("equal")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"$x/\ell$")
    if col == 0:
        ax.set_ylabel(r"$y/\ell$")
    else:
        ax.set_yticklabels([])

fig.suptitle(r"Scalar field $\phi$ at different Barnes-Hut $\theta$", y=1.02)
plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="kernel_comparison")
plt.show()


# =====================================================================
# Figure 4: Approximation error fields
# =====================================================================

all_errors = [
    np.abs(bh_results[th]["phi"] - exact_result["phi"]) for th in THETA_VALUES
]
# Consistent log-scale normalization across panels
positive_vals = np.concatenate([e[e > 0].ravel() for e in all_errors])
vmin = (
    max(float(positive_vals.min()), 1e-8) if len(positive_vals) > 0 else 1e-8
)
vmax = float(max(e.max() for e in all_errors))

fig, axes = plt.subplots(
    1, len(THETA_VALUES), figsize=(4.5 * len(THETA_VALUES), 4)
)
for ax, theta, err in zip(axes, THETA_VALUES, all_errors):
    im = ax.pcolormesh(
        X_grid,
        Y_grid,
        np.clip(err, vmin, None),
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="auto",
    )
    ax.scatter(*src_np.T, color="white", s=5, zorder=5)
    ax.set_title(rf"$\theta = {theta}$, max = {err.max():.2e}")
    ax.set_aspect("equal")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"$x/\ell$")
    if ax is axes[0]:
        ax.set_ylabel(r"$y/\ell$")
    else:
        ax.set_yticklabels([])

fig.colorbar(
    im,
    ax=axes.tolist(),
    label=r"$|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$",
    shrink=0.8,
)
fig.suptitle(
    r"Approximation error $|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$", y=1.02
)
plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="error_fields")
plt.show()


# =====================================================================
# Figure 5: Error convergence and computational cost vs theta
# =====================================================================

print("Running convergence sweep...")
# Use a smaller random target set for fast sweep evaluation
torch.manual_seed(SEED + 1)
N_SWEEP_TARGETS = 500
sweep_targets = torch.rand(N_SWEEP_TARGETS, 2, device=device) * torch.tensor(
    [X_MAX - X_MIN, Y_MAX - Y_MIN], device=device
) + torch.tensor([X_MIN, Y_MIN], device=device)
sweep_target_tree = ClusterTree.from_points(sweep_targets, leaf_size=1)

with torch.no_grad():
    exact_at_sweep = exact_kernel(
        reference_length=torch.tensor(1.0, device=device),
        source_points=source_points,
        target_points=sweep_targets,
        source_strengths=source_strengths,
        source_data=source_data,
    )

max_errors: list[float] = []
mean_errors: list[float] = []
n_kernel_evals: list[int] = []
n_dense = N_SWEEP_TARGETS * N_SOURCE_POINTS

for theta in THETA_SWEEP:
    ### Interaction statistics from the dual plan
    plan = source_tree.find_dual_interaction_pairs(
        target_tree=sweep_target_tree, theta=theta
    )
    n_kernel_evals.append(plan.n_near + plan.n_nf + plan.n_fn + plan.n_far_nodes)

    ### BH kernel evaluation at sweep targets
    with torch.no_grad():
        bh_at_sweep = bh_kernel(
            reference_length=torch.tensor(1.0, device=device),
            source_points=source_points,
            target_points=sweep_targets,
            source_strengths=source_strengths,
            source_data=source_data,
            theta=theta,
            cluster_tree=source_tree,
            target_tree=sweep_target_tree,
        )

    phi_err = (bh_at_sweep["phi"] - exact_at_sweep["phi"]).abs()
    max_errors.append(phi_err.max().item())
    mean_errors.append(phi_err.mean().item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

### Left: error convergence
ax1.loglog(
    THETA_SWEEP, max_errors, "o-", color="tab:red", label=r"Max $|\mathrm{error}|$"
)
ax1.loglog(
    THETA_SWEEP,
    mean_errors,
    "s-",
    color="tab:blue",
    label=r"Mean $|\mathrm{error}|$",
)
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$")
ax1.set_title(r"Approximation error vs. $\theta$")
ax1.legend()
ax1.grid(True, alpha=0.3)

### Right: computational cost
ax2.axhline(
    n_dense,
    color="gray",
    linestyle="--",
    alpha=0.5,
    label=f"Dense ({n_dense:,})",
)
ax2.semilogy(
    THETA_SWEEP,
    n_kernel_evals,
    "^-",
    color="tab:green",
    label="BH kernel evals",
)
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel("Number of kernel evaluations")
ax2.set_title(r"Computation cost vs. $\theta$")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="convergence")
plt.show()
