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

from physicsnemo.experimental.models.globe import BarnesHutKernel
from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent

### [Configuration]
SEED = 39
N_SOURCE_POINTS = 20
THETA_VALUES = [0.1, 0.25, 0.5, 1.0]
THETA_SWEEP = np.geomspace(0.01, 10.0, 21)
GRID_RES = 128
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = -6.0, 6.0


### [Source points: asymmetric blobby boundary with outward normals]
theta_boundary = np.linspace(0, 2 * np.pi, N_SOURCE_POINTS + 1)
r = (
    1
    + 0.25 * np.cos(2 * theta_boundary + 0.3)
    + 0.15 * np.cos(3 * theta_boundary + 1.7)
    + 0.10 * np.cos(5 * theta_boundary + 0.8)
)
boundary_coords = np.column_stack([
    r * np.cos(theta_boundary),
    r * np.sin(theta_boundary),
])
# Cell centroids and outward normals from consecutive coordinate pairs
src_centroids_np = (boundary_coords[:-1] + boundary_coords[1:]) / 2
tangents = boundary_coords[1:] - boundary_coords[:-1]
lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
tangents /= np.clip(lengths, 1e-12, None)
src_normals_np = np.column_stack([tangents[:, 1], -tangents[:, 0]])

n_source = len(src_centroids_np)
source_points = torch.as_tensor(src_centroids_np, device=device, dtype=torch.float32)
# Sinusoidal strength variation around the boundary breaks field uniformity
theta_mid = (theta_boundary[:-1] + theta_boundary[1:]) / 2
strength_variation = 1 * np.sin(7 * theta_mid + 0.5)
source_strengths = torch.as_tensor(
    strength_variation * 1e2 / n_source, device=device, dtype=torch.float32
)
source_data = TensorDict(
    {
        "normal": torch.as_tensor(src_normals_np, device=device, dtype=torch.float32),
        "other": torch.zeros(n_source, 2, device=device),
    },
    batch_size=torch.Size([n_source]),
    device=device,
)
src_np = src_centroids_np


### [Kernel: single BarnesHutKernel, using theta=0 as the exact baseline]
torch.manual_seed(SEED)
np.random.seed(SEED)
kernel = BarnesHutKernel(
    n_spatial_dims=2,
    output_field_ranks={"phi": 0, "u": 1},
    source_data_ranks={"normal": 1, "other": 1},
    hidden_layer_sizes=[64, 64],
    n_spherical_harmonics=4,
    network_type="pade",
    spectral_norm=False,
    use_gradient_checkpointing=False,
    leaf_size=1,
).to(device)
kernel.eval()


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


### [Tree visualization: higher-resolution version of the same blob shape]
N_TREE_VIZ_POINTS = 60
theta_tree = np.linspace(0, 2 * np.pi, N_TREE_VIZ_POINTS + 1)
r_tree = (
    1
    + 0.25 * np.cos(2 * theta_tree + 0.3)
    + 0.15 * np.cos(3 * theta_tree + 1.7)
    + 0.10 * np.cos(5 * theta_tree + 0.8)
)
tree_coords = np.column_stack([r_tree * np.cos(theta_tree), r_tree * np.sin(theta_tree)])
tree_viz_np = (tree_coords[:-1] + tree_coords[1:]) / 2
tree_viz_points = torch.as_tensor(tree_viz_np, device=device, dtype=torch.float32)
tree_viz_tree = ClusterTree.from_points(tree_viz_points, leaf_size=1)

tree_nodes = walk_tree(tree_viz_tree)
max_depth = max(n["depth"] for n in tree_nodes)
cmap_tree = plt.cm.viridis

# =====================================================================
# Figure 1a: ClusterTree AABB hierarchy (2D)
# =====================================================================

fig, ax = plt.subplots(figsize=(6, 5))
# Draw from shallowest to deepest so deeper (smaller) nodes paint on top.
for node in sorted(tree_nodes, key=lambda n: n["depth"]):
    frac = node["depth"] / max(max_depth, 1)
    color = cmap_tree(frac)
    xy = node["aabb_min"]
    size = node["aabb_max"] - node["aabb_min"]
    lw = max(2.5 - 0.3 * node["depth"], 0.5)
    alpha = 0.03 if node["is_leaf"] else 0.08
    rect = mpatches.FancyBboxPatch(
        xy,
        size[0],
        size[1],
        boxstyle="round,pad=0.02",
        linewidth=lw,
        edgecolor=color,
        facecolor=(*color[:3], alpha),
    )
    ax.add_patch(rect)

ax.scatter(*tree_viz_np.T, color="black", s=12, zorder=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("ClusterTree AABB hierarchy")
pad = 0.15
ax.set_xlim(tree_viz_np[:, 0].min() - pad, tree_viz_np[:, 0].max() + pad)
ax.set_ylim(tree_viz_np[:, 1].min() - pad, tree_viz_np[:, 1].max() + pad)
sm = plt.cm.ScalarMappable(cmap=cmap_tree, norm=plt.Normalize(0, max_depth))
plt.colorbar(sm, ax=ax, label="Tree depth")
p.show_plot(show=False)
save_figure(fig, stem="tree_hierarchy_2d")
plt.show()


# =====================================================================
# Figure 1b: ClusterTree AABB hierarchy (3D stacked)
# =====================================================================

fig, ax = p.figure3d(box_aspect=[1, 1, 0.4], figsize=(7, 6))
Z_SPACING = 0.7

# Index nodes by ID for parent-child line drawing
nodes_by_id = {n["id"]: n for n in tree_nodes}

for node in sorted(tree_nodes, key=lambda n: n["depth"]):
    frac = node["depth"] / max(max_depth, 1)
    color = cmap_tree(frac)
    z = (max_depth - node["depth"]) * Z_SPACING
    x0, y0 = node["aabb_min"]
    x1, y1 = node["aabb_max"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    verts = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
    ax.add_collection3d(
        Poly3DCollection(
            [verts],
            facecolor=(*color[:3], 0.12),
            edgecolor=(*color[:3], 0.6),
            linewidth=1.2,
        )
    )

    ### Draw lines from this node's center down to each child's center
    if not node["is_leaf"]:
        nid = node["id"]
        for child_attr in ("node_left_child", "node_right_child"):
            child_id = getattr(tree_viz_tree, child_attr)[nid].item()
            if child_id >= 0 and child_id in nodes_by_id:
                ch = nodes_by_id[child_id]
                ch_cx = (ch["aabb_min"][0] + ch["aabb_max"][0]) / 2
                ch_cy = (ch["aabb_min"][1] + ch["aabb_max"][1]) / 2
                ch_z = (max_depth - ch["depth"]) * Z_SPACING
                ax.plot(
                    [cx, ch_cx], [cy, ch_cy], [z, ch_z],
                    color="gray", alpha=0.4, linewidth=0.8,
                )

ax.scatter(
    *tree_viz_np.T, zs=0, zdir="z",
    color="black", s=8, depthshade=False, zorder=5,
)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
# ax.set_zlabel("Depth")
ax.set_title("ClusterTree AABB hierarchy (3D)")
# ax.set_zticks([i * Z_SPACING for i in range(max_depth + 1)])
# ax.set_zticklabels([f"depth {max_depth - i}" for i in range(max_depth + 1)])
ax.view_init(elev=30, azim=-50)
p.show_plot(show=False)
save_figure(fig, stem="tree_hierarchy_3d")
plt.show()


# =====================================================================
# Precompute kernel results for Figures 3 and 4
# =====================================================================

EXPAND_MODES = [False, True]
EXPAND_LABELS = {False: "Broadcast (default)", True: "Expanded targets"}

print("Evaluating kernel on grid (theta=0, exact)...")
exact_result = evaluate_on_grid(
    kernel, theta=0.0, cluster_tree=source_tree, target_tree=grid_target_tree,
)

grid_results: dict[bool, dict[float, dict[str, np.ndarray]]] = {}
for expand in EXPAND_MODES:
    grid_results[expand] = {}
    for theta in THETA_VALUES:
        print(
            f"Evaluating kernel on grid "
            f"(theta={theta}, expand_far_targets={expand})..."
        )
        grid_results[expand][theta] = evaluate_on_grid(
            kernel,
            theta=theta,
            cluster_tree=source_tree,
            target_tree=grid_target_tree,
            expand_far_targets=expand,
        )


# =====================================================================
# Figure 3: Kernel scalar field comparison (broadcast vs expanded)
# =====================================================================

col_labels = ["Exact"] + [rf"$\theta = {th}$" for th in THETA_VALUES]
n_cols = len(col_labels)
n_rows = len(EXPAND_MODES)

phi_scale = float(
    np.max(np.abs(np.percentile(exact_result["phi"], [0.1, 99.9])))
)
levels = np.linspace(-phi_scale, phi_scale, 31)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
for row, expand in enumerate(EXPAND_MODES):
    phi_fields = [exact_result["phi"]] + [
        grid_results[expand][th]["phi"] for th in THETA_VALUES
    ]
    for col, (ax, label, phi) in enumerate(
        zip(axes[row], col_labels, phi_fields)
    ):
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
        if row == 0:
            ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        if row == n_rows - 1:
            ax.set_xlabel(r"$x/\ell$")
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(EXPAND_LABELS[expand] + "\n" + r"$y/\ell$")
        else:
            ax.set_yticklabels([])

fig.suptitle(r"Scalar field $\phi$ at different Barnes-Hut $\theta$", y=1.02)
plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="kernel_comparison")
plt.show()


# =====================================================================
# Figure 4: Approximation error fields (broadcast vs expanded)
# =====================================================================

# Consistent log-scale normalization across ALL panels and both modes
all_error_arrays = [
    np.abs(grid_results[expand][th]["phi"] - exact_result["phi"])
    for expand in EXPAND_MODES
    for th in THETA_VALUES
]
positive_vals = np.concatenate([e[e > 0].ravel() for e in all_error_arrays])
vmin = (
    max(float(positive_vals.min()), 1e-8) if len(positive_vals) > 0 else 1e-8
)
vmax = float(max(e.max() for e in all_error_arrays))

n_cols = len(THETA_VALUES)
n_rows = len(EXPAND_MODES)
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3.5 * n_cols + 1, 3.5 * n_rows),
    gridspec_kw={"right": 0.88},
)
for row, expand in enumerate(EXPAND_MODES):
    errors = [
        np.abs(grid_results[expand][th]["phi"] - exact_result["phi"])
        for th in THETA_VALUES
    ]
    for col, (ax, theta, err) in enumerate(
        zip(axes[row], THETA_VALUES, errors)
    ):
        im = ax.pcolormesh(
            X_grid,
            Y_grid,
            np.clip(err, vmin, None),
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )
        ax.scatter(*src_np.T, color="white", s=5, zorder=5)
        if row == 0:
            ax.set_title(rf"$\theta = {theta}$")
        ax.text(
            0.98, 0.02, f"max = {err.max():.2e}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )
        ax.set_aspect("equal")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        if row == n_rows - 1:
            ax.set_xlabel(r"$x/\ell$")
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(EXPAND_LABELS[expand] + "\n" + r"$y/\ell$")
        else:
            ax.set_yticklabels([])

cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
fig.colorbar(
    im, cax=cbar_ax,
    label=r"$|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$",
)
fig.suptitle(
    r"Approximation error $|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$", y=1.02
)
p.show_plot(show=False)
save_figure(fig, stem="error_fields")
plt.show()


# =====================================================================
# Figure 5: Error convergence and computational cost vs theta
# =====================================================================

print("Running convergence sweep...")
torch.manual_seed(SEED + 1)
N_SWEEP_TARGETS = 500
sweep_targets = torch.rand(N_SWEEP_TARGETS, 2, device=device) * torch.tensor(
    [X_MAX - X_MIN, Y_MAX - Y_MIN], device=device
) + torch.tensor([X_MIN, Y_MIN], device=device)
sweep_target_tree = ClusterTree.from_points(sweep_targets, leaf_size=1)

with torch.no_grad():
    exact_at_sweep = kernel(
        reference_length=torch.tensor(1.0, device=device),
        source_points=source_points,
        target_points=sweep_targets,
        source_strengths=source_strengths,
        source_data=source_data,
        theta=0.0,
        cluster_tree=source_tree,
        target_tree=sweep_target_tree,
    )

n_dense = N_SWEEP_TARGETS * n_source

### Sweep both modes
sweep_data: dict[bool, dict] = {}
for expand in EXPAND_MODES:
    max_errs: list[float] = []
    mean_errs: list[float] = []
    n_evals: list[int] = []

    for theta in THETA_SWEEP:
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=sweep_target_tree,
            theta=theta,
            expand_far_targets=expand,
        )
        n_evals.append(
            plan.n_near + plan.n_nf + plan.n_fn + plan.n_far_nodes
        )

        with torch.no_grad():
            result_at_sweep = kernel(
                reference_length=torch.tensor(1.0, device=device),
                source_points=source_points,
                target_points=sweep_targets,
                source_strengths=source_strengths,
                source_data=source_data,
                theta=theta,
                cluster_tree=source_tree,
                target_tree=sweep_target_tree,
                expand_far_targets=expand,
            )

        phi_err = (result_at_sweep["phi"] - exact_at_sweep["phi"]).abs()
        max_errs.append(phi_err.max().item())
        mean_errs.append(phi_err.mean().item())

    sweep_data[expand] = {
        "max_errors": max_errs,
        "mean_errors": mean_errs,
        "n_kernel_evals": n_evals,
    }

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

EXPAND_COLORS = {False: "C0", True: "C1"}

### Shared index for skipping theta=0 on log-scale axes
theta_nonzero = [th for th in THETA_SWEEP if th > 0]
idx = [i for i, th in enumerate(THETA_SWEEP) if th > 0]

### Left: error vs theta
for expand in EXPAND_MODES:
    c = EXPAND_COLORS[expand]
    d = sweep_data[expand]
    ax1.loglog(
        theta_nonzero, [d["mean_errors"][i] for i in idx], "o-",
        color=c, label=EXPAND_LABELS[expand],
    )
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"Mean $|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$")
ax1.set_title(r"Error vs. $\theta$")
ax1.legend()
ax1.grid(True, alpha=0.3)

### Center: cost vs theta
ax2.axhline(
    n_dense, color="gray", linestyle="--", alpha=0.5,
    label=f"Dense ({n_dense:,})",
)
for expand in EXPAND_MODES:
    c = EXPAND_COLORS[expand]
    d = sweep_data[expand]
    ax2.loglog(
        theta_nonzero, [d["n_kernel_evals"][i] for i in idx], "^-",
        color=c, label=EXPAND_LABELS[expand],
    )
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel("Kernel evaluations")
ax2.set_title(r"Cost vs. $\theta$")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

### Right: error vs cost (Pareto-style)
for expand in EXPAND_MODES:
    c = EXPAND_COLORS[expand]
    d = sweep_data[expand]
    evals_nz = [d["n_kernel_evals"][i] for i in idx]
    errs_nz = [d["mean_errors"][i] for i in idx]
    ax3.loglog(
        evals_nz, errs_nz, "o-",
        color=c, label=EXPAND_LABELS[expand],
    )
    # Label min and max theta at the endpoints
    for j, label_idx in enumerate([0, -1]):
        th = theta_nonzero[label_idx]
        ax3.annotate(
            rf"$\theta$={th}",
            xy=(evals_nz[label_idx], errs_nz[label_idx]),
            textcoords="offset points",
            xytext=(8, 8 if j == 0 else -12),
            fontsize=7, color=c,
            arrowprops=dict(arrowstyle="-", color=c, alpha=0.4, lw=0.8),
        )
ax3.set_xlabel("Kernel evaluations")
ax3.set_ylabel(r"Mean $|\phi_\mathrm{BH} - \phi_\mathrm{exact}|$")
ax3.set_title("Error vs. cost")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
p.show_plot(show=False)
save_figure(fig, stem="convergence")
plt.show()
