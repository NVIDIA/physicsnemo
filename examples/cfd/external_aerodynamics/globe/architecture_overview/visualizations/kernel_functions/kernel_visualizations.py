from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict
from utils import (
    plot_scalar_and_vector,
    save_figure,
)

from physicsnemo.experimental.models.globe import Kernel

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent

n_spatial_dims = 2

# Domain in nondimensional coordinates (r_ts / ell)
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = -6.0, 6.0
GRID_RES: int = 256

for fig_number in range(5 + 1):
    if fig_number == 0:
        source_points = torch.tensor([[0.0, 0.0]], device=device)
        source_strengths = torch.ones(len(source_points), device=device) * 1e2
        source_data = TensorDict(
            {
                "normal": torch.tensor([[0.0, 1.0]], device=device),
                "other": torch.tensor([[0.0, 0.0]], device=device),
            },
            batch_size=torch.Size([len(source_points)]),
            device=device,
        )
    elif fig_number == 1:
        source_points = torch.tensor([[3.0, -1.0]], device=device)
        source_strengths = torch.ones(len(source_points), device=device) * 1e2
        source_data = TensorDict(
            {
                "normal": torch.tensor([[-0.6, 0.8]], device=device),
                "other": torch.tensor([[0.0, 0.0]], device=device),
            },
            batch_size=torch.Size([len(source_points)]),
            device=device,
        )
    elif fig_number == 2:
        source_points = torch.tensor([[0.0, 0.0]], device=device)
        source_strengths = torch.ones(len(source_points), device=device) * 1e2
        source_data = TensorDict(
            {
                "normal": torch.tensor([[0.0, 1.0]], device=device),
                "other": torch.tensor([[0.7, 0.15]], device=device),
            },
            batch_size=torch.Size([len(source_points)]),
            device=device,
        )
    elif fig_number == 3:
        source_points = torch.tensor([[0.0, 0.0]], device=device)
        source_strengths = torch.ones(len(source_points), device=device) * 1e2
        source_data = TensorDict(
            {
                "normal": torch.tensor([[0.0, 0.0]], device=device),
                "other": torch.tensor([[0.0, 0.0]], device=device),
            },
            batch_size=torch.Size([len(source_points)]),
            device=device,
        )
    elif fig_number == 4 or fig_number == 5:
        t = torch.linspace(0, 1, 4 if fig_number == 4 else 20, device=device)
        # source_points = torch.stack([torch.sin(t), torch.cos(t)], dim=1)
        source_points = torch.stack(
            [2 * (t - 0.5), torch.sin((t - 0.5) * 5) / 5 - t / 3], dim=1
        )
        source_strengths = torch.ones(len(source_points), device=device) * 1e2 / len(t)
        source_data = TensorDict(
            {
                "normal": torch.stack(
                    [torch.sin((t - 0.5 + 1) / 2), torch.cos((t - 0.5 + 1) / 2)], dim=1
                ),
                "other": torch.zeros_like(source_points),
            },
            batch_size=torch.Size([len(source_points)]),
            device=device,
        )

    SEED: int = 39
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    kernel = Kernel(
        n_spatial_dims=n_spatial_dims,
        output_field_ranks={"phi": 0, "u": 1},
        source_data_ranks={"normal": 1, "other": 1},
        hidden_layer_sizes=[64, 64],
        n_spherical_harmonics=4,
        network_type="pade",
        spectral_norm=False,
    ).to(device)

    def f(
        x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> dict[str, np.ndarray]:
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        with torch.no_grad():
            result = kernel(
                reference_length=torch.tensor(1.0, device=device),
                source_points=source_points,
                target_points=torch.stack([x.flatten(), y.flatten()], dim=1),
                source_strengths=source_strengths,
                source_data=source_data,
            )
        return {
            k: v.detach()
            .reshape([*x.shape, *v.shape[1:]])
            .cpu()
            .numpy()
            .astype(np.float64)
            for k, v in result.items()
        }

    ### [Grid + Evaluate]
    x = np.linspace(X_MIN, X_MAX, GRID_RES)
    y = np.linspace(Y_MIN, Y_MAX, GRID_RES)
    X, Y = np.meshgrid(x, y, indexing="xy")

    res = f(X, Y)
    Z = res["phi"]
    U, V = res["u"][..., 0], res["u"][..., 1]

    fig, ax = plot_scalar_and_vector(
        X=X,
        Y=Y,
        scalar=Z,
        U=U,
        V=V,
        figure_size=(10, 5),
        suptitle="",
        suptitle_y=0.96,
    )
    for a in ax:
        a.scatter(
            *source_points.cpu().numpy().T, color="black", marker="o", zorder=5, s=9
        )
        for i in range(len(source_points)):
            for vec in source_data.values():
                a.annotate(
                    text="",
                    xy=source_points[i].cpu().numpy(),
                    xytext=source_points[i].cpu().numpy() + 2 * vec[i].cpu().numpy(),
                    arrowprops=dict(
                        arrowstyle="<-", color="black", linewidth=1.2, alpha=1
                    ),
                    ha="center",
                    va="center",
                    color="black",
                    zorder=4,
                )

    save_figure(fig, output_dir=OUTPUT_DIR, stem=f"kernel_visualization_{fig_number}")
    plt.show()
