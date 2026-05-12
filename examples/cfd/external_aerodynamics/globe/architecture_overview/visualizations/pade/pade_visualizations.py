from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

from physicsnemo.nn import Mlp, Pade

device = "cpu"
OUTPUT_DIR = Path(__file__).parent

### [Configuration]
N_RANDOM_INSTANCES = 25
x_rng = 100
N_POINTS = 501

for network_type in ["pade", "mlp"]:
    x = torch.linspace(-x_rng, x_rng, N_POINTS, device=device).unsqueeze(-1)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    plt.sca(ax)

    for i, seed in enumerate(range(N_RANDOM_INSTANCES)):
        torch.manual_seed(i)
        if network_type == "pade":
            net = Pade(
                in_features=1,
                hidden_features=[64, 64],
                out_features=1,
                numerator_order=1,
                denominator_order=3,
                share_denominator_across_channels=False,
                use_separate_mlps=True,
            ).to(device)
        elif network_type == "mlp":
            net = Mlp(
                in_features=1,
                hidden_features=[64, 64],
                out_features=1,
            ).to(device)

        with torch.no_grad():
            y = net(x)

        basecolor = "C0" if network_type == "pade" else "C1"

        plt.plot(
            x.cpu().numpy(),
            y.cpu().numpy(),
            color=p.adjust_lightness(
                basecolor,
                amount=0 + 1.5 * i / (N_RANDOM_INSTANCES - 1),
            ),
        )
        if i == 0:
            plt.plot(
                [],
                [],
                color=basecolor,
                label="Padé approximant MLP"
                if network_type == "pade"
                else "Standard MLP",
            )

    ax.set_xlabel(r"Input")
    ax.set_ylabel(r"Output")
    all_y_values = np.concatenate(
        [line.get_ydata().flatten() for line in ax.get_lines()]
    )
    y_rng = np.max(np.abs(np.percentile(all_y_values, [1, 99])))
    plt.xlim(-x_rng, x_rng)
    plt.ylim(-y_rng, y_rng)
    plt.legend(loc="upper center")
    p.show_plot(
        show=True,
        legend=False,
        savefig=str(OUTPUT_DIR / f"{network_type}_random_init.pdf"),
    )
