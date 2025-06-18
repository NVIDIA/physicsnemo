from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

files = {
    "Raw Sensitivities": Path(__file__).parent / "drag_gradients_raw.txt",
    "Smooth Sensitivities": Path(__file__).parent / "drag_gradients_smooth.txt",
}

plt.figure(figsize=(9, 7))

for name, file in files.items():
    data = np.loadtxt(file, delimiter=",")
    epsilon = data[:, 0]
    drag = data[:, 1]

    baseline_drag = drag[epsilon == 0][0]
    drag_delta = drag - baseline_drag

    plt.plot(
        epsilon,
        drag_delta,
        ".",
        label=name,
    )

x = np.unique(np.concatenate([line.get_xdata() for line in plt.gca().get_lines()]))
x_minscale = np.min(np.abs(x[x != 0]))
x_maxscale = np.max(np.abs(x[x != 0]))

sorted_x = np.sort(
    np.concatenate(
        (
            np.logspace(np.log10(x_minscale) - 3, np.log10(x_maxscale), 100),
            np.abs(epsilon),
        )
    )
)
sorted_x = np.concatenate((-sorted_x[::-1], sorted_x))

analytical_gradient = -376531.0

plt.plot(
    sorted_x,
    analytical_gradient * sorted_x,
    "--k",
    label="Adjoint-Based Gradient",
    zorder=1.9,
)

# Set up logit axes with symmetric linear range
plt.xscale("symlog", linthresh=x_minscale)
plt.yscale("symlog", linthresh=np.abs(analytical_gradient) * x_minscale)

plt.gca().xaxis.set_major_locator(
    ticker.SymmetricalLogLocator(base=100, linthresh=x_minscale)
)
plt.gca().yaxis.set_major_locator(
    ticker.SymmetricalLogLocator(
        base=100, linthresh=np.abs(analytical_gradient) * x_minscale
    )
)

# Add grid for better readability
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.xlabel("Epsilon")
plt.ylabel("(Drag) - (Drag Baseline)\n[N]")
plt.title("Adjoint-Predicted Gradient vs. Finite-Differences")
plt.legend()
plt.show()
