from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import matplotlib.patches


### [Plotting]
def plot_scalar_and_vector(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    scalar: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    figure_size: tuple[float, float] = (10, 5),
    suptitle: str | None = None,
    suptitle_y: float = 0.9,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots a scalar field (left) and a vector field (right) with the
    exact styling used in 01_kernel_2d_basics.py."""
    fig, ax = plt.subplots(1, 2, figsize=figure_size)

    # Consistent axis formatting
    for a in ax:
        a.set_xlabel(r"$({\bf r}_{ts}\cdot\hat{\bf e}_x)/\ell$")
        a.set_ylabel(r"$({\bf r}_{ts}\cdot\hat{\bf e}_y)/\ell$")
        a.set_aspect("equal", adjustable="box")

    # Left: scalar
    plt.sca(ax[0])
    c_scale = float(np.max(np.abs(np.percentile(scalar, [0.1, 99.9]))))
    p.contour(
        X,
        Y,
        scalar,
        levels=np.linspace(-c_scale, c_scale, 41),
        extend="both",
        cmap="RdBu_r",
        colorbar=False,
        linelabels=False,
    )
    plt.clim(-c_scale, c_scale)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    p.equal()
    plt.colorbar(aspect=30, shrink=0.6)
    p.show_plot(show=False)
    ax[0].set_title(r"Scalar field $\phi$")

    # Right: vector
    plt.sca(ax[1])
    mag = (U**2 + V**2) ** 0.5
    c_scale = np.percentile(mag, 99.9).item()
    streamplot = plt.streamplot(
        np.linspace(X.min(), X.max(), X.shape[1]),
        np.linspace(Y.min(), Y.max(), Y.shape[0]),
        U,
        V,
        color=mag,
        cmap="plasma",
        density=2,
        linewidth=3 * np.log1p(mag / np.median(mag)),
        arrowstyle="fancy",
        arrowsize=1,
    )
    # Improve vector export quality by rounding caps of tiny line segments
    # See: https://stackoverflow.com/questions/72357224/is-there-a-way-to-improve-the-line-quality-when-exporting-streamplots-from-matpl
    streamplot.lines.set_capstyle("round")
    # Ensure arrows remain visible on top of lines
    streamplot.lines.set_zorder(1.9)

    for patch in plt.gca().patches:
        if not isinstance(patch, matplotlib.patches.FancyArrowPatch):
            continue
        patch.set_facecolor(p.adjust_lightness(patch.get_facecolor(), 0.8))
        patch.set_edgecolor(p.adjust_lightness(patch.get_edgecolor(), 0.8))
        patch.set_linewidth(0.8)

    p.contour(
        X,
        Y,
        mag,
        levels=np.linspace(0, c_scale, 10),
        cmap="plasma",
        alpha=0.2,
        colorbar=False,
        linelabels=False,
        zorder=1.9,
    )
    plt.clim(0, c_scale)
    plt.colorbar(streamplot.lines, aspect=30, shrink=0.6)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    p.equal()
    ax[1].set_title(r"Vector field $\bf u$")

    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    return fig, list(ax)


def save_figure(fig: plt.Figure, *, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {pdf_path}")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {png_path}")
