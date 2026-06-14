"""
Publication-ready figures (matplotlib, headless Agg backend).

Each function returns a Matplotlib Figure and, given an out_path, saves a
300-dpi PNG suitable for a manuscript. Styling is deliberately plain:
readable fonts, no chartjunk, colour used only to carry meaning.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ACCENT = "#1f6feb"
_WARN = "#d29922"
_MUTE = "#6e7681"


def _save(fig, out_path):
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def _smooth(grid, size):
    """Light NaN-aware median smoothing for display (denoises ratio speckle)."""
    if not size or size < 2:
        return grid
    from scipy.ndimage import median_filter
    g = np.array(grid, dtype="float64")
    mask = np.isnan(g)
    g_filled = np.where(mask, np.nanmedian(g), g)
    sm = median_filter(g_filled, size=size)
    sm[mask] = np.nan
    return sm


def raster_map(grid, extent, title, cbar_label, *, cmap="viridis",
               vmin=None, vmax=None, base=None, center=None, smooth=0,
               marker=None, marker_label="known deposit", out_path=None):
    """Imshow a 2D index grid over a lon/lat extent (real-imagery outputs).

    ``center`` makes the colour scale diverging and symmetric about that value
    (use with a diverging cmap like 'RdBu_r' so e.g. red = above-typical,
    blue = below). ``smooth`` applies an NxN median filter for display only.
    ``marker`` (lat, lon) draws a labelled ring (e.g. the known deposit).
    ``base`` draws a faint grey footprint of valid pixels beneath the overlay.
    """
    grid = _smooth(grid, smooth)
    if center is not None:
        lo = np.nanpercentile(grid, 2) if vmin is None else vmin
        hi = np.nanpercentile(grid, 98) if vmax is None else vmax
        half = max(center - lo, hi - center, 1e-9)
        vmin, vmax = center - half, center + half
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if base is not None:
        ax.imshow(np.where(np.isfinite(base), 0.0, np.nan), extent=extent,
                  origin="upper", cmap="Greys", vmin=0, vmax=1, aspect="auto")
    im = ax.imshow(grid, extent=extent, origin="upper", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(im, label=cbar_label)
    if marker is not None:
        import matplotlib.patheffects as pe
        sc = ax.scatter([marker[1]], [marker[0]], s=260, facecolors="none",
                        edgecolors="white", linewidths=2.6, label=marker_label,
                        zorder=6)
        sc.set_path_effects([pe.withStroke(linewidth=5.2, foreground="black")])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, out_path)


def leakage_figure(report: dict, title="Random vs spatial-block CV", out_path=None):
    """Bar chart contrasting random K-fold and spatial-block CV scores."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    means = [report["random_mean"], report["spatial_mean"]]
    errs = [report["random_std"], report["spatial_std"]]
    labels = ["Random K-fold\n(optimistic)", "Spatial-block CV\n(honest)"]
    bars = ax.bar(labels, means, yerr=errs, capsize=6,
                  color=[_MUTE, _ACCENT], width=0.6)
    metric = report.get("metric", "score").upper()
    ax.set_ylabel(metric)
    ax.set_title(title)
    gap = report["leakage_gap"]
    ax.annotate(f"leakage gap = {gap:+.3f}",
                xy=(0.5, max(means) + max(errs) + 0.02),
                ha="center", fontsize=10, color=_WARN, fontweight="bold")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m - 0.06, f"{m:.3f}",
                ha="center", color="white", fontweight="bold")
    ax.set_ylim(0, max(1.0, max(means) + max(errs) + 0.12))
    fig.tight_layout()
    return _save(fig, out_path)


def gap_figure(projection, commodities=("Gr", "Li", "REE", "Cu"),
               title="Projected supply-demand gap", out_path=None):
    """Gap-index trajectories for selected commodities."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c in commodities:
        sub = projection[projection["commodity"] == c].sort_values("year")
        if len(sub):
            ax.plot(sub["year"], sub["gap_index"], marker="o", ms=3,
                    label=f"{c} ({sub['name'].iloc[0]})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap index (demand - domestic, consumption=100)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_path)


def monitoring_figure(series, flags=None, n_sites=6,
                      title="Mine-footprint signal (BSI) over time", out_path=None):
    """Footprint index trends; expanding sites highlighted."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    expanding = set(flags[flags["is_expanding"]]["name"]) if flags is not None else set()
    names = list(series["name"].unique())[:n_sites]
    for name in names:
        g = series[series["name"] == name].sort_values("t")
        hot = name in expanding
        ax.plot(g["t"], g["bsi"], lw=2 if hot else 1,
                color=_WARN if hot else _MUTE, alpha=0.95 if hot else 0.6,
                label=f"{name}{' (expanding)' if hot else ''}")
    ax.set_xlabel("Time (period)")
    ax.set_ylabel("BSI (bare-ground signal)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_path)
