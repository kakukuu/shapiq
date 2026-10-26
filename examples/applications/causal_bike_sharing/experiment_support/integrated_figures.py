"""Native figures for paired, full-holdout experiment narratives.

Inputs are already paired arrays; this module never selects or truncates rows.
Interaction columns follow ``combinations(range(n_features), 2)``. SII edges
describe output interactions and must not be interpreted as causal graph edges.
Every figure is descriptive: sample dispersion is not estimator uncertainty.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

BASE_COLOR = "#356a9b"
VARIANT_COLOR = "#dd893a"
POSITIVE_COLOR = "#b63e43"
NEGATIVE_COLOR = "#366a9d"


def _arrays(baseline, variant, feature_names, *, interactions=False):
    """Validate the paired contract before calculating any summaries."""
    names = list(feature_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique feature names are required.")
    width = len(names) * (len(names) - 1) // 2 if interactions else len(names)
    left, right = np.asarray(baseline, dtype=float), np.asarray(variant, dtype=float)
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] != width or len(left) == 0:
        raise ValueError(f"Expected equally shaped nonempty paired arrays with {width} columns.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Resolve missing/nonfinite values before plotting paired samples.")
    return left, right, names


def _limit(*arrays):
    return max(max(float(np.max(np.abs(array))) for array in arrays), 1e-12)


def _title(title_prefix, text):
    return f"{title_prefix} | {text}" if title_prefix else text


def _finish(fig, output_path):
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    return fig


def _clean(ax, *, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#e1e5e9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_first_order_distribution(
    baseline_phi,
    variant_phi,
    *,
    feature_names,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    output_path=None,
):
    """Paired dots and distributions for every supplied sample and feature.

    The same deterministic jitter is used for the paired arms. Thin connectors
    refer to identical rows; bars show sample quartiles and white dots medians.
    No confidence interval is drawn or implied by a distribution width.
    """
    base, variant, names = _arrays(baseline_phi, variant_phi, feature_names)
    n, d = base.shape
    fig, ax = plt.subplots(figsize=(13, max(6.4, d * 0.85)), layout="constrained")
    # A deterministic sequence avoids random state and keeps rows visibly paired.
    jitter = (((np.arange(n) * 0.61803398875) % 1) - 0.5) * 0.14
    for j in range(d):
        y0, y1 = j - 0.18 + jitter, j + 0.18 + jitter
        segments = np.stack(
            [np.column_stack([base[:, j], y0]), np.column_stack([variant[:, j], y1])], axis=1
        )
        ax.add_collection(
            LineCollection(segments, colors="#66717b", alpha=0.12, linewidths=0.4, zorder=1)
        )
        for values, y, center, color in (
            (base[:, j], y0, j - 0.18, BASE_COLOR),
            (variant[:, j], y1, j + 0.18, VARIANT_COLOR),
        ):
            if np.ptp(values) > 0 and n > 1:
                violin = ax.violinplot(
                    values,
                    positions=[center],
                    orientation="horizontal",
                    widths=0.30,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.16)
            ax.scatter(
                values,
                y,
                s=10,
                color=color,
                alpha=0.46,
                edgecolors="none",
                zorder=2,
                rasterized=True,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot([q25, q75], [center, center], color=color, linewidth=3, zorder=3)
            ax.scatter(
                [median],
                [center],
                s=23,
                facecolors="white",
                edgecolors=color,
                linewidths=1.2,
                zorder=4,
            )
    limit = _limit(base, variant) * 1.06
    ax.set_xlim(-limit, limit)
    ax.set_yticks(range(d), names)
    ax.set_ylim(d - 0.45, -0.6)
    ax.axvline(0, color="#596570", linewidth=0.7)
    ax.set_xlabel("First-order attribution (model-output units)")
    ax.set_title(_title(title_prefix, f"Paired first-order distributions | n = {n}"), pad=45)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BASE_COLOR, linestyle="", label=baseline_label),
            Line2D([], [], marker="o", color=VARIANT_COLOR, linestyle="", label=variant_label),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    ax.text(
        0.5,
        -0.13,
        "Each dot is one held-out row; connectors pair the same row. Bars: 25th–75th percentiles; white dots: medians.\n"
        "Distributions describe variation across rows, not Monte Carlo uncertainty.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4c5660",
    )
    _clean(ax)
    return _finish(fig, output_path)


def plot_first_order_summary(
    baseline_phi,
    variant_phi,
    *,
    feature_names,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    output_path=None,
):
    """Mean attribution and the three distinct change summaries, in 2 × 2 panels."""
    base, variant, names = _arrays(baseline_phi, variant_phi, feature_names)
    delta = variant - base
    y = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    ax = axes[0, 0]
    ax.barh(y - 0.17, base.mean(axis=0), height=0.30, color=BASE_COLOR, label=baseline_label)
    ax.barh(y + 0.17, variant.mean(axis=0), height=0.30, color=VARIANT_COLOR, label=variant_label)
    ax.set_title("Average signed attribution: mean(φ)", fontsize=12, pad=36)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9, frameon=False)
    summaries = (
        (axes[0, 1], delta.mean(axis=0), "Directional shift: mean(Δφ)", False),
        (axes[1, 0], np.abs(delta).mean(axis=0), "Change size: mean(|Δφ|)", True),
        (
            axes[1, 1],
            (np.abs(variant) - np.abs(base)).mean(axis=0),
            "Magnitude change: mean(|φ modified| − |φ baseline|)",
            False,
        ),
    )
    for ax, values, title, absolute in summaries:
        colors = (
            VARIANT_COLOR if absolute else np.where(values >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
        )
        ax.barh(y, values, height=0.62, color=colors)
        ax.set_title(title, fontsize=12)
        limit = _limit(values)
        ax.set_xlim((0, limit * 1.30) if absolute else (-limit * 1.30, limit * 1.30))
        for j, value in enumerate(values):
            ax.text(
                value + (0.025 * limit if value >= 0 else -0.025 * limit),
                j,
                f"{value:,.1f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
    for ax in axes.flat:
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.axvline(0, color="#596570", linewidth=0.7)
        ax.set_xlabel("Model-output units")
        _clean(ax)
    fig.suptitle(
        _title(title_prefix, f"First-order averages and change definitions | n = {len(base)}")
        + "\nΔφ = modified − baseline; panels have separate scales. Descriptive full-holdout summaries.",
        fontsize=13,
    )
    return _finish(fig, output_path)


def plot_sample_heterogeneity(
    baseline_phi,
    variant_phi,
    *,
    feature_names,
    context_values,
    context_label,
    color_values=None,
    color_label=None,
    feature=None,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    output_path=None,
):
    """Relate paired change to observed input context without claiming causation.

    ``feature`` may be a feature name or a column index. If omitted, the feature
    with the largest mean absolute paired change is selected descriptively.
    ``context_values`` and optional numeric ``color_values`` must follow the
    exact supplied sample order, including date pairing for Lag comparisons.
    """
    base, variant, names = _arrays(baseline_phi, variant_phi, feature_names)
    n = len(base)
    context = np.asarray(context_values, dtype=float)
    if context.shape != (n,) or not np.isfinite(context).all():
        raise ValueError("context_values must be one finite numeric value per paired row.")
    color = None if color_values is None else np.asarray(color_values, dtype=float)
    if color is not None and (color.shape != (n,) or not np.isfinite(color).all()):
        raise ValueError("color_values must be one finite numeric value per paired row.")
    delta = variant - base
    if feature is None:
        index = int(np.abs(delta).mean(axis=0).argmax())
        selection_note = "Feature selected by largest mean(|Δφ|) across held-out rows."
    else:
        index = names.index(feature) if isinstance(feature, str) else int(feature)
        if not 0 <= index < len(names):
            raise ValueError("feature index is out of bounds.")
        selection_note = "Feature specified before plotting."
    selected = delta[:, index]
    shift_size = np.abs(delta).mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    scatter_kw = {"c": color, "cmap": "viridis"} if color is not None else {"color": BASE_COLOR}
    image = None
    for ax, values in zip(axes, (selected, shift_size), strict=True):
        image = ax.scatter(
            context, values, s=29, alpha=0.86, edgecolors="#596570", linewidths=0.35, **scatter_kw
        )
        ax.set_xlabel(context_label)
        _clean(ax, axis="both")
    axes[0].axhline(0, color="#656d75", linewidth=0.8)
    axes[0].axhline(
        selected.mean(),
        color=POSITIVE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Mean Δφ = {selected.mean():,.1f}",
    )
    axes[0].set_ylabel(f"Δφ({names[index]}) | {variant_label} − {baseline_label}")
    axes[0].set_title(f"Signed change: {names[index]}\nMean |Δφ| = {np.abs(selected).mean():,.1f}")
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    axes[1].set_ylabel("Mean absolute first-order change across features")
    axes[1].set_title("First-order change size per row\nBoth signs contribute to change size")
    axes[1].set_ylim(bottom=0)
    if color is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label=color_label or "Observed context")
    fig.suptitle(
        _title(title_prefix, f"Sample heterogeneity | n = {n}")
        + f"\n{selection_note} Associations with context are descriptive.",
        fontsize=12,
    )
    return _finish(fig, output_path)


def _interaction_matrix(values, d):
    matrix = np.full((d, d), np.nan)
    for value, (i, j) in zip(values, combinations(range(d), 2), strict=True):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def plot_mean_interaction_matrices(
    baseline_interactions,
    variant_interactions,
    *,
    feature_names,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    output_path=None,
):
    """Mean pairwise SII matrices and signed mean difference.

    Baseline/modified share a symmetric color scale; difference has a separate
    symmetric scale, explicitly labelled. Diagonals are masked, not set to zero.
    """
    base, variant, names = _arrays(
        baseline_interactions, variant_interactions, feature_names, interactions=True
    )
    averages = [base.mean(axis=0), variant.mean(axis=0), (variant - base).mean(axis=0)]
    common_limit = _limit(*averages[:2])
    limits = [common_limit, common_limit, _limit(averages[2])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), layout="constrained")
    images = []
    for ax, values, label, limit in zip(
        axes,
        averages,
        (baseline_label, variant_label, "Mean difference: modified − baseline"),
        limits,
        strict=True,
    ):
        matrix = _interaction_matrix(values, len(names))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        images.append(im)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title(label, fontsize=11)
        for i, j in combinations(range(len(names)), 2):
            for row, col in ((i, j), (j, i)):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[row, col]) > 0.65 * limit else "#17222c",
                )
    fig.colorbar(images[0], ax=axes[:2], shrink=0.72, label="Mean SII | shared arm scale")
    fig.colorbar(images[2], ax=axes[2], shrink=0.72, label="Mean ΔSII | separate difference scale")
    fig.suptitle(
        _title(title_prefix, f"Average pairwise interactions | n = {len(base)}")
        + "\nOutput SII interactions, not input causal edges. Color denotes signed mean, not mean absolute change.",
        fontsize=12,
    )
    return _finish(fig, output_path)


def plot_mean_interaction_networks(
    baseline_interactions,
    variant_interactions,
    *,
    feature_names,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    top_k=5,
    output_path=None,
):
    """Compare mean SII networks on fixed nodes and one edge color/width scale.

    Both panels display the union of the top-k absolute signed-mean interaction
    pairs in either arm. The union is held fixed; no panel-specific threshold or
    normalization can make edges appear or disappear solely due to scaling.
    """
    base, variant, names = _arrays(
        baseline_interactions, variant_interactions, feature_names, interactions=True
    )
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    means = [base.mean(axis=0), variant.mean(axis=0)]
    pairs = list(combinations(range(len(names)), 2))
    k = min(int(top_k), len(pairs))
    chosen = sorted(
        set(np.argsort(-np.abs(means[0]), kind="stable")[:k])
        | set(np.argsort(-np.abs(means[1]), kind="stable")[:k])
    )
    scale = _limit(*means)
    norm, cmap = Normalize(-scale, scale), plt.get_cmap("RdBu_r")
    angles = np.pi / 2 - np.arange(len(names)) * 2 * np.pi / len(names)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), layout="constrained")
    for ax, values, label in zip(axes, means, (baseline_label, variant_label), strict=True):
        for index in sorted(chosen, key=lambda item: abs(values[item])):
            i, j = pairs[index]
            value = values[index]
            ax.plot(
                positions[[i, j], 0],
                positions[[i, j], 1],
                color=cmap(norm(value)),
                linewidth=0.5 + 5 * abs(value) / scale,
                alpha=0.88,
                zorder=1,
            )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=530,
            c="#f6f8fa",
            edgecolors="#738292",
            linewidths=1.2,
            zorder=2,
        )
        for name, (x, y) in zip(names, positions, strict=True):
            horizontal = "left" if x > 0.2 else "right" if x < -0.2 else "center"
            ax.text(x * 1.14, y * 1.18, name, ha=horizontal, va="center", fontsize=10)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.38, 1.38)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=12, pad=12)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(scalar, ax=axes, shrink=0.65, label="Signed mean SII | same scale in both arms")
    values_for_legend = np.array([0.2, 0.5, 1.0]) * scale
    handles = [
        Line2D([], [], color="#627386", linewidth=0.5 + 5 * value / scale, label=f"{value:,.1f}")
        for value in values_for_legend
    ]
    axes[0].legend(
        handles=handles,
        title="Width: |mean SII|",
        loc="lower center",
        bbox_to_anchor=(1.08, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        _title(title_prefix, f"Mean interaction networks | n = {len(base)}")
        + f"\nFixed union of each arm's top {k} pairs ({len(chosen)} pairs); matched nodes, widths and colors."
        "\nThese are output SII interactions, not input causal edges.",
        fontsize=12,
    )
    return _finish(fig, output_path)
