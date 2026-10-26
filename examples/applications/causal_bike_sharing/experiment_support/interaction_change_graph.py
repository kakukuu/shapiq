"""Directly annotated baseline / modified / change SII interaction networks.

Pair columns always follow ``combinations(range(n_features), 2)``. These are
output interactions, not input causal edges. Matrix inputs summarize signed
means: a change of signed means is not the mean absolute sample change.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

_CMAP = LinearSegmentedColormap.from_list("interaction_change", ["#356a9b", "#f3f3f2", "#b7664e"])
_FOCUS_COLOR = "#b48115"


def _focus_indices(names, focus_features, focus_pairs):
    """Normalize selected targets without silently dropping invalid labels."""
    feature_set = set(() if focus_features is None else focus_features)
    unknown = feature_set.difference(names)
    if unknown:
        raise ValueError(f"Unknown focus features: {sorted(unknown)}")
    pair_lookup = {pair: index for index, pair in enumerate(combinations(range(len(names)), 2))}
    pair_indices = set()
    for pair in () if focus_pairs is None else focus_pairs:
        parts = [part.strip() for part in pair.split("×")] if isinstance(pair, str) else list(pair)
        if len(parts) != 2 or parts[0] == parts[1] or any(part not in names for part in parts):
            raise ValueError(f"Invalid focus pair: {pair!r}")
        indices = tuple(sorted(names.index(part) for part in parts))
        pair_indices.add(pair_lookup[indices])
    return feature_set, pair_indices


def _validate(base_phi, variant_phi, base_interactions, variant_interactions, feature_names):
    names = list(feature_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique feature names are required.")
    arrays = [
        np.asarray(a, dtype=float)
        for a in (base_phi, variant_phi, base_interactions, variant_interactions)
    ]
    ndim = arrays[0].ndim
    if ndim not in (1, 2) or any(a.ndim != ndim for a in arrays):
        raise ValueError("Supply four row vectors, or four paired row matrices.")
    d = len(names)
    widths = [d, d, d * (d - 1) // 2, d * (d - 1) // 2]
    n = arrays[0].shape[0] if ndim == 2 else 1
    for array, width in zip(arrays, widths, strict=True):
        shape = (n, width) if ndim == 2 else (width,)
        if array.shape != shape or n == 0 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite paired input with shape {shape}.")
    return [a.reshape(n, width) for a, width in zip(arrays, widths, strict=True)], names, ndim


def _summaries(base, modified):
    left, right = base.mean(axis=0), modified.mean(axis=0)
    delta = right - left
    magnitude = np.abs(right) - np.abs(left)
    equal = np.isclose(magnitude, 0.0, rtol=1e-10, atol=1e-10)
    return dict(
        baseline=left,
        modified=right,
        delta=delta,
        magnitude_change=magnitude,
        mean_absolute_sample_change=np.abs(modified - base).mean(axis=0),
        mean_sample_magnitude_change=(np.abs(modified) - np.abs(base)).mean(axis=0),
        magnitude_direction=np.where(
            equal, "unchanged", np.where(magnitude > 0, "stronger", "weaker")
        ),
        sign_flip=((left < 0) & (right > 0)) | ((left > 0) & (right < 0)),
    )


def summarize_change_graph(
    base_phi,
    variant_phi,
    base_interactions,
    variant_interactions,
    feature_names,
    *,
    top_k=3,
    focus_features=None,
    focus_pairs=None,
):
    """Return exact values for all nodes / pairs, including unshown pairs.

    ``magnitude_change`` is abs(modified signed mean) minus abs(baseline
    signed mean). The distinct ``mean_sample_magnitude_change`` and
    ``mean_absolute_sample_change`` columns preserve sample-level estimands.
    A sign flip refers to nonzero signed means, or nonzero single-row values.
    Ties in top-k selection use the input pair order deterministically.
    Explicit focus pairs are always selected, including exactly zero pairs.
    They do not change the top-k flags or discard any of the full pair table.
    """
    arrays, names, ndim = _validate(
        base_phi, variant_phi, base_interactions, variant_interactions, feature_names
    )
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    nodes = pd.DataFrame({"feature": names, **_summaries(*arrays[:2])})
    feature_focus, pair_focus = _focus_indices(names, focus_features, focus_pairs)
    nodes["is_focus"] = nodes.feature.isin(feature_focus)
    pairs = list(combinations(range(len(names)), 2))
    edges = pd.DataFrame(
        {
            "pair_index": np.arange(len(pairs)),
            "source": [names[i] for i, j in pairs],
            "target": [names[j] for i, j in pairs],
            **_summaries(*arrays[2:]),
        }
    )
    k = min(int(top_k), len(edges))
    selected = set()
    for column in ("baseline", "modified", "delta"):
        order = np.argsort(-np.abs(edges[column].to_numpy()), kind="stable")[:k]
        # Exclude exactly zero entries from the top lists: a zero-weight pair
        # must not be fabricated merely to reach the requested count.
        chosen = [int(i) for i in order if edges.loc[i, column] != 0]
        edges[f"top_{column}"] = edges.index.isin(chosen)
        selected.update(chosen)
    edges["is_focus"] = edges.index.isin(pair_focus)
    edges["selected"] = edges.index.isin(selected | pair_focus)
    return {
        "nodes": nodes,
        "edges": edges,
        "n": len(arrays[0]),
        "is_mean": ndim == 2,
        "top_k": k,
        "pairs": pairs,
        "feature_names": names,
    }


def _curve(points, pair):
    start, stop = points[list(pair)]
    chord = stop - start
    normal = np.array([-chord[1], chord[0]]) / max(np.linalg.norm(chord), 1e-12)
    # Fixed pair-specific gentle bends keep the geometry identical across all
    # arms, samples and means and avoid piling every chord at the origin.
    bend = (0.12 if (pair[0] + pair[1]) % 2 else -0.12) * np.linalg.norm(chord)
    control = (start + stop) / 2 + bend * normal
    return np.array([start, control, stop])


def _on_curve(vertices, t):
    return (1 - t) ** 2 * vertices[0] + 2 * t * (1 - t) * vertices[1] + t**2 * vertices[2]


def _label_positions(curves, points):
    """Assign distinct fixed label positions along edges without hiding nodes."""
    locations = {}
    candidates = (0.50, 0.36, 0.64, 0.25, 0.75, 0.43, 0.57)
    # Fewer alternatives on short peripheral edges; place them first.
    order = sorted(curves, key=lambda i: np.linalg.norm(curves[i][2] - curves[i][0]))
    for index in order:
        options = [_on_curve(curves[index], t) for t in candidates]

        def clearance(point):
            separation = [
                np.linalg.norm((point - other) * [1.0, 1.65]) for other in locations.values()
            ]
            separation.extend(np.linalg.norm(point - node) - 0.13 for node in points)
            return min(separation)

        locations[index] = max(options, key=clearance)
    return locations


def _number(value, decimals):
    # Avoid printing -0.0 when the rounded label is zero.
    value = 0.0 if round(float(value), decimals) == 0 else float(value)
    return f"{value:+,.{decimals}f}"


def _scale(values):
    return max(float(np.max(np.abs(values))), 1e-12)


def plot_interaction_change_graph(
    base_phi,
    variant_phi,
    base_interactions,
    variant_interactions,
    feature_names,
    *,
    title_prefix="",
    baseline_label="Baseline",
    variant_label="Modified",
    scope_label="",
    top_k=3,
    decimals=1,
    output_path=None,
    focus_features=None,
    focus_pairs=None,
):
    """Plot paired networks with direct numeric node and interaction changes.

    Defaults to the union of each arm's three largest absolute interactions
    and the three largest absolute changes, on all three panels. Use
    ``top_k=21`` to display every nonzero pair for seven features (crowded).
    Baseline and modified share edge and node scales; difference has its own
    explicitly reported scales. Node colors / areas encode first-order phi;
    edge colors / widths encode pairwise SII. Dashed edges mark sign flips;
    bold delta labels identify the top-k absolute signed changes. Every edge
    gets its exact signed numeric label (rounded only for display).
    Explicit focus features receive gold outlines; focus pair labels receive
    gold frames and remain visible even below the top-k cutoff. An exactly
    zero focus pair has a numeric label but no stroke encoding false strength.

    The full unrounded summary is attached as ``fig.change_graph_summary``;
    no samples, pairs or attributions are discarded from that summary.
    """
    summary = summarize_change_graph(
        base_phi,
        variant_phi,
        base_interactions,
        variant_interactions,
        feature_names,
        top_k=top_k,
        focus_features=focus_features,
        focus_pairs=focus_pairs,
    )
    if not isinstance(decimals, (int, np.integer)) or not 0 <= decimals <= 5:
        raise ValueError("decimals must be an integer from 0 through 5.")
    nodes, edges = summary["nodes"], summary["edges"]
    names, pairs = summary["feature_names"], summary["pairs"]
    d = len(names)
    angle = np.pi / 2 - np.arange(d) * 2 * np.pi / d
    points = np.column_stack([np.cos(angle), np.sin(angle)])
    selected = edges.index[edges.selected].tolist()
    curves = {i: _curve(points, pairs[i]) for i in selected}
    label_xy = _label_positions(curves, points)
    phi_scale = _scale(nodes[["baseline", "modified"]].to_numpy())
    edge_scale = _scale(edges[["baseline", "modified"]].to_numpy())
    dphi_scale, dedge_scale = _scale(nodes.delta), _scale(edges.delta)
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 8.2))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.79, bottom=0.23, wspace=0.08)
    for panel, (ax, column, label) in enumerate(
        zip(
            axes,
            ("baseline", "modified", "delta"),
            (baseline_label, variant_label, "Change: modified − baseline"),
            strict=True,
        )
    ):
        node_limit = dphi_scale if panel == 2 else phi_scale
        edge_limit = dedge_scale if panel == 2 else edge_scale
        norm = Normalize(-edge_limit, edge_limit)
        for index in sorted(selected, key=lambda i: abs(edges.loc[i, column])):
            row, vertices = edges.loc[index], curves[index]
            value = row[column]
            width = 0.7 + 4.5 * abs(value) / edge_limit
            patch = PathPatch(
                MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
                facecolor="none",
                edgecolor=_CMAP(norm(value)),
                linewidth=width,
                linestyle="--" if row.sign_flip else "-",
                zorder=1,
                alpha=0.93,
            )
            patch.set_gid(f"interaction-pair-{index}")
            if row.is_focus and value == 0:
                patch.set_linewidth(0.0)
            ax.add_patch(patch)
        values = nodes[column].to_numpy()
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=180 + 570 * np.abs(values) / node_limit,
            c=values,
            cmap=_CMAP,
            vmin=-node_limit,
            vmax=node_limit,
            edgecolors=[_FOCUS_COLOR if focus else "#53606d" for focus in nodes.is_focus],
            linewidths=np.where(nodes.is_focus, 3.0, 1.1),
            zorder=3,
        )
        scatter.set_gid("first-order-attributions")
        for i, name in enumerate(names):
            x, y = points[i]
            label_pos = np.array([x, y]) * 1.23
            alignment = "left" if x > 0.35 else "right" if x < -0.35 else "center"
            marker = "* " if nodes.is_focus.iloc[i] else ""
            text = f"{marker}{name}\n{'Δφ' if panel == 2 else 'φ'} {_number(values[i], decimals)}"
            ax.text(
                *label_pos,
                text,
                ha=alignment,
                va="center",
                fontsize=10.1,
                linespacing=1.5,
                color="#202c36",
                zorder=5,
            )
        for index in selected:
            row = edges.loc[index]
            text = _number(row[column], decimals)
            if panel == 2:
                direction = {"stronger": "|I| ↑", "weaker": "|I| ↓", "unchanged": "|I| ="}
                suffix = " · flip" if row.sign_flip else ""
                text = f"ΔI {text}\n{direction[row.magnitude_direction]}{suffix}"
            if row.is_focus:
                text = "* " + text
            ax.text(
                *label_xy[index],
                text,
                ha="center",
                va="center",
                fontsize=8.7 if panel == 2 else 9.2,
                linespacing=1.15,
                weight="bold" if row.is_focus or (panel == 2 and row.top_delta) else "normal",
                color="#202c36",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=.17",
                    facecolor="white",
                    edgecolor=_FOCUS_COLOR if row.is_focus else "#d4dce1",
                    linewidth=1.8 if row.is_focus else 0.55,
                    alpha=0.95,
                ),
            )
        ax.set_title(label, fontsize=13, fontweight="semibold", pad=14)
        ax.set_aspect("equal")
        ax.set_xlim(-1.72, 1.72)
        ax.set_ylim(-1.50, 1.50)
        ax.set_axis_off()
        phi_text, i_text = ("Δφ", "ΔI") if panel == 2 else ("φ", "I")
        scale_note = "separate change scales" if panel == 2 else "shared arm scales"
        ax.text(
            0.5,
            -0.025,
            f"{scale_note}: |{phi_text}| max {node_limit:,.1f}; |{i_text}| max {edge_limit:,.1f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#485561",
        )
    scope = (
        f"Signed means over n = {summary['n']} paired rows"
        if summary["is_mean"]
        else "One paired sample"
    )
    if scope_label:
        scope = f"{scope_label} | {scope}"
    title = "Direct attribution and interaction changes"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    fig.suptitle(title + "\n" + scope, fontsize=14, y=0.985, linespacing=1.6)
    focus_note = " + selected focus pairs" if edges.is_focus.any() else ""
    fig.text(
        0.5,
        0.855,
        f"Fixed union: top {summary['top_k']} |baseline I|, |modified I| and |ΔI|{focus_note} "
        f"({len(selected)} of {len(edges)} pairs shown). Identical node and edge positions.",
        ha="center",
        fontsize=10,
        color="#45515c",
    )
    legend = [
        Line2D([], [], color="#b7664e", linewidth=3, label="Positive signed value"),
        Line2D([], [], color="#356a9b", linewidth=3, label="Negative signed value"),
        Line2D(
            [], [], color="#65717b", linewidth=2, linestyle="--", label="Sign flip between arms"
        ),
    ]
    if nodes.is_focus.any() or edges.is_focus.any():
        legend.append(
            Line2D(
                [],
                [],
                color=_FOCUS_COLOR,
                linewidth=2,
                label="* Gold frame: selected explanatory target",
            )
        )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=len(legend),
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.083,
        "Nodes: first-order φ (area grows with |φ|). Edges: pairwise SII I (width grows with |I|). "
        "Bold ΔI labels: top absolute signed changes. All values in model-output units.",
        ha="center",
        fontsize=9.5,
        color="#45515c",
    )
    strength_note = (
        "|I| arrows compare magnitudes of signed means, not mean row-wise magnitudes."
        if summary["is_mean"]
        else "|I| arrows compare interaction magnitudes; positive ΔI can still weaken a negative interaction."
    )
    fig.text(
        0.5,
        0.054,
        strength_note + "  ΔI = I modified − I baseline.",
        ha="center",
        fontsize=9.5,
        color="#45515c",
    )
    fig.text(
        0.5,
        0.024,
        "Output interaction edges are not input causal edges. "
        "Descriptive comparison; omitted pairs remain in the full numeric table.",
        ha="center",
        fontsize=9.5,
        color="#45515c",
    )
    fig.change_graph_summary = summary
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig
