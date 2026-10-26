"""Display independent-seed validation for the targets that selected a case.

The selection reasons are fixed discovery records. Validation never changes
which targets appear, even when their changes fail to reproduce. Seed ranges
describe Monte Carlo variability, not confidence intervals over the dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _target_key(order, item):
    """Treat pair orientation as a label convention, not a different target."""
    if order not in (1, 2):
        raise ValueError("Target order must be 1 or 2.")
    if order == 1:
        return 1, str(item).strip()
    pair = [part.strip() for part in str(item).split("×")]
    if len(pair) != 2 or pair[0] == pair[1] or not all(pair):
        raise ValueError(f"Invalid interaction target label: {item!r}")
    return 2, tuple(sorted(pair))


def _case_frames(details, reasons, arm, row_id):
    details, reasons = details.copy(), reasons.copy()
    if "arm" not in details and "variant" in details:
        details["arm"] = details["variant"]
    if "arm" not in reasons and "variant" in reasons:
        reasons["arm"] = reasons["variant"]
    if "case_rule" not in reasons and "rule" in reasons:
        reasons["case_rule"] = reasons["rule"]
    for frame, required, name in (
        (
            details,
            {"arm", "row_id", "seed", "order", "item", "baseline", "modified", "delta"},
            "details",
        ),
        (reasons, {"order", "item", "case_rule", "baseline", "modified", "delta"}, "reasons"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    for column, value in (("arm", arm), ("row_id", row_id)):
        if value is not None:
            details = details[details[column] == value]
            if column in reasons:
                reasons = reasons[reasons[column] == value]
        elif column in reasons and reasons[column].nunique() == 1:
            details = details[details[column] == reasons[column].iloc[0]]
    cases = details[["arm", "row_id"]].drop_duplicates()
    if len(cases) != 1:
        raise ValueError("Select exactly one nonempty (arm, row_id) case.")
    case = cases.iloc[0]
    for column in ("arm", "row_id"):
        if column in reasons:
            reasons = reasons[reasons[column] == case[column]]
    if reasons.empty:
        raise ValueError("The selected case has no discovery selection reasons.")
    for frame in (details, reasons):
        if frame[["order", "item"]].isna().any().any():
            raise ValueError("Target order and item must not be missing.")
        for column in ("baseline", "modified", "delta"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy()).all():
                raise ValueError("Discovery and validation attributions must be finite.")
        if not np.allclose(frame.modified - frame.baseline, frame.delta, rtol=1e-9, atol=1e-9):
            raise ValueError("Delta must equal modified minus baseline.")
        frame["_target"] = [
            _target_key(order, item) for order, item in zip(frame.order, frame.item, strict=True)
        ]
    if details.seed.isna().any():
        raise ValueError("Validation seeds must not be missing.")
    return details, reasons, case


def summarize_target_validation(details, reasons, *, arm=None, row_id=None):
    """Summarize every selected target for one case without selecting anew.

    ``details`` has one row per (arm, row_id, seed, order, item), including
    baseline, modified, and delta; it may contain unselected targets as well.
    ``reasons`` supplies fixed discovery values and ``case_rule`` (or ``rule``).
    A target chosen by multiple rules is represented once with all its rules.

    Raw sign-flip counts always use the validation baseline and modified signs.
    Material-flip counts additionally require both endpoint magnitudes to meet
    ``endpoint_threshold`` (or ``sign_flip_threshold``). An ordinary delta
    ``threshold`` is never misinterpreted as an endpoint threshold. Counts and
    fractions are descriptive checks across seeds, not significance tests.
    """
    details, reasons, case = _case_frames(details, reasons, arm, row_id)
    rows, common_seeds = [], None
    for key in dict.fromkeys(reasons._target):
        selected = reasons[reasons._target.map(lambda value, target=key: value == target)]
        repeated = details[
            details._target.map(lambda value, target=key: value == target)
        ].sort_values("seed")
        discovery = selected.iloc[0]
        if repeated.empty:
            raise ValueError(f"Missing validation for selected target {discovery['item']!r}.")
        if repeated.seed.duplicated().any():
            raise ValueError(f"Duplicate target/seed rows for {discovery['item']!r}.")
        seeds = set(repeated.seed)
        if common_seeds is not None and seeds != common_seeds:
            raise ValueError("Selected targets must have the same validation seed set.")
        common_seeds = seeds
        for column in ("baseline", "modified", "delta"):
            if not np.allclose(selected[column], discovery[column], rtol=1e-9, atol=1e-9):
                raise ValueError("Rules for the same target must use identical discovery values.")
        baseline, modified, delta = (
            repeated[column].to_numpy() for column in ("baseline", "modified", "delta")
        )
        flips = (np.sign(baseline) * np.sign(modified)) < 0
        discovery_flip = np.sign(discovery.baseline) * np.sign(discovery.modified) < 0
        same_flip = (
            flips
            & (np.sign(baseline) == np.sign(discovery.baseline))
            & (np.sign(modified) == np.sign(discovery.modified))
        )
        direction_count = int(np.count_nonzero(np.sign(delta) == np.sign(discovery.delta)))
        endpoint_threshold = np.nan
        for column in ("endpoint_threshold", "sign_flip_threshold"):
            if column in selected:
                candidates = pd.to_numeric(selected[column], errors="raise").dropna().to_numpy()
                if not np.isfinite(candidates).all() or (candidates < 0).any():
                    raise ValueError("Endpoint thresholds must be finite and nonnegative.")
                if len(candidates):
                    endpoint_threshold = float(np.max(candidates))
                    break
        material = (
            flips
            & (np.abs(baseline) >= endpoint_threshold)
            & (np.abs(modified) >= endpoint_threshold)
        )
        row = {
            "arm": case.arm,
            "row_id": case.row_id,
            "date": discovery.get("date", repeated.iloc[0].get("date", "")),
            "order": int(discovery.order),
            "item": discovery["item"],
            "case_rules": " | ".join(dict.fromkeys(selected.case_rule.astype(str))),
            "discovery_baseline": float(discovery.baseline),
            "discovery_modified": float(discovery.modified),
            "discovery_delta": float(discovery.delta),
            "discovery_sign_flip": bool(discovery_flip),
            "n_seeds": len(repeated),
            "seeds": tuple(repeated.seed),
            "delta_values": tuple(float(value) for value in delta),
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
            "delta_q10": float(np.quantile(delta, 0.10)),
            "delta_q90": float(np.quantile(delta, 0.90)),
            "delta_direction_count": direction_count,
            "delta_direction_fraction": direction_count / len(repeated),
            "sign_flip_count": int(flips.sum()),
            "sign_flip_fraction": float(flips.mean()),
            "same_flip_direction_count": int(same_flip.sum()) if discovery_flip else 0,
            "same_flip_direction_fraction": float(same_flip.mean()) if discovery_flip else 0.0,
            "endpoint_threshold": endpoint_threshold,
            "material_sign_flip_count": int(material.sum())
            if np.isfinite(endpoint_threshold)
            else np.nan,
            "material_sign_flip_fraction": float(material.mean())
            if np.isfinite(endpoint_threshold)
            else np.nan,
        }
        for prefix, values, initial in (
            ("baseline", baseline, discovery.baseline),
            ("modified", modified, discovery.modified),
        ):
            row[f"{prefix}_positive_count"] = int((values > 0).sum())
            row[f"{prefix}_negative_count"] = int((values < 0).sum())
            row[f"{prefix}_zero_count"] = int((values == 0).sum())
            row[f"{prefix}_sign_agreement_count"] = int((np.sign(values) == np.sign(initial)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_target_validation(
    details, reasons, *, arm=None, row_id=None, title_prefix="", output_path=None
):
    """Plot all fixed selected targets once, with every seed and no target cap.

    The returned figure has ``target_validation_summary`` containing exact
    validation values and sign checks. Repeated rules share one target row.
    """
    summary = summarize_target_validation(details, reasons, arm=arm, row_id=row_id)
    count = len(summary)
    height = max(4.3, 2.1 + 0.55 * count)
    fig, ax = plt.subplots(figsize=(13.5, height))
    # Reserve physical space for the axis label and legend even for two targets.
    fig.subplots_adjust(left=0.23, right=0.71, bottom=1.0 / height, top=1.0 - 0.70 / height)
    ax.axvline(0, color="#a9b2bb", lw=1, zorder=0)
    for y, row in summary.iterrows():
        delta = np.asarray(row.delta_values)
        jitter = np.linspace(-0.12, 0.12, len(delta)) if len(delta) > 1 else np.zeros(1)
        color = "#356a9b" if row.order == 1 else "#775599"
        ax.scatter(delta, y + jitter, s=25, color=color, alpha=0.55, zorder=2)
        ax.plot([row.delta_q10, row.delta_q90], [y, y], lw=3, color=color, zorder=1)
        ax.scatter([row.delta_median], [y], marker="D", color=color, s=35, zorder=3)
        ax.scatter(
            [row.discovery_delta],
            [y],
            marker="*",
            color="#b48115",
            edgecolor="#725411",
            linewidth=0.5,
            s=120,
            zorder=4,
        )
        statistics = f"Direction {row.delta_direction_count}/{row.n_seeds} · sign flip {row.sign_flip_count}/{row.n_seeds}"
        if np.isfinite(row.endpoint_threshold):
            statistics += f"\nMaterial flip {int(row.material_sign_flip_count)}/{row.n_seeds} (both |values| ≥ {row.endpoint_threshold:,.1f})"
        ax.text(
            1.015,
            y,
            statistics,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color="#45515c",
        )
    labels = [
        f"{'Feature' if row.order == 1 else 'Pair'}: {row['item']}" for _, row in summary.iterrows()
    ]
    ax.set_yticks(np.arange(count), labels)
    ax.set_ylim(count - 0.4, -0.6)
    ax.set_xlabel("Attribution change: modified − baseline (model-output units)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.grid(axis="x", alpha=0.16)
    case = summary.iloc[0]
    date = f" | {case.date}" if str(case.date) else ""
    title = f"{case.arm} | row {case.row_id}{date} | validation of selected targets"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    fig.suptitle(title, y=0.98, fontsize=12)
    legend = [
        Line2D(
            [],
            [],
            color="#53606d",
            marker="o",
            linestyle="none",
            markersize=5,
            label="Individual validation seeds",
        ),
        Line2D(
            [],
            [],
            color="#53606d",
            marker="D",
            linewidth=3,
            markersize=5,
            label="Median and 10–90% seed range",
        ),
        Line2D(
            [],
            [],
            color="#b48115",
            marker="*",
            linestyle="none",
            markersize=10,
            label="Fixed discovery result",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.30 / height),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.07 / height,
        "Direction: same delta sign as discovery. Sign flip: baseline and modified have opposite signs. "
        "Seed ranges describe computation variability, not statistical significance.",
        ha="center",
        fontsize=8,
        color="#45515c",
    )
    fig.target_validation_summary = summary
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig
