"""Render an already selected/validated sample across structural variants.

This module never explains a new point, reselects cases within an arm, or edits
the pipeline manifest. Call ``render_common_case(output_root)`` only after the
main pipeline has completed; the caller then updates artifact hashes.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd


def _choose_shared_case(selected, arm_order):
    """Choose by frequency of *arms*, first from strong cases, then all roles."""
    for pool in ("strong", "any_frozen_role"):
        occurrences = defaultdict(list)
        for arm in arm_order:
            rows = selected[arm]
            if pool == "strong":
                rows = rows.loc[rows["case_rule"] == "strong"]
            for row_id in rows["row_id"].drop_duplicates():
                occurrences[int(row_id)].append(arm)
        shared = [(row_id, arms) for row_id, arms in occurrences.items() if len(arms) >= 2]
        if shared:
            row_id, eligible_arms = min(shared, key=lambda pair: (-len(pair[1]), pair[0]))
            arms = eligible_arms[:4]
            roles = {}
            for arm in arms:
                rows = selected[arm].loc[selected[arm]["row_id"] == row_id]
                if pool == "strong":
                    rows = rows.loc[rows["case_rule"] == "strong"]
                roles[arm] = rows["case_rule"].drop_duplicates().tolist()
            return {
                "row_id": row_id,
                "pool": pool,
                "arms": arms,
                "eligible_arms": eligible_arms,
                "original_roles": roles,
            }
    return None


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_common_case(output_root):
    """Return paths/layout, or ``None`` with a warning if no shared case exists.

    The most frequent source row among frozen ``strong`` cases is preferred;
    ties use ascending row ID. If no strong case is shared, repeat over all
    frozen roles. Show at most four eligible arms in structural-catalog order.
    Each displayed arm must already have independent-seed validation records.

    Plotted values are the original discovery-seed attributions. Validation
    establishes that a separate seed check was performed, not significance.
    """
    root = Path(output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not manifest.get("completed", False):
        raise RuntimeError("Render the common case only after the full pipeline completes.")
    selection_path = root / "selection_manifest.json"
    selection_hash = _hash(selection_path)
    if manifest.get("selection_sha256") != selection_hash:
        raise ValueError("The completed manifest and frozen selection hash disagree.")
    frozen = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    catalog = pd.read_csv(root / "tables" / "structural_catalog.csv")
    catalog = catalog.loc[catalog["variant"] != "baseline"]
    arm_order = catalog["variant"].tolist()
    selected = {}
    for arm in arm_order:
        frame = pd.read_csv(root / "tables" / f"{arm}_selected.csv")
        frozen_rows = frozen["selection"][arm]
        expected = sorted((int(row["row_id"]), row["case_rule"]) for row in frozen_rows)
        actual = sorted((int(row.row_id), row.case_rule) for row in frame.itertuples())
        if actual != expected:
            raise ValueError(f"Selected CSV differs from frozen cases for {arm}.")
        selected[arm] = frame
    chosen = _choose_shared_case(selected, arm_order)
    if chosen is None:
        warnings.warn(
            "No source row is already a frozen case in at least two structural arms; common-case rendering omitted.",
            stacklevel=2,
        )
        return None

    row_id, arms = chosen["row_id"], chosen["arms"]
    feature_names = None
    base_phi, base_interaction = None, None
    modified, dates, provenance = [], set(), {}
    for arm in arms:
        matching = selected[arm].loc[selected[arm]["row_id"] == row_id]
        dates.update(str(date)[:10] for date in matching["date"])
        feature_table = pd.read_csv(root / "tables" / f"{arm}_features.csv")
        names = feature_table.loc[feature_table["row_id"] == row_id, "item"].tolist()
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Feature labels/order differ across arms.")
        validation_path = root / "tables" / f"{arm}_validation_summary.csv"
        validation = pd.read_csv(validation_path)
        validation = validation.loc[
            (validation["row_id"] == row_id)
            & validation["case_rule"].isin(chosen["original_roles"][arm])
        ]
        if set(validation["case_rule"]) != set(chosen["original_roles"][arm]):
            raise ValueError(f"Independent-seed validation is incomplete for shared case in {arm}.")
        if (validation["n_validation_seeds"] != len(frozen["validation_seeds"])).any():
            raise ValueError(f"Unexpected validation-seed count for {arm}.")
        draws_path = root / "draws" / f"{arm}.npz"
        with np.load(draws_path, allow_pickle=False) as draws:
            positions = np.flatnonzero(draws["row_ids"] == row_id)
            if len(positions) != 1:
                raise ValueError(f"Source row must occur exactly once in {arm} draws.")
            position = int(positions[0])
            phi0 = draws["base_phi"][position].copy()
            int0 = draws["base_interaction"][position].copy()
            phi1 = draws["variant_phi"][position].copy()
            int1 = draws["variant_interaction"][position].copy()
        if base_phi is None:
            base_phi, base_interaction = phi0, int0
        elif not (np.array_equal(base_phi, phi0) and np.array_equal(base_interaction, int0)):
            raise ValueError("A shared baseline is required across all displayed arms.")
        modified.append((arm, phi1, int1))
        provenance[arm] = {
            "original_roles": chosen["original_roles"][arm],
            "discovery_percentile": float(matching.iloc[0]["effect_score_percentile"]),
            "selected_csv_sha256": _hash(root / "tables" / f"{arm}_selected.csv"),
            "draws_sha256": _hash(draws_path),
            "validation_summary_sha256": _hash(validation_path),
            "n_validation_seeds": len(frozen["validation_seeds"]),
        }
    if len(dates) != 1:
        raise ValueError("Shared source row has inconsistent target dates across arms.")
    if feature_names is None or base_phi is None or base_interaction is None:
        raise ValueError("No shared attribution data loaded.")
    panels = [("baseline", base_phi, base_interaction)] + modified
    d = len(feature_names)
    pair_indices = list(combinations(range(d), 2))
    for _, phi, interaction in panels:
        if phi.shape != (d,) or interaction.shape != (len(pair_indices),):
            raise ValueError("Attribution dimensions differ from feature labels.")
        if not np.isfinite(phi).all() or not np.isfinite(interaction).all():
            raise ValueError("Attribution arrays contain non-finite values.")
    phi_limit = max(float(np.max(np.abs(phi))) for _, phi, _ in panels) * 1.08
    interaction_limit = max(float(np.max(np.abs(inter))) for _, _, inter in panels)
    phi_limit, interaction_limit = max(phi_limit, 1e-12), max(interaction_limit, 1e-12)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(panels), figsize=(4.2 * len(panels), 9.0), constrained_layout=True
    )
    labels = catalog.set_index("variant")["label"].to_dict()
    for col, (arm, phi, interaction) in enumerate(panels):
        ax = axes[0, col]
        ax.barh(np.arange(d), phi, color=np.where(phi >= 0, "#bb3d3d", "#346ea8"))
        ax.set_yticks(np.arange(d), feature_names)
        ax.invert_yaxis()
        ax.set_xlim(-phi_limit, phi_limit)
        ax.axvline(0, color="0.45", linewidth=0.7)
        ax.set_xlabel("First-order attribution")
        title = "Baseline" if arm == "baseline" else f"{arm}: {labels[arm]}"
        roles = (
            "shared reference" if arm == "baseline" else ", ".join(chosen["original_roles"][arm])
        )
        ax.set_title(fill(title, 31) + "\n" + roles, fontsize=10)
        matrix = np.full((d, d), np.nan)
        for value, (i, j) in zip(interaction, pair_indices, strict=True):
            matrix[i, j] = matrix[j, i] = value
        ax = axes[1, col]
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-interaction_limit, vmax=interaction_limit)
        ax.set_xticks(np.arange(d), feature_names, rotation=50, ha="right", fontsize=8)
        ax.set_yticks(np.arange(d), feature_names, fontsize=8)
        ax.set_title("Output SII interactions", fontsize=10)
        for i in range(d):
            for j in range(d):
                if i != j:
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if abs(matrix[i, j]) > interaction_limit * 0.65 else "black",
                    )
    fig.colorbar(image, ax=list(axes[1]), shrink=0.65, label="SII attribution units")
    date = next(iter(dates))
    pool_label = (
        "shared frozen strong cases"
        if chosen["pool"] == "strong"
        else "shared frozen cases (roles shown)"
    )
    fig.suptitle(
        f"Source row {row_id} | {date} | {pool_label}\n"
        f"Discovery seed {frozen['discovery_seed']}; each displayed case has separate checks with "
        f"{len(frozen['validation_seeds'])} new seeds.\n"
        "All panels share scales within order; output SII interactions are not input causal edges.",
        fontsize=12,
    )
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png, pdf = figures / "common_case_comparison.png", figures / "common_case_comparison.pdf"
    try:
        fig.savefig(png, dpi=180, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
    finally:
        plt.close(fig)
    layout = {
        "row_id": row_id,
        "date": date,
        "arms": arms,
        "eligible_arms_in_catalog_order": chosen["eligible_arms"],
        "pool": chosen["pool"],
        "original_roles": chosen["original_roles"],
        "selection_rule": "Most arms sharing a frozen strong case; if fewer than two, most arms sharing any frozen case. "
        "Tie: ascending row ID. Display first four arms in catalog order.",
        "selection_manifest_sha256": selection_hash,
        "discovery_seed": frozen["discovery_seed"],
        "validation_seeds": frozen["validation_seeds"],
        "feature_names": feature_names,
        "first_order_xlim": [-phi_limit, phi_limit],
        "interaction_clim": [-interaction_limit, interaction_limit],
        "panels_use": "Discovery draws; independent validation summaries checked for every displayed arm and role.",
        "arm_provenance": provenance,
        "figure_paths": [str(png.relative_to(root)), str(pdf.relative_to(root))],
    }
    layout_path = root / "common_case_layout.json"
    layout_path.write_text(
        json.dumps(layout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"png": png, "pdf": pdf, "layout_path": layout_path, "layout": layout}
