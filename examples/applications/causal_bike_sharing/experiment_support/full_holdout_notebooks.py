"""Install and render saved full-holdout results without rerunning explanations.

Only the explicitly marked extension cells are owned by this module. Historical
notebook sources and outputs are retained, including their original numbering.
Call ``install_extensions()`` after the full-holdout artifact run completes, then
execute the new render cells. The renderer never trains or explains a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = "results"
NOTEBOOK_ARMS = ("a", "b1", "b2", "b3", "lag")
STRUCTURAL_ARMS = (
    "b1",
    "b2",
    "b3",
    "temp_root",
    "hum_root",
    "windspeed_root",
    "b3_unconfounded",
    "b2_unconfounded",
)


def _artifact_root(root=ROOT):
    """Resolve an experiment directory to its completed artifact directory."""
    return Path(root).resolve() / ARTIFACT_DIRECTORY


def _run_instruction(root):
    return (
        "Run run_full_holdout.py from the experiments directory first and wait "
        "for it to finish successfully. Expected artifacts under "
        f"{_artifact_root(root)}. This cell does not recompute missing results."
    )


def _completed_manifest(root=ROOT):
    artifact_root = _artifact_root(root)
    path = artifact_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. {_run_instruction(root)}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("The artifact manifest must be a JSON object.")
    if manifest.get("completed") is not True:
        raise RuntimeError(
            f"The artifact manifest is not marked completed=True. {_run_instruction(root)}"
        )
    if "artifacts_sha256" in manifest and not isinstance(manifest["artifacts_sha256"], dict):
        raise ValueError("manifest.artifacts_sha256 must be a path-to-SHA256 mapping.")
    return artifact_root, manifest


def _verify_artifact(path, root):
    """Verify every displayed export against the completed run when hashes exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. {_run_instruction(root)}")
    artifact_root, manifest = _completed_manifest(root)
    hashes = manifest.get("artifacts_sha256")
    if hashes is None:
        return
    relative = path.resolve().relative_to(artifact_root).as_posix()
    # Manifests written on Windows use backslashes; both encodings describe
    # the same relative artifact, and remain readable after export elsewhere.
    normalized = {str(name).replace("\\", "/"): value for name, value in hashes.items()}
    if relative not in normalized:
        raise ValueError(f"Artifact is not recorded in the completed manifest: {relative}")
    expected = normalized[relative]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Invalid SHA256 in artifact manifest: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Artifact SHA256 mismatch: {relative}. {_run_instruction(root)}")


def _csv(path, root):
    import pandas as pd

    _verify_artifact(path, root)
    return pd.read_csv(path)


def _markdown(text):
    from IPython.display import Markdown, display

    display(Markdown(text))


def _table(frame, title):
    """Print every row and column; notebook frontend truncation is separate."""
    _markdown(f"**{title}** — {len(frame):,} rows")
    print(frame.to_string(index=False, max_rows=None, max_cols=None, max_colwidth=None))


def _summary_for_arm(summary, arm):
    for column in ("arm", "variant", "experiment", "comparison", "name"):
        if column in summary:
            values = summary[column].astype(str).str.lower()
            selected = summary.loc[values.eq(arm.lower())]
            if not selected.empty:
                return selected
    # Retaining all rows is preferable to silently attributing another arm's
    # statistics to this comparison when the export schema changes.
    raise ValueError(f"aggregate_summary.csv has no identifiable row for {arm!r}")


def _image(path, title, root):
    from IPython.display import Image, display

    _verify_artifact(path, root)
    _markdown(f"**{title}**")
    display(Image(filename=str(path)))


def _detail_links(artifact_root, arm):
    names = [f"{arm}_features.csv", f"{arm}_interactions.csv"]
    links = ", ".join(f"[{name}]({ARTIFACT_DIRECTORY}/tables/{name})" for name in names)
    _markdown(
        "The complete feature and interaction detail tables are exported as "
        f"{links}. They retain the point identifier for joining to the overview."
    )
    for name in names:
        _verify_artifact(artifact_root / "tables" / name, artifact_root.parent)


def _selected_images(artifact_root, arm, selected, root):
    """Resolve only current frozen selections, preserving their recorded order.

    A rerun can leave earlier case files beside current exports. Never discover
    the selection from filenames or directory globs: the saved selected table
    is authoritative, and its contents are checked before this function runs.
    """
    if not {"case_rule", "row_id"}.issubset(selected.columns) or selected.empty:
        raise ValueError(f"{arm}: selected table must contain case_rule and row_id rows.")
    images = {"case": [], "validation": []}
    for row in selected.itertuples(index=False):
        rule = str(row.case_rule)
        if not re.fullmatch(r"[A-Za-z0-9_]+", rule):
            raise ValueError(f"Invalid selected case rule: {rule!r}")
        if not float(row.row_id).is_integer():
            raise ValueError(f"Invalid selected original row ID: {row.row_id!r}")
        row_id = int(row.row_id)
        for kind in images:
            path = artifact_root / "figures" / f"{arm}_{kind}_{rule}_{row_id}.png"
            _verify_artifact(path, root)
            images[kind].append(path)
    return images


def _current_common_case(root, manifest):
    """Accept a shared-case image only when its inputs belong to this selection.

    A normal compute rerun can hash an old optional image without regenerating
    its layout. The separate figure-export step records the provenance needed
    to distinguish that file from the current frozen-case comparison.
    """
    layout = manifest.get("common_frozen_case")
    selection_hash = manifest.get("selection_sha256")
    if not isinstance(layout, dict) or not isinstance(selection_hash, str):
        return False
    if layout.get("selection_manifest_sha256") != selection_hash:
        return False
    hashes = {
        str(name).replace("\\", "/"): value
        for name, value in manifest.get("artifacts_sha256", {}).items()
    }
    if "figures/common_case_comparison.png" not in hashes:
        return False
    arms = layout.get("arms")
    provenance = layout.get("arm_provenance")
    if (
        not isinstance(arms, list)
        or len(arms) < 2
        or any(not isinstance(arm, str) or arm not in STRUCTURAL_ARMS for arm in arms)
        or len(set(arms)) != len(arms)
        or not isinstance(provenance, dict)
    ):
        return False
    references = []
    for arm in arms:
        record = provenance.get(arm)
        if not isinstance(record, dict):
            return False
        for field, relative in (
            ("selected_csv_sha256", f"tables/{arm}_selected.csv"),
            ("draws_sha256", f"draws/{arm}.npz"),
            ("validation_summary_sha256", f"tables/{arm}_validation_summary.csv"),
        ):
            recorded = record.get(field)
            if not isinstance(recorded, str) or recorded != hashes.get(relative):
                return False
            references.append(relative)
    artifact_root = _artifact_root(root)
    selection_path = artifact_root / "selection_manifest.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing {selection_path}. {_run_instruction(root)}")
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != selection_hash:
        raise ValueError("Frozen selection manifest SHA256 mismatch.")
    for relative in references:
        _verify_artifact(artifact_root / relative, root)
    return True


def _scope_note(arm):
    if arm == "lag":
        return (
            "**New paired full holdout: 146 target dates.** Same-day and lagged "
            "arms use the same held-out target dates and split. Their feature "
            "semantics and fitted models differ, so this is a temporal-input "
            "comparison, not an isolated graph perturbation. Earlier 50-row "
            "outputs and the following legacy Summary retain their historical "
            "unmatched-sample interpretation."
        )
    note = (
        "**New full holdout: all 147 same-day test rows.** Comparisons use the "
        "same fitted model, background data and held-out input rows. Earlier "
        "50-row outputs and the following legacy Summary remain historical "
        "results. Full-holdout tables and figures use a separate FH numbering."
    )
    if arm == "a":
        note += (
            " This arm compares causal and marginal imputation; marginal "
            "imputation is not conditional observational imputation."
        )
    if arm == "b3":
        note += (
            " The structure-sweep notebook also reports B3 with its merged "
            "layer's confounding flag disabled (b3_unconfounded)."
        )
    return note


def render_extension(arm, root=ROOT):
    """Print one arm's saved tables and display its figures in an existing notebook.

    ``root`` is the experiments directory, primarily exposed for reproducible
    verification in an isolated test directory. A completed manifest and every
    requested artifact are required; missing files never trigger computation.
    """
    if arm not in NOTEBOOK_ARMS:
        raise ValueError(f"Unknown notebook arm {arm!r}; expected {NOTEBOOK_ARMS}")
    artifact_root, _ = _completed_manifest(root)
    tables = artifact_root / "tables"
    figures = artifact_root / "figures"
    prefix = arm.upper()
    _markdown(_scope_note(arm))
    _markdown(
        "Signed change describes direction; absolute change describes "
        "displacement without cancellation. Neither a large absolute change "
        "nor an extreme selected example alone establishes statistical "
        "significance. Selection is exploratory; fresh-seed validation is "
        "conditional on the fitted model, background and selected inputs."
    )

    points = _csv(tables / f"{arm}_points.csv", root)
    expected_rows = 146 if arm == "lag" else 147
    if len(points) != expected_rows:
        raise ValueError(f"{arm}: expected {expected_rows} full-holdout rows, found {len(points)}")
    _table(points, f"Table {prefix}-FH1. Complete per-point overview")
    _detail_links(artifact_root, arm)
    aggregate = _csv(tables / "aggregate_summary.csv", root)
    _table(
        _summary_for_arm(aggregate, arm),
        f"Table {prefix}-FH2. Full-holdout aggregate statistics",
    )
    selected = None
    for number, suffix, title in (
        (3, "selection_rules", "Frozen case-selection rules"),
        (4, "selected", "Selected examples and selection context"),
        (5, "validation_summary", "Fresh-seed validation summaries"),
        (6, "validation", "All selected-case validation repeats"),
    ):
        frame = _csv(tables / f"{arm}_{suffix}.csv", root)
        if suffix == "selected":
            selected = frame
        _table(frame, f"Table {prefix}-FH{number}. {title}")

    images = _selected_images(artifact_root, arm, selected, root)
    ordered = [(figures / f"{arm}_averages.png", "Full-holdout average changes")]
    ordered += [(path, f"Selected example: {path.stem}") for path in images["case"]]
    ordered += [(path, f"Fresh-seed validation: {path.stem}") for path in images["validation"]]
    for number, (path, title) in enumerate(ordered, start=1):
        _image(path, f"Figure {prefix}-FH{number}. {title}", root)


def _sweep_arms(catalog):
    """Obtain structural comparison names without treating baseline as an arm."""
    for column in ("arm", "variant", "name", "id", "key"):
        if column in catalog:
            candidates = catalog[column].astype(str).str.lower().tolist()
            arms = [name for name in candidates if name in STRUCTURAL_ARMS]
            if arms:
                return list(dict.fromkeys(arms))
    return list(STRUCTURAL_ARMS)


def render_sweep(root=ROOT, section="all"):
    """Render saved structural results: all, catalog, averages, points, or cases."""
    sections = ("catalog", "averages", "points", "cases")
    if section not in ("all",) + sections:
        raise ValueError(f"Unknown sweep section {section!r}")
    artifact_root, manifest = _completed_manifest(root)
    tables = artifact_root / "tables"
    figures = artifact_root / "figures"
    catalog = _csv(tables / "structural_catalog.csv", root)
    arms = _sweep_arms(catalog)

    if section in ("all", "catalog"):
        _markdown(
            "**Exploratory structural perturbations relative to an assumed "
            "baseline.** Layer moves can change several ancestor relationships, "
            "conditioning sets and joint-sampling groups simultaneously. They "
            "are not single-edge interventions or verified causal errors. "
            "Input-structure arrows and output interaction edges have different "
            "meanings. The model, background and 147 test rows are shared."
        )
        _table(catalog, "Table S-FH1. Complete structural configuration catalog")
        _image(
            figures / "input_structures.png",
            "Figure S-FH1. Input ordering and confounding configurations",
            root,
        )
    if section in ("all", "averages"):
        aggregate = _csv(tables / "aggregate_summary.csv", root)
        import pandas as pd

        rows = pd.concat([_summary_for_arm(aggregate, arm) for arm in arms], ignore_index=True)
        _table(rows, "Table S-FH2. Average and distribution summaries by perturbation")
        _image(
            figures / "structural_averages.png",
            "Figure S-FH2. Average changes across structural perturbations",
            root,
        )
        _image(
            figures / "sample_by_variant.png",
            "Figure S-FH3. Per-point sensitivity across structural perturbations",
            root,
        )
    if section in ("all", "points"):
        import pandas as pd

        frames = []
        for arm in arms:
            frame = _csv(tables / f"{arm}_points.csv", root)
            if len(frame) != 147:
                raise ValueError(f"{arm}: expected 147 rows, found {len(frame)}")
            frame["structural_variant"] = arm
            frame = frame[
                ["structural_variant"] + [name for name in frame if name != "structural_variant"]
            ]
            frames.append(frame)
        _table(
            pd.concat(frames, ignore_index=True),
            "Table S-FH3. All points in every structural perturbation",
        )
        _markdown(
            "Feature and interaction detail tables are saved separately for "
            "each arm under "
            f"[{ARTIFACT_DIRECTORY}/tables/]({ARTIFACT_DIRECTORY}/tables/). "
            "Rows retain sample identifiers; no point filtering is applied "
            "to this overview."
        )
    if section in ("all", "cases"):
        _markdown(
            "Case rules are fixed before the fresh-seed reruns. Both arms are "
            "recomputed for each validation seed, disjoint from discovery and "
            "earlier repeat seeds. The repeat distributions assess explanation "
            "stability for these selected examples; they do not provide "
            "population confidence intervals or multiplicity-adjusted claims "
            "across the sweep."
        )
        figure_number = 4
        table_number = 4
        # Only a file recorded in the current completed run is eligible. An
        # unlisted file left by an earlier run must not add a new displayed case.
        if _current_common_case(root, manifest):
            _image(
                figures / "common_case_comparison.png",
                "Figure S-FH4. Shared case from frozen selections",
                root,
            )
            figure_number += 1
        for arm in arms:
            _markdown(f"### {arm}: selected cases and fresh-seed stability")
            selected = None
            for suffix, title in (
                ("selection_rules", "Selection rules"),
                ("selected", "Selected examples"),
                ("validation_summary", "Validation summaries"),
            ):
                frame = _csv(tables / f"{arm}_{suffix}.csv", root)
                if suffix == "selected":
                    selected = frame
                _table(frame, f"Table S-FH{table_number}. {arm}: {title}")
                table_number += 1
            images = _selected_images(artifact_root, arm, selected, root)
            for kind, label in (
                ("case", "Selected example"),
                ("validation", "Fresh-seed stability"),
            ):
                for path in images[kind]:
                    _image(path, f"Figure S-FH{figure_number}. {arm}: {label} ({path.stem})", root)
                    figure_number += 1


def _cell(kind, identifier, source):
    cell = {
        "cell_type": kind,
        "id": identifier,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def _reuse_unchanged_cell(cell, previous):
    """Preserve outputs and execution counts if an installed cell is unchanged."""
    old = previous.get(cell["id"])
    if old and old.get("cell_type") == cell["cell_type"]:
        old_source = old.get("source", [])
        old_source = old_source if isinstance(old_source, str) else "".join(old_source)
        if old_source == "".join(cell["source"]):
            return old
    return cell


def _write_notebook(path, notebook):
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == serialized:
        return
    temporary = path.with_name(path.name + ".fullholdout.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def install_extensions(root=ROOT):
    """Add marked render cells to five notebooks and create the sweep notebook.

    This function writes notebook JSON only. It does not run cells, load result
    tables, train models, or recompute explanations. Reinstallation preserves
    unchanged marked outputs and every unmarked historical cell verbatim as a
    JSON object. Callers can render after completion through the public APIs.
    """
    root = Path(root).resolve()
    if (root / "exports" / "archive").is_dir():
        raise RuntimeError(
            "Append-only layout retired. Run integrate_notebooks.py for the five integrated notebooks."
        )
    changed = []
    template_metadata = None
    for arm in NOTEBOOK_ARMS:
        path = root / f"experiment_{arm}.ipynb"
        if not path.is_file():
            raise FileNotFoundError(path)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if template_metadata is None:
            template_metadata = notebook.get("metadata", {})
        prefix = f"fullholdout-{arm}-"
        previous = {
            cell.get("id"): cell
            for cell in notebook["cells"]
            if str(cell.get("id", "")).startswith(prefix)
        }
        historical = [
            cell for cell in notebook["cells"] if not str(cell.get("id", "")).startswith(prefix)
        ]
        intro = _cell(
            "markdown",
            prefix + "intro",
            "> **Historical and full-holdout results are separated.** The "
            "original cells below retain the earlier 50-row analysis. A new "
            "full-holdout section immediately before the historical Summary "
            "uses all 147 test rows (A/B) or 146 paired target dates (Lag). "
            "It has independent Table/Figure FH numbering and saved "
            "fresh-seed case validation. Execute its render cell to display "
            "the completed artifact run; it never recomputes explanations.\n",
        )
        insertion = next(
            (
                i
                for i, cell in enumerate(historical)
                if cell.get("cell_type") == "markdown"
                and re.search(r"(?im)^#{1,6}\s+Summary\s*$", "".join(cell.get("source", [])))
            ),
            len(historical),
        )
        extension = [
            _cell(
                "markdown",
                prefix + "heading",
                f"## Full-holdout extension — {arm.upper()}\n\n" + _scope_note(arm) + "\n",
            ),
            _cell(
                "code",
                prefix + "render",
                f"from full_holdout_notebooks import render_extension\nrender_extension({arm!r})\n",
            ),
            _cell(
                "markdown",
                prefix + "legacy-summary-note",
                "> **The Summary below belongs to the historical 50-row "
                "analysis.** Its numbers and conclusions are retained for "
                "traceability; the full-holdout results are reported in "
                "the separate section above.\n",
            ),
        ]
        cells = historical[:insertion] + extension + historical[insertion:]
        cells.insert(1 if cells else 0, intro)
        notebook["cells"] = [_reuse_unchanged_cell(cell, previous) for cell in cells]
        _write_notebook(path, notebook)
        changed.append(path)

    sweep_path = root / "experiment_structure_sweep.ipynb"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    else:
        sweep = {
            "cells": [],
            "metadata": template_metadata or {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    prefix = "fullholdout-sweep-"
    previous = {
        cell.get("id"): cell
        for cell in sweep["cells"]
        if str(cell.get("id", "")).startswith(prefix)
    }
    unmanaged = [cell for cell in sweep["cells"] if not str(cell.get("id", "")).startswith(prefix)]
    managed = [
        _cell(
            "markdown",
            prefix + "title",
            "# Full-holdout structural sensitivity sweep\n\n"
            "All structural arms share the model, background and 147 held-out "
            "same-day samples. Configurations are perturbations of a declared "
            "baseline, not established causal truth. Full point tables support "
            "the aggregate view; frozen case rules and fresh-seed reruns support "
            "selected comparisons. Input structure and output interaction graphs "
            "must not be interpreted as the same kinds of edges.\n\n"
            "Artifacts: `results/`. If missing or incomplete, run "
            "`run_full_holdout.py` from this experiments directory first. The "
            "cells below only render completed saved results.\n",
        )
    ]
    section_titles = {
        "catalog": "1. Declared configurations and changed conditioning rules",
        "averages": "2. Average changes and sensitivity across every point",
        "points": "3. Complete per-point comparison tables",
        "cases": "4. Frozen case selection and fresh-seed validation",
    }
    for section, title in section_titles.items():
        managed.append(_cell("markdown", prefix + section + "-heading", f"## {title}\n"))
        managed.append(
            _cell(
                "code",
                prefix + section + "-render",
                "from full_holdout_notebooks import render_sweep\n"
                f"render_sweep(section={section!r})\n",
            )
        )
    sweep["cells"] = [_reuse_unchanged_cell(cell, previous) for cell in managed] + unmanaged
    _write_notebook(sweep_path, sweep)
    changed.append(sweep_path)
    return changed
