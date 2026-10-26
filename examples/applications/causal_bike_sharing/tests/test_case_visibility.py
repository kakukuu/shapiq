"""Selected cases must be visible as notebook images without opening HTML cards."""

from __future__ import annotations

import copy
import json

import case_selection_figures
import interaction_change_graph
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sample_readings
import targeted_cases as targeted
from IPython.display import HTML, Image, Markdown


def focused_cases():
    """Five dates in a non-positional order exercise complete case rendering."""
    row_ids = np.array([1010, 20, 730, 451, 92])
    readings = pd.DataFrame(
        dict(
            row_id=row_ids,
            date=[f"2011-01-{day:02d}" for day in range(11, 16)],
            temp=12.75 + np.arange(5),
            atemp=15.25 + np.arange(5),
            hum=62.5 + np.arange(5),
            windspeed=7.25 + np.arange(5),
            trend=10 + np.arange(5),
            cosyear=np.linspace(0.1, 0.5, 5),
            sinyear=np.linspace(0.6, 1.0, 5),
            y_true=2200 + 101 * np.arange(5),
            prediction_base=2300.5 + 90 * np.arange(5),
            prediction_variant=2300.5 + 90 * np.arange(5),
        )
    )
    reasons = pd.DataFrame(
        [
            dict(
                row_id=int(row_id),
                order=order,
                item=item,
                case_rule="largest_target_change",
                baseline=100.0,
                modified=200.0,
                delta=100.0,
                eligible_count=5,
                cohort_size=5,
                selection_reason="A fixed target, selected before validation.",
            )
            for row_id in row_ids
            for order, item in ((1, "temp"), (2, "temp × atemp"))
        ]
    )
    return dict(
        arm="b3",
        data=dict(
            row_ids=row_ids,
            points=readings,
            base_phi=np.ones((5, 7)),
            variant_phi=np.full((5, 7), 2.0),
            base_interaction=np.ones((5, 21)),
            variant_interaction=np.full((5, 21), 2.0),
            protocol=dict(discovery_seed=42),
        ),
        selected_unique=pd.DataFrame(dict(row_id=row_ids[[2, 0, 4, 1, 3]])),
        reasons=reasons,
        rules=pd.DataFrame(dict(case_rule=["largest_target_change"], eligible_count=[5])),
        candidates=reasons.assign(eligible=True, selected=True),
        validation_details=pd.DataFrame(
            [dict(row_id=int(row_id), seed=seed) for row_id in row_ids for seed in (809, 907)]
        ),
        validation_summary=reasons.assign(delta_mean=100.0, n_seeds=2),
        reason_validation=reasons.assign(phenomenon_repeat_count=2, n_validation_seeds=2),
        manifest=dict(artifacts_sha256={"fixture": "fixed-inputs"}),
    )


@pytest.fixture
def renderer(monkeypatch, tmp_path):
    displayed, plots, decorated = [], [], []

    def small_figure():
        return plt.figure(figsize=(0.2, 0.2))

    def network(*arrays, **kwargs):
        plots.append(("network", kwargs))
        figure = small_figure()
        figure.case_figure_kind = "network"
        figure.change_graph_summary = dict(
            nodes=pd.DataFrame(dict(value=[1.0])), edges=pd.DataFrame(dict(value=[1.0]))
        )
        return figure

    def repeats(*args, **kwargs):
        plots.append(("validation", kwargs))
        figure = small_figure()
        figure.case_figure_kind = "validation"
        return figure

    def attach_readings(figure, readings):
        # Geometry is covered by sample_readings tests; these tiny stand-ins
        # exercise which point and which output receive the information.
        figure.sample_readings = copy.deepcopy(readings)
        decorated.append((figure.case_figure_kind, copy.deepcopy(readings)))
        return figure

    monkeypatch.setattr(
        IPython.display, "display", lambda *items, **kwargs: displayed.extend(items)
    )
    monkeypatch.setattr(interaction_change_graph, "plot_interaction_change_graph", network)
    monkeypatch.setattr(case_selection_figures, "plot_target_validation", repeats)
    monkeypatch.setattr(sample_readings, "add_sample_readings", attach_readings)
    focused = focused_cases()
    targeted._folder(tmp_path, focused["arm"]).mkdir(parents=True)
    return focused, displayed, plots, decorated, tmp_path


@pytest.mark.parametrize("expanded_tables", [False, True])
def test_every_selected_case_has_native_images_independent_of_table_expansion(
    renderer, expanded_tables
):
    focused, displayed, plots, decorated, root = renderer
    original = copy.deepcopy(focused)
    targeted.render_targeted_cases(focused, "B3", root=root, expanded_tables=expanded_tables)

    images = [item for item in displayed if isinstance(item, Image)]
    assert len(images) == 2 * len(focused["selected_unique"])
    for image in images:
        bundle = image._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert "image/png" in data and data["image/png"]

    # Large HTML cards must not hide the two figures from notebook renderers.
    html = "\n".join(item.data for item in displayed if isinstance(item, HTML))
    assert "data:image/png;base64," not in html
    assert 'class="targeted-case"' not in html
    last_image = max(index for index, item in enumerate(displayed) if isinstance(item, Image))
    for index, item in enumerate(displayed):
        if isinstance(item, (HTML, Markdown)) and "candidate audit" in item.data.lower():
            assert index > last_image
    headings = "\n".join(item.data for item in displayed if isinstance(item, Markdown))
    for row_id in focused["selected_unique"].row_id:
        assert f"row {row_id}" in headings

    networks = [kwargs for kind, kwargs in plots if kind == "network"]
    assert len(networks) == 5
    assert all(set(item["focus_features"]) == {"temp", "atemp"} for item in networks)
    assert all(item["focus_pairs"] == ["temp × atemp"] for item in networks)
    pd.testing.assert_frame_equal(focused["reasons"], original["reasons"])
    pd.testing.assert_frame_equal(focused["selected_unique"], original["selected_unique"])
    for field in ("base_phi", "variant_phi", "base_interaction", "variant_interaction"):
        np.testing.assert_array_equal(focused["data"][field], original["data"][field])

    # Readings must follow original row IDs even when selection order differs
    # from the saved array/point-table order, and must reach both case figures.
    points = focused["data"]["points"].set_index("row_id", drop=False)
    expected_order = [int(row) for row in focused["selected_unique"].row_id]
    assert [readings["row_id"] for _, readings in decorated] == [
        row for row in expected_order for _ in range(2)
    ]
    for row_id in expected_order:
        actual = [(kind, value) for kind, value in decorated if value["row_id"] == row_id]
        assert [kind for kind, _ in actual] == ["network", "validation"]
        expected = sample_readings.build_sample_readings(points.loc[row_id], "b3")
        assert actual[0][1] == actual[1][1] == expected
        assert expected["observed_raw_values"]["temp"] == points.loc[row_id, "temp"]
        assert expected["outcomes"]["y_true"] == points.loc[row_id, "y_true"]
        saved = targeted._folder(root, "b3") / f"case_{row_id}_readings.json"
        assert json.loads(saved.read_text(encoding="utf-8")) == expected


def test_cached_figures_are_still_displayed_without_replotting(renderer):
    focused, displayed, plots, decorated, root = renderer
    targeted.render_targeted_cases(focused, "B3", root=root)
    assert len(plots) == 10
    displayed.clear()
    plots.clear()
    decorated.clear()

    targeted.render_targeted_cases(focused, "B3", root=root)

    assert plots == []
    assert decorated == []
    assert len([item for item in displayed if isinstance(item, Image)]) == 10
    assert not plt.get_fignums()


@pytest.mark.parametrize("change", ["raw_readings", "missing_json", "tampered_json"])
def test_reading_changes_invalidate_only_the_corresponding_case_render(renderer, change):
    focused, displayed, plots, decorated, root = renderer
    targeted.render_targeted_cases(focused, "B3", root=root)
    row_id = int(focused["selected_unique"].iloc[0].row_id)
    folder = targeted._folder(root, "b3")
    path = folder / f"case_{row_id}_readings.json"
    points = focused["data"]["points"]
    expected = json.loads(path.read_text(encoding="utf-8"))
    if change == "raw_readings":
        points.loc[points.row_id.eq(row_id), "temp"] += 4.5
        expected = sample_readings.build_sample_readings(
            points.loc[points.row_id.eq(row_id)].iloc[0], "b3"
        )
    elif change == "missing_json":
        path.unlink()
    else:
        path.write_text('{"tampered": true}', encoding="utf-8")
    displayed.clear()
    plots.clear()
    decorated.clear()

    targeted.render_targeted_cases(focused, "B3", root=root)

    assert [kind for kind, _ in plots] == ["network", "validation"]
    assert [kind for kind, _ in decorated] == ["network", "validation"]
    assert all(value == expected for _, value in decorated)
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert len([item for item in displayed if isinstance(item, Image)]) == 10
