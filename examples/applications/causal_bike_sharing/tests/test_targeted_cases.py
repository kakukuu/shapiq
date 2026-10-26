"""Independent checks of frozen-target validation and numerical cache provenance."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext

import case_selection as selection
import data_utils as du
import experiment_context
import numpy as np
import pandas as pd
import pytest
import targeted_cases as targeted


def case_data():
    baseline = np.array([[100.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]])
    pairs = np.arange(100.0, 2200.0, 100.0)[None, :]
    modified = baseline.copy()
    modified[0, 0] += 200
    return dict(
        arm="b3",
        row_ids=np.array([10]),
        base_phi=baseline,
        variant_phi=modified,
        base_interaction=pairs,
        variant_interaction=pairs.copy(),
        points=pd.DataFrame({"row_id": [10], "date": ["2011-01-11"]}),
    )


def seed_details(
    data, seed=809, *, phi_base=None, phi_modified=None, pair_base=None, pair_modified=None
):
    rows = []
    labels = {
        1: du.FEATURE_NAMES,
        2: [f"{du.FEATURE_NAMES[i]} × {du.FEATURE_NAMES[j]}" for i, j in targeted.pipeline.PAIRS],
    }
    arrays = {
        1: (
            data["base_phi"][0] if phi_base is None else phi_base,
            data["variant_phi"][0] if phi_modified is None else phi_modified,
        ),
        2: (
            data["base_interaction"][0] if pair_base is None else pair_base,
            data["variant_interaction"][0] if pair_modified is None else pair_modified,
        ),
    }
    for order, (baseline, modified) in arrays.items():
        for label, left, right in zip(labels[order], baseline, modified, strict=True):
            rows.append(
                dict(
                    arm="b3",
                    row_id=10,
                    seed=seed,
                    order=order,
                    item=label,
                    baseline=left,
                    modified=right,
                    delta=right - left,
                    abs_delta=abs(right - left),
                    sign_flip=left * right < 0,
                )
            )
    return pd.DataFrame(rows)


def selected_reasons(data, order=1, item="trend"):
    return selection.select_target_cases(data, [dict(order=order, item=item)])["reasons"]


def validated_rule(reasons, details, rule):
    result = targeted._reason_validation(reasons, details, pd.DataFrame())
    return result.loc[result.case_rule == rule].iloc[0]


def test_rank_repeat_uses_target_order_only_and_frozen_delta_quantile():
    data = case_data()
    reasons = selected_reasons(data)
    reversed_pairs = data["variant_interaction"][0][::-1]
    good = seed_details(data, pair_modified=reversed_pairs)
    small_phi = data["base_phi"][0].copy()
    small_phi[0] += 100  # Material but below frozen target q75 = 200.
    below_quantile = seed_details(data, seed=907, phi_modified=small_phi)
    row = validated_rule(reasons, pd.concat([good, below_quantile]), "rank_preserved_large_change")
    assert row.n_validation_seeds == 2 and row.phenomenon_repeat_count == 1


def test_material_flip_repeat_requires_both_endpoint_sizes_and_same_orientation():
    data = case_data()
    data["variant_phi"][0, 0] = -100
    reasons = selected_reasons(data)
    good = seed_details(data)
    near_zero = data["variant_phi"][0].copy()
    near_zero[0] = -10
    bad_small = seed_details(data, seed=907, phi_modified=near_zero)
    reverse_left, reverse_right = data["base_phi"][0].copy(), data["variant_phi"][0].copy()
    reverse_left[0], reverse_right[0] = -100, 100
    bad_orientation = seed_details(
        data, seed=1009, phi_base=reverse_left, phi_modified=reverse_right
    )
    row = validated_rule(
        reasons, pd.concat([good, bad_small, bad_orientation]), "material_sign_flip"
    )
    assert row.n_validation_seeds == 3 and row.phenomenon_repeat_count == 1


def test_pair_repeat_requires_an_associated_material_singleton_in_opposite_direction():
    data = case_data()
    data["variant_interaction"][0, 0] -= 200
    reasons = selected_reasons(data, 2, "trend × cosyear")
    good = seed_details(data)
    singles = data["base_phi"][0].copy()
    singles[2] += 999  # An unrelated feature does not support this pair's reason.
    unrelated = seed_details(data, seed=907, phi_modified=singles)
    singles[0] += 10  # Correct member/direction but below its 25-unit threshold.
    tiny = seed_details(data, seed=1009, phi_modified=singles)
    row = validated_rule(reasons, pd.concat([good, unrelated, tiny]), "paired_redistribution")
    assert row.n_validation_seeds == 3 and row.phenomenon_repeat_count == 1


def test_typical_repeat_is_only_direction_check_and_never_reselects():
    data = case_data()
    reasons = selected_reasons(data)
    changed = data["base_phi"][0].copy()
    changed[0] += 1  # No longer near discovery median, but selected date stays fixed.
    row = validated_rule(reasons, seed_details(data, phi_modified=changed), "typical_target_change")
    assert row.row_id == 10 and row.delta == 200
    assert row.phenomenon_repeat_count == 1
    assert "not re-ranked" in row.validation_note


class FakeContext:
    label = "ordinary"

    def __init__(self):
        self.X_test = pd.DataFrame(np.arange(14.0).reshape(2, 7), index=[10, 20])

    def fingerprints(self):
        return {"model": "fixed-model", "background": "fixed-background"}

    def predict(self, rows):
        return np.zeros(len(rows))


def protocol():
    return dict(
        source_sha256="engine",
        versions={"numpy": "fixture"},
        sampling=dict(
            causal_sample_size=500,
            marginal_sample_size="all",
            causal_sampling_method="copula",
            marginal_sampling_method="joint",
        ),
        budget=5000,
        discovery_seed=42,
        protocol_version="fixture-v1",
        validation_seeds=[809, 907],
    )


def cache_file(folder, *, context=None, metadata_overrides=None, arrays_overrides=None):
    folder.mkdir(parents=True, exist_ok=True)
    context = FakeContext() if context is None else context
    spec = targeted.pipeline.VARIANTS["baseline"]
    metadata = targeted._numeric_identity(context, spec, protocol())
    ids = np.array([10, 20])
    metadata.update(
        seed=809,
        row_ids=ids.tolist(),
        sample_size=500,
        sampling_method="copula",
        row_values_sha256=hashlib.sha256(context.X_test.loc[ids].to_numpy().tobytes()).hexdigest(),
    )
    metadata.update(metadata_overrides or {})
    serialized = json.dumps(metadata, sort_keys=True)
    path = folder / (hashlib.sha256(serialized.encode()).hexdigest() + ".npz")
    arrays = dict(
        phi=np.zeros((2, 7)), interaction=np.zeros((2, 21)), empty=np.zeros(2), row_ids=ids
    )
    arrays.update(arrays_overrides or {})
    np.savez(path, metadata=np.array(serialized), **arrays)
    return path


def read_cache(paths, requested=(10,)):
    return targeted._read_cached_rows(
        paths, FakeContext(), targeted.pipeline.VARIANTS["baseline"], protocol(), set(requested)
    )


def test_cache_reuses_only_requested_rows_and_independent_seeds(tmp_path):
    valid = cache_file(tmp_path / "valid")
    discovery = cache_file(tmp_path / "discovery", metadata_overrides={"seed": 42})
    result = read_cache([valid, discovery])
    assert list(result) == [(809, 10)]
    np.testing.assert_array_equal(result[(809, 10)]["phi"], np.zeros(7))


@pytest.mark.parametrize(
    "field,value",
    [("source_sha256", "other-engine"), ("sample_size", 100), ("sampling_method", "gaussian")],
)
def test_cache_rejects_mismatched_actual_sampling_and_engine_metadata(tmp_path, field, value):
    path = cache_file(tmp_path, metadata_overrides={field: value})
    assert read_cache([path]) == {}


def test_cache_validates_empty_and_raw_row_id_shapes(tmp_path):
    bad_empty = cache_file(tmp_path / "empty", arrays_overrides={"empty": np.zeros((2, 1))})
    with pytest.raises(ValueError, match="shape"):
        read_cache([bad_empty])
    bad_ids = cache_file(tmp_path / "ids", arrays_overrides={"row_ids": np.array([10.5, 20.5])})
    with pytest.raises(ValueError, match="identit|integer|row"):
        read_cache([bad_ids])


def test_cache_rejects_tampered_key_input_fingerprint_and_conflicts(tmp_path):
    path = cache_file(tmp_path / "wrong-name")
    renamed = path.with_name("not-a-digest.npz")
    path.rename(renamed)
    with pytest.raises(ValueError, match="cache key"):
        read_cache([renamed])
    bad_inputs = cache_file(tmp_path / "inputs", metadata_overrides={"row_values_sha256": "bad"})
    with pytest.raises(ValueError, match="Cached inputs"):
        read_cache([bad_inputs])
    valid = cache_file(tmp_path / "valid")
    changed_phi, changed_interactions = np.zeros((2, 7)), np.zeros((2, 21))
    changed_phi[0, 0], changed_interactions[0, 0] = 25.0, -25.0  # Efficiency still holds.
    contradictory = cache_file(
        tmp_path / "conflict",
        arrays_overrides={"phi": changed_phi, "interaction": changed_interactions},
    )
    with pytest.raises(ValueError, match="Conflicting"):
        read_cache([valid, contradictory])


def test_all_requested_arm_selections_are_frozen_before_validation_draws(monkeypatch, tmp_path):
    data, context = case_data(), FakeContext()
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({context.label: {"fingerprints": context.fingerprints()}}), encoding="utf-8"
    )
    monkeypatch.setattr(targeted, "manifest", lambda root: {"config": protocol()})
    monkeypatch.setattr(targeted, "load_experiment", lambda arm, root: {**data, "arm": arm})
    monkeypatch.setattr(targeted, "_provenance", lambda arm, root, config, targets: {"arm": arm})
    monkeypatch.setattr(targeted, "verified_path", lambda name, root: models)
    monkeypatch.setattr(targeted, "calculation_lock", lambda root: nullcontext())
    arms = ["b1", "b3"]
    targets = {arm: [dict(order=1, item="trend")] for arm in arms}

    class StopBeforeValidationDraws(Exception):
        pass

    def inspect_frozen_then_stop(*args):
        for arm in arms:
            folder = targeted._folder(tmp_path, arm)
            frozen = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
            saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            assert saved["completed"] is False
            assert frozen["selected_unique"][0]["row_id"] == 10
            assert len(frozen["reasons"]) == len(pd.read_csv(folder / "reasons.csv"))
        raise StopBeforeValidationDraws

    monkeypatch.setattr(experiment_context, "build_context", lambda: context)
    monkeypatch.setattr(targeted, "_read_cached_rows", inspect_frozen_then_stop)
    with pytest.raises(StopBeforeValidationDraws):
        targeted.compute_targeted_cases(arms, root=tmp_path, targets=targets)


def mock_cached_pipeline(monkeypatch, root, *, mutate_frozen=False):
    """Use opposite-direction repeats without launching models or subprocesses."""
    data, context = case_data(), FakeContext()
    models = root / "models.json"
    models.write_text(
        json.dumps({context.label: {"fingerprints": context.fingerprints()}}), encoding="utf-8"
    )
    monkeypatch.setattr(targeted, "manifest", lambda root: {"config": protocol()})
    monkeypatch.setattr(targeted, "load_experiment", lambda arm, root: data)
    monkeypatch.setattr(targeted, "_provenance", lambda arm, root, config, targets: {"arm": arm})
    monkeypatch.setattr(targeted, "verified_path", lambda name, root: models)
    monkeypatch.setattr(targeted, "calculation_lock", lambda root: nullcontext())
    monkeypatch.setattr(experiment_context, "build_context", lambda: context)

    def cached(paths, fitted, spec, proto, requested):
        if mutate_frozen:
            path = targeted._folder(root, "b3") / "selection.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["reasons"][0]["row_id"] = 20
            path.write_text(json.dumps(value), encoding="utf-8")
        phi = data["base_phi"][0].copy()
        interactions = data["base_interaction"][0].copy()
        if spec != targeted.pipeline.VARIANTS["baseline"]:
            phi[0] -= 200
        return {
            (seed, row_id): dict(
                phi=phi.copy(),
                interaction=interactions.copy(),
                empty=np.array(-phi.sum() - interactions.sum()),
            )
            for seed in proto["validation_seeds"]
            for row_id in requested
        }

    monkeypatch.setattr(targeted, "_read_cached_rows", cached)
    return data


def test_unsuccessful_repetitions_do_not_remove_or_replace_frozen_cases(monkeypatch, tmp_path):
    data = mock_cached_pipeline(monkeypatch, tmp_path)
    targets = {"b3": [dict(order=1, item="trend")]}
    targeted.compute_targeted_cases(["b3"], root=tmp_path, targets=targets)
    folder = targeted._folder(tmp_path, "b3")
    selected = pd.read_csv(folder / "selected_unique.csv")
    checks = pd.read_csv(folder / "reason_validation.csv")
    assert selected.row_id.tolist() == [10]
    assert len(checks) == len(selected_reasons(data))
    assert checks.phenomenon_repeat_count.eq(0).all()
    assert checks.n_validation_seeds.eq(2).all()
    saved = targeted._read_manifest("b3", tmp_path)
    assert saved["completed"] and saved["selection_frozen_before_validation_stage"]
    assert saved["selection_uses_validation_values"] is False


def test_change_to_frozen_selection_during_validation_aborts_publication(monkeypatch, tmp_path):
    mock_cached_pipeline(monkeypatch, tmp_path, mutate_frozen=True)
    targets = {"b3": [dict(order=1, item="trend")]}
    with pytest.raises(ValueError, match="Selection changed during validation"):
        targeted.compute_targeted_cases(["b3"], root=tmp_path, targets=targets)
    folder = targeted._folder(tmp_path, "b3")
    saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert not saved["completed"]


def test_model_mismatch_keeps_existing_case_results(monkeypatch, tmp_path):
    mock_cached_pipeline(monkeypatch, tmp_path)
    folder = targeted._folder(tmp_path, "b3")
    folder.mkdir(parents=True)
    saved = folder / "manifest.json"
    saved.write_text('{"completed": true}', encoding="utf-8")
    original = saved.read_bytes()
    context = FakeContext()
    monkeypatch.setattr(context, "fingerprints", lambda: {"model": "changed"})
    monkeypatch.setattr(experiment_context, "build_context", lambda: context)
    with pytest.raises(ValueError, match="Reconstructed model/background differ"):
        targeted.compute_targeted_cases(
            ["b3"], root=tmp_path, targets={"b3": [dict(order=1, item="trend")]}
        )
    assert saved.read_bytes() == original
    assert not (tmp_path / "archive").exists()


def test_invalid_selection_keeps_existing_case_results(monkeypatch, tmp_path):
    mock_cached_pipeline(monkeypatch, tmp_path)
    folder = targeted._folder(tmp_path, "b3")
    folder.mkdir(parents=True)
    saved = folder / "manifest.json"
    saved.write_text('{"completed": true}', encoding="utf-8")
    original = saved.read_bytes()
    with pytest.raises(ValueError):
        targeted.compute_targeted_cases(
            ["b3"], root=tmp_path, targets={"b3": [dict(order=1, item="unknown")]}
        )
    assert saved.read_bytes() == original
    assert not (tmp_path / "archive").exists()


def test_completed_manifest_requires_every_table_and_frozen_selection_checksum(tmp_path):
    folder = targeted._folder(tmp_path, "b3")
    folder.mkdir(parents=True)
    value = dict(completed=True, provenance={}, artifacts_sha256={})
    (folder / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing targeted case checksums"):
        targeted._read_manifest("b3", tmp_path)


def test_manifest_provenance_must_match_frozen_selection_even_if_hashes_match(tmp_path):
    folder = targeted._folder(tmp_path, "b3")
    folder.mkdir(parents=True)
    for name in targeted.TABLES:
        (folder / f"{name}.csv").write_text("row_id\n10\n", encoding="utf-8")
    (folder / "selection.json").write_text(
        json.dumps({"provenance": {"arm": "b1"}}), encoding="utf-8"
    )
    hashes = {path.name: targeted._hash(path) for path in folder.iterdir()}
    value = dict(completed=True, provenance={"arm": "b3"}, artifacts_sha256=hashes)
    (folder / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen selection provenance"):
        targeted._read_manifest("b3", tmp_path)
