"""Fast provenance, graph and notebook-integration regression tests.

The numeric cache worker is exercised with a deterministic fake explanation;
these tests never fit a model or run the expensive full-holdout computation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import run_full_holdout as pipeline


class _Context:
    """Only the fitted-context contract consumed by the cache worker."""

    def __init__(self):
        self.X_test = pd.DataFrame(
            np.arange(21, dtype=float).reshape(3, 7),
            index=[11, 4, 29],
            columns=pipeline.FEATURES,
        )
        self.provenance = {
            "model_sha256": "model-A",
            "background_sha256": "background-A",
            "feature_names": list(pipeline.FEATURES),
        }
        self.explainer_calls = []

    def fingerprints(self):
        return deepcopy(self.provenance)

    def predict(self, rows=None):
        rows = self.X_test if rows is None else rows
        return rows.iloc[:, 0].to_numpy(dtype=float) + 500.0

    def explainer(self, **kwargs):
        self.explainer_calls.append(kwargs)
        return SimpleNamespace(
            imputer=SimpleNamespace(
                sampling_method="copula", sample_size=kwargs.get("sample_size", 500)
            )
        )


def _fake_explain(explainer, row, *, budget, seed):
    value = float(row.iloc[0]) + seed / 1000
    values = {(): 12.5}
    values.update({(i,): value + i for i in range(7)})
    values.update({pair: value - i for i, pair in enumerate(pipeline.PAIRS)})
    # Changes in RNG may redistribute contributions but cannot change the
    # prediction that this complete k-SII decomposition must reconstruct.
    values[pipeline.PAIRS[-1]] += float(row.iloc[0]) + 500.0 - sum(values.values())
    return SimpleNamespace(dict_values=values, baseline_value=12.5)


class GraphAndSeedTests(unittest.TestCase):
    def test_catalog_counts_match_exhaustive_conditioning_audit(self):
        # Expected values are an independent audit of 128 coalitions and their
        # 448 missing-feature conditioning sets; they are not edge counts.
        expected = {
            "baseline": (0, 0, 0, 0),
            "b1": (0, 0, 300, 120),
            "b2": (6, 2, 336, 124),
            "b3": (0, 5, 64, 64),
            "temp_root": (4, 1, 320, 124),
            "hum_root": (4, 1, 320, 124),
            "windspeed_root": (4, 1, 320, 124),
            "b3_unconfounded": (0, 5, 372, 124),
            "b2_unconfounded": (6, 2, 336, 124),
        }
        frame = pipeline.catalog().set_index("variant")
        self.assertEqual(set(frame.index), set(expected))
        self.assertTrue((frame["conditioning_items_total"] == 448).all())
        columns = [
            "added_ancestor_relations",
            "removed_ancestor_relations",
            "changed_conditioning_items",
            "changed_coalitions",
        ]
        for arm, values in expected.items():
            with self.subTest(arm=arm):
                self.assertEqual(tuple(frame.loc[arm, columns]), values)
                ordering = json.loads(frame.loc[arm, "ordering"])
                self.assertEqual(sorted(sum(ordering, [])), list(range(7)))

    def test_invalid_partition_is_rejected(self):
        variants = deepcopy(pipeline.VARIANTS)
        variants["temp_root"]["ordering"] = [[0, 3], [1, 2, 5, 5], [4]]
        with patch.object(pipeline, "VARIANTS", variants), self.assertRaises(ValueError):
            pipeline.catalog()

    def test_validation_seeds_are_unique_and_disjoint_from_all_discovery_seeds(self):
        discovery = {pipeline.DISCOVERY_SEED, *pipeline.REFERENCE_SEEDS}
        validation = set(pipeline.VALIDATION_SEEDS)
        self.assertEqual(len(validation), len(pipeline.VALIDATION_SEEDS))
        self.assertGreaterEqual(len(validation), 8)
        self.assertFalse(validation & discovery)


class DrawRoundTripTests(unittest.TestCase):
    def test_as_sii_preserves_all_orders_and_empty_value(self):
        draw = {
            "phi": np.array([[1, -2, 3, -4, 5, -6, 7], [8, 6, 4, 2, 0, -2, -4]], dtype=float),
            "interaction": np.arange(42, dtype=float).reshape(2, 21) - 20.5,
            "empty": np.array([4.75, -3.125]),
            "row_ids": np.array([29, 4]),
        }
        results = pipeline.as_sii(draw)
        self.assertEqual(len(results), 2)
        for i, result in enumerate(results):
            np.testing.assert_array_equal(
                [result.dict_values[(j,)] for j in range(7)], draw["phi"][i]
            )
            np.testing.assert_array_equal(
                [result.dict_values[pair] for pair in pipeline.PAIRS], draw["interaction"][i]
            )
            self.assertEqual(result.baseline_value, draw["empty"][i])
            self.assertEqual(result.dict_values[()], draw["empty"][i])
            self.assertEqual(result.index, "k-SII")

    def test_subset_reorders_every_array_by_source_id(self):
        draw = {
            "phi": np.arange(21).reshape(3, 7),
            "interaction": np.arange(63).reshape(3, 21),
            "empty": np.array([7, 8, 9]),
            "row_ids": np.array([11, 4, 29]),
        }
        subset = pipeline.subset(draw, [29, 11])
        for name in draw:
            np.testing.assert_array_equal(subset[name], draw[name][[2, 0]])
        with self.assertRaises(KeyError):
            pipeline.subset(draw, [999])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.context = _Context()
        self.base_spec = deepcopy(pipeline.VARIANTS["baseline"])

    def draw(
        self, *, context=None, spec=None, seed=42, ids=(11, 4), source="engine-A", versions=None
    ):
        job = (
            self.context if context is None else context,
            self.base_spec if spec is None else spec,
            seed,
            list(ids),
            source,
            {"numpy": "test-version"} if versions is None else versions,
            self.temporary.name,
        )
        return pipeline._draw_worker(job)

    def test_exact_request_reuses_cache_without_another_explanation(self):
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain) as call:
            path, was_cached = self.draw()
            again, second_cached = self.draw()
        self.assertFalse(was_cached)
        self.assertTrue(second_cached)
        self.assertEqual(path, again)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(len(self.context.explainer_calls), 1)
        draw = pipeline.load_draw(path)
        np.testing.assert_array_equal(draw["row_ids"], [11, 4])

    def test_material_provenance_changes_invalidate_cache(self):
        requests = []
        graph = deepcopy(self.base_spec)
        graph["confounding"][1] = False
        requests.extend(
            [
                {"seed": 809},
                {"spec": graph},
                {"ids": (4, 11)},
                {"source": "engine-B"},
                {"versions": {"numpy": "new-version"}},
            ]
        )
        for key in ("model_sha256", "background_sha256", "feature_names"):
            context = _Context()
            context.provenance[key] = (
                "changed" if key != "feature_names" else list(reversed(pipeline.FEATURES))
            )
            requests.append({"context": context})
        context = _Context()
        context.X_test.loc[11, "temp"] += 0.01
        requests.append({"context": context})
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            first, _ = self.draw()
            for request in requests:
                with self.subTest(request=list(request)):
                    path, cached = self.draw(**request)
                    self.assertNotEqual(path, first)
                    self.assertFalse(cached)
            with patch.object(pipeline, "BUDGET", pipeline.BUDGET + 1):
                changed_budget, cached = self.draw()
                self.assertNotEqual(changed_budget, first)
                self.assertFalse(cached)

    def test_cosmetic_graph_label_does_not_invalidate_numeric_cache(self):
        renamed = deepcopy(self.base_spec)
        renamed["label"] = "A clearer display title"
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            path, _ = self.draw()
            renamed_path, cached = self.draw(spec=renamed)
        self.assertEqual(path, renamed_path)
        self.assertTrue(cached)

    def test_cache_rejects_wrong_row_ids_even_when_numeric_shape_matches(self):
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            path, _ = self.draw()
            with np.load(path, allow_pickle=False) as stored:
                corrupted = {key: stored[key].copy() for key in stored.files}
            corrupted["row_ids"] = np.array([4, 11])
            np.savez_compressed(path, **corrupted)
            with self.assertRaises(ValueError):
                self.draw()

    def test_cache_rejects_wrong_empty_array_shape(self):
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            path, _ = self.draw()
            with np.load(path, allow_pickle=False) as stored:
                corrupted = {key: stored[key].copy() for key in stored.files}
            corrupted["empty"] = np.array([12.5])
            np.savez_compressed(path, **corrupted)
            with self.assertRaises(ValueError):
                self.draw()

    def test_new_explanation_must_reconstruct_model_prediction(self):
        def inconsistent(*args, **kwargs):
            result = _fake_explain(*args, **kwargs)
            result.dict_values[(0,)] += 100.0
            return result

        with (
            patch.object(pipeline.du, "explain_one", side_effect=inconsistent),
            self.assertRaises((ValueError, AssertionError)),
        ):
            self.draw()
        self.assertFalse(list(Path(self.temporary.name).glob("*.npz")))

    def test_cached_decomposition_must_reconstruct_prediction(self):
        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            path, _ = self.draw()
        with np.load(path, allow_pickle=False) as stored:
            corrupted = {key: stored[key].copy() for key in stored.files}
        corrupted["phi"][0, 0] += 100.0
        np.savez_compressed(path, **corrupted)
        with self.assertRaises((ValueError, AssertionError)):
            self.draw()

    def test_sample_size_is_explicit_and_part_of_cache_key(self):
        def request(sample_size):
            return pipeline._draw_worker(
                (
                    self.context,
                    self.base_spec,
                    42,
                    [11, 4],
                    "engine-A",
                    {"numpy": "test-version"},
                    self.temporary.name,
                    {"causal_sample_size": sample_size, "marginal_sample_size": "all"},
                )
            )

        with patch.object(pipeline.du, "explain_one", side_effect=_fake_explain):
            first, first_cached = request(500)
            larger, larger_cached = request(1000)
            reused, reused_cached = request(1000)
        self.assertFalse(first_cached)
        self.assertFalse(larger_cached)
        self.assertTrue(reused_cached)
        self.assertNotEqual(first, larger)
        self.assertEqual(larger, reused)
        self.assertEqual(
            [call["sample_size"] for call in self.context.explainer_calls], [500, 1000]
        )
