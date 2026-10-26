"""Fresh-process notebook workflow and stale-evidence regression checks.

Small isolated fixtures exercise actual source/data checksums and protocol
validation without fitting models or trusting any pre-fix result artifacts.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import integrated_support as support
import run_full_holdout as pipeline


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        helper = self.root / "experiment_support"
        helper.mkdir()
        for name in (
            "data_utils.py",
            "experiment_context.py",
            "run_full_holdout.py",
            "local_analysis.py",
        ):
            (helper / name).write_text("# fixture source\n", encoding="utf-8")
        self.library = self.root / "src/shapiq/imputer/causal_imputer.py"
        self.library.parent.mkdir(parents=True)
        self.library.write_text("# initial numerical implementation\n", encoding="utf-8")
        (self.root / "src/shapiq/__init__.py").write_text("", encoding="utf-8")
        self.dataset = self.root / "data/bike_sharing_daily.csv"
        self.dataset.parent.mkdir(parents=True)
        self.dataset.write_text("temp,cnt\n1,3\n", encoding="utf-8")
        self.results = self.root / support.RESULTS
        self.results.mkdir()
        self.versions = {"numpy": "fixture"}
        self.version_patch = patch.object(
            pipeline, "dependency_versions", return_value=self.versions
        )
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)
        self.write_manifest()

    def write_manifest(self, **config_changes):
        config = {
            "protocol_version": pipeline.PROTOCOL_VERSION,
            "source_sha256": pipeline.source_hash(self.root),
            "versions": deepcopy(self.versions),
            "sampling": pipeline.sampling_config(),
            "budget": pipeline.BUDGET,
            "discovery_seed": pipeline.DISCOVERY_SEED,
            "reference_seeds": list(pipeline.REFERENCE_SEEDS),
            "validation_seeds": list(pipeline.VALIDATION_SEEDS),
            "variants": deepcopy(pipeline.VARIANTS),
        }
        config.update(config_changes)
        hashes = {}
        for name in (
            "models.json",
            "rank_calibration.json",
            "protocol.json",
            "selection_manifest.json",
        ):
            path = self.results / name
            path.write_text(json.dumps(config if name == "protocol.json" else {}), encoding="utf-8")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        value = {
            "completed": True,
            "postprocessing_completed": True,
            "config": config,
            "artifacts_sha256": hashes,
        }
        (self.results / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    def test_source_and_input_data_changes_reject_pre_fix_results(self):
        for path in (self.library, self.dataset):
            with self.subTest(path=path.name):
                self.write_manifest()
                path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "different code or data"):
                    support.manifest(root=self.root)

    def test_protocol_or_dependency_changes_reject_saved_results(self):
        for changes in ({"protocol_version": "old"}, {"versions": {"numpy": "old"}}):
            with self.subTest(changes=changes):
                self.write_manifest(**changes)
                with self.assertRaises(RuntimeError):
                    support.manifest(root=self.root)

    def test_computation_without_complete_publication_is_not_reused(self):
        value = self.write_manifest()
        value["postprocessing_completed"] = False
        (self.results / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
        with (
            patch.object(support, "recompute_results") as compute,
            self.assertRaisesRegex(RuntimeError, "publication is incomplete"),
        ):
            support.ensure_results(root=self.root, mode="load")
        compute.assert_not_called()

    def test_auto_reuses_matching_results_without_computation(self):
        with patch.object(support, "recompute_results") as compute:
            result = support.ensure_results(root=self.root, mode="auto")
        compute.assert_not_called()
        self.assertTrue(result["completed"])

    def test_load_refuses_sampling_changes_without_running(self):
        with (
            patch.object(support, "recompute_results") as compute,
            self.assertRaisesRegex(RuntimeError, "sampling settings"),
        ):
            support.ensure_results(root=self.root, mode="load", causal_sample_size=1000)
        compute.assert_not_called()

    def test_auto_recomputes_old_code_and_passes_requested_sampling(self):
        self.library.write_text("# corrected implementation\n", encoding="utf-8")

        def refresh(root, causal, marginal):
            self.assertEqual(Path(root), self.root)
            self.write_manifest(sampling=pipeline.sampling_config(causal, marginal))

        with patch.object(support, "recompute_results", side_effect=refresh) as compute:
            result = support.ensure_results(root=self.root, mode="auto", causal_sample_size=1000)
        compute.assert_called_once_with(self.root, 1000, "all")
        self.assertEqual(result["config"]["sampling"]["causal_sample_size"], 1000)

    def test_recompute_mode_runs_even_for_matching_results(self):
        with patch.object(support, "recompute_results") as compute:
            support.ensure_results(root=self.root, mode="recompute")
        compute.assert_called_once_with(self.root, 500, "all")

    def test_rank_and_model_json_tampering_cannot_be_loaded(self):
        for name in ("models.json", "rank_calibration.json"):
            with self.subTest(name=name):
                self.write_manifest()
                (self.results / name).write_text('{"changed":true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    support.verified_path(name, root=self.root)
                with (
                    patch.object(support, "recompute_results") as compute,
                    self.assertRaisesRegex(RuntimeError, "modified result"),
                ):
                    support.ensure_results(root=self.root, mode="load")
                compute.assert_not_called()

    def test_invalid_mode_and_sampling_fail_before_any_computation(self):
        with patch.object(support, "recompute_results") as compute:
            for options in (
                {"mode": "typo"},
                {"causal_sample_size": 0},
                {"causal_sample_size": True},
                {"marginal_sample_size": "random"},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    support.ensure_results(root=self.root, **options)
        compute.assert_not_called()

    def test_recompute_launches_same_interpreter_and_preserves_old_evidence(self):
        old_models = (self.results / "models.json").read_bytes()
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = io.StringIO("calculation progress\n")
        process.wait.return_value = 0
        with patch.object(support.subprocess, "Popen", return_value=process) as launch:
            support.recompute_results(
                self.root, causal_sample_size=1000, marginal_sample_size="all"
            )
        args = launch.call_args.args[0]
        self.assertEqual(args[0], support.sys.executable)
        self.assertIn(str(self.root / "experiment_support/run_full_holdout.py"), args)
        self.assertEqual(args[-4:], ["--causal-samples", "1000", "--marginal-samples", "all"])
        archives = list((self.root / "archive").glob("results_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "models.json").read_bytes(), old_models)

    def test_recompute_archives_stale_outputs_and_keeps_cache_even_after_failed_run(self):
        for completed in (True, False):
            with self.subTest(completed=completed):
                value = self.write_manifest()
                value["completed"] = completed
                (self.results / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
                old_case = self.results / "figures/b3_case_strong_10.png"
                old_case.parent.mkdir(exist_ok=True)
                old_case.write_bytes(b"old selected case image")
                cache = self.results / "cache/verified_numeric_draw.npz"
                cache.parent.mkdir(exist_ok=True)
                cache.write_bytes(b"reusable numeric cache")
                exported = self.root / support.INTEGRATED / "figures/B3-11.png"
                exported.parent.mkdir(parents=True)
                exported.write_bytes(b"old notebook export")
                before = set((self.root / "archive").glob("results_*"))
                process = MagicMock()
                process.__enter__.return_value = process
                process.stdout = io.StringIO("")
                process.wait.return_value = 0
                with patch.object(support.subprocess, "Popen", return_value=process):
                    support.recompute_results(self.root)
                archived = set((self.root / "archive").glob("results_*")) - before
                self.assertEqual(len(archived), 1)
                destination = archived.pop()
                self.assertFalse(old_case.exists())
                self.assertFalse(exported.exists())
                self.assertEqual(
                    (destination / "figures" / old_case.name).read_bytes(),
                    b"old selected case image",
                )
                self.assertEqual(
                    (destination / support.INTEGRATED / "figures/B3-11.png").read_bytes(),
                    b"old notebook export",
                )
                self.assertEqual(
                    json.loads((destination / "manifest.json").read_text())["completed"], completed
                )
                self.assertEqual(cache.read_bytes(), b"reusable numeric cache")
                self.assertFalse((destination / "cache").exists())

    def test_calculation_lock_rejects_concurrent_owner_without_releasing_first_lock(self):
        lock_path = self.root / ".experiment_run.lock"
        with support.calculation_lock(self.root):
            self.assertTrue(lock_path.is_file())
            first_owner = lock_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Another notebook"):
                with support.calculation_lock(self.root):
                    self.fail("A second live owner must not enter the protected calculation.")
            self.assertEqual(lock_path.read_bytes(), first_owner)
        self.assertFalse(lock_path.exists())
        with support.calculation_lock(self.root):
            self.assertTrue(lock_path.is_file())
        self.assertFalse(lock_path.exists())

    def test_calculation_lock_releases_after_exception(self):
        lock_path = self.root / ".experiment_run.lock"
        with self.assertRaisesRegex(ValueError, "simulated calculation failure"):
            with support.calculation_lock(self.root):
                raise ValueError("simulated calculation failure")
        self.assertFalse(lock_path.exists())
        with support.calculation_lock(self.root):
            self.assertTrue(lock_path.is_file())


if __name__ == "__main__":
    unittest.main()
