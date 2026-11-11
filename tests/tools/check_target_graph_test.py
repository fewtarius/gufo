#!/usr/bin/env python3
"""Regression coverage for the serving/runtime link-graph guard (Card 09)."""

import importlib.util
import json
import tempfile
import sys
import unittest
from pathlib import Path

BUILD_DIR = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_target_graph", ROOT / "tools/ci/check-target-graph.py")
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


class LinkGraphTest(unittest.TestCase):
    def report(self, targets, forbidden=("gufo_serving", "gufo_http")):
        return {"hip": True, "absent": [], "excluded": [],
                "forbidden": list(forbidden), "targets": targets}

    def test_clean_graph_has_no_violation(self):
        report = self.report({
            "gufo_ext_runtime": ["gufo_logging", "gufo_platform"],
            "gufo_logging": ["gufo_platform"],
            "gufo_platform": [],
            "gufo_serving": ["gufo_core", "gufo_ext_runtime"],
        })
        self.assertEqual(GRAPH.violations(report, ["gufo_ext_runtime"]), [])

    def test_direct_edge_to_serving_fails(self):
        report = self.report({
            "gufo_ext_runtime": ["gufo_logging", "gufo_serving"],
            "gufo_serving": ["gufo_core"],
        })
        problems = GRAPH.violations(report, ["gufo_ext_runtime"])
        self.assertEqual(len(problems), 1)
        self.assertIn("gufo_ext_runtime reaches gufo_serving", problems[0])

    def test_edge_through_an_intermediate_library_fails(self):
        # The closure is transitive, so hiding the edge one level down is not a
        # way past the guard.
        report = self.report({
            "gufo_cache": ["gufo_platform", "gufo_http"],
            "gufo_platform": [],
            "gufo_http": ["gufo_core"],
        })
        problems = GRAPH.violations(report, ["gufo_cache"])
        self.assertEqual(len(problems), 1)
        self.assertIn("gufo_cache reaches gufo_http", problems[0])

    def test_missing_protected_target_fails(self):
        # A library that never made it into the report must not pass silently.
        report = self.report({"gufo_core": ["gufo_platform"]})
        problems = GRAPH.violations(report, ["gufo_core", "gufo_new_model"])
        self.assertIn("protected target gufo_new_model is missing from the "
                      "link graph", problems)

    def test_absent_target_covers_a_configuration_that_skips_it(self):
        # The HIP tiers are undefined in a HIP-off build and are reported as
        # absent, which is coverage, not a hole.
        report = self.report({"gufo_core": ["gufo_platform"]})
        report["absent"] = ["gufo_cache_hip"]
        report["hip"] = False
        self.assertEqual(
            GRAPH.violations(report, ["gufo_core", "gufo_cache_hip"]), [])

    def test_absent_target_still_blocks_a_forbidden_edge(self):
        report = self.report({"gufo_cache_hip": ["gufo_http"]})
        report["absent"] = []
        problems = GRAPH.violations(report, ["gufo_cache_hip"])
        self.assertEqual(len(problems), 1)
        self.assertIn("gufo_cache_hip reaches gufo_http", problems[0])

    def test_absent_non_hip_target_fails(self):
        report = self.report({})
        report.update(hip=False, absent=["gufo_core"])
        self.assertTrue(GRAPH.violations(report, ["gufo_core"]))

    def test_absent_hip_target_in_hip_build_fails(self):
        report = self.report({})
        report["absent"] = ["gufo_cache_hip"]
        self.assertTrue(GRAPH.violations(report, ["gufo_cache_hip"]))

    def test_platform_excluded_target_covers_the_gap(self):
        # An extension whose manifest leaves this platform out has no target
        # to report, and its libraries are recorded as excluded.
        report = self.report({"gufo_core": ["gufo_platform"]})
        report["excluded"] = ["gufo_laguna"]
        self.assertEqual(
            GRAPH.violations(report, ["gufo_core", "gufo_laguna"]), [])

    def test_excluded_target_cannot_also_be_present(self):
        report = self.report({"gufo_laguna": ["gufo_core"]})
        report["excluded"] = ["gufo_laguna"]
        problems = GRAPH.violations(report, ["gufo_laguna"])
        self.assertEqual(len(problems), 1)
        self.assertIn("both present and excluded", problems[0])

    def test_excluded_and_absent_cannot_overlap(self):
        report = self.report({})
        report["excluded"] = ["gufo_laguna"]
        report["absent"] = ["gufo_laguna"]
        problems = GRAPH.violations(report, ["gufo_laguna"])
        self.assertTrue(any("both absent and excluded" in problem
                            for problem in problems), problems)

    def test_report_cannot_disable_forbidden_policy(self):
        report = self.report({"gufo_core": ["gufo_http"]}, forbidden=())
        problems = GRAPH.violations(report, ["gufo_core"])
        self.assertEqual(len(problems), 2)

    def test_missing_hip_configuration_fails(self):
        report = self.report({})
        del report["hip"]
        self.assertTrue(GRAPH.violations(report, []))

    def test_manifest_targets_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, target in (
                    ("src/models/demo/library.json", "gufo_demo"),
                    ("src/families/demo/text/library.json", "gufo_demo_text")):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"schema": 1, "name": name,
                                            "target": target}) + "\n")
            self.assertEqual(GRAPH.manifest_targets(root),
                             ["gufo_demo", "gufo_demo_text"])

    def test_repository_link_graph_is_clean(self):
        if BUILD_DIR is None:
            self.skipTest("supply the current CMake binary directory for integration")
        report = BUILD_DIR / "gufo-link-graph.json"
        self.assertTrue(report.is_file(), f"missing configured report: {report}")
        document = json.loads(report.read_text(encoding="utf-8"))
        protected = sorted(set(GRAPH.PROTECTED)
                           | set(GRAPH.manifest_targets(ROOT)))
        self.assertEqual(GRAPH.violations(document, protected), [])


if __name__ == "__main__":
    unittest.main()
