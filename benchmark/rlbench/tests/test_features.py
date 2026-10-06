"""Offline tests for the feature contract (no cluster, stdlib unittest).

    python3 -m unittest discover -s benchmark/rlbench/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlbench.cli import _render_run_config  # noqa: E402
from rlbench.setup_folder import (  # noqa: E402
    FeatureFolder,
    SetupFolder,
    SetupFolderError,
    feature_builtins,
    feature_variables,
    render_manifest,
)


def make_setup(root: Path) -> Path:
    setup = root / "my-setup"
    (setup / "setup").mkdir(parents=True)
    (setup / "setup" / "00-ns.yaml").write_text("apiVersion: v1\nkind: Namespace\nmetadata:\n  name: ns-${RUN_ID}\n")
    (setup / "job.yaml").write_text(
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: j\n  namespace: ns-${RUN_ID}\n"
        "spec:\n  template:\n    spec:\n      containers: []\n"
    )
    (setup / "config" / "rung").mkdir(parents=True)
    (setup / "config" / "rung" / "run.sh").write_text("export KNOB=${KNOB:-base}\n")
    f = setup / "features" / "fancy-router"
    (f / "setup").mkdir(parents=True)
    (f / "config").mkdir()
    (f / "hooks").mkdir()
    (f / "vars.env").write_text("# comment\nKNOB=feature\nROUTER_PROFILE=p1\n\n")
    (f / "setup" / "10-cm.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: router-${ROUTER_PROFILE}\n  namespace: ns-${RUN_ID}\n"
        "data:\n  features: '${RLBENCH_FEATURES}'\n"
    )
    (f / "config" / "feature-fancy-router.sh").write_text("export ROUTER=${ROUTER_PROFILE}\n")
    (f / "config" / "router.yaml").write_text("profile: ${ROUTER_PROFILE}\n")
    (f / "hooks" / "post-run.sh").write_text("#!/bin/sh\n")
    (f / "scrape-targets.txt").write_text("pods ns-x app=router 9090 /metrics\n")
    (setup / "features" / "other").mkdir()
    return setup


class FeatureFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.setup = SetupFolder.load(make_setup(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_parses_contract(self):
        f = self.setup.feature("fancy-router")
        self.assertEqual(f.vars, {"KNOB": "feature", "ROUTER_PROFILE": "p1"})
        self.assertEqual([m.name for m in f.setup_manifests], ["10-cm.yaml"])
        self.assertEqual([c.name for c in f.config_files], ["feature-fancy-router.sh", "router.yaml"])
        self.assertEqual(f.scrape_targets(), [("ns-x", "app=router", "9090", "/metrics")])
        self.assertIsNotNone(f.hook("post-run.sh"))
        self.assertIsNone(f.hook("pre-setup.sh"))
        self.assertEqual(f.env_name, "FEATURE_FANCY_ROUTER")

    def test_unknown_feature_lists_available(self):
        with self.assertRaises(SetupFolderError) as cm:
            self.setup.feature("nope")
        self.assertIn("fancy-router", str(cm.exception))
        with self.assertRaises(SetupFolderError):
            self.setup.feature("Bad_Name")

    def test_bad_vars_env_line(self):
        f = self.setup.feature("fancy-router")
        (f.root / "vars.env").write_text("not a var\n")
        with self.assertRaises(SetupFolderError):
            FeatureFolder.load(self.setup.root, "fancy-router")

    def test_builtins_and_precedence(self):
        f = self.setup.feature("fancy-router")
        self.assertEqual(feature_builtins([]), {"RLBENCH_FEATURES": ""})
        self.assertEqual(feature_builtins([f]), {"RLBENCH_FEATURES": "fancy-router", "FEATURE_FANCY_ROUTER": "1"})
        env = {"KNOB": "env", "ROUTER_PROFILE": "env"}
        var = {"ROUTER_PROFILE": "cli"}
        merged = {**env, **feature_variables([f]), **var, **feature_builtins([f])}
        self.assertEqual(merged["KNOB"], "feature")        # vars.env beats environment
        self.assertEqual(merged["ROUTER_PROFILE"], "cli")  # --var beats vars.env

    def test_feature_manifest_renders_with_builtins_and_label(self):
        f = self.setup.feature("fancy-router")
        variables = {**feature_variables([f]), "RUN_ID": "r1", **feature_builtins([f])}
        docs = render_manifest(f.setup_manifests[0], variables)
        self.assertEqual(docs[0]["metadata"]["name"], "router-p1")
        self.assertEqual(docs[0]["data"]["features"], "fancy-router")
        self.assertEqual(docs[0]["metadata"]["labels"]["rlbench/run"], "r1")

    def test_render_run_config_flattens_feature_files(self):
        f = self.setup.feature("fancy-router")
        variables = {**feature_variables([f]), "RUN_ID": "r1", **feature_builtins([f])}
        out = Path(self.tmp.name) / "run" / "config"
        out.mkdir(parents=True)
        rendered = _render_run_config(out, self.setup.root / "config" / "rung",
                                      [(f, c) for c in f.config_files], variables)
        self.assertEqual(sorted(p.name for p in rendered.iterdir()),
                         ["feature-fancy-router.sh", "router.yaml", "run.sh"])
        self.assertEqual((rendered / "run.sh").read_text(), "export KNOB=feature\n")
        self.assertEqual((rendered / "router.yaml").read_text(), "profile: p1\n")

    def test_render_run_config_without_config_dir(self):
        f = self.setup.feature("fancy-router")
        out = Path(self.tmp.name) / "run2" / "config"
        out.mkdir(parents=True)
        rendered = _render_run_config(out, None, [(f, c) for c in f.config_files], {"RUN_ID": "r1"})
        self.assertEqual(rendered.name, "run-config")
        self.assertTrue((rendered / "router.yaml").exists())

    def test_render_run_config_collision(self):
        f = self.setup.feature("fancy-router")
        (self.setup.root / "config" / "rung" / "router.yaml").write_text("x")
        out = Path(self.tmp.name) / "run3" / "config"
        out.mkdir(parents=True)
        with self.assertRaises(ValueError):
            _render_run_config(out, self.setup.root / "config" / "rung", [(f, c) for c in f.config_files], {})

    def test_features_json_shape(self):
        from rlbench.runfolder import RunFolder
        run = RunFolder(Path(self.tmp.name) / "runs", "20260101-000000-x")
        run.record_features([self.setup.feature("fancy-router")])
        data = json.loads((run.config / "features.json").read_text())
        self.assertEqual(data[0]["name"], "fancy-router")
        self.assertEqual(data[0]["vars"]["KNOB"], "feature")


if __name__ == "__main__":
    unittest.main()
