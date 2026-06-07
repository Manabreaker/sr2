from pathlib import Path
import os
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_core_analysis.py"
REPORT = ROOT / "reports" / "okun_us_time_series_report.md"


class CoreAnalysisRunnerTest(unittest.TestCase):
    def test_runner_file_exists(self) -> None:
        self.assertTrue(SCRIPT.exists(), f"Missing {SCRIPT}")

    def test_runner_does_not_generate_reports_or_presentations(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden_fragments = [
            "REPORT.write_text",
            "insert_micro_section",
            "build_micro_markdown",
            "patch_report(",
            "Presentation(",
            "prs.save",
            ".docx",
            ".pptx",
        ]
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, text)

    def test_runner_generates_core_artifacts_without_touching_main_report(self) -> None:
        before = REPORT.read_text(encoding="utf-8") if REPORT.exists() else None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / ".python_deps")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=240,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        if before is not None:
            self.assertEqual(before, REPORT.read_text(encoding="utf-8"))

        expected = [
            ROOT / "reports" / "figures" / "01_levels_us_macro_series.png",
            ROOT / "reports" / "figures" / "07_var_irf_unemployment_response.png",
            ROOT / "reports" / "tables" / "best_adl_coefficients.csv",
            ROOT / "reports" / "tables" / "vecm_diagnostic_summary.csv",
            ROOT / "reports" / "calculation_screenshots" / "calc_03_vecm_ecm.png",
            ROOT / "reports" / "microqualification" / "figures" / "10_world_map_kmeans_clusters.png",
            ROOT / "reports" / "microqualification" / "tables" / "country_cluster_assignments.csv",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing {path}")
            min_size = 1000 if path.suffix.lower() == ".png" else 50
            self.assertGreater(path.stat().st_size, min_size, f"Empty {path}")


if __name__ == "__main__":
    unittest.main()
