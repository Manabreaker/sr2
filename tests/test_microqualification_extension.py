from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "okun_us_time_series_report.md"
MICRO = ROOT / "reports" / "microqualification"


class MicroqualificationExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = REPORT.read_text(encoding="utf-8")

    def test_main_report_contains_microqualification_section(self) -> None:
        self.assertIn("## 10. Микроквалификация", self.text)
        self.assertIn("Кластеризация стран", self.text)
        self.assertIn("типичный представитель", self.text.lower())

    def test_required_microqualification_artifacts_exist(self) -> None:
        expected = [
            MICRO / "data" / "world_bank_macro_panel.csv",
            MICRO / "tables" / "country_cluster_assignments.csv",
            MICRO / "tables" / "cluster_characteristics.csv",
            MICRO / "tables" / "distance_method_comparison.csv",
            MICRO / "figures" / "10_world_map_kmeans_clusters.png",
            MICRO / "figures" / "11_cluster_pca_scatter.png",
            MICRO / "figures" / "12_hierarchical_dendrogram.png",
            MICRO / "figures" / "13_distance_heatmap.png",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing {path}")
            self.assertGreater(path.stat().st_size, 1000, f"Empty {path}")

    def test_report_links_to_existing_microqualification_figures(self) -> None:
        figure_links = re.findall(r"!\[[^\]]+\]\((microqualification/figures/[^)]+)\)", self.text)
        self.assertGreaterEqual(len(figure_links), 4)
        for link in figure_links:
            path = REPORT.parent / link
            self.assertTrue(path.exists(), f"Broken image link: {link}")


if __name__ == "__main__":
    unittest.main()
