from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "okun_us_time_series_report.md"


def read_report() -> str:
    return REPORT.read_text(encoding="utf-8")


class ReportReviewFixesTest(unittest.TestCase):
    def test_report_mentions_bidirectional_granger_link(self) -> None:
        text = read_report()
        self.assertIn("g ⇄ Δu", text)
        self.assertIn("двусторон", text.lower())

    def test_adl_selection_is_named_as_interpretation_not_unconditional_best(self) -> None:
        text = read_report()
        self.assertIn("Выбрана для интерпретации", text)
        self.assertNotIn("Лучшая модель", text)

    def test_covid_sensitivity_and_excluding_2020_correlation_are_reported(self) -> None:
        text = read_report()
        self.assertIn("2020Q2–2020Q3", text)
        self.assertIn("без 2020Q2–2020Q3", text)

    def test_irf_and_fevd_order_dependence_is_explicit(self) -> None:
        text = read_report()
        self.assertIn("IRF и FEVD построены для ортогональных шоков", text)
        self.assertIn("g, Δu, π, Δi", text)

    def test_ecm_vecm_diagnostic_block_is_present(self) -> None:
        text = read_report()
        self.assertIn("Диагностическая VECM/ECM-проверка", text)
        self.assertIn("VECM как ECM-представление", text)
        self.assertIn("vecm_diagnostic_summary.csv", text)

    def test_calculation_screenshot_appendix_is_present(self) -> None:
        text = read_report()
        self.assertIn("Приложение C. Расчетные приложения Python", text)
        self.assertIn("calculation_screenshots", text)
        expected = [
            ROOT / "reports" / "calculation_screenshots" / "calc_01_stationarity.png",
            ROOT / "reports" / "calculation_screenshots" / "calc_02_adl_var_models.png",
            ROOT / "reports" / "calculation_screenshots" / "calc_03_vecm_ecm.png",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing {path}")
            self.assertGreater(path.stat().st_size, 1000, f"Empty {path}")


if __name__ == "__main__":
    unittest.main()
