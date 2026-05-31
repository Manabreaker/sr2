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


if __name__ == "__main__":
    unittest.main()
