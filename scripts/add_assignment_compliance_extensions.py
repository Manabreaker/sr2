#!/usr/bin/env python3
"""Add assignment-compliance blocks to the current Markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM


REPORT = ROOT / "reports" / "okun_us_time_series_report.md"
TABLE_DIR = ROOT / "reports" / "tables"
SHOT_DIR = ROOT / "reports" / "calculation_screenshots"
SHOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
    }
)


def fmt_num(value: float | int | np.floating, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}".replace(",", " ")
    return f"{value:.{digits}f}"


def fmt_p(value: float | np.floating) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    work = df.copy()
    if index:
        work = work.reset_index()
        if str(work.columns[0]) == "index":
            work = work.rename(columns={work.columns[0]: df.index.name or "Показатель"})
    headers = [str(c) for c in work.columns]
    rows: list[list[str]] = []
    for _, row in work.iterrows():
        current = []
        for value in row:
            if isinstance(value, (float, np.floating, int, np.integer)):
                current.append(fmt_num(value, digits=digits))
            else:
                current.append("" if pd.isna(value) else str(value))
        rows.append(current)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index, encoding="utf-8")
    (TABLE_DIR / f"{name}.md").write_text(markdown_table(df, index=index), encoding="utf-8")


def render_table_image(title: str, tables: list[tuple[str, pd.DataFrame]], filename: str) -> Path:
    fig_height = 1.1 + sum(max(2.0, 0.42 * (len(df) + 2)) for _, df in tables)
    fig, axes = plt.subplots(len(tables), 1, figsize=(15, fig_height), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    for ax, (subtitle, df) in zip(axes.flat, tables):
        ax.axis("off")
        display = df.copy()
        for col in display.columns:
            display[col] = display[col].map(lambda value: fmt_num(value) if isinstance(value, (float, np.floating, int, np.integer)) else str(value))
        ax.set_title(subtitle, loc="left", fontsize=12, fontweight="bold", pad=8)
        table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e9eef5")
            cell.set_edgecolor("#c7c7c7")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = SHOT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_vecm_ecm_tables() -> dict[str, pd.DataFrame]:
    levels = pd.read_csv(ROOT / "reports" / "data" / "fred_quarterly_us_macro_1990_2026.csv")
    levels["quarter"] = pd.PeriodIndex(levels["quarter"], freq="Q")
    levels = levels.set_index("quarter")
    model_levels = pd.DataFrame(
        {
            "ln_gdp": np.log(levels["GDPC1"]),
            "unrate": levels["UNRATE"],
            "ln_cpi": np.log(levels["CPIAUCSL"]),
            "fedfunds": levels["FEDFUNDS"],
        },
        index=levels.index,
    ).dropna()

    vecm = VECM(model_levels, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    variables = list(model_levels.columns)
    alpha_beta = pd.DataFrame(
        {
            "variable": variables,
            "alpha_adjustment": vecm.alpha[:, 0],
            "beta_cointegration": vecm.beta[:, 0],
        }
    )
    save_table(alpha_beta, "vecm_alpha_beta", index=False)

    summary = pd.DataFrame(
        [
            {"Показатель": "Модель", "Значение": "VECM как ECM-представление"},
            {"Показатель": "Переменные", "Значение": ", ".join(variables)},
            {"Показатель": "k_ar_diff", "Значение": 1},
            {"Показатель": "coint_rank", "Значение": 1},
            {"Показатель": "deterministic", "Значение": "co"},
            {"Показатель": "log-likelihood", "Значение": vecm.llf},
            {"Показатель": "nobs", "Значение": vecm.nobs},
        ]
    )
    save_table(summary, "vecm_diagnostic_summary", index=False)

    beta = pd.Series(vecm.beta[:, 0], index=variables)
    ect = model_levels @ beta
    diffs = model_levels.diff()
    ecm_data = pd.DataFrame(
        {
            "d_unrate": diffs["unrate"],
            "ECT_L1": ect.shift(1),
            "d_ln_gdp_L1": diffs["ln_gdp"].shift(1),
            "d_unrate_L1": diffs["unrate"].shift(1),
            "d_ln_cpi_L1": diffs["ln_cpi"].shift(1),
            "d_fedfunds_L1": diffs["fedfunds"].shift(1),
        }
    ).dropna()
    ecm = sm.OLS(ecm_data["d_unrate"], sm.add_constant(ecm_data.drop(columns=["d_unrate"]))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}
    )
    ecm_table = pd.DataFrame(
        {
            "Параметр": ecm.params.index,
            "coef": ecm.params.values,
            "std err HAC": ecm.bse.values,
            "t": ecm.tvalues.values,
            "p-value": ecm.pvalues.values,
        }
    )
    ecm_table["p-value"] = ecm_table["p-value"].map(fmt_p)
    save_table(ecm_table, "ecm_unrate_equation", index=False)

    return {
        "vecm_summary": summary,
        "alpha_beta": alpha_beta,
        "ecm_table": ecm_table,
    }


def build_screenshot_images(vecm_tables: dict[str, pd.DataFrame]) -> None:
    stationarity = pd.read_csv(TABLE_DIR / "stationarity_adf_kpss.csv")
    stationarity = stationarity.head(10)
    adl = pd.read_csv(TABLE_DIR / "adl_model_comparison.csv")
    var = pd.read_csv(TABLE_DIR / "var_model_comparison.csv")
    granger = pd.read_csv(TABLE_DIR / "strongest_granger_links.csv")
    cluster = pd.read_csv(ROOT / "reports" / "microqualification" / "tables" / "cluster_characteristics.csv")

    render_table_image(
        "Расчетные таблицы Python: стационарность",
        [("ADF/KPSS: уровни и преобразованные ряды", stationarity)],
        "calc_01_stationarity.png",
    )
    render_table_image(
        "Расчетные таблицы Python: модели и причинность",
        [
            ("ADL-спецификации", adl),
            ("VAR-спецификации", var),
            ("Наиболее сильные связи Грейнджера", granger),
        ],
        "calc_02_adl_var_models.png",
    )
    render_table_image(
        "Расчетные таблицы Python: VECM/ECM",
        [
            ("VECM summary", vecm_tables["vecm_summary"]),
            ("VECM alpha/beta", vecm_tables["alpha_beta"]),
            ("ECM equation for Δu", vecm_tables["ecm_table"]),
        ],
        "calc_03_vecm_ecm.png",
    )
    render_table_image(
        "Расчетные таблицы Python: кластеры стран",
        [("Характеристики кластеров", cluster)],
        "calc_04_micro_clusters.png",
    )


def insert_or_replace(text: str, start_marker: str, end_marker: str, block: str) -> str:
    if start_marker in text:
        start = text.index(start_marker)
        end = text.index(end_marker, start) if end_marker in text[start:] else len(text)
        return text[:start].rstrip() + "\n\n" + block.strip() + "\n\n" + text[end:].lstrip()
    insert_at = text.index(end_marker) if end_marker in text else len(text)
    return text[:insert_at].rstrip() + "\n\n" + block.strip() + "\n\n" + text[insert_at:].lstrip()


def patch_report(vecm_tables: dict[str, pd.DataFrame]) -> None:
    text = REPORT.read_text(encoding="utf-8")
    text = text.replace(
        "ECM/VECM не используется как основная модель.",
        "ECM/VECM не используется как основная модель; ниже приведена диагностическая VECM/ECM-проверка, чтобы явно показать ограничение этого подхода для текущих данных.",
    )
    text = text.replace(
        "ECM/VECM в работе не используется как основная модель, потому что для этого нужна более уверенная коинтеграционная структура.",
        "ECM/VECM не используется как основная модель, потому что для этого нужна более уверенная коинтеграционная структура. При этом диагностическая VECM/ECM-проверка оценена отдельно и используется как проверка устойчивости, а не как базовая спецификация.",
    )

    vecm_block = f"""
### 5.3. Диагностическая VECM/ECM-проверка

Так как тест Йохансена показывает один возможный коинтеграционный вектор, дополнительно оценена VECM как ECM-представление системы уровней. Эта модель не выбрана основной из-за смешанных выводов о порядке интегрируемости безработицы и ставки, но она закрывает проверку коинтеграционного ECM-подхода и показывает, насколько быстро переменные реагируют на отклонение от долгосрочного соотношения.

Сводка VECM:

{markdown_table(vecm_tables["vecm_summary"], index=False)}

Коэффициенты корректировки `alpha` и коинтеграционный вектор `beta`:

{markdown_table(vecm_tables["alpha_beta"], index=False)}

Также оценено одно ECM-уравнение для прироста безработицы:

$$\\Delta u_t = c + \\lambda ECT_{{t-1}} + \\Gamma_1\\Delta y_{{t-1}} + \\varepsilon_t,$$

где `ECT` построен на основе оцененного коинтеграционного вектора VECM.

{markdown_table(vecm_tables["ecm_table"], index=False)}

Файлы расчетов сохранены в `tables/vecm_diagnostic_summary.csv`, `tables/vecm_alpha_beta.csv` и `tables/ecm_unrate_equation.csv`. Вывод: ECM/VECM дает диагностически полезное представление долгосрочной связи, но для интерпретации основной динамики в работе надежнее использовать ADL и VAR на стационарных преобразованиях.
"""
    if "### 5.3. Диагностическая VECM/ECM-проверка" in text:
        start = text.index("### 5.3. Диагностическая VECM/ECM-проверка")
        end = text.index("## 6. Спецификации моделей", start)
        text = text[:start].rstrip() + "\n\n" + vecm_block.strip() + "\n\n" + text[end:].lstrip()
    else:
        marker = "## 6. Спецификации моделей"
        text = text[: text.index(marker)].rstrip() + "\n\n" + vecm_block.strip() + "\n\n" + text[text.index(marker) :].lstrip()

    appendix_c = """
## Приложение C. Расчетные приложения Python

В PDF-задании требуется привести скрины расчетов из статистического пакета. В этой версии отчета расчеты выполнены воспроизводимым Python-кодом, поэтому в приложение добавлены PNG-таблицы с ключевыми результатами. Полные форматированные таблицы также сохранены в CSV/MD-файлах.

![Расчет стационарности](calculation_screenshots/calc_01_stationarity.png)

![Расчет ADL/VAR/Грейнджера](calculation_screenshots/calc_02_adl_var_models.png)

![Расчет VECM/ECM](calculation_screenshots/calc_03_vecm_ecm.png)

![Расчет кластеров стран](calculation_screenshots/calc_04_micro_clusters.png)

Код расчетов:

- `../scripts/build_time_series_report.py` — основная часть 1–8;
- `../scripts/build_microqualification_clustering.py` — микроквалификация;
- `../scripts/add_assignment_compliance_extensions.py` — VECM/ECM-дополнение и расчетные PNG-приложения.
"""
    text = insert_or_replace(text, "## Приложение C. Расчетные приложения Python", "## Приложение B.", appendix_c)
    text = text.replace("## Приложение B. Источники", "## Приложение B. Список литературы и источники")
    text = text.replace(
        "* Скрипт генерации отчета: `../scripts/build_time_series_report.py`",
        "* Скрипт генерации основной части: `../scripts/build_time_series_report.py`\n"
        "* Скрипт микроквалификации: `../scripts/build_microqualification_clustering.py`\n"
        "* Скрипт VECM/ECM и расчетных приложений: `../scripts/add_assignment_compliance_extensions.py`",
    )
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    vecm_tables = build_vecm_ecm_tables()
    build_screenshot_images(vecm_tables)
    patch_report(vecm_tables)
    print("Added VECM/ECM diagnostics and Python calculation appendix")
    print(f"Screenshots: {len(list(SHOT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
