#!/usr/bin/env python3
"""Build a Markdown time-series econometrics report with FRED data."""

from __future__ import annotations

import math
import sys
import textwrap
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyArrowPatch
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, adfuller, coint, kpss, pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order


OUT = ROOT / "reports"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
REPORT = OUT / "okun_us_time_series_report.md"

for directory in (FIG, TAB, DATA):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
    }
)

SERIES_META = {
    "GDPC1": {
        "label_ru": "Реальный ВВП США",
        "short": "Реальный ВВП",
        "unit": "млрд цепных долларов 2017 г., SAAR",
        "frequency": "квартальная",
        "source": "BEA через FRED",
        "url": "https://fred.stlouisfed.org/series/GDPC1",
    },
    "UNRATE": {
        "label_ru": "Уровень безработицы",
        "short": "Безработица",
        "unit": "% рабочей силы, сезонно скорректировано",
        "frequency": "месячная, усреднена до квартала",
        "source": "BLS через FRED",
        "url": "https://fred.stlouisfed.org/series/UNRATE",
    },
    "CPIAUCSL": {
        "label_ru": "Индекс потребительских цен CPI-U",
        "short": "CPI",
        "unit": "индекс 1982-1984=100, сезонно скорректировано",
        "frequency": "месячная, усреднена до квартала",
        "source": "BLS через FRED",
        "url": "https://fred.stlouisfed.org/series/CPIAUCSL",
    },
    "FEDFUNDS": {
        "label_ru": "Эффективная ставка федеральных фондов",
        "short": "Ставка Fed funds",
        "unit": "% годовых, среднее за месяц",
        "frequency": "месячная, усреднена до квартала",
        "source": "Board of Governors через FRED",
        "url": "https://fred.stlouisfed.org/series/FEDFUNDS",
    },
}

LEVEL_LABELS = {
    "ln_gdp": "ln(реального ВВП)",
    "unrate": "безработица",
    "ln_cpi": "ln(CPI)",
    "fedfunds": "ставка Fed funds",
}

TRANS_LABELS = {
    "gdp_growth": "темп роста реального ВВП",
    "d_unrate": "прирост безработицы",
    "inflation": "инфляция CPI",
    "d_fedfunds": "прирост ставки Fed funds",
}

MODEL_LABELS = {
    "gdp_growth": "g, рост ВВП",
    "d_unrate": "Δu, прирост безработицы",
    "inflation": "π, инфляция CPI",
    "d_fedfunds": "Δi, прирост ставки",
}


def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def load_fred_series(series_id: str) -> pd.Series:
    url = fred_url(series_id)
    raw = pd.read_csv(url, parse_dates=["observation_date"], na_values=[".", ""])
    raw = raw.rename(columns={"observation_date": "date"})
    raw[series_id] = pd.to_numeric(raw[series_id], errors="coerce")
    s = raw.set_index("date")[series_id].dropna().sort_index()
    (DATA / f"fred_raw_{series_id}.csv").write_text(raw.to_csv(index=False), encoding="utf-8")
    return s


def to_quarterly(series_id: str, s: pd.Series) -> pd.Series:
    if series_id == "GDPC1":
        q = s.copy()
        q.index = q.index.to_period("Q")
        return q
    q = s.resample("QS").mean()
    q.index = q.index.to_period("Q")
    return q


def fmt_num(value: float | int | np.floating, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{value:d}"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}".replace(",", " ")
    return f"{value:.{digits}f}"


def fmt_p(value: float | np.floating) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    work = df.copy()
    if index:
        work = work.reset_index()
        first_col = work.columns[0]
        if str(first_col) == "index":
            work = work.rename(columns={first_col: df.index.name or "Показатель"})
    headers = [str(c) for c in work.columns]
    rows: list[list[str]] = []
    for _, row in work.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating, int, np.integer)):
                values.append(fmt_num(value, digits=digits))
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append(values)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    line_header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    line_sep = "| " + " | ".join("-" * w for w in widths) + " |"
    line_rows = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([line_header, line_sep, *line_rows])


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=True, encoding="utf-8")
    (TAB / f"{name}.md").write_text(markdown_table(df, index=True), encoding="utf-8")


def period_to_ts(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp(how="end")


def test_adf(series: pd.Series, regression: str) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(series.dropna(), autolag="AIC", regression=regression)
    return float(result[0]), float(result[1])


def test_kpss(series: pd.Series, regression: str) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression=regression, nlags="auto")
    return float(result[0]), float(result[1])


def stationarity_conclusion(adf_p: float, kpss_p: float) -> str:
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05
    if adf_stationary and kpss_stationary:
        return "стационарен"
    if (not adf_stationary) and (not kpss_stationary):
        return "нестационарен"
    return "смешанный вывод"


def lagged(frame: pd.DataFrame, column: str, lag: int) -> pd.Series:
    return frame[column].shift(lag).rename(f"{column}_L{lag}")


def fit_adl(transformed: pd.DataFrame, name: str, g_lags: int, du_lags: int, covid: bool = False):
    y = transformed["d_unrate"].rename("d_unrate")
    parts = [transformed["gdp_growth"].rename("gdp_growth")]
    for lag in range(1, g_lags + 1):
        parts.append(lagged(transformed, "gdp_growth", lag))
    for lag in range(1, du_lags + 1):
        parts.append(lagged(transformed, "d_unrate", lag))
    if covid:
        dummy = pd.Series((transformed.index == pd.Period("2020Q2")).astype(int), index=transformed.index, name="D_2020Q2")
        parts.append(dummy)
    x = pd.concat(parts, axis=1)
    data = pd.concat([y, x], axis=1).dropna()
    model = sm.OLS(data["d_unrate"], sm.add_constant(data.drop(columns=["d_unrate"])))
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    residuals = result.resid
    ljung_p = acorr_ljungbox(residuals, lags=[min(8, max(2, len(residuals) // 8))], return_df=True)["lb_pvalue"].iloc[0]
    jb = jarque_bera(residuals)
    rmse = math.sqrt(float(np.mean(np.square(residuals))))
    return {
        "name": name,
        "result": result,
        "nobs": int(result.nobs),
        "rmse": rmse,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "ljung_p": float(ljung_p),
        "normal_p": float(jb[1]),
        "r2_adj": float(result.rsquared_adj),
    }


def var_params_frame(fit, columns: list[str]) -> pd.DataFrame:
    raw = np.asarray(fit._results.params)
    lag_count = int(fit.k_ar)
    lag_param_count = lag_count * len(columns)
    deterministic_count = raw.shape[0] - lag_param_count
    names: list[str] = []
    if deterministic_count == 1:
        names.append("const")
    elif deterministic_count == 2:
        names.extend(["const", "trend"])
    elif deterministic_count > 0:
        names.extend([f"det_{i + 1}" for i in range(deterministic_count)])
    for lag in range(1, lag_count + 1):
        for col in columns:
            names.append(f"L{lag}.{col}")
    if len(names) != raw.shape[0]:
        names = [f"param_{i + 1}" for i in range(raw.shape[0])]
    return pd.DataFrame(raw, index=names, columns=columns)


def var_resid_frame(fit, columns: list[str]) -> pd.DataFrame:
    raw = np.asarray(fit._results.resid)
    model_index = fit.model.data.row_labels
    if model_index is None:
        index = pd.RangeIndex(raw.shape[0])
    else:
        index = pd.Index(model_index[-raw.shape[0] :])
    return pd.DataFrame(raw, index=index, columns=columns)


def plot_levels(levels: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    variables = [
        ("GDPC1", "Реальный ВВП, млрд цепных долл. 2017"),
        ("UNRATE", "Безработица, %"),
        ("CPIAUCSL", "CPI, индекс"),
        ("FEDFUNDS", "Ставка федеральных фондов, %"),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    dates = period_to_ts(levels.index)
    for ax, (col, title), color in zip(axes.flat, variables, colors):
        ax.plot(dates, levels[col], color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("")
    path = FIG / "01_levels_us_macro_series.png"
    fig.suptitle("Исходные квартальные временные ряды, США", fontsize=14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transformed(transformed: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    dates = period_to_ts(transformed.index)
    for ax, col, color in zip(axes.flat, transformed.columns, colors):
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.plot(dates, transformed[col], color=color, linewidth=1.6)
        ax.set_title(MODEL_LABELS[col])
    path = FIG / "02_stationary_transformations_growth_changes.png"
    fig.suptitle("Стационарные преобразования: темпы роста и приросты", fontsize=14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_okun_scatter(transformed: pd.DataFrame) -> Path:
    x = transformed["gdp_growth"]
    y = transformed["d_unrate"]
    x_const = sm.add_constant(x)
    fit = sm.OLS(y, x_const).fit()
    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = fit.params["const"] + fit.params["gdp_growth"] * x_grid
    fig, ax = plt.subplots(figsize=(8, 5))
    dates = transformed.index.astype(str)
    covid_mask = transformed.index.isin([pd.Period("2020Q2"), pd.Period("2020Q3")])
    ax.scatter(x[~covid_mask], y[~covid_mask], s=32, alpha=0.75, color="#1f77b4", label="кварталы")
    ax.scatter(x[covid_mask], y[covid_mask], s=58, color="#d62728", label="COVID-выбросы")
    for idx in np.where(covid_mask)[0]:
        ax.annotate(dates[idx], (x.iloc[idx], y.iloc[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(x_grid, y_grid, color="#111111", linewidth=1.5, label="OLS")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title("Диаграмма закона Оукена")
    ax.set_xlabel("Темп роста реального ВВП, % годовых")
    ax.set_ylabel("Прирост безработицы, п.п.")
    ax.legend(frameon=False)
    path = FIG / "03_okun_scatter_gdp_growth_unemployment_change.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_acf_pacf(transformed: pd.DataFrame) -> Path:
    lags = np.arange(0, 13)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    for row, col in enumerate(transformed.columns):
        values = transformed[col].dropna()
        acf_values = acf(values, nlags=12, fft=False)
        pacf_values = pacf(values, nlags=12, method="ywm")
        for j, (vals, title) in enumerate(((acf_values, "ACF"), (pacf_values, "PACF"))):
            ax = axes[row, j]
            ax.axhline(0, color="#555555", linewidth=0.8)
            conf = 1.96 / math.sqrt(len(values))
            ax.axhline(conf, color="#d62728", linestyle="--", linewidth=0.8)
            ax.axhline(-conf, color="#d62728", linestyle="--", linewidth=0.8)
            ax.vlines(lags, 0, vals, color="#1f77b4", linewidth=1.5)
            ax.scatter(lags, vals, color="#1f77b4", s=18)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{title}: {MODEL_LABELS[col]}")
    path = FIG / "04_acf_pacf_stationary_series.png"
    fig.suptitle("Коррелограммы стационарных рядов", fontsize=14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cointegration_heatmap(pairwise: pd.DataFrame) -> Path:
    labels = list(pairwise.index)
    values = pairwise.astype(float).values
    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=0.2)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                text = "-"
            else:
                text = fmt_p(values[i, j])
            ax.text(j, i, text, ha="center", va="center", color="#111111", fontsize=9)
    ax.set_title("Матрица p-value теста Энгла-Грейнджера")
    fig.colorbar(image, ax=ax, label="p-value")
    path = FIG / "05_cointegration_engle_granger_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_granger_network(granger_p: pd.DataFrame) -> Path:
    label_map = {
        "g, рост ВВП": "g",
        "Δu, прирост безработицы": "Δu",
        "π, инфляция CPI": "π",
        "Δi, прирост ставки": "Δi",
    }
    descriptions = {
        "g": "g — рост ВВП",
        "Δu": "Δu — прирост безработицы",
        "π": "π — инфляция CPI",
        "Δi": "Δi — прирост ставки Fed funds",
    }
    labels = list(granger_p.index)
    positions = {
        "g, рост ВВП": np.array([-0.55, 0.35]),
        "Δu, прирост безработицы": np.array([0.55, 0.35]),
        "π, инфляция CPI": np.array([-0.55, -0.45]),
        "Δi, прирост ставки": np.array([0.55, -0.45]),
    }
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_title("Причинность по Грейнджеру, p<0.05")
    for label in labels:
        pos = positions[label]
        short = label_map[label]
        ax.scatter(pos[0], pos[1], s=2300, color="#f4f6f8", edgecolor="#333333", linewidth=1.2, zorder=3)
        ax.text(pos[0], pos[1], short, ha="center", va="center", fontsize=18, fontweight="bold", zorder=4)

    significant_edges: set[tuple[str, str]] = set()
    for cause in labels:
        for effect in labels:
            if cause == effect:
                continue
            p_value = float(granger_p.loc[cause, effect])
            if p_value < 0.05:
                significant_edges.add((cause, effect))

    edge_count = 0
    for cause, effect in sorted(significant_edges):
        edge_count += 1
        p_value = float(granger_p.loc[cause, effect])
        reverse_exists = (effect, cause) in significant_edges
        rad = 0.28 if reverse_exists else 0.08
        start = positions[cause]
        end = positions[effect]
        direction = end - start
        start2 = start + direction * 0.22
        end2 = end - direction * 0.22
        arrow = FancyArrowPatch(
            start2,
            end2,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.3 + (0.05 - p_value) * 26,
            color="#1f77b4",
            alpha=0.85,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(arrow)
    if edge_count == 0:
        ax.text(0, 0, "Нет связей на 5%", ha="center", va="center", fontsize=12)
    legend = "\n".join(descriptions[label_map[label]] for label in labels)
    ax.text(0, -1.05, legend, ha="center", va="top", fontsize=10)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.2, 0.95)
    ax.axis("off")
    path = FIG / "06_granger_causality_network.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_var_irf(best_var, columns: list[str]) -> tuple[Path, dict[str, float]]:
    horizon = 12
    irf = best_var.irf(horizon)
    orth = irf.orth_irfs
    response_idx = columns.index("d_unrate")
    periods = np.arange(horizon + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for impulse, color in zip(columns, colors):
        impulse_idx = columns.index(impulse)
        ax.plot(periods, orth[:, response_idx, impulse_idx], marker="o", linewidth=1.5, color=color, label=MODEL_LABELS[impulse])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("IRF: отклик прироста безработицы на ортогональные шоки VAR")
    ax.set_xlabel("Горизонт, кварталы")
    ax.set_ylabel("Отклик Δu, п.п.")
    ax.legend(frameon=False, fontsize=8)
    path = FIG / "07_var_irf_unemployment_response.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    gdp_response = orth[:, response_idx, columns.index("gdp_growth")]
    return path, {
        "gdp_irf_min": float(np.min(gdp_response)),
        "gdp_irf_h_min": int(np.argmin(gdp_response)),
        "gdp_irf_cumulative_12": float(np.sum(gdp_response)),
    }


def plot_var_fevd(best_var, columns: list[str]) -> tuple[Path, pd.DataFrame]:
    horizon = 12
    fevd = best_var.fevd(horizon)
    decomp = fevd.decomp
    target_idx = columns.index("d_unrate")
    shares = pd.DataFrame(decomp[target_idx, :, :], columns=columns, index=np.arange(1, horizon + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        shares.index,
        [shares[col] for col in columns],
        labels=[MODEL_LABELS[col] for col in columns],
        colors=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_title("FEVD: вклад шоков в ошибку прогноза Δu")
    ax.set_xlabel("Горизонт, кварталы")
    ax.set_ylabel("Доля дисперсии")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    path = FIG / "08_var_fevd_unemployment_change.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, shares


def plot_residual_diagnostics(adl_result, best_var, columns: list[str]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    adl_resid = adl_result.resid
    axes[0, 0].plot(period_to_ts(adl_resid.index), adl_resid, color="#1f77b4", linewidth=1.2)
    axes[0, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[0, 0].set_title("Остатки ADL")
    axes[0, 1].hist(adl_resid, bins=18, color="#1f77b4", alpha=0.8)
    axes[0, 1].set_title("Распределение остатков ADL")
    var_resid = var_resid_frame(best_var, columns)["d_unrate"]
    axes[1, 0].plot(period_to_ts(var_resid.index), var_resid, color="#d62728", linewidth=1.2)
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_title("Остатки уравнения Δu в VAR")
    axes[1, 1].hist(var_resid, bins=18, color="#d62728", alpha=0.8)
    axes[1, 1].set_title("Распределение остатков VAR, уравнение Δu")
    path = FIG / "09_model_residual_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    series = {sid: load_fred_series(sid) for sid in SERIES_META}
    quarterly = pd.concat({sid: to_quarterly(sid, s) for sid, s in series.items()}, axis=1).sort_index()
    quarterly = quarterly.loc["1990Q1":].dropna()
    quarterly.index.name = "quarter"
    quarterly.to_csv(DATA / "fred_quarterly_us_macro_1990_2026.csv", encoding="utf-8")

    levels = quarterly.copy()
    level_model = pd.DataFrame(
        {
            "ln_gdp": np.log(levels["GDPC1"]),
            "unrate": levels["UNRATE"],
            "ln_cpi": np.log(levels["CPIAUCSL"]),
            "fedfunds": levels["FEDFUNDS"],
        },
        index=levels.index,
    )
    transformed = pd.DataFrame(
        {
            "gdp_growth": 400 * np.log(levels["GDPC1"]).diff(),
            "d_unrate": levels["UNRATE"].diff(),
            "inflation": 400 * np.log(levels["CPIAUCSL"]).diff(),
            "d_fedfunds": levels["FEDFUNDS"].diff(),
        },
        index=levels.index,
    ).dropna()
    transformed.index.name = "quarter"
    transformed.to_csv(DATA / "fred_transformed_stationary_series.csv", encoding="utf-8")

    figure_paths = {
        "levels": plot_levels(levels),
        "transformed": plot_transformed(transformed),
        "okun": plot_okun_scatter(transformed),
        "acf_pacf": plot_acf_pacf(transformed),
    }

    source_table = pd.DataFrame(
        [
            {
                "Показатель": SERIES_META[sid]["label_ru"],
                "Единицы": SERIES_META[sid]["unit"],
                "Частота в источнике": SERIES_META[sid]["frequency"],
                "Источник": SERIES_META[sid]["source"],
                "Ссылка": SERIES_META[sid]["url"],
            }
            for sid in SERIES_META
        ]
    )
    save_table(source_table, "source_data_description")

    period_table = pd.DataFrame(
        {
            "Параметр": [
                "Страна",
                "Период уровней",
                "Наблюдений уровней",
                "Период преобразованных рядов",
                "Наблюдений преобразованных рядов",
                "Дата выгрузки",
            ],
            "Значение": [
                "США",
                f"{levels.index.min()} - {levels.index.max()}",
                str(len(levels)),
                f"{transformed.index.min()} - {transformed.index.max()}",
                str(len(transformed)),
                pd.Timestamp.today().strftime("%Y-%m-%d"),
            ],
        }
    )
    save_table(period_table, "sample_period")

    level_summary = levels.rename(columns={sid: SERIES_META[sid]["short"] for sid in SERIES_META}).describe().T
    level_summary["var"] = levels.rename(columns={sid: SERIES_META[sid]["short"] for sid in SERIES_META}).var()
    level_summary = level_summary[["count", "mean", "var", "std", "min", "max"]]
    level_summary.columns = ["N", "Среднее", "Дисперсия", "Ст. откл.", "Минимум", "Максимум"]
    level_summary.index.name = "Показатель"
    save_table(level_summary, "summary_levels")

    trans_summary = transformed.rename(columns=MODEL_LABELS).describe().T
    trans_summary["var"] = transformed.rename(columns=MODEL_LABELS).var()
    trans_summary = trans_summary[["count", "mean", "var", "std", "min", "max"]]
    trans_summary.columns = ["N", "Среднее", "Дисперсия", "Ст. откл.", "Минимум", "Максимум"]
    trans_summary.index.name = "Показатель"
    save_table(trans_summary, "summary_transformed")

    growth_tail = transformed.rename(columns=MODEL_LABELS).tail(10)
    save_table(growth_tail, "last_10_transformed_observations")

    acf_table = []
    for col in transformed.columns:
        values = transformed[col].dropna()
        acf_values = acf(values, nlags=8, fft=False)
        pacf_values = pacf(values, nlags=8, method="ywm")
        acf_table.append(
            {
                "Ряд": MODEL_LABELS[col],
                "ACF(1)": acf_values[1],
                "ACF(4)": acf_values[4],
                "ACF(8)": acf_values[8],
                "PACF(1)": pacf_values[1],
                "PACF(4)": pacf_values[4],
                "PACF(8)": pacf_values[8],
            }
        )
    acf_table = pd.DataFrame(acf_table)
    save_table(acf_table, "acf_pacf_key_lags")

    stationarity_rows = []
    for col in level_model.columns:
        adf_stat, adf_p = test_adf(level_model[col], regression="ct")
        kpss_stat, kpss_p = test_kpss(level_model[col], regression="ct")
        stationarity_rows.append(
            {
                "Показатель": LEVEL_LABELS[col],
                "Преобразование": "уровень",
                "ADF stat": adf_stat,
                "ADF p": fmt_p(adf_p),
                "KPSS stat": kpss_stat,
                "KPSS p": fmt_p(kpss_p),
                "Вывод": stationarity_conclusion(adf_p, kpss_p),
            }
        )
    for col in transformed.columns:
        adf_stat, adf_p = test_adf(transformed[col], regression="c")
        kpss_stat, kpss_p = test_kpss(transformed[col], regression="c")
        stationarity_rows.append(
            {
                "Показатель": MODEL_LABELS[col],
                "Преобразование": "первая разность/темп",
                "ADF stat": adf_stat,
                "ADF p": fmt_p(adf_p),
                "KPSS stat": kpss_stat,
                "KPSS p": fmt_p(kpss_p),
                "Вывод": stationarity_conclusion(adf_p, kpss_p),
            }
        )
    stationarity = pd.DataFrame(stationarity_rows)
    save_table(stationarity, "stationarity_adf_kpss")

    pairwise_p = pd.DataFrame(np.nan, index=list(LEVEL_LABELS.values()), columns=list(LEVEL_LABELS.values()))
    pairwise_label = pd.DataFrame("", index=list(LEVEL_LABELS.values()), columns=list(LEVEL_LABELS.values()))
    pairwise_p.index.name = "Показатель"
    pairwise_label.index.name = "Показатель"
    for i, col_i in enumerate(level_model.columns):
        for j, col_j in enumerate(level_model.columns):
            if i == j:
                pairwise_label.iloc[i, j] = "-"
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_value, _ = coint(level_model[col_i], level_model[col_j], trend="c", autolag="aic")
            pairwise_p.iloc[i, j] = p_value
            pairwise_label.iloc[i, j] = f"{fmt_p(p_value)}; {'есть' if p_value < 0.05 else 'нет'}"
    save_table(pairwise_p, "cointegration_pairwise_pvalues")
    save_table(pairwise_label, "cointegration_pairwise_labels")
    figure_paths["cointegration"] = plot_cointegration_heatmap(pairwise_p)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lag_order = select_order(level_model, maxlags=8, deterministic="c").selected_orders
        johansen_diff_lags = int(lag_order.get("aic") or 2)
        johansen_diff_lags = max(1, min(johansen_diff_lags, 4))
    except Exception:
        johansen_diff_lags = 2
    johansen = coint_johansen(level_model, det_order=0, k_ar_diff=johansen_diff_lags)
    johansen_table = pd.DataFrame(
        {
            "H0: rank <=": list(range(len(level_model.columns))),
            "trace stat": johansen.lr1,
            "crit 90%": johansen.cvt[:, 0],
            "crit 95%": johansen.cvt[:, 1],
            "crit 99%": johansen.cvt[:, 2],
            "reject 5%": ["да" if stat > crit else "нет" for stat, crit in zip(johansen.lr1, johansen.cvt[:, 1])],
        }
    )
    save_table(johansen_table, "cointegration_johansen_trace")
    johansen_rank_5 = int(sum(johansen.lr1 > johansen.cvt[:, 1]))

    var_model = VAR(transformed)
    selected_orders = var_model.select_order(maxlags=8).selected_orders
    selected_lag = int(selected_orders.get("aic") or selected_orders.get("bic") or 2)
    selected_lag = max(1, min(selected_lag, 8))

    var_fits = []
    candidate_lags = sorted(set([1, 2, selected_lag, max(1, selected_lag + 1)]))
    for lag in candidate_lags:
        fit = var_model.fit(lag, trend="c")
        whiteness_p = fit.test_whiteness(nlags=max(lag + 4, 8)).pvalue
        normal_p = fit.test_normality().pvalue
        var_fits.append(
            {
                "name": f"VAR({lag}), const",
                "lag": lag,
                "fit": fit,
                "stable": bool(fit.is_stable(verbose=False)),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "hqic": float(fit.hqic),
                "whiteness_p": float(whiteness_p),
                "normal_p": float(normal_p),
            }
        )
    fit_trend = var_model.fit(selected_lag, trend="ct")
    var_fits.append(
        {
            "name": f"VAR({selected_lag}), const+trend",
            "lag": selected_lag,
            "fit": fit_trend,
            "stable": bool(fit_trend.is_stable(verbose=False)),
            "aic": float(fit_trend.aic),
            "bic": float(fit_trend.bic),
            "hqic": float(fit_trend.hqic),
            "whiteness_p": float(fit_trend.test_whiteness(nlags=max(selected_lag + 4, 8)).pvalue),
            "normal_p": float(fit_trend.test_normality().pvalue),
        }
    )
    best_var_info = sorted([v for v in var_fits if v["stable"]], key=lambda x: x["aic"])[0]
    best_var = best_var_info["fit"]

    granger_labels = [MODEL_LABELS[col] for col in transformed.columns]
    granger_p = pd.DataFrame(np.nan, index=granger_labels, columns=granger_labels)
    granger_p.index.name = "Причина"
    for cause in transformed.columns:
        for effect in transformed.columns:
            if cause == effect:
                continue
            test = best_var.test_causality(effect, [cause], kind="f")
            granger_p.loc[MODEL_LABELS[cause], MODEL_LABELS[effect]] = test.pvalue
    save_table(granger_p, "granger_causality_pvalues")
    figure_paths["granger"] = plot_granger_network(granger_p)

    adl_specs = [
        fit_adl(transformed, "ADL-1: Δu_t = c + βg_t", 0, 0, False),
        fit_adl(transformed, "ADL-2: g_t,g_{t-1},g_{t-2},Δu_{t-1}", 2, 1, False),
        fit_adl(transformed, "ADL-3: ADL-2 + D2020Q2", 2, 1, True),
    ]
    best_adl = sorted(adl_specs, key=lambda x: x["aic"])[0]

    adl_compare = pd.DataFrame(
        [
            {
                "Спецификация": spec["name"],
                "N": spec["nobs"],
                "Adj.R2": spec["r2_adj"],
                "RMSE": spec["rmse"],
                "AIC": spec["aic"],
                "BIC": spec["bic"],
                "Ljung-Box p": fmt_p(spec["ljung_p"]),
                "Jarque-Bera p": fmt_p(spec["normal_p"]),
                "Выбрана для интерпретации": "да" if spec is best_adl else "",
            }
            for spec in adl_specs
        ]
    )
    save_table(adl_compare, "adl_model_comparison")

    adl_coef = pd.DataFrame(
        {
            "coef": best_adl["result"].params,
            "std err HAC": best_adl["result"].bse,
            "t": best_adl["result"].tvalues,
            "p-value": best_adl["result"].pvalues,
        }
    )
    adl_coef.index.name = "Параметр"
    save_table(adl_coef, "best_adl_coefficients")

    g_betas = [v for k, v in best_adl["result"].params.items() if k.startswith("gdp_growth")]
    du_phis = [v for k, v in best_adl["result"].params.items() if k.startswith("d_unrate_L")]
    short_run_okun = float(best_adl["result"].params.get("gdp_growth", np.nan))
    cumulative_okun = float(np.sum(g_betas) / (1 - np.sum(du_phis))) if (1 - np.sum(du_phis)) != 0 else np.nan

    var_compare = pd.DataFrame(
        [
            {
                "Спецификация": info["name"],
                "Устойчива": "да" if info["stable"] else "нет",
                "AIC": info["aic"],
                "BIC": info["bic"],
                "HQIC": info["hqic"],
                "Portmanteau p": fmt_p(info["whiteness_p"]),
                "Normality p": fmt_p(info["normal_p"]),
                "Выбрана по AIC": "да" if info is best_var_info else "",
            }
            for info in var_fits
        ]
    )
    save_table(var_compare, "var_model_comparison")

    var_du_eq = pd.DataFrame({"coef Δu equation": var_params_frame(best_var, list(transformed.columns))["d_unrate"]})
    var_du_eq.index.name = "Параметр"
    save_table(var_du_eq, "best_var_unemployment_equation")

    combined_compare = pd.DataFrame(
        [
            {
                "Спецификация модели": row["Спецификация"],
                "Стационарность модели": "стационарные переменные; OLS",
                "Ошибка модели, инф.критерии": f"RMSE={fmt_num(row['RMSE'])}; AIC={fmt_num(row['AIC'])}; BIC={fmt_num(row['BIC'])}",
                "Анализ остатков": f"Ljung-Box p={row['Ljung-Box p']}; JB p={row['Jarque-Bera p']}",
                "Выбор модели": row["Выбрана для интерпретации"],
            }
            for _, row in adl_compare.iterrows()
        ]
        + [
            {
                "Спецификация модели": row["Спецификация"],
                "Стационарность модели": f"устойчива: {row['Устойчива']}",
                "Ошибка модели, инф.критерии": f"AIC={fmt_num(row['AIC'])}; BIC={fmt_num(row['BIC'])}; HQIC={fmt_num(row['HQIC'])}",
                "Анализ остатков": f"Portmanteau p={row['Portmanteau p']}; Normality p={row['Normality p']}",
                "Выбор модели": row["Выбрана по AIC"],
            }
            for _, row in var_compare.iterrows()
        ]
    )
    save_table(combined_compare, "combined_model_comparison")

    figure_paths["irf"], irf_stats = plot_var_irf(best_var, list(transformed.columns))
    figure_paths["fevd"], fevd_shares = plot_var_fevd(best_var, list(transformed.columns))
    fevd_selected = fevd_shares.loc[[4, 8, 12]].rename(columns=MODEL_LABELS)
    fevd_selected.index.name = "Горизонт"
    save_table(fevd_selected, "fevd_unemployment_selected_horizons")
    figure_paths["residuals"] = plot_residual_diagnostics(best_adl["result"], best_var, list(transformed.columns))

    strongest_granger = (
        granger_p.stack()
        .dropna()
        .sort_values()
        .head(6)
        .reset_index()
        .rename(columns={"level_0": "Причина", "level_1": "Следствие", 0: "p-value"})
    )
    save_table(strongest_granger, "strongest_granger_links")

    data_preview = pd.concat([levels.head(5), levels.tail(5)]).reset_index()
    data_preview["quarter"] = data_preview["quarter"].astype(str)
    save_table(data_preview, "data_preview_levels")

    observations = len(levels)
    transformed_observations = len(transformed)
    start_period = levels.index.min()
    end_period = levels.index.max()
    transformed_start = transformed.index.min()
    transformed_end = transformed.index.max()

    latest = levels.iloc[-1]
    latest_transformed = transformed.iloc[-1]
    corr_growth_unemp = transformed["gdp_growth"].corr(transformed["d_unrate"])
    no_covid_mask = ~transformed.index.isin([pd.Period("2020Q2"), pd.Period("2020Q3")])
    corr_growth_unemp_no_covid = transformed.loc[no_covid_mask, "gdp_growth"].corr(
        transformed.loc[no_covid_mask, "d_unrate"]
    )
    covid_du = transformed.loc[pd.Period("2020Q2"), "d_unrate"] if pd.Period("2020Q2") in transformed.index else np.nan
    covid_growth = transformed.loc[pd.Period("2020Q2"), "gdp_growth"] if pd.Period("2020Q2") in transformed.index else np.nan

    fig_rel = {key: path.relative_to(OUT).as_posix() for key, path in figure_paths.items()}
    data_rel = (DATA / "fred_quarterly_us_macro_1990_2026.csv").relative_to(OUT).as_posix()
    trans_data_rel = (DATA / "fred_transformed_stationary_series.csv").relative_to(OUT).as_posix()
    script_rel = Path("..") / "scripts" / "build_time_series_report.py"

    report = f"""
# Взаимосвязь экономического роста, безработицы, инфляции и процентной ставки в США

## 1. Введение

Работа проверяет динамическую версию закона Оукена: при ускорении реального выпуска безработица должна снижаться, а при замедлении роста — повышаться. В качестве расширения в модель добавлены инфляция и ставка федеральных фондов, поскольку денежно-кредитная политика и ценовая динамика влияют на совокупный спрос и рынок труда.

Научная основа: Knotek, E. S. II (2007). *How useful is Okun's law?* Economic Review, Federal Reserve Bank of Kansas City, 92(Q IV), 73-103. Автор показывает, что отрицательная связь между ростом ВВП и изменением безработицы сохраняет прогностическую полезность, но является статистической, а не структурной, и меняется во времени и по фазам цикла. В этой работе идея статьи проверяется на современной квартальной выборке США с добавлением инфляции и процентной ставки.

Проблема исследования: насколько устойчиво рост выпуска связан с изменением безработицы после кризиса 2008-2009 гг., COVID-шока и инфляционно-процентного цикла 2021-2024 гг. Гипотеза: темп роста реального ВВП отрицательно влияет на прирост безработицы; инфляция и ставка федеральных фондов могут добавлять лаговую информацию, но связь должна быть чувствительна к структурным шокам.

## 2. Исходные данные

{markdown_table(source_table, index=False)}

{markdown_table(period_table, index=False)}

Исходные месячные ряды UNRATE, CPIAUCSL и FEDFUNDS агрегированы в квартальные средние. Ряд GDPC1 уже публикуется как квартальный. Основной файл данных сохранен: `{data_rel}`. Файл со стационарными преобразованиями сохранен: `{trans_data_rel}`.

Для моделирования использованы преобразования:

$$g_t = 400\\Delta\\ln(GDPC1_t),\\quad \\Delta u_t = u_t-u_{{t-1}},\\quad \\pi_t = 400\\Delta\\ln(CPI_t),\\quad \\Delta i_t=i_t-i_{{t-1}}.$$

Темпы роста ВВП и инфляции умножены на 400, то есть выражены как годовые темпы квартал-к-кварталу. На последнем наблюдении {end_period}: ВВП = {fmt_num(latest['GDPC1'])} млрд долл., безработица = {fmt_num(latest['UNRATE'])}%, CPI = {fmt_num(latest['CPIAUCSL'])}, ставка Fed funds = {fmt_num(latest['FEDFUNDS'])}%.

![Исходные временные ряды]({fig_rel['levels']})

## 3. Описание динамики

{markdown_table(level_summary, index=True)}

{markdown_table(trans_summary, index=True)}

![Стационарные преобразования]({fig_rel['transformed']})

В уровнях видны выраженные тренды в реальном ВВП и CPI, циклические всплески безработицы в 2008-2009 и 2020 гг., а также смена режима процентной ставки после 2022 г. Сезонность в основных макроэкономических рядах в значительной степени устранена источниками данных: ВВП, CPI и безработица публикуются сезонно скорректированными; ставка FEDFUNDS усреднена по кварталам.

На преобразованных рядах выделяются структурные сдвиги: в 2020Q2 рост ВВП составил {fmt_num(covid_growth)}% годовых, а прирост безработицы — {fmt_num(covid_du)} п.п. Корреляция между квартальным ростом ВВП и приростом безработицы равна {fmt_num(corr_growth_unemp)}, а без 2020Q2–2020Q3 — {fmt_num(corr_growth_unemp_no_covid)}. Знак остается отрицательным, но сила связи заметно зависит от кризисных COVID-наблюдений.

Следует учитывать, что оценка связи существенно усиливается кризисными наблюдениями 2020Q2–2020Q3. Поэтому результаты интерпретируются не как стабильная структурная зависимость, а как эмпирическая связь, чувствительная к экстремальным макроэкономическим шокам.

![Диаграмма закона Оукена]({fig_rel['okun']})

Коррелограммы показывают умеренную инерционность инфляции и процентной ставки; изменение безработицы содержит резкие кризисные выбросы и слабую краткосрочную автокорреляцию. Это все равно делает лаговую спецификацию ADL и VAR уместной.

{markdown_table(acf_table, index=False)}

![ACF и PACF]({fig_rel['acf_pacf']})

Последние 10 наблюдений преобразованных рядов:

{markdown_table(growth_tail.reset_index().assign(quarter=lambda x: x["quarter"].astype(str)), index=False)}

## 4. Стационарность

ADF-тест проверяет нулевую гипотезу единичного корня. KPSS-тест использован как проверка устойчивости вывода: его нулевая гипотеза — стационарность вокруг константы или тренда. Для уровней в ADF/KPSS включался тренд, для преобразованных рядов — константа.

{markdown_table(stationarity, index=False)}

Вывод: уровни логарифма ВВП и CPI не подходят для VAR без преобразования; для них естественная рабочая форма — первые логарифмические разности. Ставка и безработица дают более смешанную картину из-за ограниченности и смены режимов, поэтому в системной модели использованы приросты. После преобразований ряды демонстрируют признаки стационарности по ADF и KPSS, поэтому дальнейшее моделирование проводится в темпах роста и приростах. При этом выводы следует интерпретировать с учетом кризисных выбросов 2008-2009 и 2020 гг.

## 5. Наличие взаимосвязей

### 5.1. Коинтеграционная матрица

Тест Энгла-Грейнджера применяется для проверки одной долгосрочной связи между двумя I(1)-рядами. Тест Йохансена используется в системе из нескольких I(1)-переменных и позволяет обнаружить несколько коинтеграционных векторов. Поскольку часть рядов дает смешанные выводы о порядке интегрируемости, результаты коинтеграции рассматриваются как диагностические, а не как основание для обязательного VECM.

Матрица ниже содержит `p-value; вывод на 5%`. Матрица Энгла-Грейнджера не является строго симметричной, поскольку результат зависит от выбора зависимой переменной в первом шаге регрессии. Поэтому пары с p-value ниже 5% рассматриваются только как индикаторы возможной долгосрочной связи, а не как окончательное доказательство коинтеграции.

{markdown_table(pairwise_label, index=True)}

![Матрица коинтеграции]({fig_rel['cointegration']})

Тест Йохансена, trace statistic, число лагов разностей = {johansen_diff_lags}:

{markdown_table(johansen_table, index=False)}

На 5% тест Йохансена указывает rank = {johansen_rank_5}. С учетом смешанных тестов стационарности для безработицы и ставки базовой моделью выбрана VAR на стационарных преобразованиях; ECM/VECM не используется как основная спецификация.

### 5.2. Причинность по Грейнджеру

В таблице строка — предполагаемая причина, столбец — следствие; значения — p-value F-теста в выбранной VAR.

{markdown_table(granger_p, index=True)}

Наиболее сильные связи по p-value:

{markdown_table(strongest_granger, index=False)}

![Схема Грейнджера]({fig_rel['granger']})

На 5% обнаружена двусторонняя причинность по Грейнджеру между выпуском и рынком труда: g ⇄ Δu. Это означает, что лаги роста ВВП помогают объяснять будущий прирост безработицы, а лаги прироста безработицы содержат информацию о будущем росте ВВП.

Результаты поддерживают динамическую спецификацию: лаги макроэкономических переменных содержат информацию о будущих изменениях рынка труда и процентной ставки, поэтому статическая регрессия закона Оукена недостаточна.

## 6. Спецификации моделей

Оценены две группы моделей.

ADL-модель для закона Оукена:

$$\\Delta u_t = \\alpha + \\sum_{{j=0}}^q \\beta_j g_{{t-j}} + \\sum_{{k=1}}^p \\phi_k \\Delta u_{{t-k}} + \\delta D_{{2020Q2}} + \\varepsilon_t.$$

VAR-модель для системы стационарных рядов:

$$y_t = c + A_1y_{{t-1}} + \\dots + A_py_{{t-p}} + e_t,$$

где $y_t=(g_t,\\Delta u_t,\\pi_t,\\Delta i_t)'$.

Ограничения спецификаций: ADL описывает одно уравнение и не решает проблему одновременности полностью; VAR чувствителен к выбору лагов и порядку переменных при ортогонализации шоков; ECM/VECM требует убедительной I(1)-структуры и коинтеграции, что для данной выборки не является устойчивым выводом.

## 7. Моделирование

Сравнение ADL-спецификаций:

{markdown_table(adl_compare, index=False)}

Коэффициенты выбранной ADL:

{markdown_table(adl_coef, index=True)}

Краткосрочный коэффициент Оукена при текущем росте ВВП равен {fmt_num(short_run_okun)}: увеличение квартального годового темпа роста ВВП на 1 п.п. связано с изменением прироста безработицы примерно на {fmt_num(short_run_okun)} п.п. в том же квартале при прочих равных. Например, если квартальный годовой темп роста ВВП выше на 2 п.п., то текущий прирост безработицы в среднем ниже примерно на {fmt_num(2 * abs(short_run_okun))} п.п. Динамический суммарный эффект с учетом лагов равен {fmt_num(cumulative_okun)} п.п.

Сравнение VAR-спецификаций:

{markdown_table(var_compare, index=False)}

Уравнение прироста безработицы в выбранной VAR:

{markdown_table(var_du_eq, index=True)}

Итоговая таблица спецификаций:

{markdown_table(combined_compare, index=False)}

![Диагностика остатков]({fig_rel['residuals']})

Диагностика показывает компромисс. ADL-3 с COVID-фиктивной переменной существенно улучшает AIC/RMSE и поэтому используется для интерпретации коэффициента Оукена, но Ljung-Box указывает на остаточную автокорреляцию; ADL-2 слабее по ошибке, зато чище по автокорреляции. Поэтому коэффициенты ADL-3 используются как эмпирическая оценка краткосрочной связи, а стандартные ошибки оцениваются в HAC-форме.

Для VAR выбрана устойчивая спецификация с минимальным AIC среди кандидатов. В базовой VAR фиктивные переменные не включались, чтобы сохранить простую сопоставимую спецификацию. Однако нарушение нормальности остатков показывает, что расширенная VAR с COVID-фиктивными переменными могла бы быть полезной проверкой устойчивости. Ненормальность остатков в основном связана не с общей непригодностью модели, а с экстремальными кризисными наблюдениями, прежде всего COVID-периодом. Для более строгой версии работы можно оценить альтернативные спецификации без 2020Q2–2020Q3 или с дополнительными фиктивными переменными.

## 8. Интерпретация

![IRF VAR]({fig_rel['irf']})

IRF и FEVD построены для ортогональных шоков и зависят от порядка переменных в VAR. В данной работе использован порядок: g, Δu, π, Δi. В выбранном порядке ортогонализации положительный шок роста ВВП дает немедленный отрицательный отклик Δu, после чего эффект быстро затухает; суммарный отклик за 12 кварталов равен {fmt_num(irf_stats['gdp_irf_cumulative_12'])} п.п. Это согласуется с законом Оукена, но величина эффекта зависит от спецификации VAR.

![FEVD VAR]({fig_rel['fevd']})

Разложение дисперсии ошибки прогноза для Δu:

{markdown_table(fevd_selected, index=True)}

В выбранном порядке ортогонализации основной вклад в ошибку прогноза Δu дают шоки выпуска, затем собственные шоки безработицы; вклад инфляции и процентной ставки невелик. Практически это означает, что для прогноза рынка труда в данной системе ключевым остается состояние выпуска, а лаговая инерция самой безработицы занимает второе место.

## 9. Заключение

Гипотеза об отрицательной связи между ростом реального ВВП и изменением безработицы в целом подтверждается. Корреляция между квартальным темпом роста ВВП и приростом безработицы имеет отрицательный знак, коэффициенты ADL-модели соответствуют закону Оукена, а импульсные отклики VAR показывают снижение прироста безработицы после положительного шока выпуска.

При этом выявленная связь не должна интерпретироваться как стабильная механическая зависимость. Оценки чувствительны к кризисным наблюдениям, прежде всего к COVID-периоду 2020Q2–2020Q3. ADL-модель с фиктивной переменной 2020Q2 существенно улучшает качество аппроксимации по RMSE и информационным критериям, однако диагностика остатков указывает на наличие автокорреляции. Поэтому коэффициенты используются как эмпирическая оценка краткосрочной связи, а не как строгий структурный параметр.

VAR-модель на стационарных преобразованиях подтверждает наличие двусторонней причинности по Грейнджеру между ростом ВВП и приростом безработицы. IRF и FEVD показывают, что в выбранном порядке ортогонализации шоки выпуска играют основную роль в объяснении ошибки прогноза прироста безработицы, тогда как вклад инфляции и изменения ставки федеральных фондов оказывается небольшим.

Таким образом, результаты согласуются с выводами Knotek (2007): закон Оукена остается полезным эмпирическим правилом и инструментом анализа рынка труда, но его параметры зависят от фазы делового цикла, структурных шоков и выбранной спецификации модели.

## Приложение A. Файлы расчетов

- Данные уровней: `{data_rel}`
- Стационарные преобразования: `{trans_data_rel}`
- Таблицы расчетов: `tables/*.csv` и `tables/*.md`
- Графики: `figures/*.png`
- Скрипт генерации отчета: `{script_rel.as_posix()}`

Фрагмент исходной таблицы уровней:

{markdown_table(data_preview, index=False)}

## Приложение B. Источники

- Knotek, E. S. II (2007). *How useful is Okun's law?* Economic Review, Federal Reserve Bank of Kansas City, 92(Q IV), 73-103. https://www.kansascityfed.org/documents/955/2007-How%20Useful%20is%20Okun%27s%20Law%3F.pdf
- FRED: Real Gross Domestic Product [GDPC1]. https://fred.stlouisfed.org/series/GDPC1
- FRED: Unemployment Rate [UNRATE]. https://fred.stlouisfed.org/series/UNRATE
- FRED: Consumer Price Index for All Urban Consumers [CPIAUCSL]. https://fred.stlouisfed.org/series/CPIAUCSL
- FRED: Federal Funds Effective Rate [FEDFUNDS]. https://fred.stlouisfed.org/series/FEDFUNDS
"""

    REPORT.write_text(textwrap.dedent(report).strip() + "\n", encoding="utf-8")
    print(f"Report written: {REPORT}")
    print(f"Figures: {len(list(FIG.glob('*.png')))}")
    print(f"Tables: {len(list(TAB.glob('*.csv')))}")
    print(f"Observations: levels={observations}, transformed={transformed_observations}")
    print(f"Period: {start_period}-{end_period}; transformed: {transformed_start}-{transformed_end}")
    print(f"Best ADL: {best_adl['name']}")
    print(f"Best VAR: {best_var_info['name']}")


if __name__ == "__main__":
    main()
