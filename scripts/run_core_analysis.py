#!/usr/bin/env python3
"""Run core calculations, tables, and figures without building narrative outputs."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, coint, pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order

import add_assignment_compliance_extensions as compliance
import build_microqualification_clustering as micro
import build_time_series_report as us


def load_fred_series_cached(series_id: str) -> pd.Series:
    """Load FRED data from local raw CSV when available, otherwise download it."""
    raw_path = us.DATA / f"fred_raw_{series_id}.csv"
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        raw = pd.read_csv(raw_path, na_values=[".", ""])
        date_col = "observation_date" if "observation_date" in raw.columns else "date"
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw[series_id] = pd.to_numeric(raw[series_id], errors="coerce")
        return raw.set_index(date_col)[series_id].dropna().sort_index()
    return us.load_fred_series(series_id)


def build_us_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = {sid: load_fred_series_cached(sid) for sid in us.SERIES_META}
    quarterly = pd.concat({sid: us.to_quarterly(sid, s) for sid, s in series.items()}, axis=1).sort_index()
    quarterly = quarterly.loc["1990Q1":].dropna()
    quarterly.index.name = "quarter"
    quarterly.to_csv(us.DATA / "fred_quarterly_us_macro_1990_2026.csv", encoding="utf-8")

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
    transformed.to_csv(us.DATA / "fred_transformed_stationary_series.csv", encoding="utf-8")
    return levels, level_model, transformed


def save_descriptive_tables(levels: pd.DataFrame, transformed: pd.DataFrame) -> None:
    source_table = pd.DataFrame(
        [
            {
                "Показатель": us.SERIES_META[sid]["label_ru"],
                "Единицы": us.SERIES_META[sid]["unit"],
                "Частота в источнике": us.SERIES_META[sid]["frequency"],
                "Источник": us.SERIES_META[sid]["source"],
                "Ссылка": us.SERIES_META[sid]["url"],
            }
            for sid in us.SERIES_META
        ]
    )
    us.save_table(source_table, "source_data_description")

    period_table = pd.DataFrame(
        {
            "Параметр": [
                "Страна",
                "Период уровней",
                "Наблюдений уровней",
                "Период преобразованных рядов",
                "Наблюдений преобразованных рядов",
                "Дата расчета",
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
    us.save_table(period_table, "sample_period")

    level_summary = levels.rename(columns={sid: us.SERIES_META[sid]["short"] for sid in us.SERIES_META}).describe().T
    level_summary["var"] = levels.rename(columns={sid: us.SERIES_META[sid]["short"] for sid in us.SERIES_META}).var()
    level_summary = level_summary[["count", "mean", "var", "std", "min", "max"]]
    level_summary.columns = ["N", "Среднее", "Дисперсия", "Ст. откл.", "Минимум", "Максимум"]
    level_summary.index.name = "Показатель"
    us.save_table(level_summary, "summary_levels")

    trans_summary = transformed.rename(columns=us.MODEL_LABELS).describe().T
    trans_summary["var"] = transformed.rename(columns=us.MODEL_LABELS).var()
    trans_summary = trans_summary[["count", "mean", "var", "std", "min", "max"]]
    trans_summary.columns = ["N", "Среднее", "Дисперсия", "Ст. откл.", "Минимум", "Максимум"]
    trans_summary.index.name = "Показатель"
    us.save_table(trans_summary, "summary_transformed")

    growth_tail = transformed.rename(columns=us.MODEL_LABELS).tail(10)
    us.save_table(growth_tail, "last_10_transformed_observations")


def build_stationarity_and_correlation_tables(level_model: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    acf_rows = []
    for col in transformed.columns:
        values = transformed[col].dropna()
        acf_values = acf(values, nlags=8, fft=False)
        pacf_values = pacf(values, nlags=8, method="ywm")
        acf_rows.append(
            {
                "Ряд": us.MODEL_LABELS[col],
                "ACF(1)": acf_values[1],
                "ACF(4)": acf_values[4],
                "ACF(8)": acf_values[8],
                "PACF(1)": pacf_values[1],
                "PACF(4)": pacf_values[4],
                "PACF(8)": pacf_values[8],
            }
        )
    us.save_table(pd.DataFrame(acf_rows), "acf_pacf_key_lags")

    stationarity_rows = []
    for col in level_model.columns:
        adf_stat, adf_p = us.test_adf(level_model[col], regression="ct")
        kpss_stat, kpss_p = us.test_kpss(level_model[col], regression="ct")
        stationarity_rows.append(
            {
                "Показатель": us.LEVEL_LABELS[col],
                "Преобразование": "уровень",
                "ADF stat": adf_stat,
                "ADF p": us.fmt_p(adf_p),
                "KPSS stat": kpss_stat,
                "KPSS p": us.fmt_p(kpss_p),
                "Вывод": us.stationarity_conclusion(adf_p, kpss_p),
            }
        )
    for col in transformed.columns:
        adf_stat, adf_p = us.test_adf(transformed[col], regression="c")
        kpss_stat, kpss_p = us.test_kpss(transformed[col], regression="c")
        stationarity_rows.append(
            {
                "Показатель": us.MODEL_LABELS[col],
                "Преобразование": "первая разность/темп",
                "ADF stat": adf_stat,
                "ADF p": us.fmt_p(adf_p),
                "KPSS stat": kpss_stat,
                "KPSS p": us.fmt_p(kpss_p),
                "Вывод": us.stationarity_conclusion(adf_p, kpss_p),
            }
        )
    stationarity = pd.DataFrame(stationarity_rows)
    us.save_table(stationarity, "stationarity_adf_kpss")
    return stationarity


def build_cointegration_tables(level_model: pd.DataFrame) -> None:
    pairwise_p = pd.DataFrame(np.nan, index=list(us.LEVEL_LABELS.values()), columns=list(us.LEVEL_LABELS.values()))
    pairwise_label = pd.DataFrame("", index=list(us.LEVEL_LABELS.values()), columns=list(us.LEVEL_LABELS.values()))
    pairwise_p.index.name = "Показатель"
    pairwise_label.index.name = "Показатель"
    for i, col_i in enumerate(level_model.columns):
        for j, col_j in enumerate(level_model.columns):
            if i == j:
                pairwise_label.iloc[i, j] = "-"
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_value, _ = coint(level_model[col_i], level_model[col_j], trend="c", autolag="aic")
            pairwise_p.iloc[i, j] = p_value
            pairwise_label.iloc[i, j] = f"{us.fmt_p(p_value)}; {'есть' if p_value < 0.05 else 'нет'}"
    us.save_table(pairwise_p, "cointegration_pairwise_pvalues")
    us.save_table(pairwise_label, "cointegration_pairwise_labels")
    us.plot_cointegration_heatmap(pairwise_p)

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
    us.save_table(johansen_table, "cointegration_johansen_trace")


def fit_var_models(transformed: pd.DataFrame) -> tuple[object, dict[str, object], pd.DataFrame]:
    var_model = VAR(transformed)
    selected_orders = var_model.select_order(maxlags=8).selected_orders
    selected_lag = int(selected_orders.get("aic") or selected_orders.get("bic") or 2)
    selected_lag = max(1, min(selected_lag, 8))

    var_fits = []
    candidate_lags = sorted(set([1, 2, selected_lag, max(1, selected_lag + 1)]))
    for lag in candidate_lags:
        fit = var_model.fit(lag, trend="c")
        var_fits.append(
            {
                "name": f"VAR({lag}), const",
                "lag": lag,
                "fit": fit,
                "stable": bool(fit.is_stable(verbose=False)),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "hqic": float(fit.hqic),
                "whiteness_p": float(fit.test_whiteness(nlags=max(lag + 4, 8)).pvalue),
                "normal_p": float(fit.test_normality().pvalue),
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
    best_var_info = sorted([v for v in var_fits if v["stable"]], key=lambda item: item["aic"])[0]
    best_var = best_var_info["fit"]

    var_compare = pd.DataFrame(
        [
            {
                "Спецификация": info["name"],
                "Устойчива": "да" if info["stable"] else "нет",
                "AIC": info["aic"],
                "BIC": info["bic"],
                "HQIC": info["hqic"],
                "Portmanteau p": us.fmt_p(info["whiteness_p"]),
                "Normality p": us.fmt_p(info["normal_p"]),
                "Выбрана по AIC": "да" if info is best_var_info else "",
            }
            for info in var_fits
        ]
    )
    us.save_table(var_compare, "var_model_comparison")

    var_du_eq = pd.DataFrame({"coef Δu equation": us.var_params_frame(best_var, list(transformed.columns))["d_unrate"]})
    var_du_eq.index.name = "Параметр"
    us.save_table(var_du_eq, "best_var_unemployment_equation")
    return best_var, best_var_info, var_compare


def build_granger_tables(best_var: object, transformed: pd.DataFrame) -> pd.DataFrame:
    granger_labels = [us.MODEL_LABELS[col] for col in transformed.columns]
    granger_p = pd.DataFrame(np.nan, index=granger_labels, columns=granger_labels)
    granger_p.index.name = "Причина"
    for cause in transformed.columns:
        for effect in transformed.columns:
            if cause == effect:
                continue
            test = best_var.test_causality(effect, [cause], kind="f")
            granger_p.loc[us.MODEL_LABELS[cause], us.MODEL_LABELS[effect]] = test.pvalue
    us.save_table(granger_p, "granger_causality_pvalues")
    us.plot_granger_network(granger_p)

    strongest_granger = (
        granger_p.stack()
        .dropna()
        .sort_values()
        .head(6)
        .reset_index()
        .rename(columns={"level_0": "Причина", "level_1": "Следствие", 0: "p-value"})
    )
    us.save_table(strongest_granger, "strongest_granger_links")
    return granger_p


def fit_adl_models(transformed: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    adl_specs = [
        us.fit_adl(transformed, "ADL-1: Δu_t = c + βg_t", 0, 0, False),
        us.fit_adl(transformed, "ADL-2: g_t,g_{t-1},g_{t-2},Δu_{t-1}", 2, 1, False),
        us.fit_adl(transformed, "ADL-3: ADL-2 + D2020Q2", 2, 1, True),
    ]
    best_adl = sorted(adl_specs, key=lambda item: item["aic"])[0]

    adl_compare = pd.DataFrame(
        [
            {
                "Спецификация": spec["name"],
                "N": spec["nobs"],
                "Adj.R2": spec["r2_adj"],
                "RMSE": spec["rmse"],
                "AIC": spec["aic"],
                "BIC": spec["bic"],
                "Ljung-Box p": us.fmt_p(spec["ljung_p"]),
                "Jarque-Bera p": us.fmt_p(spec["normal_p"]),
                "Выбрана для интерпретации": "да" if spec is best_adl else "",
            }
            for spec in adl_specs
        ]
    )
    us.save_table(adl_compare, "adl_model_comparison")

    adl_coef = pd.DataFrame(
        {
            "coef": best_adl["result"].params,
            "std err HAC": best_adl["result"].bse,
            "t": best_adl["result"].tvalues,
            "p-value": best_adl["result"].pvalues,
        }
    )
    adl_coef.index.name = "Параметр"
    us.save_table(adl_coef, "best_adl_coefficients")
    return best_adl, adl_compare


def save_model_comparison(adl_compare: pd.DataFrame, var_compare: pd.DataFrame) -> None:
    combined_compare = pd.DataFrame(
        [
            {
                "Спецификация модели": row["Спецификация"],
                "Стационарность модели": "стационарные переменные; OLS",
                "Ошибка модели, инф.критерии": f"RMSE={us.fmt_num(row['RMSE'])}; AIC={us.fmt_num(row['AIC'])}; BIC={us.fmt_num(row['BIC'])}",
                "Анализ остатков": f"Ljung-Box p={row['Ljung-Box p']}; JB p={row['Jarque-Bera p']}",
                "Выбор модели": row["Выбрана для интерпретации"],
            }
            for _, row in adl_compare.iterrows()
        ]
        + [
            {
                "Спецификация модели": row["Спецификация"],
                "Стационарность модели": f"устойчива: {row['Устойчива']}",
                "Ошибка модели, инф.критерии": f"AIC={us.fmt_num(row['AIC'])}; BIC={us.fmt_num(row['BIC'])}; HQIC={us.fmt_num(row['HQIC'])}",
                "Анализ остатков": f"Portmanteau p={row['Portmanteau p']}; Normality p={row['Normality p']}",
                "Выбор модели": row["Выбрана по AIC"],
            }
            for _, row in var_compare.iterrows()
        ]
    )
    us.save_table(combined_compare, "combined_model_comparison")


def build_us_core_analysis() -> dict[str, object]:
    levels, level_model, transformed = build_us_data()

    us.plot_levels(levels)
    us.plot_transformed(transformed)
    us.plot_okun_scatter(transformed)
    us.plot_acf_pacf(transformed)

    save_descriptive_tables(levels, transformed)
    build_stationarity_and_correlation_tables(level_model, transformed)
    build_cointegration_tables(level_model)

    best_var, best_var_info, var_compare = fit_var_models(transformed)
    build_granger_tables(best_var, transformed)
    best_adl, adl_compare = fit_adl_models(transformed)
    save_model_comparison(adl_compare, var_compare)

    us.plot_var_irf(best_var, list(transformed.columns))
    _, fevd_shares = us.plot_var_fevd(best_var, list(transformed.columns))
    fevd_selected = fevd_shares.loc[[4, 8, 12]].rename(columns=us.MODEL_LABELS)
    fevd_selected.index.name = "Горизонт"
    us.save_table(fevd_selected, "fevd_unemployment_selected_horizons")
    us.plot_residual_diagnostics(best_adl["result"], best_var, list(transformed.columns))

    data_preview = pd.concat([levels.head(5), levels.tail(5)]).reset_index()
    data_preview["quarter"] = data_preview["quarter"].astype(str)
    us.save_table(data_preview, "data_preview_levels")

    return {
        "levels": levels,
        "transformed": transformed,
        "best_adl": best_adl["name"],
        "best_var": best_var_info["name"],
    }


def build_micro_core_analysis() -> dict[str, object]:
    panel = micro.load_world_bank_panel()
    analysis = micro.cluster_analysis(panel)
    micro.build_figures(panel, analysis)
    representative = micro.representative_analysis(panel, analysis)
    return {
        "countries": int(analysis["assignments"]["iso3"].nunique()),
        "clusters": int(analysis["best_k"]),
        "representative": f"{representative['country']} ({representative['iso3']})",
    }


def build_vecm_and_calculation_images() -> None:
    vecm_tables = compliance.build_vecm_ecm_tables()
    compliance.build_screenshot_images(vecm_tables)


def main() -> None:
    us_result = build_us_core_analysis()
    micro_result = build_micro_core_analysis()
    build_vecm_and_calculation_images()

    print("Core analysis artifacts generated")
    print(f"US observations: levels={len(us_result['levels'])}, transformed={len(us_result['transformed'])}")
    print(f"Best ADL: {us_result['best_adl']}")
    print(f"Best VAR: {us_result['best_var']}")
    print(f"Countries: {micro_result['countries']}")
    print(f"Clusters: {micro_result['clusters']}")
    print(f"Representative: {micro_result['representative']}")
    print(f"US figures: {len(list(us.FIG.glob('*.png')))}")
    print(f"US tables: {len(list(us.TAB.glob('*.csv')))}")
    print(f"Micro figures: {len(list(micro.FIG.glob('*.png')))}")
    print(f"Micro tables: {len(list(micro.TAB.glob('*.csv')))}")
    print(f"Calculation images: {len(list(compliance.SHOT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
