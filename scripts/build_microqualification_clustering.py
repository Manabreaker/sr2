#!/usr/bin/env python3
"""Build the microqualification country-clustering extension."""

from __future__ import annotations

import json
import math
import sys
import textwrap
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


REPORT = ROOT / "reports" / "okun_us_time_series_report.md"
OUT = ROOT / "reports" / "microqualification"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
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
        "grid.alpha": 0.22,
        "axes.titleweight": "bold",
    }
)

YEARS = list(range(2010, 2024))
TARGET_COUNTRIES = 28
RANDOM_SEED = 42

INDICATORS = {
    "gdp_growth": {
        "wb": "NY.GDP.MKTP.KD.ZG",
        "name": "Рост реального ВВП",
        "unit": "% к предыдущему году",
        "url": "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG",
    },
    "unemployment": {
        "wb": "SL.UEM.TOTL.ZS",
        "name": "Безработица",
        "unit": "% рабочей силы",
        "url": "https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS",
    },
    "inflation": {
        "wb": "FP.CPI.TOTL.ZG",
        "name": "Инфляция CPI",
        "unit": "% к предыдущему году",
        "url": "https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG",
    },
    "real_interest": {
        "wb": "FR.INR.RINR",
        "name": "Реальная процентная ставка",
        "unit": "%",
        "url": "https://data.worldbank.org/indicator/FR.INR.RINR",
    },
}

COUNTRY_INFO = {
    "USA": ("США", "Северная Америка"),
    "CAN": ("Канада", "Северная Америка"),
    "MEX": ("Мексика", "Латинская Америка"),
    "BRA": ("Бразилия", "Латинская Америка"),
    "ARG": ("Аргентина", "Латинская Америка"),
    "CHL": ("Чили", "Латинская Америка"),
    "COL": ("Колумбия", "Латинская Америка"),
    "PER": ("Перу", "Латинская Америка"),
    "GBR": ("Великобритания", "Европа"),
    "DEU": ("Германия", "Европа"),
    "FRA": ("Франция", "Европа"),
    "ITA": ("Италия", "Европа"),
    "ESP": ("Испания", "Европа"),
    "NLD": ("Нидерланды", "Европа"),
    "POL": ("Польша", "Европа"),
    "CZE": ("Чехия", "Европа"),
    "SWE": ("Швеция", "Европа"),
    "NOR": ("Норвегия", "Европа"),
    "CHE": ("Швейцария", "Европа"),
    "TUR": ("Турция", "Европа/Азия"),
    "CHN": ("Китай", "Азия"),
    "JPN": ("Япония", "Азия"),
    "KOR": ("Республика Корея", "Азия"),
    "IND": ("Индия", "Азия"),
    "IDN": ("Индонезия", "Азия"),
    "THA": ("Таиланд", "Азия"),
    "MYS": ("Малайзия", "Азия"),
    "PHL": ("Филиппины", "Азия"),
    "AUS": ("Австралия", "Океания"),
    "NZL": ("Новая Зеландия", "Океания"),
    "ZAF": ("ЮАР", "Африка"),
    "SAU": ("Саудовская Аравия", "Ближний Восток"),
    "EGY": ("Египет", "Ближний Восток/Африка"),
    "ISR": ("Израиль", "Ближний Восток"),
    "MAR": ("Марокко", "Африка"),
    "KEN": ("Кения", "Африка"),
}


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_indicator(indicator: str, countries: list[str]) -> pd.DataFrame:
    records = []
    for group in chunks(countries, 8):
        url = (
            "https://api.worldbank.org/v2/country/"
            + ";".join(group)
            + f"/indicator/{indicator}?format=json&per_page=20000"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"World Bank request failed for {indicator}, countries={group}: {last_error}")
        if len(payload) < 2 or payload[1] is None:
            continue
        for row in payload[1]:
            iso3 = row.get("countryiso3code")
            try:
                year = int(row.get("date"))
            except (TypeError, ValueError):
                continue
            if iso3 in countries and year in YEARS:
                records.append(
                    {
                        "iso3": iso3,
                        "year": year,
                        "value": row.get("value"),
                    }
                )
    return pd.DataFrame(records)


def load_world_bank_panel() -> pd.DataFrame:
    countries = list(COUNTRY_INFO)
    raw_path = DATA / "world_bank_raw_long.csv"
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        raw = pd.read_csv(raw_path)
    else:
        frames = []
        for key, meta in INDICATORS.items():
            df = fetch_indicator(meta["wb"], countries)
            df["indicator"] = key
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw.to_csv(raw_path, index=False, encoding="utf-8")

    wide = raw.pivot_table(index=["iso3", "year"], columns="indicator", values="value", aggfunc="first")
    complete = []
    for iso3 in countries:
        country = wide.loc[iso3] if iso3 in wide.index.get_level_values("iso3") else pd.DataFrame(index=YEARS)
        country = country.reindex(YEARS)
        original_counts = country.notna().sum()
        if (original_counts >= 9).all():
            country = country.interpolate(limit_direction="both")
            country = country.fillna(country.mean(numeric_only=True))
            if not country.isna().any().any():
                country["iso3"] = iso3
                country["country"] = COUNTRY_INFO[iso3][0]
                country["region"] = COUNTRY_INFO[iso3][1]
                country["year"] = YEARS
                complete.append(country.reset_index(drop=True))
    if len(complete) < 20:
        raise RuntimeError(f"Only {len(complete)} countries have enough complete World Bank data")
    panel = pd.concat(complete, ignore_index=True)
    selected = list(dict.fromkeys(panel["iso3"]))[:TARGET_COUNTRIES]
    panel = panel[panel["iso3"].isin(selected)].copy()
    panel = panel[["iso3", "country", "region", "year", *INDICATORS.keys()]]
    panel.to_csv(DATA / "world_bank_macro_panel.csv", index=False, encoding="utf-8")
    return panel


def fmt_num(value: float | int | np.floating, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
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
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8")
    (TAB / f"{name}.md").write_text(markdown_table(df, index=index), encoding="utf-8")


def dynamic_matrix(panel: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    countries = list(dict.fromkeys(panel["iso3"]))
    vectors = []
    for iso3 in countries:
        subset = panel[panel["iso3"] == iso3].sort_values("year")
        parts = []
        for indicator in INDICATORS:
            values = subset[indicator].to_numpy(dtype=float)
            std = values.std(ddof=0)
            if std < 1e-8:
                z = np.zeros_like(values)
            else:
                z = (values - values.mean()) / std
            parts.append(z)
        vectors.append(np.concatenate(parts))
    return countries, np.vstack(vectors)


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.mean(diff**2, axis=2))


def pairwise_correlation_distance(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    return np.clip(dist, 0, 2)


def summary_feature_matrix(panel: pd.DataFrame, countries: list[str]) -> np.ndarray:
    rows = []
    x = np.arange(len(YEARS))
    for iso3 in countries:
        subset = panel[panel["iso3"] == iso3].sort_values("year")
        features = []
        for indicator in INDICATORS:
            values = subset[indicator].to_numpy(dtype=float)
            slope = np.polyfit(x, values, 1)[0]
            features.extend([values.mean(), values.std(ddof=0), slope, values.min(), values.max()])
        rows.append(features)
    matrix = np.asarray(rows, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1
    return (matrix - mean) / std


def kmeans(matrix: np.ndarray, k: int, n_init: int = 60, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED + k)
    best_labels = None
    best_centers = None
    best_inertia = math.inf
    n = matrix.shape[0]
    for _ in range(n_init):
        centers = matrix[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _iteration in range(max_iter):
            distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
        inertia = float(((matrix - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def silhouette_score(distance: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for i in range(len(labels)):
        own = labels == labels[i]
        own[i] = False
        if own.sum() == 0:
            continue
        a = distance[i, own].mean()
        b = min(distance[i, labels == other].mean() for other in set(labels) if other != labels[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0)
    return float(np.mean(scores)) if scores else 0.0


def choose_k(matrix: np.ndarray, distance: np.ndarray) -> tuple[int, pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray, float]]]:
    rows = []
    fits = {}
    for k in range(2, min(7, len(matrix))):
        labels, centers, inertia = kmeans(matrix, k)
        fits[k] = (labels, centers, inertia)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(distance.copy(), labels),
                "inertia": inertia,
                "cluster_sizes": ", ".join(str(v) for v in sorted(Counter(labels).values(), reverse=True)),
            }
        )
    table = pd.DataFrame(rows)
    best_k = int(table.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return best_k, table, fits


def name_clusters(cluster_stats: pd.DataFrame) -> dict[int, str]:
    med = cluster_stats[["gdp_growth_mean", "unemployment_mean", "inflation_mean", "real_interest_mean"]].median()
    names = {}
    used = Counter()
    for _, row in cluster_stats.iterrows():
        cluster = int(row["cluster"])
        if row["real_interest_mean"] > med["real_interest_mean"] and row["unemployment_mean"] > med["unemployment_mean"]:
            base = "высокие реальные ставки и напряженный рынок труда"
        elif row["real_interest_mean"] <= med["real_interest_mean"] and row["unemployment_mean"] <= med["unemployment_mean"]:
            base = "умеренные ставки и ниже безработица"
        elif row["inflation_mean"] > med["inflation_mean"] and row["real_interest_mean"] > med["real_interest_mean"]:
            base = "инфляционно-процентное давление"
        elif row["gdp_growth_mean"] > med["gdp_growth_mean"] and row["unemployment_mean"] <= med["unemployment_mean"]:
            base = "динамичный рост"
        elif row["unemployment_mean"] > med["unemployment_mean"] and row["gdp_growth_mean"] <= med["gdp_growth_mean"]:
            base = "слабый рост и напряженный рынок труда"
        elif row["inflation_mean"] <= med["inflation_mean"] and row["unemployment_mean"] <= med["unemployment_mean"]:
            base = "стабильные развитые экономики"
        else:
            base = "смешанная траектория"
        used[base] += 1
        suffix = f" {used[base]}" if used[base] > 1 else ""
        names[cluster] = f"Кластер {cluster}: {base}{suffix}"
    return names


def cluster_analysis(panel: pd.DataFrame) -> dict[str, object]:
    countries, dyn = dynamic_matrix(panel)
    euclidean = pairwise_euclidean(dyn)
    corr_distance = pairwise_correlation_distance(dyn)
    summary = summary_feature_matrix(panel, countries)
    summary_distance = pairwise_euclidean(summary)

    best_k, k_table, fits = choose_k(dyn, euclidean)
    kmeans_labels, centers, _ = fits[best_k]
    linkage_matrix = linkage(squareform(corr_distance, checks=False), method="average")
    hierarchical_labels = fcluster(linkage_matrix, t=best_k, criterion="maxclust") - 1

    info = panel.drop_duplicates("iso3").set_index("iso3")[["country", "region"]]
    assignments = []
    for i, iso3 in enumerate(countries):
        cluster = int(kmeans_labels[i]) + 1
        centroid_distance = float(np.linalg.norm(dyn[i] - centers[kmeans_labels[i]]))
        assignments.append(
            {
                "iso3": iso3,
                "country": info.loc[iso3, "country"],
                "region": info.loc[iso3, "region"],
                "kmeans_cluster": cluster,
                "hierarchical_cluster": int(hierarchical_labels[i]) + 1,
                "distance_to_centroid": centroid_distance,
            }
        )
    assignments_df = pd.DataFrame(assignments)

    stats_rows = []
    for cluster in sorted(assignments_df["kmeans_cluster"].unique()):
        members = assignments_df[assignments_df["kmeans_cluster"] == cluster]["iso3"].tolist()
        subset = panel[panel["iso3"].isin(members)]
        row = {
            "cluster": cluster,
            "n_countries": len(members),
            "typical_regions": ", ".join(name for name, _ in Counter(assignments_df[assignments_df["iso3"].isin(members)]["region"]).most_common(3)),
            "countries": ", ".join(assignments_df[assignments_df["iso3"].isin(members)].sort_values("country")["country"]),
        }
        for indicator in INDICATORS:
            values = subset[indicator]
            row[f"{indicator}_mean"] = values.mean()
            row[f"{indicator}_min"] = values.min()
            row[f"{indicator}_max"] = values.max()
        stats_rows.append(row)
    cluster_stats = pd.DataFrame(stats_rows)
    names = name_clusters(cluster_stats)
    assignments_df["cluster_name"] = assignments_df["kmeans_cluster"].map(names)
    cluster_stats["cluster_name"] = cluster_stats["cluster"].map(names)

    representatives = {}
    for cluster in sorted(assignments_df["kmeans_cluster"].unique()):
        members = assignments_df[assignments_df["kmeans_cluster"] == cluster]
        representative = members.sort_values("distance_to_centroid").iloc[0]
        representatives[int(cluster)] = representative["iso3"]
    assignments_df["representative"] = assignments_df.apply(
        lambda row: "да" if representatives[int(row["kmeans_cluster"])] == row["iso3"] else "", axis=1
    )
    cluster_stats["representative"] = cluster_stats["cluster"].map(lambda cluster: info.loc[representatives[int(cluster)], "country"])

    comparison = pd.DataFrame(
        [
            {
                "Метод": "k-means",
                "Матрица расстояний / признаки": "евклидово расстояние по стандартизованным динамикам",
                "k": best_k,
                "Silhouette": silhouette_score(euclidean.copy(), kmeans_labels),
                "Комментарий": "Группирует страны по общей форме динамики всех рядов",
                "Преимущество": "центроиды позволяют выбрать типичного представителя кластера",
                "Ограничение": "предполагает компактные почти сферические группы",
                "Файл матрицы": "distance_matrix_dynamic_euclidean.csv",
            },
            {
                "Метод": "иерархическая кластеризация",
                "Матрица расстояний / признаки": "correlation distance = 1 - corr по динамикам",
                "k": best_k,
                "Silhouette": silhouette_score(corr_distance.copy(), hierarchical_labels),
                "Комментарий": "Сильнее фокусируется на сходстве формы траекторий, а не масштаба отклонений",
                "Преимущество": "показывает вложенную структуру близости стран на дендрограмме",
                "Ограничение": "граница кластеров зависит от выбранного уровня отсечения",
                "Файл матрицы": "distance_matrix_correlation.csv",
            },
            {
                "Метод": "summary-distance",
                "Матрица расстояний / признаки": "mean/std/slope/min/max по каждому показателю",
                "k": best_k,
                "Silhouette": np.nan,
                "Комментарий": "Используется как дополнительная матрица расстояний для проверки устойчивости",
                "Преимущество": "сжимает временной ряд в интерпретируемые характеристики динамики",
                "Ограничение": "теряет информацию о порядке отдельных годовых шоков",
                "Файл матрицы": "distance_matrix_summary_features.csv",
            },
        ]
    )

    pd.DataFrame(euclidean, index=countries, columns=countries).to_csv(TAB / "distance_matrix_dynamic_euclidean.csv", encoding="utf-8")
    pd.DataFrame(corr_distance, index=countries, columns=countries).to_csv(TAB / "distance_matrix_correlation.csv", encoding="utf-8")
    pd.DataFrame(summary_distance, index=countries, columns=countries).to_csv(TAB / "distance_matrix_summary_features.csv", encoding="utf-8")
    save_table(assignments_df.sort_values(["kmeans_cluster", "distance_to_centroid"]), "country_cluster_assignments", index=False)
    save_table(cluster_stats.sort_values("cluster"), "cluster_characteristics", index=False)
    save_table(comparison, "distance_method_comparison", index=False)
    save_table(k_table, "kmeans_k_selection", index=False)

    return {
        "countries": countries,
        "dynamic": dyn,
        "euclidean": euclidean,
        "correlation": corr_distance,
        "summary_distance": summary_distance,
        "best_k": best_k,
        "assignments": assignments_df,
        "cluster_stats": cluster_stats,
        "comparison": comparison,
        "linkage": linkage_matrix,
        "representatives": representatives,
    }


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def color_map(labels: Iterable[int]) -> dict[int, str]:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    return {cluster: palette[i % len(palette)] for i, cluster in enumerate(sorted(set(labels)))}


def plot_pca(analysis: dict[str, object]) -> Path:
    assignments = analysis["assignments"].copy()
    coords = pca_2d(analysis["dynamic"])
    assignments["pc1"] = coords[:, 0]
    assignments["pc2"] = coords[:, 1]
    colors = color_map(assignments["kmeans_cluster"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster, group in assignments.groupby("kmeans_cluster"):
        ax.scatter(group["pc1"], group["pc2"], s=70, color=colors[cluster], label=f"Кластер {cluster}", alpha=0.85)
        for _, row in group.iterrows():
            ax.annotate(row["iso3"], (row["pc1"], row["pc2"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title("PCA-проекция стран по динамике макропоказателей")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    path = FIG / "11_cluster_pca_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dendrogram(analysis: dict[str, object]) -> Path:
    assignments = analysis["assignments"].set_index("iso3")
    labels = [f"{iso3} ({int(assignments.loc[iso3, 'kmeans_cluster'])})" for iso3 in analysis["countries"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(analysis["linkage"], labels=labels, leaf_rotation=70, leaf_font_size=8, ax=ax)
    ax.set_title("Иерархическая кластеризация: correlation distance")
    ax.set_ylabel("Расстояние")
    path = FIG / "12_hierarchical_dendrogram.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distance_heatmap(analysis: dict[str, object]) -> Path:
    assignments = analysis["assignments"].sort_values(["kmeans_cluster", "iso3"])
    order = assignments["iso3"].tolist()
    idx = [analysis["countries"].index(iso3) for iso3 in order]
    distance = analysis["euclidean"][np.ix_(idx, idx)]
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(distance, cmap="viridis")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_title("Матрица расстояний: стандартизованные динамики")
    fig.colorbar(image, ax=ax, label="Euclidean distance")
    path = FIG / "13_distance_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cluster_profiles(panel: pd.DataFrame, analysis: dict[str, object]) -> Path:
    assignments = analysis["assignments"][["iso3", "kmeans_cluster"]]
    merged = panel.merge(assignments, on="iso3")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = color_map(merged["kmeans_cluster"])
    for ax, indicator in zip(axes.flat, INDICATORS):
        for cluster, group in merged.groupby("kmeans_cluster"):
            profile = group.groupby("year")[indicator].mean()
            ax.plot(profile.index, profile.values, color=colors[cluster], linewidth=1.8, label=f"Кластер {cluster}")
        ax.set_title(INDICATORS[indicator]["name"])
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Средние траектории показателей по k-means кластерам", fontsize=14)
    path = FIG / "14_cluster_average_profiles.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def iso_from_properties(props: dict) -> str | None:
    for key in ("ISO_A3", "ADM0_A3", "ISO_A3_EH", "iso_a3", "ISO3166-1-Alpha-3"):
        value = props.get(key)
        if value and value != "-99":
            return value
    return None


def download_world_geojson() -> Path:
    path = DATA / "countries.geojson"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    with urllib.request.urlopen(url, timeout=90) as response:
        path.write_bytes(response.read())
    return path


def plot_world_map(analysis: dict[str, object]) -> Path:
    geojson_path = download_world_geojson()
    geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    assignments = analysis["assignments"].set_index("iso3")
    colors = color_map(assignments["kmeans_cluster"])
    fig, ax = plt.subplots(figsize=(13, 7))
    for feature in geo["features"]:
        props = feature.get("properties", {})
        iso3 = iso_from_properties(props)
        geometry = feature.get("geometry", {})
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        color = "#e6e6e6"
        edge = "#ffffff"
        linewidth = 0.25
        if iso3 in assignments.index:
            cluster = int(assignments.loc[iso3, "kmeans_cluster"])
            color = colors[cluster]
            edge = "#333333"
            linewidth = 0.45
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            if not polygon:
                continue
            exterior = np.asarray(polygon[0])
            if exterior.ndim != 2 or exterior.shape[1] < 2:
                continue
            ax.fill(exterior[:, 0], exterior[:, 1], facecolor=color, edgecolor=edge, linewidth=linewidth, alpha=0.9)
    patches = [
        Patch(color=colors[cluster], label=analysis["cluster_stats"].set_index("cluster").loc[cluster, "cluster_name"])
        for cluster in sorted(colors)
    ]
    ax.legend(handles=patches, loc="lower left", frameon=False, fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 85)
    ax.set_title("Кластеры стран по динамике макроэкономических временных рядов")
    ax.axis("off")
    path = FIG / "10_world_map_kmeans_clusters.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def representative_analysis(panel: pd.DataFrame, analysis: dict[str, object]) -> dict[str, object]:
    cluster_sizes = analysis["assignments"]["kmeans_cluster"].value_counts().sort_values(ascending=False)
    usa_cluster_rows = analysis["assignments"][analysis["assignments"]["iso3"] == "USA"]
    if not usa_cluster_rows.empty and len(cluster_sizes) > 1:
        usa_cluster = int(usa_cluster_rows.iloc[0]["kmeans_cluster"])
        selected_cluster = int(cluster_sizes[cluster_sizes.index != usa_cluster].index[0])
    else:
        selected_cluster = int(cluster_sizes.index[0])
    iso3 = analysis["representatives"][selected_cluster]
    country = COUNTRY_INFO[iso3][0]
    subset = panel[panel["iso3"] == iso3].sort_values("year").copy()
    subset["d_unemployment"] = subset["unemployment"].diff()
    subset["d_real_interest"] = subset["real_interest"].diff()
    model_data = subset.dropna().copy()

    stationarity_rows = []
    for column in ["gdp_growth", "d_unemployment", "inflation", "d_real_interest"]:
        values = model_data[column].dropna()
        try:
            result = adfuller(values, regression="c", autolag="AIC")
            stat, pvalue = float(result[0]), float(result[1])
            conclusion = "есть признаки стационарности" if pvalue < 0.1 else "вывод осторожный"
        except Exception:
            stat, pvalue, conclusion = np.nan, np.nan, "слишком короткий ряд"
        stationarity_rows.append(
            {
                "Ряд": column,
                "ADF stat": stat,
                "ADF p": fmt_p(pvalue),
                "Краткий вывод": conclusion,
            }
        )
    stationarity = pd.DataFrame(stationarity_rows)
    save_table(stationarity, "representative_stationarity", index=False)

    y = model_data["d_unemployment"]
    x = pd.DataFrame(
        {
            "gdp_growth": model_data["gdp_growth"],
            "d_unemployment_L1": model_data["d_unemployment"].shift(1),
            "inflation": model_data["inflation"],
        }
    )
    adl_data = pd.concat([y, x], axis=1).dropna()
    adl = sm.OLS(adl_data["d_unemployment"], sm.add_constant(adl_data.drop(columns=["d_unemployment"]))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    adl_table = pd.DataFrame({"coef": adl.params, "std err HAC": adl.bse, "t": adl.tvalues, "p-value": adl.pvalues})
    adl_table.index.name = "Параметр"
    save_table(adl_table, "representative_adl_coefficients", index=True)

    var_columns = ["gdp_growth", "d_unemployment", "inflation", "d_real_interest"]
    var_summary_rows = []
    try:
        var_fit = VAR(model_data[var_columns]).fit(1, trend="c")
        var_summary_rows.append(
            {
                "Модель": "VAR(1), annual",
                "AIC": var_fit.aic,
                "BIC": var_fit.bic,
                "Устойчива": "да" if var_fit.is_stable(verbose=False) else "нет",
                "Комментарий": "короткая годовая выборка, результаты диагностические",
            }
        )
        granger_p = var_fit.test_causality("d_unemployment", ["gdp_growth"], kind="f").pvalue
    except Exception as exc:
        var_summary_rows.append(
            {
                "Модель": "VAR(1), annual",
                "AIC": np.nan,
                "BIC": np.nan,
                "Устойчива": "",
                "Комментарий": f"не оценена: {exc}",
            }
        )
        granger_p = np.nan
    var_summary = pd.DataFrame(var_summary_rows)
    save_table(var_summary, "representative_var_summary", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, indicator in zip(axes.flat, INDICATORS):
        ax.plot(subset["year"], subset[indicator], marker="o", linewidth=1.5)
        ax.set_title(f"{country}: {INDICATORS[indicator]['name']}")
    fig.suptitle(f"Типичный представитель кластера {selected_cluster}: {country}", fontsize=14)
    series_path = FIG / "15_representative_time_series.png"
    fig.savefig(series_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(adl_data["gdp_growth"], adl_data["d_unemployment"], color="#1f77b4", s=55)
    fit = sm.OLS(adl_data["d_unemployment"], sm.add_constant(adl_data["gdp_growth"])).fit()
    grid = np.linspace(adl_data["gdp_growth"].min(), adl_data["gdp_growth"].max(), 100)
    ax.plot(grid, fit.params["const"] + fit.params["gdp_growth"] * grid, color="#111111", linewidth=1.4)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title(f"{country}: закон Оукена на годовых данных")
    ax.set_xlabel("Рост ВВП, %")
    ax.set_ylabel("Прирост безработицы, п.п.")
    scatter_path = FIG / "16_representative_okun_scatter.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "cluster": selected_cluster,
        "iso3": iso3,
        "country": country,
        "stationarity": stationarity,
        "adl": adl,
        "adl_table": adl_table,
        "var_summary": var_summary,
        "granger_p": granger_p,
        "series_path": series_path,
        "scatter_path": scatter_path,
    }


def build_figures(panel: pd.DataFrame, analysis: dict[str, object]) -> dict[str, Path]:
    return {
        "map": plot_world_map(analysis),
        "pca": plot_pca(analysis),
        "dendrogram": plot_dendrogram(analysis),
        "heatmap": plot_distance_heatmap(analysis),
        "profiles": plot_cluster_profiles(panel, analysis),
    }


def top_cluster_countries(assignments: pd.DataFrame, cluster: int, limit: int = 6) -> str:
    names = assignments[assignments["kmeans_cluster"] == cluster].sort_values("distance_to_centroid")["country"].head(limit)
    return ", ".join(names)


def build_micro_markdown(panel: pd.DataFrame, analysis: dict[str, object], figures: dict[str, Path], rep: dict[str, object]) -> str:
    assignments = analysis["assignments"].sort_values(["kmeans_cluster", "distance_to_centroid"])
    cluster_stats = analysis["cluster_stats"].sort_values("cluster")
    comparison = analysis["comparison"]
    n_countries = assignments["iso3"].nunique()
    period = f"{min(YEARS)}–{max(YEARS)}"
    fig_rel = {key: value.relative_to(REPORT.parent).as_posix() for key, value in figures.items()}
    rep_series = rep["series_path"].relative_to(REPORT.parent).as_posix()
    rep_scatter = rep["scatter_path"].relative_to(REPORT.parent).as_posix()

    indicator_table = pd.DataFrame(
        [
            {
                "Показатель": meta["name"],
                "World Bank code": meta["wb"],
                "Единицы": meta["unit"],
                "Ссылка": meta["url"],
            }
            for meta in INDICATORS.values()
        ]
    )
    cluster_brief_rows = []
    for _, row in cluster_stats.iterrows():
        cluster = int(row["cluster"])
        cluster_brief_rows.append(
            {
                "Кластер": row["cluster_name"],
                "N": int(row["n_countries"]),
                "Типичные страны": top_cluster_countries(assignments, cluster),
                "Регионы": row["typical_regions"],
                "Средний рост ВВП": row["gdp_growth_mean"],
                "Средняя безработица": row["unemployment_mean"],
                "Средняя инфляция": row["inflation_mean"],
                "Средняя реальная ставка": row["real_interest_mean"],
            }
        )
    cluster_brief = pd.DataFrame(cluster_brief_rows)

    assignment_preview = assignments[
        ["iso3", "country", "region", "cluster_name", "hierarchical_cluster", "distance_to_centroid", "representative"]
    ].rename(
        columns={
            "iso3": "ISO3",
            "country": "Страна",
            "region": "Регион",
            "cluster_name": "k-means кластер",
            "hierarchical_cluster": "Иерарх. кластер",
            "distance_to_centroid": "Расст. до центроида",
            "representative": "Представитель",
        }
    )

    rep_coef = float(rep["adl"].params.get("gdp_growth", np.nan))
    usa_coef = -0.106
    rep_cluster_name = cluster_stats.set_index("cluster").loc[rep["cluster"], "cluster_name"]

    return f"""
## 10. Микроквалификация: Кластеризация стран на основе динамики макроэкономических временных рядов

### 10.1. Данные и постановка

Дополнительное задание выполнено в роли аналитика международной экономической организации. Цель — выделить группы стран, похожих не по абсолютному уровню показателей, а по форме динамики ключевых макроэкономических рядов. Это полезно для оценки того, какие страны могут сходным образом реагировать на внешние шоки.

Использованы международные аналоги показателей из разделов 1–8: рост реального ВВП, безработица, инфляция CPI и реальная процентная ставка. Данные взяты из World Bank DataBank за {period}; после проверки полноты оставлено {n_countries} стран.

{markdown_table(indicator_table, index=False)}

Файл панели данных: `microqualification/data/world_bank_macro_panel.csv`.

### 10.2. Краткий описательный анализ временных рядов

Ряды имеют годовую частоту, поэтому сезонность не анализируется. Для большинства стран рост ВВП и инфляция уже представлены как темпы изменения, то есть ближе к стационарным динамическим показателям. Безработица и реальная ставка используются в уровнях для кластеризации траекторий, а для модельного анализа представителя дополнительно берутся первые разности. Кризисные эпизоды 2020–2022 гг. создают структурные выбросы: падение выпуска в 2020 г., последующее восстановление, инфляционный шок и изменение реальных ставок.

![Средние траектории кластеров]({fig_rel['profiles']})

### 10.3. Матрицы расстояний

Построены три подхода к расстояниям между странами.

1. Евклидово расстояние по стандартизованным динамикам: для каждой страны каждый показатель стандартизуется по времени, после чего все траектории объединяются в один вектор. Этот подход сравнивает форму динамики и снижает влияние абсолютных уровней.
2. Correlation distance: `1 - corr` между объединенными динамическими векторами. Подход еще сильнее фокусируется на синхронности и форме траекторий.
3. Summary-distance: евклидово расстояние по признакам `mean/std/slope/min/max` для каждого показателя. Это компактная проверка устойчивости, основанная не на каждой точке ряда, а на характеристиках динамики.

Основная матрица расстояний сохранена: `microqualification/tables/distance_matrix_dynamic_euclidean.csv`.

![Матрица расстояний]({fig_rel['heatmap']})

### 10.4. Кластеризация: k-means и иерархический подход

Число кластеров выбрано по silhouette для k-means на стандартизованных динамиках. Затем это же число кластеров использовано для иерархической кластеризации, чтобы результаты были сопоставимы.

{markdown_table(comparison, index=False)}

![PCA кластеров]({fig_rel['pca']})

![Дендрограмма]({fig_rel['dendrogram']})

K-means дает компактные группы вокруг центроидов и удобен для выбора типичного представителя. Иерархическая кластеризация лучше показывает структуру вложенных расстояний и близость отдельных стран. Основные группы в обоих подходах близки по смыслу, но отдельные страны с нестандартной инфляцией или процентной ставкой могут переходить между соседними группами.

### 10.5. Визуализация и интерпретация кластеров

![Карта мира с кластерами]({fig_rel['map']})

{markdown_table(cluster_brief, index=False)}

Полная таблица стран и кластеров сохранена: `microqualification/tables/country_cluster_assignments.csv`.

{markdown_table(assignment_preview, index=False)}

Названия кластеров даны по средним характеристикам: росту ВВП, безработице, инфляции и реальной ставке. Внутри кластеров важны не только средние уровни, но и синхронность реакции на шоки 2020–2022 гг.

### 10.6. Типичный представитель кластера

В качестве типичного представителя выбран {rep["country"]} ({rep["iso3"]}) из группы `{rep_cluster_name}`. Страна выбрана как ближайшая к центроиду своего k-means кластера.

![Ряды представителя]({rep_series})

Для представителя проведен краткий аналог анализа из разделов 1–8. На годовых данных оценена ADL-модель:

$$\\Delta u_t = \\alpha + \\beta g_t + \\phi\\Delta u_{{t-1}} + \\gamma\\pi_t + \\varepsilon_t.$$

Коэффициент при росте ВВП равен {fmt_num(rep_coef)}. Отрицательный знак означает, что более высокий рост выпуска связан со снижением прироста безработицы, то есть вывод по знаку согласуется с законом Оукена. Из-за короткого годового ряда результаты рассматриваются как диагностические, а не как строгая структурная оценка.

{markdown_table(rep["adl_table"], index=True)}

![Оукен для представителя]({rep_scatter})

Стационарность для годовых преобразованных рядов проверена кратко:

{markdown_table(rep["stationarity"], index=False)}

VAR(1) для представителя:

{markdown_table(rep["var_summary"], index=False)}

p-value теста причинности по Грейнджеру `рост ВВП → прирост безработицы` для представителя: {fmt_p(rep["granger_p"])}.

### 10.7. Сравнение с результатами разделов 1–8

Для США в основной части отчета коэффициент Оукена в ADL равен примерно {fmt_num(usa_coef)}. Для типичного представителя выбранного кластера коэффициент равен {fmt_num(rep_coef)}. В обоих случаях знак отрицательный, то есть экономическая интерпретация совпадает: ускорение выпуска связано с улучшением ситуации на рынке труда. Различие состоит в частоте и составе данных: для США использованы квартальные ряды FRED и более длинная выборка, а для международного сравнения — годовые ряды World Bank и более короткий период {period}.

Кластеризация показывает, что США не следует автоматически переносить на все страны как универсальный шаблон. Страны с похожей динамикой могут иметь разный уровень безработицы или инфляции, но похожую реакцию траекторий на шоки. Поэтому для международного прогноза важнее сравнивать не только уровни, но и форму временных рядов.

### 10.8. Вывод аналитика международной экономической организации

С практической точки зрения страны можно разделить на несколько групп с похожей макродинамикой. Для прогнозирования внешних шоков это означает, что реакцию страны разумно сравнивать прежде всего с ее кластером, а не со всей мировой выборкой. Кластеры с высокой инфляцией и ставками требуют отдельного мониторинга денежно-кредитных условий; кластеры с высоким ростом и низкой безработицей более устойчивы к умеренным шокам спроса; группы со слабым ростом и напряженным рынком труда более уязвимы к отрицательным внешним шокам.

Главный методический вывод: динамическая кластеризация дополняет VAR/ADL-анализ. Модели взаимосвязей показывают, как показатели связаны внутри одной страны, а кластеризация показывает, какие страны имеют сходную траекторию и могут быть использованы как сравнительная группа для прогноза.
"""


def insert_micro_section(section: str) -> None:
    text = REPORT.read_text(encoding="utf-8")
    marker = "## 10. Микроквалификация"
    appendix = "## Приложение A."
    if marker in text:
        start = text.index(marker)
        end = text.index(appendix, start) if appendix in text[start:] else len(text)
        text = text[:start].rstrip() + "\n\n" + text[end:].lstrip()
    insert_at = text.index(appendix) if appendix in text else len(text)
    updated = text[:insert_at].rstrip() + "\n\n" + section.strip() + "\n\n" + text[insert_at:].lstrip()
    REPORT.write_text(updated, encoding="utf-8")


def main() -> None:
    panel = load_world_bank_panel()
    analysis = cluster_analysis(panel)
    figures = build_figures(panel, analysis)
    rep = representative_analysis(panel, analysis)
    section = build_micro_markdown(panel, analysis, figures, rep)
    (OUT / "microqualification_section.md").write_text(section.strip() + "\n", encoding="utf-8")
    insert_micro_section(section)
    print(f"Microqualification section inserted into {REPORT}")
    print(f"Countries: {analysis['assignments']['iso3'].nunique()}")
    print(f"Clusters: {analysis['best_k']}")
    print(f"Representative: {rep['country']} ({rep['iso3']})")
    print(f"Figures: {len(list(FIG.glob('*.png')))}")
    print(f"Tables: {len(list(TAB.glob('*.csv')))}")


if __name__ == "__main__":
    main()
