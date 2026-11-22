"""
공간계량 분석용 시각화 모듈

matplotlib/geopandas 기반 표준 시각화.

제공 함수:
    - plot_choropleth          : 변수 단계구분도
    - plot_moran_scatter        : Moran 산점도
    - plot_lisa_cluster         : LISA 클러스터 맵
    - plot_weights_connectivity : W 행렬 연결 구조
    - plot_gwr_coefficients     : GWR 국지 계수 맵
    - plot_residual_map         : 잔차 공간 패턴
    - plot_model_comparison     : 모형 적합도 비교 (R²/AIC/BIC)
"""
from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import W, lag_spatial

logger = logging.getLogger(__name__)

# Brand palette
COLORS = {
    "navy": "#1B3A5C",
    "gold": "#D4AF37",
    "mint": "#4ecca3",
    "red": "#e74c3c",
    "blue": "#3498db",
    "orange": "#f39c12",
    "gray": "#95a5a6",
}

# LISA 표준 색상 (splot 기본)
LISA_COLORS = {
    "HH": "#d7191c",   # 진빨강
    "LL": "#2c7bb6",   # 진파랑
    "HL": "#fdae61",   # 주황
    "LH": "#abd9e9",   # 하늘
    "NS": "#cccccc",   # 회색 (비유의)
}


def _ensure_ax(ax=None, figsize=(8, 8)) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    scheme: str = "Quantiles",
    k: int = 5,
    cmap: str = "YlOrRd",
    ax=None,
    title: str | None = None,
) -> plt.Axes:
    """
    단계구분도.

    Args:
        gdf: 폴리곤 GeoDataFrame
        column: 매핑할 컬럼명
        scheme: 'Quantiles' | 'EqualInterval' | 'NaturalBreaks' | 'FisherJenks'
        k: 계급 수
    """
    if column not in gdf.columns:
        raise ValueError(f"컬럼 없음: {column}")
    ax = _ensure_ax(ax)
    gdf.plot(
        column=column,
        scheme=scheme,
        k=k,
        cmap=cmap,
        legend=True,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
        legend_kwds={"loc": "lower right", "fontsize": 8},
    )
    ax.set_axis_off()
    ax.set_title(title or column, fontsize=13)
    return ax


def plot_moran_scatter(
    values: np.ndarray,
    w: W,
    ax=None,
    title: str | None = None,
) -> plt.Axes:
    """
    Moran 산점도.

    x축: 표준화된 값 z
    y축: 공간지연 Wz
    기울기 = Moran's I

    4분면:
        Q1 (HH): +z, +Wz → 핫스팟
        Q2 (LH): -z, +Wz → 음의 아웃라이어
        Q3 (LL): -z, -Wz → 콜드스팟
        Q4 (HL): +z, -Wz → 양의 아웃라이어
    """
    values = np.asarray(values, dtype=float)
    z = (values - values.mean()) / values.std()
    Wz = lag_spatial(w, z)

    ax = _ensure_ax(ax)
    ax.scatter(z, Wz, s=40, alpha=0.7, color=COLORS["navy"], edgecolor="white")
    slope = float(np.polyfit(z, Wz, 1)[0])  # ≈ Moran's I (row-standardized W)
    xs = np.linspace(z.min(), z.max(), 100)
    ax.plot(xs, slope * xs + np.polyfit(z, Wz, 1)[1],
            color=COLORS["red"], lw=2, label=f"slope ≈ I = {slope:.3f}")
    ax.axhline(0, color=COLORS["gray"], lw=1, ls="--")
    ax.axvline(0, color=COLORS["gray"], lw=1, ls="--")
    ax.set_xlabel("z (standardized value)")
    ax.set_ylabel("Wz (spatial lag)")
    ax.set_title(title or "Moran Scatterplot", fontsize=13)
    ax.legend()
    return ax


def plot_lisa_cluster(
    gdf: gpd.GeoDataFrame,
    lisa_result,
    ax=None,
    title: str | None = None,
) -> plt.Axes:
    """
    LISA 클러스터 맵.

    Args:
        gdf: 분석 대상 GeoDataFrame
        lisa_result: diagnostics.local_moran() 결과
    """
    from matplotlib.patches import Patch

    ax = _ensure_ax(ax)
    labels = np.asarray(lisa_result.cluster_labels)
    colors = [LISA_COLORS.get(lb, "#cccccc") for lb in labels]
    gdf.plot(color=colors, edgecolor="white", linewidth=0.4, ax=ax)
    present = [lb for lb in ("HH", "LL", "HL", "LH", "NS") if lb in set(labels)]
    ax.legend(
        handles=[Patch(facecolor=LISA_COLORS[lb], label=lb) for lb in present],
        loc="lower right",
        fontsize=9,
    )
    ax.set_axis_off()
    ax.set_title(title or "LISA Cluster Map", fontsize=13)
    return ax


def plot_weights_connectivity(
    gdf: gpd.GeoDataFrame,
    w: W,
    ax=None,
    edge_alpha: float = 0.3,
    edge_color: str | None = None,
) -> plt.Axes:
    """W 행렬의 연결 구조 시각화 (이웃 엣지 그리기)."""
    ax = _ensure_ax(ax, figsize=(10, 10))
    edge_color = edge_color or COLORS["navy"]

    gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="white", linewidth=0.5)
    cent = gdf.geometry.centroid
    cx = pd.Series(cent.x.values, index=gdf.index)
    cy = pd.Series(cent.y.values, index=gdf.index)

    for i in w.id_order:
        for j in w.neighbors[i]:
            ax.plot(
                [cx[i], cx[j]], [cy[i], cy[j]],
                color=edge_color, alpha=edge_alpha, lw=0.8, zorder=2,
            )
    ax.scatter(cx, cy, s=12, color=COLORS["gold"], zorder=3)
    ax.set_axis_off()
    return ax


def plot_gwr_coefficients(
    gdf: gpd.GeoDataFrame,
    gwr_result,
    variable: str,
    cmap: str = "RdBu_r",
    ax=None,
) -> plt.Axes:
    """
    GWR의 국지적 계수 시각화 (변수 1개에 대해).

    Args:
        gwr_result: models.SpatialModel(method='GWR').fit()의 결과
            (SpatialResults) 또는 그 raw_result(mgwr GWRResults).
        variable: 계수를 그릴 변수명 (예: 'INC', 'CONSTANT')
    """
    # SpatialResults / mgwr GWRResults 모두 허용
    raw = getattr(gwr_result, "raw_result", gwr_result)
    names = getattr(raw, "name_x", None) or list(
        getattr(gwr_result, "coefficients", pd.Series(dtype=float)).index
    )
    if variable not in names:
        raise ValueError(f"변수 없음: {variable!r}. 가능한 변수: {names}")
    idx = names.index(variable)
    local_beta = np.asarray(raw.params)[:, idx]

    ax = _ensure_ax(ax)
    vmax = float(np.abs(local_beta).max())
    gdf.assign(local_beta=local_beta).plot(
        column="local_beta",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        legend=True,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(f"GWR local β — {variable}", fontsize=13)
    return ax


def plot_residual_map(
    gdf: gpd.GeoDataFrame,
    residuals: np.ndarray,
    title: str = "Model Residuals",
    ax=None,
) -> plt.Axes:
    """
    모형 잔차의 공간 분포.

    공간 패턴이 보이면 → SLM/SEM 추가 적용 필요 신호.
    """
    residuals = np.asarray(residuals, dtype=float).flatten()
    ax = _ensure_ax(ax)
    vmax = float(np.abs(residuals).max())
    gdf.assign(resid=residuals).plot(
        column="resid",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        legend=True,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    return ax


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    여러 모형의 R²/AIC/BIC 비교 막대그래프 (3-panel).

    Args:
        comparison_df: compare_models()의 반환 DataFrame
    """
    metrics = [("r2", "R² (↑ better)"), ("aic", "AIC (↓ better)"),
               ("bic", "BIC (↓ better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = comparison_df["method"].astype(str)
    palette = [COLORS["navy"], COLORS["blue"], COLORS["mint"], COLORS["gold"],
               COLORS["orange"], COLORS["red"]]
    for ax_i, (metric, label) in zip(axes, metrics, strict=True):
        vals = pd.to_numeric(comparison_df[metric], errors="coerce")
        ax_i.bar(x, vals, color=palette[: len(x)])
        ax_i.set_title(label, fontsize=12)
        ax_i.grid(axis="y", alpha=0.3)
        for spine in ("top", "right"):
            ax_i.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
