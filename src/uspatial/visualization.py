"""
공간계량 분석용 시각화 모듈

splot 기반 표준 시각화 + custom plot 확장.

제공 함수:
    - plot_choropleth          : 변수 단계구분도
    - plot_moran_scatter        : Moran 산점도
    - plot_lisa_cluster         : LISA 클러스터 맵
    - plot_weights_connectivity : W 행렬 연결 구조
    - plot_gwr_coefficients     : GWR 국지 계수 맵
    - plot_residual_map         : 잔차 공간 패턴
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import W

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


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    scheme: str = "Quantiles",
    k: int = 5,
    cmap: str = "YlOrRd",
    ax=None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    단계구분도.

    Args:
        gdf: 폴리곤 GeoDataFrame
        column: 매핑할 컬럼명
        scheme: 'Quantiles' | 'EqualInterval' | 'NaturalBreaks' | 'FisherJenks'
        k: 계급 수
    """
    raise NotImplementedError(
        "TODO: gdf.plot(column=column, scheme=scheme, k=k, cmap=cmap, legend=True, ax=ax)"
    )


def plot_moran_scatter(
    values: np.ndarray,
    w: W,
    ax=None,
    title: Optional[str] = None,
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
    raise NotImplementedError(
        "TODO: from splot.esda import moran_scatterplot; "
        "또는 z = (values-mean)/std; Wz = w.sparse @ z; ax.scatter(z, Wz)"
    )


def plot_lisa_cluster(
    gdf: gpd.GeoDataFrame,
    lisa_result,
    ax=None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    LISA 클러스터 맵.

    Args:
        gdf: 분석 대상 GeoDataFrame
        lisa_result: diagnostics.local_moran() 결과
    """
    raise NotImplementedError(
        "TODO: gdf.assign(cluster=lisa_result.cluster_labels)"
        ".plot(column='cluster', categorical=True, "
        "color=[LISA_COLORS[c] for c in ...])"
    )


def plot_weights_connectivity(
    gdf: gpd.GeoDataFrame,
    w: W,
    ax=None,
    edge_alpha: float = 0.3,
    edge_color: str = None,
) -> plt.Axes:
    """W 행렬의 연결 구조 시각화 (이웃 엣지 그리기)."""
    raise NotImplementedError(
        "TODO: gdf.plot(ax=ax); for i in w.id_order: for j in w.neighbors[i]: "
        "ax.plot([centroid_i.x, centroid_j.x], [centroid_i.y, centroid_j.y], ...)"
    )


def plot_gwr_coefficients(
    gdf: gpd.GeoDataFrame,
    gwr_result,
    variable: str,
    cmap: str = "RdBu_r",
    ax=None,
) -> plt.Axes:
    """GWR의 국지적 계수 시각화 (변수 1개에 대해)."""
    raise NotImplementedError(
        "TODO: gdf['local_beta'] = gwr_result.params[:, var_idx]; "
        "gdf.plot(column='local_beta', cmap=cmap, legend=True)"
    )


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
    raise NotImplementedError(
        "TODO: gdf.assign(resid=residuals).plot(column='resid', cmap='RdBu_r', legend=True)"
    )


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    여러 모형의 AIC/BIC/R² 비교 막대그래프 (3-panel).

    Args:
        comparison_df: compare_models()의 반환 DataFrame
    """
    raise NotImplementedError(
        "TODO: fig, axes = plt.subplots(1, 3, figsize=(15, 4)); "
        "for i, metric in enumerate(['r2', 'aic', 'bic']): axes[i].bar(...)"
    )
