"""
공간가중행렬 (Spatial Weights Matrix) 생성

PySAL의 libpysal.weights 래퍼. 5가지 방식 지원:

1. **Queen**      : 폴리곤 인접성 (공유 엣지 또는 꼭짓점, 8방향)
2. **Rook**       : 폴리곤 인접성 (공유 엣지만, 4방향)
3. **KNN**        : K-Nearest Neighbors (포인트 기본)
4. **Distance**   : 거리 밴드 (임계거리 내 모두 이웃)
5. **Kernel**     : 커널 가중 (거리가 가까울수록 큰 가중치)

모든 W는 기본적으로 row-standardize 되어 반환 (Σⱼ wᵢⱼ = 1).
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import geopandas as gpd
from libpysal.weights import W, Queen, Rook, KNN, DistanceBand, Kernel

logger = logging.getLogger(__name__)

WeightType = Literal["queen", "rook", "knn", "distance", "kernel"]


def build_weights(
    gdf: gpd.GeoDataFrame,
    method: WeightType = "queen",
    k: int = 8,
    threshold: Optional[float] = None,
    kernel_function: str = "triangular",
    bandwidth: Optional[float] = None,
    row_standardize: bool = True,
) -> W:
    """
    공간가중행렬 생성 통합 인터페이스.

    Args:
        gdf: 공간 단위. Queen/Rook는 Polygon, KNN/Distance는 Point/Polygon centroid.
        method: 'queen' | 'rook' | 'knn' | 'distance' | 'kernel'
        k: KNN에서 최근접 이웃 수
        threshold: DistanceBand 임계거리 (미터). None이면 최소 임계(≥ 1 neighbor) 자동 탐색
        kernel_function: 'uniform' | 'triangular' | 'quadratic' | 'quartic' | 'gaussian'
        bandwidth: 커널 bandwidth. None이면 자동 (max nearest-neighbor distance)
        row_standardize: True면 W_ij = 1/n_i

    Returns:
        libpysal.weights.W

    Raises:
        ValueError: method 미지원 또는 gdf가 빈 DataFrame인 경우
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError("빈 GeoDataFrame입니다.")

    if method in ("queen", "rook"):
        ctor = Queen if method == "queen" else Rook
        w = ctor.from_dataframe(gdf, use_index=True)
    elif method == "knn":
        pts = _as_points(gdf)
        w = KNN.from_dataframe(pts, k=k)
    elif method == "distance":
        pts = _as_points(gdf)
        thr = threshold if threshold is not None else auto_distance_threshold(pts)
        w = DistanceBand.from_dataframe(
            pts, threshold=thr, binary=True, silence_warnings=True
        )
    elif method == "kernel":
        pts = _as_points(gdf)
        w = Kernel.from_dataframe(
            pts, function=kernel_function, bandwidth=bandwidth, fixed=True
        )
    else:
        raise ValueError(
            f"지원하지 않는 method: {method!r}. "
            "'queen'|'rook'|'knn'|'distance'|'kernel' 중 선택."
        )

    if row_standardize:
        w.transform = "R"

    return w


def _as_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """폴리곤이면 centroid 포인트 GeoDataFrame으로 변환."""
    import warnings

    geom_types = set(gdf.geometry.geom_type.unique())
    if geom_types <= {"Point", "MultiPoint"}:
        return gdf
    pts = gdf.copy()
    with warnings.catch_warnings():
        # 지리좌표계 centroid 경고 억제 (예제 데이터는 평면좌표 취급)
        warnings.simplefilter("ignore", category=UserWarning)
        pts["geometry"] = pts.geometry.centroid
    return pts


def weights_summary(w: W) -> dict:
    """
    W 행렬 기초 통계.

    Returns:
        {
            'n': int,
            'mean_neighbors': float,
            'min_neighbors': int,
            'max_neighbors': int,
            'median_neighbors': float,
            'pct_islands': float,
            'sparsity': float,
            'symmetric': bool,
        }
    """
    cards = np.array(list(w.cardinalities.values()), dtype=float)
    n = w.n
    nonzero = float(cards.sum())
    try:
        symmetric = len(w.asymmetry()) == 0
    except Exception:
        symmetric = False

    return {
        "n": int(n),
        "mean_neighbors": float(cards.mean()) if n else 0.0,
        "min_neighbors": int(cards.min()) if n else 0,
        "max_neighbors": int(cards.max()) if n else 0,
        "median_neighbors": float(np.median(cards)) if n else 0.0,
        "pct_islands": float(len(w.islands) / n) if n else 0.0,
        "sparsity": float(1.0 - nonzero / (n * n)) if n else 1.0,
        "symmetric": bool(symmetric),
    }


def auto_distance_threshold(
    gdf: gpd.GeoDataFrame,
    percentile: float = 1.0,
) -> float:
    """
    최소 거리 임계값 자동 탐색.

    모든 점이 최소 1개 이웃을 갖도록 하는 최소 임계거리 반환.
    percentile=1.0 → max(min_nn_dist), 1.5 → 1.5배 안전마진

    Args:
        gdf: 포인트 GeoDataFrame (미터 좌표계)
        percentile: 최소 임계거리에 곱할 계수

    Returns:
        임계거리 (미터)
    """
    from scipy.spatial import cKDTree

    pts = _as_points(gdf)
    coords = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
    tree = cKDTree(coords)
    # 자기 자신 포함 2개 → 두 번째가 최근접 이웃
    dists, _ = tree.query(coords, k=2)
    nn_dist = dists[:, 1]
    return float(nn_dist.max() * percentile)


def weights_to_matrix(w: W) -> np.ndarray:
    """W 객체를 dense numpy 2D array로 변환 (작은 n만 권장)."""
    return w.full()[0]
