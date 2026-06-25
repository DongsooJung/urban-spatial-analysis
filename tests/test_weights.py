"""공간가중행렬 테스트 — build_weights / weights_summary / auto_distance_threshold.

공간계량 스택(geopandas/libpysal)이 없으면 모듈 전체를 skip.
Columbus 예제는 최초 1회 libpysal이 다운로드한다(네트워크 필요).
"""
import sys
from pathlib import Path

import pytest
import numpy as np

gpd = pytest.importorskip("geopandas")
pytest.importorskip("libpysal")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uspatial.weights import (
    build_weights,
    weights_summary,
    auto_distance_threshold,
    weights_to_matrix,
)
from uspatial.data import load_example, describe_dataset


@pytest.fixture(scope="module")
def columbus_gdf():
    """Columbus 내장 예제 데이터 (49 폴리곤)."""
    try:
        return load_example("columbus")
    except Exception as e:  # 네트워크 불가 등
        pytest.skip(f"Columbus 로드 실패: {e}")


class TestLoadExample:
    def test_columbus_shape(self, columbus_gdf):
        assert columbus_gdf.shape[0] == 49

    def test_columbus_is_polygon(self, columbus_gdf):
        assert set(columbus_gdf.geom_type.unique()) <= {"Polygon", "MultiPolygon"}

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError):
            load_example("nonexistent")

    def test_describe_keys(self):
        d = describe_dataset("columbus")
        assert d["n_obs"] == 49
        assert set(d.keys()) >= {"name", "n_obs", "geometry_type", "citation"}


class TestBuildWeights:
    def test_queen_for_polygon(self, columbus_gdf):
        """Queen 인접성은 Columbus 49개 구역에서 최소 1개 이상 이웃 생성."""
        w = build_weights(columbus_gdf, method="queen", row_standardize=False)
        assert w.n == 49
        assert all(len(w.neighbors[i]) >= 1 for i in w.id_order)

    def test_rook_subset_of_queen(self, columbus_gdf):
        """Rook 이웃 수 ≤ Queen 이웃 수."""
        wq = build_weights(columbus_gdf, method="queen", row_standardize=False)
        wr = build_weights(columbus_gdf, method="rook", row_standardize=False)
        assert sum(wr.cardinalities.values()) <= sum(wq.cardinalities.values())

    def test_knn_returns_fixed_k(self, columbus_gdf):
        """KNN(k=5)이면 모든 점이 정확히 5개 이웃."""
        w = build_weights(columbus_gdf, method="knn", k=5, row_standardize=False)
        assert all(len(w.neighbors[i]) == 5 for i in w.id_order)

    def test_row_standardization(self, columbus_gdf):
        """row_standardize=True면 각 행 합이 1.0."""
        w = build_weights(columbus_gdf, method="queen", row_standardize=True)
        for i in w.id_order:
            if w.cardinalities[i] > 0:
                assert sum(w.weights[i]) == pytest.approx(1.0)

    def test_distance_auto_threshold_no_islands(self, columbus_gdf):
        """threshold=None 자동탐색이면 고립점이 없어야 함."""
        w = build_weights(columbus_gdf, method="distance", row_standardize=False)
        assert len(w.islands) == 0

    def test_invalid_method_raises(self, columbus_gdf):
        with pytest.raises(ValueError):
            build_weights(columbus_gdf, method="invalid")

    def test_empty_gdf_raises(self):
        empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        with pytest.raises(ValueError):
            build_weights(empty, method="queen")


class TestWeightsSummary:
    def test_summary_keys(self, columbus_gdf):
        w = build_weights(columbus_gdf, method="queen")
        keys = {"n", "mean_neighbors", "min_neighbors", "max_neighbors",
                "pct_islands", "sparsity", "symmetric"}
        assert set(weights_summary(w)) >= keys

    def test_summary_values(self, columbus_gdf):
        w = build_weights(columbus_gdf, method="knn", k=5, row_standardize=False)
        s = weights_summary(w)
        assert s["n"] == 49
        assert s["min_neighbors"] == 5
        assert s["max_neighbors"] == 5
        assert s["mean_neighbors"] == pytest.approx(5.0)
        assert s["pct_islands"] == 0.0

    def test_knn_is_asymmetric(self, columbus_gdf):
        w = build_weights(columbus_gdf, method="knn", k=5, row_standardize=False)
        assert weights_summary(w)["symmetric"] is False


class TestAutoThreshold:
    def test_threshold_positive(self, columbus_gdf):
        thr = auto_distance_threshold(columbus_gdf)
        assert thr > 0

    def test_percentile_scales(self, columbus_gdf):
        t1 = auto_distance_threshold(columbus_gdf, percentile=1.0)
        t2 = auto_distance_threshold(columbus_gdf, percentile=2.0)
        assert t2 == pytest.approx(2.0 * t1)


class TestWeightsToMatrix:
    def test_matrix_shape(self, columbus_gdf):
        w = build_weights(columbus_gdf, method="queen", row_standardize=False)
        mat = weights_to_matrix(w)
        assert mat.shape == (49, 49)

    def test_matrix_binary_symmetric(self, columbus_gdf):
        w = build_weights(columbus_gdf, method="queen", row_standardize=False)
        mat = weights_to_matrix(w)
        # Queen 인접은 대칭
        assert np.allclose(mat, mat.T)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
