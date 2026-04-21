"""공간가중행렬 테스트 스켈레톤."""
import pytest
import numpy as np

# from uspatial.weights import build_weights, weights_summary, auto_distance_threshold
# from uspatial.data import load_example


@pytest.fixture
def columbus_gdf():
    """Columbus 내장 예제 데이터."""
    pytest.skip("load_example 미구현")
    # return load_example("columbus")


class TestBuildWeights:
    def test_queen_for_polygon(self, columbus_gdf):
        """Queen 인접성은 Columbus 49개 구역에서 최소 1개 이상 이웃 생성."""
        pytest.skip("미구현")
        # w = build_weights(columbus_gdf, method="queen")
        # assert w.n == 49
        # assert all(len(w.neighbors[i]) >= 1 for i in w.id_order)

    def test_knn_returns_fixed_k(self, columbus_gdf):
        """KNN(k=5)이면 모든 점이 정확히 5개 이웃."""
        pytest.skip("미구현")

    def test_row_standardization(self, columbus_gdf):
        """row_standardize=True면 각 행 합이 1.0."""
        pytest.skip("미구현")

    def test_invalid_method_raises(self, columbus_gdf):
        pytest.skip("미구현")
        # with pytest.raises(ValueError):
        #     build_weights(columbus_gdf, method="invalid")


class TestWeightsSummary:
    def test_summary_keys(self):
        pytest.skip("미구현")
        # keys = {"n", "mean_neighbors", "min_neighbors", "max_neighbors",
        #         "pct_islands", "sparsity", "symmetric"}
        # assert set(weights_summary(w)) >= keys
