"""진단 테스트 — Moran's I / LISA / LM 검정 / VIF / Chow.

Columbus CRIME은 문헌상 Moran's I ≈ 0.5 (p < 0.01)로 잘 알려진 벤치마크.
"""
import numpy as np
import pandas as pd
import pytest

from uspatial.diagnostics import (
    lm_diagnostics,
    local_moran,
    morans_i,
    spatial_chow_test,
    variance_inflation_factor,
)
from uspatial.models import SpatialModel


@pytest.fixture(scope="module")
def crime(columbus_gdf):
    return columbus_gdf["CRIME"].to_numpy()


class TestMoransI:
    def test_columbus_crime_positive_sa(self, crime, queen_w):
        mi = morans_i(crime, queen_w, permutations=999)
        assert 0.4 < mi.statistic < 0.6  # 문헌값 ≈ 0.5
        assert mi.is_significant()
        assert mi.expected == pytest.approx(-1 / 48)

    def test_interpret_mentions_direction(self, crime, queen_w):
        mi = morans_i(crime, queen_w, permutations=99)
        assert "양(+)" in mi.interpret()

    def test_random_data_not_significant(self, queen_w):
        rng = np.random.default_rng(42)
        mi = morans_i(rng.normal(size=queen_w.n), queen_w, permutations=999)
        assert abs(mi.statistic) < 0.2


class TestLisa:
    def test_shapes_and_labels(self, crime, queen_w):
        lisa = local_moran(crime, queen_w)
        n = queen_w.n
        assert lisa.local_i.shape == (n,)
        assert lisa.quadrants.shape == (n,)
        assert set(lisa.cluster_labels.unique()) <= {"HH", "LL", "HL", "LH", "NS"}

    def test_significant_matches_labels(self, crime, queen_w):
        lisa = local_moran(crime, queen_w, significance_level=0.05)
        assert ((lisa.cluster_labels == "NS") == ~lisa.significant).all()

    def test_columbus_has_clusters(self, crime, queen_w):
        lisa = local_moran(crime, queen_w)
        assert lisa.significant.sum() > 0


class TestLMDiagnostics:
    def test_columbus_recommends_spatial_model(self, columbus_gdf, queen_w):
        ols = SpatialModel(
            columbus_gdf, "CRIME", ["INC", "HOVAL"], w=queen_w, method="OLS"
        ).fit()
        lm = lm_diagnostics(ols.raw_result, queen_w)
        # Columbus에서 LM-Lag/LM-Error 모두 유의 → OLS는 아님
        assert lm.recommendation in ("SLM", "SEM", "SDM")
        assert lm.lm_lag > 0 and lm.lm_error > 0
        assert 0 <= lm.lm_lag_p <= 1


class TestVif:
    def test_vif_positive(self, columbus_gdf):
        vif = variance_inflation_factor(columbus_gdf[["INC", "HOVAL"]])
        assert set(vif.index) == {"INC", "HOVAL"}
        assert (vif >= 1.0).all()

    def test_perfect_collinearity_detected(self):
        x = np.linspace(0, 10, 50)
        X = pd.DataFrame({"a": x, "b": 2 * x + np.random.default_rng(0).normal(0, 1e-6, 50)})
        vif = variance_inflation_factor(X)
        assert (vif > 100).all()


class TestChow:
    def test_stable_coefficients_not_rejected(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 2))
        y = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(size=200)
        groups = np.repeat([0, 1], 100)
        out = spatial_chow_test(y, X, groups)
        assert out["p_value"] > 0.05
        assert out["n_groups"] == 2

    def test_structural_break_rejected(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 1))
        y = np.concatenate(
            [1 + 5 * X[:100, 0], 1 - 5 * X[100:, 0]]
        ) + rng.normal(size=200) * 0.1
        out = spatial_chow_test(y, X, np.repeat([0, 1], 100))
        assert out["reject_h0"]

    def test_single_group_raises(self):
        with pytest.raises(ValueError):
            spatial_chow_test(np.zeros(10), np.zeros((10, 1)), np.zeros(10))
