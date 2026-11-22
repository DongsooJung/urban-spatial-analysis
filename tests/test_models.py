"""공간계량 모형 테스트 — Columbus CRIME ~ INC + HOVAL.

Anselin (1988) ch.12의 고전 예제이므로 기대 부호/크기가 잘 알려져 있다:
    - INC, HOVAL 계수는 음수 (소득·주택가치 높을수록 범죄율 낮음)
    - SLM rho > 0, SEM lambda > 0 (양의 공간 자기상관)
"""
import numpy as np
import pytest

from uspatial.models import SpatialModel, compare_models, impacts_decomposition

X_VARS = ["INC", "HOVAL"]


@pytest.fixture(scope="module")
def ols_res(columbus_gdf, queen_w):
    return SpatialModel(columbus_gdf, "CRIME", X_VARS, w=queen_w, method="OLS").fit()


@pytest.fixture(scope="module")
def slm_res(columbus_gdf, queen_w):
    return SpatialModel(columbus_gdf, "CRIME", X_VARS, w=queen_w, method="SLM").fit()


@pytest.fixture(scope="module")
def sem_res(columbus_gdf, queen_w):
    return SpatialModel(columbus_gdf, "CRIME", X_VARS, w=queen_w, method="SEM").fit()


@pytest.fixture(scope="module")
def sdm_res(columbus_gdf, queen_w):
    return SpatialModel(columbus_gdf, "CRIME", X_VARS, w=queen_w, method="SDM").fit()


class TestValidation:
    def test_missing_y_raises(self, columbus_gdf):
        with pytest.raises(ValueError, match="y 컬럼"):
            SpatialModel(columbus_gdf, "NOPE", X_VARS)

    def test_missing_x_raises(self, columbus_gdf):
        with pytest.raises(ValueError, match="X 컬럼"):
            SpatialModel(columbus_gdf, "CRIME", ["INC", "NOPE"])

    def test_spatial_method_requires_w(self, columbus_gdf):
        with pytest.raises(ValueError, match="w"):
            SpatialModel(columbus_gdf, "CRIME", X_VARS, w=None, method="SLM")


class TestOLS:
    def test_shapes(self, ols_res):
        assert ols_res.n == 49
        assert list(ols_res.coefficients.index) == ["CONSTANT", "INC", "HOVAL"]

    def test_known_signs(self, ols_res):
        assert ols_res.coefficients["INC"] < 0
        assert ols_res.coefficients["HOVAL"] < 0

    def test_r2_range(self, ols_res):
        assert 0.4 < ols_res.r2 < 0.7  # 문헌값 ≈ 0.55

    def test_residuals_and_fitted(self, ols_res):
        assert ols_res.residuals.shape == (49,)
        y = ols_res.fitted_values + ols_res.residuals
        assert np.isfinite(y).all()

    def test_summary_and_frame(self, ols_res):
        text = ols_res.summary()
        assert "OLS" in text and "INC" in text
        frame = ols_res.to_frame()
        assert set(frame.columns) == {"coef", "std_err", "p_value"}


class TestSpatialModels:
    def test_slm_rho_positive(self, slm_res):
        assert slm_res.rho is not None
        assert 0 < slm_res.rho < 1

    def test_sem_lambda_positive(self, sem_res):
        assert sem_res.lambda_ is not None
        assert 0 < sem_res.lambda_ < 1

    def test_sdm_has_theta(self, sdm_res):
        assert sdm_res.theta is not None
        assert list(sdm_res.theta.index) == ["W_INC", "W_HOVAL"]
        # 본계수에서는 WX가 분리되어야 함
        assert "W_INC" not in sdm_res.coefficients.index

    def test_spatial_fit_improves_loglik(self, ols_res, slm_res, sem_res):
        assert slm_res.log_likelihood > ols_res.log_likelihood
        assert sem_res.log_likelihood > ols_res.log_likelihood

    def test_gwr_runs(self, columbus_gdf, queen_w):
        res = SpatialModel(
            columbus_gdf, "CRIME", X_VARS, w=queen_w, method="GWR"
        ).fit()
        assert res.raw_result.params.shape == (49, 3)
        assert res.residuals.shape == (49,)


class TestCompareAndImpacts:
    def test_compare_models(self, ols_res, slm_res, sem_res):
        df = compare_models([ols_res, slm_res, sem_res])
        assert list(df["method"]) == ["OLS", "SLM", "SEM"]
        assert {"aic", "bic", "log_L", "r2"} <= set(df.columns)

    def test_impacts_slm(self, slm_res):
        imp = impacts_decomposition(slm_res)
        assert list(imp.columns) == ["variable", "direct", "indirect", "total"]
        assert set(imp["variable"]) == {"INC", "HOVAL"}
        # rho > 0이면 direct/indirect 부호 동일, total = direct + indirect
        row = imp.set_index("variable").loc["INC"]
        assert row["total"] == pytest.approx(row["direct"] + row["indirect"])
        assert np.sign(row["direct"]) == np.sign(row["indirect"])

    def test_impacts_requires_rho(self, ols_res):
        with pytest.raises(ValueError, match="rho"):
            impacts_decomposition(ols_res)

    def test_impacts_sdm(self, sdm_res):
        imp = impacts_decomposition(sdm_res).set_index("variable")
        assert np.isfinite(imp.to_numpy()).all()
