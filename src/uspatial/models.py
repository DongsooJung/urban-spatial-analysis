"""
공간계량 모형 5종 통합 인터페이스

OLS / SLM / SEM / SDM / GWR을 단일 클래스로 다룰 수 있도록 설계.

모형 수식 요약:
    OLS : y = Xβ + ε
    SLM : y = ρWy + Xβ + ε               (spatial lag of y)
    SEM : y = Xβ + u, u = λWu + ε         (spatial error)
    SDM : y = ρWy + Xβ + WXθ + ε          (Durbin = Lag + spatial X)
    GWR : yᵢ = β₀(uᵢ,vᵢ) + Σₖβₖ(uᵢ,vᵢ)xᵢₖ + εᵢ  (local regression)
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import W

logger = logging.getLogger(__name__)

ModelMethod = Literal["OLS", "SLM", "SEM", "SDM", "GWR"]


# ======================================================================
# 결과 컨테이너
# ======================================================================
@dataclass
class SpatialResults:
    """공간계량 모형 추정 결과 표준 컨테이너."""

    method: ModelMethod
    coefficients: pd.Series
    std_errors: pd.Series
    p_values: pd.Series
    r2: float
    adj_r2: Optional[float] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    rho: Optional[float] = None               # SLM/SDM
    lambda_: Optional[float] = None           # SEM
    theta: Optional[pd.Series] = None         # SDM WX 계수
    residuals: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    n: int = 0
    k: int = 0
    raw_result: Any = field(default=None, repr=False)  # 원본 spreg 객체

    def summary(self) -> str:
        """가독성 있는 요약 문자열."""
        raise NotImplementedError(
            "TODO: method + N + R² + AIC + coef 표를 멀티라인 문자열로 조립"
        )

    def to_frame(self) -> pd.DataFrame:
        """계수 테이블 DataFrame 반환."""
        raise NotImplementedError(
            "TODO: pd.DataFrame({'coef': self.coefficients, 'std_err': ..., 'p': ...})"
        )


# ======================================================================
# 메인 클래스
# ======================================================================
class SpatialModel:
    """
    공간계량 모형 통합 인터페이스.

    Example:
        >>> from uspatial import SpatialModel, build_weights
        >>> w = build_weights(columbus_gdf, method="queen")
        >>> model = SpatialModel(
        ...     df=columbus_gdf,
        ...     y="CRIME",
        ...     X=["INC", "HOVAL"],
        ...     w=w,
        ...     method="SEM",
        ... )
        >>> res = model.fit()
        >>> print(res.summary())
    """

    def __init__(
        self,
        df: gpd.GeoDataFrame | pd.DataFrame,
        y: str,
        X: list[str],
        w: Optional[W] = None,
        method: ModelMethod = "OLS",
    ):
        self.df = df
        self.y_col = y
        self.X_cols = X
        self.w = w
        self.method = method

        self._validate()

    # ------------------------------------------------------------------
    def fit(self) -> SpatialResults:
        """Estimate the specified spatial model."""
        fitters = {
            "OLS": self._fit_ols,
            "SLM": self._fit_slm,
            "SEM": self._fit_sem,
            "SDM": self._fit_sdm,
            "GWR": self._fit_gwr,
        }
        if self.method not in fitters:
            raise ValueError(f"Unknown method: {self.method}")
        return fitters[self.method]()

    # ------------------------------------------------------------------
    def _fit_ols(self) -> SpatialResults:
        """spreg.OLS (또는 statsmodels.OLS)."""
        raise NotImplementedError(
            "TODO: from spreg import OLS; "
            "res = OLS(y, X, name_y=..., name_x=..., w=self.w, spat_diag=True); "
            "SpatialResults 채우기"
        )

    def _fit_slm(self) -> SpatialResults:
        """spreg.ML_Lag (Maximum Likelihood) or GM_Lag (GMM)."""
        raise NotImplementedError("TODO: ML_Lag(y, X, w=self.w)")

    def _fit_sem(self) -> SpatialResults:
        """spreg.ML_Error or GM_Error."""
        raise NotImplementedError("TODO: ML_Error(y, X, w=self.w)")

    def _fit_sdm(self) -> SpatialResults:
        """
        Spatial Durbin Model.

        Implementation: lag transform X (WX) then ML_Lag with [X, WX].
        """
        raise NotImplementedError(
            "TODO: WX = self.w.sparse @ X_matrix; "
            "X_extended = np.hstack([X_matrix, WX]); "
            "ML_Lag(y, X_extended, w=self.w)"
        )

    def _fit_gwr(self) -> SpatialResults:
        """mgwr.gwr.GWR — 국지적 회귀."""
        raise NotImplementedError(
            "TODO: from mgwr.gwr import GWR; "
            "from mgwr.sel_bw import Sel_BW; "
            "bw = Sel_BW(coords, y, X).search(); "
            "gwr = GWR(coords, y, X, bw).fit()"
        )

    def _validate(self) -> None:
        """입력 검증."""
        if self.y_col not in self.df.columns:
            raise ValueError(f"y 컬럼 없음: {self.y_col}")
        missing = [c for c in self.X_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"X 컬럼 누락: {missing}")
        if self.method in ("SLM", "SEM", "SDM", "GWR") and self.w is None:
            raise ValueError(f"method={self.method}에는 w가 필요합니다")


# ======================================================================
# 모형 비교
# ======================================================================
def compare_models(results: list[SpatialResults]) -> pd.DataFrame:
    """
    여러 모형의 적합도를 단일 DataFrame으로 비교.

    Returns:
        columns=['method', 'n', 'k', 'log_L', 'aic', 'bic', 'r2', 'rho', 'lambda']

    Example:
        >>> df = compare_models([ols_res, slm_res, sem_res, sdm_res])
        >>> df.sort_values('aic').head()
    """
    raise NotImplementedError(
        "TODO: rows = [{'method': r.method, 'n': r.n, 'aic': r.aic, ...} for r in results]"
    )


def impacts_decomposition(slm_result: SpatialResults) -> pd.DataFrame:
    """
    SLM/SDM의 Direct / Indirect / Total Effect 분해.

    LeSage & Pace (2009): SLM의 β는 직접 해석 불가.
    실제 한계효과는 (I - ρW)⁻¹ X β 로 계산되며,
    대각 원소 = Direct, 비대각 합 = Indirect, 총합 = Total.

    Returns:
        columns=['variable', 'direct', 'indirect', 'total']
    """
    raise NotImplementedError(
        "TODO: spreg.diagnostics_sp 또는 수동 계산으로 A_inv = np.linalg.inv(I - rho*W)"
    )
