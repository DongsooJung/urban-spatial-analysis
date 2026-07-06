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
from dataclasses import dataclass, field
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import W, lag_spatial

logger = logging.getLogger(__name__)

ModelMethod = Literal["OLS", "SLM", "SEM", "SDM", "GWR"]

CONSTANT_NAME = "CONSTANT"


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
    adj_r2: float | None = None
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    rho: float | None = None               # SLM/SDM
    lambda_: float | None = None           # SEM
    theta: pd.Series | None = None         # SDM WX 계수
    residuals: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    n: int = 0
    k: int = 0
    w: W | None = field(default=None, repr=False)   # 추정에 사용한 가중행렬
    raw_result: Any = field(default=None, repr=False)  # 원본 spreg/mgwr 객체

    def summary(self) -> str:
        """가독성 있는 요약 문자열."""
        lines = [
            "=" * 64,
            f"Spatial Model Results — {self.method}",
            "=" * 64,
            f"N = {self.n}    k = {self.k}    R² = {self.r2:.4f}"
            + (f"    adj-R² = {self.adj_r2:.4f}" if self.adj_r2 is not None else ""),
        ]
        stats = []
        if self.log_likelihood is not None:
            stats.append(f"log-L = {self.log_likelihood:.2f}")
        if self.aic is not None:
            stats.append(f"AIC = {self.aic:.2f}")
        if self.bic is not None:
            stats.append(f"BIC = {self.bic:.2f}")
        if stats:
            lines.append("    ".join(stats))
        if self.rho is not None:
            lines.append(f"rho (spatial lag)     = {self.rho:.4f}")
        if self.lambda_ is not None:
            lines.append(f"lambda (spatial error) = {self.lambda_:.4f}")
        lines.append("-" * 64)
        lines.append(f"{'variable':<16}{'coef':>12}{'std_err':>12}{'p-value':>12}")
        lines.append("-" * 64)
        for name in self.coefficients.index:
            coef = self.coefficients[name]
            se = self.std_errors.get(name, np.nan)
            p = self.p_values.get(name, np.nan)
            lines.append(f"{name:<16}{coef:>12.4f}{se:>12.4f}{p:>12.4f}")
        if self.theta is not None and len(self.theta):
            lines.append("-" * 64)
            lines.append("WX (Durbin) coefficients:")
            for name, val in self.theta.items():
                lines.append(f"{name:<16}{val:>12.4f}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """계수 테이블 DataFrame 반환."""
        return pd.DataFrame(
            {
                "coef": self.coefficients,
                "std_err": self.std_errors,
                "p_value": self.p_values,
            }
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
        w: W | None = None,
        method: ModelMethod = "OLS",
    ):
        self.df = df
        self.y_col = y
        self.X_cols = list(X)
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
        logger.info("Fitting %s (y=%s, X=%s)", self.method, self.y_col, self.X_cols)
        return fitters[self.method]()

    # ------------------------------------------------------------------
    @property
    def _y(self) -> np.ndarray:
        return self.df[self.y_col].to_numpy(dtype=float).reshape(-1, 1)

    @property
    def _X(self) -> np.ndarray:
        return self.df[self.X_cols].to_numpy(dtype=float)

    def _names(self, x_names: list[str] | None = None) -> list[str]:
        return [CONSTANT_NAME] + (x_names if x_names is not None else self.X_cols)

    # ------------------------------------------------------------------
    def _fit_ols(self) -> SpatialResults:
        """spreg.OLS — 공간진단(spat_diag) 포함 베이스라인."""
        from spreg import OLS

        res = OLS(
            self._y,
            self._X,
            w=self.w,
            spat_diag=self.w is not None,
            name_y=self.y_col,
            name_x=self.X_cols,
        )
        names = self._names()
        stats = np.asarray(res.t_stat, dtype=float)  # (k, 2) = (t, p)
        return SpatialResults(
            method="OLS",
            coefficients=pd.Series(res.betas.flatten(), index=names),
            std_errors=pd.Series(np.asarray(res.std_err).flatten(), index=names),
            p_values=pd.Series(stats[:, 1], index=names),
            r2=float(res.r2),
            adj_r2=float(res.ar2),
            log_likelihood=_maybe_float(getattr(res, "logll", None)),
            aic=_maybe_float(getattr(res, "aic", None)),
            bic=_maybe_float(getattr(res, "schwarz", None)),
            residuals=res.u.flatten(),
            fitted_values=res.predy.flatten(),
            n=int(res.n),
            k=int(res.k),
            w=self.w,
            raw_result=res,
        )

    def _fit_slm(self) -> SpatialResults:
        """spreg.ML_Lag (Maximum Likelihood)."""
        from spreg import ML_Lag

        res = ML_Lag(
            self._y, self._X, w=self.w, name_y=self.y_col, name_x=self.X_cols
        )
        return self._from_ml_lag(res, "SLM", self.X_cols)

    def _fit_sem(self) -> SpatialResults:
        """spreg.ML_Error (Maximum Likelihood)."""
        from spreg import ML_Error

        res = ML_Error(
            self._y, self._X, w=self.w, name_y=self.y_col, name_x=self.X_cols
        )
        names = self._names()
        betas = res.betas.flatten()
        stats = np.asarray(res.z_stat, dtype=float)
        std = np.asarray(res.std_err).flatten()
        # 마지막 원소 = lambda
        return SpatialResults(
            method="SEM",
            coefficients=pd.Series(betas[:-1], index=names),
            std_errors=pd.Series(std[: len(names)], index=names),
            p_values=pd.Series(stats[: len(names), 1], index=names),
            r2=float(res.pr2),
            log_likelihood=_maybe_float(getattr(res, "logll", None)),
            aic=_maybe_float(getattr(res, "aic", None)),
            bic=_maybe_float(getattr(res, "schwarz", None)),
            lambda_=float(res.lam),
            residuals=res.u.flatten(),
            fitted_values=res.predy.flatten(),
            n=int(res.n),
            k=int(res.k),
            w=self.w,
            raw_result=res,
        )

    def _fit_sdm(self) -> SpatialResults:
        """
        Spatial Durbin Model.

        구현: X를 공간시차(WX)로 확장한 뒤 ML_Lag로 추정.
        """
        from spreg import ML_Lag

        X = self._X
        WX = np.column_stack(
            [lag_spatial(self.w, X[:, j]) for j in range(X.shape[1])]
        )
        wx_names = [f"W_{c}" for c in self.X_cols]
        res = ML_Lag(
            self._y,
            np.hstack([X, WX]),
            w=self.w,
            name_y=self.y_col,
            name_x=self.X_cols + wx_names,
        )
        out = self._from_ml_lag(res, "SDM", self.X_cols + wx_names)
        # WX 계수를 theta로 분리
        theta = out.coefficients[wx_names]
        out.theta = theta
        out.coefficients = out.coefficients.drop(index=wx_names)
        out.std_errors = out.std_errors.drop(index=wx_names)
        out.p_values = out.p_values.drop(index=wx_names)
        return out

    def _from_ml_lag(
        self, res: Any, method: ModelMethod, x_names: list[str]
    ) -> SpatialResults:
        """ML_Lag 계열(spreg) 결과 → SpatialResults 변환."""
        names = self._names(x_names)
        betas = res.betas.flatten()          # [..., rho]
        stats = np.asarray(res.z_stat, dtype=float)
        std = np.asarray(res.std_err).flatten()
        return SpatialResults(
            method=method,
            coefficients=pd.Series(betas[:-1], index=names),
            std_errors=pd.Series(std[: len(names)], index=names),
            p_values=pd.Series(stats[: len(names), 1], index=names),
            r2=float(res.pr2),
            log_likelihood=_maybe_float(getattr(res, "logll", None)),
            aic=_maybe_float(getattr(res, "aic", None)),
            bic=_maybe_float(getattr(res, "schwarz", None)),
            rho=float(res.rho),
            residuals=res.u.flatten(),
            fitted_values=res.predy.flatten(),
            n=int(res.n),
            k=int(res.k),
            w=self.w,
            raw_result=res,
        )

    def _fit_gwr(self) -> SpatialResults:
        """mgwr.gwr.GWR — 국지적 회귀 (bandwidth는 AICc로 자동 탐색)."""
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW

        if not isinstance(self.df, gpd.GeoDataFrame):
            raise TypeError("GWR에는 geometry가 있는 GeoDataFrame이 필요합니다")

        cent = self.df.geometry.centroid
        coords = list(zip(cent.x, cent.y, strict=True))
        y, X = self._y, self._X

        bw = Sel_BW(coords, y, X).search()
        gwr_res = GWR(coords, y, X, bw).fit()
        # 시각화 등에서 변수명을 복원할 수 있도록 부착
        gwr_res.name_x = self._names()

        params = gwr_res.params  # (n, k+1) 국지 계수
        names = self._names()
        return SpatialResults(
            method="GWR",
            # 국지 계수의 평균/표준편차 (p-value는 국지적이므로 전역값 없음)
            coefficients=pd.Series(params.mean(axis=0), index=names),
            std_errors=pd.Series(params.std(axis=0), index=names),
            p_values=pd.Series(np.full(len(names), np.nan), index=names),
            r2=_maybe_float(getattr(gwr_res, "R2", None)) or float("nan"),
            aic=_maybe_float(getattr(gwr_res, "aic", None)),
            bic=_maybe_float(getattr(gwr_res, "bic", None)),
            residuals=np.asarray(gwr_res.resid_response).flatten(),
            fitted_values=np.asarray(gwr_res.predy).flatten(),
            n=int(len(y)),
            k=int(params.shape[1]),
            w=self.w,
            raw_result=gwr_res,
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


def _maybe_float(v: Any) -> float | None:
    return None if v is None else float(v)


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
    rows = [
        {
            "method": r.method,
            "n": r.n,
            "k": r.k,
            "log_L": r.log_likelihood,
            "aic": r.aic,
            "bic": r.bic,
            "r2": r.r2,
            "rho": r.rho,
            "lambda": r.lambda_,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def impacts_decomposition(
    result: SpatialResults, w: W | None = None
) -> pd.DataFrame:
    """
    SLM/SDM의 Direct / Indirect / Total Effect 분해.

    LeSage & Pace (2009): SLM의 β는 직접 해석 불가.
    실제 한계효과 행렬은 S_k(W) = (I - ρW)⁻¹ (I·β_k + W·θ_k) 이며,
    대각 평균 = Direct, 나머지 = Indirect, 행합 평균 = Total.

    Args:
        result: SLM/SDM 추정 결과 (rho 필요)
        w: 가중행렬. 생략하면 result.w 사용

    Returns:
        columns=['variable', 'direct', 'indirect', 'total']
    """
    if result.rho is None:
        raise ValueError("impacts 분해는 rho가 있는 SLM/SDM 결과에만 적용됩니다")
    w = w if w is not None else result.w
    if w is None:
        raise ValueError("가중행렬 w가 필요합니다 (result.w 또는 인자로 전달)")

    W_dense = w.full()[0]
    n = W_dense.shape[0]
    A_inv = np.linalg.inv(np.eye(n) - result.rho * W_dense)

    rows = []
    for var in result.coefficients.index:
        if var == CONSTANT_NAME:
            continue
        beta_k = result.coefficients[var]
        S_k = A_inv * beta_k
        if result.theta is not None:
            theta_k = result.theta.get(f"W_{var}")
            if theta_k is not None:
                S_k = A_inv @ (np.eye(n) * beta_k + W_dense * theta_k)
        direct = float(np.mean(np.diag(S_k)))
        total = float(S_k.sum() / n)
        rows.append(
            {
                "variable": var,
                "direct": direct,
                "indirect": total - direct,
                "total": total,
            }
        )
    return pd.DataFrame(rows)
