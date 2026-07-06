"""
공간 자기상관 진단 및 모형 선택 검정

3단계 의사결정 지원:
    1. 전역 자기상관 존재 여부 (Moran's I)
    2. 국지적 패턴 식별 (LISA)
    3. SLM vs SEM 중 어느 모형인가 (LM / Robust LM)

참고: Anselin, L. (1995). Local indicators of spatial association—LISA.
      Geographical Analysis, 27(2), 93-115.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from libpysal.weights import W

logger = logging.getLogger(__name__)

QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


# ======================================================================
# 전역 Moran's I
# ======================================================================
@dataclass
class MoranResult:
    """전역 Moran's I 결과."""

    statistic: float
    expected: float
    variance: float
    z_score: float
    p_value: float
    p_sim: float               # 순열검정 p-value
    permutations: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_sim < alpha

    def interpret(self) -> str:
        """결과 해석 문자열."""
        if self.statistic > self.expected:
            direction = "양(+)의 공간 자기상관 (유사값 군집)"
        elif self.statistic < self.expected:
            direction = "음(-)의 공간 자기상관 (체커보드 패턴)"
        else:
            direction = "공간적 무작위"
        sig = (
            f"통계적으로 유의함 (p_sim={self.p_sim:.4f} < 0.05)"
            if self.is_significant()
            else f"통계적으로 유의하지 않음 (p_sim={self.p_sim:.4f} ≥ 0.05)"
        )
        return (
            f"Moran's I = {self.statistic:.4f} (E[I] = {self.expected:.4f}, "
            f"z = {self.z_score:.3f}) → {direction}, {sig}"
        )


def morans_i(
    values: np.ndarray,
    w: W,
    permutations: int = 999,
    transformation: str = "R",
) -> MoranResult:
    """
    전역 Moran's I 계산.

    I = (n / Σᵢⱼwᵢⱼ) × (Σᵢⱼwᵢⱼ(xᵢ - x̄)(xⱼ - x̄)) / Σᵢ(xᵢ - x̄)²

    Args:
        values: 관심 변수 1D array
        w: 공간가중행렬
        permutations: 순열검정 반복 수 (Monte Carlo)
        transformation: 'R' (row-standardize), 'B' (binary), 'V' (var-stabilize)

    Returns:
        MoranResult
    """
    from esda.moran import Moran

    m = Moran(
        np.asarray(values, dtype=float),
        w,
        transformation=transformation,
        permutations=permutations,
    )
    return MoranResult(
        statistic=float(m.I),
        expected=float(m.EI),
        variance=float(m.VI_norm),
        z_score=float(m.z_norm),
        p_value=float(m.p_norm),
        p_sim=float(m.p_sim),
        permutations=permutations,
    )


# ======================================================================
# 국지적 LISA (Local Moran's I)
# ======================================================================
@dataclass
class LisaResult:
    """LISA 분석 결과."""

    local_i: np.ndarray           # 각 관측치의 Iᵢ
    z_scores: np.ndarray
    p_values: np.ndarray
    quadrants: np.ndarray         # 1=HH, 2=LH, 3=LL, 4=HL
    significant: np.ndarray       # Boolean mask
    cluster_labels: pd.Series     # 'HH' | 'LL' | 'HL' | 'LH' | 'NS'


def local_moran(
    values: np.ndarray,
    w: W,
    significance_level: float = 0.05,
    permutations: int = 999,
) -> LisaResult:
    """
    LISA (Local Moran's I) 계산.

    Quadrant 해석:
        HH (1): 본인·이웃 모두 high → 핫스팟
        LH (2): 본인 low, 이웃 high → 음의 아웃라이어
        LL (3): 본인·이웃 모두 low → 콜드스팟
        HL (4): 본인 high, 이웃 low → 양의 아웃라이어

    Returns:
        LisaResult
    """
    from esda.moran import Moran_Local

    lm = Moran_Local(
        np.asarray(values, dtype=float), w, permutations=permutations
    )
    significant = lm.p_sim < significance_level
    labels = pd.Series(
        np.where(
            significant,
            pd.Series(lm.q).map(QUADRANT_LABELS).to_numpy(),
            "NS",
        ),
        name="lisa_cluster",
    )
    return LisaResult(
        local_i=np.asarray(lm.Is),
        z_scores=np.asarray(lm.z_sim),
        p_values=np.asarray(lm.p_sim),
        quadrants=np.asarray(lm.q),
        significant=np.asarray(significant),
        cluster_labels=labels,
    )


# ======================================================================
# LM 진단 (모형 선택)
# ======================================================================
@dataclass
class LMTestResult:
    """
    Anselin-Florax-Rey LM 검정 결과.

    Decision rule (Anselin & Florax 1995):
        1. LM-Lag, LM-Error 모두 유의하지 않음 → OLS
        2. 하나만 유의 → 해당 모형
        3. 둘 다 유의 → Robust LM 중 더 유의한 쪽 (둘 다 강하게 유의하면 SDM)
    """

    lm_lag: float
    lm_lag_p: float
    lm_error: float
    lm_error_p: float
    robust_lm_lag: float
    robust_lm_lag_p: float
    robust_lm_error: float
    robust_lm_error_p: float
    recommendation: Literal["OLS", "SLM", "SEM", "SDM"]


def lm_diagnostics(
    ols_result: Any, w: W, alpha: float = 0.05
) -> LMTestResult:
    """
    OLS 잔차 기반 Lagrange Multiplier 진단.

    Args:
        ols_result: spreg.OLS 결과 객체
        w: 공간가중행렬
        alpha: 유의수준 (의사결정 룰에 사용)

    Returns:
        LMTestResult with recommendation
    """
    from spreg.diagnostics_sp import LMtests

    lm = LMtests(ols_result, w, tests=["all"])
    lm_lag, lm_lag_p = (float(v) for v in lm.lml)
    lm_err, lm_err_p = (float(v) for v in lm.lme)
    rlm_lag, rlm_lag_p = (float(v) for v in lm.rlml)
    rlm_err, rlm_err_p = (float(v) for v in lm.rlme)

    lag_sig, err_sig = lm_lag_p < alpha, lm_err_p < alpha
    if not lag_sig and not err_sig:
        rec: Literal["OLS", "SLM", "SEM", "SDM"] = "OLS"
    elif lag_sig and not err_sig:
        rec = "SLM"
    elif err_sig and not lag_sig:
        rec = "SEM"
    else:
        # 둘 다 유의 → Robust LM으로 판별
        r_lag_sig, r_err_sig = rlm_lag_p < alpha, rlm_err_p < alpha
        if r_lag_sig and r_err_sig:
            rec = "SDM"  # 두 효과가 모두 강함 → 일반형 (LeSage & Pace 2009)
        elif r_lag_sig:
            rec = "SLM"
        elif r_err_sig:
            rec = "SEM"
        else:
            rec = "SLM" if rlm_lag_p < rlm_err_p else "SEM"

    return LMTestResult(
        lm_lag=lm_lag,
        lm_lag_p=lm_lag_p,
        lm_error=lm_err,
        lm_error_p=lm_err_p,
        robust_lm_lag=rlm_lag,
        robust_lm_lag_p=rlm_lag_p,
        robust_lm_error=rlm_err,
        robust_lm_error_p=rlm_err_p,
        recommendation=rec,
    )


# ======================================================================
# 추가 진단
# ======================================================================
def variance_inflation_factor(X: pd.DataFrame) -> pd.Series:
    """다중공선성 진단 (VIF). 5 이상이면 경고, 10 이상은 심각."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor as _vif,
    )

    Xc = sm.add_constant(X.astype(float), has_constant="add")
    vals = {
        col: float(_vif(Xc.values, i))
        for i, col in enumerate(Xc.columns)
        if col != "const"
    }
    return pd.Series(vals, name="VIF")


def spatial_chow_test(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    w: W | None = None,
) -> dict:
    """
    구조 안정성 검정 (Chow test) — region별 계수 동일성 귀무가설.

    그룹별 OLS와 풀링 OLS의 SSR을 비교하는 고전적 Chow F-검정.
    (w 인자는 API 호환용이며 현재 검정에는 사용되지 않음 —
    잔차의 공간상관이 강하면 p-value가 낙관적일 수 있음에 유의.)

    Returns:
        {'f_stat', 'p_value', 'df1', 'df2', 'n_groups', 'reject_h0'}
    """
    from scipy import stats

    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xc = np.column_stack([np.ones(len(y)), X])
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("groups에는 2개 이상의 그룹이 필요합니다")

    def ssr(y_, X_):
        beta, *_ = np.linalg.lstsq(X_, y_, rcond=None)
        resid = y_ - X_ @ beta
        return float(resid @ resid)

    ssr_pooled = ssr(y, Xc)
    ssr_groups = sum(ssr(y[groups == g], Xc[groups == g]) for g in uniq)

    k = Xc.shape[1]
    G = len(uniq)
    df1 = (G - 1) * k
    df2 = len(y) - G * k
    if df2 <= 0:
        raise ValueError("관측치 수가 부족합니다 (n ≤ G·k)")
    f_stat = ((ssr_pooled - ssr_groups) / df1) / (ssr_groups / df2)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "n_groups": G,
        "reject_h0": p_value < 0.05,
    }
