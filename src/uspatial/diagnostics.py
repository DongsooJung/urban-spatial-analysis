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
from typing import Literal

import numpy as np
import pandas as pd
from libpysal.weights import W

logger = logging.getLogger(__name__)


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
        raise NotImplementedError(
            "TODO: I > E[I] 이면 positive SA, 음수면 negative, 유의성도 함께"
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
    raise NotImplementedError(
        "TODO: from esda.moran import Moran; "
        "m = Moran(values, w, permutations=permutations, transformation=transformation); "
        "MoranResult(statistic=m.I, ...)"
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
    raise NotImplementedError(
        "TODO: from esda.moran import Moran_Local; "
        "lm = Moran_Local(values, w, permutations=permutations); "
        "significance = lm.p_sim < significance_level"
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
        3. 둘 다 유의 → Robust LM 중 더 유의한 쪽
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


def lm_diagnostics(ols_result: Any, w: W) -> LMTestResult:  # noqa: F821
    """
    OLS 잔차 기반 Lagrange Multiplier 진단.

    spreg.OLS는 이미 lm_lag, lm_error 등을 계산하므로 래핑만 필요.

    Args:
        ols_result: spreg.OLS 결과 객체
        w: 공간가중행렬

    Returns:
        LMTestResult with recommendation
    """
    raise NotImplementedError(
        "TODO: ols_result.lm_lag, ols_result.rlm_lag 등 속성 추출 후 "
        "의사결정 룰로 recommendation 계산"
    )


# ======================================================================
# 추가 진단
# ======================================================================
def variance_inflation_factor(X: pd.DataFrame) -> pd.Series:
    """다중공선성 진단 (VIF). 5 이상이면 경고, 10 이상은 심각."""
    raise NotImplementedError(
        "TODO: from statsmodels.stats.outliers_influence import variance_inflation_factor"
    )


def spatial_chow_test(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    w: W,
) -> dict:
    """공간적 구조 안정성 검정 (Chow test, region별 계수 차이)."""
    raise NotImplementedError(
        "TODO: 그룹별 SSR 비교, F-통계량 계산"
    )
