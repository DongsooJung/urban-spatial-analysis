"""
Urban Spatial Econometric Analysis

공간계량경제학(Spatial Econometrics) 표준 모형 5종을 통합 인터페이스로 제공한다.

구현 모형:
    - OLS  : 공간 효과 무시 (베이스라인)
    - SLM  : Spatial Lag Model (y의 공간 자기상관)
    - SEM  : Spatial Error Model (error의 공간 자기상관)
    - SDM  : Spatial Durbin Model (Lag + X의 공간 spillover)
    - GWR  : Geographically Weighted Regression (국지적 회귀)

이론적 배경:
    - Anselin (1988): Spatial Econometrics - Methods and Models
    - LeSage & Pace (2009): Introduction to Spatial Econometrics
    - Fotheringham, Brunsdon & Charlton (2002): Geographically Weighted Regression

주 의존 패키지: PySAL (libpysal, spreg, esda, mgwr, splot)

사용 예:
    >>> from uspatial import SpatialModel, build_weights
    >>> w = build_weights(gdf, method="queen")
    >>> model = SpatialModel(df=gdf, y="log_value", X=["x1", "x2"], w=w, method="SEM")
    >>> results = model.fit()
    >>> print(results.summary())
"""

__version__ = "0.1.0"
__author__ = "Dongsoo Jung"
__email__ = "jds068888@gmail.com"

from uspatial.models import SpatialModel  # noqa: F401
from uspatial.weights import build_weights  # noqa: F401
from uspatial.diagnostics import morans_i, lm_diagnostics  # noqa: F401

__all__ = [
    "SpatialModel",
    "build_weights",
    "morans_i",
    "lm_diagnostics",
]
