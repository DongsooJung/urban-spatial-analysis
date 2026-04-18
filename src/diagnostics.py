"""
Spatial Autocorrelation Diagnostics
===================================

Utilities for testing spatial structure *before* model specification and
for evaluating residuals *after* estimation.

- :func:`morans_i` — global Moran's I statistic + pseudo p-value
- :func:`lisa`     — Local Indicators of Spatial Association
- :func:`lm_tests` — Lagrange Multiplier tests (LM-lag, LM-error,
                     robust variants) from OLS residuals
                     """
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from libpysal import weights
from esda.moran import Moran, Moran_Local
from spreg import OLS


def morans_i(y: np.ndarray, w: weights.W, permutations: int = 999) -> Dict:
      """Global Moran's I with pseudo p-value via permutation inference."""
      mi = Moran(y, w, permutations=permutations)
      return {
          "I": float(mi.I),
          "EI": float(mi.EI),
          "z_sim": float(mi.z_sim),
          "p_sim": float(mi.p_sim),
          "p_norm": float(mi.p_norm),
      }


def lisa(y: np.ndarray, w: weights.W, permutations: int = 999) -> pd.DataFrame:
      """Local Moran's I for hotspot/coldspot detection.

          Quadrants: 1=HH, 2=LH, 3=LL, 4=HL (LeSage & Pace 2009 convention).
              """
      lm = Moran_Local(y, w, permutations=permutations)
      return pd.DataFrame({
          "Is": lm.Is,
          "q": lm.q,
          "p_sim": lm.p_sim,
          "significant_05": lm.p_sim < 0.05,
      })


def lm_tests(df: pd.DataFrame, y: str, x: Sequence[str], w: weights.W) -> Dict:
      """Run OLS and extract LM-lag / LM-error / robust variants.

          Interpretation (Anselin 1988):
              - If LM-lag only is significant → use SLM
                  - If LM-error only is significant → use SEM
                      - If both are significant → compare *robust* variants;
                            follow the robust test that remains significant.
                                """
      y_arr = df[[y]].values
      x_arr = df[list(x)].values
      ols = OLS(y_arr, x_arr, w=w, spat_diag=True, moran=True,
                name_y=y, name_x=list(x))

    def _pair(name):
              stat = getattr(ols, name, None)
              if stat is None:
                            return None
                        # spreg returns (statistic, p-value) tuples
                        return {"stat": float(stat[0]), "p": float(stat[1])}

    return {
              "LM_lag": _pair("lm_lag"),
              "LM_error": _pair("lm_error"),
              "Robust_LM_lag": _pair("rlm_lag"),
              "Robust_LM_error": _pair("rlm_error"),
              "SARMA": _pair("lm_sarma"),
              "Moran_residual": _pair("moran_res") if hasattr(ols, "moran_res") else None,
    }
