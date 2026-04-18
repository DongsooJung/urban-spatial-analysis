"""
Spatial Econometric Models
==========================

Thin wrappers around `spreg` estimators for consistent API:

- :func:`fit_ols`  — baseline OLS
- :func:`fit_slm`  — Spatial Lag Model via Maximum Likelihood
- :func:`fit_sem`  — Spatial Error Model via Maximum Likelihood
- :func:`fit_sdm`  — Spatial Durbin Model (lag of X + lag of y)

All estimators return a dictionary with parameter estimates, standard
errors, fit statistics (AIC, log-likelihood, pseudo-R2), and the raw
`spreg` result object under the key ``"raw"`` for advanced inspection.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from libpysal import weights
from spreg import OLS, ML_Lag, ML_Error, GM_Lag


def _pack(res, model_name: str, x_names: Sequence[str]) -> Dict:
      """Normalize spreg result objects into a plain dict."""
      return {
          "model": model_name,
          "x_names": list(x_names),
          "coef": dict(zip(x_names, np.asarray(res.betas).flatten().tolist())),
          "std_err": getattr(res, "std_err", None),
          "z_stat": getattr(res, "z_stat", None),
          "pr2": getattr(res, "pr2", None),
          "aic": getattr(res, "aic", None),
          "logll": getattr(res, "logll", None),
          "rho": getattr(res, "rho", None),
          "lam": getattr(res, "lam", None),
          "n": res.n,
          "k": res.k,
          "raw": res,
      }


def fit_ols(df: pd.DataFrame, y: str, x: Sequence[str]) -> Dict:
      """OLS baseline — ignores spatial structure."""
      y_arr = df[[y]].values
      x_arr = df[list(x)].values
      res = OLS(y_arr, x_arr, name_y=y, name_x=list(x))
      return _pack(res, "OLS", x)


def fit_slm(df: pd.DataFrame, y: str, x: Sequence[str], w: weights.W) -> Dict:
      """Spatial Lag Model (SLM) via ML.

          y = ρ * W*y + Xβ + ε
              """
      y_arr = df[[y]].values
      x_arr = df[list(x)].values
      res = ML_Lag(y_arr, x_arr, w=w, name_y=y, name_x=list(x))
      return _pack(res, "SLM", x)


def fit_sem(df: pd.DataFrame, y: str, x: Sequence[str], w: weights.W) -> Dict:
      """Spatial Error Model (SEM) via ML.

          y = Xβ + u,  u = λ * W*u + ε
              """
      y_arr = df[[y]].values
      x_arr = df[list(x)].values
      res = ML_Error(y_arr, x_arr, w=w, name_y=y, name_x=list(x))
      return _pack(res, "SEM", x)


def fit_sdm(df: pd.DataFrame, y: str, x: Sequence[str], w: weights.W) -> Dict:
      """Spatial Durbin Model — OLS with lagged X plus lagged y (GM).

          Implemented here via GM_Lag with ``w_lags=1`` on covariates.
              """
      y_arr = df[[y]].values
      x_arr = df[list(x)].values
      res = GM_Lag(y_arr, x_arr, w=w, w_lags=1, name_y=y, name_x=list(x))
      return _pack(res, "SDM", x)


def compare(results: Sequence[Dict]) -> pd.DataFrame:
      """Return a side-by-side comparison table of fit statistics."""
      rows = []
      for r in results:
                rows.append({
                              "model": r["model"],
                              "n": r["n"],
                              "k": r["k"],
                              "pseudo_R2": r.get("pr2"),
                              "AIC": r.get("aic"),
                              "logLik": r.get("logll"),
                              "rho": r.get("rho"),
                              "lambda": r.get("lam"),
                })
            return pd.DataFrame(rows).set_index("model")
