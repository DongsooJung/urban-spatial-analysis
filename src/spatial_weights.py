"""
Spatial Weight Matrix Construction
===================================

Provides factory functions for building spatial weight matrices W used
throughout spatial econometric models. Supports Queen/Rook contiguity,
K-nearest neighbors, and distance-band weights.

References
----------
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Springer.
Rey, S.J. & Anselin, L. (2010). PySAL: A Python Library of Spatial
Analytical Methods. In Handbook of Applied Spatial Analysis.
"""
from __future__ import annotations

from typing import Literal, Optional

import geopandas as gpd
from libpysal import weights


ContiguityRule = Literal["queen", "rook"]


def build_contiguity(
      gdf: gpd.GeoDataFrame,
      rule: ContiguityRule = "queen",
      row_standardize: bool = True,
) -> weights.W:
      """Build a contiguity-based spatial weight matrix.

          Parameters
              ----------
                  gdf : GeoDataFrame
                          Polygon geometries representing spatial units (e.g., census tracts).
                              rule : {"queen", "rook"}
                                      Queen shares edges *or* vertices; Rook shares edges only.
                                          row_standardize : bool
                                                  If True, transform to row-standardized form (sum of each row = 1),
                                                          which is standard for SLM/SEM estimation.
                                                              """
      if rule == "queen":
                w = weights.Queen.from_dataframe(gdf, use_index=True)
elif rule == "rook":
        w = weights.Rook.from_dataframe(gdf, use_index=True)
else:  # pragma: no cover
        raise ValueError(f"Unknown contiguity rule: {rule!r}")

    if row_standardize:
              w.transform = "r"
          return w


def build_knn(
      gdf: gpd.GeoDataFrame,
      k: int = 5,
      row_standardize: bool = True,
) -> weights.W:
      """K-nearest-neighbor spatial weights (centroid-based)."""
      centroids = gdf.geometry.centroid
      coords = list(zip(centroids.x, centroids.y))
      w = weights.KNN.from_array(coords, k=k)
      if row_standardize:
                w.transform = "r"
            return w


def build_distance_band(
      gdf: gpd.GeoDataFrame,
      threshold: Optional[float] = None,
      binary: bool = True,
      row_standardize: bool = True,
) -> weights.W:
      """Distance-band weights (all neighbors within `threshold` meters)."""
    centroids = gdf.geometry.centroid
    coords = list(zip(centroids.x, centroids.y))
    if threshold is None:
              threshold = weights.min_threshold_dist_from_shapefile(None, coords) \
                  if False else weights.util.get_points_array_from_shapefile  # fallback
        # In practice, compute min-threshold to guarantee no islands:
              threshold = weights.util.min_threshold_distance(coords)
          w = weights.DistanceBand.from_array(coords, threshold=threshold, binary=binary)
    if row_standardize:
              w.transform = "r"
          return w


def summarize(w: weights.W) -> dict:
      """Return basic diagnostics for a weight matrix."""
    return {
              "n": w.n,
              "mean_neighbors": w.mean_neighbors,
              "min_neighbors": w.min_neighbors,
              "max_neighbors": w.max_neighbors,
              "pct_nonzero": w.pct_nonzero,
              "islands": len(w.islands),
              "transform": w.transform,
    }
