"""
예제 데이터 로더

PySAL 내장 예제 데이터셋을 즉시 로드하여 튜토리얼에 활용.
외부 API 없이 공간계량 모형을 바로 실험할 수 있다.

제공 데이터셋:
    - Columbus (Ohio, USA)     : 49개 census tract × 범죄·주거 변수
    - Baltimore                : 211개 주택 × 가격·입지
    - NAT                      : 3,085개 US county × 살인율 패널 (1960-90)
    - NCOVR                    : NAT의 서브셋, 간단한 튜토리얼용
    - Boston                   : 506 × 13 속성 (Harrison-Rubinfeld 1978)
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

DatasetName = Literal["columbus", "baltim", "nat", "ncovr", "boston"]


def load_example(
    name: DatasetName = "columbus",
    return_gdf: bool = True,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    PySAL 내장 예제 데이터셋 로드.

    Args:
        name: 데이터셋 이름
        return_gdf: True면 GeoDataFrame, False면 속성만 담은 DataFrame

    Returns:
        선택된 데이터셋

    Example:
        >>> gdf = load_example("columbus")
        >>> gdf.shape
        (49, 21)
        >>> gdf.columns.tolist()[:5]
        ['AREA', 'PERIMETER', 'COLUMBUS_', 'COLUMBUS_I', 'POLYID']
    """
    from libpysal.examples import load_example as _pysal_load

    # libpysal 예제명 + shapefile 매핑
    mapping = {
        "columbus": ("Columbus", "columbus.shp"),
        "baltim": ("Baltimore", "baltim.shp"),
        "nat": ("NAT", "NAT.shp"),
        "ncovr": ("NAT", "NAT.shp"),
        "boston": ("Bostonhsg", "bostonhsg.shp"),
    }
    if name not in mapping:
        raise ValueError(
            f"지원하지 않는 데이터셋: {name!r}. "
            f"{list(mapping)} 중 선택."
        )

    proper, shp = mapping[name]
    example = _pysal_load(proper)
    path = example.get_path(shp)
    gdf = gpd.read_file(path)

    if return_gdf:
        return gdf
    return pd.DataFrame(gdf.drop(columns="geometry"))


def describe_dataset(name: DatasetName) -> dict:
    """
    데이터셋 메타정보 반환.

    Returns:
        {
            'name': str,
            'n_obs': int,
            'n_vars': int,
            'geometry_type': str,
            'crs': str,
            'key_variables': list[str],
            'citation': str,
        }
    """
    meta = {
        "columbus": dict(n_obs=49, n_vars=21, geometry_type="Polygon"),
        "baltim": dict(n_obs=211, n_vars=17, geometry_type="Point"),
        "nat": dict(n_obs=3085, n_vars=69, geometry_type="Polygon"),
        "ncovr": dict(n_obs=3085, n_vars=69, geometry_type="Polygon"),
        "boston": dict(n_obs=506, n_vars=23, geometry_type="Polygon"),
    }
    if name not in meta:
        raise ValueError(f"지원하지 않는 데이터셋: {name!r}")

    info = DATASET_INFO.get(name, {})
    m = meta[name]
    return {
        "name": name,
        "n_obs": m["n_obs"],
        "n_vars": m["n_vars"],
        "geometry_type": m["geometry_type"],
        "crs": "EPSG:4326",
        "key_variables": info.get("key_vars", []),
        "citation": info.get("citation", ""),
    }


# ----------------------------------------------------------------------
# 데이터셋 설명 (참조용)
# ----------------------------------------------------------------------
DATASET_INFO = {
    "columbus": {
        "description": "49 neighborhoods in Columbus, OH (1980)",
        "key_vars": ["CRIME", "INC", "HOVAL"],
        "citation": "Anselin (1988), Spatial Econometrics, ch.12",
    },
    "baltim": {
        "description": "211 houses sold in Baltimore (1978)",
        "key_vars": ["PRICE", "SQFT", "AGE", "ROOMS"],
        "citation": "Dubin (1992), Regional Science and Urban Economics",
    },
    "nat": {
        "description": "3,085 US counties, homicide 1960-1990",
        "key_vars": ["HR60", "HR70", "HR80", "HR90", "PO60-PO90"],
        "citation": "Messner & Anselin (1999)",
    },
    "boston": {
        "description": "506 census tracts in Boston (1970)",
        "key_vars": ["MEDV", "NOX", "RM", "DIS", "TAX"],
        "citation": "Harrison & Rubinfeld (1978), JEEM",
    },
}
