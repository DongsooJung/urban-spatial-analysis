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
    raise NotImplementedError(
        "TODO: libpysal.examples.load_example(name)을 사용하여 "
        "shapefile + dbf 로드, return_gdf에 따라 반환형 결정"
    )


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
    raise NotImplementedError(
        "TODO: 각 데이터셋 메타정보를 dict 반환 (하드코딩)"
    )


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
