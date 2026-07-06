# Urban Spatial Econometric Analysis · 도시 공간계량 분석

> 파이썬과 GIS 기반 도시정책 평가용 공간계량 모형
> Spatial econometric models for urban policy evaluation using Python and GIS

[![CI](https://github.com/DongsooJung/urban-spatial-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/DongsooJung/urban-spatial-analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PySAL](https://img.shields.io/badge/PySAL-spreg%20%C2%B7%20esda-orange?style=flat-square)](https://pysal.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 개요 · Overview

도시 데이터(주택가격·범죄율·인프라 품질)는 인접 지역 간 **공간 자기상관**을 갖습니다. 이를 무시한 전통적 회귀모형은 편향된 계수와 잘못된 정책 결론을 낳습니다. 본 저장소는 공간 가중행렬 구축부터 SLM·SEM·SDM·GWR 추정까지 완결된 공간계량 파이프라인을 제공합니다.

Urban data — property values, crime rates, infrastructure quality — exhibits **spatial autocorrelation** across neighboring areas. Traditional regression that ignores this yields biased coefficients and invalid policy recommendations. This repository implements a complete spatial econometric pipeline, from spatial weight matrix construction to SLM/SEM/SDM/GWR estimation.

## 문제와 접근 · Problem & Approach

1. **공간 가중행렬 / Spatial Weight Matrix** — Queen 인접성, K-최근접, 거리 기반 (`weights.py`)
2. **공간 시차 모형 / Spatial Lag Model (SLM)** — 내생적 공간 상호작용 포착
3. **공간 오차 모형 / Spatial Error Model (SEM)** — 공간 상관 잔차 보정
4. **공간 더빈 모형 / Spatial Durbin Model (SDM)** — 시차·오차 효과 결합
5. **지리가중회귀 / Geographically Weighted Regression (GWR)** — 국지적 계수 추정
6. **진단 / Diagnostics** — Moran's I, LM 검정, LISA 군집 (`diagnostics.py`)

## 기술 스택 · Tech Stack

- **Core:** Python 3.11+, PySAL (libpysal, spreg, esda), GeoPandas
- **Visualization:** Matplotlib, Folium, Plotly
- **Data:** 통계청 KOSIS, 국가공간정보 NSDI (National Spatial Data Infrastructure)
- **Spatial DB:** PostGIS, GeoJSON

## 프로젝트 구조 · Repository Structure

```
urban-spatial-analysis/
├── src/uspatial/
│   ├── weights.py              # 공간 가중행렬 (W) 구축 — Queen/Rook/KNN/Distance/Kernel
│   ├── models.py               # OLS·SLM·SEM·SDM·GWR 통합 인터페이스 + impacts 분해
│   ├── diagnostics.py          # Moran's I, LISA, LM 검정, VIF, Chow
│   ├── data.py                 # 예제 데이터 로더 (Columbus, Baltimore, NAT, Boston)
│   └── visualization.py        # Choropleth · Moran scatter · LISA · GWR 계수 지도
├── notebooks/
│   └── 01_columbus_full_analysis.ipynb   # 전체 분석 워크플로(Columbus 예제, 실행 완료)
├── tests/                      # pytest — weights/models/diagnostics/data (55 tests)
├── data/README.md              # 데이터 출처 및 다운로드 안내
├── docs/ARCHITECTURE.md        # 설계·방법론 문서
├── .github/workflows/ci.yml    # CI — ruff lint + pytest (Python 3.11/3.12)
├── pyproject.toml              # 패키지 메타데이터·의존성
├── CONTRIBUTING.md
└── LICENSE
```

## 빠른 시작 · Quick Start

```bash
git clone https://github.com/DongsooJung/urban-spatial-analysis.git
cd urban-spatial-analysis
pip install -e ".[dev,notebooks]"

# 전체 워크플로 노트북 실행
jupyter notebook notebooks/01_columbus_full_analysis.ipynb
```

```python
from uspatial import SpatialModel, build_weights, load_example, morans_i

gdf = load_example("columbus")                       # 49개 Columbus 구역
w = build_weights(gdf, method="queen")               # Queen 인접 가중행렬

mi = morans_i(gdf["CRIME"].values, w)                # 전역 공간 자기상관
print(mi.interpret())                                # I ≈ 0.5, p < 0.001

sem = SpatialModel(gdf, y="CRIME", X=["INC", "HOVAL"], w=w, method="SEM").fit()
print(sem.summary())
```

## 테스트 · Testing

```bash
pytest          # 55개 테스트 (Columbus 벤치마크 회귀 검증 포함)
ruff check src tests
```

## 연구 맥락 · Research Context

서울대학교 건설환경공학부(스마트도시공학) 박사과정 연구를 지원하며, 군 공항 이전 영향 분석과 대중교통중심개발(TOD) 평가 등 도시정책 평가에 적용됩니다.

This work supports doctoral research at Seoul National University, Department of Civil & Environmental Engineering (Smart City Engineering), applied to urban policy evaluation including military airport relocation impact analysis and transit-oriented development assessment.

## 참고문헌 · References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Springer.
- LeSage, J. & Pace, R.K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Rey, S.J. & Anselin, L. (2010). PySAL: A Python Library of Spatial Analytical Methods.

## 라이선스 · License

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참조.

## 저자 · Author

**정동수 (Dongsoo Jung)** — 서울대학교 박사과정 · 스마트도시공학
[GitHub @DongsooJung](https://github.com/DongsooJung)
