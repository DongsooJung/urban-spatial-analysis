# 🏗 Architecture & Methodology

> Urban Spatial Econometric Analysis — 설계 및 방법론

---

## 📐 프로젝트 구조

```
urban-spatial-analysis/
├── src/uspatial/
│   ├── __init__.py        # 공개 API
│   ├── data.py            # 예제 데이터 로더 (Columbus, Baltimore, NAT)
│   ├── weights.py         # Queen/Rook/KNN/Distance/Kernel
│   ├── models.py          # OLS/SLM/SEM/SDM/GWR 통합 I/F
│   ├── diagnostics.py     # Moran's I, LISA, LM tests
│   └── visualization.py   # choropleth, LISA map, GWR coef map
│
├── notebooks/
│   └── 01_columbus_full_analysis.ipynb  # 8단계 전 과정 시연
│
├── data/                  # 원본·가공 데이터
├── tests/                 # pytest
├── docs/                  # 방법론 문서
└── scripts/               # CLI 실행 스크립트
```

---

## 📚 5대 모형 — 수식과 의미

### 1. OLS (Ordinary Least Squares)

$$y = X\beta + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I)$$

**가정**: 오차항은 독립 동분포(iid).
**한계**: 공간 데이터에서 잔차의 공간자기상관이 흔히 위반됨 → 표준오차 과소추정, 계수 편향.

---

### 2. SLM — Spatial Lag Model

$$y = \rho W y + X\beta + \varepsilon$$

**해석**: 종속변수의 공간 spillover — 이웃의 y값이 내 y값에 직접 영향.

**예시**: 이웃 동네 범죄율이 내 동네 범죄율에 직접 전이 (range effect).

**추정**: Maximum Likelihood (ML_Lag) 또는 Instrumental Variables (GM_Lag).

**주의**: β 계수를 직접 한계효과로 해석하면 안 됨 → Impact decomposition 필요.

---

### 3. SEM — Spatial Error Model

$$y = X\beta + u, \quad u = \lambda W u + \varepsilon$$

**해석**: 관측되지 않은 공간적 공통 요인이 오차항을 통해 전파.

**예시**: 학군·재개발 기대감 같은 비관측 요인이 인접 지역에 공유됨.

**추정**: ML_Error 또는 GM_Error.

**장점**: β 계수는 여전히 직접 해석 가능 (효율성만 개선).

---

### 4. SDM — Spatial Durbin Model

$$y = \rho W y + X\beta + W X \theta + \varepsilon$$

**해석**: Lag + 설명변수의 공간 spillover 모두 포함.

**예시**: 이웃 동네의 소득(WX)이 내 동네 범죄율에 영향.

**일반성**: LeSage & Pace (2009)는 SDM을 기본 모형으로 권장 (nesting 관계로 SLM/SEM을 특수 케이스로 포함).

---

### 5. GWR — Geographically Weighted Regression

$$y_i = \beta_0(u_i, v_i) + \sum_k \beta_k(u_i, v_i) x_{ik} + \varepsilon_i$$

**해석**: **계수 자체가 위치에 따라 변함** (국지적 회귀).

**예시**: 소득이 범죄율에 미치는 영향이 북부와 남부에서 다름.

**Bandwidth 선택**: AICc 또는 CV로 최적 bandwidth 탐색 (mgwr.sel_bw.Sel_BW).

---

## 🔍 모형 선택 의사결정 흐름

```
              [ OLS 추정 ]
                    │
                    ▼
         [ 잔차 Moran's I 검정 ]
                    │
         ┌──────────┴──────────┐
         │ p < 0.05            │ p ≥ 0.05
         ▼                     ▼
   [ LM 진단 ]             OLS 채택
         │
   ┌─────┴─────┬──────┐
   ▼           ▼      ▼
LM-Lag      LM-Error  둘 다 유의
유의만      유의만
   │           │      │
   ▼           ▼      ▼
  SLM         SEM   [ Robust LM 비교 ]
                       │
               ┌───────┴────────┐
               ▼                ▼
             R-LM-Lag          R-LM-Error
             더 유의            더 유의
               │                │
               ▼                ▼
              SLM              SEM
```

**Anselin & Florax (1995)** 규칙. 둘 다 유의하지 않은데 글로벌 Moran's I가 유의하면
**heteroskedasticity** 점검 후 GWR 고려.

---

## 🎨 시각화 표준

| 시각화 | 용도 | 권장 색상 |
|--------|------|-----------|
| Choropleth | 변수 단계구분도 | YlOrRd (+), RdBu_r (±) |
| Moran Scatter | 전역 자기상관 | 4사분면 색 구분 |
| LISA Map | 국지 클러스터 | 표준 (HH=빨강, LL=파랑) |
| GWR Coef Map | 계수 공간 이질성 | RdBu_r (중앙 0) |
| Residual Map | 잔차 패턴 | RdBu_r (대칭) |

---

## 📊 성능 권고

- **n ≤ 500**: 모든 방법 즉시 실행 가능
- **n = 1K ~ 10K**: `spreg.GM_Lag` / `GM_Error` (GMM) 권장, ML보다 빠름
- **n > 10K**: sparse matrix 필수, `libpysal.weights.set_operations` 활용
- **GWR 대안**: n > 2K면 `mgwr.MGWR` (다중 bandwidth) 또는 FastGWR 고려

---

## 📖 주요 참고문헌

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
2. Anselin, L., & Florax, R. J. G. M. (Eds.) (1995). *New Directions in Spatial Econometrics*. Springer.
3. LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
4. Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression*. Wiley.
5. Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93-115.

---

## 🚧 향후 확장

- [ ] Spatial Panel Models (Elhorst 2014) — 패널 버전
- [ ] Bayesian Spatial Models (MCMC 기반)
- [ ] Spatial Quantile Regression
- [ ] Machine Learning 통합 (Spatial XGBoost, GeoAI)
- [ ] Interactive dashboard (Streamlit + Folium)
