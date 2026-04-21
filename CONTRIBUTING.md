# 🤝 Contributing

> Urban Spatial Econometric Analysis — 연구 포트폴리오

---

## 🛠 개발 환경

```bash
git clone https://github.com/DongsooJung/urban-spatial-analysis.git
cd urban-spatial-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 테스트
pytest tests/ -v

# 노트북 실행
jupyter lab notebooks/01_columbus_full_analysis.ipynb
```

## 📋 구현 우선순위

### Phase 1 — 데이터 (30분)
- [ ] `data.load_example("columbus")` — PySAL 예제 로더

### Phase 2 — 공간가중행렬 (1시간)
- [ ] `weights.build_weights("queen")` — Polygon
- [ ] `weights.build_weights("knn")` — Point
- [ ] `weights.weights_summary` — 기초 통계

### Phase 3 — 진단 (1시간)
- [ ] `diagnostics.morans_i` — 전역
- [ ] `diagnostics.local_moran` — LISA
- [ ] `diagnostics.lm_diagnostics` — 모형 선택

### Phase 4 — 모형 (2시간)
- [ ] `SpatialModel._fit_ols`
- [ ] `SpatialModel._fit_slm`
- [ ] `SpatialModel._fit_sem`
- [ ] `SpatialModel._fit_sdm`
- [ ] `SpatialModel._fit_gwr`

### Phase 5 — 시각화 (1시간)
- [ ] `plot_choropleth`
- [ ] `plot_moran_scatter`
- [ ] `plot_lisa_cluster`
- [ ] `plot_gwr_coefficients`

## ✅ 코드 스타일

- PEP 8 (black + ruff)
- Type hints 필수
- Google-style docstring
- Functional 스타일 우선, class는 state가 필요할 때만

## 🧪 테스트

- 단위 테스트: `tests/test_*.py`
- Integration: Columbus 데이터로 end-to-end 스모크 테스트
- Coverage 목표: `models.py` 80%+

## 📬 Contact

- GitHub Issues: 기술적 질문
- Email: jds068888@gmail.com
