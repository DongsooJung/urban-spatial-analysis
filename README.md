# Urban Spatial Econometric Analysis

> Spatial econometric models for urban policy evaluation using Python and GIS

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey?style=flat-square)]()

## Problem

Traditional regression models fail to capture **spatial dependencies** in urban data — property values, crime rates, and infrastructure quality are inherently correlated across neighboring areas. Ignoring spatial autocorrelation leads to biased coefficients and invalid policy recommendations.

## Solution

This repository implements a complete spatial econometric pipeline:

1. **Spatial Weight Matrix** construction (Queen contiguity, K-nearest, distance-based)
2. **Spatial Lag Model (SLM)** — captures endogenous spatial interaction
3. **Spatial Error Model (SEM)** — corrects for spatially correlated residuals
4. **Spatial Durbin Model (SDM)** — combines both lag and error effects
5. **Geographically Weighted Regression (GWR)** — local coefficient estimation

## Tech Stack

- **Core:** Python 3.11+, PySAL (libpysal, spreg, esda), GeoPandas
- **Visualization:** Matplotlib, Folium, Plotly
- **Data:** Korean Statistical Information Service (KOSIS), National Spatial Data Infrastructure (NSDI)
- **Spatial DB:** PostGIS, GeoJSON

## Repository Structure

```
urban-spatial-analysis/
├── src/
│   ├── spatial_weights.py      # W matrix construction
│   ├── models.py               # SLM, SEM, SDM, GWR wrappers
│   ├── diagnostics.py          # Moran's I, LM tests, LISA
│   └── visualization.py        # Choropleth & cluster maps
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_spatial_analysis.ipynb
│   ├── 03_model_estimation.ipynb
│   └── 04_policy_simulation.ipynb
├── data/
│   └── README.md               # Data sources & download instructions
├── docs/
│   └── methodology.md          # Mathematical formulation
├── tests/
│   └── test_models.py
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
git clone https://github.com/DongsooJung/urban-spatial-analysis.git
cd urban-spatial-analysis
pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb
```

## Key Results

| Model | AIC | Log-Likelihood | Spatial ρ/λ | R² |
|-------|-----|----------------|-------------|-----|
| OLS (baseline) | — | — | — | — |
| Spatial Lag | — | — | — | — |
| Spatial Error | — | — | — | — |
| SDM | — | — | — | — |

*Results populated upon running analysis with target dataset.*

## Research Context

This work supports doctoral research at **Seoul National University**, Department of Civil & Environmental Engineering (Smart City Engineering). Applied to urban policy evaluation including military airport relocation impact analysis and transit-oriented development assessment.

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Springer.
- LeSage, J. & Pace, R.K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Rey, S.J. & Anselin, L. (2010). PySAL: A Python Library of Spatial Analytical Methods.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Dongsoo Jung** — Ph.D. Candidate, Seoul National University  
[DongsooJung](https://github.com/DongsooJung) (Business) · [DongsooJung](https://github.com/DongsooJung) (Research)
