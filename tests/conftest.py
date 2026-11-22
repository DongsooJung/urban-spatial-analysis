"""공용 픽스처 — 공간계량 스택이 없으면 전체 skip."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("libpysal")

# 패키지 미설치 환경(pip install -e . 이전)에서도 임포트 가능하도록
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def columbus_gdf():
    """Columbus 내장 예제 데이터 (49 폴리곤). 최초 1회 네트워크 필요."""
    from uspatial.data import load_example

    try:
        return load_example("columbus")
    except (Exception, SystemExit) as e:  # 네트워크 불가 등 (libpysal은 SystemExit도 던짐)
        pytest.skip(f"Columbus 로드 실패: {e}")


@pytest.fixture(scope="session")
def queen_w(columbus_gdf):
    from uspatial.weights import build_weights

    return build_weights(columbus_gdf, method="queen")
