"""데이터 로더 테스트 — load_example / describe_dataset."""
import pytest

from uspatial.data import describe_dataset, load_example


class TestLoadExample:
    def test_columbus_shape(self, columbus_gdf):
        assert columbus_gdf.shape[0] == 49

    def test_columbus_is_polygon(self, columbus_gdf):
        assert set(columbus_gdf.geom_type.unique()) <= {"Polygon", "MultiPolygon"}

    def test_columbus_key_variables_present(self, columbus_gdf):
        assert {"CRIME", "INC", "HOVAL"} <= set(columbus_gdf.columns)

    def test_return_dataframe(self, columbus_gdf):
        df = load_example("columbus", return_gdf=False)
        assert "geometry" not in df.columns
        assert len(df) == 49

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError):
            load_example("nonexistent")


class TestDescribeDataset:
    def test_describe_keys(self):
        d = describe_dataset("columbus")
        assert d["n_obs"] == 49
        assert set(d.keys()) >= {"name", "n_obs", "geometry_type", "citation"}

    @pytest.mark.parametrize("name", ["columbus", "baltim", "nat", "ncovr", "boston"])
    def test_all_supported_have_metadata(self, name):
        d = describe_dataset(name)
        assert d["n_obs"] > 0
        assert d["key_variables"], f"{name}: key_variables 비어 있음"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            describe_dataset("nope")
