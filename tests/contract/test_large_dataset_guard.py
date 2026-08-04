from __future__ import annotations

import pytest

from astronomicAL.platform.datasets import DatasetManager

from tests.helpers.large_dataset_guard import (
    FullMaterialisationError,
    HugeDatasetSource,
)


@pytest.fixture
def huge_datasets() -> tuple[DatasetManager, HugeDatasetSource]:
    source = HugeDatasetSource(row_count=50_000_000)

    datasets = DatasetManager()
    datasets.register_source(
        "huge",
        source,
        name="Huge guarded dataset",
        backend="huge_guard_source",
        row_count=source.row_count(),
    )
    datasets.set_active("huge")
    datasets.set_mapping("huge", "record_id", "id")
    datasets.set_mapping("huge", "target_label", "label")

    return datasets, source


@pytest.mark.plugin_quality
@pytest.mark.contract
def test_huge_dataset_source_allows_metadata_access(huge_datasets) -> None:
    datasets, source = huge_datasets

    assert datasets.active_id() == "huge"
    assert datasets.row_count("huge") == 50_000_000
    assert datasets.list_columns("huge") == ["id", "x", "y", "label", "ra", "dec"]

    methods = [call["method"] for call in source.calls]

    assert "row_count" in methods
    assert "columns" in methods


@pytest.mark.plugin_quality
@pytest.mark.contract
def test_huge_dataset_source_allows_bounded_pandas_views(huge_datasets) -> None:
    datasets, _source = huge_datasets

    limited = datasets.get_df("huge", limit=3)
    assert len(limited) == 3
    assert list(limited.columns) == ["id", "x", "y", "label", "ra", "dec"]

    column_limited = datasets.get_df("huge", columns=["id", "x"], limit=2)
    assert column_limited.to_dict(orient="records") == [
        {"id": "r0", "x": 0.0},
        {"id": "r1", "x": 1.0},
    ]

    row = datasets.get_row_by_id(
        "huge",
        "r123",
        id_column="id",
        columns=["id", "label"],
    )
    assert row.to_dict(orient="records") == [{"id": "r123", "label": -1}]


@pytest.mark.plugin_quality
@pytest.mark.contract
def test_huge_dataset_source_blocks_unbounded_get_df(huge_datasets) -> None:
    datasets, _source = huge_datasets

    with pytest.raises(FullMaterialisationError):
        datasets.get_df("huge")


@pytest.mark.plugin_quality
@pytest.mark.contract
def test_huge_dataset_source_blocks_dataset_df_compatibility_property(huge_datasets) -> None:
    datasets, _source = huge_datasets

    with pytest.raises(FullMaterialisationError):
        _ = datasets.get("huge").df