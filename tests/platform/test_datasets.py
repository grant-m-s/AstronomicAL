from __future__ import annotations

import pandas as pd
import pytest

from astronomicAL.platform.datasets import DatasetManager


@pytest.mark.unit
def test_register_and_access_active_dataset(datasets: DatasetManager) -> None:
    assert datasets.list_ids() == ["main"]
    assert datasets.active_id() == "main"

    df = datasets.get_df()
    assert list(df["id"]) == ["r1", "r2", "r3", "r4"]

    assert datasets.row_count("main") == 4
    assert set(datasets.list_columns("main")) >= {"id", "x", "y", "label"}


@pytest.mark.unit
def test_column_subset_materialisation(datasets: DatasetManager) -> None:
    df = datasets.get_df("main", columns=["id", "x"], limit=2)

    assert list(df.columns) == ["id", "x"]
    assert df.to_dict(orient="records") == [
        {"id": "r1", "x": 1.0},
        {"id": "r2", "x": 2.0},
    ]


@pytest.mark.unit
def test_head_uses_dataset_source(datasets: DatasetManager) -> None:
    head = datasets.head("main", n=2, columns=["id", "label"])

    assert list(head.columns) == ["id", "label"]
    assert head.to_dict(orient="records") == [
        {"id": "r1", "label": 0},
        {"id": "r2", "label": 1},
    ]


@pytest.mark.unit
def test_dataset_mappings_are_dataset_local(datasets: DatasetManager) -> None:
    datasets.register(
        "secondary",
        pd.DataFrame({"row_key": ["a", "b"], "class": [1, 0]}),
        name="Secondary",
    )

    datasets.set_mapping("secondary", "record_id", "row_key")
    datasets.set_mapping("secondary", "target_label", "class")

    assert datasets.get_mapping("main", "record_id") == "id"
    assert datasets.get_mapping("main", "target_label") == "label"

    assert datasets.get_mapping("secondary", "record_id") == "row_key"
    assert datasets.get_mapping("secondary", "target_label") == "class"


@pytest.mark.unit
def test_set_active_unknown_dataset_raises(datasets: DatasetManager) -> None:
    with pytest.raises(KeyError):
        datasets.set_active("missing")


@pytest.mark.unit
def test_dataset_snapshot_restores_mappings_after_dataset_is_registered(
    datasets: DatasetManager,
    sample_df: pd.DataFrame,
) -> None:
    snapshot = datasets.snapshot()

    restored = DatasetManager()

    # Simulate workspace restore before file/data loading has completed.
    restored.restore_metadata_snapshot(snapshot)

    # Registering the matching dataset should apply pending restored mappings.
    restored.register("main", sample_df.copy(), name="Main test dataset")

    assert restored.active_id() == "main"
    assert restored.get_mapping("main", "record_id") == "id"
    assert restored.get_mapping("main", "target_label") == "label"


@pytest.mark.unit
def test_get_row_by_id_uses_mapped_column_convention(datasets: DatasetManager) -> None:
    id_column = datasets.get_mapping("main", "record_id")

    row = datasets.get_row_by_id("main", "r3", id_column=id_column)

    assert row.iloc[0]["id"] == "r3"
    assert row.iloc[0]["x"] == 3.0