from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from astronomicAL.platform.context import AppContext
from astronomicAL.platform.dataset_sources import DuckDBParquetDatasetSource
from astronomicAL.platform.persistence import WorkspacePersistence

from .bundled_plugin_helpers import (
    core_panel_registrations,
    enable_core_plugins,
    ensure_required_mappings_for_panel,
    get_workspace_record,
    make_bundled_plugin_manager,
    make_fake_workspace_manager_compatible,
    prepare_context_for_bundled_plugins,
    wait_for_workspace_panel_kind,
)


@pytest.fixture
def parquet_dependencies() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


def _large_test_dataframe(rows: int = 1_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"row-{index:05d}" for index in range(rows)],
            "value": [float(index) for index in range(rows)],
            "other": [float(rows - index) for index in range(rows)],
            "group": [f"group-{index % 5}" for index in range(rows)],
            "label": [index % 3 for index in range(rows)],
        }
    )


def _write_parquet(
    tmp_path: Path,
    df: pd.DataFrame,
    name: str = "dataset.parquet",
) -> Path:
    path = tmp_path / name
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def _register_parquet_active_dataset(
    context: AppContext,
    path: Path | list[Path],
    *,
    dataset_id: str = "parquet_main",
    columns: list[str],
    row_count: int,
) -> None:
    context.datasets.register_parquet(
        dataset_id,
        path,
        name="Parquet Main",
        columns=columns,
        row_count=row_count,
        provenance="test-suite",
    )
    context.datasets.set_active(dataset_id)

    context.datasets.set_mapping(dataset_id, "record_id", "id")
    context.datasets.set_mapping(dataset_id, "target_label", "label")
    context.datasets.set_mapping(dataset_id, "x", "value")
    context.datasets.set_mapping(dataset_id, "y", "other")
    context.datasets.set_mapping(dataset_id, "group", "group")


def _prepare_parquet_plugin_context(context: AppContext) -> None:
    prepare_context_for_bundled_plugins(context)
    make_fake_workspace_manager_compatible(context)
    context.persistence = WorkspacePersistence(context)


class GuardedDuckDBParquetDatasetSource(DuckDBParquetDatasetSource):
    """
    DuckDBParquetDatasetSource test double that fails if code asks for an
    unbounded all-column pandas materialisation.

    This does not ban pandas views entirely. It only bans the dangerous
    compatibility pattern equivalent to "load the whole dataset".
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.to_pandas_calls: list[dict[str, Any]] = []

    def to_pandas(
        self,
        *,
        columns=None,
        limit=None,
        where_sql=None,
        params=None,
    ) -> pd.DataFrame:
        self.to_pandas_calls.append(
            {
                "columns": None if columns is None else list(columns),
                "limit": limit,
                "where_sql": where_sql,
                "params": None if params is None else list(params),
            }
        )

        if columns is None and limit is None and where_sql is None:
            raise AssertionError(
                "Full pandas materialisation requested from a Parquet source. "
                "Use columns, limit, where_sql, get_row_by_id, or "
                "get_row_by_position instead."
            )

        return super().to_pandas(
            columns=columns,
            limit=limit,
            where_sql=where_sql,
            params=params,
        )


def test_register_parquet_preserves_duckdb_backend_and_metadata_hints(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-basic")
    df = _large_test_dataframe(rows=1_000)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )

    source = context.datasets.get_source("parquet_main")
    meta = context.datasets.get_meta("parquet_main")

    assert source.backend_name == "duckdb_parquet"
    assert meta["backend"] == "duckdb_parquet"
    assert meta["source_format"] == "parquet"
    assert meta["row_count"] == len(df)
    assert meta["columns"] == list(df.columns)

    assert context.datasets.list_columns("parquet_main") == list(df.columns)
    assert context.datasets.row_count("parquet_main") == len(df)

    head = context.datasets.head(
        "parquet_main",
        n=3,
        columns=["id", "value"],
    )

    assert list(head.columns) == ["id", "value"]
    assert list(head["id"]) == ["row-00000", "row-00001", "row-00002"]


def test_parquet_source_supports_filtered_limited_column_materialisation(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-filtered")
    df = _large_test_dataframe(rows=1_000)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )

    filtered = context.datasets.get_df(
        "parquet_main",
        columns=["id", "value", "group"],
        where_sql='"value" >= ?',
        params=[995.0],
        limit=3,
    )

    assert list(filtered.columns) == ["id", "value", "group"]
    assert len(filtered) == 3
    assert list(filtered["id"]) == ["row-00995", "row-00996", "row-00997"]
    assert list(filtered["value"]) == [995.0, 996.0, 997.0]


def test_parquet_source_supports_row_lookup_by_position_and_record_id(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-row-lookup")
    df = _large_test_dataframe(rows=250)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )

    by_position = context.datasets.get_row_by_position(
        "parquet_main",
        42,
        columns=["id", "value"],
    )

    assert len(by_position) == 1
    assert by_position.iloc[0]["id"] == "row-00042"
    assert by_position.iloc[0]["value"] == 42.0

    by_id = context.datasets.get_row_by_id(
        "parquet_main",
        "row-00123",
        id_column="id",
        columns=["id", "group", "label"],
    )

    assert len(by_id) == 1
    assert by_id.iloc[0]["id"] == "row-00123"
    assert by_id.iloc[0]["group"] == "group-3"
    assert by_id.iloc[0]["label"] == 0

    position = context.datasets.find_position_by_id(
        "parquet_main",
        "row-00123",
        id_column="id",
    )

    assert position == 123

    missing = context.datasets.get_row_by_id(
        "parquet_main",
        "does-not-exist",
        id_column="id",
        columns=["id", "value"],
    )

    assert missing.empty
    assert list(missing.columns) == ["id", "value"]


def test_parquet_source_handles_quoted_and_awkward_column_names(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-quoted-columns")

    df = pd.DataFrame(
        {
            "source id": ["src-0", "src-1", "src-2"],
            "value with space": [10.0, 20.0, 30.0],
            'quoted " value': [1, 2, 3],
            "label/class": [0, 1, 1],
        }
    )

    path = _write_parquet(tmp_path, df, name="quoted-columns.parquet")

    context.datasets.register_parquet(
        "quoted",
        path,
        name="Quoted Columns",
        columns=list(df.columns),
        row_count=len(df),
    )
    context.datasets.set_active("quoted")
    context.datasets.set_mapping("quoted", "record_id", "source id")
    context.datasets.set_mapping("quoted", "target_label", "label/class")

    selected = context.datasets.get_df(
        "quoted",
        columns=["source id", "value with space", 'quoted " value'],
        limit=2,
    )

    assert list(selected.columns) == [
        "source id",
        "value with space",
        'quoted " value',
    ]
    assert list(selected["source id"]) == ["src-0", "src-1"]

    by_id = context.datasets.get_row_by_id(
        "quoted",
        "src-2",
        id_column="source id",
        columns=["source id", "value with space", "label/class"],
    )

    assert len(by_id) == 1
    assert by_id.iloc[0]["source id"] == "src-2"
    assert by_id.iloc[0]["value with space"] == 30.0
    assert by_id.iloc[0]["label/class"] == 1


def test_parquet_source_supports_multiple_files(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-multiple-files")

    df = _large_test_dataframe(rows=40)
    first = df.iloc[:20].copy()
    second = df.iloc[20:].copy()

    part_one = _write_parquet(tmp_path, first, name="part-1.parquet")
    part_two = _write_parquet(tmp_path, second, name="part-2.parquet")

    context.datasets.register_parquet(
        "parts",
        [part_one, part_two],
        name="Partitioned Parquet",
        columns=list(df.columns),
    )
    context.datasets.set_active("parts")

    assert context.datasets.list_columns("parts") == list(df.columns)
    assert context.datasets.row_count("parts") == 40

    tail = context.datasets.get_df(
        "parts",
        columns=["id", "value"],
        where_sql='"value" >= ?',
        params=[37.0],
        limit=10,
    )

    assert list(tail["id"]) == ["row-00037", "row-00038", "row-00039"]
    assert list(tail["value"]) == [37.0, 38.0, 39.0]


def test_hinted_parquet_source_does_not_connect_for_columns_or_row_count(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-hints-no-connect")

    source = DuckDBParquetDatasetSource(
        tmp_path / "does-not-need-to-exist.parquet",
        dataset_name="Hinted Dataset",
        columns_hint=["id", "value", "label"],
        row_count_hint=123_456,
    )

    def fail_connect():
        raise AssertionError("DuckDB connection should not be opened for hints")

    source._connect = fail_connect  # type: ignore[method-assign]

    context.datasets.register_source(
        "hinted",
        source,
        name="Hinted Dataset",
        columns=["id", "value", "label"],
        row_count=123_456,
    )

    assert context.datasets.list_columns("hinted") == ["id", "value", "label"]
    assert context.datasets.row_count("hinted") == 123_456


def test_dataset_snapshot_of_parquet_source_does_not_materialise_full_dataframe(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-snapshot-no-fullscan")
    df = _large_test_dataframe(rows=500)
    path = _write_parquet(tmp_path, df)

    source = GuardedDuckDBParquetDatasetSource(
        path,
        dataset_name="Guarded Parquet",
        columns_hint=list(df.columns),
        row_count_hint=len(df),
    )

    context.datasets.register_source(
        "guarded",
        source,
        name="Guarded Parquet",
        columns=list(df.columns),
        row_count=len(df),
    )
    context.datasets.set_active("guarded")

    context.datasets.set_mapping("guarded", "record_id", "id")
    context.datasets.set_mapping("guarded", "target_label", "label")

    snapshot = context.datasets.snapshot()

    assert snapshot["active_id"] == "guarded"
    item = next(item for item in snapshot["items"] if item["id"] == "guarded")
    assert item["columns"] == list(df.columns)
    assert item["mappings"]["record_id"] == "id"
    assert item["mappings"]["target_label"] == "label"

    assert source.to_pandas_calls == []


def test_guarded_parquet_source_allows_targeted_access_but_rejects_full_scan(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-guarded-access")
    df = _large_test_dataframe(rows=500)
    path = _write_parquet(tmp_path, df)

    source = GuardedDuckDBParquetDatasetSource(
        path,
        dataset_name="Guarded Parquet",
        columns_hint=list(df.columns),
        row_count_hint=len(df),
    )

    context.datasets.register_source(
        "guarded",
        source,
        name="Guarded Parquet",
        columns=list(df.columns),
        row_count=len(df),
    )
    context.datasets.set_active("guarded")

    preview = context.datasets.get_df(
        "guarded",
        columns=["id", "value"],
        limit=5,
    )
    assert list(preview["id"]) == [
        "row-00000",
        "row-00001",
        "row-00002",
        "row-00003",
        "row-00004",
    ]

    by_id = context.datasets.get_row_by_id(
        "guarded",
        "row-00499",
        id_column="id",
        columns=["id", "label"],
    )
    assert by_id.iloc[0]["id"] == "row-00499"

    with pytest.raises(AssertionError, match="Full pandas materialisation"):
        context.datasets.get_df("guarded")

    assert any(call["limit"] == 5 for call in source.to_pandas_calls)
    assert any(call["where_sql"] is not None for call in source.to_pandas_calls)


def test_bundled_core_panels_open_against_parquet_active_dataset(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-core-panels")
    df = _large_test_dataframe(rows=1_000)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )
    _prepare_parquet_plugin_context(context)

    manager = make_bundled_plugin_manager(context)
    enable_core_plugins(manager, context)

    opened: list[str] = []

    for registration in core_panel_registrations(manager):
        ensure_required_mappings_for_panel(context, registration)

        workspace_id = manager.open_panel(
            registration.id,
            context=context,
            instance_id=f"parquet-open:{registration.id}",
        )

        record = wait_for_workspace_panel_kind(context, workspace_id)
        opened.append(workspace_id)

        assert record.kind == "plugin_panel"
        assert record.plugin_id == registration.plugin_id
        assert record.registration_id == registration.id

    assert opened

    failed_events = [
        payload
        for _, topic, payload in context.events.recent_events(1_000)
        if topic == "plugin.panel.open_failed"
    ]
    assert failed_events == []

    for workspace_id in opened:
        context.workspace.remove_panel(workspace_id)

    assert context.events.list_subscriptions() == []


def test_record_browser_uses_parquet_record_id_mapping_for_initial_focus(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    context = make_app_context("parquet-record-browser")
    df = _large_test_dataframe(rows=100)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )
    _prepare_parquet_plugin_context(context)

    manager = make_bundled_plugin_manager(context)
    enable_core_plugins(manager, context)

    registration_id = "core.record_browser.panel"
    registration = manager.get_panel(registration_id)
    ensure_required_mappings_for_panel(context, registration)

    workspace_id = manager.open_panel(
        registration_id,
        context=context,
        instance_id="parquet-record-browser",
    )

    record = wait_for_workspace_panel_kind(context, workspace_id)

    assert record.kind == "plugin_panel"
    assert record.plugin_id == "core.record_browser"

    focus = context.selection.get_focus()

    assert focus.dataset_id == "parquet_main"
    assert focus.row_id == "row-00000"

    row = context.datasets.get_row_by_id(
        "parquet_main",
        focus.row_id,
        id_column="id",
        columns=["id", "value", "label"],
    )

    assert len(row) == 1
    assert row.iloc[0]["id"] == "row-00000"
    assert row.iloc[0]["value"] == 0.0

    context.workspace.remove_panel(workspace_id)
    assert get_workspace_record(context, workspace_id) is None
    assert context.events.list_subscriptions() == []


def test_workspace_persistence_snapshot_round_trips_parquet_dataset_metadata(
    make_app_context,
    tmp_path: Path,
    parquet_dependencies: None,
) -> None:
    source_context = make_app_context("parquet-persistence-source")
    df = _large_test_dataframe(rows=250)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        source_context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )
    _prepare_parquet_plugin_context(source_context)

    manager = make_bundled_plugin_manager(source_context)
    enable_core_plugins(manager, source_context)

    registration_id = "core.plugin_manager.panel"
    registration = manager.get_panel(registration_id)
    ensure_required_mappings_for_panel(source_context, registration)

    workspace_id = manager.open_panel(
        registration_id,
        context=source_context,
        instance_id="parquet-persist-plugin-manager",
    )
    wait_for_workspace_panel_kind(source_context, workspace_id)

    source_context.selection.set_focus(
        "parquet_main",
        "row-00200",
        origin="parquet-persistence-test",
    )

    snapshot = source_context.persistence.snapshot()

    dataset_snapshot = snapshot["datasets"]
    assert dataset_snapshot["active_id"] == "parquet_main"

    parquet_item = next(
        item
        for item in dataset_snapshot["items"]
        if item["id"] == "parquet_main"
    )

    assert parquet_item["mappings"]["record_id"] == "id"
    assert parquet_item["mappings"]["target_label"] == "label"
    assert parquet_item["mappings"]["x"] == "value"
    assert parquet_item["mappings"]["y"] == "other"
    assert parquet_item["columns"] == list(df.columns)
    assert parquet_item["meta"]["backend"] == "duckdb_parquet"
    assert parquet_item["meta"]["source_format"] == "parquet"

    restored_context = make_app_context("parquet-persistence-target")
    _register_parquet_active_dataset(
        restored_context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )
    _prepare_parquet_plugin_context(restored_context)

    restored_manager = make_bundled_plugin_manager(restored_context)
    enable_core_plugins(restored_manager, restored_context)

    restored_context.workspace.clear()

    issues = restored_context.persistence.restore(snapshot)

    assert issues == []

    restored_record = wait_for_workspace_panel_kind(
        restored_context,
        "parquet-persist-plugin-manager",
    )
    assert restored_record.kind == "plugin_panel"
    assert restored_record.plugin_id == "core.plugin_manager"

    assert restored_context.datasets.active_id() == "parquet_main"
    assert restored_context.datasets.get_mapping(
        "parquet_main",
        "record_id",
    ) == "id"
    assert restored_context.datasets.get_mapping(
        "parquet_main",
        "target_label",
    ) == "label"

    focus = restored_context.selection.get_focus()
    assert focus.dataset_id == "parquet_main"
    assert focus.row_id == "row-00200"

def test_table_tools_create_subset_from_selection_subset_over_parquet_does_not_materialise(
    make_app_context,
    tmp_path,
    parquet_dependencies,
) -> None:
    from astronomicAL.plugins.selection_tools.plugin import (
        SelectionSubsetDatasetSource,
    )
    from astronomicAL.plugins.table_tools.plugin import create_subset_action
    from astronomicAL.platform.plugins.specs import ActionRequest

    context = make_app_context("table-tools-selection-subset-parquet")

    df = _large_test_dataframe(rows=10_000)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )

    base_source = context.datasets.get_source("parquet_main")

    selected_ids = [f"row-{index:05d}" for index in range(0, 10_000, 2)]

    subset_source = SelectionSubsetDatasetSource(
        base_source=base_source,
        row_ids=selected_ids,
        id_column="id",
        columns=list(df.columns),
    )

    context.datasets.register_source(
        "selected_parquet",
        subset_source,
        name="Selected Parquet",
        derived_from="parquet_main",
        column_mappings={
            "record_id": "id",
            "target_label": "label",
            "x": "value",
            "y": "other",
        },
    )
    context.datasets.set_active("selected_parquet")

    result = create_subset_action(
        context,
        ActionRequest(
            dataset_id="selected_parquet",
            params={
                "subset_name": "Selected high values",
                "expression": "value >= 9000",
                "set_active": True,
            },
        ),
    )

    new_dataset_id = result.value["dataset_id"]

    assert context.datasets.get_source(new_dataset_id).backend_name == "duckdb_parquet"
    assert context.datasets.row_count(new_dataset_id) == 500

    preview = context.datasets.head(
        new_dataset_id,
        n=5,
        columns=["id", "value"],
    )

    assert list(preview.columns) == ["id", "value"]
    assert preview["value"].min() >= 9000

def test_table_tools_add_column_to_parquet_dataset_registers_lazy_derived_source(
    make_app_context,
    tmp_path,
    parquet_dependencies,
) -> None:
    from astronomicAL.plugins.table_tools.plugin import add_column_action
    from astronomicAL.platform.plugins.specs import ActionRequest

    context = make_app_context("table-tools-add-column-parquet")

    df = _large_test_dataframe(rows=1_000)
    path = _write_parquet(tmp_path, df)

    _register_parquet_active_dataset(
        context,
        path,
        columns=list(df.columns),
        row_count=len(df),
    )

    result = add_column_action(
        context,
        ActionRequest(
            dataset_id="parquet_main",
            params={
                "new_column": "sum_value",
                "expression": "value + other",
            },
        ),
    )

    assert result.value["column"] == "sum_value"
    assert result.value["rows"] == 1_000
    assert result.value["materialized"] is False

    source = context.datasets.get_source("parquet_main")
    assert source.backend_name == "duckdb_parquet_derived_column"

    assert "sum_value" in context.datasets.list_columns("parquet_main")

    preview = context.datasets.head(
        "parquet_main",
        n=5,
        columns=["id", "value", "other", "sum_value"],
    )

    assert list(preview.columns) == ["id", "value", "other", "sum_value"]
    assert list(preview["sum_value"]) == [1000.0] * 5