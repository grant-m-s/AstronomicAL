from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DuckDBParquetDatasetSource,
    PandasDatasetSource,
)

from .label_storage import LabelTableRef

OVERLAY_NAME = "core.active_learning.labels"

def attach_session_labels(
    context: Any,
    *,
    dataset_id: str,
    record_id_column: str,
    target_column: str,
    labelled_items: Sequence[Mapping[str, Any]],
    session_id: str,
    session_artifact_id: str,
    round_index: int,
    origin: str,
    label_table_ref: Mapping[str, Any] | LabelTableRef | None = None,
) -> Dict[str, Any]:
    """Attach verified labels to the existing pool dataset.

    The dataset ID and base source remain unchanged. A named sidecar overlay is
    replaced on each round so training never creates ``__al_train_rN`` datasets.
    """

    overlay_source = None
    overlay_record_id_column = str(record_id_column)
    durable_ref = None
    if label_table_ref is None and session_artifact_id:
        try:
            session_payload = context.artifacts.get(str(session_artifact_id))
        except Exception:
            session_payload = None
        if isinstance(session_payload, Mapping):
            label_table_ref = session_payload.get("label_table_ref")
    if label_table_ref is not None:
        try:
            durable_ref = LabelTableRef.from_value(label_table_ref)
        except Exception:
            durable_ref = None
    if durable_ref is not None and durable_ref.parquet_parts:
        overlay_record_id_column = durable_ref.record_id_column
        overlay_source = DuckDBParquetDatasetSource(
            durable_ref.parquet_parts,
            dataset_name=f"{session_id}:labels",
            columns_hint=[durable_ref.record_id_column, str(target_column)],
            row_count_hint=durable_ref.row_count,
        )

    rows: list[Dict[str, Any]] = []
    if overlay_source is None:
        rows = [
            {
                str(record_id_column): str(item["row_id"]),
                str(target_column): item.get("label"),
            }
            for item in labelled_items
            if item.get("row_id") not in (None, "")
        ]
        if not rows:
            raise ValueError("No verified labels are available for the pool overlay.")
        frame = pd.DataFrame.from_records(
            rows,
            columns=[str(record_id_column), str(target_column)],
        )
        ids = frame[str(record_id_column)].map(str)
        if ids.duplicated().any():
            duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
            raise ValueError(
                "Active Learning labels contain duplicate record IDs: "
                f"{duplicates[:10]!r}"
            )
        overlay_source = PandasDatasetSource(frame)

    datasets = getattr(context, "datasets", None)
    upsert = getattr(datasets, "upsert_column_overlay", None)
    if not callable(upsert):
        raise RuntimeError(
            "DatasetManager.upsert_column_overlay() is required to attach "
            "Active Learning labels without creating another dataset."
        )

    changed_columns = upsert(
        str(dataset_id),
        overlay_name=OVERLAY_NAME,
        overlay_source=overlay_source,
        base_record_id_column=str(record_id_column),
        overlay_record_id_column=str(overlay_record_id_column),
        columns=[str(target_column)],
        preserve_base_on_missing=True,
        origin=str(origin),
        al_label_overlay=True,
        al_label_session_id=str(session_id),
        al_label_session_artifact_id=str(session_artifact_id),
        al_label_round=int(round_index),
        al_label_row_count=int(durable_ref.row_count if durable_ref is not None else len(rows)),
        al_label_table_ref=(durable_ref.to_dict() if durable_ref is not None else None),
        al_label_target_column=str(target_column),
    )
    return {
        "overlay_name": OVERLAY_NAME,
        "dataset_id": str(dataset_id),
        "record_id_column": str(record_id_column),
        "target_column": str(target_column),
        "row_count": int(durable_ref.row_count if durable_ref is not None else len(rows)),
        "storage_mode": "parquet_sidecar" if durable_ref is not None and durable_ref.parquet_parts else "pandas_compatibility",
        "changed_columns": list(changed_columns),
    }
