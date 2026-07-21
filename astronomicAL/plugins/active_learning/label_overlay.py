from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import pandas as pd

from astronomicAL.platform.dataset_sources import PandasDatasetSource

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
) -> Dict[str, Any]:
    """Attach verified labels to the existing pool dataset.

    The dataset ID and base source remain unchanged. A named sidecar overlay is
    replaced on each round so training never creates ``__al_train_rN`` datasets.
    """

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
        overlay_source=PandasDatasetSource(frame),
        base_record_id_column=str(record_id_column),
        overlay_record_id_column=str(record_id_column),
        columns=[str(target_column)],
        preserve_base_on_missing=True,
        origin=str(origin),
        al_label_overlay=True,
        al_label_session_id=str(session_id),
        al_label_session_artifact_id=str(session_artifact_id),
        al_label_round=int(round_index),
        al_label_row_count=int(len(frame)),
        al_label_target_column=str(target_column),
    )
    return {
        "overlay_name": OVERLAY_NAME,
        "dataset_id": str(dataset_id),
        "record_id_column": str(record_id_column),
        "target_column": str(target_column),
        "row_count": int(len(frame)),
        "changed_columns": list(changed_columns),
    }