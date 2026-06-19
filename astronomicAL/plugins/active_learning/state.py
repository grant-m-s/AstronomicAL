from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional


ARTIFACT_SESSION = "al.session"
ARTIFACT_TRAINING_SET = "al.training_set"
ARTIFACT_BATCH = "ml.active_learning_batch"

UNSURE_LABEL = "__unsure__"
UNSURE_DISPLAY = "Unsure"


def now() -> float:
    return time.time()


def parse_label_options(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("\n", ",").split(",")]
    elif isinstance(value, Iterable):
        parts = [str(part).strip() for part in value]
    else:
        parts = [str(value).strip()]

    labels: List[str] = []
    for label in parts:
        if not label:
            continue
        if normalise_label(label) == UNSURE_LABEL:
            continue
        if label not in labels:
            labels.append(label)

    return labels


def normalise_label(label: Any) -> str:
    value = str(label or "").strip()
    if value.lower() in {"unsure", "uncertain", "unknown", "skip", UNSURE_LABEL.lower()}:
        return UNSURE_LABEL
    return value


def display_label(label: Any) -> str:
    value = normalise_label(label)
    if value == UNSURE_LABEL:
        return UNSURE_DISPLAY
    return value


def create_session(
    *,
    dataset_id: str,
    label_options: Iterable[str],
    seed: int = 42,
    target_column: str = "al_label",
    session_id: Optional[str] = None,
    recipe_id: Optional[str] = None,
    pool_dataset_id: Optional[str] = None,
    validation_dataset_id: Optional[str] = None,
    test_dataset_id: Optional[str] = None,
    al_protocol: str = "review",
) -> Dict[str, Any]:
    timestamp = now()
    return {
        "schema_version": 1,
        "session_id": session_id or f"al:{uuid.uuid4().hex[:12]}",
        "revision": 0,
        "previous_session_artifact_id": None,
        "dataset_id": str(dataset_id),
        "pool_dataset_id": str(pool_dataset_id or dataset_id),
        "validation_dataset_id": str(validation_dataset_id or ""),
        "test_dataset_id": str(test_dataset_id or ""),
        "recipe_id": str(recipe_id or ""),
        "al_protocol": str(al_protocol or "review"),
        "target_column": str(target_column or "al_label"),
        "label_options": parse_label_options(label_options),
        "seed": int(seed or 0),
        "round": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
        "labels": {},
        "ignored_row_ids": [],
        "training_row_ids": [],
        "last_batch": None,
        "history": [],
    }

def coerce_session(payload: Mapping[str, Any]) -> Dict[str, Any]:
    session = copy.deepcopy(dict(payload or {}))
    session.setdefault("schema_version", 1)
    session.setdefault("session_id", f"al:{uuid.uuid4().hex[:12]}")
    session.setdefault("revision", 0)
    session.setdefault("previous_session_artifact_id", None)
    session.setdefault("dataset_id", "")
    session.setdefault("pool_dataset_id", session.get("dataset_id", ""))
    session.setdefault("validation_dataset_id", "")
    session.setdefault("test_dataset_id", "")
    session.setdefault("recipe_id", "")
    session.setdefault("al_protocol", "review")
    session.setdefault("target_column", "al_label")
    session["label_options"] = parse_label_options(session.get("label_options"))
    session.setdefault("seed", 42)
    session.setdefault("round", 0)
    session.setdefault("created_at", now())
    session.setdefault("updated_at", now())
    session.setdefault("labels", {})
    session.setdefault("ignored_row_ids", [])
    session.setdefault("training_row_ids", [])
    session.setdefault("last_batch", None)
    session.setdefault("history", [])
    return session


def record_label(
    session_payload: Mapping[str, Any],
    *,
    row_id: Any,
    label: Any,
    source: str = "manual",
    round_index: Optional[int] = None,
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    row_id_str = str(row_id)
    label_value = normalise_label(label)
    timestamp = now()

    if not row_id_str:
        raise ValueError("row_id is required.")
    if not label_value:
        raise ValueError("label is required.")

    is_unsure = label_value == UNSURE_LABEL
    entry = {
        "row_id": row_id_str,
        "label": label_value,
        "display_label": display_label(label_value),
        "status": "unsure" if is_unsure else "verified",
        "source": str(source or "manual"),
        "round": int(round_index if round_index is not None else session.get("round", 0)),
        "timestamp": timestamp,
    }

    labels = dict(session.get("labels") or {})
    labels[row_id_str] = entry
    session["labels"] = labels

    ignored = _stable_unique([*session.get("ignored_row_ids", []), row_id_str])
    session["ignored_row_ids"] = ignored

    training = [
        str(item)
        for item in session.get("training_row_ids", [])
        if str(item) != row_id_str
    ]
    if not is_unsure:
        training.append(row_id_str)
    session["training_row_ids"] = _stable_unique(training)

    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": "label_recorded",
            "row_id": row_id_str,
            "label": label_value,
            "status": entry["status"],
            "timestamp": timestamp,
        }
    )
    return session


def with_last_batch(
    session_payload: Mapping[str, Any],
    *,
    batch_artifact_id: str,
    strategy_id: str,
    row_ids: Iterable[Any],
    predictions_artifact_id: Optional[str] = None,
    kind: str = "query",
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    row_ids_list = [str(row_id) for row_id in row_ids]
    timestamp = now()

    session["last_batch"] = {
        "kind": str(kind),
        "batch_artifact_id": str(batch_artifact_id),
        "predictions_artifact_id": predictions_artifact_id,
        "strategy_id": str(strategy_id),
        "row_ids": row_ids_list,
        "count": len(row_ids_list),
        "round": int(session.get("round", 0)),
        "timestamp": timestamp,
    }
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": "batch_created",
            "kind": str(kind),
            "batch_artifact_id": str(batch_artifact_id),
            "strategy_id": str(strategy_id),
            "count": len(row_ids_list),
            "timestamp": timestamp,
        }
    )
    return session


def with_completed_training_round(
    session_payload: Mapping[str, Any],
    *,
    training_dataset_id: str,
    training_artifact_id: str,
    ml_result: Mapping[str, Any],
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    timestamp = now()
    next_round = int(session.get("round", 0)) + 1

    session["round"] = next_round
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": "training_round_completed",
            "round": next_round,
            "training_dataset_id": str(training_dataset_id),
            "training_artifact_id": str(training_artifact_id),
            "ml_result": _json_safe_summary(ml_result),
            "timestamp": timestamp,
        }
    )
    return session


def labelled_training_items(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session = coerce_session(session_payload)
    labels = dict(session.get("labels") or {})

    items: List[Dict[str, Any]] = []
    for row_id in session.get("training_row_ids", []):
        entry = labels.get(str(row_id))
        if not entry:
            continue
        if normalise_label(entry.get("label")) == UNSURE_LABEL:
            continue
        items.append(dict(entry))

    return items


def label_rows(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session = coerce_session(session_payload)
    rows = [dict(entry) for entry in dict(session.get("labels") or {}).values()]
    rows.sort(key=lambda row: float(row.get("timestamp") or 0.0), reverse=True)
    return rows


def counts(session_payload: Mapping[str, Any]) -> Dict[str, int]:
    session = coerce_session(session_payload)
    label_entries = dict(session.get("labels") or {})
    unsure_count = sum(
        1
        for entry in label_entries.values()
        if normalise_label(entry.get("label")) == UNSURE_LABEL
    )
    training_count = len(labelled_training_items(session))
    ignored_count = len(_stable_unique(session.get("ignored_row_ids", [])))
    return {
        "labelled_or_verified": training_count,
        "unsure": unsure_count,
        "ignored": ignored_count,
        "total_reviewed": len(label_entries),
    }

def with_revision(
    session_payload: Mapping[str, Any],
    *,
    previous_session_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    previous = str(previous_session_artifact_id or "").strip()
    if previous:
        session["previous_session_artifact_id"] = previous
        session["revision"] = int(session.get("revision", 0) or 0) + 1
    session["updated_at"] = now()
    return session

def _stable_unique(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _json_safe_summary(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"traceback", "records", "payload"}:
                continue
            out[str(key)] = _json_safe_summary(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe_summary(item) for item in value[:20]]
    return str(value)