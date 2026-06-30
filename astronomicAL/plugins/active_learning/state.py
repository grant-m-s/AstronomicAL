from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ARTIFACT_SESSION = "al.session"
ARTIFACT_TRAINING_SET = "al.training_set"
ARTIFACT_BATCH = "ml.active_learning_batch"

UNSURE_LABEL = "__unsure__"
UNSURE_DISPLAY = "Unsure"

ROW_UNLABELLED = "unlabelled"
ROW_QUEUED = "queued"
ROW_VERIFIED = "verified"
ROW_UNSURE = "unsure"
ROW_TRAINING = "training"
ROW_DEFERRED = "deferred"
ROW_EXCLUDED = "excluded"

TERMINAL_QUERY_STATES = frozenset(
    {
        ROW_QUEUED,
        ROW_VERIFIED,
        ROW_UNSURE,
        ROW_TRAINING,
        ROW_DEFERRED,
        ROW_EXCLUDED,
    }
)

LATEST_REFERENCE_KEYS = (
    "training_dataset_id",
    "training_artifact_id",
    "run_artifact_id",
    "model_artifact_id",
    "split_spec_artifact_id",
    "evaluation_report_artifact_id",
    "predictions_artifact_id",
    "prediction_dataset_id",
    "acquisition_artifact_id",
)


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
        if not label or normalise_label(label) == UNSURE_LABEL:
            continue
        if label not in labels:
            labels.append(label)
    return labels


def normalise_label(label: Any) -> str:
    value = str(label or "").strip()
    if value.lower() in {
        "unsure",
        "uncertain",
        "unknown",
        "skip",
        UNSURE_LABEL.lower(),
    }:
        return UNSURE_LABEL
    return value


def display_label(label: Any) -> str:
    value = normalise_label(label)
    return UNSURE_DISPLAY if value == UNSURE_LABEL else value


def _blank_latest() -> Dict[str, Optional[str]]:
    return {key: None for key in LATEST_REFERENCE_KEYS}


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
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp = now()
    return {
        "schema_version": 3,
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
        "row_states": {},
        "latest": _blank_latest(),
        "last_batch": None,
        "history": [],
        "contract": copy.deepcopy(dict(contract or {})),
    }


def coerce_session(payload: Mapping[str, Any]) -> Dict[str, Any]:
    session = copy.deepcopy(dict(payload or {}))
    session.setdefault("schema_version", 3)
    session["schema_version"] = max(3, int(session.get("schema_version") or 0))
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
    session.setdefault("row_states", {})
    session.setdefault("last_batch", None)
    session.setdefault("history", [])
    session.setdefault("contract", {})

    latest = _blank_latest()
    latest.update(dict(session.get("latest") or {}))
    for key in LATEST_REFERENCE_KEYS:
        direct = session.get(key)
        if direct and not latest.get(key):
            latest[key] = str(direct)
    session["latest"] = latest

    # Derive lifecycle state for sessions written before schema v3.
    row_states = {str(k): str(v) for k, v in dict(session.get("row_states") or {}).items()}
    for row_id, entry in dict(session.get("labels") or {}).items():
        status = str((entry or {}).get("status") or "")
        row_states.setdefault(
            str(row_id),
            ROW_UNSURE if status == "unsure" else ROW_VERIFIED,
        )
    for row_id in session.get("training_row_ids") or []:
        row_states[str(row_id)] = ROW_TRAINING
    last_batch = session.get("last_batch") or {}
    if isinstance(last_batch, Mapping):
        for row_id in last_batch.get("row_ids") or []:
            row_states.setdefault(str(row_id), ROW_QUEUED)
    session["row_states"] = row_states
    return session


def with_contract(
    session_payload: Mapping[str, Any],
    *,
    contract: Optional[Mapping[str, Any]],
    event: str = "contract_updated",
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    timestamp = now()
    session["contract"] = copy.deepcopy(dict(contract or {}))
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": str(event or "contract_updated"),
            "recipe_id": (session.get("contract") or {}).get("recipe", {}).get("id"),
            "timestamp": timestamp,
        }
    )
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
    session["ignored_row_ids"] = _stable_unique(
        [*session.get("ignored_row_ids", []), row_id_str]
    )

    training = [
        str(item)
        for item in session.get("training_row_ids", [])
        if str(item) != row_id_str
    ]
    if not is_unsure:
        training.append(row_id_str)
    session["training_row_ids"] = _stable_unique(training)

    row_states = dict(session.get("row_states") or {})
    row_states[row_id_str] = ROW_UNSURE if is_unsure else ROW_VERIFIED
    session["row_states"] = row_states
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

    row_states = dict(session.get("row_states") or {})
    for row_id in row_ids_list:
        current = row_states.get(row_id, ROW_UNLABELLED)
        if current not in {ROW_VERIFIED, ROW_UNSURE, ROW_TRAINING, ROW_EXCLUDED}:
            row_states[row_id] = ROW_QUEUED
    session["row_states"] = row_states

    latest = dict(session.get("latest") or _blank_latest())
    latest["acquisition_artifact_id"] = str(batch_artifact_id)
    if predictions_artifact_id:
        latest["predictions_artifact_id"] = str(predictions_artifact_id)
    session["latest"] = latest

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

    latest = dict(session.get("latest") or _blank_latest())
    latest["training_dataset_id"] = str(training_dataset_id)
    latest["training_artifact_id"] = str(training_artifact_id)
    for key in (
        "run_artifact_id",
        "model_artifact_id",
        "split_spec_artifact_id",
        "evaluation_report_artifact_id",
        "predictions_artifact_id",
    ):
        value = _find_result_value(ml_result, key)
        if value:
            latest[key] = str(value)
    session["latest"] = latest

    row_states = dict(session.get("row_states") or {})
    for row_id in session.get("training_row_ids") or []:
        row_states[str(row_id)] = ROW_TRAINING
    session["row_states"] = row_states

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


def with_prediction_result(
    session_payload: Mapping[str, Any],
    *,
    prediction_result: Mapping[str, Any],
) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    timestamp = now()
    artifact_id = (
        _find_result_value(prediction_result, "predictions_artifact_id")
        or _find_result_value(prediction_result, "artifact_id")
    )
    derived_dataset_id = (
        _find_result_value(prediction_result, "derived_dataset_id")
        or _find_result_value(prediction_result, "prediction_dataset_id")
    )
    if not artifact_id:
        raise ValueError("core.ml.predict did not return a predictions artifact id.")

    latest = dict(session.get("latest") or _blank_latest())
    latest["predictions_artifact_id"] = str(artifact_id)
    if derived_dataset_id:
        latest["prediction_dataset_id"] = str(derived_dataset_id)
    session["latest"] = latest
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": "pool_prediction_completed",
            "predictions_artifact_id": str(artifact_id),
            "prediction_dataset_id": str(derived_dataset_id or ""),
            "count": _find_result_value(prediction_result, "count"),
            "timestamp": timestamp,
        }
    )
    return session


def latest_reference(session_payload: Mapping[str, Any], key: str) -> Optional[str]:
    session = coerce_session(session_payload)
    value = (session.get("latest") or {}).get(str(key))
    return str(value) if value not in (None, "") else None


def row_ids_in_states(
    session_payload: Mapping[str, Any],
    states: Sequence[str] = tuple(TERMINAL_QUERY_STATES),
) -> List[str]:
    wanted = {str(state) for state in states}
    session = coerce_session(session_payload)
    return [
        str(row_id)
        for row_id, state in dict(session.get("row_states") or {}).items()
        if str(state) in wanted
    ]


def labelled_training_items(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session = coerce_session(session_payload)
    labels = dict(session.get("labels") or {})
    items: List[Dict[str, Any]] = []
    for row_id in session.get("training_row_ids", []):
        entry = labels.get(str(row_id))
        if not entry or normalise_label(entry.get("label")) == UNSURE_LABEL:
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
    row_states = dict(session.get("row_states") or {})
    return {
        "labelled_or_verified": training_count,
        "unsure": unsure_count,
        "ignored": ignored_count,
        "queued": sum(1 for state in row_states.values() if state == ROW_QUEUED),
        "training": sum(1 for state in row_states.values() if state == ROW_TRAINING),
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


def _find_result_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        direct = value.get(key)
        if direct not in (None, ""):
            return direct
        for nested_key in ("summary", "result", "ml_result", "artifacts", "outputs"):
            nested = value.get(nested_key)
            found = _find_result_value(nested, key)
            if found not in (None, ""):
                return found
        for nested in value.values():
            found = _find_result_value(nested, key)
            if found not in (None, ""):
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found = _find_result_value(nested, key)
            if found not in (None, ""):
                return found
    return None


def _json_safe_summary(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"traceback", "records", "payload"}:
                continue
            output[str(key)] = _json_safe_summary(item)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_summary(item) for item in list(value)[:100]]
    return str(value)
