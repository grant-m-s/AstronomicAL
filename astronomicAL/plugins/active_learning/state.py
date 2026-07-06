from __future__ import annotations

import copy
import time
import uuid
from collections import Counter
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
    "training_log_artifact_id",
    "run_artifact_id",
    "model_artifact_id",
    "split_spec_artifact_id",
    "evaluation_report_artifact_id",
    "training_predictions_artifact_id",
    "predictions_artifact_id",
    "prediction_dataset_id",
    "acquisition_artifact_id",
)


def now() -> float:
    return time.time()


def normalise_label(label: Any) -> str:
    value = str(label or "").strip()
    if value.lower() in {"unsure", "uncertain", "unknown", "skip", UNSURE_LABEL.lower()}:
        return UNSURE_LABEL
    return value


def display_label(label: Any) -> str:
    value = normalise_label(label)
    return UNSURE_DISPLAY if value == UNSURE_LABEL else value


def parse_label_options(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("\n", ",").split(",")]
    elif isinstance(value, Iterable):
        parts = [str(part).strip() for part in value]
    else:
        parts = [str(value).strip()]

    out: List[str] = []
    for label in parts:
        if not label or normalise_label(label) == UNSURE_LABEL:
            continue
        if label not in out:
            out.append(label)
    return out


def _blank_latest() -> Dict[str, Optional[str]]:
    return {key: None for key in LATEST_REFERENCE_KEYS}


def stable_unique(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def create_session(
    *,
    dataset_id: str,
    label_options: Iterable[str] = (),
    seed: int = 42,
    target_column: str = "al_label",
    session_id: Optional[str] = None,
    pool_dataset_id: Optional[str] = None,
    validation_dataset_id: Optional[str] = None,
    test_dataset_id: Optional[str] = None,
    recipe_id: Optional[str] = None,
    recipe_profile_id: Optional[str] = None,
    recipe_profile_name: Optional[str] = None,
    al_protocol: str = "review",
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp = now()
    return {
        "schema_version": 4,
        "session_id": session_id or f"al:{uuid.uuid4().hex[:12]}",
        "revision": 0,
        "previous_session_artifact_id": None,
        "dataset_id": str(dataset_id),
        "pool_dataset_id": str(pool_dataset_id or dataset_id),
        "validation_dataset_id": str(validation_dataset_id or ""),
        "test_dataset_id": str(test_dataset_id or ""),
        "recipe_id": str(recipe_id or ""),
        "recipe_profile_id": str(recipe_profile_id or ""),
        "recipe_profile_name": str(recipe_profile_name or ""),
        "al_protocol": str(al_protocol or "review"),
        "target_column": str(target_column or "al_label"),
        "label_options": parse_label_options(label_options),
        "seed": int(seed or 0),
        "round": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
        # Canonical human decisions.
        "labels": {},
        # Compatibility/fast lookup fields retained for existing panels/actions.
        "ignored_row_ids": [],
        "training_row_ids": [],
        "row_states": {},
        "latest": _blank_latest(),
        "last_batch": None,
        "history": [],
        # Optional recipe/data preflight metadata.  Not required for an AL session.
        "contract": copy.deepcopy(dict(contract or {})),
    }


def coerce_session(payload: Mapping[str, Any]) -> Dict[str, Any]:
    session = copy.deepcopy(dict(payload or {}))
    session.setdefault("schema_version", 4)
    session["schema_version"] = max(4, int(session.get("schema_version") or 0))
    session.setdefault("session_id", f"al:{uuid.uuid4().hex[:12]}")
    session.setdefault("revision", 0)
    session.setdefault("previous_session_artifact_id", None)
    session.setdefault("dataset_id", "")
    session.setdefault("pool_dataset_id", session.get("dataset_id", ""))
    session.setdefault("validation_dataset_id", "")
    session.setdefault("test_dataset_id", "")
    session.setdefault("recipe_id", "")
    session.setdefault("recipe_profile_id", "")
    session.setdefault("recipe_profile_name", "")
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
        if direct not in (None, "") and not latest.get(key):
            latest[key] = str(direct)
    session["latest"] = latest

    labels = {str(k): dict(v or {}) for k, v in dict(session.get("labels") or {}).items()}
    session["labels"] = labels

    row_states = {str(k): str(v) for k, v in dict(session.get("row_states") or {}).items()}
    for row_id, entry in labels.items():
        status = str(entry.get("status") or "")
        row_states.setdefault(row_id, ROW_UNSURE if status == "unsure" else ROW_VERIFIED)
    for row_id in session.get("training_row_ids") or []:
        if str(row_id) in labels and normalise_label(labels[str(row_id)].get("label")) != UNSURE_LABEL:
            row_states[str(row_id)] = ROW_TRAINING
    last_batch = session.get("last_batch") or {}
    if isinstance(last_batch, Mapping):
        for row_id in last_batch.get("row_ids") or []:
            row_states.setdefault(str(row_id), ROW_QUEUED)
    session["row_states"] = row_states

    ignored = stable_unique(session.get("ignored_row_ids") or [])
    for row_id, entry in labels.items():
        if normalise_label(entry.get("label")) == UNSURE_LABEL:
            ignored.append(row_id)
    session["ignored_row_ids"] = stable_unique(ignored)
    session["training_row_ids"] = stable_unique(
        row_id
        for row_id, entry in labels.items()
        if normalise_label(entry.get("label")) != UNSURE_LABEL
        and str(entry.get("status") or "verified") in {"verified", "training"}
    )
    return session


def with_revision(session_payload: Mapping[str, Any], *, previous_session_artifact_id: Optional[str] = None) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    previous = str(previous_session_artifact_id or "").strip()
    if previous:
        session["previous_session_artifact_id"] = previous
        session["revision"] = int(session.get("revision", 0) or 0) + 1
    session["updated_at"] = now()
    return session


def with_contract(session_payload: Mapping[str, Any], *, contract: Optional[Mapping[str, Any]], event: str = "contract_updated") -> Dict[str, Any]:
    session = coerce_session(session_payload)
    timestamp = now()
    session["contract"] = copy.deepcopy(dict(contract or {}))
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": str(event or "contract_updated"),
            "recipe_id": (session.get("contract") or {}).get("recipe", {}).get("id"),
            "recipe_profile_id": (session.get("contract") or {}).get("recipe", {}).get("profile_id"),
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
    row_id_str = str(row_id or "").strip()
    label_value = normalise_label(label)
    if not row_id_str:
        raise ValueError("row_id is required.")
    if not label_value:
        raise ValueError("label is required.")

    timestamp = now()
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

    ignored = [row_id for row_id in session.get("ignored_row_ids", []) if str(row_id) != row_id_str]
    training = [row_id for row_id in session.get("training_row_ids", []) if str(row_id) != row_id_str]
    if is_unsure:
        ignored.append(row_id_str)
    else:
        training.append(row_id_str)
    session["ignored_row_ids"] = stable_unique(ignored)
    session["training_row_ids"] = stable_unique(training)

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
    row_ids_list = stable_unique(row_ids)
    timestamp = now()
    session["last_batch"] = {
        "kind": str(kind),
        "batch_artifact_id": str(batch_artifact_id),
        "predictions_artifact_id": str(predictions_artifact_id or ""),
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


def clear_queued_rows(
    session_payload: Mapping[str, Any],
    *,
    reason: str = "training_started",
) -> Dict[str, Any]:
    """Invalidate queued query rows and return them to the available pool.

    Query batches are tied to the predictions/model that produced their scores.
    Once a new training round begins those scores are stale, so rows that were
    merely queued should stop being excluded from future acquisition.  Verified,
    unsure, training, deferred, and explicitly excluded rows are left untouched.
    """

    session = coerce_session(session_payload)
    row_states = dict(session.get("row_states") or {})
    changed = False
    for row_id, state in list(row_states.items()):
        if str(state) == ROW_QUEUED:
            row_states.pop(row_id, None)
            changed = True

    last_batch = dict(session.get("last_batch") or {}) if isinstance(session.get("last_batch"), Mapping) else {}
    if last_batch:
        session["last_batch"] = None
        changed = True

    latest = dict(session.get("latest") or _blank_latest())
    for key in ("acquisition_artifact_id", "predictions_artifact_id", "prediction_dataset_id"):
        if latest.get(key):
            latest[key] = None
            changed = True
    session["latest"] = latest
    session["row_states"] = row_states

    if changed:
        timestamp = now()
        session["updated_at"] = timestamp
        session.setdefault("history", []).append(
            {
                "event": "queued_rows_invalidated",
                "reason": str(reason or "training_started"),
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
    session["round"] = int(session.get("round", 0) or 0) + 1
    latest = dict(session.get("latest") or _blank_latest())
    latest["training_dataset_id"] = str(training_dataset_id)
    latest["training_artifact_id"] = str(training_artifact_id)
    for key in (
        "training_log_artifact_id",
        "run_artifact_id",
        "model_artifact_id",
        "split_spec_artifact_id",
        "evaluation_report_artifact_id",
    ):
        value = find_nested_value(ml_result, key)
        if value not in (None, ""):
            latest[key] = str(value)

    # core.ml recipe runs may emit predictions on the materialised AL training
    # dataset.  Those are useful provenance, but they are not valid acquisition
    # inputs for the AL pool.  Keep them separate so the Query tab never picks a
    # training-dataset prediction artifact by accident.
    training_predictions = find_nested_value(ml_result, "predictions_artifact_id")
    if training_predictions not in (None, ""):
        latest["training_predictions_artifact_id"] = str(training_predictions)
        latest["predictions_artifact_id"] = None
        latest["prediction_dataset_id"] = None
    session["latest"] = latest

    row_states = dict(session.get("row_states") or {})
    for row_id in session.get("training_row_ids") or []:
        row_states[str(row_id)] = ROW_TRAINING
    session["row_states"] = row_states
    session["updated_at"] = timestamp
    session.setdefault("history", []).append(
        {
            "event": "training_round_completed",
            "round": session["round"],
            "training_dataset_id": str(training_dataset_id),
            "training_artifact_id": str(training_artifact_id),
            "ml_result": json_safe_summary(ml_result),
            "metrics": extract_metrics(ml_result),
            "labelled_count": len(labelled_training_items(session)),
            "timestamp": timestamp,
        }
    )
    return session


def with_prediction_result(session_payload: Mapping[str, Any], *, prediction_result: Mapping[str, Any]) -> Dict[str, Any]:
    session = coerce_session(session_payload)
    artifact_id = find_nested_value(prediction_result, "predictions_artifact_id") or find_nested_value(prediction_result, "artifact_id")
    if not artifact_id:
        raise ValueError("core.ml.predict did not return a predictions artifact id.")
    derived_dataset_id = find_nested_value(prediction_result, "derived_dataset_id") or find_nested_value(prediction_result, "prediction_dataset_id")
    latest = dict(session.get("latest") or _blank_latest())
    latest["predictions_artifact_id"] = str(artifact_id)
    if derived_dataset_id:
        latest["prediction_dataset_id"] = str(derived_dataset_id)
    session["latest"] = latest
    session["updated_at"] = now()
    session.setdefault("history", []).append(
        {
            "event": "pool_prediction_completed",
            "predictions_artifact_id": str(artifact_id),
            "prediction_dataset_id": str(derived_dataset_id or ""),
            "count": find_nested_value(prediction_result, "count"),
            "timestamp": session["updated_at"],
        }
    )
    return session


def latest_reference(session_payload: Mapping[str, Any], key: str) -> Optional[str]:
    value = (coerce_session(session_payload).get("latest") or {}).get(str(key))
    return str(value) if value not in (None, "") else None


def row_ids_in_states(session_payload: Mapping[str, Any], states: Sequence[str] = tuple(TERMINAL_QUERY_STATES)) -> List[str]:
    wanted = {str(state) for state in states}
    session = coerce_session(session_payload)
    return [str(row_id) for row_id, state in dict(session.get("row_states") or {}).items() if str(state) in wanted]


def labelled_training_items(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session = coerce_session(session_payload)
    labels = dict(session.get("labels") or {})
    out: List[Dict[str, Any]] = []
    for row_id in session.get("training_row_ids") or []:
        entry = labels.get(str(row_id))
        if not entry or normalise_label(entry.get("label")) == UNSURE_LABEL:
            continue
        out.append(dict(entry))
    return out


def label_rows(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(entry) for entry in dict(coerce_session(session_payload).get("labels") or {}).values()]
    rows.sort(key=lambda row: float(row.get("timestamp") or 0.0), reverse=True)
    return rows


def counts(session_payload: Mapping[str, Any]) -> Dict[str, int]:
    session = coerce_session(session_payload)
    labels = dict(session.get("labels") or {})
    unsure_count = sum(1 for entry in labels.values() if normalise_label(entry.get("label")) == UNSURE_LABEL)
    training_items = labelled_training_items(session)
    current_round = int(session.get("round", 0) or 0)
    labelled_since_last_train = sum(
        1
        for item in training_items
        if int(item.get("round", 0) or 0) >= current_round
    )
    row_states = dict(session.get("row_states") or {})
    return {
        "labelled_or_verified": len(training_items),
        "labelled_since_last_train": labelled_since_last_train,
        "unsure": unsure_count,
        "ignored": len(stable_unique(session.get("ignored_row_ids") or [])),
        "queued": sum(1 for state in row_states.values() if state == ROW_QUEUED),
        "training": sum(1 for state in row_states.values() if state == ROW_TRAINING),
        "total_reviewed": len(labels),
    }


def label_counts(session_payload: Mapping[str, Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in labelled_training_items(session_payload):
        counter[str(item.get("label") or "")] += 1
    return dict(counter)


def find_nested_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        direct = value.get(key)
        if direct not in (None, ""):
            return direct
        for nested_key in ("summary", "result", "ml_result", "artifacts", "outputs"):
            found = find_nested_value(value.get(nested_key), key)
            if found not in (None, ""):
                return found
        for nested in value.values():
            found = find_nested_value(nested, key)
            if found not in (None, ""):
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found = find_nested_value(nested, key)
            if found not in (None, ""):
                return found
    return None


def extract_metrics(value: Any) -> Dict[str, float]:
    """Best-effort extraction of scalar performance metrics from nested ML results."""

    metric_like_keys = {
        "metrics",
        "metric",
        "scores",
        "score",
        "evaluation",
        "evaluation_metrics",
        "eval_metrics",
        "test_metrics",
        "validation_metrics",
        "val_metrics",
        "history",
    }
    ignored_keys = {
        "seed",
        "round",
        "epoch",
        "step",
        "count",
        "n",
        "k",
        "timestamp",
        "schema_version",
    }
    out: Dict[str, float] = {}

    def walk(item: Any, prefix: str = "", in_metric_context: bool = False) -> None:
        if isinstance(item, Mapping):
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                key_lower = key.lower()
                child_prefix = f"{prefix}.{key}" if prefix else key
                metric_context = in_metric_context or key_lower in metric_like_keys or key_lower.endswith("_metrics")
                if isinstance(raw_value, (int, float, bool)) and not isinstance(raw_value, bool):
                    if metric_context or looks_like_metric_name(key_lower):
                        if key_lower not in ignored_keys:
                            metric_name = key if in_metric_context else child_prefix
                            out[metric_name] = float(raw_value)
                else:
                    walk(raw_value, child_prefix, metric_context)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            # Prefer the latest scalar values in a metric history, but avoid generating
            # hundreds of per-epoch AL plot options.
            values = list(item)
            if values and all(isinstance(entry, Mapping) for entry in values):
                for entry in reversed(values):
                    before = len(out)
                    walk(entry, prefix, in_metric_context)
                    if len(out) > before:
                        break
            elif values and (in_metric_context or looks_like_metric_name(prefix.lower())):
                for raw_value in reversed(values):
                    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                        metric_name = prefix.split(".")[-1] if in_metric_context else prefix
                        if metric_name:
                            out[metric_name] = float(raw_value)
                        break

    walk(value)
    return {key: val for key, val in sorted(out.items())}


def artifact_ids(value: Any) -> List[str]:
    """Best-effort extraction of artifact ids from nested action/event payloads."""

    ids: List[str] = []
    artifact_type_keys = {"type", "artifact_type", "kind"}
    interesting_types = {
        "ml.evaluation_report",
        "ml.training_log",
        "ml.run",
        "ml.model",
        "ml.predictions",
        "al.training_set",
    }

    def append(raw: Any) -> None:
        if raw not in (None, "", {}, []):
            ids.append(str(raw))

    def walk(item: Any, parent_key: str = "") -> None:
        if isinstance(item, Mapping):
            lowered = {str(key).lower(): val for key, val in item.items()}
            item_type = str(
                lowered.get("type")
                or lowered.get("artifact_type")
                or lowered.get("kind")
                or ""
            )
            if item_type in interesting_types or item_type.startswith("ml."):
                for key in ("artifact_id", "id", "ref", "artifact_ref"):
                    append(lowered.get(key))
            for raw_key, raw_value in item.items():
                key = str(raw_key).lower()
                if key.endswith("artifact_id") or key in {"artifact_id", "artifact_ref"}:
                    append(raw_value)
                    continue
                if key in {"id", "ref"} and parent_key in artifact_type_keys:
                    append(raw_value)
                    continue
                walk(raw_value, key)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in list(item)[:500]:
                walk(nested, parent_key)

    walk(value)
    return stable_unique(ids)


def looks_like_metric_name(key: str) -> bool:
    return any(
        token in key
        for token in (
            "accuracy",
            "acc",
            "auc",
            "f1",
            "precision",
            "recall",
            "loss",
            "mae",
            "mse",
            "rmse",
            "r2",
            "score",
            "metric",
            "balanced",
            "logloss",
            "roc",
        )
    )


def json_safe_summary(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"traceback", "records", "payload"}:
                continue
            out[str(key)] = json_safe_summary(item)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe_summary(item) for item in list(value)[:100]]
    return str(value)
