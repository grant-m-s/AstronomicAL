from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from . import state as al_state

def session_summary(session_payload: Mapping[str, Any]) -> Dict[str, Any]:
    session = al_state.coerce_session(session_payload)
    return {
        "session_id": session.get("session_id"),
        "dataset_id": session.get("dataset_id"),
        "round": session.get("round"),
        "counts": al_state.counts(session),
        "label_counts": al_state.label_counts(session),
        "latest": dict(session.get("latest") or {}),
        "last_batch": dict(session.get("last_batch") or {}),
    }

def label_history_rows(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return al_state.label_rows(session_payload)

def batch_rows(batch_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = batch_payload.get("records") or []
    return [dict(record) for record in records if isinstance(record, Mapping)]

def cumulative_label_counts(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = sorted(al_state.label_rows(session_payload), key=lambda row: float(row.get("timestamp") or 0.0))
    counter: Counter[str] = Counter()
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        label = al_state.display_label(row.get("label"))
        counter[label] += 1
        out.append({"index": idx, "row_id": row.get("row_id"), "label": label, **dict(counter)})
    return out

def performance_rows(session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return _performance_rows(session_payload, context=None)

def performance_rows_with_artifacts(context: Any, session_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return AL performance rows, falling back to referenced core.ml artifacts.

    Older AL histories often stored only artifact ids in ``ml_result``.  The core_ml
    training-curves panel can still show those runs because it reads the referenced
    training-log/report artifacts directly; this helper gives the AL performance
    tab the same behaviour.
    """

    return _performance_rows(session_payload, context=context)

def _performance_rows(session_payload: Mapping[str, Any], *, context: Any = None) -> List[Dict[str, Any]]:
    session = al_state.coerce_session(session_payload or {})
    out: List[Dict[str, Any]] = []
    for entry in session.get("history") or []:
        if not isinstance(entry, Mapping) or entry.get("event") != "training_round_completed":
            continue
        metrics = _entry_metrics(entry, context=context)
        round_index = int(entry.get("round") or 0)
        labelled_count = int(entry.get("labelled_count") or 0)
        if not labelled_count:
            labelled_count = len(entry.get("row_ids") or []) or int((entry.get("counts") or {}).get("labelled_or_verified") or 0)
        for metric, value in metrics.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            out.append(
                {
                    "round": round_index,
                    "labelled_count": labelled_count,
                    "metric": str(metric),
                    "value": numeric,
                    "training_dataset_id": entry.get("training_dataset_id"),
                    "training_artifact_id": entry.get("training_artifact_id"),
                    "timestamp": entry.get("timestamp"),
                }
            )
    if context is not None:
        out.extend(_latest_artifact_rows(context, session, existing_rows=out))
    return out

def _entry_metrics(entry: Mapping[str, Any], *, context: Any = None) -> Dict[str, float]:
    metrics = dict(entry.get("metrics") or {})
    if metrics:
        return metrics

    ml_result = entry.get("ml_result") or {}
    metrics = al_state.extract_metrics(ml_result)
    if metrics:
        return metrics

    if context is None:
        return {}

    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return {}

    merged: Dict[str, float] = {}
    for artifact_id in _artifact_ids_from_training_entry(entry):
        try:
            payload = artifacts.get(str(artifact_id))
        except Exception:
            continue
        for key, value in al_state.extract_metrics(payload).items():
            merged.setdefault(key, value)
    return merged

def _artifact_ids_from_training_entry(entry: Mapping[str, Any]) -> List[str]:
    ids: List[str] = list(al_state.artifact_ids(entry))

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_key, raw_value in value.items():
                key = str(raw_key).lower()
                if key.endswith("artifact_id") or key in {"artifact_id", "run_id", "log_id", "report_id"}:
                    if raw_value not in (None, "", {}, []):
                        ids.append(str(raw_value))
                else:
                    walk(raw_value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            for item in list(value)[:100]:
                walk(item)

    walk(entry)
    return list(dict.fromkeys(ids))

def _latest_artifact_rows(context: Any, session: Mapping[str, Any], *, existing_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create a performance row from latest core.ml artifacts when history lacks metrics."""

    existing_keys = {
        (int(row.get("round") or 0), int(row.get("labelled_count") or 0), str(row.get("metric") or ""))
        for row in existing_rows
    }
    existing_x_keys = {
        (int(row.get("labelled_count") or 0), str(row.get("metric") or ""))
        for row in existing_rows
    }
    round_index = int(session.get("round") or 0)
    labelled_count = len(al_state.labelled_training_items(session))
    if not round_index or not labelled_count:
        return []

    latest = dict(session.get("latest") or {})
    pseudo_entry: Dict[str, Any] = {"latest": latest}
    metrics = _entry_metrics(pseudo_entry, context=context)
    if not metrics:
        metrics = _metrics_from_recent_artifacts(context, session)
    rows: List[Dict[str, Any]] = []
    for metric, value in metrics.items():
        key = (round_index, labelled_count, str(metric))
        x_key = (labelled_count, str(metric))
        if key in existing_keys or x_key in existing_x_keys:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        rows.append(
            {
                "round": round_index,
                "labelled_count": labelled_count,
                "metric": str(metric),
                "value": numeric,
                "training_dataset_id": latest.get("training_dataset_id"),
                "training_artifact_id": latest.get("training_artifact_id"),
                "timestamp": session.get("updated_at"),
            }
        )
    return rows

def _metrics_from_recent_artifacts(context: Any, session: Mapping[str, Any]) -> Dict[str, float]:
    artifacts = getattr(context, "artifacts", None)
    find = getattr(artifacts, "find", None)
    get = getattr(artifacts, "get", None)
    if not callable(find) or not callable(get):
        return {}

    merged: Dict[str, float] = {}
    for artifact_type in ("ml.evaluation_report", "ml.run", "ml.training_log"):
        try:
            refs = find(type=artifact_type)
        except TypeError:
            try:
                refs = find(artifact_type)
            except Exception:
                refs = []
        except Exception:
            refs = []
        for ref in list(refs or [])[-20:]:
            artifact_id = getattr(ref, "artifact_id", None)
            if artifact_id is None and isinstance(ref, Mapping):
                artifact_id = ref.get("artifact_id") or ref.get("id")
            if not artifact_id:
                continue
            try:
                payload = get(str(artifact_id))
            except Exception:
                continue
            if not _artifact_matches_session(payload, session):
                continue
            for metric, value in al_state.extract_metrics(payload).items():
                merged.setdefault(metric, value)
    return merged

def _artifact_matches_session(payload: Any, session: Mapping[str, Any]) -> bool:
    if not isinstance(payload, Mapping):
        return True
    session_id = str(session.get("session_id") or "")
    if not session_id:
        return True
    text_values: List[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, value in item.items():
                if str(key).lower() in {"al_session_id", "session_id", "source_session_id", "origin_session_id"}:
                    text_values.append(str(value))
                elif isinstance(value, (Mapping, list, tuple)):
                    walk(value)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray)):
            for nested in list(item)[:100]:
                walk(nested)

    walk(payload)
    return not text_values or session_id in text_values
