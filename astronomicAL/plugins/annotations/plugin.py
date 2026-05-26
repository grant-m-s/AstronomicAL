# BUG: Annotation panel reloads labels but summary doesnt

from __future__ import annotations

import html
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ArtifactResult


ANNOTATION_NOTE_TYPE = "annotation.note"
REVIEW_STATUS_TYPE = "review.status"
SUMMARY_TABLE_TYPE = "table.annotations_summary"
SUMMARY_CSV_TYPE = "annotation.summary.csv"


manifest = PluginManifest(
    id="core.annotations",
    name="Annotations",
    version="0.2.0",
    description=(
        "Record-level notes, review state, tags, confidence, and optional "
        "label suggestions for human-in-the-loop analysis workflows."
    ),
    capabilities=[
        "panel",
        "action",
        "datasets",
        "selection",
        "events",
        "artifacts",
        "persistence",
    ],
    tags=[
        "core",
        "annotations",
        "notes",
        "review",
        "labels",
        "human-in-the-loop",
    ],
)


def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Annotations",
        factory=create_annotations_panel,
        description=(
            "Attach notes and review decisions to the currently focused record. "
            "Works with any dataset that has a mapped record_id."
        ),
        category="Core",
        icon="edit",
        tags=["core", "annotations", "review", "notes"],
        required_mappings=["record_id"],
        optional_mappings=["target_label"],
        produces=[
            "annotation.created",
            "review.status.changed",
            "labels.suggested",
            "artifact.created",
            ANNOTATION_NOTE_TYPE,
            REVIEW_STATUS_TYPE,
        ],
        default_layout={"x": 8, "y": 0, "w": 4, "h": 7},
        state_version=2,
        persist_layout=True,
        persist_state=True,
    )

    api.register_panel(
        id="summary",
        title="Annotation Summary",
        factory=create_annotation_summary_panel,
        description=(
            "Summarise all annotation.note and review.status artifacts for the "
            "active dataset, with options to create a summary dataset or CSV artifact."
        ),
        category="Core",
        icon="table",
        tags=["core", "annotations", "summary", "export"],
        produces=[
            "artifact.created",
            "dataset.loaded",
            SUMMARY_TABLE_TYPE,
            SUMMARY_CSV_TYPE,
        ],
        default_layout={"x": 8, "y": 7, "w": 4, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
    )

    api.register_action(
        id="build_summary",
        title="Build Annotation Summary",
        handler=build_annotation_summary_action,
        outputs=[SUMMARY_TABLE_TYPE],
        run_in_job=False,
        description=(
            "Build a table summarising all annotation.note and review.status "
            "artifacts for the active dataset."
        ),
        category="Core",
        icon="table",
        tags=["core", "annotations", "summary", "export"],
    )


# ---------------------------------------------------------------------
# Panel factories
# ---------------------------------------------------------------------


def create_annotations_panel(
    context,
    *,
    instance_id: Optional[str] = None,
    restore_state: Optional[Dict[str, Any]] = None,
    restore_metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    controller = AnnotationsPanel(
        context=context,
        instance_id=instance_id,
        restore_state=restore_state,
        restore_metadata=restore_metadata,
        **kwargs,
    )
    return controller.panel(), controller


def create_annotation_summary_panel(
    context,
    *,
    instance_id: Optional[str] = None,
    restore_state: Optional[Dict[str, Any]] = None,
    restore_metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    controller = AnnotationSummaryPanel(
        context=context,
        instance_id=instance_id,
        restore_state=restore_state,
        restore_metadata=restore_metadata,
        **kwargs,
    )
    return controller.panel(), controller


# ---------------------------------------------------------------------
# Action handler
# ---------------------------------------------------------------------


def build_annotation_summary_action(
    context,
    request=None,
    cancel_token=None,
):
    if cancel_token is not None and cancel_token.cancelled():
        return None

    dataset_id = getattr(request, "dataset_id", None) or _active_dataset_id(context)
    df = build_annotation_summary_dataframe(context, dataset_id=dataset_id)

    if cancel_token is not None and cancel_token.cancelled():
        return None

    return ArtifactResult(
        type=SUMMARY_TABLE_TYPE,
        payload=df.to_dict(orient="records"),
        dataset_id=dataset_id,
        row_ids=None,
        params={
            "source": "core.annotations",
            "row_count": int(len(df)),
            "created_at": _now(),
        },
        publish=True,
    )


# ---------------------------------------------------------------------
# Shared artifact/history helpers
# ---------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _active_dataset_id(context) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None
    try:
        return datasets.active_id()
    except Exception:
        return None


def _artifact_created_at(ref: Any) -> float:
    try:
        return float(getattr(ref, "created_at", 0.0) or 0.0)
    except Exception:
        return 0.0


def _artifact_row_id(ref: Any) -> Optional[str]:
    try:
        row_ids = getattr(ref, "row_ids", None)
    except Exception:
        row_ids = None
    if row_ids:
        return str(row_ids[0])
    return None


def _safe_artifact_payload(context, artifact_id: str) -> Any:
    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return None
    try:
        return artifacts.get(artifact_id)
    except Exception:
        return None


def _find_artifacts(
    context,
    *,
    artifact_type: str,
    dataset_id: Optional[str] = None,
    row_id: Optional[str] = None,
) -> List[Any]:
    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return []

    try:
        return list(
            artifacts.find(
                type=artifact_type,
                dataset_id=dataset_id,
                row_id=row_id,
            )
        )
    except TypeError:
        try:
            return list(
                artifacts.find(
                    artifact_type=artifact_type,
                    dataset_id=dataset_id,
                    row_id=row_id,
                )
            )
        except Exception:
            return []
    except Exception:
        return []


def _iter_artifact_payloads(
    context,
    *,
    artifact_type: str,
    dataset_id: Optional[str] = None,
    row_id: Optional[str] = None,
):
    refs = _find_artifacts(
        context,
        artifact_type=artifact_type,
        dataset_id=dataset_id,
        row_id=row_id,
    )

    for ref in refs:
        artifact_id = getattr(ref, "artifact_id", None)
        if not artifact_id:
            continue

        payload = _safe_artifact_payload(context, artifact_id)
        if not isinstance(payload, dict):
            continue

        yield ref, payload


def _record_key(dataset_id: Optional[str], row_id: Optional[str]) -> Optional[str]:
    if dataset_id is None or row_id is None:
        return None
    return f"{dataset_id}::{row_id}"


def _default_record_state(
    *,
    dataset_id: Optional[str] = None,
    row_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "row_id": row_id,
        "status": "unreviewed",
        "confidence": 0.0,
        "tags": [],
        "label_suggestion": "",
        "notes": [],
        "reviews": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def _normalise_note_payload(ref: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    artifact_id = getattr(ref, "artifact_id", None)
    dataset_id = payload.get("dataset_id") or getattr(ref, "dataset_id", None)
    row_id = payload.get("row_id") or _artifact_row_id(ref)

    created_at = payload.get("created_at") or payload.get("updated_at") or _now()
    updated_at = payload.get("updated_at") or created_at

    annotation_id = (
        payload.get("annotation_id")
        or payload.get("id")
        or f"annotation:{artifact_id or uuid.uuid4().hex}"
    )

    tags = payload.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

    return {
        "annotation_id": str(annotation_id),
        "artifact_id": artifact_id,
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "row_id": str(row_id) if row_id is not None else None,
        "text": str(payload.get("text", "") or ""),
        "status": str(payload.get("status", "unreviewed") or "unreviewed"),
        "confidence": _safe_float(payload.get("confidence", 0.0), default=0.0),
        "tags": list(tags),
        "label_suggestion": str(payload.get("label_suggestion", "") or ""),
        "created_at": str(created_at),
        "updated_at": str(updated_at),
        "source": str(payload.get("source", "core.annotations") or "core.annotations"),
        "instance_id": payload.get("instance_id"),
        "_sort_time": _artifact_created_at(ref),
    }


def _normalise_review_payload(ref: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    artifact_id = getattr(ref, "artifact_id", None)
    dataset_id = payload.get("dataset_id") or getattr(ref, "dataset_id", None)
    row_id = payload.get("row_id") or _artifact_row_id(ref)

    updated_at = payload.get("updated_at") or payload.get("created_at") or _now()

    review_id = (
        payload.get("review_id")
        or payload.get("id")
        or f"review:{artifact_id or uuid.uuid4().hex}"
    )

    tags = payload.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

    return {
        "review_id": str(review_id),
        "artifact_id": artifact_id,
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "row_id": str(row_id) if row_id is not None else None,
        "status": str(payload.get("status", "unreviewed") or "unreviewed"),
        "confidence": _safe_float(payload.get("confidence", 0.0), default=0.0),
        "tags": list(tags),
        "label_suggestion": str(payload.get("label_suggestion", "") or ""),
        "created_at": str(payload.get("created_at", updated_at) or updated_at),
        "updated_at": str(updated_at),
        "source": str(payload.get("source", "core.annotations") or "core.annotations"),
        "instance_id": payload.get("instance_id"),
        "_sort_time": _artifact_created_at(ref),
    }


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _dedupe_items(items: Iterable[Dict[str, Any]], id_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for item in items:
        identity = None
        for key in id_keys:
            if item.get(key):
                identity = (key, str(item.get(key)))
                break

        if identity is None:
            identity = (
                "content",
                str(item.get("dataset_id")),
                str(item.get("row_id")),
                str(item.get("created_at") or item.get("updated_at")),
                str(item.get("text", "")),
                str(item.get("status", "")),
            )

        if identity in seen:
            continue

        seen.add(identity)
        out.append(dict(item))

    out.sort(
        key=lambda item: (
            _safe_float(item.get("_sort_time", 0.0), default=0.0),
            str(item.get("created_at") or item.get("updated_at") or ""),
        )
    )
    return out


def _apply_latest_state_from_history(state: Dict[str, Any]) -> Dict[str, Any]:
    reviews = state.get("reviews") or []
    notes = state.get("notes") or []

    latest = None
    if reviews:
        latest = reviews[-1]
    elif notes:
        latest = notes[-1]

    if latest:
        state["status"] = latest.get("status", "unreviewed")
        state["confidence"] = _safe_float(latest.get("confidence", 0.0), default=0.0)
        state["tags"] = list(latest.get("tags") or [])
        state["label_suggestion"] = str(latest.get("label_suggestion", "") or "")
        state["updated_at"] = str(
            latest.get("updated_at") or latest.get("created_at") or state.get("updated_at") or _now()
        )

    if notes:
        state["created_at"] = str(notes[0].get("created_at") or state.get("created_at") or _now())
    elif reviews:
        state["created_at"] = str(
            reviews[0].get("created_at") or reviews[0].get("updated_at") or state.get("created_at") or _now()
        )

    return state


def _merge_record_states(
    old_state: Optional[Dict[str, Any]],
    artifact_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if old_state is None and artifact_state is None:
        return _default_record_state()

    if old_state is None:
        return _apply_latest_state_from_history(dict(artifact_state or {}))

    if artifact_state is None:
        return _apply_latest_state_from_history(dict(old_state or {}))

    merged = dict(old_state)

    merged["dataset_id"] = artifact_state.get("dataset_id") or old_state.get("dataset_id")
    merged["row_id"] = artifact_state.get("row_id") or old_state.get("row_id")

    merged["notes"] = _dedupe_items(
        list(old_state.get("notes") or []) + list(artifact_state.get("notes") or []),
        ("annotation_id", "artifact_id"),
    )
    merged["reviews"] = _dedupe_items(
        list(old_state.get("reviews") or []) + list(artifact_state.get("reviews") or []),
        ("review_id", "artifact_id"),
    )

    return _apply_latest_state_from_history(merged)


def build_record_state_from_artifacts(
    context,
    *,
    dataset_id: Optional[str],
    row_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if dataset_id is None or row_id is None:
        return None

    state = _default_record_state(dataset_id=str(dataset_id), row_id=str(row_id))

    for ref, payload in _iter_artifact_payloads(
        context,
        artifact_type=ANNOTATION_NOTE_TYPE,
        dataset_id=str(dataset_id),
        row_id=str(row_id),
    ):
        note = _normalise_note_payload(ref, payload)
        if note.get("dataset_id") == str(dataset_id) and note.get("row_id") == str(row_id):
            state["notes"].append(note)

    for ref, payload in _iter_artifact_payloads(
        context,
        artifact_type=REVIEW_STATUS_TYPE,
        dataset_id=str(dataset_id),
        row_id=str(row_id),
    ):
        review = _normalise_review_payload(ref, payload)
        if review.get("dataset_id") == str(dataset_id) and review.get("row_id") == str(row_id):
            state["reviews"].append(review)

    state["notes"] = _dedupe_items(state["notes"], ("annotation_id", "artifact_id"))
    state["reviews"] = _dedupe_items(state["reviews"], ("review_id", "artifact_id"))

    if not state["notes"] and not state["reviews"]:
        return None

    return _apply_latest_state_from_history(state)


def build_all_record_states_from_artifacts(
    context,
    *,
    dataset_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}

    for ref, payload in _iter_artifact_payloads(
        context,
        artifact_type=ANNOTATION_NOTE_TYPE,
        dataset_id=dataset_id,
        row_id=None,
    ):
        note = _normalise_note_payload(ref, payload)
        ds = note.get("dataset_id")
        row = note.get("row_id")
        key = _record_key(ds, row)
        if key is None:
            continue

        states.setdefault(
            key,
            _default_record_state(dataset_id=ds, row_id=row),
        )
        states[key]["notes"].append(note)

    for ref, payload in _iter_artifact_payloads(
        context,
        artifact_type=REVIEW_STATUS_TYPE,
        dataset_id=dataset_id,
        row_id=None,
    ):
        review = _normalise_review_payload(ref, payload)
        ds = review.get("dataset_id")
        row = review.get("row_id")
        key = _record_key(ds, row)
        if key is None:
            continue

        states.setdefault(
            key,
            _default_record_state(dataset_id=ds, row_id=row),
        )
        states[key]["reviews"].append(review)

    for key, state in list(states.items()):
        state["notes"] = _dedupe_items(state.get("notes") or [], ("annotation_id", "artifact_id"))
        state["reviews"] = _dedupe_items(state.get("reviews") or [], ("review_id", "artifact_id"))
        states[key] = _apply_latest_state_from_history(state)

    return states


def build_annotation_summary_dataframe(
    context,
    *,
    dataset_id: Optional[str] = None,
) -> pd.DataFrame:
    states = build_all_record_states_from_artifacts(context, dataset_id=dataset_id)

    rows: List[Dict[str, Any]] = []
    for state in states.values():
        notes = state.get("notes") or []
        reviews = state.get("reviews") or []

        latest_note = notes[-1] if notes else {}
        latest_review = reviews[-1] if reviews else {}

        rows.append(
            {
                "dataset_id": state.get("dataset_id"),
                "row_id": state.get("row_id"),
                "latest_status": state.get("status", "unreviewed"),
                "latest_confidence": state.get("confidence", 0.0),
                "tags": ", ".join(str(tag) for tag in (state.get("tags") or [])),
                "label_suggestion": state.get("label_suggestion", ""),
                "note_count": len(notes),
                "review_count": len(reviews),
                "latest_note": latest_note.get("text", ""),
                "latest_note_at": latest_note.get("created_at", ""),
                "latest_review_at": latest_review.get("updated_at", ""),
                "first_annotated_at": state.get("created_at", ""),
                "updated_at": state.get("updated_at", ""),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "row_id",
                "latest_status",
                "latest_confidence",
                "tags",
                "label_suggestion",
                "note_count",
                "review_count",
                "latest_note",
                "latest_note_at",
                "latest_review_at",
                "first_annotated_at",
                "updated_at",
            ]
        )

    return df.sort_values(
        by=["dataset_id", "updated_at", "row_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main Annotations panel
# ---------------------------------------------------------------------


class AnnotationsPanel:
    REVIEW_STATUSES = [
        "unreviewed",
        "in_review",
        "approved",
        "rejected",
        "unsure",
        "needs_follow_up",
    ]

    def __init__(
        self,
        *,
        context,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> None:
        self.context = context
        self.instance_id = instance_id or f"core.annotations:{uuid.uuid4().hex[:10]}"
        self.restore_metadata = restore_metadata or {}

        self._subscriptions: List[Any] = []
        self._watchers: List[Any] = []
        self._disposed = False

        self.dataset_id: Optional[str] = None
        self.row_id: Optional[str] = None
        self.row: Optional[pd.Series] = None

        self._columns_cache: Dict[str, List[str]] = {}

        # Fallback/local cache keyed by "dataset_id::row_id".
        # Artifacts are the canonical shared store; this helps persistence and
        # protects notes during transitional states.
        self._records: Dict[str, Dict[str, Any]] = {}

        self._build_widgets()
        self._build_root()
        self._subscribe_events()

        if restore_state:
            self.restore_state(restore_state)

        self._sync_from_current_focus()
        self._render()

    # ------------------------------------------------------------------
    # Public lifecycle / persistence API
    # ------------------------------------------------------------------

    def panel(self):
        return self.root

    def dispose(self) -> None:
        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

        for watcher in list(self._watchers):
            try:
                watcher.obj.param.unwatch(watcher)
            except Exception:
                try:
                    watcher.inst.param.unwatch(watcher)
                except Exception:
                    pass
        self._watchers.clear()

    def get_state(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "records": self._records,
            "current_dataset_id": self.dataset_id,
            "current_row_id": self.row_id,
            "draft": {
                "note": self.note_input.value,
                "tags": self.tags_input.value,
                "label_suggestion": self.label_suggestion_input.value,
                "status": self.status_select.value,
                "confidence": self.confidence_slider.value,
            },
            "show_history": self.show_history_checkbox.value,
        }

    def snapshot_state(self) -> Dict[str, Any]:
        return self.get_state()

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        records = state.get("records")
        if isinstance(records, dict):
            for key, value in records.items():
                if isinstance(value, dict):
                    self._records[str(key)] = value

        draft = state.get("draft")
        if isinstance(draft, dict):
            self.note_input.value = str(draft.get("note", "") or "")
            self.tags_input.value = str(draft.get("tags", "") or "")
            self.label_suggestion_input.value = str(
                draft.get("label_suggestion", "") or ""
            )

            status = draft.get("status")
            if status in self.REVIEW_STATUSES:
                self.status_select.value = status

            self.confidence_slider.value = max(
                0.0,
                min(1.0, _safe_float(draft.get("confidence", 0.0), default=0.0)),
            )

        if "show_history" in state:
            self.show_history_checkbox.value = bool(state.get("show_history"))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.status_select = pn.widgets.Select(
            name="Review status",
            options=list(self.REVIEW_STATUSES),
            value="unreviewed",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.confidence_slider = pn.widgets.FloatSlider(
            name="Confidence",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.0,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.tags_input = pn.widgets.TextInput(
            name="Tags",
            placeholder="comma-separated tags, e.g. ambiguous, follow-up",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.label_suggestion_input = pn.widgets.TextInput(
            name="Suggested label",
            placeholder="optional label suggestion",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.note_input = pn.widgets.TextAreaInput(
            name="Note",
            placeholder="Write a note for the focused record...",
            rows=5,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.save_note_button = pn.widgets.Button(
            name="Save note",
            button_type="primary",
            width=110,
            height=32,
            margin=(0, 8, 8, 0),
        )
        self.save_review_button = pn.widgets.Button(
            name="Save review state",
            button_type="success",
            width=150,
            height=32,
            margin=(0, 8, 8, 0),
        )
        self.reload_button = pn.widgets.Button(
            name="Reload from artifacts",
            button_type="default",
            width=155,
            height=32,
            margin=(0, 8, 8, 0),
        )
        self.clear_draft_button = pn.widgets.Button(
            name="Clear draft",
            button_type="default",
            width=110,
            height=32,
            margin=(0, 0, 8, 0),
        )

        self.show_history_checkbox = pn.widgets.Checkbox(
            name="Show history",
            value=True,
            margin=(0, 0, 8, 0),
        )

        self.save_note_button.on_click(self._on_save_note)
        self.save_review_button.on_click(self._on_save_review)
        self.reload_button.on_click(self._on_reload_from_artifacts)
        self.clear_draft_button.on_click(self._on_clear_draft)

        self._watchers.append(
            self.show_history_checkbox.param.watch(self._on_show_history_changed, "value")
        )

    def _build_root(self) -> None:
        self.header_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.record_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.history_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.message_pane = pn.pane.Alert(
            "",
            alert_type="info",
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.root = pn.Column(
            self.header_pane,
            self.message_pane,
            self._card("Focused record", self.record_pane),
            self._card(
                "Review state",
                self.status_select,
                self.confidence_slider,
                self.tags_input,
                self.label_suggestion_input,
                pn.Row(
                    self.save_review_button,
                    self.reload_button,
                    self.clear_draft_button,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                ),
            ),
            self._card(
                "Note",
                self.note_input,
                pn.Row(
                    self.save_note_button,
                    self.show_history_checkbox,
                    sizing_mode="stretch_width",
                    margin=(0, 0, 0, 0),
                ),
            ),
            self._card("Annotation history", self.history_pane),
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "padding": "4px",
                "box-sizing": "border-box",
            },
        )

    def _card(self, title: str, *objects):
        return pn.Column(
            pn.pane.HTML(
                f"""
                <div style="font-weight: 700; font-size: 14px;
                            margin-bottom: 8px;">
                    {_escape(title)}
                </div>
                """,
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            ),
            *objects,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 0, 8, 0),
            styles={
                "border": "1px solid #d9d9d9",
                "border-radius": "8px",
                "background": "#ffffff",
                "padding": "10px 12px",
                "box-sizing": "border-box",
                "width": "100%",
                "overflow": "hidden",
            },
        )

    # ------------------------------------------------------------------
    # Platform helpers
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        self._subscribe("selection.focus.changed", self._on_selection_focus_changed)
        self._subscribe("selection.focus.cleared", self._on_selection_focus_cleared)
        self._subscribe("dataset.active.changed", self._on_dataset_active_changed)
        self._subscribe("dataset.updated", self._on_dataset_updated)
        self._subscribe("dataset.mapping_updated", self._on_dataset_mapping_updated)
        self._subscribe("artifact.created", self._on_artifact_created)

    def _subscribe(self, topic: str, callback) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            sub = events.subscribe(topic, callback)
            self._subscriptions.append(sub)
        except Exception:
            pass

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            events.publish(topic, payload)
        except Exception:
            pass

    def _active_dataset_id(self) -> Optional[str]:
        return _active_dataset_id(self.context)

    def _active_columns(self) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._active_dataset_id()

        if datasets is None or dataset_id is None:
            return []

        try:
            return [str(col) for col in datasets.list_columns(dataset_id)]
        except Exception:
            return []


    def _column_exists(self, column: Optional[str]) -> bool:
        if not column:
            return False
        return str(column) in set(self._active_columns())
    
    def _active_df(self) -> Optional[pd.DataFrame]:
        raise RuntimeError(
            "AnnotationsPanel must not materialise the full active dataset. "
            "Use list_columns(), get_row_by_id(), or get_row_by_position()."
        )

    def _get_mapping(self, semantic_name: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._active_dataset_id()

        if datasets is None or dataset_id is None:
            return None

        for method_name in ("get_mapping", "mapping", "get_column_mapping"):
            method = getattr(datasets, method_name, None)
            if not callable(method):
                continue

            try:
                value = method(dataset_id, semantic_name)
            except TypeError:
                try:
                    value = method(semantic_name)
                except Exception:
                    value = None
            except Exception:
                value = None

            if value is not None and str(value).strip():
                return str(value)

        return None

    def _is_index_mapping(self, value: Optional[str]) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {
            "use index",
            "index",
            "__index__",
            "_index",
        }

    def _resolve_record_id_col(self) -> Optional[str]:
        mapped = self._get_mapping("record_id")

        if self._is_index_mapping(mapped):
            return "Use Index"

        columns = set(self._active_columns())

        if mapped and mapped in columns:
            return mapped

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}

        return None

    def _resolve_label_col(self) -> Optional[str]:
        mapped = self._get_mapping("target_label")
        columns = set(self._active_columns())

        if mapped and mapped in columns:
            return mapped

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}

        return None
    
    def _record_preview_columns(self) -> Optional[List[str]]:
        columns = self._active_columns()
        if not columns:
            return None

        record_id_col = self._resolve_record_id_col()
        label_col = self._resolve_label_col()

        selected: List[str] = []

        if record_id_col and record_id_col != "Use Index":
            selected.append(record_id_col)

        if label_col:
            selected.append(label_col)

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}

        extra_cols = settings.get("extra_info_cols") or []

        if isinstance(extra_cols, str):
            extra_cols = [extra_cols]

        for col in extra_cols:
            if col and col in columns and col not in selected:
                selected.append(str(col))

        for col in columns:
            if col not in selected:
                selected.append(str(col))
            if len(selected) >= 10:
                break

        return selected or None

    def _row_df_to_series(self, row_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if row_df is None or row_df.empty:
            return None

        try:
            return row_df.iloc[0]
        except Exception:
            return None

    def _current_focus(self) -> Optional[Dict[str, Any]]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            focus = selection.get_focus()
        except Exception:
            return None

        if focus is None:
            return None

        if isinstance(focus, dict):
            return dict(focus)

        return {
            "dataset_id": getattr(focus, "dataset_id", None),
            "row_id": getattr(focus, "row_id", None),
            "origin": getattr(focus, "origin", None),
        }

    # ------------------------------------------------------------------
    # Selection / dataset events
    # ------------------------------------------------------------------

    def _sync_from_current_focus(self) -> None:
        focus = self._current_focus()
        if not focus:
            self.dataset_id = self._active_dataset_id()
            self.row_id = None
            self.row = None
            return

        self._set_focus(
            dataset_id=focus.get("dataset_id") or self._active_dataset_id(),
            row_id=focus.get("row_id"),
            render=False,
        )

    def _on_selection_focus_changed(self, _topic: str, payload: Dict[str, Any]) -> None:
        if self._disposed or not isinstance(payload, dict):
            return

        old_dataset_id = self.dataset_id
        old_row_id = self.row_id

        new_dataset_id = payload.get("dataset_id") or self._active_dataset_id()
        new_row_id = payload.get("row_id")

        if str(new_dataset_id) != str(old_dataset_id) or str(new_row_id) != str(old_row_id):
            self._clear_message()

        self._set_focus(
            dataset_id=new_dataset_id,
            row_id=new_row_id,
            render=True,
        )

    def _on_selection_focus_cleared(self, _topic: str, _payload: Dict[str, Any]) -> None:
        if self._disposed:
            return
        self._clear_message()
        self.row_id = None
        self.row = None
        self._render()

    def _on_dataset_active_changed(self, _topic: str, _payload: Dict[str, Any]) -> None:
        if self._disposed:
            return
        self._clear_message()
        self.dataset_id = self._active_dataset_id()
        self.row_id = None
        self.row = None
        self._render()

    def _on_dataset_updated(self, _topic: str, payload: Dict[str, Any]) -> None:
        if self._disposed:
            return

        if isinstance(payload, dict):
            dataset_id = payload.get("dataset_id")
            if self.dataset_id and dataset_id and dataset_id != self.dataset_id:
                return

        if self.row_id is not None:
            self._set_focus(self.dataset_id, self.row_id, render=True)
        else:
            self._render()

    def _on_dataset_mapping_updated(self, _topic: str, payload: Dict[str, Any]) -> None:
        if self._disposed:
            return

        if isinstance(payload, dict):
            dataset_id = payload.get("dataset_id")
            if self.dataset_id and dataset_id and dataset_id != self.dataset_id:
                return

        if self.row_id is not None:
            self._set_focus(self.dataset_id, self.row_id, render=True)
        else:
            self._render()

    def _on_artifact_created(self, _topic: str, payload: Dict[str, Any]) -> None:
        if self._disposed or not isinstance(payload, dict):
            return

        artifact_type = payload.get("type")
        if artifact_type not in {ANNOTATION_NOTE_TYPE, REVIEW_STATUS_TYPE}:
            return

        dataset_id = payload.get("dataset_id")
        row_ids = payload.get("row_ids") or []

        if (
            self.dataset_id is not None
            and self.row_id is not None
            and str(dataset_id) == str(self.dataset_id)
            and str(self.row_id) in [str(row_id) for row_id in row_ids]
        ):
            self._refresh_current_record_from_artifacts()
            self._load_record_state_into_widgets()
            self._render()

    def _set_focus(
        self,
        dataset_id: Optional[str],
        row_id: Any,
        *,
        render: bool,
    ) -> None:
        self.dataset_id = str(dataset_id) if dataset_id is not None else None
        self.row_id = str(row_id) if row_id is not None else None
        self.row = self._find_row(self.dataset_id, self.row_id)

        self._refresh_current_record_from_artifacts()
        self._load_record_state_into_widgets()

        if render:
            self._render()

    # ------------------------------------------------------------------
    # Dataset row helpers
    # ------------------------------------------------------------------

    def _find_row(
        self,
        dataset_id: Optional[str],
        row_id: Optional[str],
    ) -> Optional[pd.Series]:
        if dataset_id is None or row_id is None:
            return None

        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None

        columns = self._record_preview_columns()
        record_id_col = self._resolve_record_id_col()

        try:
            if record_id_col and record_id_col != "Use Index":
                row_df = datasets.get_row_by_id(
                    dataset_id,
                    row_id,
                    id_column=record_id_col,
                    columns=columns,
                )
                return self._row_df_to_series(row_df)

            # Index-based mapping: treat row_id as a position.
            if record_id_col == "Use Index":
                try:
                    position = int(row_id)
                except Exception:
                    return None

                row_df = datasets.get_row_by_position(
                    dataset_id,
                    position,
                    columns=columns,
                )
                return self._row_df_to_series(row_df)

            # Last-resort compatibility fallback: numeric row_id as position.
            try:
                position = int(row_id)
            except Exception:
                return None

            row_count = None
            try:
                row_count = datasets.row_count(dataset_id)
            except Exception:
                pass

            if row_count is not None and not (0 <= position < row_count):
                return None

            row_df = datasets.get_row_by_position(
                dataset_id,
                position,
                columns=columns,
            )
            return self._row_df_to_series(row_df)

        except Exception:
            return None

    def _record_key(self) -> Optional[str]:
        return _record_key(self.dataset_id, self.row_id)

    def _refresh_current_record_from_artifacts(self) -> None:
        key = self._record_key()
        if key is None:
            return

        artifact_state = build_record_state_from_artifacts(
            self.context,
            dataset_id=self.dataset_id,
            row_id=self.row_id,
        )

        existing = self._records.get(key)
        merged = _merge_record_states(existing, artifact_state)
        merged["dataset_id"] = self.dataset_id
        merged["row_id"] = self.row_id

        self._records[key] = merged

    def _get_record_state(self) -> Optional[Dict[str, Any]]:
        key = self._record_key()
        if key is None:
            return None

        if key not in self._records:
            self._records[key] = _default_record_state(
                dataset_id=self.dataset_id,
                row_id=self.row_id,
            )

        return self._records[key]

    def _load_record_state_into_widgets(self) -> None:
        state = self._get_record_state()
        if not state:
            self.status_select.value = "unreviewed"
            self.confidence_slider.value = 0.0
            self.tags_input.value = ""
            self.label_suggestion_input.value = ""
            return

        status = state.get("status", "unreviewed")
        if status not in self.REVIEW_STATUSES:
            status = "unreviewed"

        self.status_select.value = status
        self.confidence_slider.value = max(
            0.0,
            min(1.0, _safe_float(state.get("confidence", 0.0), default=0.0)),
        )

        tags = state.get("tags") or []
        if isinstance(tags, str):
            self.tags_input.value = tags
        else:
            self.tags_input.value = ", ".join(str(tag) for tag in tags)

        self.label_suggestion_input.value = str(state.get("label_suggestion", "") or "")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_save_note(self, _event=None) -> None:
        if not self._ensure_can_save():
            return

        note_text = self.note_input.value.strip()
        if not note_text:
            self._show_message("Write a note before saving.", "warning")
            return

        record_state = self._get_record_state()
        if record_state is None:
            self._show_message("No focused record to annotate.", "warning")
            return

        annotation_id = f"annotation:{uuid.uuid4().hex}"
        timestamp = _now()
        tags = self._parse_tags(self.tags_input.value)

        note = {
            "annotation_id": annotation_id,
            "dataset_id": self.dataset_id,
            "row_id": self.row_id,
            "text": note_text,
            "status": self.status_select.value,
            "confidence": float(self.confidence_slider.value or 0.0),
            "tags": tags,
            "label_suggestion": self.label_suggestion_input.value.strip(),
            "created_at": timestamp,
            "updated_at": timestamp,
            "source": "core.annotations",
            "instance_id": self.instance_id,
        }

        artifact_id = self._put_artifact(
            artifact_type=ANNOTATION_NOTE_TYPE,
            payload=note,
            params={"annotation_id": annotation_id},
        )
        note["artifact_id"] = artifact_id

        record_state.setdefault("notes", []).append(note)
        record_state["notes"] = _dedupe_items(
            record_state.get("notes") or [],
            ("annotation_id", "artifact_id"),
        )
        record_state["status"] = note["status"]
        record_state["confidence"] = note["confidence"]
        record_state["tags"] = list(tags)
        record_state["label_suggestion"] = note["label_suggestion"]
        record_state["updated_at"] = timestamp

        event_payload = {
            "dataset_id": self.dataset_id,
            "row_id": self.row_id,
            "annotation_id": annotation_id,
            "artifact_id": artifact_id,
            "status": note["status"],
            "confidence": note["confidence"],
            "tags": list(tags),
            "label_suggestion": note["label_suggestion"],
            "source": "core.annotations",
            "instance_id": self.instance_id,
        }

        self._publish("annotation.created", event_payload)

        if artifact_id:
            self._publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": ANNOTATION_NOTE_TYPE,
                    "dataset_id": self.dataset_id,
                    "row_ids": [self.row_id],
                    "source": "core.annotations",
                },
            )

        if note["label_suggestion"]:
            self._publish(
                "labels.suggested",
                {
                    "dataset_id": self.dataset_id,
                    "row_id": self.row_id,
                    "label": note["label_suggestion"],
                    "annotation_id": annotation_id,
                    "source": "core.annotations",
                    "instance_id": self.instance_id,
                },
            )

        self.note_input.value = ""
        self._show_message("Note saved.", "success")
        self._render()

    def _on_save_review(self, _event=None) -> None:
        if not self._ensure_can_save():
            return

        record_state = self._get_record_state()
        if record_state is None:
            self._show_message("No focused record to review.", "warning")
            return

        timestamp = _now()
        review_id = f"review:{uuid.uuid4().hex}"
        tags = self._parse_tags(self.tags_input.value)

        review_payload = {
            "review_id": review_id,
            "dataset_id": self.dataset_id,
            "row_id": self.row_id,
            "status": self.status_select.value,
            "confidence": float(self.confidence_slider.value or 0.0),
            "tags": tags,
            "label_suggestion": self.label_suggestion_input.value.strip(),
            "created_at": timestamp,
            "updated_at": timestamp,
            "source": "core.annotations",
            "instance_id": self.instance_id,
        }

        artifact_id = self._put_artifact(
            artifact_type=REVIEW_STATUS_TYPE,
            payload=review_payload,
            params={"review_id": review_id, "status": review_payload["status"]},
        )
        review_payload["artifact_id"] = artifact_id

        record_state.setdefault("reviews", []).append(review_payload)
        record_state["reviews"] = _dedupe_items(
            record_state.get("reviews") or [],
            ("review_id", "artifact_id"),
        )
        record_state["status"] = review_payload["status"]
        record_state["confidence"] = review_payload["confidence"]
        record_state["tags"] = list(tags)
        record_state["label_suggestion"] = review_payload["label_suggestion"]
        record_state["updated_at"] = timestamp

        self._publish(
            "review.status.changed",
            {
                **review_payload,
                "artifact_id": artifact_id,
            },
        )

        if artifact_id:
            self._publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": REVIEW_STATUS_TYPE,
                    "dataset_id": self.dataset_id,
                    "row_ids": [self.row_id],
                    "source": "core.annotations",
                },
            )

        if review_payload["label_suggestion"]:
            self._publish(
                "labels.suggested",
                {
                    "dataset_id": self.dataset_id,
                    "row_id": self.row_id,
                    "label": review_payload["label_suggestion"],
                    "review_id": review_id,
                    "source": "core.annotations",
                    "instance_id": self.instance_id,
                },
            )

        self._show_message("Review state saved.", "success")
        self._render()

    def _on_reload_from_artifacts(self, _event=None) -> None:
        if self.dataset_id is None or self.row_id is None:
            self._show_message("No focused record to reload.", "warning")
            return

        self._refresh_current_record_from_artifacts()
        self._load_record_state_into_widgets()
        self._show_message("Reloaded annotation history from artifacts.", "success")
        self._render()

    def _on_clear_draft(self, _event=None) -> None:
        self.note_input.value = ""
        self.tags_input.value = ""
        self.label_suggestion_input.value = ""
        self.status_select.value = "unreviewed"
        self.confidence_slider.value = 0.0
        self._show_message("Draft cleared.", "info")

    def _on_show_history_changed(self, _event=None) -> None:
        self._render_history()

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _put_artifact(
        self,
        *,
        artifact_type: str,
        payload: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            return None

        row_ids = [self.row_id] if self.row_id is not None else None

        try:
            return artifacts.put(
                artifact_type,
                payload,
                dataset_id=self.dataset_id or "default",
                row_ids=row_ids,
                params=params or {},
            )
        except TypeError:
            try:
                return artifacts.put(
                    type=artifact_type,
                    payload=payload,
                    dataset_id=self.dataset_id or "default",
                    row_ids=row_ids,
                    params=params or {},
                )
            except Exception:
                return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        self._set_enabled_state()
        self._render_header()
        self._render_record()
        self._render_history()

    def _render_header(self) -> None:
        dataset = _escape(self.dataset_id or "No active dataset")
        row_id = _escape(self.row_id or "No focused record")

        self.header_pane.object = f"""
        <div style="display: flex; flex-direction: column; gap: 2px;">
            <div style="font-weight: 700; font-size: 15px;">
                Annotations
            </div>
            <div style="font-size: 12px; color: #666;">
                Dataset: <code>{dataset}</code>
                &nbsp; | &nbsp;
                Focus: <code>{row_id}</code>
            </div>
        </div>
        """

    def _render_record(self) -> None:
        if self.dataset_id is None:
            self.record_pane.object = self._empty_html("No active dataset is loaded.")
            return

        if self.row_id is None:
            self.record_pane.object = self._empty_html(
                "No record is focused. Use Record Browser or a visualisation panel "
                "to focus a row."
            )
            return

        if self.row is None:
            self.record_pane.object = self._empty_html(
                "The focused record could not be found in the active dataset."
            )
            return

        label_col = self._resolve_label_col()

        rows: List[Tuple[str, Any]] = [
            ("Dataset", self.dataset_id),
            ("Record ID", self.row_id),
        ]

        if label_col and label_col in self.row.index:
            rows.append(("Current label", self.row[label_col]))

        preview_count = 0

        for col in self.row.index:
            if label_col and col == label_col:
                continue

            value = self.row[col]

            if pd.isna(value):
                continue

            rows.append((str(col), value))
            preview_count += 1

            if preview_count >= 8:
                break

        self.record_pane.object = self._table_html(rows)

    def _render_history(self) -> None:
        if not self.show_history_checkbox.value:
            self.history_pane.object = self._empty_html("History hidden.")
            return

        state = self._get_record_state()
        if not state:
            self.history_pane.object = self._empty_html(
                "No annotation history for this record yet."
            )
            return

        notes = list(state.get("notes") or [])
        reviews = list(state.get("reviews") or [])

        events: List[Dict[str, Any]] = []
        for note in notes:
            item = dict(note)
            item["_history_type"] = "note"
            item["_history_time"] = item.get("created_at") or item.get("updated_at") or ""
            events.append(item)

        for review in reviews:
            item = dict(review)
            item["_history_type"] = "review"
            item["_history_time"] = item.get("updated_at") or item.get("created_at") or ""
            events.append(item)

        if not events:
            self.history_pane.object = self._empty_html(
                "No notes or review states saved for this record yet."
            )
            return

        events.sort(
            key=lambda item: (
                str(item.get("_history_time", "")),
                _safe_float(item.get("_sort_time", 0.0), default=0.0),
            ),
            reverse=True,
        )

        cards = []
        for item in events:
            kind = item.get("_history_type", "event")
            timestamp = _escape(item.get("_history_time", ""))
            status = _escape(item.get("status", ""))
            confidence = _escape(f"{_safe_float(item.get('confidence', 0.0), default=0.0):.2f}")
            label = _escape(item.get("label_suggestion", "") or "—")
            tags = item.get("tags") or []
            tag_text = _escape(", ".join(str(tag) for tag in tags) or "—")

            if kind == "note":
                title = "Note"
                text = _escape(item.get("text", ""))
            else:
                title = "Review state"
                text = _escape(
                    f"Status set to {item.get('status', 'unreviewed')}"
                )

            cards.append(
                f"""
                <div style="border: 1px solid #e1e1e1; border-radius: 6px;
                            padding: 8px; margin-bottom: 8px;
                            background: #fafafa;">
                    <div style="display: flex; justify-content: space-between;
                                gap: 8px; margin-bottom: 4px;">
                        <strong style="font-size: 13px;">{_escape(title)}</strong>
                        <span style="font-size: 11px; color: #666;">{timestamp}</span>
                    </div>
                    <div style="font-size: 12px; margin-bottom: 4px;">
                        <strong>Status:</strong> {status}
                        &nbsp; | &nbsp;
                        <strong>Confidence:</strong> {confidence}
                    </div>
                    <div style="font-size: 12px; margin-bottom: 4px;">
                        <strong>Suggested label:</strong> {label}
                        &nbsp; | &nbsp;
                        <strong>Tags:</strong> {tag_text}
                    </div>
                    <div style="white-space: pre-wrap; font-size: 13px;">
                        {text}
                    </div>
                </div>
                """
            )

        self.history_pane.object = "\n".join(cards)

    def _set_enabled_state(self) -> None:
        enabled = self.dataset_id is not None and self.row_id is not None

        for widget in (
            self.status_select,
            self.confidence_slider,
            self.tags_input,
            self.label_suggestion_input,
            self.note_input,
            self.save_note_button,
            self.save_review_button,
            self.reload_button,
            self.clear_draft_button,
        ):
            widget.disabled = not enabled

    def _show_message(self, message: str, alert_type: str = "info") -> None:
        self.message_pane.object = message
        self.message_pane.alert_type = alert_type
        self.message_pane.visible = True
    
    def _clear_message(self) -> None:
        self.message_pane.object = ""
        self.message_pane.visible = False

    # ------------------------------------------------------------------
    # Validation / formatting
    # ------------------------------------------------------------------

    def _ensure_can_save(self) -> bool:
        if self.dataset_id is None:
            self._show_message("No active dataset is loaded.", "warning")
            return False

        if self.row_id is None:
            self._show_message("No focused record is available.", "warning")
            return False

        if self.row is None:
            self._show_message(
                "The focused record could not be found in the active dataset.",
                "warning",
            )
            return False

        return True

    def _parse_tags(self, value: str) -> List[str]:
        tags = []
        for raw_tag in (value or "").split(","):
            tag = raw_tag.strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _empty_html(self, text: str) -> str:
        return f"""
        <div style="font-size: 13px; color: #666; padding: 4px 0;">
            {_escape(text)}
        </div>
        """

    def _table_html(self, rows: List[Tuple[str, Any]]) -> str:
        body = []
        for key, value in rows:
            body.append(
                f"""
                <tr>
                    <th style="text-align: left; vertical-align: top;
                               padding: 4px 8px 4px 0; color: #555;
                               font-weight: 600; white-space: nowrap;">
                        {_escape(key)}
                    </th>
                    <td style="text-align: left; vertical-align: top;
                               padding: 4px 0; word-break: break-word;">
                        {_escape(value)}
                    </td>
                </tr>
                """
            )

        return f"""
        <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
            <tbody>
                {''.join(body)}
            </tbody>
        </table>
        """


# ---------------------------------------------------------------------
# Summary / export panel
# ---------------------------------------------------------------------


class AnnotationSummaryPanel:
    def __init__(
        self,
        *,
        context,
        instance_id: Optional[str] = None,
        restore_state: Optional[Dict[str, Any]] = None,
        restore_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> None:
        self.context = context
        self.instance_id = instance_id or f"core.annotations.summary:{uuid.uuid4().hex[:10]}"
        self.restore_metadata = restore_metadata or {}

        self._subscriptions: List[Any] = []
        self._watchers: List[Any] = []
        self._disposed = False

        self.dataset_id: Optional[str] = _active_dataset_id(self.context)
        self.summary_df = pd.DataFrame()

        self._build_widgets()
        self._build_root()
        self._subscribe_events()

        self._restored_summary_from_state = False

        self._build_widgets()
        self._build_root()
        self._subscribe_events()

        if restore_state:
            self.restore_state(restore_state)

        if self._restored_summary_from_state:
            self._render()
            self._show_message(
                f"Restored cached summary with {len(self.summary_df)} rows.",
                "info",
            )
        else:
            self.refresh()

    def panel(self):
        return self.root

    def dispose(self) -> None:
        self._disposed = True

        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    def get_state(self) -> Dict[str, Any]:
        summary_records: List[Dict[str, Any]] = []

        if isinstance(self.summary_df, pd.DataFrame) and not self.summary_df.empty:
            try:
                summary_records = self.summary_df.to_dict(orient="records")
            except Exception:
                summary_records = []

        return {
            "version": 2,
            "dataset_id": self.dataset_id,
            "summary_records": summary_records,
            "summary_columns": (
                list(self.summary_df.columns)
                if isinstance(self.summary_df, pd.DataFrame)
                else []
            ),
            "summary_row_count": int(len(self.summary_df))
            if isinstance(self.summary_df, pd.DataFrame)
            else 0,
            "cached_at": _now(),
        }

    def snapshot_state(self) -> Dict[str, Any]:
        return self.get_state()

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        if state.get("dataset_id"):
            self.dataset_id = str(state.get("dataset_id"))

        records = state.get("summary_records")
        columns = state.get("summary_columns") or []

        if isinstance(records, list):
            try:
                restored_df = pd.DataFrame(records)

                if columns:
                    # Preserve saved column order and include missing columns as empty.
                    for col in columns:
                        if col not in restored_df.columns:
                            restored_df[col] = ""
                    restored_df = restored_df[[col for col in columns]]

                self.summary_df = restored_df
                self._restored_summary_from_state = True
            except Exception:
                self.summary_df = pd.DataFrame()
                self._restored_summary_from_state = False

    def _build_widgets(self) -> None:
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            button_type="primary",
            width=90,
            height=32,
            margin=(0, 8, 8, 0),
        )
        self.dataset_button = pn.widgets.Button(
            name="Create summary dataset",
            button_type="success",
            width=175,
            height=32,
            margin=(0, 8, 8, 0),
        )
        self.csv_button = pn.widgets.Button(
            name="Create CSV artifact",
            button_type="default",
            width=150,
            height=32,
            margin=(0, 0, 8, 0),
        )

        self.refresh_button.on_click(lambda _event: self.refresh())
        self.dataset_button.on_click(self._on_create_summary_dataset)
        self.csv_button.on_click(self._on_create_csv_artifact)

    def _build_root(self) -> None:
        self.header_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.message_pane = pn.pane.Alert(
            "",
            alert_type="info",
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.table_pane = pn.pane.DataFrame(
            pd.DataFrame(),
            index=False,
            sizing_mode="stretch_both",
            margin=(0, 0, 8, 0),
        )

        self.root = pn.Column(
            self.header_pane,
            self.message_pane,
            pn.Row(
                self.refresh_button,
                self.dataset_button,
                self.csv_button,
                sizing_mode="stretch_width",
            ),
            self.table_pane,
            sizing_mode="stretch_both",
            scroll=True,
            margin=(0, 0, 0, 0),
            styles={
                "padding": "4px",
                "box-sizing": "border-box",
            },
        )

    def _subscribe_events(self) -> None:
        self._subscribe("dataset.active.changed", self._on_dataset_active_changed)
        self._subscribe("artifact.created", self._on_artifact_created)

    def _subscribe(self, topic: str, callback) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            sub = events.subscribe(topic, callback)
            self._subscriptions.append(sub)
        except Exception:
            pass

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            events.publish(topic, payload)
        except Exception:
            pass

    def _on_dataset_active_changed(self, _topic: str, _payload: Dict[str, Any]) -> None:
        if self._disposed:
            return
        self.dataset_id = _active_dataset_id(self.context)
        self.refresh()

    def _on_artifact_created(self, _topic: str, payload: Dict[str, Any]) -> None:
        if self._disposed or not isinstance(payload, dict):
            return

        if payload.get("type") in {ANNOTATION_NOTE_TYPE, REVIEW_STATUS_TYPE}:
            if self.dataset_id is None or str(payload.get("dataset_id")) == str(self.dataset_id):
                self.refresh(show_message=False)

    def refresh(self, show_message: bool = True) -> None:
        self.dataset_id = self.dataset_id or _active_dataset_id(self.context)

        self.summary_df = build_annotation_summary_dataframe(
            self.context,
            dataset_id=self.dataset_id,
        )
        self._restored_summary_from_state = False

        self._render()

        if show_message:
            self._show_message(
                f"Summary refreshed. {len(self.summary_df)} annotated records found.",
                "success",
            )

    def _render(self) -> None:
        dataset = _escape(self.dataset_id or "No active dataset")
        count = len(self.summary_df)

        self.header_pane.object = f"""
        <div style="display: flex; flex-direction: column; gap: 2px;">
            <div style="font-weight: 700; font-size: 15px;">
                Annotation Summary
            </div>
            <div style="font-size: 12px; color: #666;">
                Dataset: <code>{dataset}</code>
                &nbsp; | &nbsp;
                Annotated records: <strong>{count}</strong>
            </div>
        </div>
        """

        self.table_pane.object = self.summary_df

    def _on_create_summary_dataset(self, _event=None) -> None:
        self.refresh(show_message=False)

        if self.summary_df.empty:
            self._show_message("No annotations to export as a dataset.", "warning")
            return

        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            self._show_message("Dataset manager is unavailable.", "danger")
            return

        dataset_id = f"annotation_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            datasets.register(
                dataset_id,
                self.summary_df.copy(),
                name="Annotation Summary",
                derived_from=self.dataset_id,
                provenance="core.annotations",
                created_at=_now(),
            )
        except Exception as exc:
            self._show_message(f"Could not create summary dataset: {exc}", "danger")
            return

        self._publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "name": "Annotation Summary",
                "derived_from": self.dataset_id,
                "source": "core.annotations",
            },
        )

        self._show_message(
            f"Created summary dataset: {dataset_id}",
            "success",
        )

    def _on_create_csv_artifact(self, _event=None) -> None:
        self.refresh(show_message=False)

        if self.summary_df.empty:
            self._show_message("No annotations to export as CSV.", "warning")
            return

        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            self._show_message("Artifact store is unavailable.", "danger")
            return

        csv_text = self.summary_df.to_csv(index=False)

        try:
            artifact_id = artifacts.put(
                SUMMARY_CSV_TYPE,
                {
                    "filename": "annotation_summary.csv",
                    "csv": csv_text,
                    "row_count": int(len(self.summary_df)),
                    "dataset_id": self.dataset_id,
                    "created_at": _now(),
                    "source": "core.annotations",
                },
                dataset_id=self.dataset_id or "default",
                row_ids=None,
                params={
                    "source": "core.annotations",
                    "format": "csv",
                    "row_count": int(len(self.summary_df)),
                },
            )
        except Exception as exc:
            self._show_message(f"Could not create CSV artifact: {exc}", "danger")
            return

        self._publish(
            "artifact.created",
            {
                "artifact_id": artifact_id,
                "type": SUMMARY_CSV_TYPE,
                "dataset_id": self.dataset_id,
                "row_ids": None,
                "source": "core.annotations",
            },
        )

        self._show_message(
            f"Created CSV artifact: {artifact_id}",
            "success",
        )

    def _show_message(self, message: str, alert_type: str = "info") -> None:
        self.message_pane.object = message
        self.message_pane.alert_type = alert_type
        self.message_pane.visible = True