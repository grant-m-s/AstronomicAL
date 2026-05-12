from __future__ import annotations

import html
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.annotations",
    name="Annotations",
    version="0.1.0",
    description=(
        "Record-level notes, review state, tags, confidence, and optional "
        "label suggestions for human-in-the-loop analysis workflows."
    ),
    capabilities=[
        "panel",
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
        ],
        default_layout={"x": 8, "y": 0, "w": 4, "h": 7},
        state_version=1,
        persist_layout=True,
        persist_state=True,
    )


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


class AnnotationsPanel:
    """Generic record-level annotation and review panel.

    This panel intentionally avoids astronomy assumptions.

    It only assumes:

    - there is an active dataset
    - the dataset has a semantic `record_id` mapping
    - another panel, such as Record Browser or Visualisation, publishes focus
      through `context.selection.set_focus(...)`

    Saved notes/review decisions are kept in the controller state for workspace
    persistence and are also emitted as artifacts/events so other plugins can
    react to them.
    """

    REVIEW_STATUSES = [
        "unreviewed",
        "in_review",
        "approved",
        "rejected",
        "unsure",
        "needs_follow_up",
    ]

    CONFIDENCE_LABELS = {
        0.0: "No confidence set",
        0.25: "Low",
        0.5: "Medium",
        0.75: "High",
        1.0: "Certain",
    }

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

        # Keyed by "dataset_id::row_id".
        self._records: Dict[str, Dict[str, Any]] = {}

        self._build_widgets()
        self._build_root()
        self._subscribe_events()

        if restore_state:
            self.restore_state(restore_state)

        self._sync_from_current_focus()
        self._render()

    # ------------------------------------------------------------------
    # Public panel/lifecycle/persistence API
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
            "version": 1,
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
            # Keep this deliberately tolerant: older saved states may contain
            # partially serialized annotation records.
            self._records.update(records)

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

            try:
                confidence = float(draft.get("confidence", 0.0) or 0.0)
                self.confidence_slider.value = max(0.0, min(1.0, confidence))
            except Exception:
                self.confidence_slider.value = 0.0

        if "show_history" in state:
            self.show_history_checkbox.value = bool(state.get("show_history"))

    # ------------------------------------------------------------------
    # UI construction
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
        self.clear_draft_button.on_click(self._on_clear_draft)

        self._watchers.append(
            self.status_select.param.watch(self._on_local_review_widget_changed, "value")
        )
        self._watchers.append(
            self.confidence_slider.param.watch(
                self._on_local_review_widget_changed, "value"
            )
        )
        self._watchers.append(
            self.tags_input.param.watch(self._on_local_review_widget_changed, "value")
        )
        self._watchers.append(
            self.label_suggestion_input.param.watch(
                self._on_local_review_widget_changed, "value"
            )
        )
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
            self._card(
                "Focused record",
                self.record_pane,
            ),
            self._card(
                "Review state",
                self.status_select,
                self.confidence_slider,
                self.tags_input,
                self.label_suggestion_input,
                pn.Row(
                    self.save_review_button,
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
            self._card(
                "Annotation history",
                self.history_pane,
            ),
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
                    {self._escape(title)}
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
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None
        try:
            return datasets.active_id()
        except Exception:
            return None

    def _active_df(self) -> Optional[pd.DataFrame]:
        datasets = getattr(self.context, "datasets", None)
        dataset_id = self._active_dataset_id()
        if datasets is None or dataset_id is None:
            return None
        try:
            return datasets.get_df(dataset_id)
        except TypeError:
            try:
                return datasets.get_df()
            except Exception:
                return None
        except Exception:
            return None

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
            if value:
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

        df = self._active_df()
        if df is not None and mapped in df.columns:
            return mapped

        # Compatibility bridge for older settings-backed panels.
        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}
        legacy = settings.get("id_col")
        if self._is_index_mapping(legacy):
            return "Use Index"
        if df is not None and legacy in df.columns:
            return legacy

        return None

    def _resolve_label_col(self) -> Optional[str]:
        mapped = self._get_mapping("target_label")
        df = self._active_df()
        if df is not None and mapped in df.columns:
            return mapped

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}
        legacy = settings.get("label_col")
        if df is not None and legacy in df.columns:
            return legacy

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

        self._set_focus(
            dataset_id=payload.get("dataset_id") or self._active_dataset_id(),
            row_id=payload.get("row_id"),
            render=True,
        )

    def _on_selection_focus_cleared(self, _topic: str, _payload: Dict[str, Any]) -> None:
        if self._disposed:
            return
        self.row_id = None
        self.row = None
        self._render()

    def _on_dataset_active_changed(self, _topic: str, _payload: Dict[str, Any]) -> None:
        if self._disposed:
            return
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

        df = self._active_df()
        if df is None or len(df) == 0:
            return None

        record_id_col = self._resolve_record_id_col()

        try:
            if record_id_col == "Use Index":
                matches = df.index.astype(str) == str(row_id)
                if matches.any():
                    position = list(matches).index(True)
                    return df.iloc[position]

            if record_id_col and record_id_col in df.columns:
                matches = df[record_id_col].astype(str) == str(row_id)
                if matches.any():
                    return df.loc[matches].iloc[0]

            # Last-resort fallback: allow focus row_id to be an integer index.
            try:
                idx = int(row_id)
                if 0 <= idx < len(df):
                    return df.iloc[idx]
            except Exception:
                pass

        except Exception:
            return None

        return None

    def _record_key(
        self,
        dataset_id: Optional[str] = None,
        row_id: Optional[str] = None,
    ) -> Optional[str]:
        dataset_id = dataset_id if dataset_id is not None else self.dataset_id
        row_id = row_id if row_id is not None else self.row_id

        if dataset_id is None or row_id is None:
            return None
        return f"{dataset_id}::{row_id}"

    def _default_record_state(self) -> Dict[str, Any]:
        return {
            "status": "unreviewed",
            "confidence": 0.0,
            "tags": [],
            "label_suggestion": "",
            "notes": [],
            "created_at": self._now(),
            "updated_at": self._now(),
        }

    def _get_record_state(self) -> Optional[Dict[str, Any]]:
        key = self._record_key()
        if key is None:
            return None

        if key not in self._records:
            self._records[key] = self._default_record_state()

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

        try:
            confidence = float(state.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        self.confidence_slider.value = max(0.0, min(1.0, confidence))

        tags = state.get("tags") or []
        if isinstance(tags, str):
            self.tags_input.value = tags
        else:
            self.tags_input.value = ", ".join(str(tag) for tag in tags)

        self.label_suggestion_input.value = str(state.get("label_suggestion", "") or "")

    # ------------------------------------------------------------------
    # Button/watch callbacks
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
        timestamp = self._now()
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

        record_state.setdefault("notes", []).append(note)
        record_state["status"] = self.status_select.value
        record_state["confidence"] = float(self.confidence_slider.value or 0.0)
        record_state["tags"] = tags
        record_state["label_suggestion"] = self.label_suggestion_input.value.strip()
        record_state["updated_at"] = timestamp

        artifact_id = self._put_artifact(
            artifact_type="annotation.note",
            payload=note,
            params={"annotation_id": annotation_id},
        )

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
                    "type": "annotation.note",
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

        timestamp = self._now()
        tags = self._parse_tags(self.tags_input.value)

        review_payload = {
            "dataset_id": self.dataset_id,
            "row_id": self.row_id,
            "status": self.status_select.value,
            "confidence": float(self.confidence_slider.value or 0.0),
            "tags": tags,
            "label_suggestion": self.label_suggestion_input.value.strip(),
            "updated_at": timestamp,
            "source": "core.annotations",
            "instance_id": self.instance_id,
        }

        record_state["status"] = review_payload["status"]
        record_state["confidence"] = review_payload["confidence"]
        record_state["tags"] = list(tags)
        record_state["label_suggestion"] = review_payload["label_suggestion"]
        record_state["updated_at"] = timestamp

        artifact_id = self._put_artifact(
            artifact_type="review.status",
            payload=review_payload,
            params={"status": review_payload["status"]},
        )

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
                    "type": "review.status",
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
                    "source": "core.annotations",
                    "instance_id": self.instance_id,
                },
            )

        self._show_message("Review state saved.", "success")
        self._render()

    def _on_clear_draft(self, _event=None) -> None:
        self.note_input.value = ""
        self.tags_input.value = ""
        self.label_suggestion_input.value = ""
        self.status_select.value = "unreviewed"
        self.confidence_slider.value = 0.0
        self._show_message("Draft cleared.", "info")

    def _on_local_review_widget_changed(self, _event=None) -> None:
        # Widget changes are deliberately not published immediately. They become
        # platform state only when the user clicks Save review state or Save note.
        pass

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

        # The branch examples use positional type/payload. Some versions also
        # support keyword names, so keep both paths.
        try:
            return artifacts.put(
                artifact_type,
                payload,
                dataset_id=self.dataset_id,
                row_ids=row_ids,
                params=params or {},
            )
        except TypeError:
            try:
                return artifacts.put(
                    type=artifact_type,
                    payload=payload,
                    dataset_id=self.dataset_id,
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
        dataset = self._escape(self.dataset_id or "No active dataset")
        row_id = self._escape(self.row_id or "No focused record")

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
            self.record_pane.object = self._empty_html(
                "No active dataset is loaded."
            )
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

        # Show a compact preview of non-empty row values.
        preview_count = 0
        for col in self.row.index:
            if label_col and col == label_col:
                continue
            if preview_count >= 8:
                break

            value = self.row[col]
            if pd.isna(value):
                continue

            rows.append((str(col), value))
            preview_count += 1

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

        notes = state.get("notes") or []
        if not notes:
            self.history_pane.object = self._empty_html(
                "No notes saved for this record yet."
            )
            return

        cards = []
        for note in reversed(notes):
            created = self._escape(note.get("created_at", ""))
            status = self._escape(note.get("status", ""))
            confidence = self._escape(f"{float(note.get('confidence', 0.0)):.2f}")
            label = self._escape(note.get("label_suggestion", "") or "—")
            tags = note.get("tags") or []
            tag_text = self._escape(", ".join(str(tag) for tag in tags) or "—")
            text = self._escape(note.get("text", ""))

            cards.append(
                f"""
                <div style="border: 1px solid #e1e1e1; border-radius: 6px;
                            padding: 8px; margin-bottom: 8px;
                            background: #fafafa;">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">
                        {created}
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
            self.clear_draft_button,
        ):
            widget.disabled = not enabled

    def _show_message(self, message: str, alert_type: str = "info") -> None:
        self.message_pane.object = message
        self.message_pane.alert_type = alert_type
        self.message_pane.visible = True

    # ------------------------------------------------------------------
    # Validation / formatting helpers
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

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _escape(self, value: Any) -> str:
        return html.escape(str(value), quote=True)

    def _empty_html(self, text: str) -> str:
        return f"""
        <div style="font-size: 13px; color: #666; padding: 4px 0;">
            {self._escape(text)}
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
                        {self._escape(key)}
                    </th>
                    <td style="text-align: left; vertical-align: top;
                               padding: 4px 0; word-break: break-word;">
                        {self._escape(value)}
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