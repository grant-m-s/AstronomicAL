from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import uuid

# selection.focus.changed
# selection.focus.cleared
# selection.set.changed
# selection.set.cleared
# selection.set.promoted

@dataclass
class FocusState:
    dataset_id: Optional[str] = None
    row_id: Optional[str] = None
    origin: Optional[str] = None
    panel_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionSetState:
    selection_set_id: str
    dataset_id: str
    row_ids: List[str]
    origin: Optional[str] = None
    panel_id: Optional[str] = None
    mode: str = "replace"
    artifact_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelectionManager:
    """
    Owns current focus and current multi-selection set.

    Responsibilities:
    - store current focus state
    - store current active selection set
    - publish canonical selection events
    - optionally materialize a selection-set artifact

    Non-responsibilities:
    - reading dataframe rows
    - mutating Bokeh/Panel state directly
    - deciding how panels render selection
    """

    def __init__(self, events, artifacts=None):
        self.events = events
        self.artifacts = artifacts

        self._focus = FocusState()
        self._active_set: Optional[SelectionSetState] = None

    # -------------------------
    # Read API
    # -------------------------

    def get_focus(self) -> FocusState:
        return self._focus

    def get_active_set(self) -> Optional[SelectionSetState]:
        return self._active_set

    # -------------------------
    # Focus API
    # -------------------------

    def clear_focus(self, *, origin=None, panel_id=None):
        dataset_id = self._focus.dataset_id
        self._focus = FocusState(origin=origin, panel_id=panel_id)

        self.events.publish(
            "selection.focus.cleared",
            {
                "dataset_id": dataset_id,
                "origin": origin,
                "panel_id": panel_id,
            },
        )

    def set_focus(
        self,
        dataset_id: str,
        row_id: str,
        *,
        origin=None,
        panel_id=None,
        selection_set_id=None,
        metadata=None,
    ):
        row_id = str(row_id)
        metadata = dict(metadata or {})

        self._focus = FocusState(
            dataset_id=dataset_id,
            row_id=row_id,
            origin=origin,
            panel_id=panel_id,
            metadata=metadata,
        )

        self.events.publish(
            "selection.focus.changed",
            {
                "dataset_id": dataset_id,
                "row_id": row_id,
                "origin": origin,
                "panel_id": panel_id,
                "selection_set_id": selection_set_id,
                "timestamp": self._focus.timestamp,
                "metadata": metadata,
            },
        )

    # -------------------------
    # Selection-set API
    # -------------------------

    def clear_selection_set(self, *, origin=None, panel_id=None):
        old = self._active_set
        self._active_set = None

        self.events.publish(
            "selection.set.cleared",
            {
                "dataset_id": old.dataset_id if old else None,
                "selection_set_id": old.selection_set_id if old else None,
                "origin": origin,
                "panel_id": panel_id,
            },
        )

    def set_selection_set(
        self,
        dataset_id: str,
        row_ids: List[str],
        *,
        origin=None,
        panel_id=None,
        mode="replace",
        metadata=None,
        create_artifact=True,
        update_focus_policy="preserve_or_first",
    ) -> SelectionSetState:
        row_ids = [str(r) for r in row_ids]
        row_ids = list(dict.fromkeys(row_ids))  # stable dedupe

        selection_set_id = f"selset:{uuid.uuid4().hex[:8]}"
        artifact_id = None

        metadata = metadata or {}

        if create_artifact and self.artifacts is not None:
            artifact_id = self.artifacts.put(
                "selection.ids",
                {
                    "dataset_id": dataset_id,
                    "row_ids": row_ids,
                    "mode": mode,
                    "origin": origin,
                    "panel_id": panel_id,
                    "metadata": metadata,
                },
                dataset_id=dataset_id,
                row_ids=row_ids,
                params={"selection_set_id": selection_set_id},
            )

        state = SelectionSetState(
            selection_set_id=selection_set_id,
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin=origin,
            panel_id=panel_id,
            mode=mode,
            artifact_id=artifact_id,
            metadata=metadata,
        )
        self._active_set = state

        focus_row_id = self._apply_focus_policy(
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin=origin,
            panel_id=panel_id,
            selection_set_id=selection_set_id,
            policy=update_focus_policy,
        )

        self.events.publish(
            "selection.set.changed",
            {
                "dataset_id": dataset_id,
                "selection_set_id": selection_set_id,
                "count": len(row_ids),
                "row_ids": row_ids,
                "artifact_id": artifact_id,
                "mode": mode,
                "origin": origin,
                "panel_id": panel_id,
                "focus_row_id": focus_row_id,
                "timestamp": state.timestamp,
            },
        )

        return state

    def _apply_focus_policy(
        self,
        *,
        dataset_id: str,
        row_ids: List[str],
        origin=None,
        panel_id=None,
        selection_set_id=None,
        policy="preserve_or_first",
    ):
        if not row_ids:
            return None

        current = self._focus

        if policy == "unchanged":
            return current.row_id

        if policy == "first":
            self.set_focus(
                dataset_id=dataset_id,
                row_id=row_ids[0],
                origin=origin,
                panel_id=panel_id,
                selection_set_id=selection_set_id,
            )
            return row_ids[0]

        if policy == "preserve_or_first":
            if current.dataset_id == dataset_id and current.row_id in row_ids:
                return current.row_id

            self.set_focus(
                dataset_id=dataset_id,
                row_id=row_ids[0],
                origin=origin,
                panel_id=panel_id,
                selection_set_id=selection_set_id,
            )
            return row_ids[0]

        return None
    
    def snapshot(self) -> dict[str, Any]:
        focus = self.get_focus()
        active_set = self.get_active_set()

        focus_snapshot = None
        if focus is not None and focus.dataset_id is not None and focus.row_id is not None:
            focus_snapshot = {
                "dataset_id": focus.dataset_id,
                "row_id": focus.row_id,
                "origin": focus.origin,
                "panel_id": focus.panel_id,
                "timestamp": focus.timestamp,
                "metadata": dict(getattr(focus, "metadata", {}) or {}),
            }

        selection_set_snapshot = None
        if active_set is not None:
            selection_set_snapshot = {
                "selection_set_id": active_set.selection_set_id,
                "dataset_id": active_set.dataset_id,
                "row_ids": list(active_set.row_ids),
                "origin": active_set.origin,
                "panel_id": active_set.panel_id,
                "mode": active_set.mode,
                "artifact_id": active_set.artifact_id,
                "timestamp": active_set.timestamp,
                "metadata": dict(active_set.metadata or {}),
            }

        return {
            "focus": focus_snapshot,
            "selection_set": selection_set_snapshot,
        }

    def restore_snapshot(self, snapshot: dict[str, Any], *, publish: bool = True) -> None:
        if not snapshot:
            return

        selection_set = snapshot.get("selection_set")
        focus = snapshot.get("focus")

        if selection_set:
            if publish:
                self.set_selection_set(
                    dataset_id=selection_set["dataset_id"],
                    row_ids=list(selection_set.get("row_ids") or []),
                    origin=selection_set.get("origin") or "workspace.restore",
                    panel_id=selection_set.get("panel_id"),
                    mode=selection_set.get("mode") or "replace",
                    metadata=dict(selection_set.get("metadata") or {}),
                    create_artifact=False,
                    update_focus_policy="unchanged",
                )

                if self._active_set is not None:
                    self._active_set.selection_set_id = selection_set.get(
                        "selection_set_id",
                        self._active_set.selection_set_id,
                    )
                    self._active_set.artifact_id = selection_set.get("artifact_id")
            else:
                self._active_set = SelectionSetState(
                    selection_set_id=selection_set.get("selection_set_id") or f"selset:{uuid.uuid4().hex[:8]}",
                    dataset_id=selection_set["dataset_id"],
                    row_ids=[str(row_id) for row_id in selection_set.get("row_ids") or []],
                    origin=selection_set.get("origin"),
                    panel_id=selection_set.get("panel_id"),
                    mode=selection_set.get("mode") or "replace",
                    artifact_id=selection_set.get("artifact_id"),
                    metadata=dict(selection_set.get("metadata") or {}),
                )
        else:
            self._active_set = None

        if focus:
            if publish:
                self.set_focus(
                    dataset_id=focus["dataset_id"],
                    row_id=focus["row_id"],
                    origin=focus.get("origin") or "workspace.restore",
                    panel_id=focus.get("panel_id"),
                    metadata=dict(focus.get("metadata") or {}),
                )
            else:
                self._focus = FocusState(
                    dataset_id=focus.get("dataset_id"),
                    row_id=str(focus.get("row_id")),
                    origin=focus.get("origin"),
                    panel_id=focus.get("panel_id"),
                    metadata=dict(focus.get("metadata") or {}),
                )