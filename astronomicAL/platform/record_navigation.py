from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Iterable, Literal, Optional

NavigationScope = Literal["dataset", "selection"]


class NavigationError(RuntimeError):
    """Raised when a requested record-navigation operation is unavailable."""


@dataclass(frozen=True)
class NavigationState:
    dataset_id: Optional[str] = None
    scope: NavigationScope = "dataset"
    row_id: Any = None
    position: Optional[int] = None
    row_count: int = 0
    dataset_row_count: int = 0
    selection_count: int = 0
    id_column: Optional[str] = None
    has_focus: bool = False
    focus_in_scope: bool = False
    can_previous: bool = False
    can_next: bool = False
    can_add_to_selection: bool = False
    can_remove_from_selection: bool = False
    message: str = "Load a dataset to start browsing."

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class RecordNavigationManager:
    """Owns global record navigation while SelectionManager owns focus.

    The manager deliberately does not keep a second canonical focused record.
    Navigation changes are published by calling SelectionManager.set_focus(), and
    external focus changes are reflected back into NavigationState.
    """

    EVENT_TOPICS = (
        "dataset.active.changed",
        "dataset.updated",
        "dataset.mapping.updated",
        "selection.focus.changed",
        "selection.focus.cleared",
        "selection.set.changed",
        "selection.set.cleared",
    )

    def __init__(self, *, datasets: Any, selection: Any, events: Any) -> None:
        self.datasets = datasets
        self.selection = selection
        self.events = events
        self._lock = RLock()
        self._scope: NavigationScope = "dataset"
        self._position_cache: dict[tuple[str, str], int] = {}
        self._subscriptions: list[Any] = []
        self._disposed = False
        self._state = NavigationState()

        for topic in self.EVENT_TOPICS:
            self._subscriptions.append(
                events.subscribe(
                    topic,
                    self._on_platform_event,
                    owner_id="platform.record_navigation",
                    owner_label="Record navigation",
                    owner_kind="platform_service",
                )
            )
        self.refresh(force_publish=False)

    @property
    def scope(self) -> NavigationScope:
        return self._scope

    def get_state(self) -> NavigationState:
        with self._lock:
            return self._state

    def set_scope(self, scope: NavigationScope, *, origin: str = "toolbar") -> NavigationState:
        if scope not in ("dataset", "selection"):
            raise ValueError(f"Unknown navigation scope: {scope!r}")
        with self._lock:
            self._scope = scope
        state = self.refresh(force_publish=True)
        self.events.publish(
            "navigation.scope.changed",
            {
                "scope": scope,
                "dataset_id": state.dataset_id,
                "origin": origin,
            },
        )
        return state

    def refresh(self, *, force_publish: bool = False) -> NavigationState:
        previous = self.get_state()
        state = self._build_state()
        with self._lock:
            self._state = state
        if force_publish or state != previous:
            self.events.publish("navigation.state.changed", state.to_payload())
        return state

    def go_to_position(self, position: int, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.dataset_id is None:
            raise NavigationError("No active dataset is available.")
        if state.id_column is None:
            raise NavigationError("Map record_id before navigating records.")
        if state.row_count <= 0:
            if state.scope == "selection":
                raise NavigationError("The active selection is empty.")
            raise NavigationError("The active dataset is empty.")

        position = int(position)
        if position < 0 or position >= state.row_count:
            raise NavigationError(
                "Record position must be between "
                f"0 and {state.row_count - 1:,}."
            )

        if state.scope == "selection":
            row_ids = self._selection_row_ids(state.dataset_id)
            row_id = row_ids[position]
            dataset_position = self._cached_position(state.dataset_id, row_id)
        else:
            row_id = self._row_id_at_dataset_position(
                state.dataset_id,
                position,
                state.id_column,
            )
            dataset_position = position

        self._remember_position(state.dataset_id, row_id, dataset_position)
        self.selection.set_focus(
            dataset_id=state.dataset_id,
            row_id=row_id,
            origin=origin,
            panel_id=None,
            metadata={
                "index": dataset_position,
                "navigation_position": position,
                "navigation_scope": state.scope,
                "id_column": state.id_column,
            },
        )
        return self.refresh(force_publish=True)

    def previous(self, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.position is None:
            return self.go_to_position(0, origin=origin)
        if not state.can_previous:
            raise NavigationError("Already at the first record in this scope.")
        return self.go_to_position(state.position - 1, origin=origin)

    def next(self, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.position is None:
            return self.go_to_position(0, origin=origin)
        if not state.can_next:
            raise NavigationError("Already at the last record in this scope.")
        return self.go_to_position(state.position + 1, origin=origin)

    def first(self, *, origin: str = "toolbar") -> NavigationState:
        return self.go_to_position(0, origin=origin)

    def last(self, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.row_count <= 0:
            raise NavigationError("There are no records in the current scope.")
        return self.go_to_position(state.row_count - 1, origin=origin)

    def find_position(self, row_id: Any) -> Optional[int]:
        """Return a zero-based position in the current scope.

        This may query the dataset backend and is therefore suitable for a
        JobManager worker when called from a UI controller.
        """
        state = self.refresh()
        if state.dataset_id is None or state.id_column is None:
            return None
        if state.scope == "selection":
            wanted = str(row_id)
            for position, candidate in enumerate(self._selection_row_ids(state.dataset_id)):
                if str(candidate) == wanted:
                    return position
            return None
        position = self.datasets.find_position_by_id(
            state.dataset_id,
            row_id,
            id_column=state.id_column,
        )
        if position is not None:
            self._remember_position(state.dataset_id, row_id, int(position))
        return None if position is None else int(position)

    def go_to_id(self, row_id: Any, *, origin: str = "toolbar") -> NavigationState:
        query = str(row_id).strip()
        if not query:
            raise NavigationError("Enter a record ID.")
        position = self.find_position(query)
        if position is None:
            raise NavigationError(f"No record with ID {query!r} was found in this scope.")
        return self.go_to_position(position, origin=origin)

    def resolve_focus_position(self) -> NavigationState:
        """Resolve an externally focused record's position without changing focus.

        Plot, table, and gallery plugins often publish only dataset_id and row_id.
        The toolbar can run this method through JobManager so its position display
        remains accurate without blocking the UI thread.
        """
        state = self.refresh()
        if not state.has_focus or state.row_id is None or state.position is not None:
            return state
        position = self.find_position(state.row_id)
        if position is not None and state.dataset_id is not None:
            if state.scope == "dataset":
                self._remember_position(state.dataset_id, state.row_id, position)
            return self.refresh(force_publish=True)
        return self.refresh()

    def add_focus_to_selection(self, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.dataset_id is None or not state.has_focus:
            raise NavigationError("Focus a record before adding it to the selection.")
        row_ids = self._selection_row_ids(state.dataset_id)
        if not self._contains_id(row_ids, state.row_id):
            row_ids.append(state.row_id)
        self.selection.set_selection_set(
            dataset_id=state.dataset_id,
            row_ids=row_ids,
            origin=origin,
            panel_id=None,
            mode="replace",
            metadata={"source": "application_toolbar"},
            create_artifact=False,
            update_focus_policy="preserve_or_first",
        )
        return self.refresh(force_publish=True)

    def remove_focus_from_selection(self, *, origin: str = "toolbar") -> NavigationState:
        state = self.refresh()
        if state.dataset_id is None or not state.has_focus:
            raise NavigationError("Focus a record before removing it from the selection.")
        row_ids = [
            candidate
            for candidate in self._selection_row_ids(state.dataset_id)
            if str(candidate) != str(state.row_id)
        ]
        if row_ids:
            self.selection.set_selection_set(
                dataset_id=state.dataset_id,
                row_ids=row_ids,
                origin=origin,
                panel_id=None,
                mode="replace",
                metadata={"source": "application_toolbar"},
                create_artifact=False,
                update_focus_policy="preserve_or_first",
            )
        else:
            self.selection.clear_selection_set(origin=origin, panel_id=None)
        return self.refresh(force_publish=True)

    def clear_selection(self, *, origin: str = "toolbar") -> NavigationState:
        self.selection.clear_selection_set(origin=origin, panel_id=None)
        return self.refresh(force_publish=True)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for subscription in list(self._subscriptions):
            try:
                self.events.unsubscribe(subscription)
            except Exception:
                pass
        self._subscriptions.clear()

    def _build_state(self) -> NavigationState:
        try:
            dataset_id = self.datasets.active_id()
        except Exception:
            dataset_id = None
        if not dataset_id:
            return NavigationState(scope=self._scope)

        try:
            dataset_row_count = max(0, int(self.datasets.row_count(dataset_id)))
        except Exception:
            dataset_row_count = 0

        id_column = self.datasets.get_mapping(dataset_id, "record_id", default=None)
        if id_column is not None:
            id_column = str(id_column)

        focus = self.selection.get_focus()
        focus_matches = bool(
            focus is not None and str(getattr(focus, "dataset_id", "")) == str(dataset_id)
        )
        row_id = getattr(focus, "row_id", None) if focus_matches else None
        metadata = dict(getattr(focus, "metadata", {}) or {}) if focus_matches else {}

        selected_ids = self._selection_row_ids(dataset_id)
        selection_count = len(selected_ids)
        scope = self._scope
        scope_count = dataset_row_count if scope == "dataset" else selection_count

        position: Optional[int] = None
        if focus_matches:
            if scope == "selection":
                position = self._index_of_id(selected_ids, row_id)
            else:
                metadata_position = metadata.get("index")
                if isinstance(metadata_position, int) and 0 <= metadata_position < dataset_row_count:
                    position = metadata_position
                else:
                    position = self._cached_position(dataset_id, row_id)

        focus_in_scope = position is not None
        in_selection = focus_matches and self._contains_id(selected_ids, row_id)

        if id_column is None:
            message = "Map record_id to enable record navigation."
        elif scope == "selection" and selection_count == 0:
            message = "The active selection is empty."
        elif scope_count == 0:
            message = "The active dataset is empty."
        elif focus_matches and not focus_in_scope:
            message = "Focused record is outside the current navigation scope."
        elif focus_matches:
            message = "Ready"
        else:
            message = "Choose a record or press Next to begin."

        return NavigationState(
            dataset_id=str(dataset_id),
            scope=scope,
            row_id=row_id,
            position=position,
            row_count=scope_count,
            dataset_row_count=dataset_row_count,
            selection_count=selection_count,
            id_column=id_column,
            has_focus=focus_matches,
            focus_in_scope=focus_in_scope,
            can_previous=position is not None and position > 0,
            can_next=position is not None and position + 1 < scope_count,
            can_add_to_selection=focus_matches and not in_selection,
            can_remove_from_selection=focus_matches and in_selection,
            message=message,
        )

    def _selection_row_ids(self, dataset_id: str) -> list[Any]:
        active_set = self.selection.get_active_set()
        if active_set is None:
            return []
        if str(getattr(active_set, "dataset_id", "")) != str(dataset_id):
            return []
        return list(getattr(active_set, "row_ids", []) or [])

    def _row_id_at_dataset_position(
        self,
        dataset_id: str,
        position: int,
        id_column: str,
    ) -> Any:
        columns = None if id_column == "Use Index" else [id_column]
        frame = self.datasets.get_row_by_position(
            dataset_id,
            position,
            columns=columns,
        )
        if frame is None or frame.empty:
            raise NavigationError(f"Could not read record {position + 1:,}.")
        if id_column == "Use Index":
            try:
                return frame.index[0]
            except Exception as exc:
                raise NavigationError(
                    "Index-based record IDs are unavailable for this dataset backend. "
                    "Map record_id to a concrete column."
                ) from exc
        if id_column not in frame.columns:
            raise NavigationError(f"Mapped record ID column {id_column!r} is unavailable.")
        return frame.iloc[0][id_column]

    def _on_platform_event(self, _topic: str, _payload: Any) -> None:
        if not self._disposed:
            self.refresh()

    def _remember_position(
        self,
        dataset_id: str,
        row_id: Any,
        position: Optional[int],
    ) -> None:
        if position is None:
            return
        key = (str(dataset_id), str(row_id))
        with self._lock:
            if len(self._position_cache) >= 8192:
                # Dicts preserve insertion order. Remove the oldest quarter rather
                # than turning every navigation event into an LRU bookkeeping cost.
                for old_key in list(self._position_cache)[:2048]:
                    self._position_cache.pop(old_key, None)
            self._position_cache[key] = int(position)

    def _cached_position(self, dataset_id: str, row_id: Any) -> Optional[int]:
        with self._lock:
            return self._position_cache.get((str(dataset_id), str(row_id)))

    @staticmethod
    def _contains_id(row_ids: Iterable[Any], row_id: Any) -> bool:
        wanted = str(row_id)
        return any(str(candidate) == wanted for candidate in row_ids)

    @staticmethod
    def _index_of_id(row_ids: Iterable[Any], row_id: Any) -> Optional[int]:
        wanted = str(row_id)
        for position, candidate in enumerate(row_ids):
            if str(candidate) == wanted:
                return position
        return None