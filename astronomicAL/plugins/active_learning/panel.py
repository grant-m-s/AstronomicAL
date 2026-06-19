# BUG: pick random starting -> label -> train -> predict -> query -> label -> train (Attempted) -> [ActiveLearningPanel] Scratch training failed: Resolved recipe column(s) are not present in the dataset: image_path |
# BUG: protocol options for val and test dont automatically line up with what was chosen earlier.

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest

from . import actions
from . import state as al_state

_AL_LAYOUT_CSS_INSTALLED = False

class ActiveLearningPanel:
    state_version = 1

    def __init__(self, context: Any, restore_state: Optional[Mapping[str, Any]] = None) -> None:
        self.context = context
        self._install_layout_css_once()

        self._current_session_artifact_id: Optional[str] = None
        self._active_job_handle: Any = None
        self._job_running = False

        self._disposed = False
        self._subscriptions: List[Any] = []
        self.recipe_param_widgets: Dict[str, Any] = {}
        self.recipe_param_fields: Dict[str, Any] = {}
        self.protocol_widgets: Dict[str, Any] = {}
        self.protocol_fields: Dict[str, Any] = {}

        restore_state = dict(restore_state or {})
        self._current_session_artifact_id = restore_state.get("session_artifact_id")

        self.dataset_select = pn.widgets.Select(name="Pool dataset", options=[])
        self.session_select = pn.widgets.Select(name="Session", options={})
        self.predictions_select = pn.widgets.Select(name="Predictions artifact", options={})
        self.strategy_select = pn.widgets.Select(name="Query strategy", options={})

        self.recipe_select = pn.widgets.Select(name="core.ml recipe", options={})
        self.recipe_card = pn.pane.Markdown("No recipe selected.", sizing_mode="stretch_width")
        self.recipe_params_area = pn.Column(sizing_mode="stretch_width")

        self.label_options_select = pn.widgets.MultiChoice(
            name="Allowed labels",
            options=[],
            value=[],
            solid=False,
        )
        self.target_column = pn.widgets.Select(
            name="Training label column",
            options=[],
            value=None,
        )
        self.validation_dataset_select = pn.widgets.Select(
            name="Validation dataset",
            options={"Split from AL training rows": ""},
            value="",
        )
        self.test_dataset_select = pn.widgets.Select(
            name="Test dataset",
            options={"Split from AL training rows": ""},
            value="",
        )
        self.al_protocol = pn.widgets.Select(
            name="AL protocol",
            options={
                "Review mode: train on verified labels": "review",
                "Benchmark mode: fixed holdout datasets": "benchmark",
            },
            value=str(restore_state.get("al_protocol") or "review"),
        )

        self.seed = pn.widgets.IntInput(
            name="Initialisation/random seed",
            value=int(restore_state.get("seed", 42)),
            start=0,
        )
        self.initial_k = pn.widgets.IntInput(
            name="Initial random points",
            value=int(restore_state.get("initial_k", 20)),
            start=0,
        )
        self.query_k = pn.widgets.IntInput(
            name="Query batch size",
            value=int(restore_state.get("query_k", 200)),
            start=1,
        )

        self.label_select = pn.widgets.Select(name="Label current focus", options={})
        self.advance_after_label = pn.widgets.Checkbox(
            name="Advance to next selected row after labelling",
            value=bool(restore_state.get("advance_after_label", True)),
        )
        self.auto_label_n = pn.widgets.IntInput(
            name="Auto-label next N points",
            value=int(restore_state.get("auto_label_n", 50)),
            start=1,
        )

        self.auto_label_button = pn.widgets.Button(
            name="Label next N most informative points",
            button_type="primary",
        )

        self.auto_label_source_column = pn.widgets.Select(
            name="Auto-label from dataset column",
            options=[],
            value=restore_state.get("auto_label_source_column"),
        )

        self.refresh_button = pn.widgets.Button(name="Refresh", button_type="light")
        self.start_button = pn.widgets.Button(name="Start / random sample", button_type="primary")
        self.query_button = pn.widgets.Button(name="Query top-k from predictions", button_type="primary")
        self.label_button = pn.widgets.Button(name="Record label / verification", button_type="success")
        self.unsure_button = pn.widgets.Button(name="Mark current as Unsure", button_type="warning")
        self.train_button = pn.widgets.Button(
            name="Add verified labels to training set + train from scratch",
            button_type="danger",
        )

        # Short, muted hints shown next to gated buttons explaining why they
        # are disabled (e.g. "label a point before training").
        self.query_hint = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.train_hint = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.status = pn.pane.Alert("", alert_type="info", visible=False)
        self.summary = pn.pane.Markdown("")
        self.batch_table = _make_table()
        self.labels_table = _make_table()

        self.session_json = pn.pane.JSON(
            {},
            depth=2,
            sizing_mode="stretch_width",
            height=260,
            styles={
                "max-width": "100%",
                "width": "100%",
                "overflow": "auto",
                "box-sizing": "border-box",
            },
        )

        self._normalise_widget_layout()

        self.refresh_button.on_click(self._refresh_clicked)
        self.start_button.on_click(self._start_clicked)
        self.query_button.on_click(self._query_clicked)
        self.label_button.on_click(self._label_clicked)
        self.unsure_button.on_click(self._unsure_clicked)
        self.train_button.on_click(self._train_clicked)
        self.auto_label_button.on_click(self._auto_label_next_clicked)

        self.session_select.param.watch(self._session_selected, "value")
        self.dataset_select.param.watch(self._dataset_changed, "value")
        self.target_column.param.watch(self._target_column_changed, "value")
        self.recipe_select.param.watch(self._recipe_changed, "value")
        self.predictions_select.param.watch(self._predictions_changed, "value")
        self.label_options_select.param.watch(
            lambda *_: self._set_label_select_options(),
            "value",
        )

        self._subscribe_refresh_events()
        self.refresh()
        self._restore_widget_state(restore_state)

    def panel(self):
        # Tabs are back for the query -> label -> train workflow, but with
        # dynamic=True: an inactive tab's content is NOT rendered until you
        # switch to it, so it is never measured while display:none. That was
        # the cause of the overlapping/collapsed widgets with the old
        # dynamic=False tabs.
        start_section = self._tab_body(
            pn.Column(
                self.dataset_select,
                self.session_select,
                self.al_protocol,
                self.recipe_select,
                self.recipe_card,
                self.label_options_select,
                self.target_column,
                self.seed,
                self.initial_k,
                self.validation_dataset_select,
                self.test_dataset_select,
                self.start_button,
                sizing_mode="stretch_width",
            ),
        )

        query_section = self._tab_body(
            pn.Column(
                self.predictions_select,
                self.strategy_select,
                self.query_k,
                self.query_hint,
                self.query_button,
                sizing_mode="stretch_width",
            ),
        )

        review_section = self._tab_body(
            pn.Column(
                self.label_select,
                self.advance_after_label,
                self.label_button,
                self.unsure_button,
                pn.layout.Divider(margin=(8, 0, 8, 0)),
                self.auto_label_source_column,
                self.auto_label_n,
                self.auto_label_button,
                sizing_mode="stretch_width",
            ),
        )

        # The recipe params live in a fixed-height scroll box. A definite
        # height means the container can't under-report its size, which was
        # letting the train button and the (out-of-tabs) session summary draw
        # over the params tail. The train button sits BELOW this box, so it is
        # always visible without scrolling inside the box.
        params_scroller = pn.Column(
            self.recipe_params_area,
            height=340,
            scroll=True,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "border": "1px solid rgba(0,0,0,0.10)",
                "border-radius": "4px",
                "padding": "6px",
            },
        )

        train_section = self._tab_body(
            pn.Column(
                params_scroller,
                pn.layout.Divider(margin=(8, 0, 8, 0)),
                self.train_hint,
                self.train_button,
                sizing_mode="stretch_width",
            ),
        )

        controls_tabs = pn.Tabs(
            ("Start", start_section),
            ("Query", query_section),
            ("Review", review_section),
            ("Train", train_section),
            dynamic=True,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
            },
            css_classes=["al-controls-tabs"],
        )

        results_tabs = pn.Tabs(
            ("Current query batch", self.batch_table),
            ("Labelled / verified / unsure", self.labels_table),
            ("Session JSON", self.session_json),
            dynamic=True,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
            },
            css_classes=["al-results-tabs"],
        )

        return pn.Column(
            pn.pane.Markdown(
                "### Active Learning",
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
                styles={
                    "font-size": "15px",
                    "font-weight": "600",
                    "line-height": "1.2",
                    "overflow-wrap": "anywhere",
                },
            ),
            self.status,
            self.refresh_button,
            controls_tabs,
            pn.layout.Divider(margin=(8, 0, 8, 0)),
            self.summary,
            results_tabs,
            sizing_mode="stretch_width",
            scroll=True,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "padding": "8px",
            },
            css_classes=["al-panel-root"],
        )

    def refresh(self) -> None:
        if self._disposed:
            return
        self._refresh_datasets()
        self._refresh_sessions()
        self._refresh_predictions()
        self._refresh_strategies()
        self._refresh_recipes()
        self._refresh_column_widgets()
        self._refresh_auto_label_source_column()
        self._refresh_validation_test_datasets()
        self._refresh_recipe_params()
        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()
        self._set_label_select_options()
        self._refresh_review_label_for_focus()
        self._refresh_session_views()
        self._update_action_gating()

    def dispose(self) -> None:
        self._disposed = True
        handle = self._active_job_handle
        self._active_job_handle = None
        cancel = getattr(handle, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)
        if callable(unsubscribe):
            for sub in list(self._subscriptions):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions = []

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "session_artifact_id": self._current_session_artifact_id,
            "dataset_id": self.dataset_select.value,
            "recipe_id": self.recipe_select.value,
            "label_options": list(self.label_options_select.value or []),
            "target_column": self.target_column.value,
            "validation_dataset_id": self.validation_dataset_select.value,
            "test_dataset_id": self.test_dataset_select.value,
            "al_protocol": self.al_protocol.value,
            "seed": int(self.seed.value or 0),
            "initial_k": int(self.initial_k.value or 0),
            "query_k": int(self.query_k.value or 1),
            "strategy_id": self.strategy_select.value,
            "recipe_params": self._recipe_params(),
            "advance_after_label": bool(self.advance_after_label.value),
            "auto_label_n": int(self.auto_label_n.value or 1),
            "auto_label_source_column": self.auto_label_source_column.value,
        }

    def _refresh_clicked(self, *_: Any) -> None:
        self.refresh()
        self._set_status("Refreshed.", "info")

    def _start_clicked(self, *_: Any) -> None:
        try:
            dataset_id = self._require_dataset()
            labels = self._current_label_options()

            if not labels:
                raise ValueError(
                    "No label options are available. Choose a populated label column, "
                    "or choose a recipe/predictions artifact that exposes class labels."
                )

            request = ActionRequest(
                dataset_id=dataset_id,
                params={
                    "dataset_id": dataset_id,
                    "pool_dataset_id": dataset_id,
                    "recipe_id": self.recipe_select.value or "",
                    "label_options": labels,
                    "target_column": self.target_column.value or "al_label",
                    "validation_dataset_id": self.validation_dataset_select.value or "",
                    "test_dataset_id": self.test_dataset_select.value or "",
                    "al_protocol": self.al_protocol.value or "review",
                    "initial_k": int(self.initial_k.value or 0),
                    "seed": int(self.seed.value or 0),
                    "make_selection": True,
                },
                origin="core.active_learning.panel",
            )
            result = actions.start_session_action(self.context, request)
            self._use_action_session_result(result)
            self._set_status(
                f"Started session with {len(result.get('row_ids') or [])} random rows.",
                "success",
            )
        except Exception as exc:
            self._set_error("Could not start active-learning session", exc)

    def _query_clicked(self, *_: Any) -> None:
        try:
            session_artifact_id = self._require_session()
            self._require_trained()
            predictions_artifact_id = self._require_prediction_artifact()

            request = ActionRequest(
                artifact_id=predictions_artifact_id,
                params={
                    "session_artifact_id": session_artifact_id,
                    "predictions_artifact_id": predictions_artifact_id,
                    "strategy_id": self.strategy_select.value or "least_confidence",
                    "k": int(self.query_k.value or 200),
                    "seed": int(self.seed.value or 0),
                    "make_selection": True,
                },
                origin="core.active_learning.panel",
            )

            manager = getattr(self.context, "plugins", None)
            run_action = getattr(manager, "run_action", None)
            if callable(run_action):
                self._set_running(True)
                self._set_status("Querying prediction artifact...", "info")
                self._active_job_handle = run_action(
                    "core.active_learning.query_batch",
                    self.context,
                    request,
                    on_done=self._on_query_done,
                    on_error=self._on_query_error,
                )
            else:
                result = actions.query_batch_action(self.context, request)
                self._on_query_done(result)

        except Exception as exc:
            self._set_running(False)
            self._set_error("Could not create active-learning query batch", exc)

    def _label_clicked(self, *_: Any) -> None:
        self._record_current_label(self.label_select.value)

    def _unsure_clicked(self, *_: Any) -> None:
        self._record_current_label(al_state.UNSURE_LABEL)

    def _record_current_label(self, label: Any) -> None:
        try:
            session_artifact_id = self._require_session()
            row_id = self._current_focus_row_id()
            if row_id is None:
                raise ValueError("No focused row is active. Select a row first.")

            request = ActionRequest(
                params={
                    "session_artifact_id": session_artifact_id,
                    "row_id": row_id,
                    "label": label,
                    "source": "active_learning_panel",
                },
                origin="core.active_learning.panel",
            )
            result = actions.record_label_action(self.context, request)
            self._use_action_session_result(result)

            display = al_state.display_label(result.get("label"))
            self._set_status(f"Recorded {display!r} for row {row_id}.", "success")

            if bool(self.advance_after_label.value):
                self._advance_focus_after(row_id)

        except Exception as exc:
            self._set_error("Could not record label", exc)

    def _refresh_auto_label_source_column(self) -> None:
        """Refresh physical dataset columns usable for benchmark auto-labelling."""
        current = self.auto_label_source_column.value
        dataset_id = self.dataset_select.value

        columns = self._physical_columns_for_dataset(dataset_id)
        options = {col: col for col in columns}

        self.auto_label_source_column.options = options

        if current in options.values():
            self.auto_label_source_column.value = current
            return

        # Prefer common human-readable label columns first.
        for candidate in (
            "label_name",
            "target_label_name",
            "class_name",
            "target_name",
            "class_label",
            "target_label",
            "label",
            "class",
            "target",
            "fine_label",
            "coarse_label",
        ):
            if candidate in columns:
                self.auto_label_source_column.value = candidate
                return

        self.auto_label_source_column.value = columns[0] if columns else None


    def _physical_columns_for_dataset(self, dataset_id: Any) -> List[str]:
        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return []

        try:
            return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            pass

        try:
            df = self.context.datasets.get_df(dataset_id)
            return [str(col) for col in df.columns]
        except Exception:
            return []


    def _resolve_auto_label_source_column(
        self,
        *,
        dataset_id: str,
        session: Mapping[str, Any],
    ) -> str:
        """Resolve the physical source column to read benchmark labels from.

        This is intentionally separate from the AL training target column.
        """
        columns = set(self._physical_columns_for_dataset(dataset_id))

        explicit = str(self.auto_label_source_column.value or "").strip()
        if explicit and explicit in columns:
            return explicit

        # If the UI value is a semantic mapping name rather than a physical column,
        # try resolving it through DatasetManager mappings.
        for semantic in (
            explicit,
            "target_label_name",
            "target_label",
            "label_name",
            "label",
            "class_label",
            "class",
            "target",
        ):
            if not semantic:
                continue
            try:
                mapped = self.context.datasets.get_mapping(dataset_id, semantic)
            except Exception:
                mapped = None

            mapped = str(mapped or "").strip()
            if mapped and mapped in columns:
                return mapped

        # Last-resort conventional names.
        for candidate in (
            "label_name",
            "target_label_name",
            "class_name",
            "target_name",
            "class_label",
            "target_label",
            "label",
            "class",
            "target",
            "fine_label",
            "coarse_label",
        ):
            if candidate in columns:
                return candidate

        raise ValueError(
            "Could not find a physical source label column for auto-labelling. "
            f"Choose one in 'Auto-label from dataset column'. "
            f"Available columns: {sorted(columns)}"
        )


    def _normalise_auto_label_value(
        self,
        value: Any,
        *,
        label_options: List[str],
    ) -> Optional[str]:
        """Convert source label values into AL label values.

        Handles the common CIFAR/HuggingFace case where labels are integer class
        IDs and label_options contains class names.
        """
        if value is None or pd.isna(value):
            return None

        # Numeric class index -> label option.
        try:
            if label_options:
                index = int(value)
                if 0 <= index < len(label_options):
                    return str(label_options[index])
        except Exception:
            pass

        text = str(value).strip()
        return text or None

    def _auto_label_next_clicked(self, *_: Any) -> None:
        """Record labels for the next N ordered rows in the active selection set.

        "Next N" starts at the current focus row. If the current focus is the
        first queried row and N=50, rows 1..50 are labelled, then focus moves to
        row 51.

        Labels are read from the selected target column on the pool/source dataset.
        This is intended for benchmark/debug workflows where the ground-truth label
        column already exists.
        """
        try:
            session_artifact_id = self._require_session()
            session = self._load_current_session() or {}

            dataset_id = str(
                session.get("pool_dataset_id")
                or session.get("dataset_id")
                or self.dataset_select.value
                or ""
            ).strip()
            if not dataset_id:
                raise ValueError("Could not determine the active-learning pool dataset.")

            source_label_column = self._resolve_auto_label_source_column(
                dataset_id=dataset_id,
                session=session,
            )
            label_options = self._current_label_options()

            n = max(1, int(self.auto_label_n.value or 1))

            selection = getattr(self.context, "selection", None)
            if selection is None or not hasattr(selection, "get_active_set"):
                raise ValueError("No active selection set is available.")

            active_set = selection.get_active_set()
            if active_set is None:
                raise ValueError("No active selection set is available.")

            active_dataset_id = str(getattr(active_set, "dataset_id", "") or "")
            actions._assert_dataset_ids_match(
                dataset_id,
                active_dataset_id,
                action="auto_label_next active selection",
            )

            ordered_row_ids = [
                str(row_id)
                for row_id in (getattr(active_set, "row_ids", []) or [])
                if row_id is not None
            ]
            if not ordered_row_ids:
                raise ValueError("The active selection set has no rows.")

            focus_row_id = self._current_focus_row_id()
            start_index = 0
            if focus_row_id is not None:
                try:
                    start_index = ordered_row_ids.index(str(focus_row_id))
                except ValueError:
                    start_index = 0

            already_seen = set(str(row_id) for row_id in (session.get("ignored_row_ids", []) or []))
            already_seen.update(str(row_id) for row_id in ((session.get("labels") or {}).keys()))

            # Start at current focus and take the next N not-yet-recorded rows.
            to_label: List[str] = []
            cursor = start_index
            while cursor < len(ordered_row_ids) and len(to_label) < n:
                row_id = ordered_row_ids[cursor]
                if row_id not in already_seen:
                    to_label.append(row_id)
                cursor += 1

            if not to_label:
                raise ValueError("No unlabelled rows remain at or after the current focus.")

            labels_by_row_id = self._read_existing_labels_for_rows(
                dataset_id=dataset_id,
                row_ids=to_label,
                target_column=source_label_column,
            )

            recorded = 0
            skipped_missing: List[str] = []
            latest_session_artifact_id = session_artifact_id

            for row_id in to_label:
                raw_label = labels_by_row_id.get(str(row_id))
                label = self._normalise_auto_label_value(
                    raw_label,
                    label_options=label_options,
                )
                if label is None or str(label).strip() == "":
                    skipped_missing.append(str(row_id))
                    continue

                request = ActionRequest(
                    params={
                        "session_artifact_id": latest_session_artifact_id,
                        "dataset_id": dataset_id,
                        "row_id": row_id,
                        "label": label,
                        "source": "active_learning_panel.auto_label_next",
                    },
                    origin="core.active_learning.panel",
                )

                result = actions.record_label_action(self.context, request)
                latest_session_artifact_id = str(
                    result.get("session_artifact_id") or latest_session_artifact_id
                )
                recorded += 1

            if latest_session_artifact_id:
                self._current_session_artifact_id = latest_session_artifact_id

            # Move focus to the next row after the consumed block, i.e. if N=50
            # starts at rank 1, focus moves to rank 51.
            next_focus = None
            if cursor < len(ordered_row_ids):
                next_focus = ordered_row_ids[cursor]

            if next_focus is not None and hasattr(selection, "set_focus"):
                selection.set_focus(
                    dataset_id=dataset_id,
                    row_id=next_focus,
                    origin="core.active_learning.panel.auto_label_next",
                    selection_set_id=getattr(active_set, "selection_set_id", None),
                    metadata={
                        "reason": "after_auto_label_next",
                        "auto_labelled_count": recorded,
                        "requested_n": n,
                    },
                )

            self.refresh()

            message = (
                f"Auto-labelled {recorded} row(s) from `{source_label_column}`."
            )
            if skipped_missing:
                message += f" Skipped {len(skipped_missing)} row(s) with missing labels."
            if next_focus is not None:
                message += f" Focus moved to row `{next_focus}`."

            self._set_status(message, "success" if recorded else "warning")

        except Exception as exc:
            self._set_error("Could not auto-label next points", exc)


    def _read_existing_labels_for_rows(
        self,
        *,
        dataset_id: str,
        row_ids: List[str],
        target_column: str,
    ) -> Dict[str, Any]:
        """Read existing labels for specific row IDs from a physical dataset column.

        This is used by the Review tab's benchmark auto-labelling flow.

        Important:
        context.datasets.get_rows_by_ids(...) may return a reduced dataframe that
        does not contain every physical source column. If it drops target_column,
        fall back to a projected/full dataframe read and filter manually.
        """
        if not row_ids:
            return {}

        dataset_id = str(dataset_id or "").strip()
        target_column = str(target_column or "").strip()
        row_ids = [str(row_id) for row_id in row_ids]
        row_id_set = set(row_ids)

        if not dataset_id:
            raise ValueError("No dataset_id supplied for auto-labelling.")
        if not target_column:
            raise ValueError("No auto-label source column supplied.")

        id_column = actions._resolve_record_id_column(self.context, dataset_id)

        def get_df_compat(columns: Optional[List[str]] = None) -> pd.DataFrame:
            """Call DatasetManager.get_df across old/new signatures."""
            if columns:
                projected = list(
                    dict.fromkeys(
                        str(col)
                        for col in columns
                        if col and str(col).strip()
                    )
                )
                try:
                    return self.context.datasets.get_df(dataset_id, columns=projected)
                except TypeError:
                    return self.context.datasets.get_df(dataset_id)
                except Exception:
                    # Some lazy/projected backends may reject a column projection.
                    # Fall back to full materialisation.
                    return self.context.datasets.get_df(dataset_id)

            return self.context.datasets.get_df(dataset_id)

        def available_columns_debug(frame: Optional[pd.DataFrame] = None) -> List[str]:
            if frame is not None:
                try:
                    return [str(col) for col in frame.columns]
                except Exception:
                    pass

            try:
                return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
            except Exception:
                pass

            try:
                return [str(col) for col in get_df_compat().columns]
            except Exception:
                return []

        def extract_by_id_column(frame: pd.DataFrame) -> Dict[str, Any]:
            if id_column not in frame.columns:
                raise ValueError(
                    f"record_id column {id_column!r} is not present in dataset "
                    f"{dataset_id!r}. Available columns: {list(frame.columns)}"
                )

            if target_column not in frame.columns:
                raise ValueError(
                    f"Auto-label source column {target_column!r} is not present in "
                    f"dataset {dataset_id!r}. Available columns: {list(frame.columns)}"
                )

            out: Dict[str, Any] = {}
            for _, row in frame.iterrows():
                row_id = str(row[id_column])
                if row_id not in row_id_set:
                    continue

                value = row[target_column]
                if pd.notna(value):
                    out[row_id] = value

            return out

        def extract_by_index(frame: pd.DataFrame) -> Dict[str, Any]:
            if target_column not in frame.columns:
                raise ValueError(
                    f"Auto-label source column {target_column!r} is not present in "
                    f"dataset {dataset_id!r}. Available columns: {list(frame.columns)}"
                )

            out: Dict[str, Any] = {}

            # Build a str(index) -> real index lookup once.
            index_lookup = {str(idx): idx for idx in frame.index}

            for row_id in row_ids:
                real_index = index_lookup.get(str(row_id))
                if real_index is None:
                    continue

                value = frame.loc[real_index, target_column]
                if pd.notna(value):
                    out[str(row_id)] = value

            return out

        if id_column:
            row_lookup_df: Optional[pd.DataFrame] = None

            # Fast path: use row lookup first.
            try:
                row_lookup_df = self.context.datasets.get_rows_by_ids(
                    dataset_id,
                    row_ids,
                    id_column=id_column,
                )
                row_lookup_df = row_lookup_df.copy()
            except Exception:
                row_lookup_df = None

            # If fast path returned the needed columns, use it.
            if (
                row_lookup_df is not None
                and not row_lookup_df.empty
                and id_column in row_lookup_df.columns
                and target_column in row_lookup_df.columns
            ):
                return extract_by_id_column(row_lookup_df)

            # Otherwise fall back to projected/full dataset read.
            try:
                source_df = get_df_compat([id_column, target_column])
            except Exception as exc:
                returned_columns = available_columns_debug(row_lookup_df)
                raise ValueError(
                    f"Could not read columns {id_column!r} and {target_column!r} "
                    f"from dataset {dataset_id!r}. Row lookup returned columns: "
                    f"{returned_columns}"
                ) from exc

            if id_column not in source_df.columns or target_column not in source_df.columns:
                returned_columns = available_columns_debug(source_df)
                row_lookup_columns = (
                    list(row_lookup_df.columns)
                    if row_lookup_df is not None
                    else []
                )
                raise ValueError(
                    f"Could not read auto-label source column {target_column!r} "
                    f"from dataset {dataset_id!r} using id column {id_column!r}.\n"
                    f"Projected/full dataframe columns: {returned_columns}\n"
                    f"get_rows_by_ids dataframe columns: {row_lookup_columns}"
                )

            filtered = source_df[
                source_df[id_column].astype(str).isin(row_id_set)
            ].copy()

            return extract_by_id_column(filtered)

        # No record_id mapping: use dataframe index matching.
        source_df = get_df_compat([target_column])

        if target_column not in source_df.columns:
            raise ValueError(
                f"Auto-label source column {target_column!r} is not present in "
                f"dataset {dataset_id!r}. Available columns: {list(source_df.columns)}"
            )

        return extract_by_index(source_df)

    def _train_clicked(self, *_: Any) -> None:
        try:
            session_artifact_id = self._require_session()
            self._require_labelled()
            recipe_id = str(self.recipe_select.value or "").strip()
            if not recipe_id:
                raise ValueError("Choose a core.ml recipe.")

            params = {
                "session_artifact_id": session_artifact_id,
                "dataset_id": self.dataset_select.value or "",
                "recipe_id": recipe_id,
                "recipe_params": self._recipe_params(),
                "target_column": self.target_column.value or "al_label",
                "validation_dataset_id": self.validation_dataset_select.value or "",
                "test_dataset_id": self.test_dataset_select.value or "",
                "al_protocol": self.al_protocol.value or "review",
                "seed": int(self.seed.value or 0),
            }
            params.update(self._protocol_params())
            request = ActionRequest(
                params=params,
                origin="core.active_learning.panel",
            )

            self._set_running(True)
            self._set_status(
                "Training started. Active Learning will call the registered core.ml recipe action from scratch.",
                "info",
            )

            manager = getattr(self.context, "plugins", None)
            run_action = getattr(manager, "run_action", None)
            if callable(run_action):
                self._active_job_handle = run_action(
                    "core.active_learning.train_from_session",
                    self.context,
                    request,
                    on_done=self._on_train_done,
                    on_error=self._on_train_error,
                )
            else:
                submit = getattr(getattr(self.context, "jobs", None), "submit", None)
                if callable(submit):
                    key = f"core.active_learning.train:{session_artifact_id}:{recipe_id}"
                    self._active_job_handle = submit(
                        actions.train_from_session_action,
                        title="Active-learning scratch training",
                        key=key,
                        on_done=self._on_train_done,
                        on_error=self._on_train_error,
                        context=self.context,
                        request=request,
                    )
                else:
                    result = actions.train_from_session_action(self.context, request)
                    self._on_train_done(result)

        except Exception as exc:
            self._set_running(False)
            self._set_error("Could not start scratch training", exc)

    def _on_query_done(self, result: Any) -> None:
        if self._disposed:
            return
        self._set_running(False)
        self._active_job_handle = None
        if not isinstance(result, Mapping):
            self._set_status("Query finished, but returned an unexpected result shape.", "warning")
            return
        self._use_action_session_result(result)
        stats = result.get("rank_stats") or {}
        self._set_status(
            (
                f"Created ordered query batch with {len(result.get('row_ids') or [])} rows. "
                f"Seen={stats.get('total_records_seen', 'unknown')}, "
                f"excluded={stats.get('excluded_count', 'unknown')}, "
                f"unscorable={stats.get('unscorable_count', 'unknown')}."
            ),
            "success",
        )

    def _on_query_error(self, error: Any) -> None:
        if self._disposed:
            return
        self._set_running(False)
        self._active_job_handle = None
        self._set_error("Could not create active-learning query batch", error)

    def _on_train_done(self, result: Any) -> None:
        if self._disposed:
            return
        self._set_running(False)
        self._active_job_handle = None
        if not isinstance(result, Mapping):
            self._set_status("Training finished, but returned an unexpected result shape.", "warning")
            return

        self._use_action_session_result(result)
        pool_dataset_id = self._session_pool_dataset_id()
        self._set_status(
            (
                f"Training round {result.get('round')} finished from scratch with "
                f"{result.get('labelled_count')} verified labels.\n\n"
                f"Next: run `core.ml.predict` on the pool dataset "
                f"`{pool_dataset_id}`, then use the Query tab."
            ),
            "success",
        )

    def _on_train_error(self, error: Any) -> None:
        if self._disposed:
            return
        self._set_running(False)
        self._active_job_handle = None
        self._set_error("Scratch training failed", error)

    def _session_selected(self, event: Any) -> None:
        value = getattr(event, "new", None)
        if value:
            self._current_session_artifact_id = str(value)
            self._refresh_predictions()
            self._refresh_review_label_for_focus()
            self._refresh_session_views()
            self._update_action_gating()

    def _label_options_changed(self, *_: Any) -> None:
        self._set_label_select_options()

    def _refresh_datasets(self) -> None:
        current = self.dataset_select.value
        ids = self._dataset_ids()
        self.dataset_select.options = ids
        active = None
        try:
            active = self.context.datasets.active_id()
        except Exception:
            active = None
        if current in ids:
            self.dataset_select.value = current
        elif active in ids:
            self.dataset_select.value = active
        elif ids:
            self.dataset_select.value = ids[0]
        else:
            self.dataset_select.value = None

    def _refresh_sessions(self) -> None:
        current = self._current_session_artifact_id or self.session_select.value
        latest: Dict[str, Dict[str, Any]] = {}
        try:
            refs = self.context.artifacts.find(type=al_state.ARTIFACT_SESSION)
        except Exception:
            refs = []

        for ref in refs:
            try:
                payload = al_state.coerce_session(self.context.artifacts.get(ref.artifact_id))
            except Exception:
                continue
            session_id = str(payload.get("session_id") or ref.artifact_id)
            updated_at = float(payload.get("updated_at") or 0.0)
            revision = int(payload.get("revision") or 0)
            existing = latest.get(session_id)
            if existing is None or (updated_at, revision) > (existing["updated_at"], existing["revision"]):
                latest[session_id] = {
                    "artifact_id": ref.artifact_id,
                    "payload": payload,
                    "updated_at": updated_at,
                    "revision": revision,
                }

        options: Dict[str, str] = {}
        for session_id, item in sorted(
            latest.items(),
            key=lambda kv: float(kv[1]["updated_at"]),
            reverse=True,
        ):
            payload = item["payload"]
            counts = al_state.counts(payload)
            dataset_id = payload.get("dataset_id", "")
            label = (
                f"{session_id} | {dataset_id} | r{payload.get('round')} "
                f"rev{payload.get('revision', 0)} | "
                f"train={counts['labelled_or_verified']} unsure={counts['unsure']}"
            )
            options[label] = item["artifact_id"]

        self.session_select.options = options
        values = set(options.values())
        if current and current in values:
            self.session_select.value = current
            self._current_session_artifact_id = current
        elif options:
            first_value = next(iter(options.values()))
            self.session_select.value = first_value
            self._current_session_artifact_id = first_value
        else:
            self.session_select.value = None
            self._current_session_artifact_id = None

    def _refresh_predictions(self) -> None:
        current = self.predictions_select.value
        options: Dict[str, str] = {}

        session = self._load_current_session()
        expected_dataset_id = self._session_pool_dataset_id(session)

        total_prediction_artifacts = 0
        skipped_wrong_dataset = 0

        try:
            refs = self.context.artifacts.find(type="ml.predictions")
        except Exception:
            refs = []

        for ref in refs:
            artifact_id = str(getattr(ref, "artifact_id", "") or "")
            if not artifact_id:
                continue

            total_prediction_artifacts += 1

            try:
                payload = self.context.artifacts.get(artifact_id)
                if not isinstance(payload, Mapping):
                    payload = {}
            except Exception:
                payload = {}

            dataset_id = str(
                payload.get("dataset_id")
                or getattr(ref, "dataset_id", "")
                or ""
            ).strip()

            # Once a session exists, only offer predictions made over the AL
            # pool dataset. Do not offer predictions over derived datasets like
            # "...__al_train_r1_...".
            if expected_dataset_id and dataset_id != expected_dataset_id:
                skipped_wrong_dataset += 1
                continue

            label = self._prediction_option_label(
                artifact_id=artifact_id,
                payload=payload,
            )
            options[label] = artifact_id

        self.predictions_select.options = options

        if current in options.values():
            self.predictions_select.value = current
        elif options:
            self.predictions_select.value = next(iter(options.values()))
        else:
            self.predictions_select.value = None

        if expected_dataset_id and not options and total_prediction_artifacts:
            self.query_hint.object = (
                "_No predictions are available for this session's pool dataset. "
                f"Run `core.ml.predict` on `{expected_dataset_id}` using the trained "
                "model, then return here to query._"
            )
        elif expected_dataset_id and skipped_wrong_dataset:
            self.query_hint.object = (
                f"_Filtered out {skipped_wrong_dataset} prediction artifact(s) "
                "because they were not made on the AL pool dataset._"
            )

    def _refresh_strategies(self) -> None:
        current = self.strategy_select.value
        options: Dict[str, str] = {}

        try:
            registry = self.context.services.get("core.active_learning.query_strategy_registry")
            for info in registry.list():
                options[f"{info.title} ({info.id})"] = info.id
        except Exception:
            options = {
                "Least confidence (least_confidence)": "least_confidence",
                "Smallest margin (margin)": "margin",
                "Entropy (entropy)": "entropy",
                "Random (random)": "random",
            }

        self.strategy_select.options = options
        if current in options.values():
            self.strategy_select.value = current
        elif options:
            self.strategy_select.value = next(iter(options.values()))

    def _refresh_session_views(self) -> None:
        session = self._load_current_session()
        if not session:
            self.summary.object = ""
            _set_table_value(self.batch_table, pd.DataFrame())
            _set_table_value(self.labels_table, pd.DataFrame())
            self.session_json.object = {}
            return

        counts = al_state.counts(session)
        last_batch = session.get("last_batch") or {}

        self.summary.object = (
            f"### Session `{session.get('session_id')}`\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Dataset | `{session.get('dataset_id')}` |\n"
            f"| Pool | `{session.get('pool_dataset_id') or session.get('dataset_id')}` |\n"
            f"| Round | `{session.get('round')}` |\n"
            f"| Verified training labels | **{counts['labelled_or_verified']}** |\n"
            f"| Unsure / ignored only | **{counts['unsure']}** |\n"
            f"| Total ignored from pool | **{counts['ignored']}** |\n"
            f"| Last batch | `{last_batch.get('strategy_id') or 'none'}` "
            f"({last_batch.get('count') or 0} rows) |\n"
        )

        labels_df = pd.DataFrame(al_state.label_rows(session))
        _set_table_value(self.labels_table, labels_df)

        batch_df = pd.DataFrame()
        batch_artifact_id = last_batch.get("batch_artifact_id")
        if batch_artifact_id:
            try:
                batch_payload = self.context.artifacts.get(batch_artifact_id)
                records = list(batch_payload.get("records") or [])
                batch_df = pd.DataFrame(records[:1000])
            except Exception:
                batch_df = pd.DataFrame()

        _set_table_value(self.batch_table, batch_df)
        self.session_json.object = session

    def _use_action_session_result(self, result: Mapping[str, Any]) -> None:
        session_artifact_id = result.get("session_artifact_id")
        if session_artifact_id:
            self._current_session_artifact_id = str(session_artifact_id)
        self.refresh()

    def _set_label_select_options(self) -> None:
        labels = al_state.parse_label_options(self.label_options_select.value)

        options = {label: label for label in labels}
        options[al_state.UNSURE_DISPLAY] = al_state.UNSURE_LABEL

        current = self.label_select.value
        self.label_select.options = options

        if current in options.values():
            self.label_select.value = current
        elif options:
            self.label_select.value = al_state.UNSURE_LABEL

    def _advance_focus_after(self, row_id: str) -> None:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return

        active_set = selection.get_active_set()
        if active_set is None:
            return

        row_ids = [str(item) for item in getattr(active_set, "row_ids", []) or []]
        if not row_ids:
            return

        session = self._load_current_session() or {}
        ignored = set(str(item) for item in session.get("ignored_row_ids", []) or [])

        try:
            start = row_ids.index(str(row_id)) + 1
        except ValueError:
            start = 0

        for candidate in row_ids[start:]:
            if candidate in ignored:
                continue
            selection.set_focus(
                dataset_id=active_set.dataset_id,
                row_id=candidate,
                origin="core.active_learning.panel.advance",
                selection_set_id=active_set.selection_set_id,
                metadata={"reason": "advance_after_label"},
            )
            self._refresh_review_label_for_focus()
            return

    def _load_current_session(self) -> Optional[Dict[str, Any]]:
        if not self._current_session_artifact_id:
            return None
        try:
            return al_state.coerce_session(
                self.context.artifacts.get(self._current_session_artifact_id)
            )
        except Exception:
            return None

    def _session_pool_dataset_id(self, session: Optional[Mapping[str, Any]] = None) -> str:
        """Dataset that AL should query from.

        This is the original pool/source dataset, not the derived AL training
        dataset created for a particular round.
        """
        if session is None:
            session = self._load_current_session() or {}

        return str(
            session.get("pool_dataset_id")
            or session.get("dataset_id")
            or self.dataset_select.value
            or ""
        ).strip()

    def _prediction_dataset_id(self, artifact_id: str) -> str:
        artifact_id = str(artifact_id or "").strip()
        if not artifact_id:
            return ""

        try:
            payload = self.context.artifacts.get(artifact_id)
            if isinstance(payload, Mapping):
                return str(payload.get("dataset_id") or "").strip()
        except Exception:
            pass

        try:
            refs = self.context.artifacts.find(type="ml.predictions")
            for ref in refs:
                if str(getattr(ref, "artifact_id", "")) == artifact_id:
                    return str(getattr(ref, "dataset_id", "") or "").strip()
        except Exception:
            pass

        return ""

    def _prediction_matches_current_session(self, artifact_id: str) -> bool:
        artifact_id = str(artifact_id or "").strip()
        if not artifact_id:
            return False

        expected_dataset_id = self._session_pool_dataset_id()
        actual_dataset_id = self._prediction_dataset_id(artifact_id)

        if not expected_dataset_id:
            return True

        return bool(actual_dataset_id and actual_dataset_id == expected_dataset_id)

    def _prediction_option_label(
        self,
        *,
        artifact_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        dataset_id = str(payload.get("dataset_id") or "")
        count = len(payload.get("records") or [])

        model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
        recipe_id = str(payload.get("recipe_id") or model.get("recipe_id") or "")
        run_id = str(payload.get("run_id") or "")

        bits = [artifact_id[:8]]
        if dataset_id:
            bits.append(dataset_id)
        if count:
            bits.append(f"{count} rows")
        if recipe_id:
            bits.append(recipe_id)
        if run_id:
            bits.append(f"run {run_id[:8]}")

        return " | ".join(bits)

    def _refresh_review_label_for_focus(self) -> None:
        """Set the Review label dropdown from session label, source column, or Unsure."""
        if self._disposed:
            return

        label = self._label_for_current_focus()
        if label is None:
            label = al_state.UNSURE_LABEL

        label = al_state.normalise_label(label)

        # Ensure the inferred label is present in both the allowed-label widget
        # and the review dropdown. Unsure is always allowed as a review action.
        if label != al_state.UNSURE_LABEL:
            allowed = al_state.parse_label_options(self.label_options_select.value)
            if label not in allowed:
                allowed.append(label)
                self.label_options_select.options = list(
                    dict.fromkeys([*self.label_options_select.options, label])
                )
                self.label_options_select.value = allowed

        self._set_label_select_options()

        valid_values = set(self.label_select.options.values()) if isinstance(self.label_select.options, dict) else set(self.label_select.options)
        if label in valid_values:
            self.label_select.value = label
        else:
            self.label_select.value = al_state.UNSURE_LABEL

    def _label_for_current_focus(self) -> Optional[str]:
        dataset_id, row_id = self._current_focus_ref()
        if not row_id:
            return al_state.UNSURE_LABEL

        session = self._load_current_session() or {}

        session_dataset_id = str(session.get("dataset_id") or "")
        if dataset_id and session_dataset_id and dataset_id != session_dataset_id:
            return al_state.UNSURE_LABEL

        session_label = self._session_label_for_row(session, row_id)
        if session_label:
            return session_label

        source_label = self._source_label_for_row(
            dataset_id=dataset_id or session_dataset_id,
            row_id=row_id,
        )
        if source_label:
            return source_label

        return al_state.UNSURE_LABEL

    def _session_label_for_row(
        self,
        session: Mapping[str, Any],
        row_id: Any,
    ) -> Optional[str]:
        row_id = str(row_id)
        labels = dict(session.get("labels") or {})
        entry = labels.get(row_id)
        if not isinstance(entry, Mapping):
            return None

        label = entry.get("label")
        if label is None:
            return None

        label = al_state.normalise_label(label)
        return label or None

    def _source_label_for_row(
        self,
        *,
        dataset_id: Optional[str],
        row_id: Any,
    ) -> Optional[str]:
        dataset_id = str(dataset_id or "").strip()
        row_id = str(row_id or "").strip()
        column = str(self.target_column.value or "").strip()

        if not dataset_id or not row_id or not column or column == "al_label":
            return None

        value = self._dataset_value_for_row(
            dataset_id=dataset_id,
            row_id=row_id,
            column=column,
        )
        if value is None:
            return None

        label = al_state.normalise_label(value)
        return label or None

    def _dataset_value_for_row(
        self,
        *,
        dataset_id: str,
        row_id: str,
        column: str,
    ) -> Optional[Any]:
        id_column = self._dataset_mapping(dataset_id, "record_id")

        if not id_column:
            columns = self._dataset_columns(dataset_id)
            for candidate in ("record_id", "id", "ID", "source_id", "object_id", "row_id"):
                if candidate in columns:
                    id_column = candidate
                    break

        # Prefer row-id lookup when the platform supports it.
        if id_column:
            try:
                df = self.context.datasets.get_rows_by_ids(
                    dataset_id,
                    [row_id],
                    id_column=id_column,
                )
                if column in df.columns and not df.empty:
                    return df.iloc[0][column]
            except Exception:
                pass

        # Fallback to a small dataframe materialisation.
        try:
            try:
                needed = [column]
                if id_column:
                    needed.append(id_column)
                df = self.context.datasets.get_df(
                    dataset_id,
                    columns=list(dict.fromkeys(needed)),
                )
            except TypeError:
                df = self.context.datasets.get_df(dataset_id)

            if column not in df.columns:
                return None

            if id_column and id_column in df.columns:
                match = df[df[id_column].astype(str) == row_id]
                if not match.empty:
                    return match.iloc[0][column]

            # Some datasets use the dataframe index as the record id.
            index_matches = [idx for idx in df.index if str(idx) == row_id]
            if index_matches:
                return df.loc[index_matches[0], column]

        except Exception:
            return None

        return None

    def _current_focus_ref(self) -> tuple[Optional[str], Optional[str]]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None, None
        focus = selection.get_focus()
        dataset_id = getattr(focus, "dataset_id", None)
        row_id = getattr(focus, "row_id", None)
        return (
            None if dataset_id is None else str(dataset_id),
            None if row_id is None else str(row_id),
        )

    def _current_focus_row_id(self) -> Optional[str]:
        dataset_id, row_id = self._current_focus_ref()
        session = self._load_current_session() or {}
        session_dataset_id = str(session.get("dataset_id") or "")
        if dataset_id and session_dataset_id and dataset_id != session_dataset_id:
            raise ValueError(
                f"Focused row belongs to dataset {dataset_id!r}, but the AL session "
                f"belongs to {session_dataset_id!r}."
            )
        return row_id

    def _restore_widget_state(self, state: Mapping[str, Any]) -> None:
        state = dict(state or {})
        for widget_name, key in (
            ("dataset_select", "dataset_id"),
            ("recipe_select", "recipe_id"),
            ("target_column", "target_column"),
            ("validation_dataset_select", "validation_dataset_id"),
            ("test_dataset_select", "test_dataset_id"),
            ("strategy_select", "strategy_id"),
            ("al_protocol", "al_protocol"),
        ):
            widget = getattr(self, widget_name, None)
            value = state.get(key)
            if widget is None or value is None:
                continue
            try:
                options = widget.options
                valid_values = set(options.values()) if isinstance(options, dict) else set(options)
                if value in valid_values:
                    widget.value = value
            except Exception:
                pass

        labels = al_state.parse_label_options(state.get("label_options"))
        if labels:
            self.label_options_select.options = list(dict.fromkeys([*labels, *self.label_options_select.options]))
            self.label_options_select.value = labels

        params = state.get("recipe_params") or {}
        if isinstance(params, Mapping):
            for name, value in params.items():
                widget = self.recipe_param_widgets.get(str(name)) or self.protocol_widgets.get(str(name))
                if widget is not None:
                    try:
                        widget.value = value
                    except Exception:
                        pass

        self._update_action_gating()

    def _subscribe_refresh_events(self) -> None:
        events = getattr(self.context, "events", None)
        subscribe = getattr(events, "subscribe", None)
        if not callable(subscribe):
            return
        topics = [
            "plugin.enabled",
            "plugin.disabled",
            "artifact.created",
            "artifact.updated",
            "dataset.active.changed",
            "dataset.registered",
            "dataset.loaded",
            "dataset.updated",
            "dataset.*"
            "selection.focus.changed",
            "selection.focus.cleared",
            "ml.recipe.registered",
            "ml.recipe.unregistered",
            "ml.model_definition.created",
            "ml.model.created",
            "ml.run.created",
            "ml.predictions.created",
            "al.round.training_finished",
        ]
        for topic in topics:
            try:
                self._subscriptions.append(subscribe(topic, self._on_external_refresh_event))
            except Exception:
                pass

    def _on_external_refresh_event(self, *args: Any, **kwargs: Any) -> None:
        """React to platform events.

        The platform EventBus calls subscribers as cb(topic, payload). This
        method accepts multiple callback shapes so event handling never breaks
        the panel.
        """
        if self._disposed:
            return

        topic = None
        payload: Any = None

        if len(args) >= 2:
            topic = args[0]
            payload = args[1]
        elif len(args) == 1:
            event = args[0]
            if isinstance(event, Mapping):
                topic = event.get("topic")
                payload = event
            else:
                topic = getattr(event, "topic", None)
                payload = getattr(event, "payload", None)

        if topic is None:
            topic = kwargs.get("topic")
        if payload is None:
            payload = kwargs.get("payload", kwargs)

        topic = str(topic or "")

        def apply() -> None:
            if self._disposed:
                return

            if topic in {"selection.focus.changed", "selection.focus.cleared"}:
                self._refresh_review_label_for_focus()
                return

            if (
                topic.startswith("dataset.")
                or topic.startswith("mapping.")
            ):
                self._refresh_dataset_dependent_widgets()
                return

            try:
                self.refresh()
            except Exception:
                pass

        try:
            curdoc = getattr(pn.state, "curdoc", None)
            if curdoc is not None:
                curdoc.add_next_tick_callback(apply)
            else:
                apply()
        except Exception:
            apply()

    def _dataset_changed(self, *_: Any) -> None:
        self._refresh_column_widgets()
        self._refresh_validation_test_datasets()
        self._refresh_recipe_params()
        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()

    def _recipe_changed(self, *_: Any) -> None:
        self._refresh_recipe_params()
        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()

    def _predictions_changed(self, *_: Any) -> None:
        self._refresh_labels_from_available_context()
        self._update_action_gating()

    def _target_column_changed(self, *_: Any) -> None:
        self._refresh_labels_from_available_context()
        self._set_label_select_options()
        self._refresh_review_label_for_focus()

    def _current_label_options(self) -> List[str]:
        labels = al_state.parse_label_options(self.label_options_select.value)
        if labels:
            return labels

        labels = self._labels_from_selected_target_column()
        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        labels = self._labels_from_selected_recipe()
        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        labels = self._labels_from_selected_predictions()
        if labels:
            self.label_options_select.options = labels
            self.label_options_select.value = labels
            self._set_label_select_options()
            return labels

        return []

    def _labels_from_selected_target_column(self) -> List[str]:
        dataset_id = str(self.dataset_select.value or "").strip()
        column = str(self.target_column.value or "").strip()

        if not dataset_id or not column or column == "al_label":
            return []

        values = self._unique_values_from_column(dataset_id, column, max_values=200)
        return al_state.parse_label_options(values)

    def _unique_values_from_column(
        self,
        dataset_id: str,
        column: str,
        *,
        max_values: int = 200,
    ) -> List[str]:
        values: List[str] = []

        try:
            source = None
            get_source = getattr(self.context.datasets, "get_source", None)
            if callable(get_source):
                source = get_source(dataset_id)

            for method_name in ("unique_values", "distinct_values", "get_column_values"):
                method = getattr(source, method_name, None) if source is not None else None
                if callable(method):
                    try:
                        raw = method(column, limit=max_values)
                    except TypeError:
                        raw = method(column)
                    values = [str(item) for item in raw if item is not None and str(item).strip()]
                    return list(dict.fromkeys(values))[:max_values]
        except Exception:
            pass

        try:
            try:
                df = self.context.datasets.get_df(dataset_id, columns=[column])
            except TypeError:
                df = self.context.datasets.get_df(dataset_id)

            if column not in df.columns:
                return []

            series = df[column].dropna()
            for value in series.tolist():
                text = str(value).strip()
                if not text:
                    continue
                if text not in values:
                    values.append(text)
                if len(values) >= max_values:
                    break
        except Exception:
            return []

        return values

    def _restore_select_value(self, widget: Any, value: Any) -> None:
        if value is None:
            return

        try:
            options = widget.options
            valid_values = set(options.values()) if isinstance(options, dict) else set(options)
            if value in valid_values:
                widget.value = value
        except Exception:
            pass

    def _restore_widget_value(self, widget: Any, value: Any) -> None:
        try:
            if isinstance(widget, pn.widgets.Select):
                options = widget.options
                valid_values = set(options.values()) if isinstance(options, dict) else set(options)
                if value in valid_values:
                    widget.value = value
            else:
                widget.value = value
        except Exception:
            pass

    def _refresh_dataset_dependent_widgets(self) -> None:
        """Refresh controls affected by dataset registration or mapping changes."""
        if self._disposed:
            return

        previous_dataset = self.dataset_select.value
        previous_validation = self.validation_dataset_select.value
        previous_test = self.test_dataset_select.value
        previous_target = self.target_column.value
        previous_prediction = self.predictions_select.value

        previous_recipe_values: Dict[str, Any] = {}
        for name, widget in self.recipe_param_widgets.items():
            try:
                previous_recipe_values[str(name)] = widget.value
            except Exception:
                pass

        previous_protocol_values: Dict[str, Any] = {}
        for name, widget in self.protocol_widgets.items():
            try:
                previous_protocol_values[str(name)] = widget.value
            except Exception:
                pass

        self._refresh_datasets()
        self._refresh_validation_test_datasets()
        self._refresh_predictions()
        self._refresh_column_widgets()
        self._refresh_recipe_params()

        self._restore_select_value(self.dataset_select, previous_dataset)
        self._restore_select_value(self.validation_dataset_select, previous_validation)
        self._restore_select_value(self.test_dataset_select, previous_test)
        self._restore_select_value(self.target_column, previous_target)
        self._restore_select_value(self.predictions_select, previous_prediction)

        for name, value in previous_recipe_values.items():
            widget = self.recipe_param_widgets.get(name)
            if widget is not None:
                self._restore_widget_value(widget, value)

        for name, value in previous_protocol_values.items():
            widget = self.protocol_widgets.get(name)
            if widget is not None:
                self._restore_widget_value(widget, value)

        self._apply_recipe_inferred_defaults()
        self._refresh_labels_from_available_context()
        self._set_label_select_options()
        self._refresh_review_label_for_focus()
        self._update_action_gating()

    def _refresh_column_widgets(self) -> None:
        """Refresh column-backed dropdowns after dataset or mappings change.

        This keeps the target-label dropdown, recipe column params, and protocol
        column params aligned with the currently selected dataset.
        """
        dataset_id = self.dataset_select.value
        columns = self._dataset_columns(dataset_id)

        target_options = list(dict.fromkeys(["al_label", *columns]))
        current_target = self.target_column.value

        self.target_column.options = target_options

        mapped_target = self._dataset_mapping(dataset_id, "target_label")
        if current_target in target_options:
            self.target_column.value = current_target
        elif mapped_target in target_options:
            self.target_column.value = mapped_target
        else:
            guessed = self._guess_label_column(columns)
            self.target_column.value = guessed if guessed in target_options else "al_label"

        column_options = [""] + columns

        # Refresh recipe parameter widgets that represent dataset columns.
        for name, widget in list(self.recipe_param_widgets.items()):
            if not isinstance(widget, pn.widgets.Select):
                continue

            lname = str(name).lower()
            looks_like_column_param = (
                lname.endswith("_column")
                or lname in {
                    "target",
                    "label",
                    "label_column",
                    "target_column",
                    "image_column",
                    "mask_column",
                    "record_id_column",
                    "id_column",
                    "group_column",
                    "split_column",
                }
            )

            if not looks_like_column_param:
                continue

            current = widget.value
            widget.options = column_options
            widget.value = current if current in column_options else ""

        # Refresh protocol column widgets if they have already been built.
        for key in ("protocol_group_column", "protocol_split_column"):
            widget = self.protocol_widgets.get(key)
            if widget is None or not isinstance(widget, pn.widgets.Select):
                continue

            current = widget.value
            widget.options = column_options
            widget.value = current if current in column_options else ""

        self._refresh_labels_from_available_context()

    def _refresh_recipes(self) -> None:
        current = self.recipe_select.value
        options: Dict[str, str] = {}
        registry = self._recipe_registry()
        if registry is not None:
            try:
                for spec in registry.list():
                    label = f"{getattr(spec, 'title', getattr(spec, 'id', 'recipe'))} ({getattr(spec, 'id', '')})"
                    options[label] = str(getattr(spec, "id", ""))
            except Exception:
                options = {}
        self.recipe_select.options = options
        if current in options.values():
            self.recipe_select.value = current
        elif options:
            self.recipe_select.value = next(iter(options.values()))
        else:
            self.recipe_select.value = None
            self.recipe_card.object = "No core.ml recipes are registered. Enable core.ml or add a recipe."

    def _recipe_registry(self) -> Any:
        try:
            return self.context.services.get("core.ml.recipe_registry")
        except Exception:
            return None

    def _selected_recipe_spec(self) -> Any:
        registry = self._recipe_registry()
        recipe_id = str(self.recipe_select.value or "").strip()
        if registry is None or not recipe_id:
            return None
        try:
            return registry.get(recipe_id)
        except Exception:
            return None

    def _form_row(self, label: str, widget: Any, *, height: int = 46):
        return self._field(label, widget)

    def _field(self, label: str, widget: Any):
        # Use the widget's own native label (rendered inside the widget box)
        # instead of stacking a separate Markdown pane above it.
        try:
            widget.name = label
        except Exception:
            pass
        self._fit_widget(widget)

        # Reserve vertical space with min_height (not a fixed height): the
        # widget can still grow if a long label wraps, but can never collapse
        # to zero, which is the residual insurance against any flex weirdness.
        try:
            if isinstance(widget, pn.widgets.TextAreaInput):
                widget.min_height = 120
            elif isinstance(widget, pn.widgets.Checkbox):
                widget.min_height = 28
            else:
                widget.min_height = 54
            widget.margin = (0, 0, 12, 0)
        except Exception:
            pass

        return widget

    def _install_layout_css_once(self) -> None:
        global _AL_LAYOUT_CSS_INSTALLED
        if _AL_LAYOUT_CSS_INSTALLED:
            return

        css = """
        .al-panel-root,
        .al-panel-root * {
            box-sizing: border-box;
        }

        .al-panel-root {
            max-width: 480px;
            width: 100%;
            margin: 0 auto;
            overflow-x: hidden !important;
        }

        .al-controls-tabs,
        .al-results-tabs,
        .al-tab-body,
        .al-section,
        .al-field {
            max-width: 100%;
            width: 100%;
            box-sizing: border-box;
        }

        .al-tab-body {
            overflow: visible;
        }

        .al-panel-root button,
        .al-panel-root .bk-btn {
            min-height: 32px !important;
            height: auto !important;
            white-space: normal !important;
        }

        .al-panel-root .markdown,
        .al-panel-root .bk-HTML,
        .al-panel-root code {
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .al-panel-root .tabulator {
            max-width: 100%;
            width: 100%;
            overflow: auto;
        }

        .al-panel-root .bk-menu,
        .al-panel-root .choices__list--dropdown {
            z-index: 10000 !important;
        }

        .al-hint p {
            margin: 0 0 6px 0;
            font-size: 12px;
            color: #9a6a00;
        }
        """

        try:
            pn.extension(raw_css=[css])
            _AL_LAYOUT_CSS_INSTALLED = True
        except Exception:
            pass

    def _fit_widget(self, widget: Any) -> Any:

        try:
            widget.sizing_mode = "stretch_width"
        except Exception:
            pass

        try:
            styles = dict(getattr(widget, "styles", {}) or {})
            styles.update(
                {
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                }
            )
            widget.styles = styles
        except Exception:
            pass

        return widget

    def _fit_form_widget(self, widget: Any) -> Any:
        """Fit recipe/protocol widgets.

        We now rely on each widget's native label (set via ``_field``), so this
        must NOT blank out the name as an earlier version did.
        """
        return self._fit_widget(widget)

    def _normalise_widget_layout(self) -> None:
        button_names = {
            self.refresh_button: "Refresh",
            self.start_button: "Start / random sample",
            self.query_button: "Query top-k from predictions",
            self.label_button: "Record label / verification",
            self.unsure_button: "Mark current as Unsure",
            self.train_button: "Add verified labels to training set + train from scratch",
        }

        widgets = [
            self.dataset_select,
            self.session_select,
            self.predictions_select,
            self.strategy_select,
            self.recipe_select,
            self.label_options_select,
            self.target_column,
            self.validation_dataset_select,
            self.test_dataset_select,
            self.al_protocol,
            self.seed,
            self.initial_k,
            self.query_k,
            self.label_select,
            self.advance_after_label,
            self.refresh_button,
            self.start_button,
            self.query_button,
            self.label_button,
            self.unsure_button,
            self.train_button,
        ]

        for widget in widgets:
            self._fit_widget(widget)

        for button, name in button_names.items():
            try:
                button.name = name
                button.min_width = 160
                button.align = "start"
            except Exception:
                pass

        for pane in (self.status, self.summary, self.recipe_card):
            try:
                pane.sizing_mode = "stretch_width"
                pane.styles = {
                    "max-width": "100%",
                    "width": "100%",
                    "overflow-wrap": "anywhere",
                    "word-break": "break-word",
                    "box-sizing": "border-box",
                }
            except Exception:
                pass

        for hint in (self.query_hint, self.train_hint):
            try:
                hint.sizing_mode = "stretch_width"
                hint.css_classes = ["al-hint"]
                hint.styles = {
                    "max-width": "100%",
                    "width": "100%",
                    "box-sizing": "border-box",
                }
            except Exception:
                pass

        try:
            self.recipe_params_area.sizing_mode = "stretch_width"
            self.recipe_params_area.styles = {
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
            }
        except Exception:
            pass

    def _tab_body(self, body: Any, *, height: int = 430):
        # No fixed height / inner scroll: the whole panel scrolls as one
        # region, so a button at the bottom of a tab is always reachable.
        return pn.Column(
            body,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "padding": "8px 6px 12px 6px",
            },
            css_classes=["al-tab-body"],
        )

    def _section(self, title: str, body: Any):
        return pn.Column(
            pn.pane.Markdown(
                f"**{title}**",
                sizing_mode="stretch_width",
                margin=(0, 0, 6, 0),
                styles={
                    "font-size": "12px",
                    "line-height": "1.2",
                    "font-weight": "600",
                    "overflow-wrap": "anywhere",
                },
            ),
            body,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "padding-bottom": "4px",
            },
            css_classes=["al-section"],
        )

    def _widget_for_schema(self, name: str, schema: Mapping[str, Any]):
        schema = dict(schema or {})
        kind = str(schema.get("type", "string"))
        default = schema.get("default", "")
        widget_kind = str(schema.get("x-widget") or schema.get("widget") or "")

        if widget_kind in {"dataset_select", "dataset"} or name.endswith("_dataset_id"):
            options = [""] + self._dataset_ids()
            value = default if default in options else ""
            return pn.widgets.Select(name="", options=options, value=value, sizing_mode="stretch_width")

        if (
            widget_kind in {"column_select", "column"}
            or name.endswith("_column")
            or name in {
                "target",
                "label_column",
                "target_column",
                "image_column",
                "mask_column",
            }
        ):
            columns = [""] + self._dataset_columns(self.dataset_select.value)
            value = default if default in columns else ""
            return pn.widgets.Select(name="", options=columns, value=value, sizing_mode="stretch_width")

        if "enum" in schema:
            values = list(schema.get("enum") or [])
            options = {str(v): v for v in values}
            value = default if default in values else (values[0] if values else None)
            return pn.widgets.Select(name="", options=options, value=value, sizing_mode="stretch_width")

        if kind in {"integer", "int"}:
            return pn.widgets.IntInput(
                name="",
                value=int(default or 0),
                start=schema.get("minimum"),
                end=schema.get("maximum"),
                sizing_mode="stretch_width",
            )

        if kind in {"number", "float"}:
            return pn.widgets.FloatInput(
                name="",
                value=float(default or 0.0),
                start=schema.get("minimum"),
                end=schema.get("maximum"),
                sizing_mode="stretch_width",
            )

        if kind in {"boolean", "bool"}:
            return pn.widgets.Checkbox(name="", value=bool(default), sizing_mode="stretch_width")

        if kind in {"array", "object"}:
            text = json.dumps(default if default not in ("", None) else ([] if kind == "array" else {}))
            return pn.widgets.TextAreaInput(name="", value=text, height=100, sizing_mode="stretch_width")

        return pn.widgets.TextInput(
            name="",
            value="" if default is None else str(default),
            placeholder=str(schema.get("description") or ""),
            sizing_mode="stretch_width",
        )

    def _build_protocol_section(self, *, managed: bool):
        """Build the advanced validation/test protocol block.

        Returns the layout object (or None) instead of appending to
        ``recipe_params_area`` directly. The caller assembles the full list and
        assigns ``recipe_params_area.objects`` in a single shot, which avoids
        the stale-parent-height collapse caused by incremental ``.append``.
        """
        if not managed:
            return None

        column_options = [""] + self._dataset_columns(self.dataset_select.value)

        advanced_toggle = pn.widgets.Checkbox(
            name="Show advanced validation/test protocol",
            value=False,
            sizing_mode="stretch_width",
        )
        self._fit_widget(advanced_toggle)

        self.protocol_widgets = {
            "protocol_split_strategy": pn.widgets.Select(
                name="",
                options={
                    "Random split": "random",
                    "Group-aware split": "by_group",
                    "Temporal split": "temporal",
                    "Predefined split column": "predefined",
                },
                value="random",
            ),
            "protocol_validation_source": pn.widgets.Select(
                name="",
                options={
                    "Split from AL training rows": "split",
                    "Use selected validation dataset": "dataset",
                },
                value="dataset" if self.validation_dataset_select.value else "split",
            ),
            "protocol_validation_dataset_id": self.validation_dataset_select,
            "protocol_test_source": pn.widgets.Select(
                name="",
                options={
                    "Split from AL training rows": "split",
                    "Use selected test dataset": "dataset",
                    "No test set": "none",
                },
                value="dataset" if self.test_dataset_select.value else "split",
            ),
            "protocol_test_dataset_id": self.test_dataset_select,
            "protocol_group_column": pn.widgets.Select(
                name="",
                options=column_options,
                value="",
            ),
            "protocol_split_column": pn.widgets.Select(
                name="",
                options=column_options,
                value="",
            ),
            "protocol_validation_size": pn.widgets.FloatInput(
                name="",
                value=0.1,
                start=0.01,
                end=0.8,
                step=0.01,
            ),
            "protocol_test_size": pn.widgets.FloatInput(
                name="",
                value=0.2,
                start=0.0,
                end=0.8,
                step=0.01,
            ),
            "protocol_selection_metric": pn.widgets.Select(
                name="",
                options={
                    "Validation accuracy": "val_accuracy",
                    "Validation macro F1": "val_f1_macro",
                    "Validation loss": "val_loss",
                },
                value="val_accuracy",
            ),
            "protocol_random_state": pn.widgets.IntInput(
                name="",
                value=int(self.seed.value or 42),
            ),
        }

        advanced_body = pn.Column(
            self._field("Split method", self.protocol_widgets["protocol_split_strategy"]),
            self._field("Validation source", self.protocol_widgets["protocol_validation_source"]),
            self._field("Test source", self.protocol_widgets["protocol_test_source"]),
            self._field("Group/time column", self.protocol_widgets["protocol_group_column"]),
            self._field("Predefined split column", self.protocol_widgets["protocol_split_column"]),
            self._field("Validation fraction", self.protocol_widgets["protocol_validation_size"]),
            self._field("Test fraction", self.protocol_widgets["protocol_test_size"]),
            self._field("Best-epoch metric", self.protocol_widgets["protocol_selection_metric"]),
            self._field("Protocol random seed", self.protocol_widgets["protocol_random_state"]),
            visible=False,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "padding-top": "6px",
            },
        )

        def _toggle_advanced(event: Any) -> None:
            advanced_body.visible = bool(getattr(event, "new", False))

        advanced_toggle.param.watch(_toggle_advanced, "value")

        return pn.Column(
            pn.layout.Divider(margin=(8, 0, 6, 0)),
            advanced_toggle,
            advanced_body,
            sizing_mode="stretch_width",
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "visible",
                "clear": "both",
            },
        )

    def _protocol_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, widget in self.protocol_widgets.items():
            try:
                params[key] = widget.value
            except Exception:
                pass

        validation_dataset_id = self.validation_dataset_select.value or ""
        test_dataset_id = self.test_dataset_select.value or ""

        if validation_dataset_id:
            params["protocol_validation_source"] = "dataset"
            params["protocol_validation_dataset_id"] = validation_dataset_id

        if test_dataset_id:
            params["protocol_test_source"] = "dataset"
            params["protocol_test_dataset_id"] = test_dataset_id

        params["validation_dataset_id"] = validation_dataset_id
        params["test_dataset_id"] = test_dataset_id
        params["al_protocol"] = self.al_protocol.value or "review"
        return params

    def _refresh_recipe_params(self) -> None:
        previous_values: Dict[str, Any] = {}
        for name, widget in self.recipe_param_widgets.items():
            try:
                previous_values[str(name)] = widget.value
            except Exception:
                pass

        previous_protocol_values: Dict[str, Any] = {}
        for name, widget in self.protocol_widgets.items():
            try:
                previous_protocol_values[str(name)] = widget.value
            except Exception:
                pass

        spec = self._selected_recipe_spec()
        self.recipe_param_widgets = {}
        self.recipe_param_fields = {}
        self.protocol_widgets = {}
        self.protocol_fields = {}

        if spec is None:
            self.recipe_card.object = "No recipe selected."
            # Single assignment; never incrementally mutate a mounted Column.
            self.recipe_params_area.objects = []
            return

        recipe_cls = getattr(spec, "recipe_cls", None)
        execution_mode = (
            getattr(recipe_cls, "execution_mode", None)
            or getattr(spec, "execution_mode", "freeform")
        )
        managed = str(execution_mode) == "managed"

        required_mappings = getattr(spec, "required_mappings", []) or []

        self.recipe_card.object = (
            f"`{getattr(spec, 'id', '')}` v{getattr(spec, 'version', '')}\n\n"
            f"{getattr(spec, 'description', '')}\n\n"
            f"- Task: `{getattr(spec, 'task', '')}`\n"
            f"- Modality: `{getattr(spec, 'modality', '')}`\n"
            f"- Mode: `{'managed' if managed else 'freeform'}`\n"
            f"- Required mappings: `{', '.join(required_mappings) or 'none'}`"
        )

        properties = (getattr(spec, "params_schema", {}) or {}).get("properties", {}) or {}

        # Build the entire child list locally, then assign it to the mounted
        # Column in ONE operation. Incremental .append() on an already-rendered
        # Column leaves Bokeh with a stale parent height, so following siblings
        # (the divider + train button) get drawn over the params area's tail.
        new_objects: List[Any] = []

        if properties:
            for name, schema in properties.items():
                name = str(name)
                widget = self._widget_for_schema(name, schema)
                if name in previous_values:
                    try:
                        widget.value = previous_values[name]
                    except Exception:
                        pass
                field = self._form_row(str(schema.get("title") or name), widget)
                self.recipe_param_widgets[name] = widget
                self.recipe_param_fields[name] = field
                new_objects.append(field)
        else:
            new_objects.append(
                pn.pane.Markdown(
                    "This recipe exposes no extra parameters.",
                    sizing_mode="stretch_width",
                    styles={"overflow-wrap": "anywhere"},
                )
            )

        protocol_block = self._build_protocol_section(managed=managed)
        for name, value in previous_protocol_values.items():
            widget = self.protocol_widgets.get(name)
            if widget is not None:
                try:
                    widget.value = value
                except Exception:
                    pass
        if protocol_block is not None:
            new_objects.append(protocol_block)

        self.recipe_params_area.objects = new_objects

        # Reserve enough height for the children so the inner column never
        # compresses them inside the fixed-height scroll box.
        est = 0
        for widget in self.recipe_param_widgets.values():
            try:
                est += int(getattr(widget, "min_height", 54) or 54) + 14
            except Exception:
                est += 68
        if managed:
            est += 90
        try:
            self.recipe_params_area.min_height = max(est, 60)
        except Exception:
            pass

    def _refresh_validation_test_datasets(self) -> None:
        dataset_options = {"Split from AL training rows": ""}
        dataset_options.update({dataset_id: dataset_id for dataset_id in self._dataset_ids()})
        for widget in (self.validation_dataset_select, self.test_dataset_select):
            current = widget.value
            widget.options = dataset_options
            widget.value = current if current in dataset_options.values() else ""

    def _refresh_labels_from_available_context(self) -> None:
        current = al_state.parse_label_options(self.label_options_select.value)

        labels: List[str] = []
        labels.extend(self._labels_from_selected_target_column())
        labels.extend(self._labels_from_selected_recipe())
        labels.extend(self._labels_from_selected_predictions())
        labels.extend(current)

        labels = al_state.parse_label_options(labels)

        self.label_options_select.options = labels

        if current:
            self.label_options_select.value = [
                label for label in current if label in labels
            ]
        else:
            self.label_options_select.value = labels

        self._set_label_select_options()

    def _labels_from_selected_recipe(self) -> List[str]:
        spec = self._selected_recipe_spec()
        if spec is None:
            return []

        labels: List[str] = []
        schema = getattr(spec, "params_schema", {}) or {}
        properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}

        for key in ("labels", "class_labels", "classes", "label_options"):
            prop = properties.get(key, {}) if isinstance(properties, Mapping) else {}
            enum = prop.get("enum") if isinstance(prop, Mapping) else None
            default = prop.get("default") if isinstance(prop, Mapping) else None

            if enum:
                labels.extend(str(item) for item in enum)
            if isinstance(default, list):
                labels.extend(str(item) for item in default)

        for attr in ("labels", "class_labels", "classes"):
            value = getattr(spec, attr, None) or getattr(getattr(spec, "recipe_cls", None), attr, None)
            if isinstance(value, (list, tuple, set)):
                labels.extend(str(item) for item in value)

        return al_state.parse_label_options(labels)

    def _labels_from_selected_predictions(self) -> List[str]:
        artifact_id = str(self.predictions_select.value or "").strip()
        if not artifact_id:
            return []
        try:
            payload = self.context.artifacts.get(artifact_id)
            return actions._infer_label_options(payload)
        except Exception:
            return []

    def _apply_recipe_inferred_defaults(self) -> None:
        spec = self._selected_recipe_spec()
        dataset_id = self.dataset_select.value
        if spec is None or not dataset_id:
            return

        inferred: Dict[str, Any] = {}
        registry = self._recipe_registry()
        infer = getattr(registry, "infer_recipe_params", None)
        if callable(infer):
            try:
                inferred = dict(infer(self.context, dataset_id, spec) or {})
            except Exception:
                inferred = {}

        for name, value in inferred.items():
            widget = self.recipe_param_widgets.get(str(name))
            if widget is None:
                continue
            try:
                if widget.value in (None, "", [], {}):
                    widget.value = value
            except Exception:
                pass

    def _dataset_ids(self) -> List[str]:
        try:
            return list(self.context.datasets.list_ids())
        except Exception:
            return []

    def _dataset_columns(self, dataset_id: Any) -> List[str]:
        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return []
        try:
            return [str(col) for col in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            try:
                return [str(col) for col in self.context.datasets.get_df(dataset_id).columns]
            except Exception:
                return []

    def _guess_label_column(self, columns: List[str]) -> str:
        lowered = {str(col).lower(): str(col) for col in columns}

        for candidate in (
            "target_label",
            "label",
            "labels",
            "class",
            "class_label",
            "target",
            "y",
        ):
            if candidate in lowered:
                return lowered[candidate]

        return "al_label"

    def _dataset_mapping(self, dataset_id: Any, semantic_name: str) -> Optional[str]:
        try:
            value = self.context.datasets.get_mapping(str(dataset_id), semantic_name)
            return None if value is None else str(value)
        except Exception:
            return None

    def _recipe_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        spec = self._selected_recipe_spec()
        properties = (getattr(spec, "params_schema", {}) or {}).get("properties", {}) if spec else {}

        for name, widget in self.recipe_param_widgets.items():
            value = widget.value
            schema = properties.get(name, {}) if isinstance(properties, Mapping) else {}
            kind = str(schema.get("type", "string"))
            if kind in {"array", "object"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
            params[name] = value

        params.update(self._protocol_params())
        params["recipe_id"] = self.recipe_select.value
        params["dataset_id"] = self.dataset_select.value
        return params

    # ------------------------------------------------------------------
    # Workflow gating
    # ------------------------------------------------------------------
    def _labelled_count(self, session: Optional[Mapping[str, Any]]) -> int:
        if not session:
            return 0
        try:
            return int(al_state.counts(session).get("labelled_or_verified", 0))
        except Exception:
            return 0

    def _session_has_been_trained(self, session: Optional[Mapping[str, Any]]) -> bool:
        """True once the session has at least one completed training round.

        Primary signal is the round counter (incremented by training). A few
        alternative keys are checked too so the gate still works if a given
        platform records training differently. If your `start` action sets
        `round` to 1, adjust the round check below to `>= 2`.
        """
        if not session:
            return False
        try:
            if int(session.get("round") or 0) >= 1:
                return True
        except Exception:
            pass
        for key in ("model_artifact_id", "last_training", "trained_at", "training_history"):
            if session.get(key):
                return True
        return False

    def _update_action_gating(self) -> None:
        if self._disposed:
            return

        # While a job is running, _set_running owns the disabled state.
        if self._job_running:
            return

        session = self._load_current_session()
        has_session = bool(session)
        labelled = self._labelled_count(session)
        trained = self._session_has_been_trained(session)

        # Train requires at least one labelled / verified point.
        can_train = has_session and labelled >= 1
        try:
            self.train_button.disabled = not can_train
        except Exception:
            pass
        if not has_session:
            self.train_hint.object = "_Start or select a session first._"
        elif labelled < 1:
            self.train_hint.object = "_Label at least one point in the Review tab before training._"
        else:
            self.train_hint.object = ""

        # Query requires a trained model and predictions over the original pool
        # dataset. Predictions over the derived AL training dataset are not
        # valid query inputs.
        selected_prediction = str(self.predictions_select.value or "").strip()
        prediction_ok = bool(
            selected_prediction
            and self._prediction_matches_current_session(selected_prediction)
        )

        can_query = has_session and trained and prediction_ok
        try:
            self.query_button.disabled = not can_query
        except Exception:
            pass

        if not has_session:
            self.query_hint.object = "_Start or select a session first._"
        elif not trained:
            self.query_hint.object = "_Train at least once before querying from predictions._"
        elif not selected_prediction:
            expected_dataset_id = self._session_pool_dataset_id(session)
            self.query_hint.object = (
                "_No matching prediction artifact is selected. Run "
                f"`core.ml.predict` on `{expected_dataset_id}` using the trained "
                "model, then query from that prediction artifact._"
            )
        elif not prediction_ok:
            expected_dataset_id = self._session_pool_dataset_id(session)
            actual_dataset_id = self._prediction_dataset_id(selected_prediction)
            self.query_hint.object = (
                "_Selected predictions do not match the AL pool dataset. "
                f"Expected `{expected_dataset_id}`, got `{actual_dataset_id}`._"
            )
        else:
            self.query_hint.object = ""

    def _require_dataset(self) -> str:
        dataset_id = str(self.dataset_select.value or "").strip()
        if not dataset_id:
            raise ValueError("Choose a dataset.")
        return dataset_id

    def _require_session(self) -> str:
        session_artifact_id = str(self._current_session_artifact_id or self.session_select.value or "").strip()
        if not session_artifact_id:
            raise ValueError("Start or choose an active-learning session.")
        return session_artifact_id

    def _require_labelled(self) -> None:
        session = self._load_current_session()
        if self._labelled_count(session) < 1:
            raise ValueError(
                "Label at least one point in the Review tab before training."
            )

    def _require_trained(self) -> None:
        session = self._load_current_session()
        if not self._session_has_been_trained(session):
            raise ValueError(
                "Train at least once before querying a top-k batch from predictions."
            )

    def _require_prediction_artifact(self) -> str:
        artifact_id = str(self.predictions_select.value or "").strip()
        expected_dataset_id = self._session_pool_dataset_id()

        if not artifact_id:
            if expected_dataset_id:
                raise ValueError(
                    "Choose an ml.predictions artifact for the AL pool dataset. "
                    f"No matching prediction artifact is currently selected. Run "
                    f"`core.ml.predict` on `{expected_dataset_id}` using the latest "
                    "trained model, then query again."
                )
            raise ValueError("Choose an ml.predictions artifact.")

        actual_dataset_id = self._prediction_dataset_id(artifact_id)

        if expected_dataset_id and actual_dataset_id != expected_dataset_id:
            raise ValueError(
                "The selected prediction artifact was not produced for this AL "
                "session's pool dataset.\n\n"
                f"Expected pool dataset: {expected_dataset_id!r}\n"
                f"Prediction dataset: {actual_dataset_id!r}\n\n"
                "For AL querying, use predictions over the original pool/source "
                "dataset, not over the derived AL training dataset. Run "
                "`core.ml.predict` on the pool dataset with the trained model."
            )

        return artifact_id

    def _set_running(self, running: bool) -> None:
        self._job_running = running
        self.train_button.disabled = running
        self.start_button.disabled = running
        self.query_button.disabled = running
        self.label_button.disabled = running
        self.unsure_button.disabled = running
        if not running:
            # Re-apply workflow gating now that the job has finished.
            self._update_action_gating()

    def _set_status(self, message: str, alert_type: str = "info") -> None:
        self.status.object = message
        self.status.alert_type = alert_type
        self.status.visible = True

    def _set_error(self, title: str, error: Any) -> None:
        detail = str(error)
        self.status.object = f"**{title}**\n\n{detail}"
        self.status.alert_type = "danger"
        self.status.visible = True
        try:
            print(
                f"[ActiveLearningPanel] {title}: {detail}\n{traceback.format_exc()}",
                flush=True,
            )
        except Exception:
            pass


def _make_table():
    empty = pd.DataFrame()

    try:
        return pn.widgets.Tabulator(
            empty,
            pagination="remote",
            page_size=10,
            disabled=True,
            sizing_mode="stretch_width",
            height=240,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
            },
        )
    except Exception:
        return pn.widgets.DataFrame(
            empty,
            disabled=True,
            sizing_mode="stretch_width",
            height=240,
            styles={
                "max-width": "100%",
                "width": "100%",
                "box-sizing": "border-box",
                "overflow": "auto",
            },
        )


def _set_table_value(widget: Any, df: pd.DataFrame) -> None:
    try:
        widget.value = df
    except Exception:
        try:
            widget.object = df
        except Exception:
            pass