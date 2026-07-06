from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import threading

from astronomicAL.platform.plugins.specs import ActionRequest

from . import actions
from . import acquisition
from . import analytics
from . import state as al_state

try:  # pragma: no cover - UI import is environment-specific.
    import panel as pn
except Exception:  # pragma: no cover
    pn = None


class ActiveLearningPanel:
    """Thin UI around AL actions/services.

    The panel intentionally stays small: it discovers datasets, columns, labels,
    strategies, and core.ml recipes from platform services, then delegates work to
    actions.  It can be opened before data exists; dataset/platform events refresh
    the menus when data is later registered or modified.
    """

    DATASET_REFRESH_TOPICS = (
        "dataset.registered",
        "dataset.updated",
        "dataset.removed",
        "dataset.active.changed",
        "dataset.columns.changed",
        "dataset.mapping.changed",
        "datasets.registered",
        "datasets.updated",
        "datasets.removed",
        "datasets.active.changed",
        "datasets.columns.changed",
        "datasets.mapping.changed",
        "dataset.*",
        "datasets.*",
    )
    RECIPE_REFRESH_TOPICS = (
        "plugin.enabled",
        "plugin.disabled",
        "plugin.reloaded",
        "service.registered",
        "core.ml.recipe_registry.changed",
        "core.ml.recipe_profile_store.changed",
        "ml.recipe.registered",
        "ml.recipe_profile.saved",
        "ml.recipe_profiles.changed",
    )
    REVIEW_REFRESH_TOPICS = (
        "selection.focus.changed",
        "selection.changed",
    )
    SESSION_REFRESH_TOPICS = (
        "al.session.saved",
        "al.session.created",
        "al.label.recorded",
        "al.labels.bulk_recorded",
        "al.query_batch.created",
        "al.round.training_started",
        "al.round.training_finished",
        "al.round.training_failed",
        "al.round.prediction_or_query_failed",
        "ml.recipe_run.started",
        "ml.recipe_run.finished",
        "ml.recipe_run.failed",
        "ml.training.started",
        "ml.training.finished",
        "ml.training.failed",
    )

    def __init__(self, *, context: Any, restore_state: Optional[Mapping[str, Any]] = None) -> None:
        self.context = context
        self.session_artifact_id = ""
        self.predictions_artifact_id = ""
        self.model_artifact_id = ""
        self.dataset_id = ""
        self.label_column = ""
        self.selected_labels: List[str] = []
        self.recipe_profile_id = ""
        self.recipe_id = ""  # legacy fallback only
        self.status = "Ready"
        self._busy = False
        self._view = None
        self._tabs = None
        self._tab_index: Dict[str, int] = {}
        self._widgets: Dict[str, Any] = {}
        self._subscriptions: List[Any] = []
        self._performance_event_rows: List[Dict[str, Any]] = []
        self._pending_refresh = False
        self._doc = None
        if restore_state:
            self.restore_state(restore_state)
        self._subscribe_to_platform_events()

    def panel(self):
        if pn is None:
            return self
        if self._view is None:
            self._view = self._build_view()
            self.refresh_choices(status=False)
        return self._view

    def get_state(self) -> Dict[str, Any]:
        return {
            "session_artifact_id": self.session_artifact_id,
            "predictions_artifact_id": self.predictions_artifact_id,
            "model_artifact_id": self.model_artifact_id,
            "dataset_id": self._widget_value("dataset_id", self.dataset_id),
            "label_column": self._widget_value("label_column", self.label_column),
            "selected_labels": list(self._widget_value("labels", self.selected_labels) or []),
            "recipe_profile_id": self._widget_value("recipe_profile_id", self.recipe_profile_id),
            "recipe_id": self.recipe_id,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.session_artifact_id = str(state.get("session_artifact_id") or "")
        self.predictions_artifact_id = str(state.get("predictions_artifact_id") or "")
        self.model_artifact_id = str(state.get("model_artifact_id") or "")
        self.dataset_id = str(state.get("dataset_id") or "")
        self.label_column = str(state.get("label_column") or "")
        labels = state.get("selected_labels") or []
        if isinstance(labels, str):
            labels = [part.strip() for part in labels.replace("\n", ",").split(",") if part.strip()]
        self.selected_labels = [str(label) for label in labels if label not in (None, "")]
        self.recipe_profile_id = str(state.get("recipe_profile_id") or state.get("recipe_id") or "")
        self.recipe_id = str(state.get("recipe_id") or "")

    def close(self) -> None:
        for subscription in self._subscriptions:
            if callable(subscription):
                try:
                    subscription()
                except Exception:
                    pass
        self._subscriptions.clear()

    def _build_view(self):
        pn.extension()
        self._doc = getattr(pn.state, "curdoc", None)
        dataset_options = self._dataset_options()
        dataset_value = self._valid_or_default(self.dataset_id, dataset_options)
        column_options = self._label_column_options(dataset_value)
        column_value = self._valid_or_default(self.label_column, column_options, allow_blank=True)
        labels = self.selected_labels or self._infer_label_options(dataset_value, column_value)
        recipe_options = self._recipe_options()
        recipe_value = self._valid_or_default(self.recipe_profile_id, recipe_options, allow_blank=True)

        self._widgets = {
            "dataset_id": pn.widgets.Select(name="Dataset", options=dataset_options, value=dataset_value),
            "label_column": pn.widgets.Select(name="Label column", options=column_options, value=column_value),
            "labels": pn.widgets.MultiChoice(name="Labels to use", options=labels, value=labels, disabled=not bool(column_value)),
            "initial_k": pn.widgets.IntInput(name="Initial random sample", value=20, start=0),
            "seed": pn.widgets.IntInput(name="Seed", value=42),
            "session_id": pn.widgets.TextInput(name="Session artifact id", value=self.session_artifact_id),
            "model_id": pn.widgets.TextInput(name="Model artifact id", value=self.model_artifact_id),
            "predictions_id": pn.widgets.TextInput(name="Predictions artifact id", value=self.predictions_artifact_id),
            "strategy": pn.widgets.Select(name="Query strategy", options=self._strategy_options(), value="least_confidence"),
            "query_k": pn.widgets.IntInput(name="Query batch size", value=200, start=1),
            "row_id": pn.widgets.TextInput(name="Start row id", placeholder="blank = focused row or first unlabelled in latest batch"),
            "label": pn.widgets.Select(name="Label", options=self._review_label_options(labels), value=self._first_review_label(labels)),
            "bulk_n": pn.widgets.IntInput(name="Bulk label next N", value=5, start=1),
            "recipe_profile_id": pn.widgets.Select(name="core.ml recipe profile", options=recipe_options, value=recipe_value),
            "status": pn.pane.Markdown(self._status_text()),
            "session_summary": pn.pane.Markdown(self._session_summary_text()),
            "performance_metric": pn.widgets.Select(name="Performance metric", options={"No metrics yet": ""}, value=""),
            "performance_plot": pn.pane.Markdown("No AL performance points yet."),
            "xy_x": pn.widgets.Select(name="X column", options={"Select X column": ""}, value=""),
            "xy_y": pn.widgets.Select(name="Y column", options={"Select Y column": ""}, value=""),
            "xy_scope": pn.widgets.Select(
                name="Data shown",
                options={
                    "Train/pool + validation": "combined",
                    "Train/pool only": "pool",
                    "Validation only": "validation",
                },
                value="combined",
            ),
            "xy_validation_dataset_id": pn.widgets.Select(name="Validation dataset", options=self._dataset_options_with_blank("No validation dataset"), value=""),
            "xy_colour": pn.widgets.Select(
                name="Colour by",
                options={
                    "Prediction correctness": "prediction_correctness",
                    "Training status": "training_status",
                    "Last query score / informativeness": "last_query_score",
                    "Last query rank": "last_query_rank",
                    "Source label": "label",
                },
                value="prediction_correctness",
            ),
            "xy_max_points": pn.widgets.IntInput(name="Max plotted rows", value=5000, start=100),
            "xy_plot": pn.pane.Matplotlib(None, tight=True, sizing_mode="stretch_width", height=280, min_width=220, min_height=180),
        }

        start_btn = pn.widgets.Button(name="Start session", button_type="primary")
        query_btn = pn.widgets.Button(name="Create query batch", button_type="primary")
        label_btn = pn.widgets.Button(name="Record label", button_type="success")
        bulk_label_btn = pn.widgets.Button(name="Next N labels from column", button_type="success")
        train_btn = pn.widgets.Button(name="Train via core.ml", button_type="warning")
        refresh_data_btn = pn.widgets.Button(name="Refresh datasets", button_type="default")
        refresh_strategy_btn = pn.widgets.Button(name="Refresh strategies", button_type="default")
        refresh_recipe_btn = pn.widgets.Button(name="Refresh profiles", button_type="default")
        refresh_performance_btn = pn.widgets.Button(name="Refresh performance", button_type="default")
        refresh_xy_btn = pn.widgets.Button(name="Refresh XY plot", button_type="default")
        self._widgets.update({
            "start_btn": start_btn,
            "query_btn": query_btn,
            "label_btn": label_btn,
            "bulk_label_btn": bulk_label_btn,
            "train_btn": train_btn,
            "refresh_performance_btn": refresh_performance_btn,
            "refresh_xy_btn": refresh_xy_btn,
        })

        self._widgets["dataset_id"].param.watch(lambda event: self._on_dataset_changed(str(event.new or "")), "value")
        self._widgets["label_column"].param.watch(lambda event: self._on_label_column_changed(str(event.new or "")), "value")
        self._widgets["labels"].param.watch(lambda event: self._on_labels_changed(list(event.new or [])), "value")
        self._widgets["recipe_profile_id"].param.watch(lambda event: self._on_recipe_profile_changed(str(event.new or "")), "value")
        self._widgets["row_id"].param.watch(lambda event: self._on_review_row_changed(str(event.new or "")), "value")
        self._widgets["performance_metric"].param.watch(lambda event: self._refresh_performance(status=False, keep_metric=True), "value")
        self._widgets["xy_x"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_y"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_colour"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_scope"].param.watch(lambda event: self._on_xy_dataset_choice_changed(), "value")
        self._widgets["xy_validation_dataset_id"].param.watch(lambda event: self._on_xy_dataset_choice_changed(), "value")
        self._widgets["xy_max_points"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")

        start_btn.on_click(lambda event: self._run_start())
        query_btn.on_click(lambda event: self._run_query())
        label_btn.on_click(lambda event: self._run_label())
        bulk_label_btn.on_click(lambda event: self._run_bulk_label())
        train_btn.on_click(lambda event: self._run_train())
        refresh_data_btn.on_click(lambda event: self.refresh_dataset_controls())
        refresh_strategy_btn.on_click(lambda event: self._refresh_strategies())
        refresh_recipe_btn.on_click(lambda event: self._refresh_recipes())
        refresh_performance_btn.on_click(lambda event: self._refresh_performance())
        refresh_xy_btn.on_click(lambda event: self._refresh_xy_plot())

        start_tab = pn.Column(
            "### Start",
            self._widgets["dataset_id"],
            self._widgets["label_column"],
            self._widgets["labels"],
            "Select a label column first. The label set is inferred from that column; remove labels here to limit the session classes.",
            pn.Row(self._widgets["initial_k"], self._widgets["seed"]),
            pn.Row(start_btn, refresh_data_btn),
        )
        query_tab = pn.Column(
            "### Query",
            self._widgets["session_id"],
            self._widgets["model_id"],
            self._widgets["predictions_id"],
            pn.Row(self._widgets["strategy"], refresh_strategy_btn),
            self._widgets["query_k"],
            query_btn,
        )
        review_tab = pn.Column(
            "### Review",
            self._widgets["session_id"],
            self._widgets["row_id"],
            self._widgets["label"],
            label_btn,
            pn.Row(self._widgets["bulk_n"], bulk_label_btn),
            "`Next N labels from column` uses each row's pre-assigned value in the selected label column; it does not repeat the dropdown value.",
            "Use label `Unsure` to remove a row from the query pool without adding it to training.",
        )
        train_tab = pn.Column(
            "### Train",
            self._widgets["session_id"],
            pn.Row(self._widgets["recipe_profile_id"], refresh_recipe_btn),
            self._widgets["seed"],
            train_btn,
            "Training materialises the currently labelled rows and updates model/prediction artifacts. Create the next query batch from the Query tab after choosing the strategy and batch size.",
        )
        performance_tab = pn.Column(
            "### AL Performance",
            "Each point is one completed active-learning training round. The x-axis is the number of labelled training rows used in that round.",
            pn.Row(self._widgets["performance_metric"], refresh_performance_btn),
            self._widgets["performance_plot"],
        )
        xy_tab = pn.Column(
            "### XY Diagnostics",
            "Plot the pool in two dataset columns. Labelled/trained rows are overlaid, and the latest query batch can be coloured by rank or score to inspect where the selected query strategy found informative points.",
            pn.Row(self._widgets["xy_scope"], self._widgets["xy_validation_dataset_id"]),
            pn.Row(self._widgets["xy_x"], self._widgets["xy_y"]),
            pn.Row(self._widgets["xy_colour"], self._widgets["xy_max_points"], refresh_xy_btn),
            self._widgets["xy_plot"],
        )
        self._sync_start_controls()
        self._sync_train_controls()
        self._refresh_session_summary(status=False)
        self._refresh_performance(status=False)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._refresh_xy_plot(status=False)
        self._tab_index = {"Start": 0, "Query": 1, "Review": 2, "Train": 3, "Performance": 4, "XY": 5}
        self._tabs = pn.Tabs(("Start", start_tab), ("Query", query_tab), ("Review", review_tab), ("Train", train_tab), ("Performance", performance_tab), ("XY", xy_tab))
        return pn.Column(
            "## Active Learning",
            self._widgets["status"],
            self._widgets["session_summary"],
            self._tabs,
        )

    def refresh_choices(self, *, status: bool = True) -> None:
        self.refresh_dataset_controls(status=False)
        self._refresh_strategies(status=False)
        self._refresh_recipes(status=False)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._refresh_xy_plot(status=False)
        if status:
            self._set_status("Choices refreshed.")

    def refresh_dataset_controls(self, *, status: bool = True) -> None:
        if pn is None or not self._widgets:
            self._pending_refresh = True
            return
        dataset_widget = self._widgets.get("dataset_id")
        column_widget = self._widgets.get("label_column")
        if dataset_widget is None or column_widget is None:
            return
        old_dataset = str(dataset_widget.value or self.dataset_id or "")
        dataset_options = self._dataset_options()
        dataset_widget.options = dataset_options
        dataset_widget.value = self._valid_or_default(old_dataset, dataset_options)
        self._refresh_label_columns(dataset_widget.value, keep_current=True)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._refresh_xy_plot(status=False)
        if status:
            self._set_status("Dataset and label-column choices refreshed.")

    def _refresh_label_columns(self, dataset_id: str, *, keep_current: bool) -> None:
        column_widget = self._widgets.get("label_column")
        if column_widget is None:
            return
        old_column = str(column_widget.value or self.label_column or "") if keep_current else ""
        column_options = self._label_column_options(dataset_id)
        column_widget.options = column_options
        column_widget.value = self._valid_or_default(old_column, column_options, allow_blank=True)
        self._refresh_inferred_labels(dataset_id, str(column_widget.value or ""), keep_selected=keep_current)

    def _refresh_inferred_labels(self, dataset_id: str, label_column: str, *, keep_selected: bool) -> None:
        labels_widget = self._widgets.get("labels")
        if labels_widget is None:
            return
        inferred = self._infer_label_options(dataset_id, label_column) if dataset_id and label_column else []
        selected = [str(value) for value in (labels_widget.value or []) if str(value) in set(inferred)] if keep_selected else []
        if not selected:
            selected = list(inferred)
        labels_widget.options = inferred
        labels_widget.value = selected
        labels_widget.disabled = not bool(label_column)
        self._sync_label_controls(selected)
        self._sync_start_controls()

    def _on_dataset_changed(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self._refresh_label_columns(dataset_id, keep_current=False)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._refresh_xy_plot(status=False)
        self._set_status("Dataset changed. Label columns, inferred labels, and XY columns updated.")

    def _on_label_column_changed(self, label_column: str) -> None:
        self.label_column = label_column
        dataset_id = str(self._widgets.get("dataset_id").value or "") if self._widgets.get("dataset_id") is not None else ""
        self._refresh_inferred_labels(dataset_id, label_column, keep_selected=False)
        if label_column:
            self._set_status(f"Labels inferred from `{label_column}`.")
        else:
            self._set_status("Select a label column to infer labels.")

    def _on_labels_changed(self, labels: Sequence[str]) -> None:
        self.selected_labels = [str(label) for label in labels if label not in (None, "")]
        self._sync_label_controls(self.selected_labels)
        self._sync_start_controls()

    def _sync_label_controls(self, labels: Sequence[str]) -> None:
        label_widget = self._widgets.get("label")
        if label_widget is None:
            return
        options = self._review_label_options(labels)
        old_value = str(label_widget.value or "")
        label_widget.options = options
        label_widget.value = old_value if old_value in set(options.values()) else self._first_review_label(labels)

    def _sync_start_controls(self) -> None:
        button = self._widgets.get("start_btn")
        if button is None:
            return
        dataset_id = str(self._widget_value("dataset_id", "") or "")
        label_column = str(self._widget_value("label_column", "") or "")
        labels = list(self._widget_value("labels", []) or [])
        button.disabled = not bool(dataset_id and label_column and labels)

    def _sync_train_controls(self) -> None:
        button = self._widgets.get("train_btn")
        if button is None:
            return
        button.disabled = not bool(self._widget_value("recipe_profile_id", ""))

    def _run_start(self) -> None:
        self._set_status("Starting active-learning session and drawing the initial random review batch...")
        self._set_button_busy("start_btn", True)
        try:
            labels = list(self._widgets["labels"].value or [])
            dataset_id = str(self._widgets["dataset_id"].value or "").strip()
            label_column = str(self._widgets["label_column"].value or "").strip()
            if not dataset_id:
                raise ValueError("Select a dataset before starting an active-learning session.")
            if not label_column:
                raise ValueError("Select a label column so the label set can be inferred.")
            if not labels:
                raise ValueError("Select at least one label for the active-learning session.")
            result = actions.start_session_action(
                self.context,
                ActionRequest(
                    dataset_id=dataset_id,
                    row_ids=None,
                    columns=[],
                    params={
                        "dataset_id": dataset_id,
                        "label_options": labels,
                        "target_column": label_column,
                        "label_column": label_column,
                        "infer_labels_from_column": True,
                        "initial_k": self._widgets["initial_k"].value,
                        "seed": self._widgets["seed"].value,
                        "make_selection": True,
                    },
                    artifact_id=None,
                    origin="core.active_learning.panel",
                ),
            )
            self._set_session_id(result.get("session_artifact_id"))
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            initial_row_ids = [str(row_id) for row_id in (result.get("initial_row_ids") or []) if str(row_id)]
            if initial_row_ids:
                self._set_review_row(initial_row_ids[0])
            self._select_tab("Review")
            count = result.get("count", len(initial_row_ids))
            self._set_status(f"Started session `{result.get('session_id')}` with {count} initial rows. Review tab selected.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("start_btn", False)

    def _run_query(self) -> None:
        self._set_status("Creating query batch from the selected predictions and strategy...")
        self._set_button_busy("query_btn", True)
        try:
            session_id = self._widgets["session_id"].value.strip()
            predictions_id = self._widgets["predictions_id"].value.strip()
            result = actions.query_batch_action(
                self.context,
                ActionRequest(
                    dataset_id=None,
                    row_ids=None,
                    columns=[],
                    params={
                        "session_artifact_id": session_id,
                        "predictions_artifact_id": predictions_id,
                        "strategy_id": self._widgets["strategy"].value,
                        "k": self._widgets["query_k"].value,
                        "seed": self._widgets["seed"].value,
                        "make_selection": True,
                    },
                    artifact_id=predictions_id or None,
                    origin="core.active_learning.panel",
                ),
            )
            self._set_session_id(result.get("session_artifact_id"))
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            row_ids = [str(row_id) for row_id in (result.get("row_ids") or []) if str(row_id)]
            if row_ids:
                self._set_review_row(row_ids[0])
            self._set_status(f"Created query batch `{result.get('batch_artifact_id')}` with {result.get('count')} rows. Review rows are ready.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("query_btn", False)

    def _run_label(self) -> None:
        self._set_status("Recording label...")
        self._set_button_busy("label_btn", True)
        try:
            result = actions.record_label_action(
                self.context,
                ActionRequest(
                    dataset_id=None,
                    row_ids=None,
                    columns=[],
                    params={
                        "session_artifact_id": self._widgets["session_id"].value.strip(),
                        "row_id": self._widgets["row_id"].value.strip(),
                        "label": self._widgets["label"].value,
                    },
                    artifact_id=None,
                    origin="core.active_learning.panel",
                ),
            )
            self._set_session_id(result.get("session_artifact_id"))
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            self._set_status(f"Recorded `{result.get('display_label')}` for row `{result.get('row_id')}`.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("label_btn", False)

    def _run_bulk_label(self) -> None:
        self._set_status("Recording source-column labels for the next review rows...")
        self._set_button_busy("bulk_label_btn", True)
        try:
            result = actions.bulk_label_next_action(
                self.context,
                ActionRequest(
                    dataset_id=None,
                    row_ids=None,
                    columns=[],
                    params={
                        "session_artifact_id": self._widgets["session_id"].value.strip(),
                        "row_id": self._widgets["row_id"].value.strip(),
                        "label_column": self._session_label_column(),
                        "n": self._widgets["bulk_n"].value,
                        "source": "bulk_column",
                    },
                    artifact_id=None,
                    origin="core.active_learning.panel",
                ),
            )
            self._set_session_id(result.get("session_artifact_id"))
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            row_ids = result.get("row_ids") or []
            if row_ids:
                skipped = result.get("skipped") or []
                self._set_review_row(row_ids[-1])
                self._set_status(
                    f"Recorded source-column labels for {result.get('count')} rows "
                    f"from `{row_ids[0]}` to `{row_ids[-1]}`. Skipped {len(skipped)} rows."
                )
            else:
                self._set_status("No unlabelled rows were available in the latest review/query batch.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("bulk_label_btn", False)

    def _run_train(self) -> None:
        recipe_profile_id = str(self._widgets["recipe_profile_id"].value or "").strip()
        if not recipe_profile_id:
            self._set_status("Error: Select a saved core.ml recipe profile before training.")
            return
        session_id = str(self._widgets["session_id"].value or "").strip()
        if not session_id:
            self._set_status("Error: Start or select an active-learning session before training.")
            return

        seed_value = self._widgets["seed"].value
        self.recipe_profile_id = recipe_profile_id
        self.recipe_id = ""
        self._set_status(
            f"Training active-learning round with recipe profile `{recipe_profile_id}`. This may take a while; the train button will stay busy until the workflow finishes."
        )
        self._set_button_busy("train_btn", True)

        def work() -> None:
            try:
                result = actions.train_from_session_action(
                    self.context,
                    ActionRequest(
                        dataset_id=None,
                        row_ids=None,
                        columns=[],
                        params={
                            "session_artifact_id": session_id,
                            "recipe_profile_id": recipe_profile_id,
                            "target_column": self.label_column,
                            "seed": seed_value,
                            "auto_predict": True,
                            "auto_query": False,
                            "make_selection": False,
                        },
                        artifact_id=None,
                        origin="core.active_learning.panel",
                    ),
                )

                def done() -> None:
                    self._set_button_busy("train_btn", False)
                    self._apply_training_result(result)

                self._next_tick(done)
            except Exception as exc:
                def failed(exc: Exception = exc) -> None:
                    self._set_button_busy("train_btn", False)
                    self._set_status(f"Training failed: {exc}")
                    self._refresh_session_summary(status=False)
                    self._refresh_performance(status=False)

                self._next_tick(failed)

        thread = threading.Thread(target=work, name="ActiveLearningTraining", daemon=True)
        thread.start()

    def _apply_training_result(self, result: Mapping[str, Any]) -> None:
        self._set_session_id(result.get("session_artifact_id"))
        latest = self._sync_latest_from_session(result.get("session_artifact_id"))
        self._set_artifact_widget("model_id", latest.get("model_artifact_id"))
        self._set_artifact_widget("predictions_id", latest.get("predictions_artifact_id"))
        self._refresh_session_summary(status=False)
        self._refresh_performance(status=False)
        self._refresh_xy_plot(status=False)
        self._select_tab("Query")
        status = result.get("workflow_status") or "complete"
        self._set_status(
            f"Training workflow finished with status `{status}`. "
            f"Model `{self._widget_value('model_id', '')}` and predictions `{self._widget_value('predictions_id', '')}` are selected. Choose a query strategy and batch size, then create the next query batch."
        )

    def _refresh_strategies(self, *, status: bool = True) -> None:
        widget = self._widgets.get("strategy") if self._widgets else None
        if widget is not None:
            options = self._strategy_options()
            old_value = str(widget.value or "")
            widget.options = options
            values = set(options.values())
            if old_value not in values:
                widget.value = next(iter(values), "least_confidence")
        if status:
            self._set_status("Strategy list refreshed.")

    def _refresh_recipes(self, *, status: bool = True) -> None:
        # Prefer the new profile selector name, but keep recipe_id fallback so older
        # panel code still works while the UI is being migrated.
        widget = None
        if self._widgets:
            widget = self._widgets.get("recipe_profile_id") or self._widgets.get("recipe_id")

        if widget is not None:
            options = self._recipe_options()

            old_value = str(
                widget.value
                or getattr(self, "recipe_profile_id", "")
                or getattr(self, "recipe_id", "")
                or ""
            )

            widget.options = options
            widget.value = self._valid_or_default(old_value, options, allow_blank=True)

            if hasattr(self, "recipe_profile_id"):
                self.recipe_profile_id = str(widget.value or "")

            # Keep recipe_id clear when using profiles. The profile resolves to the
            # concrete recipe inside core.ml.run_ml_recipe.
            if hasattr(self, "recipe_id"):
                self.recipe_id = ""

            self._sync_train_controls()

        if status:
            self._set_status("Recipe profile list refreshed.")


    def _strategy_options(self) -> Dict[str, str]:
        try:
            registry = actions.get_strategy_registry(self.context)
            return registry.option_map()
        except Exception:
            return {
                "Least confidence": "least_confidence",
                "Smallest margin": "margin",
                "Entropy": "entropy",
                "Random": "random",
            }


    def _recipe_options(self) -> Dict[str, str]:
        """
        Return AL training options.

        After the core_ml recipe-profile update, AL should train from saved
        ml.recipe_profile entries rather than raw recipe IDs. Raw recipes are still
        discoverable in the core_ml Recipe Launcher, where users configure params,
        train/val/test protocol, bindings, and then save a reusable profile.
        """
        store = self._service("core.ml.recipe_profile_store")
        profiles: List[Tuple[str, str]] = []

        if store is not None:
            profiles.extend(self._recipe_profile_options_from_store(store))

        if not profiles:
            return {"No core.ml recipe profiles available": ""}

        deduped: Dict[str, str] = {}
        seen_values: set[str] = set()

        for profile_id, title in profiles:
            profile_id = str(profile_id or "").strip()
            if not profile_id or profile_id in seen_values:
                continue

            label = str(title or profile_id).strip() or profile_id
            deduped[label] = profile_id
            seen_values.add(profile_id)

        return deduped or {"No core.ml recipe profiles available": ""}


    def _recipe_profile_options_from_store(self, store: Any) -> List[Tuple[str, str]]:
        """
        Return [(profile_id, display_label), ...] from core.ml.recipe_profile_store.

        Supports the proposed RecipeProfileStore API:

            store.list() -> list[dict]

        but is deliberately tolerant of nearby shapes while this branch is moving.
        """
        raw_items: Any = None

        for method_name in ("option_map", "options"):
            method = getattr(store, method_name, None)
            if callable(method):
                try:
                    value = method()
                except TypeError:
                    value = method
                if isinstance(value, Mapping):
                    # Expected shape: {"Display label": "profile_id"}
                    return [(str(profile_id), str(label)) for label, profile_id in value.items()]

        for method_name in ("list", "list_profiles", "all", "profiles"):
            method = getattr(store, method_name, None)
            if callable(method):
                raw_items = method()
                break

        if raw_items is None:
            for attr_name in ("profiles", "_profiles", "items", "_items"):
                raw_items = getattr(store, attr_name, None)
                if raw_items is not None:
                    break

        if callable(raw_items):
            raw_items = raw_items()

        if isinstance(raw_items, Mapping):
            iterable = raw_items.items()
        else:
            iterable = enumerate(raw_items or [])

        out: List[Tuple[str, str]] = []

        for key, item in iterable:
            profile_id = (
                self._field(item, "profile_id", "id", "artifact_id", "recipe_profile_id")
                or key
            )

            name = self._field(item, "name", "title", "label")
            recipe_title = self._field(item, "recipe_title")
            recipe_id = self._field(item, "recipe_id")
            default_dataset_id = self._field(item, "default_dataset_id", "dataset_id")

            title_bits = [str(name or profile_id)]
            recipe_part = recipe_title or recipe_id
            if recipe_part:
                title_bits.append(f"recipe: {recipe_part}")
            if default_dataset_id:
                title_bits.append(f"dataset: {default_dataset_id}")

            title = " | ".join(title_bits)
            out.append((str(profile_id), title))

        return out

    def _dataset_options(self) -> Dict[str, str]:
        dataset_ids = self._dataset_ids()
        if not dataset_ids:
            return {"No datasets loaded": ""}
        options: Dict[str, str] = {}
        for dataset_id in dataset_ids:
            title = self._dataset_title(dataset_id)
            label = f"{title} ({dataset_id})" if title and title != dataset_id else dataset_id
            options[label] = dataset_id
        return options

    def _dataset_options_with_blank(self, blank_label: str = "None") -> Dict[str, str]:
        """Return dataset options with a selectable blank entry.

        Used by optional dataset selectors, such as the XY diagnostics
        validation dataset control.  This deliberately differs from
        _dataset_options(), which shows a non-selectable placeholder when no
        datasets exist.
        """
        options: Dict[str, str] = {str(blank_label or "None"): ""}
        for label, dataset_id in self._dataset_options().items():
            if not dataset_id:
                continue
            options[str(label)] = str(dataset_id)
        return options

    def _dataset_ids(self) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []
        values: List[str] = []
        for method_name in ("list_ids", "list_dataset_ids", "ids", "keys"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    values.extend(str(value) for value in method() if value not in (None, ""))
                except Exception:
                    pass
        for method_name in ("list_sources", "sources"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    raw = method()
                except Exception:
                    raw = None
                values.extend(self._ids_from_raw_collection(raw))
        for attr_name in ("sources", "_sources", "datasets", "_datasets"):
            raw = getattr(datasets, attr_name, None)
            values.extend(self._ids_from_raw_collection(raw))
        active = self._active_dataset_id()
        if active:
            values.append(active)
        return list(dict.fromkeys(value for value in values if value))

    def _ids_from_raw_collection(self, raw: Any) -> List[str]:
        if raw is None or callable(raw):
            return []
        if isinstance(raw, Mapping):
            return [str(key) for key in raw.keys() if key not in (None, "")]
        out: List[str] = []
        for item in raw or []:
            value = self._field(item, "dataset_id", "id", "name")
            if value not in (None, ""):
                out.append(str(value))
        return out

    def _dataset_title(self, dataset_id: str) -> str:
        source = self._dataset_source(dataset_id)
        for field_name in ("title", "name", "label", "dataset_id", "id"):
            value = self._field(source, field_name) if source is not None else None
            if value not in (None, ""):
                return str(value)
        return str(dataset_id)

    def _dataset_source(self, dataset_id: str) -> Any:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not dataset_id:
            return None
        method = getattr(datasets, "get_source", None)
        if callable(method):
            try:
                return method(dataset_id)
            except Exception:
                return None
        for attr_name in ("sources", "_sources", "datasets", "_datasets"):
            raw = getattr(datasets, attr_name, None)
            if isinstance(raw, Mapping) and dataset_id in raw:
                return raw[dataset_id]
        return None

    def _label_column_options(self, dataset_id: str) -> Dict[str, str]:
        if not dataset_id:
            return {"Select a dataset first": ""}
        columns = self._dataset_columns(dataset_id)
        if not columns:
            return {"No columns available": ""}
        options = {"Select label column": ""}
        options.update({column: column for column in columns})
        return options

    def _dataset_columns(self, dataset_id: str) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not dataset_id:
            return []
        for method_name in ("list_columns", "columns"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    values = method(dataset_id)
                    return [str(value) for value in values if value not in (None, "")]
                except Exception:
                    pass
        source = self._dataset_source(dataset_id)
        for attr_name in ("columns", "column_names"):
            value = getattr(source, attr_name, None)
            if callable(value):
                value = value()
            if value:
                return [str(column) for column in value if column not in (None, "")]
        schema = getattr(source, "schema", None)
        schema_columns = getattr(schema, "columns", None) if schema is not None else None
        if schema_columns:
            return [str(column) for column in schema_columns if column not in (None, "")]
        try:
            df = datasets.get_df(dataset_id)
            return [str(column) for column in df.columns]
        except Exception:
            return []

    def _infer_label_options(self, dataset_id: str, label_column: str, *, max_labels: int = 500) -> List[str]:
        if not dataset_id or not label_column:
            return []
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []
        try:
            try:
                df = datasets.get_df(dataset_id, columns=[label_column])
            except TypeError:
                df = datasets.get_df(dataset_id)
            if label_column not in getattr(df, "columns", []):
                return []
            series = df[label_column].dropna()
            values = series.unique().tolist() if hasattr(series, "unique") else list(series)
        except Exception:
            return []
        labels: List[str] = []
        for value in values:
            text = str(value).strip()
            if text and al_state.normalise_label(text) != al_state.UNSURE_LABEL and text not in labels:
                labels.append(text)
            if len(labels) >= max_labels:
                break
        return labels

    def _active_dataset_id(self) -> str:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return ""
        active = getattr(datasets, "active_id", None)
        try:
            return str(active() if callable(active) else active or "").strip()
        except Exception:
            return ""

    def _service(self, key: str) -> Any:
        services = getattr(self.context, "services", None)
        if services is None:
            return None
        for service_key in (key, key.split(".")[-1]):
            try:
                value = services.get(service_key)
                if value is not None:
                    return value
            except Exception:
                pass
        return None

    def _subscribe_to_platform_events(self) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        self._subscribe_topics(events, tuple(dict.fromkeys(self.DATASET_REFRESH_TOPICS + self.RECIPE_REFRESH_TOPICS)), self._handle_platform_event)
        self._subscribe_topics(events, self.REVIEW_REFRESH_TOPICS, self._handle_review_event)
        self._subscribe_topics(events, self.SESSION_REFRESH_TOPICS, self._handle_session_event)

    def _subscribe_topics(self, events: Any, topics: Sequence[str], handler: Any) -> None:
        for topic in topics:
            for method_name in ("subscribe", "on"):
                method = getattr(events, method_name, None)
                if not callable(method):
                    continue
                try:
                    subscription = method(topic, handler)
                except TypeError:
                    try:
                        subscription = method(handler, topic=topic)
                    except Exception:
                        continue
                except Exception:
                    continue
                self._subscriptions.append(subscription)
                break

    def _handle_platform_event(self, *args: Any, **kwargs: Any) -> None:
        if pn is None or self._view is None or not self._widgets:
            self._pending_refresh = True
            return
        def refresh() -> None:
            self.refresh_choices(status=False)
        self._next_tick(refresh)

    def _handle_review_event(self, *args: Any, **kwargs: Any) -> None:
        def update() -> None:
            row_id = self._row_id_from_event(*args, **kwargs)
            if row_id:
                self._set_review_row(row_id)
        self._next_tick(update)

    def _handle_session_event(self, *args: Any, **kwargs: Any) -> None:
        payload = self._payload_from_event(*args, **kwargs)

        def refresh() -> None:
            session_artifact_id = str(payload.get("session_artifact_id") or "").strip()
            if session_artifact_id:
                self._set_session_id(session_artifact_id)
                self._sync_latest_from_session(session_artifact_id)
            topic = str(payload.get("topic") or payload.get("event") or "")
            recipe_label = str(
                payload.get("recipe_profile_name")
                or payload.get("recipe_profile_id")
                or payload.get("recipe_id")
                or ""
            ).strip()
            if topic.endswith("training_started") or topic in {"al.round.training_started", "ml.training.started", "ml.recipe_run.started"}:
                self._set_status(f"Training round {payload.get('round', '')} started" + (f" with `{recipe_label}`." if recipe_label else "."))
                self._set_button_busy("train_btn", True)
            elif topic.endswith("training_finished") or topic in {"al.round.training_finished", "ml.training.finished", "ml.recipe_run.finished"}:
                self._capture_performance_event(payload)
                self._set_button_busy("train_btn", False)
                self._set_status(f"Training round {payload.get('round', '')} finished. Model and prediction artifacts refreshed.")
                self._select_tab("Query")
            elif topic.endswith("training_failed") or topic.endswith("failed") or topic in {"al.round.training_failed", "ml.training.failed", "ml.recipe_run.failed"}:
                self._set_button_busy("train_btn", False)
                self._set_status(f"Training failed: {payload.get('error', 'unknown error')}")
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            self._refresh_xy_plot(status=False)

        self._next_tick(refresh)

    def _payload_from_event(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        topic = kwargs.get("topic") or kwargs.get("event") or ""
        for item in args:
            if isinstance(item, str) and not topic:
                topic = item
            elif isinstance(item, Mapping):
                payload.update(dict(item))
        if isinstance(kwargs.get("payload"), Mapping):
            payload.update(dict(kwargs["payload"]))
        payload.update({key: value for key, value in kwargs.items() if key != "payload"})
        if topic and "topic" not in payload:
            payload["topic"] = str(topic)
        return payload

    def _capture_performance_event(self, payload: Mapping[str, Any]) -> None:
        rows = self._performance_rows_from_event(payload)
        if not rows:
            return
        existing = {
            (int(row.get("round") or 0), int(row.get("labelled_count") or 0), str(row.get("metric") or ""), float(row.get("value") or 0.0))
            for row in self._performance_event_rows
        }
        for row in rows:
            key = (int(row.get("round") or 0), int(row.get("labelled_count") or 0), str(row.get("metric") or ""), float(row.get("value") or 0.0))
            if key not in existing:
                self._performance_event_rows.append(dict(row))
                existing.add(key)

    def _performance_rows_from_event(self, payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
        metrics = al_state.extract_metrics(payload)
        if not metrics:
            artifacts = getattr(self.context, "artifacts", None)
            get = getattr(artifacts, "get", None)
            if callable(get):
                for artifact_id in al_state.artifact_ids(payload):
                    try:
                        artifact_payload = get(str(artifact_id))
                    except Exception:
                        continue
                    for key, value in al_state.extract_metrics(artifact_payload).items():
                        metrics.setdefault(key, value)
        if not metrics:
            return []

        session = self._session_payload(payload.get("session_artifact_id")) or self._session_payload()
        labelled_count = int(payload.get("labelled_count") or 0)
        if not labelled_count:
            labelled_count = len(al_state.labelled_training_items(session))
        round_index = int(payload.get("round") or session.get("round") or 0)
        rows: List[Dict[str, Any]] = []
        for metric, value in metrics.items():
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
                    "training_dataset_id": payload.get("training_dataset_id"),
                    "training_artifact_id": payload.get("training_artifact_id"),
                    "timestamp": payload.get("timestamp"),
                }
            )
        return rows

    def _next_tick(self, callback: Any) -> None:
        try:
            doc = self._doc or (getattr(pn.state, "curdoc", None) if pn is not None else None)
            if doc is not None and hasattr(doc, "add_next_tick_callback"):
                doc.add_next_tick_callback(callback)
            else:
                callback()
        except Exception:
            callback()

    def _row_id_from_event(self, *args: Any, **kwargs: Any) -> str:
        payload = kwargs.get("payload") or kwargs
        if args:
            for item in reversed(args):
                if isinstance(item, Mapping):
                    payload = item
                    break
        if isinstance(payload, Mapping):
            focus = payload.get("focus") or payload.get("selection") or payload
            if isinstance(focus, Mapping):
                return str(focus.get("row_id") or focus.get("id") or "").strip()
            return str(payload.get("row_id") or payload.get("id") or "").strip()
        return ""

    def _review_label_options(self, labels: Sequence[str]) -> Dict[str, str]:
        options = {str(label): str(label) for label in labels if label not in (None, "")}
        options["Unsure"] = "Unsure"
        return options

    def _first_review_label(self, labels: Sequence[str]) -> str:
        for label in labels:
            if label not in (None, ""):
                return str(label)
        return "Unsure"

    def _valid_or_default(self, current: Any, options: Mapping[str, str], *, allow_blank: bool = False) -> str:
        current = str(current or "")
        values = [str(value) for value in options.values()]
        if current in values:
            return current
        if allow_blank and "" in values:
            return ""
        for value in values:
            if value:
                return value
        return ""

    def _set_session_id(self, value: Any) -> None:
        self.session_artifact_id = str(value or "")
        widget = self._widgets.get("session_id")
        if widget is not None:
            widget.value = self.session_artifact_id
        self._refresh_session_summary(status=False)
        self._refresh_performance(status=False)

    def _set_artifact_widget(self, key: str, value: Any) -> None:
        text = str(value or "")
        if key == "model_id":
            self.model_artifact_id = text
        elif key == "predictions_id":
            self.predictions_artifact_id = text
        widget = self._widgets.get(key)
        if widget is not None:
            widget.value = text

    def _set_review_row(self, row_id: Any) -> None:
        text = str(row_id or "").strip()
        widget = self._widgets.get("row_id")
        if widget is not None and text:
            widget.value = text
        if text:
            self._sync_review_label_to_source(text)

    def _on_review_row_changed(self, row_id: str) -> None:
        self._sync_review_label_to_source(row_id)

    def _sync_review_label_to_source(self, row_id: Any) -> None:
        label = self._source_label_for_row(str(row_id or "").strip())
        if not label:
            return
        label_widget = self._widgets.get("label")
        if label_widget is None:
            return
        options = dict(getattr(label_widget, "options", {}) or {})
        allowed = set(str(value) for value in options.values())
        if str(label) not in allowed:
            return
        label_widget.value = str(label)

    def _source_label_for_row(self, row_id: str) -> str:
        row_id = str(row_id or "").strip()
        if not row_id:
            return ""
        dataset_id = self._session_dataset_id() or str(self._widget_value("dataset_id", "") or "")
        label_column = self._session_label_column()
        if not dataset_id or not label_column:
            return ""
        try:
            values = actions.dataset_label_values_by_row_id(
                self.context,
                dataset_id=dataset_id,
                row_ids=[row_id],
                label_column=label_column,
            )
        except Exception:
            return ""
        label = al_state.normalise_label(values.get(row_id))
        return "" if label == al_state.UNSURE_LABEL else label

    def _session_payload(self, session_artifact_id: Any = None) -> Dict[str, Any]:
        artifact_id = str(session_artifact_id or self._widget_value("session_id", "") or self.session_artifact_id or "").strip()
        if not artifact_id:
            return {}
        try:
            return al_state.coerce_session(self.context.artifacts.get(artifact_id))
        except Exception:
            return {}

    def _session_dataset_id(self) -> str:
        session = self._session_payload()
        return str(session.get("pool_dataset_id") or session.get("dataset_id") or "").strip()

    def _session_label_column(self) -> str:
        session = self._session_payload()
        return str(session.get("target_column") or self._widget_value("label_column", "") or self.label_column or "").strip()

    def _sync_latest_from_session(self, session_artifact_id: Any = None) -> Dict[str, str]:
        session = self._session_payload(session_artifact_id)
        latest = dict(session.get("latest") or {})
        model_artifact_id = str(latest.get("model_artifact_id") or "")
        predictions_artifact_id = str(latest.get("predictions_artifact_id") or "")
        if model_artifact_id:
            self._set_artifact_widget("model_id", model_artifact_id)
        if predictions_artifact_id:
            self._set_artifact_widget("predictions_id", predictions_artifact_id)
        return {"model_artifact_id": model_artifact_id, "predictions_artifact_id": predictions_artifact_id}

    def _select_tab(self, title: str) -> None:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        index = self._tab_index.get(str(title))
        if index is not None:
            tabs.active = index

    def _set_status(self, text: str) -> None:
        self.status = str(text)
        pane = self._widgets.get("status")
        if pane is not None:
            pane.object = self._status_text()

    def _refresh_session_summary(self, *, status: bool = True) -> None:
        pane = self._widgets.get("session_summary")
        if pane is not None:
            pane.object = self._session_summary_text()
        if status:
            self._set_status("Session summary refreshed.")

    def _session_summary_text(self) -> str:
        session = self._session_payload()
        if not session:
            return "**Session:** none selected  \n**Round:** —  \n**Labelled since last train:** —  \n**Labelled total:** —"
        counts = al_state.counts(session)
        latest = dict(session.get("latest") or {})
        model_id = latest.get("model_artifact_id") or "—"
        predictions_id = latest.get("predictions_artifact_id") or "—"
        return (
            f"**Session:** `{session.get('session_id')}`  \n"
            f"**Round:** {session.get('round', 0)}  \n"
            f"**Labelled since last train:** {counts.get('labelled_since_last_train', 0)}  \n"
            f"**Labelled total:** {counts.get('labelled_or_verified', 0)}  \n"
            f"**Reviewed total:** {counts.get('total_reviewed', 0)}  \n"
            f"**Unsure:** {counts.get('unsure', 0)}  \n"
            f"**Queued:** {counts.get('queued', 0)}  \n"
            f"**Model:** `{model_id}`  \n"
            f"**Predictions:** `{predictions_id}`"
        )


    def _prepare_performance_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Filter noisy core.ml metric payloads down to useful AL metrics.

        core.ml artifacts contain many scalar values (epochs, records, query
        scores, referenced artifact copies).  AL performance should present one
        concise set of validation/test-style metrics per training round.
        """

        cleaned: List[Dict[str, Any]] = []
        for row in rows:
            metric = self._canonical_performance_metric(str(row.get("metric") or ""))
            if not self._is_useful_performance_metric(metric):
                continue
            try:
                value = float(row.get("value"))
            except Exception:
                continue
            item = dict(row)
            item["metric"] = metric
            item["value"] = value
            cleaned.append(item)
        return self._dedupe_performance_rows(cleaned)

    def _canonical_performance_metric(self, metric: str) -> str:
        text = str(metric or "").strip()
        if not text:
            return ""
        parts = [part for part in text.split(".") if part]
        # Drop wrappers introduced by event/artifact summaries.
        while parts and parts[0] in {
            "ml_result",
            "result",
            "summary",
            "metrics",
            "metric",
            "evaluation",
            "evaluation_metrics",
            "eval_metrics",
            "validation_metrics",
            "val_metrics",
            "test_metrics",
            "history",
            "histories",
            "epochs",
            "events",
            "referenced_artifacts",
            "artifacts",
            "outputs",
        }:
            parts.pop(0)
            # referenced_artifacts.<artifact_id>.<metric>
            if parts and len(parts[0]) > 16 and any(ch.isdigit() for ch in parts[0]):
                parts.pop(0)
        # If the metric still has a nested prefix, use the final metric-looking
        # component.  This normalises events.val_accuracy and
        # referenced_artifacts.<id>.epochs.val_accuracy to val_accuracy.
        if len(parts) > 1:
            for part in reversed(parts):
                lower = part.lower()
                if any(token in lower for token in ("accuracy", "acc", "auc", "f1", "precision", "recall", "loss", "mae", "mse", "rmse", "r2", "score")):
                    return lower
        return (parts[-1] if parts else text).lower()

    def _is_useful_performance_metric(self, metric: str) -> bool:
        name = str(metric or "").lower().strip()
        if not name:
            return False
        if any(token in name for token in ("epoch", "step", "batch", "timestamp", "duration", "time", "record", "informativeness", "active_learning", "query", "pred_", "prob_")):
            return False
        if name in {"score", "best_score", "best_score_so_far", "best", "metric", "count", "n", "k"}:
            return False
        return any(
            token in name
            for token in (
                "accuracy",
                "acc",
                "balanced_accuracy",
                "auc",
                "roc_auc",
                "f1",
                "precision",
                "recall",
                "loss",
                "log_loss",
                "mae",
                "mse",
                "rmse",
                "r2",
            )
        )

    def _performance_metric_sort_key(self, metric: str) -> Tuple[int, str]:
        name = str(metric or "").lower()
        if name.startswith("test_"):
            priority = 0
        elif name.startswith("val_") or name.startswith("validation_"):
            priority = 1
        elif name in {"accuracy", "f1", "f1_macro", "loss", "auc", "roc_auc"}:
            priority = 2
        elif name.startswith("train_") or name.startswith("training_"):
            priority = 4
        else:
            priority = 3
        return (priority, name)

    def _refresh_performance(self, *, status: bool = True, keep_metric: bool = False) -> None:
        metric_widget = self._widgets.get("performance_metric")
        plot_pane = self._widgets.get("performance_plot")
        if metric_widget is None or plot_pane is None:
            return
        rows = analytics.performance_rows_with_artifacts(self.context, self._session_payload())
        rows.extend(self._performance_event_rows)
        rows = self._prepare_performance_rows(rows)
        metrics = sorted({str(row.get("metric")) for row in rows if row.get("metric") not in (None, "")}, key=self._performance_metric_sort_key)
        old_metric = str(metric_widget.value or "")
        options = {metric: metric for metric in metrics} if metrics else {"No metrics yet": ""}
        metric_widget.options = options
        metric = old_metric if keep_metric and old_metric in set(options.values()) else self._valid_or_default(old_metric, options, allow_blank=True)
        metric_widget.value = metric
        metric_rows = [row for row in rows if str(row.get("metric")) == str(metric)] if metric else []
        plot_pane.object = self._performance_plot_object(metric_rows, metric)
        if status:
            self._set_status("AL performance plot refreshed.")

    def _dedupe_performance_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for row in rows:
            try:
                key = (
                    int(row.get("round") or 0),
                    int(row.get("labelled_count") or 0),
                    str(row.get("metric") or ""),
                )
                value = float(row.get("value") or 0.0)
            except Exception:
                continue
            item = dict(row)
            item["value"] = value
            # Prefer the last copy seen; event payloads are appended after session
            # history and usually carry the most current artifact ids.
            by_key[key] = item
        return [by_key[key] for key in sorted(by_key)]

    def _performance_plot_object(self, rows: Sequence[Mapping[str, Any]], metric: str) -> Any:
        if not rows or not metric:
            return "No AL performance points yet. Complete a training round with an evaluation metric to populate this plot."
        ordered = sorted(rows, key=lambda row: (int(row.get("round") or 0), int(row.get("labelled_count") or 0)))
        try:
            from bokeh.models import ColumnDataSource, HoverTool
            from bokeh.plotting import figure

            source = ColumnDataSource(
                data={
                    "round": [int(row.get("round") or 0) for row in ordered],
                    "labelled_count": [int(row.get("labelled_count") or 0) for row in ordered],
                    "value": [float(row.get("value") or 0.0) for row in ordered],
                    "metric": [str(row.get("metric") or "") for row in ordered],
                }
            )
            plot = figure(
                height=320,
                sizing_mode="stretch_width",
                x_axis_label="Number of labelled training points",
                y_axis_label=metric,
                title=f"AL performance: {metric}",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            plot.line("labelled_count", "value", source=source, line_width=2)
            plot.circle("labelled_count", "value", source=source, size=8)
            plot.add_tools(HoverTool(tooltips=[("round", "@round"), ("labelled", "@labelled_count"), (metric, "@value")]))
            return plot
        except Exception:
            lines = [f"| Round | Labelled points | {metric} |", "|---:|---:|---:|"]
            for row in ordered:
                lines.append(f"| {row.get('round')} | {row.get('labelled_count')} | {row.get('value')} |")
            return "\n".join(lines)



    def _on_recipe_profile_changed(self, profile_id: str) -> None:
        self.recipe_profile_id = str(profile_id or "").strip()
        if self.recipe_profile_id:
            self._set_status("Recipe profile selected.")
        else:
            self._set_status("Select a core.ml recipe profile before training.")
        self._sync_train_controls()

    def _on_xy_dataset_choice_changed(self) -> None:
        self._refresh_xy_columns(status=False)
        self._refresh_xy_plot(status=False)

    def _refresh_xy_dataset_controls(self, *, status: bool = True) -> None:
        widget = self._widgets.get("xy_validation_dataset_id") if self._widgets else None
        if widget is None:
            return
        options = self._dataset_options_with_blank("No validation dataset")
        old_value = str(getattr(widget, "value", "") or "").strip()
        session = self._session_payload()
        preferred = old_value or str(session.get("validation_dataset_id") or "").strip()
        if not preferred:
            preferred = self._validation_dataset_id_from_profile()
        widget.options = options
        widget.value = self._valid_or_default(preferred, options, allow_blank=True)
        if status:
            self._set_status("XY dataset controls refreshed.")

    def _xy_scope_value(self) -> str:
        value = str(self._widget_value("xy_scope", "combined") or "combined").strip().lower()
        return value if value in {"combined", "pool", "validation"} else "combined"

    def _xy_validation_dataset_id(self, session: Optional[Mapping[str, Any]] = None) -> str:
        widget_value = str(self._widget_value("xy_validation_dataset_id", "") or "").strip()
        if widget_value:
            return widget_value
        session_payload = dict(session or self._session_payload() or {})
        value = str(session_payload.get("validation_dataset_id") or "").strip()
        if value:
            return value
        contract = session_payload.get("contract") or {}
        if isinstance(contract, Mapping):
            datasets = contract.get("datasets") or contract.get("data") or {}
            if isinstance(datasets, Mapping):
                validation = datasets.get("validation") or datasets.get("val") or {}
                if isinstance(validation, Mapping):
                    value = str(validation.get("dataset_id") or validation.get("id") or "").strip()
                    if value:
                        return value
        return self._validation_dataset_id_from_profile()

    def _validation_dataset_id_from_profile(self) -> str:
        profile_id = str(self._widget_value("recipe_profile_id", self.recipe_profile_id) or self.recipe_profile_id or "").strip()
        if not profile_id:
            return ""
        store = self._service("core.ml.recipe_profile_store")
        if store is None:
            return ""
        try:
            profile = store.get(profile_id)
        except Exception:
            return ""
        if not isinstance(profile, Mapping):
            return ""
        protocol = profile.get("protocol_params") or {}
        binding = profile.get("binding_params") or {}
        for source in (protocol, binding, profile):
            if not isinstance(source, Mapping):
                continue
            for key in ("protocol_validation_dataset_id", "validation_dataset_id", "val_dataset_id"):
                value = str(source.get(key) or "").strip()
                if value:
                    return value
        return ""

    def _prediction_from_dataset_row(self, row: Any) -> Dict[str, Any]:
        getter = getattr(row, "get", None)
        if not callable(getter):
            return {}
        prediction: Dict[str, Any] = {}
        for source_key, target_key in (
            ("pred_label", "predicted_label"),
            ("predicted_label", "predicted_label"),
            ("pred_class", "predicted_label"),
            ("predicted_class", "predicted_label"),
            ("pred_confidence", "confidence"),
            ("confidence", "confidence"),
            ("pred_entropy", "entropy"),
            ("pred_least_confidence", "least_confidence"),
            ("pred_margin_uncertainty", "margin_uncertainty"),
            ("pred_true_label", "true_label"),
            ("pred_correct", "correct"),
        ):
            try:
                value = getter(source_key)
            except Exception:
                continue
            if value not in (None, ""):
                prediction[target_key] = value
        probabilities: Dict[str, float] = {}
        try:
            keys = list(getattr(row, "index", []))
        except Exception:
            keys = []
        for key in keys:
            name = str(key)
            if not name.startswith("pred_prob_"):
                continue
            label = name[len("pred_prob_"):]
            try:
                value = getter(key)
                probabilities[label] = float(value)
            except Exception:
                continue
        if probabilities:
            prediction["probabilities"] = probabilities
            prediction["class_labels"] = list(probabilities.keys())
        return prediction

    def _refresh_xy_columns(self, *, status: bool = True) -> None:
        x_widget = self._widgets.get("xy_x")
        y_widget = self._widgets.get("xy_y")
        if x_widget is None or y_widget is None:
            return
        options = self._xy_column_options_for_current_scope()
        old_x = str(x_widget.value or "")
        old_y = str(y_widget.value or "")
        x_widget.options = options
        y_widget.options = options
        x_widget.value = self._valid_or_default(old_x, options, allow_blank=True)
        y_widget.value = self._valid_or_default(old_y, options, allow_blank=True)
        if not y_widget.value:
            values = [str(value) for value in options.values() if str(value)]
            if len(values) >= 2 and x_widget.value != values[1]:
                y_widget.value = values[1]
        if status:
            self._set_status("XY column list refreshed.")

    def _xy_column_options_for_current_scope(self) -> Dict[str, str]:
        session = self._session_payload()
        scope = self._xy_scope_value()
        dataset_ids: List[str] = []
        pool_dataset_id = self._session_dataset_id() or str(self._widget_value("dataset_id", "") or "")
        validation_dataset_id = self._xy_validation_dataset_id(session)
        if scope in {"combined", "pool"} and pool_dataset_id:
            dataset_ids.append(pool_dataset_id)
        if scope in {"combined", "validation"} and validation_dataset_id:
            dataset_ids.append(validation_dataset_id)
        columns: List[str] = []
        for dataset_id in dict.fromkeys(dataset_ids):
            for column in self._plottable_dataset_columns(dataset_id):
                if column not in columns:
                    columns.append(column)
        if not columns:
            return {"No plottable columns available": ""}
        options = {"Select column": ""}
        options.update({column: column for column in columns})
        return options

    def _xy_column_options(self, dataset_id: str) -> Dict[str, str]:
        columns = self._plottable_dataset_columns(dataset_id)
        if not columns:
            return {"No plottable columns available": ""}
        options = {"Select column": ""}
        options.update({column: column for column in columns})
        return options

    def _plottable_dataset_columns(self, dataset_id: str) -> List[str]:
        """Return columns that can be coerced to numeric values for XY plotting."""

        df = self._dataset_df(dataset_id)
        if df is None:
            return self._dataset_columns(dataset_id)
        out: List[str] = []
        try:
            import pandas as pd
            for column in getattr(df, "columns", []):
                try:
                    values = pd.to_numeric(df[column], errors="coerce")
                    if values.notna().sum() >= 2:
                        out.append(str(column))
                except Exception:
                    continue
        except Exception:
            return self._dataset_columns(dataset_id)
        return out

    def _numeric_dataset_columns(self, dataset_id: str) -> List[str]:
        # Compatibility wrapper retained for older callers.
        return self._plottable_dataset_columns(dataset_id)

    def _dataset_df(self, dataset_id: str) -> Any:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not dataset_id:
            return None
        try:
            return datasets.get_df(dataset_id)
        except Exception:
            return None

    def _refresh_xy_plot(self, *, status: bool = True) -> None:
        pane = self._widgets.get("xy_plot")
        if pane is None:
            return
        pane.object = self._xy_plot_object()
        if status:
            self._set_status("XY diagnostics plot refreshed.")

    def _xy_plot_object(self) -> Any:
        x_col = str(self._widget_value("xy_x", "") or "").strip()
        y_col = str(self._widget_value("xy_y", "") or "").strip()
        dataset_id = self._session_dataset_id() or str(self._widget_value("dataset_id", "") or "").strip()
        if not dataset_id:
            return self._empty_matplotlib_message("Load or select a dataset before plotting XY diagnostics.")
        if not x_col or not y_col:
            return self._empty_matplotlib_message("Choose X and Y columns to plot XY diagnostics.")

        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except Exception:
            return "Matplotlib/pandas is not available for XY diagnostics."

        session = self._session_payload()
        scope = self._xy_scope_value()
        datasets: List[Tuple[str, str, str]] = []
        if scope in {"combined", "pool"}:
            datasets.append(("pool", dataset_id, "o"))
        validation_dataset_id = self._xy_validation_dataset_id(session)
        if scope in {"combined", "validation"} and validation_dataset_id and validation_dataset_id != dataset_id:
            datasets.append(("validation", validation_dataset_id, "^"))
        elif scope == "validation" and validation_dataset_id == dataset_id:
            datasets = [("validation", validation_dataset_id, "^")]
        if not datasets:
            return self._empty_matplotlib_message("Choose train/pool or validation data to plot.")

        frames: List[Any] = []
        for role, role_dataset_id, marker in datasets:
            df = self._dataset_df(role_dataset_id)
            if df is None or x_col not in getattr(df, "columns", []) or y_col not in getattr(df, "columns", []):
                continue
            work = df.copy()
            try:
                work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
                work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
                work = work.dropna(subset=[x_col, y_col])
            except Exception:
                continue
            if work.empty:
                continue
            id_column = ""
            try:
                id_column = acquisition.resolve_record_id_column(self.context, role_dataset_id)
            except Exception:
                id_column = ""
            if id_column and id_column in work.columns:
                row_ids = work[id_column].map(lambda value: str(value))
            else:
                row_ids = work.index.map(lambda value: str(value))
            work = work.assign(_al_row_id=list(row_ids), _al_dataset_role=role, _al_marker=marker)
            frames.append(work)

        if not frames:
            return self._empty_matplotlib_message("No selected dataset has numeric values for the chosen X/Y columns.")
        work_all = pd.concat(frames, ignore_index=False)

        training_ids = {str(row_id) for row_id in (session.get("training_row_ids") or []) if str(row_id)}
        if not training_ids:
            for row in al_state.labelled_training_items(session):
                if row.get("row_id") not in (None, ""):
                    training_ids.add(str(row.get("row_id")))
        label_map = {str(row.get("row_id")): al_state.display_label(row.get("label")) for row in al_state.label_rows(session)}
        label_column = self._session_label_column()
        if label_column:
            for _, row in work_all.iterrows():
                row_id = str(row.get("_al_row_id") or "")
                if row_id and row_id not in label_map and label_column in work_all.columns:
                    value = row.get(label_column)
                    if value not in (None, ""):
                        label_map[row_id] = al_state.display_label(value)

        batch_records = self._latest_batch_records(session)
        query_ids = set(batch_records.keys())
        prediction_map = self._prediction_records_by_row_id(al_state.latest_reference(session, "predictions_artifact_id") or self._widget_value("predictions_id", ""))
        for _, row in work_all.iterrows():
            row_id = str(row.get("_al_row_id") or "")
            if row_id and row_id not in prediction_map:
                prediction = self._prediction_from_dataset_row(row)
                if prediction:
                    prediction_map[row_id] = prediction

        max_points = int(self._widget_value("xy_max_points", 5000) or 5000)
        max_points = max(100, max_points)
        special_ids = training_ids | query_ids
        special = work_all[work_all["_al_row_id"].isin(special_ids)]
        base = work_all
        if len(base) > max_points:
            non_special = base[~base["_al_row_id"].isin(special_ids)]
            keep_n = max(0, max_points - len(special))
            if keep_n > 0 and len(non_special) > keep_n:
                non_special = non_special.sample(n=keep_n, random_state=13)
            base = pd.concat([non_special, special], ignore_index=False).drop_duplicates(subset=["_al_dataset_role", "_al_row_id"])

        colour_by = str(self._widget_value("xy_colour", "prediction_correctness") or "prediction_correctness")
        fig, ax = plt.subplots(figsize=(5.4, 3.2))

        if colour_by == "prediction_correctness":
            self._scatter_correctness_groups(ax, base, x_col, y_col, label_map, prediction_map)
        else:
            for role, marker in (("pool", "o"), ("validation", "^")):
                sub = base[base["_al_dataset_role"] == role]
                if not sub.empty:
                    ax.scatter(sub[x_col], sub[y_col], s=3, alpha=0.08, marker=marker, label=f"{role} rows")

        trained = work_all[work_all["_al_row_id"].isin(training_ids)]
        if not trained.empty:
            if colour_by == "prediction_correctness":
                self._scatter_correctness_groups(ax, trained, x_col, y_col, label_map, prediction_map, prefix="trained ", marker="x", size=12, alpha=0.7)
            else:
                ax.scatter(trained[x_col], trained[y_col], s=10, alpha=0.65, marker="x", label="trained/labelled")

        queried = work_all[work_all["_al_row_id"].isin(query_ids)]
        if not queried.empty:
            values: List[Any] = []
            for row_id in queried["_al_row_id"]:
                info = batch_records.get(str(row_id), {})
                if colour_by == "last_query_rank":
                    values.append(info.get("rank"))
                elif colour_by == "label":
                    values.append(label_map.get(str(row_id), "unlabelled"))
                elif colour_by == "training_status":
                    values.append("trained" if str(row_id) in training_ids else "queried")
                elif colour_by == "prediction_correctness":
                    values.append(self._prediction_status(str(row_id), label_map.get(str(row_id)), prediction_map.get(str(row_id))))
                else:
                    values.append(info.get("score"))
            self._scatter_query_overlay(ax, queried, x_col, y_col, values, colour_by)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        strategy = str((session.get("last_batch") or {}).get("strategy_id") or "")
        title = "AL XY diagnostics"
        if scope == "pool":
            title += " — train/pool"
        elif scope == "validation":
            title += " — validation"
        if strategy:
            title += f" — latest QS: {strategy}"
        ax.set_title(title)
        ax.legend(loc="best", fontsize="small", markerscale=1.4)
        fig.tight_layout()
        return fig

    def _scatter_correctness_groups(
        self,
        ax: Any,
        df: Any,
        x_col: str,
        y_col: str,
        label_map: Mapping[str, str],
        prediction_map: Mapping[str, Mapping[str, Any]],
        *,
        prefix: str = "",
        marker: str | None = None,
        size: int = 3,
        alpha: float = 0.08,
    ) -> None:
        if df.empty:
            return
        groups = {"correct": [], "second": [], "incorrect": [], "unknown": []}
        for idx, row in df.iterrows():
            row_id = str(row.get("_al_row_id") or "")
            status = self._prediction_status(row_id, label_map.get(row_id), prediction_map.get(row_id))
            groups.setdefault(status, []).append(idx)
        style = {
            "correct": ("green", "correct"),
            "incorrect": ("red", "incorrect"),
            "second": ("gold", "true label is 2nd probability"),
            "unknown": ("0.55", "no correctness info"),
        }
        for status, indexes in groups.items():
            if not indexes:
                continue
            sub = df.loc[indexes]
            role_marker = marker or ("^" if str(sub.iloc[0].get("_al_dataset_role") or "") == "validation" else "o")
            colour, label = style.get(status, ("0.55", status))
            ax.scatter(sub[x_col], sub[y_col], s=size, alpha=alpha, marker=role_marker, c=colour, label=f"{prefix}{label}")

    def _prediction_status(self, row_id: str, true_label: Any, prediction: Mapping[str, Any] | None) -> str:
        label = str(true_label or "").strip()
        if not label or prediction is None:
            return "unknown"
        pred_label = self._prediction_label(prediction)
        if not pred_label:
            return "unknown"
        if str(pred_label) == label:
            return "correct"
        second_label, num_classes = self._second_probability_label(prediction)
        if num_classes > 3 and second_label and str(second_label) == label:
            return "second"
        return "incorrect"

    def _prediction_label(self, prediction: Mapping[str, Any]) -> str:
        for key in ("predicted_label", "prediction", "label_pred", "predicted_class", "class", "y_pred", "label"):
            value = prediction.get(key)
            if value not in (None, ""):
                return str(value)
        probs = self._probability_items(prediction)
        if probs:
            return str(max(probs, key=lambda item: item[1])[0])
        return ""

    def _second_probability_label(self, prediction: Mapping[str, Any]) -> Tuple[str, int]:
        probs = sorted(self._probability_items(prediction), key=lambda item: item[1], reverse=True)
        if len(probs) < 2:
            return "", len(probs)
        return str(probs[1][0]), len(probs)

    def _probability_items(self, prediction: Mapping[str, Any]) -> List[Tuple[str, float]]:
        raw = None
        for key in ("probabilities", "class_probabilities", "probs", "proba", "scores", "class_scores"):
            if key in prediction:
                raw = prediction.get(key)
                break
        labels = prediction.get("class_labels") or prediction.get("labels") or prediction.get("classes")
        out: List[Tuple[str, float]] = []
        if isinstance(raw, Mapping):
            for label, value in raw.items():
                try:
                    out.append((str(label), float(value)))
                except Exception:
                    pass
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            label_values = list(labels or [])
            for index, value in enumerate(raw):
                try:
                    label = str(label_values[index]) if index < len(label_values) else str(index)
                    out.append((label, float(value)))
                except Exception:
                    pass
        return out

    def _prediction_records_by_row_id(self, predictions_artifact_id: Any) -> Dict[str, Dict[str, Any]]:
        artifact_id = str(predictions_artifact_id or "").strip()
        if not artifact_id:
            return {}
        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            return {}
        class_labels = []
        if isinstance(payload, Mapping):
            class_labels = list(payload.get("class_labels") or payload.get("labels") or payload.get("classes") or [])
        records = self._prediction_record_list(payload)
        out: Dict[str, Dict[str, Any]] = {}
        for record in records:
            row_id = str(record.get("row_id") or record.get("record_id") or record.get("id") or record.get("source_id") or "").strip()
            if not row_id:
                continue
            item = dict(record)
            if class_labels and not item.get("class_labels"):
                item["class_labels"] = class_labels
            out[row_id] = item
        return out

    def _prediction_record_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, Mapping):
            for key in ("records", "rows", "predictions", "items", "data"):
                value = payload.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    return [dict(item) for item in value if isinstance(item, Mapping)]
            for value in payload.values():
                records = self._prediction_record_list(value)
                if records:
                    return records
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        return []

    def _scatter_query_overlay(self, ax: Any, df: Any, x_col: str, y_col: str, values: Sequence[Any], colour_by: str) -> None:
        if colour_by in {"last_query_score", "last_query_rank"}:
            numeric = []
            for value in values:
                try:
                    numeric.append(float(value))
                except Exception:
                    numeric.append(float("nan"))
            scatter = ax.scatter(df[x_col], df[y_col], c=numeric, s=8, alpha=0.65, label="latest query")
            try:
                ax.figure.colorbar(scatter, ax=ax, label="query score" if colour_by == "last_query_score" else "query rank")
            except Exception:
                pass
            return
        if colour_by == "prediction_correctness":
            colours = {"correct": "green", "incorrect": "red", "second": "gold", "unknown": "0.55"}
            labels = {"correct": "query correct", "incorrect": "query incorrect", "second": "query true label is 2nd probability", "unknown": "query unknown"}
            categories = [str(value or "unknown") for value in values]
            for category in list(dict.fromkeys(categories)):
                mask = [value == category for value in categories]
                sub = df[mask]
                if not sub.empty:
                    ax.scatter(sub[x_col], sub[y_col], s=9, alpha=0.7, marker="D", c=colours.get(category, "0.55"), label=labels.get(category, category))
            return
        categories = [str(value or "") for value in values]
        unique = list(dict.fromkeys(categories))
        for category in unique:
            mask = [value == category for value in categories]
            sub = df[mask]
            if not sub.empty:
                ax.scatter(sub[x_col], sub[y_col], s=8, alpha=0.65, label=f"latest query: {category}")

    def _latest_batch_records(self, session: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        last_batch = dict((session or {}).get("last_batch") or {})
        batch_artifact_id = str(last_batch.get("batch_artifact_id") or "").strip()
        records: List[Mapping[str, Any]] = []
        if batch_artifact_id:
            try:
                payload = self.context.artifacts.get(batch_artifact_id)
                if isinstance(payload, Mapping):
                    records = [dict(row) for row in (payload.get("records") or []) if isinstance(row, Mapping)]
            except Exception:
                records = []
        if not records:
            row_ids = [str(row_id) for row_id in (last_batch.get("row_ids") or []) if str(row_id)]
            records = [{"row_id": row_id, "rank": index + 1} for index, row_id in enumerate(row_ids)]
        out: Dict[str, Dict[str, Any]] = {}
        for index, record in enumerate(records):
            row_id = str(record.get("row_id") or record.get("id") or "").strip()
            if not row_id:
                continue
            item = dict(record)
            item.setdefault("rank", index + 1)
            score = item.get("score")
            if score is None:
                score = item.get("acquisition_score") or item.get("uncertainty")
            if score is not None:
                item["score"] = score
            out[row_id] = item
        return out

    def _empty_matplotlib_message(self, message: str) -> Any:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.text(0.5, 0.5, str(message), ha="center", va="center", wrap=True)
            ax.set_axis_off()
            fig.tight_layout()
            return fig
        except Exception:
            return str(message)

    def _set_button_busy(self, key: str, busy: bool) -> None:
        widget = self._widgets.get(key)
        if widget is None:
            return
        try:
            widget.loading = bool(busy)
        except Exception:
            pass
        if key == "train_btn":
            try:
                widget.disabled = bool(busy) or not bool(self._widget_value("recipe_profile_id", ""))
            except Exception:
                pass

    def _status_text(self) -> str:
        return f"**Status:** {self.status}"

    def _widget_value(self, key: str, default: Any = None) -> Any:
        widget = self._widgets.get(key)
        return getattr(widget, "value", default) if widget is not None else default

    def _field(self, item: Any, *names: str) -> Any:
        if item is None:
            return None
        if isinstance(item, Mapping):
            for name in names:
                if name in item:
                    return item.get(name)
            return None
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                return value
        return None
