from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import html

from . import actions
from . import acquisition
from . import analytics
from . import state as al_state

try:  # pragma: no cover - UI import is environment-specific.
    import panel as pn
except Exception:  # pragma: no cover
    pn = None


def _new_action_request(**kwargs: Any) -> Any:
    from astronomicAL.platform.plugins.specs import ActionRequest

    return ActionRequest(**kwargs)


MAX_AUTO_LABEL_SCAN_ROWS = 50000
XY_DEFAULT_MAX_POINTS = 5000
XY_COLUMN_SAMPLE_ROWS = 2000
BUTTON_HEIGHT = 34

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
        "dataset.mapping.updated",
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
        "al.strategy_scores.calculated",
        "al.round.training_started",
        "al.round.training_finished",
        "al.round.training_paused",
        "al.round.training_cancelled",
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
        self.strategy_scores_artifact_id = ""
        self.dataset_id = ""
        self.label_column = ""
        self.task_type = al_state.TASK_CLASSIFICATION
        self.label_profile: Dict[str, Any] = {}
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
        self._job_handles: List[Any] = []
        self._disposed = False
        self._performance_event_rows: List[Dict[str, Any]] = []
        self._pending_refresh = False
        self._doc = None
        self._dataset_columns_cache: Dict[str, List[str]] = {}
        self._plottable_columns_cache: Dict[str, List[str]] = {}
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

    def _session_protocol_controls_view(self) -> Any:
        """Optional session-protocol controls supplied by the job-backed panel."""
        return None

    def _training_controls_view(self, train_button: Any) -> Any:
        """Optional pause/resume/cancel controls supplied by the job-backed panel."""
        return train_button

    def get_state(self) -> Dict[str, Any]:
        return {
            "session_artifact_id": self.session_artifact_id,
            "predictions_artifact_id": self.predictions_artifact_id,
            "model_artifact_id": self.model_artifact_id,
            "strategy_scores_artifact_id": self.strategy_scores_artifact_id,
            "dataset_id": self._widget_value("dataset_id", self.dataset_id),
            "label_column": self._widget_value("label_column", self.label_column),
            "task_type": self.task_type,
            "label_profile": dict(self.label_profile or {}),
            "selected_labels": list(self._widget_value("labels", self.selected_labels) or []),
            "recipe_profile_id": self._widget_value("recipe_profile_id", self.recipe_profile_id),
            "recipe_id": self.recipe_id,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.session_artifact_id = str(state.get("session_artifact_id") or "")
        self.predictions_artifact_id = str(state.get("predictions_artifact_id") or "")
        self.model_artifact_id = str(state.get("model_artifact_id") or "")
        self.strategy_scores_artifact_id = str(state.get("strategy_scores_artifact_id") or "")
        self.dataset_id = str(state.get("dataset_id") or "")
        self.label_column = str(state.get("label_column") or "")
        self.task_type = al_state.parse_task_type(state.get("task_type"), default=al_state.TASK_CLASSIFICATION)
        self.label_profile = dict(state.get("label_profile") or {})
        labels = state.get("selected_labels") or []
        if isinstance(labels, str):
            labels = [part.strip() for part in labels.replace("\n", ",").split(",") if part.strip()]
        self.selected_labels = [str(label) for label in labels if label not in (None, "")]
        self.recipe_profile_id = str(state.get("recipe_profile_id") or state.get("recipe_id") or "")
        self.recipe_id = str(state.get("recipe_id") or "")

    def dispose(self) -> None:
        self._disposed = True

        for handle in list(self._job_handles):
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
        self._job_handles.clear()

        events = getattr(self.context, "events", None)
        unsubscribe = getattr(events, "unsubscribe", None)

        for subscription in list(self._subscriptions):
            try:
                if callable(unsubscribe):
                    unsubscribe(subscription)
                elif callable(subscription):
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
        self.label_profile = self._infer_label_profile(dataset_value, column_value, cheap_only=True) if dataset_value and column_value else {}
        self.task_type = al_state.parse_task_type(self.label_profile.get("task_type") or self.task_type)
        labels = [] if self.task_type == al_state.TASK_REGRESSION else (self.selected_labels or self._infer_label_options(dataset_value, column_value, cheap_only=True))
        recipe_options = self._recipe_options()
        recipe_value = self._valid_or_default(self.recipe_profile_id, recipe_options, allow_blank=True)

        self._widgets = {
            "dataset_id": pn.widgets.Select(name="Training pool dataset", options=dataset_options, value=dataset_value),
            "label_column": pn.widgets.Select(name="Label column", options=column_options, value=column_value),
            "labels": pn.widgets.MultiChoice(name="Labels to use", options=labels, value=labels, disabled=not bool(column_value) or self.task_type == al_state.TASK_REGRESSION),
            "label_profile": pn.pane.Markdown(self._label_profile_text(), sizing_mode="stretch_width"),
            "initial_k": pn.widgets.IntInput(name="Initial random sample", value=20, start=0),
            "seed": pn.widgets.IntInput(name="Seed", value=42),
            "session_id": pn.widgets.TextInput(name="Session artifact id", value=self.session_artifact_id),
            "model_id": pn.widgets.TextInput(name="Model artifact id", value=self.model_artifact_id),
            "predictions_id": pn.widgets.TextInput(name="Predictions artifact id", value=self.predictions_artifact_id),
            "strategy": pn.widgets.Select(name="Query strategy", options=self._strategy_options(), value="least_confidence"),
            "query_k": pn.widgets.IntInput(name="Query batch size", value=200, start=1),
            "row_id": pn.widgets.TextInput(name="Start row id", placeholder="blank = focused row or first unlabelled in latest batch"),
            "label": pn.widgets.Select(name="Label", options=self._review_label_options(labels), value=self._first_review_label(labels)),
            "label_value": pn.widgets.TextInput(name="Target value", placeholder="numeric regression value"),
            "bulk_n": pn.widgets.IntInput(name="Bulk label next N", value=5, start=1),
            "recipe_profile_id": pn.widgets.Select(name="core.ml recipe profile", options=recipe_options, value=recipe_value),
            "status": pn.pane.Markdown(self._status_text(), sizing_mode="stretch_width", margin=(0, 0, 4, 0)),
            "session_summary_left": pn.pane.Markdown("", sizing_mode="stretch_width", margin=(0, 0, 0, 0)),
            "session_summary_right": pn.pane.Markdown("", sizing_mode="stretch_width", margin=(0, 0, 0, 0)),
            "performance_metric": pn.widgets.Select(name="Performance metric", options={"No metrics yet": ""}, value=""),
            "performance_plot": pn.Column(pn.pane.Markdown("No AL performance points yet."), sizing_mode="stretch_width"),
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
            "query_strategy_info": pn.pane.Markdown(self._strategy_help_text("least_confidence")),
            "xy_colour": pn.widgets.Select(
                name="Colour by",
                options=self._xy_colour_options(),
                value="prediction_correctness",
            ),
            "strategy_scores_summary": pn.pane.HTML(self._strategy_scores_empty_html(), sizing_mode="stretch_width", margin=(0, 0, 8, 0)),
            "xy_show_trained": pn.widgets.Checkbox(name="Show trained/labelled overlay", value=True),
            "xy_max_points": pn.widgets.IntInput(name="Max plotted rows", value=XY_DEFAULT_MAX_POINTS, start=100),
            "xy_plot": pn.pane.Matplotlib(None, tight=True, sizing_mode="stretch_width", height=420, min_width=120, min_height=240),
        }

        start_btn = self._button("Start session", button_type="primary")
        query_btn = self._button("Create query batch", button_type="primary")
        label_btn = self._button("Record label", button_type="success")
        bulk_label_btn = self._button("Next N labels from column", button_type="success")
        train_btn = self._button("Train via core.ml", button_type="warning")
        refresh_data_btn = self._button("Refresh datasets", button_type="default")
        refresh_strategy_btn = self._button("Refresh strategies", button_type="default")
        refresh_recipe_btn = self._button("Refresh profiles", button_type="default")
        refresh_performance_btn = self._button("Refresh performance", button_type="default")
        refresh_xy_btn = self._button("Refresh XY plot", button_type="default")
        score_all_btn = self._button("Calculate QS scores over pool", button_type="primary")
        self._widgets.update({
            "start_btn": start_btn,
            "query_btn": query_btn,
            "label_btn": label_btn,
            "bulk_label_btn": bulk_label_btn,
            "train_btn": train_btn,
            "refresh_performance_btn": refresh_performance_btn,
            "refresh_xy_btn": refresh_xy_btn,
            "score_all_btn": score_all_btn,
        })

        self._widgets["dataset_id"].param.watch(lambda event: self._on_dataset_changed(str(event.new or "")), "value")
        self._widgets["label_column"].param.watch(lambda event: self._on_label_column_changed(str(event.new or "")), "value")
        self._widgets["labels"].param.watch(lambda event: self._on_labels_changed(list(event.new or [])), "value")
        self._widgets["recipe_profile_id"].param.watch(lambda event: self._on_recipe_profile_changed(str(event.new or "")), "value")
        self._widgets["strategy"].param.watch(lambda event: self._refresh_strategy_info(str(event.new or "")), "value")
        self._widgets["row_id"].param.watch(lambda event: self._on_review_row_changed(str(event.new or "")), "value")
        self._widgets["performance_metric"].param.watch(lambda event: self._refresh_performance(status=False, keep_metric=True), "value")
        self._widgets["xy_x"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_y"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_colour"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
        self._widgets["xy_scope"].param.watch(lambda event: self._on_xy_dataset_choice_changed(), "value")
        self._widgets["xy_validation_dataset_id"].param.watch(lambda event: self._on_xy_dataset_choice_changed(), "value")
        self._widgets["xy_show_trained"].param.watch(lambda event: self._refresh_xy_plot(status=False), "value")
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
        score_all_btn.on_click(lambda event: self._run_score_pool())

        start_objects: List[Any] = [
            "### Start",
            self._widgets["dataset_id"],
            self._widgets["label_column"],
            self._widgets["label_profile"],
            self._widgets["labels"],
            "For classification, the class set is inferred from the label column and can be limited here. For regression, labels are numeric target values and no class list is used.",
        ]
        protocol_controls = self._session_protocol_controls_view()
        if isinstance(protocol_controls, (list, tuple)):
            start_objects.extend(protocol_controls)
        elif protocol_controls is not None:
            start_objects.append(protocol_controls)
        start_objects.extend(
            [
                self._compact_row(self._widgets["initial_k"], self._widgets["seed"]),
                self._compact_row(start_btn, refresh_data_btn),
            ]
        )
        start_tab = self._scrollable_tab(*start_objects)
        query_tab = self._scrollable_tab(
            "### Query",
            pn.Accordion(
                (
                    "Advanced artifact ids",
                    pn.Column(
                        self._widgets["session_id"],
                        self._widgets["model_id"],
                        self._widgets["predictions_id"],
                        sizing_mode="stretch_width",
                    ),
                ),
                active=[],
                sizing_mode="stretch_width",
            ),
            self._compact_row(self._widgets["strategy"], refresh_strategy_btn),
            self._widgets["query_strategy_info"],
            self._widgets["query_k"],
            query_btn,
        )
        review_tab = self._scrollable_tab(
            "### Review",
            self._widgets["session_id"],
            self._widgets["row_id"],
            self._widgets["label"],
            self._widgets["label_value"],
            label_btn,
            self._compact_row(self._widgets["bulk_n"], bulk_label_btn),
            "`Next N labels from column` uses each row's pre-assigned value in the selected label column; it does not repeat the dropdown value.",
            "Use label `Unsure` to remove a row from the query pool without adding it to training.",
        )
        train_objects: List[Any] = [
            "### Train",
            self._widgets["session_id"],
            self._compact_row(self._widgets["recipe_profile_id"], refresh_recipe_btn),
            self._widgets["seed"],
        ]
        training_controls = self._training_controls_view(train_btn)
        if isinstance(training_controls, (list, tuple)):
            train_objects.extend(training_controls)
        elif training_controls is not None:
            train_objects.append(training_controls)
        train_objects.extend(
            [
                "Pause is cooperative: the current epoch, validation pass, scheduler update, and checkpoint save complete before the job enters the paused state. Cancel requests a clean stop through the platform JobManager.",
                "Training materialises the currently labelled rows and updates model/prediction artifacts. Create the next query batch from the Query tab after choosing the strategy and batch size.",
            ]
        )
        train_tab = self._scrollable_tab(*train_objects)
        performance_tab = self._scrollable_tab(
            "### AL Performance",
            "Each point is one completed active-learning training round. The x-axis is the number of labelled training rows used in that round.",
            self._compact_row(self._widgets["performance_metric"], refresh_performance_btn),
            self._widgets["performance_plot"],
        )
        xy_tab = self._scrollable_tab(
            "### XY Diagnostics",
            "Plot the pool in two dataset columns. Labelled/trained rows are overlaid, and the latest query batch can be coloured by rank or score to inspect where the selected query strategy found informative points.",
            self._compact_row(self._widgets["xy_scope"], self._widgets["xy_validation_dataset_id"]),
            self._compact_row(self._widgets["xy_x"], self._widgets["xy_y"]),
            self._compact_row(self._widgets["xy_colour"], self._widgets["xy_max_points"], refresh_xy_btn),
            self._compact_row(score_all_btn),
            self._widgets["strategy_scores_summary"],
            self._widgets["xy_show_trained"],
            self._widgets["xy_plot"],
        )
        self._sync_task_controls()
        self._sync_start_controls()
        self._sync_train_controls()
        self._refresh_session_summary(status=False)
        self._refresh_performance(status=False)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._refresh_strategy_scores_summary()
        self._refresh_xy_plot(status=False)
        self._tab_index = {"Start": 0, "Query": 1, "Review": 2, "Train": 3, "Performance": 4, "XY": 5}
        self._tabs = pn.Tabs(
            ("Start", start_tab),
            ("Query", query_tab),
            ("Review", review_tab),
            ("Train", train_tab),
            ("Performance", performance_tab),
            ("XY", xy_tab),
            sizing_mode="stretch_both",
            dynamic=True,
            styles={
                "flex": "1 1 auto",
                "min-width": "0",
                "min-height": "0",
                "height": "100%",
                "max-height": "100%",
                "overflow": "hidden",
            },
        )
        return self._scrollable_root(
            self._fixed_header(
                pn.pane.Markdown("## Active Learning", sizing_mode="stretch_width", margin=(0, 0, 6, 0)),
                self._header_summary_grid(),
            ),
            self._tabs,
        )

    def _header_summary_grid(self):
        left = pn.Column(
            self._widgets["status"],
            self._widgets["session_summary_left"],
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        right = pn.Column(
            self._widgets["session_summary_right"],
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )
        row = pn.Row(left, right, sizing_mode="stretch_width", margin=(0, 0, 4, 0))
        for column in (left, right):
            try:
                column.styles = {
                    **dict(getattr(column, "styles", {}) or {}),
                    "flex": "1 1 16rem",
                    "min-width": "14rem",
                    "max-width": "100%",
                    "box-sizing": "border-box",
                    "overflow": "visible",
                }
            except Exception:
                pass
        try:
            row.styles = {
                **dict(getattr(row, "styles", {}) or {}),
                "display": "flex",
                "flex-wrap": "wrap",
                "gap": "0.75rem",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "box-sizing": "border-box",
                "overflow": "visible",
            }
        except Exception:
            pass
        return row

    def _fixed_header(self, *objects: Any):
        header = pn.Column(*objects, sizing_mode="stretch_width", margin=(0, 0, 8, 0))
        try:
            header.styles = {
                **dict(getattr(header, "styles", {}) or {}),
                "flex": "0 0 auto",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "height": "auto",
                "max-height": "none",
                "overflow": "visible",
                "box-sizing": "border-box",
            }
        except Exception:
            pass
        for obj in objects:
            self._fixed_child(obj)
        return header

    def _fixed_child(self, obj: Any) -> None:
        try:
            obj.styles = {
                **dict(getattr(obj, "styles", {}) or {}),
                "flex": "0 0 auto",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "height": "auto",
                "max-height": "none",
                "overflow": "visible",
                "box-sizing": "border-box",
            }
        except Exception:
            pass
        try:
            obj.sizing_mode = "stretch_width"
        except Exception:
            pass

    def _button(self, name: str, *, button_type: str = "default") -> Any:
        button = pn.widgets.Button(name=name, button_type=button_type, height=BUTTON_HEIGHT)
        try:
            button.min_height = BUTTON_HEIGHT
            button.max_height = BUTTON_HEIGHT
        except Exception:
            pass
        try:
            button.height_policy = "fixed"
        except Exception:
            pass
        try:
            button.sizing_mode = "fixed"
        except Exception:
            pass
        try:
            button.styles = {
                **dict(getattr(button, "styles", {}) or {}),
                "height": f"{BUTTON_HEIGHT}px",
                "min-height": f"{BUTTON_HEIGHT}px",
                "max-height": f"{BUTTON_HEIGHT}px",
                "line-height": f"{BUTTON_HEIGHT - 2}px",
                "align-self": "flex-start",
                "flex": "0 0 auto",
                "overflow": "hidden",
                "white-space": "nowrap",
            }
        except Exception:
            pass
        return button

    def _compact_row(self, *objects: Any):
        row = pn.Row(*objects, sizing_mode="stretch_width", margin=(0, 0, 8, 0))
        try:
            row.styles = {
                **dict(getattr(row, "styles", {}) or {}),
                "display": "flex",
                "flex-wrap": "wrap",
                "align-items": "flex-start",
                "gap": "0.75rem",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
            }
        except Exception:
            pass
        for obj in objects:
            self._compact_child(obj)
        return row

    def _compact_child(self, obj: Any) -> None:
        class_name = obj.__class__.__name__.lower()
        is_button = class_name.endswith("button")
        try:
            obj.styles = {
                **dict(getattr(obj, "styles", {}) or {}),
                "flex": "0 1 12rem" if is_button else "1 1 12rem",
                "max-width": "100%",
                "min-width": "9rem" if not is_button else "7rem",
                "box-sizing": "border-box",
                "margin-bottom": "0.25rem",
            }
        except Exception:
            pass
        try:
            obj.margin = (0, 0, 4, 0)
        except Exception:
            pass
        if is_button:
            try:
                obj.height = BUTTON_HEIGHT
                obj.height_policy = "fixed"
                obj.sizing_mode = "fixed"
            except Exception:
                pass
            try:
                obj.styles = {
                    **dict(getattr(obj, "styles", {}) or {}),
                    "height": f"{BUTTON_HEIGHT}px",
                    "min-height": f"{BUTTON_HEIGHT}px",
                    "max-height": f"{BUTTON_HEIGHT}px",
                    "line-height": f"{BUTTON_HEIGHT - 2}px",
                    "align-self": "flex-start",
                    "flex": "0 0 auto",
                    "overflow": "hidden",
                    "white-space": "nowrap",
                }
            except Exception:
                pass
            return
        try:
            obj.sizing_mode = "stretch_width"
        except Exception:
            pass
        try:
            obj.height_policy = "auto"
        except Exception:
            pass

    def _tab_child(self, obj: Any) -> Any:
        if isinstance(obj, str):
            obj = pn.pane.Markdown(obj, sizing_mode="stretch_width", margin=(0, 0, 8, 0))
        class_name = obj.__class__.__name__.lower()
        is_row = class_name == "row"
        try:
            obj.styles = {
                **dict(getattr(obj, "styles", {}) or {}),
                "flex": "0 0 auto",
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "box-sizing": "border-box",
                "overflow-x": "hidden" if not is_row else "visible",
            }
        except Exception:
            pass
        try:
            obj.margin = getattr(obj, "margin", None) or (0, 0, 8, 0)
        except Exception:
            pass
        is_button = class_name.endswith("button")
        if is_button:
            try:
                obj.height = BUTTON_HEIGHT
                obj.height_policy = "fixed"
                obj.sizing_mode = "fixed"
            except Exception:
                pass
            try:
                obj.styles = {
                    **dict(getattr(obj, "styles", {}) or {}),
                    "height": f"{BUTTON_HEIGHT}px",
                    "min-height": f"{BUTTON_HEIGHT}px",
                    "max-height": f"{BUTTON_HEIGHT}px",
                    "line-height": f"{BUTTON_HEIGHT - 2}px",
                    "align-self": "flex-start",
                    "flex": "0 0 auto",
                    "overflow": "hidden",
                    "white-space": "nowrap",
                }
            except Exception:
                pass
        elif not is_row and class_name not in {"markdown", "spacer"}:
            try:
                obj.sizing_mode = "stretch_width"
            except Exception:
                pass
            try:
                obj.height_policy = "auto"
            except Exception:
                pass
        return obj

    def _scrollable_tab(self, *objects: Any):
        spaced_objects = [self._tab_child(obj) for obj in objects]
        column = pn.Column(*spaced_objects, pn.Spacer(height=20), sizing_mode="stretch_both")
        self._apply_scroll_styles(column, max_height="100%")
        return column

    def _scrollable_root(self, *objects: Any):
        root = pn.Column(*objects, pn.Spacer(height=20), sizing_mode="stretch_both")
        self._apply_scroll_styles(root, max_height="100%")
        return root

    def _apply_scroll_styles(self, layout: Any, *, max_height: str) -> None:
        try:
            layout.styles = {
                **dict(getattr(layout, "styles", {}) or {}),
                "display": "flex",
                "flex-direction": "column",
                "gap": "0.5rem",
                "width": "100%",
                "max-width": "100%",
                "height": "100%",
                "max-height": max_height,
                "min-width": "0",
                "min-height": "0",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
                "padding-right": "0.5rem",
                "padding-bottom": "0.25rem",
                "scrollbar-gutter": "stable",
            }
        except Exception:
            pass

    def refresh_choices(self, *, status: bool = True) -> None:
        self.refresh_dataset_controls(status=False)
        self._refresh_strategies(status=False)
        self._refresh_recipes(status=False)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        # Do not auto-render the XY plot while refreshing choices.  On large
        # datasets this can materialise millions of rows before the user has
        # explicitly asked for a plot.
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
        self.label_profile = self._infer_label_profile(dataset_id, label_column) if dataset_id and label_column else {}
        self.task_type = al_state.parse_task_type(self.label_profile.get("task_type"), default=al_state.TASK_CLASSIFICATION)
        profile_labels = [str(label) for label in ((self.label_profile or {}).get("label_options") or (self.label_profile or {}).get("class_labels") or []) if str(label).strip()]
        inferred = [] if self.task_type == al_state.TASK_REGRESSION else (profile_labels or (self._infer_label_options(dataset_id, label_column) if dataset_id and label_column else []))
        selected = [str(value) for value in (labels_widget.value or []) if str(value) in set(inferred)] if keep_selected else []
        if not selected:
            selected = list(inferred)
        labels_widget.options = inferred
        labels_widget.value = selected
        labels_widget.disabled = not bool(label_column) or self.task_type == al_state.TASK_REGRESSION
        self._update_label_profile_pane()
        self._sync_task_controls()
        self._sync_label_controls(selected)
        self._sync_start_controls()

    def _on_dataset_changed(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self._refresh_label_columns(dataset_id, keep_current=False)
        self._dataset_columns_cache.pop(str(dataset_id or ""), None)
        self._plottable_columns_cache.pop(str(dataset_id or ""), None)
        self._refresh_xy_dataset_controls(status=False)
        self._refresh_xy_columns(status=False)
        self._set_status("Dataset changed. Label columns and XY column choices updated; press Refresh XY plot to render.")

    def _on_label_column_changed(self, label_column: str) -> None:
        self.label_column = label_column
        dataset_id = str(self._widgets.get("dataset_id").value or "") if self._widgets.get("dataset_id") is not None else ""
        self._refresh_inferred_labels(dataset_id, label_column, keep_selected=False)
        if label_column:
            task = "regression" if self.task_type == al_state.TASK_REGRESSION else "classification"
            self._set_status(f"Detected `{label_column}` as a {task} target. {self._label_profile_reason()}")
        else:
            self._set_status("Select a label column to infer the target type and labels.")

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
        self._sync_task_controls()

    def _sync_task_controls(self) -> None:
        is_regression = self.task_type == al_state.TASK_REGRESSION
        labels_widget = self._widgets.get("labels")
        label_select = self._widgets.get("label")
        label_value = self._widgets.get("label_value")
        for widget, visible in ((labels_widget, not is_regression), (label_select, not is_regression), (label_value, is_regression)):
            if widget is not None:
                try:
                    widget.visible = visible
                except Exception:
                    pass
        if labels_widget is not None:
            try:
                labels_widget.disabled = is_regression or not bool(self.label_column)
            except Exception:
                pass

    def _update_label_profile_pane(self) -> None:
        pane = self._widgets.get("label_profile")
        if pane is not None:
            pane.object = self._label_profile_text()

    def _label_profile_reason(self) -> str:
        reason = str((self.label_profile or {}).get("reason") or "").strip()
        return reason[:1].upper() + reason[1:] + "." if reason else ""

    def _label_profile_text(self) -> str:
        if not self.label_column:
            return "**Target type:** —"
        profile = dict(self.label_profile or {})
        task = al_state.parse_task_type(profile.get("task_type") or self.task_type)
        sample_count = profile.get("sample_count")
        unique_count = profile.get("unique_count_sampled")
        numeric = profile.get("is_numeric")
        reason = str(profile.get("reason") or "")
        if task == al_state.TASK_REGRESSION:
            return (
                "**Target type:** regression  \n"
                f"**Reason:** {reason or 'numeric continuous target'}  \n"
                f"**Profile:** sampled {sample_count if sample_count not in (None, '') else '—'} non-null values; "
                f"numeric={bool(numeric)}; distinct sampled values={unique_count if unique_count not in (None, '') else '—'}."
            )
        labels = profile.get("label_options") or profile.get("class_labels") or []
        label_text = ", ".join(str(label) for label in labels[:12])
        if len(labels) > 12:
            label_text += f", … +{len(labels) - 12} more"
        return (
            "**Target type:** classification  \n"
            f"**Reason:** {reason or 'categorical target'}  \n"
            f"**Detected classes:** {label_text or '—'}"
        )

    def _sync_start_controls(self) -> None:
        button = self._widgets.get("start_btn")
        if button is None:
            return
        dataset_id = str(self._widget_value("dataset_id", "") or "")
        label_column = str(self._widget_value("label_column", "") or "")
        labels = list(self._widget_value("labels", []) or [])
        needs_classes = self.task_type != al_state.TASK_REGRESSION
        button.disabled = not bool(dataset_id and label_column and (labels or not needs_classes))

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
            if self.task_type != al_state.TASK_REGRESSION and not labels:
                raise ValueError("Select at least one class label for the active-learning session.")
            result = actions.start_session_action(
                self.context,
                _new_action_request(
                    dataset_id=dataset_id,
                    row_ids=None,
                    columns=[],
                    params={
                        "dataset_id": dataset_id,
                        "label_options": [] if self.task_type == al_state.TASK_REGRESSION else labels,
                        "task_type": self.task_type,
                        "problem_type": self.task_type,
                        "label_profile": dict(self.label_profile or {}),
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
            self._set_status(f"Started {self.task_type} session `{result.get('session_id')}` with {count} initial rows. Review tab selected.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("start_btn", False)

    def _track_job_handle(self, handle: Any) -> None:
        if handle is None:
            return

        cancel = getattr(handle, "cancel", None)
        if callable(cancel) and handle not in self._job_handles:
            self._job_handles.append(handle)

    def _discard_job_handle(self, handle: Any) -> None:
        try:
            self._job_handles.remove(handle)
        except ValueError:
            pass

    def _run_plugin_action(
        self,
        action_id: str,
        request: Any,
        *,
        on_done: Any,
        on_error: Any,
    ) -> Any:
        plugins = getattr(self.context, "plugins", None)
        run_action = getattr(plugins, "run_action", None)

        if not callable(run_action):
            raise RuntimeError(
                "Plugin manager is not available on context; active-learning "
                "panel actions must run through context.plugins.run_action()."
            )

        handle = run_action(
            action_id,
            self.context,
            request,
            on_done=on_done,
            on_error=on_error,
        )
        self._track_job_handle(handle)
        return handle

    def _run_query(self) -> None:
        self._set_status("Creating query batch from the selected predictions and strategy...")
        self._set_button_busy("query_btn", True)

        try:
            session_id = str(self._widget_value("session_id", "") or "").strip()
            predictions_id = str(self._widget_value("predictions_id", "") or "").strip()

            if not session_id:
                raise ValueError("Start or select an active-learning session before creating a query batch.")
            if not predictions_id:
                raise ValueError("Train/predict first, or enter an ml.predictions artifact id.")

            request = _new_action_request(
                dataset_id=None,
                row_ids=None,
                columns=[],
                params={
                    "session_artifact_id": session_id,
                    "predictions_artifact_id": predictions_id,
                    "strategy_id": self._widget_value("strategy", "least_confidence"),
                    "k": int(self._widget_value("query_k", 200) or 200),
                    "seed": int(self._widget_value("seed", 42) or 42),
                    "make_selection": True,
                },
                artifact_id=predictions_id or None,
                origin="core.active_learning.panel",
            )
        except Exception as exc:
            self._set_button_busy("query_btn", False)
            self._set_status(f"Error: {exc}")
            return

        handle = None

        def done(result: Any) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            payload = dict(result or {}) if isinstance(result, Mapping) else {}

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("query_btn", False)
                self._set_session_id(payload.get("session_artifact_id"))
                self._refresh_session_summary(status=False)
                self._refresh_performance(status=False)

                row_ids = [str(row_id) for row_id in (payload.get("row_ids") or []) if str(row_id)]
                if row_ids:
                    self._set_review_row(row_ids[0])

                self._refresh_xy_plot(status=False)
                self._set_status(self._query_feedback_text(payload))

            self._next_tick(update)

        def failed(exc: BaseException) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("query_btn", False)
                self._set_status(f"Error: {exc}")

            self._next_tick(update)

        try:
            handle = self._run_plugin_action(
                "core.active_learning.query_batch",
                request,
                on_done=done,
                on_error=failed,
            )
        except Exception as exc:
            self._set_button_busy("query_btn", False)
            self._set_status(f"Error: {exc}")

    def _current_label_value(self) -> str:
        if self.task_type == al_state.TASK_REGRESSION:
            return str(self._widgets.get("label_value").value if self._widgets.get("label_value") is not None else "").strip()
        return str(self._widgets.get("label").value if self._widgets.get("label") is not None else "").strip()

    def _run_label(self) -> None:
        self._set_status("Recording label...")
        self._set_button_busy("label_btn", True)
        try:
            result = actions.record_label_action(
                self.context,
                _new_action_request(
                    dataset_id=None,
                    row_ids=None,
                    columns=[],
                    params={
                        "session_artifact_id": self._widgets["session_id"].value.strip(),
                        "row_id": self._widgets["row_id"].value.strip(),
                        "label": self._current_label_value(),
                    },
                    artifact_id=None,
                    origin="core.active_learning.panel",
                ),
            )
            labelled_row_id = str(result.get("row_id") or "")
            self._set_session_id(result.get("session_artifact_id"))
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            # Keep single-label navigation cheap: the action updates platform focus,
            # but we intentionally avoid rebuilding the whole selection set on every click.
            next_row_id = str(result.get("next_row_id") or "").strip() or self._next_unverified_review_row(after_row_id=labelled_row_id)
            if next_row_id:
                self._set_review_row(next_row_id)
                self._set_status(f"Recorded `{result.get('display_label')}` for row `{labelled_row_id}`. Next unverified row `{next_row_id}` selected.")
            else:
                self._set_status(f"Recorded `{result.get('display_label')}` for row `{labelled_row_id}`. No more unverified rows in this batch.")
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
                _new_action_request(
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
                next_row_id = str(result.get("next_row_id") or "").strip()
                if next_row_id:
                    self._set_review_row(next_row_id)
                self._refresh_xy_plot(status=False)
                tail = f" Next unverified row `{next_row_id}` selected." if next_row_id else " No more unverified rows remain in this batch."
                self._set_status(
                    f"Recorded source-column labels for {result.get('count')} rows "
                    f"from `{row_ids[0]}` to `{row_ids[-1]}`. Skipped {len(skipped)} rows." + tail
                )
            else:
                self._set_status("No unlabelled rows were available in the latest review/query batch.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_button_busy("bulk_label_btn", False)

    def _run_score_pool(self) -> None:
        session_id = str(self._widget_value("session_id", "") or "").strip()
        predictions_id = str(self._widget_value("predictions_id", "") or "").strip()

        if not session_id:
            self._set_status("Error: Start or select an active-learning session before calculating query-strategy scores.")
            return
        if not predictions_id:
            self._set_status("Error: Select pool predictions before calculating query-strategy scores.")
            return

        strategies = [str(value) for value in self._strategy_options().values() if str(value)]
        self._set_status(
            f"Calculating query-strategy scores for {len(strategies)} strategies over the eligible pool. "
            "This uses the current prediction artifact; it does not retrain the model."
        )
        self._set_button_busy("score_all_btn", True)

        request = _new_action_request(
            dataset_id=None,
            row_ids=None,
            columns=[],
            params={
                "session_artifact_id": session_id,
                "predictions_artifact_id": predictions_id,
                "strategy_ids": strategies,
                "seed": int(self._widget_value("seed", 42) or 42),
            },
            artifact_id=predictions_id or None,
            origin="core.active_learning.panel",
        )

        handle = None

        def done(result: Any) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            payload = dict(result or {}) if isinstance(result, Mapping) else {}

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("score_all_btn", False)
                self._apply_score_pool_result(payload)

            self._next_tick(update)

        def failed(exc: BaseException) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("score_all_btn", False)
                self._set_status(f"Strategy-score calculation failed: {exc}")
                self._refresh_session_summary(status=False)

            self._next_tick(update)

        try:
            handle = self._run_plugin_action(
                "core.active_learning.score_pool",
                request,
                on_done=done,
                on_error=failed,
            )
        except Exception as exc:
            self._set_button_busy("score_all_btn", False)
            self._set_status(f"Strategy-score calculation failed: {exc}")

    def _apply_score_pool_result(self, result: Mapping[str, Any]) -> None:
        self._set_session_id(result.get("session_artifact_id"))
        artifact_id = str(result.get("strategy_scores_artifact_id") or "").strip()
        if artifact_id:
            self.strategy_scores_artifact_id = artifact_id
        self._refresh_session_summary(status=False)
        self._refresh_strategy_scores_summary(result)
        self._refresh_xy_colour_options(keep_current=True)
        self._refresh_xy_plot(status=False)
        self._set_status(self._score_pool_feedback_text(result))

    def _query_feedback_text(self, result: Mapping[str, Any]) -> str:
        stats = dict(result.get("rank_stats") or {})
        title = str(stats.get("strategy_title") or result.get("strategy_id") or "strategy")
        strategy_id = str(stats.get("strategy_id") or result.get("strategy_id") or "")
        pool_count = stats.get("eligible_pool_count") or stats.get("scored_pool_count") or stats.get("prediction_record_count") or "?"
        selected_count = result.get("count", stats.get("selected_count", "?"))
        batch_id = result.get("batch_artifact_id")
        pieces = [f"Created query batch `{batch_id}` with {selected_count} rows using **{title}**."]
        pieces.append(f"Scored/considered {pool_count} eligible pool rows before taking the top {selected_count}.")
        if bool(stats.get("batch_aware")):
            pieces.append("This is a batch-aware strategy, so row choices can depend on the rest of the selected batch, not only independent per-row scores.")
        sources = self._score_source_summary(stats.get("score_source_counts"))
        if sources:
            pieces.append(f"Score source: {sources}.")
        if strategy_id in {"learning_loss", "badge", "coreset"}:
            pieces.append("Querying uses the existing prediction artifact and required emitted model outputs; it does not start a new training run.")
        pieces.append("Review rows are ready.")
        return " ".join(str(piece) for piece in pieces if piece)

    def _score_pool_feedback_text(self, result: Mapping[str, Any]) -> str:
        strategy_ids = [str(value) for value in (result.get("strategy_ids") or []) if str(value)]
        eligible = result.get("eligible_pool_count", "?")
        artifact_id = result.get("strategy_scores_artifact_id")
        stats_by_strategy = dict(result.get("stats_by_strategy") or {})
        failed = [strategy_id for strategy_id in strategy_ids if dict(stats_by_strategy.get(strategy_id) or {}).get("error")]
        text = (
            f"Calculated whole-pool query-strategy scores for {len(strategy_ids) - len(failed)} of {len(strategy_ids)} strategies over {eligible} eligible rows. "
            f"Score artifact `{artifact_id}` is selected for XY diagnostics. "
            "Choose any `QS score: ...` entry in Colour by to visualise its informativeness map."
        )
        if failed:
            text += " Some strategies need extra model outputs and were not scored: " + ", ".join(failed) + "."
        return text

    def _score_source_summary(self, value: Any) -> str:
        if not isinstance(value, Mapping) or not value:
            return ""
        items = sorted(((str(key), int(count or 0)) for key, count in value.items()), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{key} × {count}" for key, count in items if count)

    def _refresh_strategy_scores_summary(self, result: Mapping[str, Any] | None = None) -> None:
        pane = self._widgets.get("strategy_scores_summary")
        if pane is None:
            return
        if result is None:
            session = self._session_payload()
            artifact_id = self._strategy_scores_artifact_id(session)
            if not artifact_id:
                pane.object = self._strategy_scores_empty_html()
                return
            try:
                payload = self.context.artifacts.get(artifact_id)
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                result = {
                    "strategy_scores_artifact_id": artifact_id,
                    "strategy_ids": list(payload.get("strategy_ids") or []),
                    "eligible_pool_count": payload.get("eligible_pool_count"),
                    "stats_by_strategy": payload.get("stats_by_strategy") or {},
                }
            else:
                pane.object = self._strategy_scores_empty_html(f"Latest strategy-score artifact {artifact_id} could not be read.")
                return
        pane.object = self._strategy_scores_summary_html(result)

    def _strategy_scores_empty_html(self, message: str = "No whole-pool query-strategy scores calculated yet.") -> str:
        return (
            '<div style="font-size: 0.92em; margin: 0.15rem 0 0.4rem 0;">'
            f'<span style="color: #666;">{html.escape(str(message))}</span>'
            '</div>'
        )

    def _strategy_scores_summary_html(self, result: Mapping[str, Any]) -> str:
        strategy_ids = [str(value) for value in (result.get("strategy_ids") or []) if str(value)]
        eligible = result.get("eligible_pool_count", "?")
        artifact_id = str(result.get("strategy_scores_artifact_id") or "—")
        stats_by_strategy = dict(result.get("stats_by_strategy") or {})
        successes: List[Tuple[str, str, str]] = []
        failures: List[Tuple[str, str, str]] = []
        for strategy_id in strategy_ids:
            stats = dict(stats_by_strategy.get(strategy_id) or {})
            title = str(stats.get("strategy_title") or strategy_id)
            error = str(stats.get("error") or "").strip()
            if error:
                failures.append((title, self._score_failure_summary(strategy_id, error), error))
                continue
            detail = self._score_source_summary(stats.get("score_source_counts")) or "direct score"
            warnings = self._score_warning_summary(stats)
            if warnings:
                detail = f"{detail}; warning: {warnings}"
            successes.append((title, detail, detail))

        def list_html(items: Sequence[Tuple[str, str, str]], *, ok: bool) -> str:
            if not items:
                label = "No successes" if ok else "No failures"
                return f'<div style="color: #777; font-size: 0.9em;">{html.escape(label)}</div>'
            marker = "✓" if ok else "✗"
            colour = "#207245" if ok else "#9a3412"
            rows = []
            for title, detail, full_detail in items:
                rows.append(
                    '<li style="margin: 0 0 0.18rem 0; line-height: 1.25;" '
                    f'title="{html.escape(str(full_detail), quote=True)}">'
                    f'<span style="color: {colour}; font-weight: 700;">{marker}</span> '
                    f'<strong>{html.escape(str(title))}</strong>'
                    f'<br><span style="color: #555; font-size: 0.86em;">{html.escape(str(detail))}</span>'
                    '</li>'
                )
            return '<ul style="margin: 0.25rem 0 0 1.1rem; padding: 0;">' + "".join(rows) + '</ul>'

        header = (
            '<div style="font-size: 0.92em; margin: 0.1rem 0 0.35rem 0;">'
            f'<strong>Whole-pool QS scores:</strong> <code>{html.escape(artifact_id)}</code> — '
            f'{len(successes)} / {len(strategy_ids)} strategies over {html.escape(str(eligible))} eligible rows. '
            '<span style="color: #666;">Choose a <code>QS score: ...</code> entry in Colour by.</span>'
            '</div>'
        )
        grid = (
            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); '
            'gap: 0.75rem; align-items: start; margin-bottom: 0.4rem;">'
            '<div style="min-width: 0;">'
            '<div style="font-weight: 700;">Calculated</div>'
            f'{list_html(successes, ok=True)}'
            '</div>'
            '<div style="min-width: 0;">'
            '<div style="font-weight: 700;">Not calculated</div>'
            f'{list_html(failures, ok=False)}'
            '</div>'
            '</div>'
        )
        return header + grid

    def _score_warning_summary(self, stats: Mapping[str, Any]) -> str:
        values: List[str] = []
        raw_values = stats.get("warnings") or []
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        for value in raw_values:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
        single = str(stats.get("warning") or "").strip()
        if single and single not in values:
            values.append(single)
        return "; ".join(values[:2])

    def _score_failure_summary(self, strategy_id: str, error: str) -> str:
        text = " ".join(str(error or "").split())
        lower = text.lower()
        if strategy_id == "learning_loss" or "learning loss" in lower:
            return "requires loss-prediction output"
        if strategy_id == "badge" or "badge" in lower:
            return "requires gradient embedding, or probabilities + embeddings/features"
        if strategy_id == "coreset" or "core-set" in lower or "embedding" in lower:
            return "requires embedding/feature vectors"
        if not text:
            return "not calculated"
        first_sentence = text.split(".", 1)[0].strip() or text
        return first_sentence[:120] + ("…" if len(first_sentence) > 120 else "")

    def _run_train(self) -> None:
        self._set_button_busy("train_btn", True)
        self._set_status("Training from active-learning session...")

        try:
            session_id = str(self._widget_value("session_id", "") or "").strip()
            recipe_profile_id = str(self._widget_value("recipe_profile_id", "") or "").strip()
            label_column = str(self._widget_value("label_column", self.label_column) or "").strip()

            if not session_id:
                raise ValueError("Start or select an active-learning session before training.")
            if not recipe_profile_id:
                raise ValueError("Select a saved core.ml recipe profile before training.")
            if not label_column:
                raise ValueError("Select a label column before training.")

            seed_value = int(self._widget_value("seed", 42) or 42)

            self.recipe_profile_id = recipe_profile_id
            self.recipe_id = ""

            request = _new_action_request(
                dataset_id=None,
                row_ids=None,
                columns=[],
                params={
                    "session_artifact_id": session_id,
                    "recipe_profile_id": recipe_profile_id,
                    "target_column": label_column,
                    "label_column": label_column,
                    "task_type": self.task_type,
                    "problem_type": self.task_type,
                    "label_profile": dict(self.label_profile or {}),
                    "seed": seed_value,
                    "auto_predict": True,
                    "auto_query": False,
                    "make_selection": False,
                },
                artifact_id=None,
                origin="core.active_learning.panel",
            )
        except Exception as exc:
            self._set_button_busy("train_btn", False)
            self._set_status(f"Training failed: {exc}")
            return

        handle = None

        def done(result: Any) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            payload = dict(result or {}) if isinstance(result, Mapping) else {}

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("train_btn", False)
                self._apply_training_result(payload)

            self._next_tick(update)

        def failed(exc: BaseException) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            def update() -> None:
                if self._disposed:
                    return

                self._set_button_busy("train_btn", False)
                self._set_status(f"Training failed: {exc}")
                self._refresh_session_summary(status=False)
                self._refresh_performance(status=False)

            self._next_tick(update)

        try:
            handle = self._run_plugin_action(
                "core.active_learning.train_from_session",
                request,
                on_done=done,
                on_error=failed,
            )
        except Exception as exc:
            self._set_button_busy("train_btn", False)
            self._set_status(f"Training failed: {exc}")

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
        current_strategy = str(getattr(widget, "value", "") or "") if widget is not None else ""
        self._refresh_strategy_info(current_strategy)
        self._refresh_xy_colour_options(keep_current=True)
        if status:
            self._set_status("Strategy list refreshed.")

    def _refresh_strategy_info(self, strategy_id: str = "") -> None:
        pane = self._widgets.get("query_strategy_info")
        if pane is not None:
            pane.object = self._strategy_help_text(strategy_id or self._widget_value("strategy", ""))

    def _strategy_help_text(self, strategy_id: str) -> str:
        strategy_id = str(strategy_id or "").strip()
        info = self._strategy_info_map().get(strategy_id)
        if not info:
            return "Uses the selected query strategy to rank the current prediction artifact. Querying does not retrain the model."
        title = str(getattr(info, "title", strategy_id) or strategy_id)
        description = str(getattr(info, "description", "") or "")
        flags: List[str] = []
        if bool(getattr(info, "batch_aware", False)):
            flags.append("batch-aware")
        if bool(getattr(info, "requires_probabilities", False)):
            flags.append("uses probabilities")
        required = list(getattr(info, "required_prediction_fields", ()) or ())
        if required:
            flags.append("expects " + ", ".join(str(value) for value in required))
        flag_text = f" ({'; '.join(flags)})" if flags else ""
        extra = " Querying uses already-generated predictions/features; only the Train tab starts a training run."
        if strategy_id == "learning_loss":
            extra += " Requires a model-emitted learning-loss/loss-prediction value for every eligible row; it does not fall back to uncertainty scores."
        elif strategy_id == "badge":
            extra += " Uses BADGE gradient embeddings directly, or computes them from probabilities and model embeddings/features; it does not fall back to entropy or random ranking."
        elif strategy_id == "coreset":
            extra += " Core-set uses embeddings/features and falls back to deterministic random ordering when no embedding columns are available."
        return f"**{title}**{flag_text}: {description}{extra}"

    def _strategy_info_map(self) -> Dict[str, Any]:
        try:
            registry = actions.get_strategy_registry(self.context)
            return {str(info.id): info for info in registry.list()}
        except Exception:
            return {}

    def _xy_colour_options(self) -> Dict[str, str]:
        options = {
            "Prediction correctness": "prediction_correctness",
            "Training status": "training_status",
            "Last query score / informativeness": "last_query_score",
            "Last query rank": "last_query_rank",
            "Source label": "label",
        }
        for title, strategy_id in self._strategy_options().items():
            options[f"QS score: {title}"] = f"strategy_score:{strategy_id}"
        return options

    def _refresh_xy_colour_options(self, *, keep_current: bool = True) -> None:
        widget = self._widgets.get("xy_colour") if self._widgets else None
        if widget is None:
            return
        options = self._xy_colour_options()
        old_value = str(widget.value or "") if keep_current else ""
        widget.options = options
        if old_value in set(options.values()):
            widget.value = old_value
        else:
            widget.value = "prediction_correctness"

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
                "Learning loss": "learning_loss",
                "BADGE": "badge",
                "Core-set": "coreset",
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
        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return []
        cached = self._dataset_columns_cache.get(dataset_id)
        if cached is not None:
            return list(cached)
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []
        columns: List[str] = []
        for method_name in ("list_columns", "columns"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    values = method(dataset_id)
                    columns = [str(value) for value in values if value not in (None, "")]
                    if columns:
                        self._dataset_columns_cache[dataset_id] = columns
                        return list(columns)
                except Exception:
                    pass
        source = self._dataset_source(dataset_id)
        for attr_name in ("columns", "column_names"):
            value = getattr(source, attr_name, None)
            try:
                value = value() if callable(value) else value
            except Exception:
                value = None
            if value:
                columns = [str(column) for column in value if column not in (None, "")]
                self._dataset_columns_cache[dataset_id] = columns
                return list(columns)
        schema = getattr(source, "schema", None)
        schema_columns = getattr(schema, "columns", None) if schema is not None else None
        if schema_columns:
            columns = [str(column) for column in schema_columns if column not in (None, "")]
            self._dataset_columns_cache[dataset_id] = columns
            return list(columns)
        # Last-resort compatibility fallback.  This should be rare; prefer the
        # platform/source schema APIs above so selecting a large table does not
        # materialise the data just to populate dropdowns.
        try:
            df = datasets.get_df(dataset_id, columns=[])
        except TypeError:
            try:
                df = datasets.get_df(dataset_id)
            except Exception:
                return []
        except Exception:
            return []
        columns = [str(column) for column in getattr(df, "columns", [])]
        self._dataset_columns_cache[dataset_id] = columns
        return list(columns)

    def _infer_label_profile(self, dataset_id: str, label_column: str, *, cheap_only: bool = False) -> Dict[str, Any]:
        if not dataset_id or not label_column:
            return {}
        try:
            return actions.infer_label_profile_from_column(self.context, dataset_id=dataset_id, column=label_column)
        except Exception:
            if cheap_only:
                return {"task_type": al_state.TASK_CLASSIFICATION, "column": label_column, "reason": "profile unavailable during cheap refresh"}
            return {"task_type": al_state.TASK_CLASSIFICATION, "column": label_column, "reason": "profile inference failed"}

    def _infer_label_options(self, dataset_id: str, label_column: str, *, max_labels: int = 500, cheap_only: bool = False) -> List[str]:
        if not dataset_id or not label_column:
            return []
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []

        values: List[Any] = []
        # Prefer cheap/source-level APIs when available.  Avoid a full wide
        # pandas materialisation during panel startup or mapping refresh.
        for owner in (datasets, self._dataset_source(dataset_id)):
            if owner is None:
                continue
            for method_name in ("unique_values", "distinct_values", "value_counts"):
                method = getattr(owner, method_name, None)
                if not callable(method):
                    continue
                attempts = (
                    lambda: method(dataset_id=dataset_id, column=label_column, limit=max_labels),
                    lambda: method(column=label_column, limit=max_labels),
                    lambda: method(label_column, limit=max_labels),
                )
                for attempt in attempts:
                    try:
                        raw = attempt()
                    except TypeError:
                        continue
                    except Exception:
                        raw = None
                    if raw is None:
                        continue
                    if isinstance(raw, Mapping):
                        values = list(raw.keys())
                    else:
                        values = list(raw)
                    break
                if values:
                    break
            if values:
                break

        if not values and cheap_only:
            return []

        if not values:
            try:
                try:
                    df = datasets.get_df(dataset_id, columns=[label_column])
                except TypeError:
                    df = datasets.get_df(dataset_id)
                if label_column not in getattr(df, "columns", []):
                    return []
                series = df[label_column].dropna()
                # Bound automatic inference work.  For very large tables this is
                # only a UI convenience; the chosen label column itself remains
                # authoritative for bulk labelling/training.
                try:
                    series = series.head(MAX_AUTO_LABEL_SCAN_ROWS)
                except Exception:
                    pass
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
            elif topic == "al.strategy_scores.calculated" or topic.endswith("strategy_scores.calculated"):
                artifact_id = str(payload.get("strategy_scores_artifact_id") or "").strip()
                if artifact_id:
                    self.strategy_scores_artifact_id = artifact_id
                self._refresh_strategy_scores_summary(payload)
            self._refresh_session_summary(status=False)
            self._refresh_performance(status=False)
            self._refresh_xy_colour_options(keep_current=True)
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
        session = self._session_payload(self.session_artifact_id)
        if session:
            self.task_type = al_state.parse_task_type(session.get("task_type") or session.get("problem_type"), default=self.task_type)
            self.label_profile = dict(session.get("label_profile") or self.label_profile or {})
            self._sync_task_controls()
            self._update_label_profile_pane()
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

    def _next_unverified_review_row(self, *, after_row_id: Any = "") -> str:
        session = self._session_payload()
        last_batch = dict(session.get("last_batch") or {}) if isinstance(session.get("last_batch"), Mapping) else {}
        row_ids = [str(row_id) for row_id in (last_batch.get("row_ids") or []) if str(row_id)]
        if not row_ids:
            return ""
        try:
            return actions.next_review_row_id(session, row_ids=row_ids, after_row_id=after_row_id)
        except Exception:
            return ""

    def _on_review_row_changed(self, row_id: str) -> None:
        self._sync_review_label_to_source(row_id)

    def _sync_review_label_to_source(self, row_id: Any) -> None:
        label = self._source_label_for_row(str(row_id or "").strip())

        if self.task_type == al_state.TASK_REGRESSION:
            value_widget = self._widgets.get("label_value")
            if value_widget is not None:
                value_widget.value = "" if label == al_state.UNSURE_DISPLAY else str(label or "")
            return

        label_widget = self._widgets.get("label")
        if label_widget is None:
            return

        options = dict(getattr(label_widget, "options", {}) or {})
        allowed = set(str(value) for value in options.values())

        if label and str(label) in allowed:
            label_widget.value = str(label)
        else:
            label_widget.value = al_state.UNSURE_DISPLAY

    def _source_label_for_row(self, row_id: str) -> str:
        row_id = str(row_id or "").strip()
        if not row_id:
            return al_state.UNSURE_DISPLAY

        dataset_id = self._session_dataset_id() or str(self._widget_value("dataset_id", "") or "")
        label_column = self._session_label_column()

        if not dataset_id or not label_column:
            return al_state.UNSURE_DISPLAY

        try:
            values = actions.dataset_label_values_by_row_id(
                self.context,
                dataset_id=dataset_id,
                row_ids=[row_id],
                label_column=label_column,
            )
        except Exception:
            return al_state.UNSURE_DISPLAY

        raw_label = values.get(row_id)

        if al_state.is_missing_label_value(raw_label):
            return al_state.UNSURE_DISPLAY

        label = al_state.normalise_label(raw_label)

        if not label or label == al_state.UNSURE_LABEL:
            return al_state.UNSURE_DISPLAY

        return label

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
        strategy_scores_artifact_id = str(latest.get("strategy_scores_artifact_id") or "")
        if model_artifact_id:
            self._set_artifact_widget("model_id", model_artifact_id)
        if predictions_artifact_id:
            self._set_artifact_widget("predictions_id", predictions_artifact_id)
        if strategy_scores_artifact_id:
            self.strategy_scores_artifact_id = strategy_scores_artifact_id
        return {
            "model_artifact_id": model_artifact_id,
            "predictions_artifact_id": predictions_artifact_id,
            "strategy_scores_artifact_id": strategy_scores_artifact_id,
        }

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
        left, right = self._session_summary_columns()
        left_pane = self._widgets.get("session_summary_left")
        right_pane = self._widgets.get("session_summary_right")
        if left_pane is not None:
            left_pane.object = left
        if right_pane is not None:
            right_pane.object = right
        if status:
            self._set_status("Session summary refreshed.")

    def _session_summary_text(self) -> str:
        left, right = self._session_summary_columns()
        return f"{left}  \n{right}"

    def _session_summary_columns(self) -> Tuple[str, str]:
        session = self._session_payload()
        if not session:
            return (
                "**Session:** none selected  \n**Round:** —  \n**Target type:** —  \n**Pool remaining:** —  \n**Labelled since last train:** —",
                "**Labelled total:** —  \n**Reviewed total:** —  \n**Unsure:** —  \n**Queued:** —  \n**Model:** —  \n**Predictions:** —",
            )
        counts = al_state.counts(session)
        latest = dict(session.get("latest") or {})
        model_id = latest.get("model_artifact_id") or "—"
        predictions_id = latest.get("predictions_artifact_id") or "—"
        pool_remaining = self._pool_remaining_summary(session)
        left = (
            f"**Session:** `{session.get('session_id')}`  \n"
            f"**Round:** {session.get('round', 0)}  \n"
            f"**Target type:** {al_state.parse_task_type(session.get('task_type') or session.get('problem_type'))}  \n"
            f"**Pool remaining:** {pool_remaining}  \n"
            f"**Labelled since last train:** {counts.get('labelled_since_last_train', 0)}"
        )
        right = (
            f"**Labelled total:** {counts.get('labelled_or_verified', 0)}  \n"
            f"**Reviewed total:** {counts.get('total_reviewed', 0)}  \n"
            f"**Unsure:** {counts.get('unsure', 0)}  \n"
            f"**Queued:** {counts.get('queued', 0)}  \n"
            f"**Model:** `{model_id}`  \n"
            f"**Predictions:** `{predictions_id}`"
        )
        return left, right

    def _pool_remaining_summary(
        self,
        session_payload: Mapping[str, Any],
    ) -> str:
        remaining, total, source = self._pool_remaining_counts(session_payload)
        if remaining is None:
            return "—"
        if total is None:
            return f"{remaining}"

        labels = {
            "session_dataset": "session pool",
            "session_metadata": "session pool",
            "prediction_metadata": "prediction metadata",
        }
        return f"{remaining} of {total} ({labels.get(source, 'session pool')})"

    def _pool_remaining_counts(
        self,
        session_payload: Mapping[str, Any],
    ) -> Tuple[Optional[int], Optional[int], str]:
        session = al_state.coerce_session(session_payload)
        excluded = acquisition.session_query_exclude_row_ids(session, {})

        pool_dataset_id = acquisition.session_pool_dataset_id(session)
        total = (
            self._dataset_row_count(pool_dataset_id)
            if pool_dataset_id
            else None
        )
        source = "session_dataset"

        if total is None:
            total = self._session_pool_row_count(session)
            source = "session_metadata"

        if total is None:
            total = self._prediction_pool_row_count(
                session,
                pool_dataset_id=pool_dataset_id,
            )
            source = "prediction_metadata"

        if total is None:
            return None, None, "unknown"

        total = max(0, int(total))
        excluded_count = min(total, len(excluded))
        return total - excluded_count, total, source

    def _session_pool_row_count(
        self,
        session: Mapping[str, Any],
    ) -> Optional[int]:
        """Return the persisted pool count without reading dataset rows."""

        candidates: List[Any] = []

        partition_counts = session.get("partition_counts")
        if isinstance(partition_counts, Mapping):
            candidates.append(partition_counts.get("pool"))

        session_split = session.get("session_split")
        if isinstance(session_split, Mapping):
            counts = session_split.get("counts")
            if isinstance(counts, Mapping):
                candidates.append(counts.get("pool"))

        contract = session.get("contract")
        if isinstance(contract, Mapping):
            contract_split = contract.get("session_split")
            if isinstance(contract_split, Mapping):
                counts = contract_split.get("counts")
                if isinstance(counts, Mapping):
                    candidates.append(counts.get("pool"))

        for value in candidates:
            if value in (None, ""):
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue

        return None

    def _prediction_pool_row_count(
        self,
        session: Mapping[str, Any],
        *,
        pool_dataset_id: str,
    ) -> Optional[int]:
        """Use complete prediction metadata only as a restoration fallback.

        Inline prediction records may be a bounded preview and must never be used
        as the pool denominator.
        """

        artifact_id = str(
            al_state.latest_reference(session, "predictions_artifact_id")
            or self._widget_value("predictions_id", "")
            or ""
        ).strip()
        if not artifact_id:
            return None

        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None

        matches_dataset = getattr(
            acquisition,
            "prediction_payload_matches_dataset",
            None,
        )
        if (
            pool_dataset_id
            and callable(matches_dataset)
            and not matches_dataset(payload, pool_dataset_id)
        ):
            return None

        candidates: List[Any] = [payload.get("row_count")]

        prediction_ref = payload.get("prediction_ref")
        if isinstance(prediction_ref, Mapping):
            candidates.append(prediction_ref.get("row_count"))

        prediction_table = payload.get("prediction_table")
        if isinstance(prediction_table, Mapping):
            candidates.append(prediction_table.get("row_count"))

            storage = prediction_table.get("storage")
            if isinstance(storage, Mapping):
                candidates.append(storage.get("row_count"))

        for value in candidates:
            if value in (None, ""):
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue

        return None

    def _dataset_row_count(self, dataset_id: str) -> Optional[int]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not dataset_id:
            return None
        source = None
        get_source = getattr(datasets, "get_source", None)
        if callable(get_source):
            try:
                source = get_source(dataset_id)
            except Exception:
                source = None
        for owner in (source, datasets):
            if owner is None:
                continue
            for name in ("row_count", "n_rows", "num_rows", "count"):
                value = getattr(owner, name, None)
                if callable(value):
                    for args in ((dataset_id,), ()):  # tolerate manager-style and source-style APIs
                        try:
                            raw = value(*args)
                            if raw not in (None, ""):
                                return int(raw)
                        except Exception:
                            continue
                elif value not in (None, ""):
                    try:
                        return int(value)
                    except Exception:
                        pass
            try:
                return int(len(owner))
            except Exception:
                pass
        return None

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
        self._set_dynamic_pane_object("performance_plot", self._performance_plot_object(metric_rows, metric))
        if status:
            self._set_status("AL performance plot refreshed.")

    def _set_dynamic_pane_object(self, key: str, obj: Any) -> None:
        target = self._widgets.get(key)
        if target is None:
            return
        if pn is not None and hasattr(target, "objects"):
            try:
                target.objects = [pn.panel(obj, sizing_mode="stretch_width")]
                return
            except Exception:
                pass
        if hasattr(target, "object"):
            try:
                target.object = obj
            except Exception:
                pass

    def _dedupe_performance_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        # The same training result can reach the panel twice: once from AL session
        # history and again from the latest referenced core.ml artifact/event.  Some
        # paths assign different round ids before the session revision is finalised,
        # which previously produced two y-values at the same labelled-count x-value.
        # A performance curve should have one point per metric/training-set size.
        by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
        by_exact_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for source_index, row in enumerate(rows):
            try:
                round_index = int(row.get("round") or 0)
                labelled_count = int(row.get("labelled_count") or 0)
                metric = str(row.get("metric") or "")
                value = float(row.get("value") or 0.0)
            except Exception:
                continue
            item = dict(row)
            item["round"] = round_index
            item["labelled_count"] = labelled_count
            item["metric"] = metric
            item["value"] = value
            item["_source_index"] = source_index
            by_exact_key[(round_index, labelled_count, metric)] = item

        for item in by_exact_key.values():
            key = (int(item.get("labelled_count") or 0), str(item.get("metric") or ""))
            existing = by_key.get(key)
            if existing is None or self._prefer_performance_row(item, existing):
                by_key[key] = item

        out: List[Dict[str, Any]] = []
        for item in by_key.values():
            clean = dict(item)
            clean.pop("_source_index", None)
            out.append(clean)
        return sorted(out, key=lambda row: (int(row.get("labelled_count") or 0), int(row.get("round") or 0), str(row.get("metric") or "")))

    def _prefer_performance_row(self, candidate: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
        def timestamp_value(row: Mapping[str, Any]) -> float:
            raw = row.get("timestamp") or row.get("updated_at") or row.get("created_at")
            try:
                return float(raw)
            except Exception:
                pass
            if isinstance(raw, str):
                try:
                    from datetime import datetime

                    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                except Exception:
                    return 0.0
            return 0.0

        candidate_rank = (
            timestamp_value(candidate),
            int(candidate.get("round") or 0),
            int(candidate.get("_source_index") or 0),
        )
        existing_rank = (
            timestamp_value(existing),
            int(existing.get("round") or 0),
            int(existing.get("_source_index") or 0),
        )
        return candidate_rank >= existing_rank

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
            plot.scatter("labelled_count", "value", source=source, size=8, marker="circle")
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
        # Avoid plotting automatically after dataset-scope changes; plotting may
        # require reading sampled numeric columns from a large table.

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
        """Return likely plottable columns without loading the full dataset.

        Large catalogues can be millions of rows by hundreds of columns.  The
        previous implementation called ``get_df(dataset_id)`` and coerced every
        column to numeric just to fill the X/Y dropdowns.  That is a large eager
        pandas materialisation after dataset mapping.  Use metadata/schema when
        available, otherwise expose the full column list and validate only the
        selected X/Y columns when the user explicitly refreshes the plot.
        """

        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return []
        cached = self._plottable_columns_cache.get(dataset_id)
        if cached is not None:
            return list(cached)

        source = self._dataset_source(dataset_id)
        typed_columns: List[str] = []
        schema = getattr(source, "schema", None)
        candidates = []
        for attr_name in ("dtypes", "types", "column_types"):
            raw = getattr(source, attr_name, None)
            try:
                raw = raw() if callable(raw) else raw
            except Exception:
                raw = None
            if isinstance(raw, Mapping):
                candidates.append(raw)
        schema_dtypes = getattr(schema, "dtypes", None) if schema is not None else None
        if isinstance(schema_dtypes, Mapping):
            candidates.append(schema_dtypes)
        for dtypes in candidates:
            for column, dtype in dtypes.items():
                text = str(dtype).lower()
                if any(token in text for token in ("int", "float", "double", "decimal", "number", "numeric")):
                    typed_columns.append(str(column))
        columns = typed_columns or self._dataset_columns(dataset_id)
        self._plottable_columns_cache[dataset_id] = list(columns)
        return list(columns)

    def _numeric_dataset_columns(self, dataset_id: str) -> List[str]:
        # Compatibility wrapper retained for older callers.
        return self._plottable_dataset_columns(dataset_id)

    def _dataset_df(self, dataset_id: str, columns: Optional[Sequence[str]] = None, *, max_rows: Optional[int] = None) -> Any:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None or not dataset_id:
            return None
        requested_columns = [str(column) for column in (columns or []) if column not in (None, "")]
        try:
            if requested_columns:
                df = datasets.get_df(dataset_id, columns=requested_columns)
            else:
                df = datasets.get_df(dataset_id)
        except TypeError:
            try:
                df = datasets.get_df(dataset_id)
            except Exception:
                return None
        except Exception:
            return None
        if max_rows is not None:
            try:
                if len(df) > int(max_rows):
                    return df.sample(n=int(max_rows), random_state=13)
            except Exception:
                try:
                    return df.head(int(max_rows))
                except Exception:
                    pass
        return df

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

        max_points = int(self._widget_value("xy_max_points", XY_DEFAULT_MAX_POINTS) or XY_DEFAULT_MAX_POINTS)
        max_points = max(100, max_points)
        # Load only the columns required for the requested plot.  Do not read a
        # full 300-column catalogue merely to draw two axes.
        frames: List[Any] = []
        for role, role_dataset_id, marker in datasets:
            id_column = ""
            try:
                id_column = acquisition.resolve_record_id_column(self.context, role_dataset_id) or ""
            except Exception:
                id_column = ""
            required_columns = [x_col, y_col]
            if id_column and id_column not in required_columns:
                required_columns.append(id_column)
            label_column = self._session_label_column()
            if label_column and label_column not in required_columns:
                required_columns.append(label_column)
            # Prediction-table datasets may contain these lightweight diagnostic
            # columns; include them only if present in metadata so narrow source
            # readers do not error on missing columns.
            available = set(self._dataset_columns(role_dataset_id))
            for pred_col in (
                "pred_label", "predicted_label", "pred_class", "predicted_class",
                "pred_confidence", "confidence", "pred_entropy",
                "pred_least_confidence", "pred_margin_uncertainty", "pred_true_label",
                "pred_correct",
            ):
                if pred_col in available and pred_col not in required_columns:
                    required_columns.append(pred_col)
            df = self._dataset_df(role_dataset_id, columns=required_columns, max_rows=max_points * 4)
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

        show_trained = bool(self._widget_value("xy_show_trained", True))
        special_ids = (training_ids if show_trained else set()) | query_ids
        special = work_all[work_all["_al_row_id"].isin(special_ids)]
        base = work_all
        if len(base) > max_points:
            non_special = base[~base["_al_row_id"].isin(special_ids)]
            keep_n = max(0, max_points - len(special))
            if keep_n > 0 and len(non_special) > keep_n:
                non_special = non_special.sample(n=keep_n, random_state=13)
            base = pd.concat([non_special, special], ignore_index=False).drop_duplicates(subset=["_al_dataset_role", "_al_row_id"])

        colour_by = str(self._widget_value("xy_colour", "prediction_correctness") or "prediction_correctness")
        strategy_score_id = self._strategy_colour_id(colour_by)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))

        if colour_by == "prediction_correctness":
            self._scatter_correctness_groups(ax, base, x_col, y_col, label_map, prediction_map)
        elif strategy_score_id:
            score_map = self._strategy_score_map(session, strategy_score_id)
            self._scatter_strategy_score_map(ax, base, x_col, y_col, score_map, strategy_score_id)
        else:
            for role, marker in (("pool", "o"), ("validation", "^")):
                sub = base[base["_al_dataset_role"] == role]
                if not sub.empty:
                    ax.scatter(sub[x_col], sub[y_col], s=3, alpha=0.08, marker=marker, label=f"{role} rows")

        trained = work_all[work_all["_al_row_id"].isin(training_ids)] if show_trained else work_all.iloc[0:0]
        if show_trained and not trained.empty:
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
                elif strategy_score_id:
                    values.append("queried")
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

    def _strategy_colour_id(self, colour_by: str) -> str:
        prefix = "strategy_score:"
        value = str(colour_by or "")
        return value[len(prefix):].strip() if value.startswith(prefix) else ""

    def _strategy_scores_artifact_id(self, session: Mapping[str, Any]) -> str:
        latest = dict((session or {}).get("latest") or {})
        return str(latest.get("strategy_scores_artifact_id") or self.strategy_scores_artifact_id or "").strip()

    def _strategy_score_map(self, session: Mapping[str, Any], strategy_id: str) -> Dict[str, float]:
        strategy_id = str(strategy_id or "").strip()
        artifact_id = self._strategy_scores_artifact_id(session)
        if not strategy_id or not artifact_id:
            return {}
        try:
            payload = self.context.artifacts.get(artifact_id)
        except Exception:
            return {}
        if not isinstance(payload, Mapping):
            return {}
        records = []
        by_strategy = payload.get("by_strategy")
        if isinstance(by_strategy, Mapping):
            records = by_strategy.get(strategy_id) or []
        if not records:
            records = [row for row in (payload.get("records") or []) if isinstance(row, Mapping) and str(row.get("strategy_id") or "") == strategy_id]
        out: Dict[str, float] = {}
        for record in records:
            if not isinstance(record, Mapping):
                continue
            row_id = str(record.get("row_id") or record.get("id") or "").strip()
            if not row_id:
                continue
            for key in ("score", "informativeness_score", "active_learning_score"):
                try:
                    value = record.get(key)
                    if value not in (None, ""):
                        out[row_id] = float(value)
                        break
                except Exception:
                    continue
        return out

    def _scatter_strategy_score_map(self, ax: Any, df: Any, x_col: str, y_col: str, score_map: Mapping[str, float], strategy_id: str) -> None:
        import math

        if df.empty:
            return
        values: List[float] = []
        valid_mask: List[bool] = []
        for row_id in df["_al_row_id"]:
            try:
                value = float(score_map.get(str(row_id), float("nan")))
                is_valid = math.isfinite(value)
            except Exception:
                value = float("nan")
                is_valid = False
            values.append(value)
            valid_mask.append(is_valid)
        if any(valid_mask):
            valid_df = df[valid_mask]
            valid_values = [value for value, keep in zip(values, valid_mask) if keep]
            scatter = ax.scatter(valid_df[x_col], valid_df[y_col], c=valid_values, s=5, alpha=0.5, label=f"{strategy_id} score")
            try:
                ax.figure.colorbar(scatter, ax=ax, label=f"{strategy_id} informativeness score")
            except Exception:
                pass
            missing = df[[not keep for keep in valid_mask]]
            if not missing.empty:
                ax.scatter(missing[x_col], missing[y_col], s=3, alpha=0.06, label="rows without calculated QS score")
        else:
            for role, marker in (("pool", "o"), ("validation", "^")):
                sub = df[df["_al_dataset_role"] == role]
                if not sub.empty:
                    ax.scatter(sub[x_col], sub[y_col], s=3, alpha=0.08, marker=marker, label=f"{role} rows")
            ax.text(
                0.5,
                0.96,
                f"No whole-pool scores for `{strategy_id}` yet. Click 'Calculate QS scores over pool'.",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize="small",
            )

    def _scatter_query_overlay(self, ax: Any, df: Any, x_col: str, y_col: str, values: Sequence[Any], colour_by: str) -> None:
        if colour_by in {"last_query_score", "last_query_rank"}:
            import math
            numeric = []
            for value in values:
                try:
                    number = float(value)
                    numeric.append(number if math.isfinite(number) else float("nan"))
                except Exception:
                    numeric.append(float("nan"))
            valid = [math.isfinite(value) for value in numeric]
            if any(valid):
                plot_df = df[valid]
                plot_values = [value for value, keep in zip(numeric, valid) if keep]
                scatter = ax.scatter(plot_df[x_col], plot_df[y_col], c=plot_values, s=12, alpha=0.75, label="latest query")
                try:
                    label = "informativeness score" if colour_by == "last_query_score" else "query rank"
                    ax.figure.colorbar(scatter, ax=ax, label=label)
                except Exception:
                    pass
                missing = df[[not keep for keep in valid]]
                if not missing.empty:
                    ax.scatter(missing[x_col], missing[y_col], s=8, alpha=0.35, label="latest query: no score")
            else:
                ax.scatter(df[x_col], df[y_col], s=8, alpha=0.65, label="latest query: no numeric score")
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
            score = None
            for key in (
                "informativeness_score",
                "active_learning_score",
                "score",
                "acquisition_score",
                "uncertainty",
                "entropy",
                "least_confidence",
                "margin_uncertainty",
                "learning_loss",
            ):
                value = item.get(key)
                if value not in (None, ""):
                    score = value
                    break
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
        elif key in {"score_all_btn", "query_btn", "label_btn", "bulk_label_btn"}:
            try:
                widget.disabled = bool(busy)
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