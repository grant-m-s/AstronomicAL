from __future__ import annotations

from typing import Any, Mapping, Optional

import json
import panel as pn

from . import state as al_state
from .panel import ActiveLearningPanel


def _new_action_request(**kwargs: Any) -> Any:
    from astronomicAL.platform.plugins.specs import ActionRequest

    return ActionRequest(**kwargs)


TRAIN_ACTION_ID = "core.active_learning.train_from_session"
TRAINING_CONTROL_SERVICE = "core.active_learning.training_controls"


class JobBackedActiveLearningPanel(ActiveLearningPanel):
    """Active Learning panel using platform jobs and Core ML pause controls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        restore_state = kwargs.get("restore_state")
        restored = self._restored_widget_values(restore_state)
        self._active_training_handle: Any = None
        self._active_training_job_id: Optional[str] = None
        self._training_control_id: Optional[str] = None
        self._resume_params: dict[str, Any] = {
            key: str(restored[key])
            for key in (
                "resume_checkpoint_artifact_id",
                "resume_manifest_path",
                "resume_checkpoint_path",
            )
            if restored.get(key) not in (None, "")
        }
        self._pending_resume_params: dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        self._build_protocol_widgets(restore_state)
        self._build_training_widgets()

    # ------------------------------------------------------------------
    # Explicit base-panel extension points
    # ------------------------------------------------------------------

    def _session_protocol_controls_view(self) -> Any:
        return list(self.protocol_controls)

    def _training_controls_view(self, train_button: Any) -> Any:
        controls = self._compact_row(
            train_button,
            self.pause_train_button,
            self.resume_train_button,
            self.cancel_train_button,
        )
        return [
            controls,
            self.training_control_status,
            self.resume_diagnostics,
        ]

    def _register_widget(self, name: str, widget: Any) -> None:
        setattr(self, name, widget)
        widgets = getattr(self, "_widgets", None)
        if isinstance(widgets, dict):
            widgets[name] = widget

    def get_state(self) -> dict[str, Any]:
        state = dict(super().get_state())
        state.update(
            {
                "validation_source": str(self.validation_source.value),
                "validation_dataset_id": str(
                    self.validation_dataset_id.value or ""
                ),
                "validation_fraction": float(self.validation_fraction.value),
                "test_source": str(self.test_source.value),
                "test_dataset_id": str(self.test_dataset_id.value or ""),
                "test_fraction": float(self.test_fraction.value),
            }
        )
        state.update(self._resume_params)
        return state

    # ------------------------------------------------------------------
    # Session protocol controls
    # ------------------------------------------------------------------

    def _build_protocol_widgets(self, restore_state: Any) -> None:
        restored = self._restored_widget_values(restore_state)
        dataset_options = self._holdout_dataset_options()

        self.validation_source = pn.widgets.Select(
            name="Validation source",
            options={
                "Split from training pool": "split",
                "Use existing dataset": "dataset",
            },
            value=self._valid_source(restored.get("validation_source")),
            sizing_mode="stretch_width",
        )
        self.validation_dataset_id = pn.widgets.Select(
            name="Validation dataset",
            options=dataset_options,
            value=self._valid_option(
                restored.get("validation_dataset_id"),
                dataset_options,
            ),
            sizing_mode="stretch_width",
        )
        self.validation_fraction = pn.widgets.FloatInput(
            name="Validation fraction",
            value=float(restored.get("validation_fraction") or 0.1),
            start=0.000001,
            end=0.999999,
            step=0.01,
            sizing_mode="stretch_width",
        )
        self.test_source = pn.widgets.Select(
            name="Test source",
            options={
                "Split from training pool": "split",
                "Use existing dataset": "dataset",
            },
            value=self._valid_source(restored.get("test_source")),
            sizing_mode="stretch_width",
        )
        self.test_dataset_id = pn.widgets.Select(
            name="Test dataset",
            options=dataset_options,
            value=self._valid_option(
                restored.get("test_dataset_id"),
                dataset_options,
            ),
            sizing_mode="stretch_width",
        )
        self.test_fraction = pn.widgets.FloatInput(
            name="Test fraction",
            value=float(restored.get("test_fraction") or 0.2),
            start=0.000001,
            end=0.999999,
            step=0.01,
            sizing_mode="stretch_width",
        )

        for name, widget in (
            ("validation_source", self.validation_source),
            ("validation_dataset_id", self.validation_dataset_id),
            ("validation_fraction", self.validation_fraction),
            ("test_source", self.test_source),
            ("test_dataset_id", self.test_dataset_id),
            ("test_fraction", self.test_fraction),
        ):
            self._register_widget(name, widget)

        self.validation_source.param.watch(
            lambda *_: self._sync_protocol_visibility(),
            "value",
        )
        self.test_source.param.watch(
            lambda *_: self._sync_protocol_visibility(),
            "value",
        )
        self._sync_protocol_visibility()

        protocol_help = pn.pane.Markdown(
            "### Validation and test data\n"
            "Choose each holdout independently. When both use existing "
            "datasets, every row in the selected training dataset remains in "
            "the Active Learning pool. Existing datasets are referenced "
            "without copying or re-registering them.",
            sizing_mode="stretch_width",
            height_policy="auto",
            margin=(4, 0, 6, 0),
            styles={
                "flex": "0 0 auto",
                "height": "auto",
                "min-height": "0",
                "max-height": "none",
                "overflow": "visible",
            },
        )
        validation_row = pn.Row(
            self.validation_source,
            self.validation_fraction,
            self.validation_dataset_id,
            sizing_mode="stretch_width",
            height_policy="auto",
            margin=(0, 0, 4, 0),
            styles={
                "flex": "0 0 auto",
                "height": "auto",
                "min-height": "0",
                "max-height": "none",
                "align-items": "flex-start",
                "overflow": "visible",
            },
        )
        test_row = pn.Row(
            self.test_source,
            self.test_fraction,
            self.test_dataset_id,
            sizing_mode="stretch_width",
            height_policy="auto",
            margin=(0, 0, 6, 0),
            styles={
                "flex": "0 0 auto",
                "height": "auto",
                "min-height": "0",
                "max-height": "none",
                "align-items": "flex-start",
                "overflow": "visible",
            },
        )
        self.protocol_controls = [
            protocol_help,
            validation_row,
            test_row,
        ]

    def _sync_protocol_visibility(self) -> None:
        validation_is_dataset = self.validation_source.value == "dataset"
        test_is_dataset = self.test_source.value == "dataset"
        self.validation_dataset_id.visible = validation_is_dataset
        self.validation_fraction.visible = not validation_is_dataset
        self.test_dataset_id.visible = test_is_dataset
        self.test_fraction.visible = not test_is_dataset

    def _holdout_dataset_options(self) -> Mapping[str, str]:
        return self._dataset_options_with_blank("Select an existing dataset")

    def _refresh_dataset_options(self) -> None:
        options = self._holdout_dataset_options()
        values = set(options.values())
        for widget in (self.validation_dataset_id, self.test_dataset_id):
            current = str(widget.value or "")
            widget.options = options
            widget.value = current if current in values else ""

    def refresh_dataset_controls(self, *, status: bool = True) -> None:
        super().refresh_dataset_controls(status=status)
        if hasattr(self, "validation_dataset_id"):
            self._refresh_dataset_options()

    @staticmethod
    def _valid_source(value: Any) -> str:
        resolved = str(value or "split").strip().lower()
        return resolved if resolved in {"split", "dataset"} else "split"

    @staticmethod
    def _valid_option(value: Any, options: Mapping[str, str]) -> str:
        resolved = str(value or "")
        return resolved if resolved in set(options.values()) else ""

    @staticmethod
    def _restored_widget_values(restore_state: Any) -> dict[str, Any]:
        if not isinstance(restore_state, Mapping):
            return {}
        nested = restore_state.get("widgets")
        if isinstance(nested, Mapping):
            values = dict(nested)
            values.update(
                {
                    key: value
                    for key, value in restore_state.items()
                    if key not in values
                }
            )
            return values
        return dict(restore_state)

    # ------------------------------------------------------------------
    # Background session creation
    # ------------------------------------------------------------------

    def _run_start(self) -> None:
        self._set_status(
            "Creating the Active Learning data protocol in the background and "
            "drawing the initial random review batch..."
        )
        self._set_button_busy("start_btn", True)

        try:
            request = self._start_session_request()
        except Exception as exc:
            self._set_button_busy("start_btn", False)
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
                self._set_button_busy("start_btn", False)
                self._apply_start_session_result(payload)

            self._next_tick(update)

        def failed(exc: BaseException) -> None:
            if handle is not None:
                self._discard_job_handle(handle)

            def update() -> None:
                if self._disposed:
                    return
                self._set_button_busy("start_btn", False)
                self._set_status(
                    f"Active Learning session creation failed: {exc}"
                )

            self._next_tick(update)

        try:
            handle = self._run_plugin_action(
                "core.active_learning.start_session",
                request,
                on_done=done,
                on_error=failed,
            )
        except Exception as exc:
            self._set_button_busy("start_btn", False)
            self._set_status(
                f"Active Learning session creation failed: {exc}"
            )

    def _start_session_request(self) -> Any:
        self._refresh_dataset_options()
        dataset_id = str(self._widget_value("dataset_id", "") or "").strip()
        label_column = str(
            self._widget_value("label_column", "") or ""
        ).strip()
        labels = [
            str(value)
            for value in (self._widget_value("labels", []) or [])
            if str(value).strip()
        ]
        if not dataset_id:
            raise ValueError(
                "Select a dataset before starting an active-learning session."
            )
        if not label_column:
            raise ValueError(
                "Select a label column so the target contract can be inferred."
            )
        if self.task_type != al_state.TASK_REGRESSION and not labels:
            raise ValueError(
                "Select at least one class label for the active-learning session."
            )

        validation_source = str(self.validation_source.value)
        test_source = str(self.test_source.value)
        validation_dataset_id = str(
            self.validation_dataset_id.value or ""
        ).strip()
        test_dataset_id = str(self.test_dataset_id.value or "").strip()
        if validation_source == "dataset" and not validation_dataset_id:
            raise ValueError("Select an existing validation dataset.")
        if test_source == "dataset" and not test_dataset_id:
            raise ValueError("Select an existing test dataset.")
        if validation_dataset_id and validation_dataset_id == dataset_id:
            raise ValueError(
                "The validation dataset must differ from the training pool."
            )
        if test_dataset_id and test_dataset_id == dataset_id:
            raise ValueError(
                "The test dataset must differ from the training pool."
            )
        if (
            validation_source == "dataset"
            and test_source == "dataset"
            and validation_dataset_id == test_dataset_id
        ):
            raise ValueError(
                "Validation and test must use different existing datasets."
            )

        return _new_action_request(
            dataset_id=dataset_id,
            row_ids=None,
            columns=[],
            params={
                "dataset_id": dataset_id,
                "label_options": (
                    [] if self.task_type == al_state.TASK_REGRESSION else labels
                ),
                "task_type": self.task_type,
                "problem_type": self.task_type,
                "label_profile": dict(self.label_profile or {}),
                "target_column": label_column,
                "label_column": label_column,
                "infer_labels_from_column": True,
                "initial_k": int(self._widget_value("initial_k", 20) or 0),
                "seed": int(self._widget_value("seed", 42) or 42),
                "make_selection": True,
                "validation_source": validation_source,
                "validation_dataset_id": (
                    validation_dataset_id
                    if validation_source == "dataset"
                    else ""
                ),
                "test_source": test_source,
                "test_dataset_id": (
                    test_dataset_id if test_source == "dataset" else ""
                ),
                "session_validation_size": float(
                    self.validation_fraction.value
                ),
                "session_test_size": float(self.test_fraction.value),
            },
            artifact_id=None,
            origin="core.active_learning.panel",
        )

    def _apply_start_session_result(self, result: Mapping[str, Any]) -> None:
        self._set_session_id(result.get("session_artifact_id"))
        self._refresh_session_summary(status=False)
        self._refresh_performance(status=False)

        row_ids = [
            str(row_id)
            for row_id in (result.get("initial_row_ids") or [])
            if str(row_id).strip()
        ]
        focused_row_id = str(result.get("focused_row_id") or "").strip()
        review_row_id = focused_row_id or (row_ids[0] if row_ids else "")
        if review_row_id:
            self._set_review_row(review_row_id)
        self._select_tab("Review")

        count = int(result.get("count", len(row_ids)) or 0)
        counts = dict(result.get("partition_counts") or {})
        protocol_text = self._protocol_count_text(
            counts,
            validation_source=str(result.get("validation_source") or ""),
            test_source=str(result.get("test_source") or ""),
        )
        suffix = f" {protocol_text}" if protocol_text else ""
        self._set_status(
            f"Started {self.task_type} session "
            f"`{result.get('session_id')}` with {count} initial review rows."
            f"{suffix} Review tab selected."
        )

    @staticmethod
    def _protocol_count_text(
        counts: Mapping[str, Any],
        *,
        validation_source: str,
        test_source: str,
    ) -> str:
        parts = []
        for key, title, source in (
            ("pool", "Pool", "session"),
            ("validation", "validation", validation_source),
            ("test", "test", test_source),
        ):
            value = counts.get(key)
            if value in (None, ""):
                continue
            try:
                count = f"{int(value):,}"
            except (TypeError, ValueError):
                continue
            suffix = " existing" if source == "dataset" else ""
            parts.append(f"{title}: {count}{suffix}")
        return "; ".join(parts) + "." if parts else ""

    # ------------------------------------------------------------------
    # Training pause, resume, and cancellation
    # ------------------------------------------------------------------

    def _build_training_widgets(self) -> None:
        self.pause_train_button = pn.widgets.Button(
            name="Pause after epoch",
            button_type="warning",
            disabled=False,
            width=170,
            height=34,
        )
        self.resume_train_button = pn.widgets.Button(
            name="Resume paused",
            button_type="primary",
            disabled=False,
            width=150,
            height=34,
        )
        self.cancel_train_button = pn.widgets.Button(
            name="Cancel training",
            button_type="danger",
            disabled=False,
            width=150,
            height=34,
        )
        self.training_control_status = pn.pane.Markdown(
            (
                "A paused checkpoint is available."
                if self._resume_params
                else (
                    "Training controls are ready. Pause and Cancel report when no "
                    "job is active; Resume reports when no paused checkpoint exists."
                )
            ),
            sizing_mode="stretch_width",
            height_policy="auto",
            margin=(2, 0, 6, 0),
            styles={
                "flex": "0 0 auto",
                "height": "auto",
                "min-height": "0",
                "max-height": "none",
                "overflow": "visible",
            },
        )
        self.resume_diagnostics = pn.widgets.TextAreaInput(
            name="Full resume diagnostics — copy this complete text",
            value="",
            visible=False,
            height=420,
            sizing_mode="stretch_width",
            margin=(4, 0, 8, 0),
        )
        self.pause_train_button.on_click(self._pause_training)
        self.resume_train_button.on_click(self._resume_training)
        self.cancel_train_button.on_click(self._cancel_training)
        self._set_training_control_state(running=False)

    def _sync_train_controls(self) -> None:
        super()._sync_train_controls()
        button = self._widgets.get("train_btn")
        if button is not None and self._active_training_handle is not None:
            button.disabled = True

    def _run_plugin_action(
        self,
        action_id: str,
        request: Any,
        *,
        on_done: Any = None,
        on_error: Any = None,
        **kwargs: Any,
    ):
        if action_id != TRAIN_ACTION_ID:
            return super()._run_plugin_action(
                action_id,
                request,
                on_done=on_done,
                on_error=on_error,
                **kwargs,
            )
        if self._active_training_handle is not None:
            raise RuntimeError(
                "An Active Learning training job is already running."
            )

        registry = self._training_controls()
        control_id, _control = registry.create()
        self._training_control_id = control_id
        params = dict(request.params or {})
        params["training_control_id"] = control_id
        if self._pending_resume_params:
            params.update(self._pending_resume_params)
            self._pending_resume_params = {}
        controlled_request = _new_action_request(
            dataset_id=request.dataset_id,
            row_ids=request.row_ids,
            columns=list(request.columns or []),
            params=params,
            artifact_id=request.artifact_id,
            origin=request.origin,
        )

        submission_finished = False

        def completed(result: Any) -> None:
            nonlocal submission_finished
            submission_finished = True
            payload = dict(result or {}) if isinstance(result, Mapping) else {}
            status = str(
                payload.get("workflow_status") or payload.get("status") or ""
            ).lower()
            self._capture_resume_params(payload)
            if payload.get("session_artifact_id"):
                self._set_session_id(payload.get("session_artifact_id"))
            self._finish_training_control(payload=payload)

            if status in {"paused", "cancelled"}:
                def update_interrupted() -> None:
                    if self._disposed:
                        return
                    self._set_button_busy("train_btn", False)
                    self._refresh_session_summary(status=False)
                    self._refresh_performance(status=False)
                    if status == "paused":
                        self._set_status(
                            "Training paused at a safe epoch boundary. "
                            "Use Resume paused to continue from the saved checkpoint."
                        )
                    else:
                        self._set_status("Training cancelled.")

                self._next_tick(update_interrupted)
                return

            if callable(on_done):
                on_done(result)

        def failed(exc: BaseException) -> None:
            nonlocal submission_finished
            submission_finished = True
            payload = getattr(exc, "pause_payload", None)
            if isinstance(payload, Mapping):
                self._capture_resume_params(payload)
            cancelled = type(exc).__name__ in {
                "CancelledError",
                "MLRecipeCancelled",
                "JobCancelled",
            }
            self._finish_training_control(
                error=exc,
                cancelled=cancelled,
            )
            failure_text = self._resume_failure_text(exc)
            failure_plain_text = self._resume_failure_plain_text(exc)
            failure_payload = getattr(exc, "failure_payload", None)
            has_resume_diagnostics = (
                isinstance(failure_payload, Mapping)
                and isinstance(
                    failure_payload.get("resume_debug"),
                    Mapping,
                )
            )

            def update_failure_status() -> None:
                if self._disposed:
                    return
                self._set_button_busy("train_btn", False)
                self.training_control_status.object = failure_text
                self.resume_diagnostics.value = failure_plain_text
                self.resume_diagnostics.visible = has_resume_diagnostics
                self._set_status(failure_text)
                self._refresh_session_summary(status=False)
                self._refresh_performance(status=False)

            self._next_tick(update_failure_status)

            # The base callback replaces the detailed report with a generic
            # one-line status. Detailed resume failures already perform the
            # required cleanup above, so do not invoke that callback.
            if not has_resume_diagnostics and callable(on_error):
                on_error(exc)

        try:
            handle = super()._run_plugin_action(
                action_id,
                controlled_request,
                on_done=completed,
                on_error=failed,
                **kwargs,
            )
        except Exception:
            registry.release(control_id)
            self._training_control_id = None
            raise

        if not submission_finished:
            self.resume_diagnostics.value = ""
            self.resume_diagnostics.visible = False
            self._active_training_handle = handle
            self._active_training_job_id = str(
                getattr(handle, "job_id", None)
                or getattr(handle, "id", None)
                or ""
            ) or None
            self._set_training_control_state(running=True)
        return handle

    def _pause_training(self, *_: Any) -> None:
        if (
            self._active_training_handle is None
            or not self._training_control_id
        ):
            message = "No Active Learning training job is currently running."
            self.training_control_status.object = message
            self._set_status(message)
            return
        try:
            requested = self._training_controls().request_pause(
                self._training_control_id,
                "Pause requested from the Active Learning Train tab.",
            )
        except Exception as exc:
            self.training_control_status.object = (
                f"Could not request a pause: {exc}"
            )
            return
        if not requested:
            self.training_control_status.object = (
                "The training pause control is no longer active."
            )
            return
        self.training_control_status.object = (
            "Pause requested. The current epoch, validation pass, and scheduler "
            "update will finish before Core ML saves a resumable checkpoint."
        )

    def _cancel_training(self, *_: Any) -> None:
        handle = self._active_training_handle
        if handle is None:
            message = "No Active Learning training job is currently running."
            self.training_control_status.object = message
            self._set_status(message)
            return
        try:
            handle.cancel()
        except Exception:
            token = getattr(handle, "token", None)
            cancel = getattr(token, "cancel", None)
            if callable(cancel):
                cancel()
        self.training_control_status.object = (
            "Cancellation requested. Training will stop at the next safe "
            "cancellation point."
        )

    def _resume_training(self, *_: Any) -> None:
        if self._active_training_handle is not None:
            message = "A training job is already running."
            self.training_control_status.object = message
            self._set_status(message)
            return
        if not self._resume_params:
            message = "No paused checkpoint is available from this session."
            self.training_control_status.object = message
            self._set_status(message)
            return
        self._pending_resume_params = {
            **dict(self._resume_params),
            "resume_al_training": True,
        }
        self.training_control_status.object = (
            "Submitting the paused checkpoint with its exact saved training, "
            "validation, and test partitions."
        )
        run_train = getattr(self, "_run_train", None)
        if not callable(run_train):
            raise RuntimeError(
                "The Active Learning panel does not expose its training action."
            )
        run_train()

    def _capture_resume_params(self, payload: Mapping[str, Any]) -> None:
        params: dict[str, Any] = {}
        for key in (
            "resume_checkpoint_artifact_id",
            "resume_manifest_path",
            "resume_checkpoint_path",
        ):
            value = self._find_nested(payload, key)
            if value not in (None, ""):
                params[key] = str(value)
        if params:
            self._resume_params = params

    @classmethod
    def _find_nested(cls, value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            if value.get(key) not in (None, ""):
                return value.get(key)
            for nested in value.values():
                found = cls._find_nested(nested, key)
                if found not in (None, ""):
                    return found
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found = cls._find_nested(nested, key)
                if found not in (None, ""):
                    return found
        return None


    @staticmethod
    def _debug_code_block(value: Any, *, limit: int = 12000) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                text = repr(value)
        if len(text) > limit:
            text = text[:limit] + "\n... <truncated>"
        return "```json\n" + text.replace("```", "` ` `") + "\n```"

    @staticmethod
    def _resume_failure_plain_text(exc: BaseException) -> str:
        payload = getattr(exc, "failure_payload", None)
        if not isinstance(payload, Mapping):
            return f"Training failed: {exc}"

        error = str(payload.get("error") or exc)
        status = str(payload.get("resume_debug_status") or "").strip()
        report = str(payload.get("resume_debug_report") or "").strip()
        parts = [f"Training failed: {error}"]
        if status:
            parts.append(status)
        if report:
            parts.append(report)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _resume_failure_text(exc: BaseException) -> str:
        payload = getattr(exc, "failure_payload", None)
        if not isinstance(payload, Mapping):
            return f"Training failed: {exc}"

        debug = payload.get("resume_debug")
        status = str(payload.get("resume_debug_status") or "").strip()
        full_report = str(
            payload.get("resume_debug_report") or ""
        ).strip()
        lines = [f"**Training failed:** {payload.get('error') or exc}"]
        if status:
            lines.extend(["", f"**Resume preflight:** {status}"])
        if isinstance(debug, Mapping):
            artifact_id = str(
                debug.get("diagnostic_artifact_id") or ""
            ).strip()
            lines.extend(
                [
                    "",
                    "**Resume comparison values**",
                    f"- Exact saved request replayed: "
                    f"`{bool(debug.get('exact_request_replayed'))}`",
                    f"- Training dataset: "
                    f"`{debug.get('request_dataset_id')}`",
                    f"- Training rows: "
                    f"`{debug.get('request_row_count')}`",
                    f"- Row signature: "
                    f"`{debug.get('request_row_ids_sha256')}`",
                    f"- Validation dataset: "
                    f"`{debug.get('validation_dataset_id')}`",
                    f"- Test dataset: "
                    f"`{debug.get('test_dataset_id')}`",
                    f"- Checkpoint: "
                    f"`{debug.get('checkpoint_reference')}`",
                ]
            )
            if artifact_id:
                lines.append(
                    f"- Full diagnostic artifact: `{artifact_id}`"
                )

            request_protocol = (
                debug.get("request_protocol_json")
                or debug.get("request_protocol")
            )
            checkpoint_summary = (
                debug.get("checkpoint_artifact_summary_json")
                or debug.get("checkpoint_artifact_summary")
            )
            exception_chain = (
                debug.get("exception_chain_json")
                or debug.get("exception_chain")
            )
            if full_report:
                lines.extend(
                    [
                        "",
                        "**Complete diagnostic report**",
                        JobBackedActiveLearningPanel._debug_code_block(
                            full_report,
                            limit=40000,
                        ),
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "**Exact protocol submitted to Core ML**",
                        JobBackedActiveLearningPanel._debug_code_block(
                            request_protocol
                        ),
                        "",
                        "**Paused checkpoint partition metadata**",
                        JobBackedActiveLearningPanel._debug_code_block(
                            checkpoint_summary
                        ),
                        "",
                        "**Core ML exception chain and attributes**",
                        JobBackedActiveLearningPanel._debug_code_block(
                            exception_chain
                        ),
                    ]
                )
        return "\n".join(lines)
    def _finish_training_control(
        self,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException] = None,
        cancelled: bool = False,
    ) -> None:
        control_id = self._training_control_id
        if control_id:
            try:
                self._training_controls().release(control_id)
            except Exception:
                pass
        handle = self._active_training_handle
        if handle is not None:
            self._discard_job_handle(handle)
        self._active_training_handle = None
        self._active_training_job_id = None
        self._training_control_id = None

        status = str(
            (payload or {}).get("workflow_status")
            or (payload or {}).get("status")
            or ""
        ).lower()
        if status not in {"paused", "cancelled"} and error is None:
            self._resume_params = {}
        if status == "paused":
            message = (
                "Training paused at a safe epoch boundary. A resumable "
                "checkpoint is available."
            )
        elif status == "cancelled" or cancelled:
            message = "Training cancelled."
        elif error is not None:
            message = f"Training failed: {error}"
        else:
            message = "Training controls are idle."
        self._set_training_control_state(running=False, message=message)

    def _set_training_control_state(
        self,
        *,
        running: bool,
        message: Optional[str] = None,
    ) -> None:
        def apply() -> None:
            # Keep every control clickable so users receive immediate feedback.
            # The handlers enforce whether pause, resume, or cancellation is
            # currently meaningful.
            self.pause_train_button.disabled = False
            self.cancel_train_button.disabled = False
            self.resume_train_button.disabled = False
            train_button = self._widgets.get("train_btn")
            if train_button is not None and running:
                train_button.disabled = True
            if message is not None:
                self.training_control_status.object = message
            elif running:
                self.training_control_status.object = (
                    "Training is running through the platform JobManager. "
                    "Pause waits for the current epoch boundary; Cancel stops at "
                    "the next safe cancellation point."
                )

        # Calls made by button handlers need to update synchronously. Worker
        # completion callbacks are already routed through the panel's next-tick
        # callbacks before they alter the main status/result panes.
        apply()

    def _training_controls(self):
        services = getattr(getattr(self, "context", None), "services", None)
        if services is None:
            raise RuntimeError(
                "Active Learning training controls require the platform "
                "service registry."
            )
        return services.get(TRAINING_CONTROL_SERVICE)

    def _set_session_id(self, value: Any) -> None:
        super()._set_session_id(value)
        self._load_resume_from_session()

    def _load_resume_from_session(self) -> None:
        session_id = str(getattr(self, "session_artifact_id", "") or "")
        session = self._session_payload(session_id)
        if not isinstance(session, Mapping):
            return

        history = list(session.get("history") or [])
        last_training_event = ""
        for item in reversed(history):
            if not isinstance(item, Mapping):
                continue
            event = str(item.get("event") or "")
            if event.startswith("training_"):
                last_training_event = event
                break

        if last_training_event != "training_paused":
            self._resume_params = {}
        else:
            latest = dict(session.get("latest") or {})
            self._resume_params = {}
            self._capture_resume_params(
                {
                    "latest": latest,
                    "history": history[-1:] if history else [],
                }
            )

        if hasattr(self, "resume_train_button"):
            self._set_training_control_state(running=False)

    def dispose(self) -> None:
        handle = self._active_training_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
        if self._training_control_id:
            try:
                self._training_controls().release(self._training_control_id)
            except Exception:
                pass
        self._active_training_handle = None
        self._training_control_id = None
        super().dispose()