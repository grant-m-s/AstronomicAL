from __future__ import annotations

from typing import Any, Mapping

from astronomicAL.platform.plugins.specs import ActionRequest

from . import state as al_state
from .panel import ActiveLearningPanel


class JobBackedActiveLearningPanel(ActiveLearningPanel):
    """Active Learning panel with background session creation.

    The base panel predates the job-backed start-session action and invokes the
    handler directly. This subclass keeps the existing panel implementation and
    routes only session creation through ``PluginManager.run_action()`` so the
    full split scan and Parquet writes execute through ``JobManager``.
    """

    def _run_start(self) -> None:
        self._set_status(
            "Creating the Active Learning partitions in the background and "
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
                self._set_status(f"Active Learning session creation failed: {exc}")

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
            self._set_status(f"Active Learning session creation failed: {exc}")

    def _start_session_request(self) -> ActionRequest:
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

        return ActionRequest(
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
                "partition_whole_dataset": True,
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
        counts = dict(
            result.get("partition_counts")
            or (result.get("session_split") or {}).get("counts")
            or {}
        )
        partition_text = self._partition_count_text(counts)
        suffix = f" {partition_text}" if partition_text else ""
        self._set_status(
            f"Started {self.task_type} session "
            f"`{result.get('session_id')}` with {count} initial review rows."
            f"{suffix} Review tab selected."
        )

    @staticmethod
    def _partition_count_text(counts: Mapping[str, Any]) -> str:
        parts = []
        for key, title in (
            ("pool", "Pool"),
            ("validation", "validation"),
            ("test", "test"),
        ):
            value = counts.get(key)
            if value in (None, ""):
                continue
            try:
                parts.append(f"{title}: {int(value):,}")
            except (TypeError, ValueError):
                continue
        return "; ".join(parts) + "." if parts else ""