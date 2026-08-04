from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

from .serialization import json_safe

PROGRESS_SCHEMA_VERSION = 1
PROGRESS_EVENT = "ml.recipe_run.progress"

_STAGE_LABELS = {
    "initializing": "Initialising run",
    "preflight": "Checking recipe and dataset",
    "partitioning": "Creating dataset partitions",
    "split_manifest": "Saving split manifest",
    "normalization": "Calculating training normalisation",
    "building_model": "Building model",
    "configuring_training": "Configuring training",
    "loading_data": "Preparing data access",
    "restoring": "Restoring paused state",
    "output_validation": "Validating model output",
    "preprocessing": "Fitting preprocessing",
    "external_memory": "Building external-memory data",
    "training": "Training model",
    "validation": "Evaluating validation data",
    "selecting_checkpoint": "Selecting best checkpoint",
    "test_evaluation": "Evaluating test data",
    "saving_model": "Saving trained model",
    "saving_checkpoint": "Saving resumable checkpoint",
    "saving_evaluation": "Saving evaluation report",
    "saving_predictions": "Saving predictions",
    "finalizing": "Finalising run",
    "complete": "Run complete",
    "paused": "Run paused",
    "cancelling": "Stopping run",
    "cancelled": "Run cancelled",
    "failed": "Run failed",
}

_TERMINAL_STATUSES = {"complete", "completed", "finished", "paused", "cancelled", "failed", "error"}

def stage_label(stage: str) -> str:
    value = str(stage or "running").strip().lower()
    return _STAGE_LABELS.get(value, value.replace("_", " ").strip().title() or "Running")

def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None

def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def cancellation_requested(run: Any) -> bool:
    """Return whether a cooperative cancellation token has been requested.

    DataLoader worker processes may hold a copied token, so callers in the main
    process must perform this check as well. The attribute probes support the
    platform token plus Future/Event-like compatibility tokens used by tests and
    integrations.
    """

    token = getattr(run, "cancel_token", None)
    if token is None:
        return False

    for name in ("cancelled", "is_cancelled", "cancel_requested", "is_set"):
        value = getattr(token, name, None)
        if value is None:
            continue
        try:
            if bool(value() if callable(value) else value):
                return True
        except Exception:
            continue

    check = getattr(run, "check_cancelled", None)
    if callable(check):
        try:
            check()
        except BaseException:
            return True
    return False

def progress_percent(payload: Mapping[str, Any] | None) -> Optional[float]:
    data = dict(payload or {})
    explicit = _finite_float(data.get("percent"))
    if explicit is not None:
        return min(100.0, max(0.0, explicit))
    current = _finite_float(data.get("current"))
    total = _finite_float(data.get("total"))
    if current is None or total is None or total <= 0:
        return None
    return min(100.0, max(0.0, current * 100.0 / total))

def progress_alert_type(payload: Mapping[str, Any] | None) -> str:
    status = str(dict(payload or {}).get("status") or "running").lower()
    if status in {"failed", "error"}:
        return "danger"
    if status in {"paused", "cancelling", "cancelled"}:
        return "warning"
    if status in {"complete", "completed", "finished"}:
        return "success"
    return "info"

def format_duration(seconds: Any) -> str:
    value = _finite_float(seconds)
    if value is None or value < 0:
        return ""
    rounded = int(round(value))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def progress_markdown(payload: Mapping[str, Any] | None) -> str:
    data = dict(payload or {})
    stage = str(data.get("stage") or data.get("status") or "running")
    label = str(data.get("stage_label") or stage_label(stage))
    message = str(data.get("message") or "Work is in progress.")
    detail = str(data.get("detail") or "").strip()

    lines = [f"**{label}**", message]

    identity = []
    if data.get("run_id"):
        identity.append(f"run `{str(data.get('run_id'))[:12]}`")
    if data.get("recipe_id"):
        identity.append(f"recipe `{data.get('recipe_id')}`")
    if data.get("dataset_id"):
        identity.append(f"dataset `{data.get('dataset_id')}`")
    if identity:
        lines.append("Context: " + ", ".join(identity))

    current = _finite_float(data.get("current"))
    total = _finite_float(data.get("total"))
    unit = str(data.get("unit") or "items")
    percent = progress_percent(data)
    if current is not None and total is not None and total > 0:
        current_text = f"{int(current):,}" if current.is_integer() else f"{current:,.2f}"
        total_text = f"{int(total):,}" if total.is_integer() else f"{total:,.2f}"
        suffix = f" ({percent:.1f}%)" if percent is not None else ""
        lines.append(f"Progress: `{current_text}` / `{total_text}` {unit}{suffix}")
    elif current is not None:
        current_text = f"{int(current):,}" if current.is_integer() else f"{current:,.2f}"
        lines.append(f"Processed: `{current_text}` {unit}")

    epoch = _positive_int(data.get("epoch"))
    total_epochs = _positive_int(data.get("total_epochs"))
    if epoch is not None:
        epoch_text = f"Epoch: `{epoch}`"
        if total_epochs:
            epoch_text += f" / `{total_epochs}`"
        lines.append(epoch_text)

    elapsed = format_duration(data.get("elapsed_seconds"))
    eta = format_duration(data.get("eta_seconds"))
    timing = []
    if elapsed:
        timing.append(f"elapsed `{elapsed}`")
    if eta:
        timing.append(f"estimated remaining `{eta}`")
    if timing:
        lines.append("Timing: " + ", ".join(timing))

    if detail:
        lines.append(detail)
    return "  \n".join(lines)

@dataclass
class _StageState:
    name: str = ""
    started_at: float = 0.0
    current: Optional[float] = None
    total: Optional[float] = None
    unit: str = "items"
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    message: str = ""
    detail: str = ""
    status: str = "running"
    last_emit_at: float = 0.0
    last_percent: Optional[float] = None

class MLProgressReporter:
    """Publish bounded live progress while retaining the latest state in the run log.

    Stage transitions are always persisted. Repeated batch/row updates are
    throttled so the synchronous platform event bus and artifact store are not
    flooded during fast training loops.
    """

    def __init__(self, run: Any) -> None:
        self.run = run
        self._lock = threading.RLock()
        self._run_started_at = time.time()
        self._state = _StageState()
        params = dict(getattr(run, "params", {}) or {})
        self._min_interval = max(0.25, float(params.get("progress_update_interval_seconds") or 2.0))
        self._min_percent_step = max(0.1, float(params.get("progress_min_percent_step") or 1.0))

    def report(
        self,
        *,
        stage: str,
        message: str,
        status: str = "running",
        detail: str = "",
        current: Any = None,
        total: Any = None,
        unit: str = "items",
        epoch: Any = None,
        total_epochs: Any = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        force: bool = False,
        durable: bool = True,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        stage_name = str(stage or "running").strip().lower()
        status_value = str(status or "running").strip().lower()
        current_value = _finite_float(current)
        total_value = _finite_float(total)

        if (
            status_value not in _TERMINAL_STATUSES
            and stage_name != "cancelling"
            and cancellation_requested(self.run)
        ):
            previous_label = stage_label(stage_name)
            stage_name = "cancelling"
            status_value = "cancelling"
            message = (
                "Cancellation requested. Waiting for the current batch operation "
                "to reach a safe stop point."
            )
            detail = (
                f"Last active stage: `{previous_label}`. No additional training "
                "batch will be handed to the recipe after control returns to the "
                "main process."
            )
            force = True

        with self._lock:
            changed_stage = stage_name != self._state.name
            if changed_stage:
                self._state = _StageState(name=stage_name, started_at=now)
                force = True

            percent = None
            if current_value is not None and total_value is not None and total_value > 0:
                percent = min(100.0, max(0.0, current_value * 100.0 / total_value))

            if not force:
                since_last = now - self._state.last_emit_at
                percent_delta = None
                if percent is not None and self._state.last_percent is not None:
                    percent_delta = abs(percent - self._state.last_percent)
                if since_last < self._min_interval and (
                    percent_delta is None or percent_delta < self._min_percent_step
                ):
                    self._state.current = current_value
                    self._state.total = total_value
                    self._state.unit = str(unit or "items")
                    self._state.epoch = _positive_int(epoch)
                    self._state.total_epochs = _positive_int(total_epochs)
                    self._state.message = str(message or "")
                    self._state.detail = str(detail or "")
                    self._state.status = status_value
                    return None

            stage_elapsed = max(0.0, now - self._state.started_at)
            eta = None
            if current_value is not None and total_value is not None and 0 < current_value < total_value:
                rate = current_value / max(stage_elapsed, 1e-9)
                if rate > 0:
                    eta = (total_value - current_value) / rate

            payload: Dict[str, Any] = {
                "schema_version": PROGRESS_SCHEMA_VERSION,
                "run_id": str(getattr(self.run, "run_id", "") or ""),
                "dataset_id": str(getattr(self.run, "dataset_id", "") or ""),
                "recipe_id": str(getattr(self.run, "recipe_id", "") or ""),
                "recipe_version": str(getattr(self.run, "recipe_version", "") or ""),
                "training_log_artifact_id": getattr(self.run, "training_log_artifact_id", None),
                "launcher_session_id": dict(getattr(self.run, "params", {}) or {}).get("launcher_session_id"),
                "status": status_value,
                "stage": stage_name,
                "stage_label": stage_label(stage_name),
                "message": str(message or "Work is in progress."),
                "detail": str(detail or ""),
                "current": current_value,
                "total": total_value,
                "unit": str(unit or "items"),
                "percent": percent,
                "epoch": _positive_int(epoch),
                "total_epochs": _positive_int(total_epochs),
                "elapsed_seconds": max(0.0, now - self._run_started_at),
                "stage_elapsed_seconds": stage_elapsed,
                "eta_seconds": eta,
                "updated_at": now,
            }
            if metrics:
                payload["metrics"] = dict(metrics)
            if extra:
                payload.update(dict(extra))
            payload = json_safe(payload)

            self._state.current = current_value
            self._state.total = total_value
            self._state.unit = str(unit or "items")
            self._state.epoch = _positive_int(epoch)
            self._state.total_epochs = _positive_int(total_epochs)
            self._state.message = str(message or "")
            self._state.detail = str(detail or "")
            self._state.status = status_value
            self._state.last_emit_at = now
            self._state.last_percent = percent

        if durable:
            logger = getattr(self.run, "logger", None)
            update_summary = getattr(logger, "update_summary", None)
            if callable(update_summary):
                try:
                    update_summary(
                        status=status_value,
                        message=str(message or ""),
                        stage=stage_name,
                        progress=payload,
                    )
                except Exception:
                    pass
            try:
                self.run.log(
                    message=str(message or ""),
                    status=status_value,
                    step=_positive_int(epoch),
                    total=_positive_int(total_epochs),
                    metrics=metrics or {},
                    extra={"phase": "progress", "stage": stage_name, "progress": payload},
                )
            except Exception:
                pass

        try:
            self.run.publish(PROGRESS_EVENT, payload)
        except Exception:
            pass
        return payload

    def update(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return self.report(**kwargs)

    def terminal(self, *, status: str, message: str, detail: str = "") -> Optional[Dict[str, Any]]:
        status_value = str(status or "complete").lower()
        stage = status_value if status_value in _TERMINAL_STATUSES else "complete"
        return self.report(
            stage=stage,
            message=message,
            status=status_value,
            detail=detail,
            current=1,
            total=1,
            unit="run",
            force=True,
        )

    @contextmanager
    def activity(
        self,
        *,
        stage: str,
        message: str,
        detail: str = "",
        current: Any = None,
        total: Any = None,
        unit: str = "items",
        epoch: Any = None,
        total_epochs: Any = None,
        heartbeat_seconds: float = 5.0,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[None]:
        self.report(
            stage=stage,
            message=message,
            detail=detail,
            current=current,
            total=total,
            unit=unit,
            epoch=epoch,
            total_epochs=total_epochs,
            extra=extra,
            force=True,
        )
        stop = threading.Event()
        interval = max(1.0, float(heartbeat_seconds or 5.0))

        def heartbeat() -> None:
            while not stop.wait(interval):
                with self._lock:
                    if time.time() - self._state.last_emit_at < interval * 0.9:
                        continue
                    current = self._state.current
                    total = self._state.total
                    state_unit = self._state.unit
                    state_epoch = self._state.epoch
                    state_total_epochs = self._state.total_epochs
                    state_message = self._state.message or message
                    state_detail = self._state.detail or detail
                    state_status = self._state.status or "running"
                self.report(
                    stage=stage,
                    message=state_message,
                    status=state_status,
                    detail=state_detail,
                    current=current,
                    total=total,
                    unit=state_unit,
                    epoch=state_epoch,
                    total_epochs=state_total_epochs,
                    extra={**dict(extra or {}), "heartbeat": True},
                    force=True,
                    durable=False,
                )

        thread = threading.Thread(
            target=heartbeat,
            name=f"ml-progress-{str(getattr(self.run, 'run_id', ''))[:8]}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=0.2)

class ProgressIterable:
    """Main-process cancellation boundary plus bounded training progress.

    The wrapper checks cancellation before requesting a batch, after a possibly
    blocking DataLoader fetch, and after the consumer finishes the batch. This is
    essential when ``num_workers > 0`` because worker processes can hold copied
    cancellation-token state and may already have prefetched batches.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        reporter: Optional[MLProgressReporter],
        run: Any,
    ) -> None:
        self._delegate = delegate
        self._reporter = reporter
        self._run = run
        self._pass_index = 0

    def __iter__(self):
        self._pass_index += 1
        start_epoch = max(1, int(getattr(self._run, "start_epoch", 1) or 1))
        epoch = start_epoch + self._pass_index - 1
        total_epochs = _total_epochs(self._run)
        total_batches = _safe_len(self._delegate)
        message = f"Training epoch {epoch}"
        if total_epochs:
            message += f" of {total_epochs}"
        message += ": reading batches, running forward/backward passes, and updating model weights."

        self._run.check_cancelled()
        if self._reporter is not None:
            self._reporter.report(
                stage="training",
                message=message,
                current=0,
                total=total_batches,
                unit="batches",
                epoch=epoch,
                total_epochs=total_epochs,
                force=True,
            )

        completed = 0
        delegate_iterator = None
        exhausted = False
        try:
            delegate_iterator = iter(self._delegate)
            while True:
                # Main-process check prevents prefetched worker batches from being
                # consumed after the user has requested cancellation.
                self._run.check_cancelled()
                try:
                    item = next(delegate_iterator)
                except StopIteration:
                    exhausted = True
                    break

                # Cancellation may have arrived while DataLoader was blocked
                # fetching/decoding this batch. Do not hand it to the GPU.
                self._run.check_cancelled()
                completed += 1
                yield item

                # The consumer has finished forward/backward/optimiser work for
                # this batch. Stop before reporting stale running progress or
                # requesting another prefetched batch.
                self._run.check_cancelled()
                if self._reporter is not None:
                    self._reporter.report(
                        stage="training",
                        message=message,
                        current=completed,
                        total=total_batches,
                        unit="batches",
                        epoch=epoch,
                        total_epochs=total_epochs,
                    )

            self._run.check_cancelled()
            if exhausted and self._reporter is not None:
                self._reporter.report(
                    stage="training",
                    message=f"Training batches for epoch {epoch} are complete; validation is next.",
                    current=completed,
                    total=total_batches or completed,
                    unit="batches",
                    epoch=epoch,
                    total_epochs=total_epochs,
                    force=True,
                )
        finally:
            _close_loader_iterator(delegate_iterator)

    def __len__(self) -> int:
        return len(self._delegate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _close_loader_iterator(iterator: Any) -> None:
    """Release DataLoader workers promptly after cancellation or failure."""

    if iterator is None:
        return
    for name in ("_shutdown_workers", "shutdown_workers", "close"):
        method = getattr(iterator, name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            pass
        return


def _safe_len(value: Any) -> Optional[int]:
    try:
        length = int(len(value))
    except Exception:
        return None
    return max(0, length)

def _total_epochs(run: Any) -> Optional[int]:
    params = dict(getattr(run, "params", {}) or {})
    for key in ("epochs", "num_epochs", "max_epochs"):
        value = _positive_int(params.get(key))
        if value:
            return value
    return None
