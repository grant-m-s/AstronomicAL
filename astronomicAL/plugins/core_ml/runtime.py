from __future__ import annotations

import gc
import sys
import threading
from typing import Any, Mapping, Optional, Sequence

from astronomicAL.platform.plugins.specs import ActionRequest

from .serialization import json_safe

def coerce_action_request(request: Any) -> ActionRequest:
    """Normalize plugin action requests from platform, dict, or test doubles."""
    if isinstance(request, ActionRequest):
        return request
    if isinstance(request, dict):
        return ActionRequest.from_dict(request)
    return ActionRequest(
        dataset_id=getattr(request, "dataset_id", None),
        row_ids=getattr(request, "row_ids", None),
        columns=list(getattr(request, "columns", []) or []),
        params=dict(getattr(request, "params", {}) or {}),
        artifact_id=getattr(request, "artifact_id", None),
        origin=getattr(request, "origin", None),
    )

def request_dataset_id(context: Any, request: ActionRequest, params: Mapping[str, Any]) -> Optional[str]:
    from .data.dataset_access import active_dataset_id

    value = params.get("dataset_id") or request.dataset_id
    return str(value) if value else active_dataset_id(context)

class MLRecipeExecutionError(RuntimeError):
    """Lightweight job error that does not retain the training traceback."""

    def __init__(self, message: str, *, failure_payload: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.failure_payload = dict(failure_payload or {})


class MLPredictionExecutionError(RuntimeError):
    """Lightweight prediction error that does not retain accelerator tensors."""

    def __init__(self, message: str, *, failure_payload: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.failure_payload = dict(failure_payload or {})


class MLRecipeCancelled(RuntimeError):
    """Raised when a recipe run is cancelled by the user."""


class MLRecipePaused(RuntimeError):
    """Raised at a safe epoch boundary after a resumable checkpoint is saved."""

    def __init__(self, message: str, *, pause_payload: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.pause_payload = dict(pause_payload or {})


class TrainingControl:
    """Thread-safe controls that are separate from destructive cancellation.

    Pause is cooperative and is honoured only by a harness epoch boundary. This
    keeps the current epoch, validation pass, scheduler update, and checkpoint
    internally consistent.
    """

    def __init__(self) -> None:
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        self.pause_reason = "Training pause requested."

    def request_pause(self, reason: str = "Training pause requested.") -> None:
        with self._lock:
            self.pause_reason = str(reason or "Training pause requested.")
            self._pause_event.set()

    def clear_pause(self) -> None:
        self._pause_event.clear()

    @property
    def pause_requested(self) -> bool:
        return self._pause_event.is_set()

    def is_pause_requested(self) -> bool:
        return self._pause_event.is_set()


class CancellationToken:
    """Thread-safe cancellation token for recipe runs.

    The recipe launcher owns this token. Recipes should call
    run.check_cancelled() inside long loops so cancellation can happen cleanly.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = "ML recipe run was cancelled."

    def cancel(self, reason: str = "ML recipe run was cancelled.") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()

def check_cancelled(cancel_token: Any = None) -> None:
    """Raise MLRecipeCancelled if a cancellation token has been set."""

    if cancel_token is None:
        return

    cancelled = False

    for attr in ("cancelled", "is_cancelled", "is_set"):
        value = getattr(cancel_token, attr, None)

        try:
            if callable(value):
                cancelled = bool(value())
            elif value is not None:
                cancelled = bool(value)
        except Exception:
            cancelled = False

        if cancelled:
            break

    if cancelled:
        reason = getattr(
            cancel_token,
            "reason",
            "ML recipe run was cancelled.",
        )
        raise MLRecipeCancelled(str(reason))

def publish(
    context: Any,
    event_type: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Best-effort platform event publish."""

    events = getattr(context, "events", None)
    if events is None:
        return

    payload_dict = json_safe(dict(payload or {}))

    for method_name in ("publish", "emit"):
        method = getattr(events, method_name, None)
        if callable(method):
            try:
                method(event_type, payload_dict)
                return
            except TypeError:
                try:
                    method(
                        {
                            "type": event_type,
                            "payload": payload_dict,
                        }
                    )
                    return
                except Exception:
                    return
            except Exception:
                return

def put_artifact(
    context: Any,
    artifact_type: str,
    payload: Mapping[str, Any],
    *,
    dataset_id: Optional[str] = None,
    row_ids: Optional[Sequence[Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    persist: bool = True,
    required: bool = False,
) -> Optional[str]:
    """Create an artifact using the platform ArtifactStore."""

    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return None

    put = getattr(artifacts, "put", None)
    if not callable(put):
        return None

    clean_payload = json_safe(dict(payload or {}))

    kwargs = {
        "dataset_id": dataset_id,
        "row_ids": list(row_ids) if row_ids is not None else None,
        "params": json_safe(dict(params or {})),
        "persist": persist,
    }

    # Different ArtifactStore versions may not accept every kwarg.
    while True:
        try:
            return put(
                artifact_type,
                clean_payload,
                **kwargs,
            )
        except TypeError as exc:
            message = str(exc)

            removed = False
            for key in list(kwargs):
                if key in message:
                    kwargs.pop(key, None)
                    removed = True
                    break

            if not removed:
                # Fallback to the smallest likely signature.
                try:
                    return put(artifact_type, clean_payload)
                except Exception:
                    if required:
                        raise
                    return None
        except Exception:
            if required:
                raise
            return None


def is_out_of_memory_error(error: BaseException | str) -> bool:
    text = str(error or "").lower()
    markers = (
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "mps backend out of memory",
        "defaultcpuallocator: can't allocate memory",
    )
    return any(marker in text for marker in markers)


def cleanup_ml_runtime(*, reason: str = "", aggressive: bool = False) -> dict[str, Any]:
    """Release Python and accelerator caches after an ML operation.

    The function deliberately does not import torch. Importing a heavyweight
    framework during cleanup would make sklearn-only runs slower and can itself
    fail in a damaged environment. If torch was used, it is already present in
    ``sys.modules`` and its backend caches are cleared best-effort.
    """
    report: dict[str, Any] = {
        "reason": str(reason or ""),
        "aggressive": bool(aggressive),
        "python_collected": 0,
        "cuda_cache_cleared": False,
        "cuda_ipc_collected": False,
        "mps_cache_cleared": False,
        "errors": [],
    }

    try:
        report["python_collected"] = int(gc.collect())
    except Exception as exc:
        report["errors"].append(f"gc.collect: {exc}")

    torch = sys.modules.get("torch")
    if torch is None:
        return report

    try:
        clear_autocast = getattr(torch, "clear_autocast_cache", None)
        if callable(clear_autocast):
            clear_autocast()
    except Exception as exc:
        report["errors"].append(f"torch.clear_autocast_cache: {exc}")

    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
                report["cuda_cache_cleared"] = True
        except Exception as exc:
            report["errors"].append(f"torch.cuda.empty_cache: {exc}")

        try:
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
                report["cuda_ipc_collected"] = True
        except Exception as exc:
            report["errors"].append(f"torch.cuda.ipc_collect: {exc}")

        if aggressive:
            try:
                reset_peak = getattr(cuda, "reset_peak_memory_stats", None)
                if callable(reset_peak):
                    reset_peak()
            except Exception as exc:
                report["errors"].append(f"torch.cuda.reset_peak_memory_stats: {exc}")

    try:
        mps = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(mps):
            mps()
            report["mps_cache_cleared"] = True
    except Exception as exc:
        report["errors"].append(f"torch.mps.empty_cache: {exc}")

    if aggressive:
        try:
            report["python_collected_after_accelerator"] = int(gc.collect())
        except Exception as exc:
            report["errors"].append(f"second gc.collect: {exc}")

    return report
