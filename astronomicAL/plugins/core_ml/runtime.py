from __future__ import annotations

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

class MLRecipeCancelled(RuntimeError):
    """Raised when a recipe run is cancelled by the user."""

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
                    return None
        except Exception:
            return None
