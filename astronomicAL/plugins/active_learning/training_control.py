from __future__ import annotations

import threading
import uuid
from typing import Any, Optional

from astronomicAL.plugins.core_ml.runtime import TrainingControl


class ActiveLearningTrainingControls:
    """Context-owned cooperative pause controls for AL training jobs.

    Controls are addressed by opaque IDs so action requests remain serialisable.
    The service owns no jobs or threads; cancellation remains the responsibility
    of the platform JobManager.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._controls: dict[str, TrainingControl] = {}

    def create(self, control_id: Optional[str] = None) -> tuple[str, TrainingControl]:
        resolved = str(control_id or uuid.uuid4().hex)
        control = TrainingControl()
        with self._lock:
            if resolved in self._controls:
                raise ValueError(
                    f"Active Learning training control already exists: {resolved}"
                )
            self._controls[resolved] = control
        return resolved, control

    def get(self, control_id: Any) -> Optional[TrainingControl]:
        resolved = str(control_id or "").strip()
        if not resolved:
            return None
        with self._lock:
            return self._controls.get(resolved)

    def request_pause(
        self,
        control_id: Any,
        reason: str = "Pause requested from the Active Learning panel.",
    ) -> bool:
        control = self.get(control_id)
        if control is None:
            return False
        control.request_pause(reason)
        return True

    def release(self, control_id: Any) -> None:
        resolved = str(control_id or "").strip()
        if not resolved:
            return
        with self._lock:
            self._controls.pop(resolved, None)

    def clear(self) -> None:
        with self._lock:
            self._controls.clear()
