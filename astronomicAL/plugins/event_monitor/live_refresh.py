from __future__ import annotations

from typing import Any, Callable


class DocumentPeriodicRefresh:
    """Own one periodic callback on one Bokeh session document.

    The wrapper is independent of Panel so its lifecycle can be tested without
    creating a server session. Starting is idempotent, a running callback cannot
    be moved to another document, and stop always removes the callback from the
    document that created it.
    """

    def __init__(self, callback: Callable[[], None], *, period_ms: int):
        if period_ms <= 0:
            raise ValueError("period_ms must be greater than zero")

        self._callback = callback
        self._period_ms = int(period_ms)
        self._document: Any | None = None
        self._handle: Any | None = None
        self._disposed = False

    @property
    def running(self) -> bool:
        return self._handle is not None and not self._disposed

    @property
    def document(self) -> Any | None:
        return self._document

    def start(self, document: Any | None) -> bool:
        if self._disposed or document is None:
            return False
        if self._handle is not None:
            if document is not self._document:
                raise RuntimeError(
                    "A running periodic refresh cannot move between session documents"
                )
            return True

        add_callback = getattr(document, "add_periodic_callback", None)
        if not callable(add_callback):
            raise TypeError(
                "The current session document cannot schedule periodic callbacks"
            )

        handle = add_callback(self._callback, self._period_ms)
        if handle is None:
            raise RuntimeError("The session document did not return a callback handle")

        self._document = document
        self._handle = handle
        return True

    def stop(self) -> None:
        document = self._document
        handle = self._handle
        self._document = None
        self._handle = None

        if document is None or handle is None:
            return

        remove_callback = getattr(document, "remove_periodic_callback", None)
        if callable(remove_callback):
            remove_callback(handle)

    def dispose(self) -> None:
        if self._disposed:
            return
        self.stop()
        self._disposed = True