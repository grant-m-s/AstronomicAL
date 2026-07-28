from __future__ import annotations

from typing import Any, Callable, Optional

def submit_job(
    app_context: Any,
    fn: Callable[..., Any],
    *,
    title: str,
    key: Optional[str] = None,
    on_done: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
    **kwargs: Any,
):
    """Submit to the platform JobManager when available.

    JobManager.submit passes `cancel_token` to `fn`, which is exactly what the
    core ML action handlers expect. This helper centralizes the optional fallback
    behavior for tests or stripped-down contexts.
    """
    jobs = getattr(app_context, "jobs", None)
    submit = getattr(jobs, "submit", None)
    if callable(submit):
        return submit(fn, title=title, key=key, on_done=on_done, on_error=on_error, **kwargs)

    import threading

    class _FallbackToken:
        def __init__(self) -> None:
            self._cancelled = False
        def cancel(self, *_: Any) -> None:
            self._cancelled = True
        def cancelled(self) -> bool:
            return self._cancelled

    class _FallbackHandle:
        def __init__(self, thread: threading.Thread, token: _FallbackToken) -> None:
            self.future = None
            self.token = token
            self.thread = thread
        def cancel(self) -> bool:
            self.token.cancel()
            return False

    token = _FallbackToken()

    def runner() -> None:
        try:
            result = fn(cancel_token=token, **kwargs)
            if on_done:
                on_done(result)
        except BaseException as exc:
            if on_error:
                on_error(exc)
            else:
                raise

    thread = threading.Thread(target=runner, daemon=True)
    handle = _FallbackHandle(thread, token)
    thread.start()
    return handle
