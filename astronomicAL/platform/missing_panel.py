from __future__ import annotations

from html import escape
from typing import Any

import panel as pn


class MissingPanelController:
    """
    Placeholder used during workspace restore when a saved panel cannot be
    recreated.

    The important behaviour is that the tile remains in the restored grid
    instead of silently disappearing.
    """

    state_version = 1

    def __init__(self, *, reason: str, snapshot: dict[str, Any]) -> None:
        self.reason = reason
        self.snapshot = dict(snapshot or {})
        self.view = pn.Column(
            pn.pane.HTML(self._html(), sizing_mode="stretch_width"),
            sizing_mode="stretch_both",
            styles={
                "box-sizing": "border-box",
                "padding": "10px",
                "height": "100%",
                "overflow": "auto",
                "border": "1px solid #efb5b5",
                "background": "#fff5f5",
            },
        )

    def _html(self) -> str:
        title = escape(str(self.snapshot.get("title") or "Missing panel"))
        plugin_id = escape(str(self.snapshot.get("plugin_id") or "unknown"))
        registration_id = escape(str(self.snapshot.get("registration_id") or "unknown"))
        reason = escape(str(self.reason))

        return f"""
        <div style="font-family: system-ui, sans-serif;">
          <h3 style="margin: 0 0 8px 0; color: #8a1f1f;">{title}</h3>
          <p style="margin: 0 0 8px 0;">
            This panel could not be restored.
          </p>
          <table style="font-size: 13px; border-collapse: collapse;">
            <tr>
              <td style="font-weight: 700; padding: 3px 8px 3px 0;">Plugin</td>
              <td style="font-family: monospace;">{plugin_id}</td>
            </tr>
            <tr>
              <td style="font-weight: 700; padding: 3px 8px 3px 0;">Registration</td>
              <td style="font-family: monospace;">{registration_id}</td>
            </tr>
            <tr>
              <td style="font-weight: 700; padding: 3px 8px 3px 0;">Reason</td>
              <td>{reason}</td>
            </tr>
          </table>
        </div>
        """

    def get_state(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "snapshot": self.snapshot,
        }

    def dispose(self) -> None:
        return None


def create_missing_panel(*, reason: str, snapshot: dict[str, Any]) -> tuple[Any, Any]:
    controller = MissingPanelController(reason=reason, snapshot=snapshot)
    return controller.view, controller