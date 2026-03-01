import panel as pn
import pandas as pd
import time

class EventMonitor:
    def __init__(self, context):
        self.context = context
        self.table = pn.widgets.Tabulator(pd.DataFrame(columns=["time", "topic", "payload"]),
                                          height=300, sizing_mode="stretch_width")
        self.subs = pn.widgets.Tabulator(pd.DataFrame(columns=["topic", "subscribers"]),
                                         height=200, sizing_mode="stretch_width")
        self.refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary")
        self.refresh_btn.on_click(lambda _e: self.refresh())

        # auto refresh
        self._cb = pn.state.add_periodic_callback(self.refresh, period=1000)

    def refresh(self):
        events = self.context.events.recent_events(200)
        df = pd.DataFrame(
            [{
                "time": time.strftime("%H:%M:%S", time.localtime(t)),
                "topic": topic,
                "payload": str(payload)[:200],
            } for (t, topic, payload) in events]
        )
        self.table.value = df

        subs = self.context.events.subscribers()
        df2 = pd.DataFrame([{"topic": k, "subscribers": v} for k, v in sorted(subs.items())])
        self.subs.value = df2

    def panel(self):
        return pn.Column(
            pn.Row(self.refresh_btn),
            pn.pane.Markdown("### Recent events"),
            self.table,
            pn.pane.Markdown("### Subscriptions"),
            self.subs,
            sizing_mode="stretch_both",
        )

    def dispose(self):
        try:
            self._cb.stop()
        except Exception:
            pass