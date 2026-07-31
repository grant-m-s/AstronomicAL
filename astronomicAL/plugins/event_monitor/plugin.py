from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.event_monitor",
    name="Event Monitor",
    version="1.4.3",
    description=(
        "Live event-bus health, performance, callback errors, and subscription "
        "inspection for AstronomicAL."
    ),
    capabilities=["panel", "diagnostics", "events", "observability"],
    tags=["core", "diagnostics", "events", "performance"],
)

def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Event Monitor",
        factory=create_event_monitor_panel,
        description=(
            "Monitor event throughput, publish latency, slow callbacks, callback "
            "errors, active publishes, and subscriptions."
        ),
        category="Diagnostics",
        icon="activity",
        tags=["diagnostics", "events", "performance", "subscriptions"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 10},
        state_version=1,
    )

def create_event_monitor_panel(context, *, instance_id=None, **_kwargs):
    # Keep plugin discovery and registration cheap. Panel is imported only when
    # the user opens this contribution.
    from astronomicAL.plugins.event_monitor.panel import EventMonitorPanel

    controller = EventMonitorPanel(context=context, instance_id=instance_id)
    return controller.view, controller
