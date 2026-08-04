from __future__ import annotations

import pytest

from astronomicAL.platform.events import EventBus


@pytest.mark.unit
def test_subscribe_publish_unsubscribe(events: EventBus) -> None:
    seen = []

    sub = events.subscribe(
        "example.changed",
        lambda topic, payload: seen.append((topic, payload)),
    )

    events.publish("example.changed", {"value": 1})

    assert seen == [("example.changed", {"value": 1})]

    events.unsubscribe(sub)
    events.publish("example.changed", {"value": 2})

    assert seen == [("example.changed", {"value": 1})]


@pytest.mark.unit
def test_wildcard_subscription_receives_all_events(events: EventBus) -> None:
    seen = []

    events.subscribe("*", lambda topic, payload: seen.append((topic, payload)))

    events.publish("dataset.loaded", {"dataset_id": "main"})
    events.publish("selection.focus.changed", {"row_id": "r1"})

    assert seen == [
        ("dataset.loaded", {"dataset_id": "main"}),
        ("selection.focus.changed", {"row_id": "r1"}),
    ]


@pytest.mark.unit
def test_failing_subscriber_does_not_stop_other_subscribers(events: EventBus) -> None:
    seen = []

    def broken(_topic, _payload):
        raise RuntimeError("subscriber failed intentionally")

    def healthy(topic, payload):
        seen.append((topic, payload))

    events.subscribe("example.event", broken)
    events.subscribe("example.event", healthy)

    events.publish("example.event", {"ok": True})

    assert seen == [("example.event", {"ok": True})]


@pytest.mark.unit
def test_trace_records_recent_events() -> None:
    events = EventBus(trace=True, trace_limit=3)

    events.publish("one", {"n": 1})
    events.publish("two", {"n": 2})
    events.publish("three", {"n": 3})
    events.publish("four", {"n": 4})

    recent = events.recent_events(10)

    assert [topic for _timestamp, topic, _payload in recent] == [
        "two",
        "three",
        "four",
    ]


@pytest.mark.unit
def test_subscriber_counts_update(events: EventBus) -> None:
    sub_a = events.subscribe("x", lambda _topic, _payload: None)
    sub_b = events.subscribe("x", lambda _topic, _payload: None)

    assert events.subscribers()["x"] == 2

    events.unsubscribe(sub_a)
    assert events.subscribers()["x"] == 1

    events.unsubscribe(sub_b)
    assert "x" not in events.subscribers()