from __future__ import annotations

import threading

from astronomicAL.platform.artifacts import ArtifactStore
from astronomicAL.platform.context import AppContext
from astronomicAL.platform.events import EventBus
from astronomicAL.platform.jobs import JobManager
from astronomicAL.platform.services import ServiceRegistry

from .conftest import recent_event_topics


def test_event_bus_supports_wildcards_trace_and_unsubscribe() -> None:
    bus = EventBus(trace=True)
    seen: list[tuple[str, dict]] = []

    sub = bus.subscribe(
        "*",
        lambda topic, payload: seen.append((topic, payload)),
        owner_id="test.wildcard",
        owner_kind="test",
        owner_label="Wildcard test",
    )

    bus.publish("alpha.beta", {"value": 1})

    assert seen == [("alpha.beta", {"value": 1})]
    assert [topic for _, topic, _ in bus.recent_events(10)] == ["alpha.beta"]

    subscriptions = bus.list_subscriptions()
    assert any(item.owner_id == "test.wildcard" for item in subscriptions)

    bus.unsubscribe(sub)
    bus.publish("alpha.gamma", {"value": 2})

    assert seen == [("alpha.beta", {"value": 1})]


def test_artifact_store_can_put_get_ref_and_find(tmp_path) -> None:
    store = ArtifactStore(cache_dir=str(tmp_path / "artifacts"))

    first_id = store.put(
        "classifier.scores",
        {"score": 0.9},
        dataset_id="main",
        row_ids=["a"],
        params={"model": "rf"},
    )
    second_id = store.put(
        "report.summary",
        {"text": "hello"},
        dataset_id="main",
        row_ids=["b"],
    )

    assert store.get(first_id) == {"score": 0.9}

    first_ref = store.ref(first_id)
    assert first_ref.type == "classifier.scores"
    assert first_ref.dataset_id == "main"
    assert first_ref.row_ids == ["a"]
    assert first_ref.params == {"model": "rf"}

    score_refs = store.find(type="classifier.scores")
    assert [ref.artifact_id for ref in score_refs] == [first_id]

    main_refs = store.find(dataset_id="main")
    assert {ref.artifact_id for ref in main_refs} == {first_id, second_id}


def test_service_registry_lazy_factory_and_removal() -> None:
    services = ServiceRegistry()
    calls: list[str] = []
    disposed: list[str] = []

    class DisposableClient:
        def __init__(self) -> None:
            calls.append("created")

        def dispose(self) -> None:
            disposed.append("disposed")

    services.set_factory("client.lazy", lambda: DisposableClient(), lazy=True)

    assert calls == []
    client = services.get("client.lazy")
    assert isinstance(client, DisposableClient)
    assert calls == ["created"]

    same_client = services.get("client.lazy")
    assert same_client is client
    assert calls == ["created"]

    removed = services.remove("client.lazy")
    assert removed is client
    assert disposed == ["disposed"]
    assert services.get("client.lazy") is None


def test_job_manager_deduplicates_in_flight_jobs_by_key() -> None:
    jobs = JobManager(max_workers=2)
    release = threading.Event()
    calls: list[str] = []

    def slow_work(*, cancel_token):
        calls.append("started")
        release.wait(timeout=2)
        if cancel_token and cancel_token.cancelled():
            return "cancelled"
        return "done"

    try:
        first = jobs.submit(slow_work, title="Slow", key="same-job")
        second = jobs.submit(slow_work, title="Slow duplicate", key="same-job")

        assert first is second

        release.set()
        assert first.future.result(timeout=3) == "done"
        assert calls == ["started"]
    finally:
        jobs.shutdown(wait=False)


def test_job_handle_cancel_sets_cancel_token() -> None:
    jobs = JobManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def cancellable_work(*, cancel_token):
        started.set()
        release.wait(timeout=2)
        return "cancelled" if cancel_token and cancel_token.cancelled() else "done"

    try:
        handle = jobs.submit(cancellable_work, title="Cancellable", key="cancel-me")
        started.wait(timeout=2)

        handle.cancel()
        release.set()

        assert handle.future.result(timeout=3) == "cancelled"
    finally:
        jobs.shutdown(wait=False)


def test_dataset_manager_source_access_and_mappings(app_context: AppContext) -> None:
    assert app_context.datasets.active_id() == "main"
    assert app_context.datasets.list_columns("main") == [
        "id",
        "value",
        "other",
        "name",
        "label",
    ]
    assert app_context.datasets.row_count("main") == 3
    assert app_context.datasets.get_mapping("main", "record_id") == "id"
    assert app_context.datasets.get_mapping("main", "target_label") == "label"

    preview = app_context.datasets.get_df("main", columns=["id", "value"])
    assert list(preview.columns) == ["id", "value"]
    assert list(preview["id"]) == ["a", "b", "c"]


def test_platform_services_work_together_for_event_plus_artifact_pattern(
    app_context: AppContext,
) -> None:
    artifact_id = app_context.artifacts.put(
        "test.result",
        {"answer": 42},
        dataset_id="main",
        row_ids=["a"],
    )

    app_context.events.publish(
        "artifact.created",
        {
            "artifact_id": artifact_id,
            "type": "test.result",
            "dataset_id": "main",
        },
    )

    assert app_context.artifacts.get(artifact_id) == {"answer": 42}
    assert "artifact.created" in recent_event_topics(app_context)