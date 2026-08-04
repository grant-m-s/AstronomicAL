from __future__ import annotations

import pytest

from astronomicAL.platform.selection import SelectionManager


@pytest.mark.unit
def test_set_focus_updates_state_and_publishes_event(context) -> None:
    seen = []

    context.events.subscribe(
        "selection.focus.changed",
        lambda topic, payload: seen.append((topic, payload)),
    )

    context.selection.set_focus(
        dataset_id="main",
        row_id="r2",
        origin="test",
        metadata={"reason": "unit-test"},
    )

    focus = context.selection.get_focus()

    assert focus.dataset_id == "main"
    assert focus.row_id == "r2"
    assert focus.origin == "test"
    assert focus.metadata == {"reason": "unit-test"}

    assert seen[-1][0] == "selection.focus.changed"
    assert seen[-1][1]["dataset_id"] == "main"
    assert seen[-1][1]["row_id"] == "r2"
    assert seen[-1][1]["origin"] == "test"
    assert seen[-1][1]["metadata"] == {"reason": "unit-test"}


@pytest.mark.unit
def test_clear_focus_publishes_event(context) -> None:
    seen = []

    context.selection.set_focus(dataset_id="main", row_id="r1", origin="setup")

    context.events.subscribe(
        "selection.focus.cleared",
        lambda topic, payload: seen.append((topic, payload)),
    )

    context.selection.clear_focus(origin="test")

    focus = context.selection.get_focus()

    assert focus.dataset_id is None
    assert focus.row_id is None

    assert seen == [
        (
            "selection.focus.cleared",
            {
                "dataset_id": "main",
                "origin": "test",
                "panel_id": None,
            },
        )
    ]


@pytest.mark.unit
def test_set_selection_set_updates_state_publishes_event_and_creates_artifact(context) -> None:
    seen = []

    context.events.subscribe(
        "selection.set.changed",
        lambda topic, payload: seen.append((topic, payload)),
    )

    state = context.selection.set_selection_set(
        dataset_id="main",
        row_ids=["r3", "r1", "r3"],
        origin="lasso",
        metadata={"tool": "scatter"},
    )

    assert state.dataset_id == "main"
    assert state.row_ids == ["r3", "r1"]
    assert state.origin == "lasso"
    assert state.artifact_id is not None

    active = context.selection.get_active_set()
    assert active is state

    payload = seen[-1][1]
    assert seen[-1][0] == "selection.set.changed"
    assert payload["dataset_id"] == "main"
    assert payload["count"] == 2
    assert payload["row_ids"] == ["r3", "r1"]
    assert payload["artifact_id"] == state.artifact_id

    artifact_payload = context.artifacts.get(state.artifact_id)
    assert artifact_payload["dataset_id"] == "main"
    assert artifact_payload["row_ids"] == ["r3", "r1"]
    assert artifact_payload["origin"] == "lasso"


@pytest.mark.unit
def test_selection_set_focus_policy_defaults_to_first_when_focus_not_in_set(context) -> None:
    context.selection.set_focus(dataset_id="main", row_id="r4", origin="setup")

    context.selection.set_selection_set(
        dataset_id="main",
        row_ids=["r1", "r2"],
        origin="lasso",
    )

    focus = context.selection.get_focus()

    assert focus.dataset_id == "main"
    assert focus.row_id == "r1"


@pytest.mark.unit
def test_selection_snapshot_and_restore(context) -> None:
    context.selection.set_focus(
        dataset_id="main",
        row_id="r2",
        origin="before-snapshot",
    )
    context.selection.set_selection_set(
        dataset_id="main",
        row_ids=["r2", "r3"],
        origin="before-snapshot",
        create_artifact=False,
    )

    snapshot = context.selection.snapshot()

    restored = SelectionManager(context.events, artifacts=context.artifacts)
    restored.restore_snapshot(snapshot, publish=False)

    assert restored.get_focus().dataset_id == "main"
    assert restored.get_focus().row_id == "r2"

    active_set = restored.get_active_set()
    assert active_set is not None
    assert active_set.dataset_id == "main"
    assert active_set.row_ids == ["r2", "r3"]