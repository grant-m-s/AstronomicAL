from __future__ import annotations

import json

import pytest

from astronomicAL.platform.artifacts import ArtifactStore


@pytest.mark.unit
def test_put_get_and_ref_artifact(artifacts: ArtifactStore) -> None:
    artifact_id = artifacts.put(
        "classifier.scores",
        {"r1": 0.9, "r2": 0.1},
        dataset_id="main",
        row_ids=["r1", "r2"],
        params={"model": "rf"},
    )

    assert artifacts.get(artifact_id) == {"r1": 0.9, "r2": 0.1}

    ref = artifacts.ref(artifact_id)
    assert ref.artifact_id == artifact_id
    assert ref.type == "classifier.scores"
    assert ref.dataset_id == "main"
    assert ref.row_ids == ["r1", "r2"]
    assert ref.params == {"model": "rf"}
    assert ref.has_payload is True


@pytest.mark.unit
def test_find_artifacts_by_type_dataset_row_and_params(artifacts: ArtifactStore) -> None:
    matching_id = artifacts.put(
        "classifier.scores",
        {"r1": 0.9},
        dataset_id="main",
        row_ids=["r1"],
        params={"model": "rf", "version": 2},
    )
    artifacts.put(
        "classifier.scores",
        {"r2": 0.2},
        dataset_id="main",
        row_ids=["r2"],
        params={"model": "svm", "version": 1},
    )
    artifacts.put(
        "report.summary",
        {"text": "hello"},
        dataset_id="other",
        row_ids=["r1"],
    )

    refs = artifacts.find(
        type="classifier.scores",
        dataset_id="main",
        row_id="r1",
        params_subset={"model": "rf"},
    )

    assert [ref.artifact_id for ref in refs] == [matching_id]


@pytest.mark.unit
def test_unknown_artifact_raises_key_error(artifacts: ArtifactStore) -> None:
    with pytest.raises(KeyError):
        artifacts.get("does-not-exist")

    with pytest.raises(KeyError):
        artifacts.ref("does-not-exist")


@pytest.mark.unit
def test_persisted_json_artifact_roundtrip(tmp_path) -> None:
    store = ArtifactStore(cache_dir=str(tmp_path))

    payload = {"rows": [{"id": "r1", "score": 0.9}]}
    artifact_id = store.put(
        "report.summary",
        payload,
        dataset_id="main",
        persist=True,
    )

    ref = store.ref(artifact_id)

    assert ref.has_payload is False
    assert ref.uri is not None
    assert store.get(artifact_id) == payload

    with open(ref.uri, "r", encoding="utf-8") as handle:
        assert json.load(handle) == payload