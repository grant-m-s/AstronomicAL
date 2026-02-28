# astronomicAL/platform/artifacts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
import os
import json
import time
import uuid
import hashlib


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    type: str
    dataset_id: str
    created_at: float
    row_ids: Optional[List[str]]
    params: Dict[str, Any]
    has_payload: bool
    uri: Optional[str]


class ArtifactStore:
    """
    Stores computed results for sharing across panels.

    Start simple:
    - In-memory payloads for small objects
    - Optional file-backed storage via `cache_dir` for bigger payloads (JSON-friendly)
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._payloads: Dict[str, Any] = {}
        self._meta: Dict[str, ArtifactRef] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _hash_params(params: Dict[str, Any]) -> str:
        raw = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def put(
        self,
        type: str,
        payload: Any,
        *,
        dataset_id: str = "default",
        row_ids: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        persist: bool = False,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        now = time.time()
        params = params or {}
        row_ids_list = list(row_ids) if row_ids is not None else None

        uri = None
        has_payload = True

        if persist and self._cache_dir:
            # Persist JSON-serializable payloads. For non-JSON, caller should persist themselves
            # and pass uri, or store in-memory for Phase 1.
            uri = os.path.join(self._cache_dir, f"{artifact_id}.json")
            with open(uri, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            has_payload = False
        else:
            self._payloads[artifact_id] = payload

        ref = ArtifactRef(
            artifact_id=artifact_id,
            type=type,
            dataset_id=dataset_id,
            created_at=now,
            row_ids=row_ids_list,
            params=params,
            has_payload=has_payload,
            uri=uri,
        )
        self._meta[artifact_id] = ref
        return artifact_id

    def get(self, artifact_id: str) -> Any:
        ref = self._meta.get(artifact_id)
        if not ref:
            raise KeyError(f"Unknown artifact_id: {artifact_id}")

        if ref.has_payload:
            return self._payloads[artifact_id]

        if ref.uri and os.path.exists(ref.uri):
            with open(ref.uri, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(f"Artifact payload missing for {artifact_id} (uri={ref.uri})")

    def ref(self, artifact_id: str) -> ArtifactRef:
        ref = self._meta.get(artifact_id)
        if not ref:
            raise KeyError(f"Unknown artifact_id: {artifact_id}")
        return ref

    def find(
        self,
        *,
        type: Optional[str] = None,
        dataset_id: Optional[str] = None,
        row_id: Optional[str] = None,
        params_subset: Optional[Dict[str, Any]] = None,
    ) -> List[ArtifactRef]:
        out: List[ArtifactRef] = []
        for ref in self._meta.values():
            if type is not None and ref.type != type:
                continue
            if dataset_id is not None and ref.dataset_id != dataset_id:
                continue
            if row_id is not None:
                if not ref.row_ids or row_id not in ref.row_ids:
                    continue
            if params_subset:
                ok = True
                for k, v in params_subset.items():
                    if ref.params.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            out.append(ref)
        # newest first
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out