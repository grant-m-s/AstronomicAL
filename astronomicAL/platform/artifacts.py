from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import uuid


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRowIdsRef:
    """Durable identity for the complete row set represented by an artifact."""

    storage: str
    uri: str
    format: str
    row_count: int
    sha256: Optional[str] = None
    id_column: str = "row_id"
    parts: tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.row_count) < 0:
            raise ValueError("ArtifactRowIdsRef.row_count must be zero or greater")
        if not str(self.uri or "").strip():
            raise ValueError("ArtifactRowIdsRef.uri is required")
        if not str(self.format or "").strip():
            raise ValueError("ArtifactRowIdsRef.format is required")

    @classmethod
    def from_value(
        cls,
        value: ArtifactRowIdsRef | Mapping[str, Any],
    ) -> ArtifactRowIdsRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "row_ids_ref must be ArtifactRowIdsRef or a mapping, "
                f"got {type(value)!r}"
            )
        return cls(
            storage=str(value.get("storage") or "external"),
            uri=str(value.get("uri") or ""),
            format=str(value.get("format") or ""),
            row_count=int(value.get("row_count") or 0),
            sha256=(
                None
                if value.get("sha256") in (None, "")
                else str(value.get("sha256"))
            ),
            id_column=str(value.get("id_column") or "row_id"),
            parts=tuple(str(part) for part in value.get("parts") or ()),
            params=dict(value.get("params") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage": self.storage,
            "uri": self.uri,
            "format": self.format,
            "row_count": int(self.row_count),
            "sha256": self.sha256,
            "id_column": self.id_column,
            "parts": list(self.parts),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    type: str
    dataset_id: str
    created_at: float
    row_ids: Optional[List[str]]
    row_count: Optional[int]
    row_ids_ref: Optional[ArtifactRowIdsRef]
    params: Dict[str, Any]
    has_payload: bool
    uri: Optional[str]

    @property
    def rows_inline_complete(self) -> bool:
        return (
            self.row_count is not None
            and self.row_ids is not None
            and len(self.row_ids) == int(self.row_count)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "row_ids": None if self.row_ids is None else list(self.row_ids),
            "row_count": self.row_count,
            "row_ids_ref": (
                None if self.row_ids_ref is None else self.row_ids_ref.to_dict()
            ),
            "params": dict(self.params),
            "has_payload": self.has_payload,
            "uri": self.uri,
            "rows_inline_complete": self.rows_inline_complete,
        }


class ArtifactStore:
    """Store reusable outputs and their row-identity metadata.

    Small payloads may remain in memory. JSON-compatible payloads may be written
    to ``cache_dir``. Large domain outputs should persist themselves and place
    their external references in the payload, while ``row_ids_ref`` identifies
    the complete represented row set independently from any inline preview.

    When bound to the host EventBus, successful writes publish the lightweight
    ``artifact.store.created`` lifecycle event. The event is emitted after the
    artifact is fully committed and never transports the artifact payload.
    """

    def __init__(self, cache_dir: Optional[str] = None, *, events: Any = None) -> None:
        self._payloads: Dict[str, Any] = {}
        self._meta: Dict[str, ArtifactRef] = {}
        self._cache_dir = cache_dir
        self._events = events
        self._lock = threading.RLock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def bind_events(self, events: Any) -> None:
        """Bind the store to the shared host EventBus.

        Rebinding to a different bus is rejected because one ArtifactStore should
        belong to one application context. Rebinding the same instance is a no-op.
        """

        if events is None:
            raise ValueError("ArtifactStore requires a non-null EventBus")
        with self._lock:
            if self._events is not None and self._events is not events:
                raise RuntimeError("ArtifactStore is already bound to another EventBus")
            self._events = events

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
        row_ids: Optional[Iterable[Any]] = None,
        row_count: Optional[int] = None,
        row_ids_ref: Optional[
            ArtifactRowIdsRef | Mapping[str, Any]
        ] = None,
        params: Optional[Dict[str, Any]] = None,
        persist: bool = False,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        now = time.time()
        params = dict(params or {})
        row_ids_list = (
            [str(row_id) for row_id in row_ids]
            if row_ids is not None
            else None
        )
        resolved_row_ids_ref = (
            None
            if row_ids_ref is None
            else ArtifactRowIdsRef.from_value(row_ids_ref)
        )
        resolved_row_count = self._resolve_row_count(
            row_ids=row_ids_list,
            row_count=row_count,
            row_ids_ref=resolved_row_ids_ref,
        )

        uri = None
        has_payload = True
        persisted = bool(persist and self._cache_dir)
        if persisted:
            uri = os.path.join(str(self._cache_dir), f"{artifact_id}.json")
            temp_uri = f"{uri}.tmp"
            try:
                with open(temp_uri, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, default=str)
                os.replace(temp_uri, uri)
            except Exception:
                try:
                    os.unlink(temp_uri)
                except OSError:
                    pass
                raise
            has_payload = False

        ref = ArtifactRef(
            artifact_id=artifact_id,
            type=str(type),
            dataset_id=str(dataset_id),
            created_at=now,
            row_ids=row_ids_list,
            row_count=resolved_row_count,
            row_ids_ref=resolved_row_ids_ref,
            params=params,
            has_payload=has_payload,
            uri=uri,
        )

        with self._lock:
            if not persisted:
                self._payloads[artifact_id] = payload
            self._meta[artifact_id] = ref

        self._publish_created(ref)
        return artifact_id

    def _publish_created(self, ref: ArtifactRef) -> None:
        with self._lock:
            events = self._events
        publish = getattr(events, "publish", None)
        if not callable(publish):
            return

        try:
            publish(
                "artifact.store.created",
                {
                    "artifact_id": ref.artifact_id,
                    "type": ref.type,
                    "dataset_id": ref.dataset_id,
                    "row_count": ref.row_count,
                    "created_at": ref.created_at,
                },
            )
        except Exception:
            # Artifact persistence is authoritative. A notification failure must
            # not make a successfully committed artifact appear to have failed.
            logger.exception(
                "Failed to publish artifact.store.created for %s",
                ref.artifact_id,
            )

    @staticmethod
    def _resolve_row_count(
        *,
        row_ids: Optional[Sequence[str]],
        row_count: Optional[int],
        row_ids_ref: Optional[ArtifactRowIdsRef],
    ) -> Optional[int]:
        resolved = None if row_count is None else int(row_count)
        if resolved is not None and resolved < 0:
            raise ValueError("row_count must be zero or greater")
        if row_ids_ref is not None:
            ref_count = int(row_ids_ref.row_count)
            if resolved is None:
                resolved = ref_count
            elif resolved != ref_count:
                raise ValueError(
                    "row_count does not match row_ids_ref.row_count: "
                    f"{resolved} != {ref_count}"
                )
        if row_ids is not None:
            inline_count = len(row_ids)
            if resolved is None:
                resolved = inline_count
            elif inline_count > resolved:
                raise ValueError(
                    "Inline row_ids cannot contain more rows than row_count: "
                    f"{inline_count} > {resolved}"
                )
        return resolved

    def get(self, artifact_id: str) -> Any:
        with self._lock:
            ref = self._meta.get(artifact_id)
            if not ref:
                raise KeyError(f"Unknown artifact_id: {artifact_id}")
            if ref.has_payload:
                return self._payloads[artifact_id]
            uri = ref.uri

        if uri and os.path.exists(uri):
            with open(uri, "r", encoding="utf-8") as handle:
                return json.load(handle)
        raise FileNotFoundError(
            f"Artifact payload missing for {artifact_id} (uri={uri})"
        )

    def ref(self, artifact_id: str) -> ArtifactRef:
        with self._lock:
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
        requested_row_id = None if row_id is None else str(row_id)
        with self._lock:
            refs = list(self._meta.values())

        out: List[ArtifactRef] = []
        for ref in refs:
            if type is not None and ref.type != type:
                continue
            if dataset_id is not None and ref.dataset_id != dataset_id:
                continue
            if requested_row_id is not None:
                if not ref.row_ids or requested_row_id not in ref.row_ids:
                    continue
            if params_subset:
                if any(
                    ref.params.get(key) != value
                    for key, value in params_subset.items()
                ):
                    continue
            out.append(ref)
        out.sort(key=lambda value: value.created_at, reverse=True)
        return out
