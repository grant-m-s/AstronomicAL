from __future__ import annotations

import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from ..protocol import SplitManifestRef
from ..split_manifest import SPLIT_ROLE_COLUMN, iter_split_manifest
from . import action as legacy

PROV_TRAIN = "train"
PROV_VALIDATION = "validation"
PROV_TEST = "test"
PROV_NOVEL = "novel"
PROV_UNKNOWN = "unknown"


class StreamingProvenanceIndex:
    """Disk-backed provenance lookup for manifest-sized experiment splits."""

    def __init__(self, context: Any, model_payload: Mapping[str, Any]):
        self.context = context
        self.model_payload = dict(model_payload or {})
        self.verified = False
        self.split_spec_artifact_id: Optional[str] = None
        self.protocol_id: Optional[str] = None
        self.record_id_column: Optional[str] = None
        self.dataset_ids: Dict[str, Optional[str]] = {
            PROV_TRAIN: None,
            PROV_VALIDATION: None,
            PROV_TEST: None,
        }
        self.aliases: Dict[str, frozenset[str]] = {
            PROV_TRAIN: frozenset(),
            PROV_VALIDATION: frozenset(),
            PROV_TEST: frozenset(),
        }
        self._legacy_membership: Dict[str, frozenset[str]] = {}
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._connection: Optional[sqlite3.Connection] = None
        self._load()

    def _load(self) -> None:
        spec, spec_id = self._resolve_spec()
        if not isinstance(spec, Mapping):
            return
        self.split_spec_artifact_id = spec_id
        self.protocol_id = _text_or_none(spec.get("protocol_id"))
        self.record_id_column = _text_or_none(spec.get("record_id_column"))
        self.dataset_ids = {
            PROV_TRAIN: _text_or_none(spec.get("train_dataset_id")),
            PROV_VALIDATION: _text_or_none(spec.get("validation_dataset_id")),
            PROV_TEST: _text_or_none(spec.get("test_dataset_id")),
        }
        alias_resolver = getattr(legacy, "_dataset_aliases_for_provenance", None)
        for role, dataset_id in self.dataset_ids.items():
            if callable(alias_resolver):
                try:
                    aliases = alias_resolver(self.context, dataset_id)
                except Exception:
                    aliases = ()
            else:
                aliases = ()
            values = {str(value) for value in aliases or () if value}
            if dataset_id:
                values.add(dataset_id)
            self.aliases[role] = frozenset(values)

        manifest_payload = spec.get("split_manifest")
        if isinstance(manifest_payload, Mapping):
            self._build_manifest_index(SplitManifestRef.from_dict(manifest_payload))
        else:
            self._legacy_membership = {
                PROV_TRAIN: frozenset(
                    str(value) for value in spec.get("train_row_ids") or []
                ),
                PROV_VALIDATION: frozenset(
                    str(value) for value in spec.get("validation_row_ids") or []
                ),
                PROV_TEST: frozenset(
                    str(value) for value in spec.get("test_row_ids") or []
                ),
            }
        self.verified = True

    def _resolve_spec(self):
        artifacts = getattr(self.context, "artifacts", None)
        get = getattr(artifacts, "get", None)
        spec_id = self.model_payload.get("split_spec_artifact_id")
        if spec_id and callable(get):
            try:
                spec = get(spec_id)
            except Exception:
                spec = None
            if isinstance(spec, Mapping):
                return spec, str(spec_id)
        finder = getattr(legacy, "_find_split_spec_by_run", None)
        if callable(finder):
            try:
                return finder(
                    self.context,
                    run_id=self.model_payload.get("run_id"),
                    protocol_id=self.model_payload.get("protocol_id"),
                )
            except Exception:
                pass
        return None, None

    def _build_manifest_index(self, manifest: SplitManifestRef) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="astronomical-provenance-")
        database_path = Path(self._temp_dir.name) / "provenance.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE membership (record_id TEXT PRIMARY KEY, split_role TEXT NOT NULL)"
        )
        batch = []
        for row in iter_split_manifest(manifest, verify_checksum=True):
            record_id = row.get(manifest.record_id_column)
            role = _canonical_role(row.get(SPLIT_ROLE_COLUMN))
            if record_id is None or role is None:
                continue
            batch.append((str(record_id), role))
            if len(batch) >= 10_000:
                connection.executemany(
                    "INSERT INTO membership(record_id, split_role) VALUES (?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO membership(record_id, split_role) VALUES (?, ?)",
                batch,
            )
        connection.commit()
        self._connection = connection

    def classify_many(
        self,
        predict_dataset_id: Optional[str],
        row_ids: Sequence[Any],
    ) -> list[str]:
        if not self.verified:
            return [PROV_UNKNOWN for _ in row_ids]
        normalized = [str(value) for value in row_ids]
        roles = self._lookup_roles(normalized)
        return [
            self._classify_role(
                predict_dataset_id=predict_dataset_id,
                role=roles.get(row_id),
            )
            for row_id in normalized
        ]

    def _lookup_roles(self, row_ids: Sequence[str]) -> Dict[str, str]:
        if self._connection is None:
            result = {}
            for role, members in self._legacy_membership.items():
                for row_id in row_ids:
                    if row_id in members:
                        result[row_id] = role
            return result
        result: Dict[str, str] = {}
        for start in range(0, len(row_ids), 800):
            chunk = list(dict.fromkeys(row_ids[start : start + 800]))
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            cursor = self._connection.execute(
                f"SELECT record_id, split_role FROM membership WHERE record_id IN ({placeholders})",
                chunk,
            )
            result.update((str(record_id), str(role)) for record_id, role in cursor)
        return result

    def _classify_role(
        self,
        *,
        predict_dataset_id: Optional[str],
        role: Optional[str],
    ) -> str:
        if role is None:
            return PROV_NOVEL
        predicted = _text_or_none(predict_dataset_id)
        expected = self.dataset_ids.get(role)
        aliases = self.aliases.get(role, frozenset())
        if predicted and (predicted == expected or predicted in aliases):
            return role
        return PROV_NOVEL

    def summary(self, counts: Mapping[str, int]) -> Dict[str, Any]:
        normalized = {
            role: int(counts.get(role, 0) or 0)
            for role in (
                PROV_TRAIN,
                PROV_VALIDATION,
                PROV_TEST,
                PROV_NOVEL,
                PROV_UNKNOWN,
            )
        }
        reasons = {
            PROV_TRAIN: "Rows the model was fit on.",
            PROV_VALIDATION: "Rows used to select the best model.",
            PROV_TEST: "The model's held-out test rows.",
            PROV_NOVEL: "Rows not found in the model's split.",
            PROV_UNKNOWN: "No verifiable split membership was available.",
        }
        return {
            "verified": bool(self.verified),
            "split_spec_artifact_id": self.split_spec_artifact_id,
            "protocol_id": self.protocol_id,
            "record_id_column": self.record_id_column,
            "counts": normalized,
            "reportable_row_count": normalized[PROV_TEST] + normalized[PROV_NOVEL],
            "seen_row_count": normalized[PROV_TRAIN] + normalized[PROV_VALIDATION],
            "unverifiable_row_count": normalized[PROV_UNKNOWN],
            "dataset_ids": dict(self.dataset_ids),
            "dataset_aliases": {
                role: sorted(values) for role, values in self.aliases.items()
            },
            "reasons": {
                role: reasons[role]
                for role, count in normalized.items()
                if count
            },
        }

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> "StreamingProvenanceIndex":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _canonical_role(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    aliases = {
        "train": PROV_TRAIN,
        "training": PROV_TRAIN,
        "val": PROV_VALIDATION,
        "validation": PROV_VALIDATION,
        "test": PROV_TEST,
    }
    return aliases.get(text)


def _text_or_none(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None
