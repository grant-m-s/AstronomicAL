from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from .protocol import (
    SPLIT_DIGEST_ALGORITHM,
    SPLIT_IDENTITY_VERSION,
    Partition,
    PartitionRef,
    SourceRevision,
    SplitManifestRef,
)

SPLIT_MANIFEST_SCHEMA_VERSION = 2
SPLIT_ROLE_COLUMN = "split_role"
TARGET_SNAPSHOT_COLUMN = "target_snapshot"

@dataclass(frozen=True)
class PartitionManifestMetadata:
    """Metadata required to build a durable ``PartitionRef``."""

    name: str
    dataset_id: Optional[str]
    source: str = "split"
    classes: tuple[str, ...] = ()

class SplitManifestWriter:
    """Atomic incremental writer for split membership rows.

    The writer owns a temporary gzip-compressed JSONL file until ``finalize``
    succeeds. Callers may stream rows from a database cursor or dataset scan;
    no partition-wide record-ID list is required.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        run_id: str,
        source_dataset_id: str,
        protocol_id: str,
        record_id_column: str,
        target_column: Optional[str],
        source_revisions: Optional[Mapping[str, SourceRevision]] = None,
    ):
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{_safe_filename(run_id)}-split-manifest-"
            f"{uuid.uuid4().hex[:8]}.jsonl.gz"
        )
        self.final_path = root_path / filename
        self.temp_path = self.final_path.with_suffix(self.final_path.suffix + ".tmp")
        self.source_dataset_id = str(source_dataset_id)
        self.protocol_id = str(protocol_id)
        self.record_id_column = str(record_id_column)
        self.target_column = str(target_column) if target_column else None
        self.source_revisions = {
            str(dataset_id): (
                revision
                if isinstance(revision, SourceRevision)
                else SourceRevision.from_dict(revision)
            )
            for dataset_id, revision in dict(source_revisions or {}).items()
        }
        self.created_at = time.time()
        self.role_counts: Dict[str, int] = {}
        self.row_count = 0
        self._membership_digests: Dict[str, Any] = {}
        self._target_digests: Dict[str, Any] = {}
        self._last_record_keys: Dict[str, str] = {}
        self._handle: Optional[TextIO] = gzip.open(
            self.temp_path,
            "wt",
            encoding="utf-8",
            newline="\n",
        )
        self._finalized = False

    def write(
        self,
        *,
        role: str,
        record_id: Any,
        target: Any = None,
    ) -> None:
        handle = self._require_open()
        canonical_role = str(role)
        record_scalar = json_scalar(record_id)
        record_key = canonical_json(record_scalar)

        previous_key = self._last_record_keys.get(canonical_role)
        if previous_key is not None and record_key <= previous_key:
            raise ValueError(
                "Split manifest rows must be written in strict canonical "
                f"record-ID order within role {canonical_role!r}."
            )
        self._last_record_keys[canonical_role] = record_key

        target_scalar = json_scalar(target)
        row = {
            self.record_id_column: record_scalar,
            SPLIT_ROLE_COLUMN: canonical_role,
        }
        if self.target_column:
            row[TARGET_SNAPSHOT_COLUMN] = target_scalar

        handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        handle.write("\n")

        membership_digest = self._membership_digests.setdefault(
            canonical_role,
            hashlib.sha256(),
        )
        update_canonical_digest(membership_digest, record_scalar)

        if self.target_column:
            target_digest = self._target_digests.setdefault(
                canonical_role,
                hashlib.sha256(),
            )
            update_canonical_digest(
                target_digest,
                record_scalar,
                target_scalar,
            )

        self.role_counts[canonical_role] = (
            self.role_counts.get(canonical_role, 0) + 1
        )
        self.row_count += 1

    def finalize(
        self,
        partitions: Mapping[str, PartitionManifestMetadata],
    ) -> Tuple[SplitManifestRef, Dict[str, PartitionRef]]:
        if self._finalized:
            raise RuntimeError("Split manifest writer has already been finalized.")
        try:
            self._close_handle()
            os.replace(self.temp_path, self.final_path)
            manifest = self._manifest_ref()
            refs: Dict[str, PartitionRef] = {}
            for role, metadata in dict(partitions or {}).items():
                canonical_role = str(role)
                count = int(self.role_counts.get(canonical_role, 0))
                source_revision = manifest.source_revision(metadata.dataset_id)
                membership_sha256 = manifest.membership_sha256_by_role.get(
                    canonical_role,
                    "",
                )
                target_sha256 = manifest.target_sha256_by_role.get(
                    canonical_role,
                    "",
                )
                refs[canonical_role] = PartitionRef(
                    name=str(metadata.name),
                    role=canonical_role,
                    row_count=count,
                    manifest=manifest,
                    classes=list(metadata.classes),
                    dataset_id=metadata.dataset_id,
                    source=str(metadata.source),
                    fingerprint=partition_fingerprint(
                        manifest,
                        canonical_role,
                        dataset_id=metadata.dataset_id,
                    ),
                    identity_version=manifest.identity_version,
                    digest_algorithm=manifest.digest_algorithm,
                    membership_sha256=membership_sha256,
                    target_sha256=target_sha256,
                    source_revision_sha256=(
                        source_revision.revision_sha256
                        if source_revision is not None
                        else ""
                    ),
                )
            self._finalized = True
            return manifest, refs
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        self._close_handle()
        self.temp_path.unlink(missing_ok=True)
        if not self._finalized:
            self.final_path.unlink(missing_ok=True)

    def _manifest_ref(self) -> SplitManifestRef:
        columns = [self.record_id_column, SPLIT_ROLE_COLUMN]
        if self.target_column:
            columns.append(TARGET_SNAPSHOT_COLUMN)
        return SplitManifestRef(
            schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
            storage="local_file",
            uri=str(self.final_path),
            format="jsonl.gz",
            created_at=self.created_at,
            source_dataset_id=self.source_dataset_id,
            protocol_id=self.protocol_id,
            record_id_column=self.record_id_column,
            target_column=self.target_column,
            row_count=int(self.row_count),
            role_counts=dict(self.role_counts),
            columns=columns,
            sha256=file_sha256(self.final_path),
            size_bytes=self.final_path.stat().st_size,
            identity_version=SPLIT_IDENTITY_VERSION,
            digest_algorithm=SPLIT_DIGEST_ALGORITHM,
            membership_sha256_by_role={
                role: digest.hexdigest()
                for role, digest in self._membership_digests.items()
            },
            target_sha256_by_role={
                role: digest.hexdigest()
                for role, digest in self._target_digests.items()
            },
            source_revisions=dict(self.source_revisions),
        )

    def _require_open(self) -> TextIO:
        if self._finalized or self._handle is None:
            raise RuntimeError("Split manifest writer is closed.")
        return self._handle

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SplitManifestWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._finalized:
            self.abort()
        return False

def create_split_manifest(
    *,
    root: Path | str,
    run_id: str,
    source_dataset_id: str,
    protocol_id: str,
    record_id_column: str,
    target_column: Optional[str],
    partitions: Mapping[str, Partition],
) -> Tuple[SplitManifestRef, Dict[str, PartitionRef]]:
    """Persist legacy materialised partitions as a durable manifest.

    This compatibility function now delegates to ``SplitManifestWriter``. New
    split generation should stream rows directly into the writer instead of
    creating ``Partition.record_ids`` and ``Partition.labels`` first.
    """

    writer = SplitManifestWriter(
        root=root,
        run_id=run_id,
        source_dataset_id=source_dataset_id,
        protocol_id=protocol_id,
        record_id_column=record_id_column,
        target_column=target_column,
    )
    metadata: Dict[str, PartitionManifestMetadata] = {}
    try:
        for role, partition in dict(partitions or {}).items():
            if partition is None:
                continue
            metadata[str(role)] = PartitionManifestMetadata(
                name=partition.name,
                dataset_id=partition.dataset_id,
                source=partition.source,
                classes=tuple(partition.classes),
            )
            _write_partition_rows(
                writer,
                role=str(role),
                partition=partition,
                include_target=bool(target_column),
            )
        return writer.finalize(metadata)
    except Exception:
        writer.abort()
        raise

def iter_split_manifest(
    manifest: SplitManifestRef | Mapping[str, Any],
    *,
    role: Optional[str] = None,
    verify_checksum: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield manifest rows without materialising the membership table."""

    ref = (
        manifest
        if isinstance(manifest, SplitManifestRef)
        else SplitManifestRef.from_dict(manifest)
    )
    path = Path(ref.uri)
    if verify_checksum:
        verify_split_manifest(ref)
    if ref.format not in {"jsonl", "jsonl.gz"}:
        raise ValueError(f"Unsupported split manifest format {ref.format!r}.")

    with _open_manifest(path, ref.format) as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid split manifest JSON at {path}:{line_number}."
                ) from exc
            if role is not None and str(row.get(SPLIT_ROLE_COLUMN)) != str(role):
                continue
            yield row

def iter_partition_rows(
    partition: PartitionRef,
    *,
    verify_checksum: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield rows for one partition role without loading its membership table."""

    yield from iter_split_manifest(
        partition.manifest,
        role=partition.role,
        verify_checksum=verify_checksum,
    )

def iter_partition_row_batches(
    partition: PartitionRef,
    *,
    batch_size: int,
    verify_checksum: bool = False,
) -> Iterator[List[Dict[str, Any]]]:
    """Yield bounded lists of manifest rows for a partition."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be greater than zero.")

    batch: List[Dict[str, Any]] = []
    for row in iter_partition_rows(
        partition,
        verify_checksum=verify_checksum,
    ):
        batch.append(row)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch

def iter_partition_record_ids(
    partition: PartitionRef,
    *,
    verify_checksum: bool = False,
) -> Iterator[Any]:
    for row in iter_partition_rows(
        partition, verify_checksum=verify_checksum
    ):
        yield row[partition.manifest.record_id_column]

def iter_partition_record_id_batches(
    partition: PartitionRef,
    *,
    batch_size: int,
    verify_checksum: bool = False,
) -> Iterator[List[Any]]:
    record_id_column = partition.manifest.record_id_column
    for rows in iter_partition_row_batches(
        partition,
        batch_size=batch_size,
        verify_checksum=verify_checksum,
    ):
        yield [row[record_id_column] for row in rows]

def verify_split_manifest(manifest: SplitManifestRef | Mapping[str, Any]) -> None:
    ref = (
        manifest
        if isinstance(manifest, SplitManifestRef)
        else SplitManifestRef.from_dict(manifest)
    )
    path = Path(ref.uri)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    actual_size = path.stat().st_size
    if ref.size_bytes and actual_size != ref.size_bytes:
        raise ValueError(
            f"Split manifest size mismatch for {path}: expected {ref.size_bytes}, "
            f"got {actual_size}."
        )
    if ref.sha256:
        actual_hash = file_sha256(path)
        if actual_hash.lower() != ref.sha256.lower():
            raise ValueError(
                f"Split manifest checksum mismatch for {path}: expected "
                f"{ref.sha256}, got {actual_hash}."
            )

def partition_fingerprint(
    manifest: SplitManifestRef,
    role: str,
    *,
    dataset_id: Optional[str] = None,
) -> str:
    """Return the path-independent semantic identity of one partition."""

    canonical_role = str(role)
    source_revision = manifest.source_revision(dataset_id)
    payload = {
        "identity_version": int(manifest.identity_version),
        "digest_algorithm": manifest.digest_algorithm,
        "role": canonical_role,
        "row_count": int(manifest.role_counts.get(canonical_role, 0)),
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "protocol_id": manifest.protocol_id,
        "membership_sha256": manifest.membership_sha256_by_role.get(
            canonical_role,
            "",
        ),
        "target_sha256": manifest.target_sha256_by_role.get(
            canonical_role,
            "",
        ),
        "source_revision_sha256": (
            source_revision.revision_sha256
            if source_revision is not None
            else ""
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_scalar(value: Any) -> Any:
    """Convert common dataframe values to canonical JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return (
            value
            if value == value and value not in (float("inf"), float("-inf"))
            else None
        )
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): json_scalar(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [json_scalar(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [json_scalar(item) for item in value]
        return sorted(items, key=canonical_json)
    item = getattr(value, "item", None)
    if callable(item):
        return json_scalar(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_scalar(tolist())
        except Exception:
            pass
    return str(value)


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for identity hashing."""

    return json.dumps(
        json_scalar(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def update_canonical_digest(digest: Any, *values: Any) -> None:
    """Append length-framed canonical values to a hashlib-compatible digest."""

    for value in values:
        encoded = canonical_json(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _write_partition_rows(
    writer: SplitManifestWriter,
    *,
    role: str,
    partition: Partition,
    include_target: bool,
) -> int:
    record_ids = partition.record_ids
    labels = partition.labels
    if include_target and len(labels) != len(record_ids):
        raise ValueError(
            f"Partition {partition.name!r} contains {len(record_ids)} record IDs "
            f"but {len(labels)} labels."
        )

    rows = [
        (
            json_scalar(record_id),
            json_scalar(labels[index]) if include_target else None,
        )
        for index, record_id in enumerate(record_ids)
    ]
    rows.sort(key=lambda row: canonical_json(row[0]))

    for record_id, target in rows:
        writer.write(
            role=role,
            record_id=record_id,
            target=target,
        )
    return len(rows)


@contextmanager
def _open_manifest(path: Path, format_name: str) -> Iterator[TextIO]:
    if format_name == "jsonl.gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
        return
    with path.open("r", encoding="utf-8") as handle:
        yield handle

def _safe_filename(value: Any) -> str:
    text = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value or "")
    ).strip("._-")
    return text or "run"