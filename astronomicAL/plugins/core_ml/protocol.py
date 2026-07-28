from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

SPLIT_IDENTITY_VERSION = 1
SPLIT_DIGEST_ALGORITHM = "sha256-framed-json-v1"

def resolve_selection_mode(metric: str, mode: str = "auto") -> str:
    if mode in ("max", "min"):
        return mode
    metric_name = str(metric or "").lower()
    return (
        "min"
        if any(token in metric_name for token in ("loss", "error", "mae", "mse", "rmse"))
        else "max"
    )

@dataclass
class ProtocolConfig:
    """Experiment protocol enforced by the harness.

    ``split_strategy`` controls only how the selected/main dataset is split when
    ``validation_source`` or ``test_source`` is ``"split"``.

    ``validation_source``:
      - ``split``: create validation from the selected dataset
      - ``dataset``: use ``protocol_validation_dataset_id``

    ``test_source``:
      - ``split``: create test from the selected dataset
      - ``dataset``: use ``protocol_test_dataset_id``
      - ``none``: no test set
    """

    split_strategy: str = "random"  # random | by_group | temporal | predefined
    validation_source: str = "split"  # split | dataset
    test_source: str = "split"  # split | dataset | none
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    group_column: Optional[str] = None
    split_column: Optional[str] = None
    validation_size: float = 0.1
    test_size: float = 0.2
    materialize_split_datasets: bool = True
    split_dataset_prefix: Optional[str] = None
    selection_metric: str = "val_accuracy"
    selection_mode: str = "auto"
    random_state: int = 42
    protocol_id: str = ""

    def resolved_mode(self) -> str:
        return resolve_selection_mode(self.selection_metric, self.selection_mode)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ProtocolConfig":
        def num(key: str, default: float) -> float:
            try:
                return float(params.get(key, default))
            except Exception:
                return default

        cfg = cls(
            split_strategy=str(params.get("protocol_split_strategy", "random")),
            validation_source=str(params.get("protocol_validation_source", "split")),
            test_source=str(params.get("protocol_test_source", "split")),
            validation_dataset_id=params.get("protocol_validation_dataset_id") or None,
            test_dataset_id=params.get("protocol_test_dataset_id") or None,
            group_column=params.get("protocol_group_column") or None,
            split_column=params.get("protocol_split_column") or None,
            validation_size=num("protocol_validation_size", 0.1),
            test_size=num("protocol_test_size", 0.2),
            materialize_split_datasets=_bool_param(
                params.get("protocol_materialize_split_datasets"), True
            ),
            split_dataset_prefix=params.get("protocol_split_dataset_prefix") or None,
            selection_metric=str(
                params.get("protocol_selection_metric", "val_accuracy")
            ),
            selection_mode=str(params.get("protocol_selection_mode", "auto")),
            random_state=int(params.get("protocol_random_state", 42)),
        )

        allowed_split_strategies = {"random", "by_group", "temporal", "predefined"}
        if cfg.split_strategy not in allowed_split_strategies:
            raise ValueError(
                f"Unknown protocol_split_strategy {cfg.split_strategy!r}. "
                f"Expected one of {sorted(allowed_split_strategies)}."
            )
        if cfg.validation_source not in {"split", "dataset"}:
            raise ValueError(
                "protocol_validation_source must be 'split' or 'dataset'."
            )
        if cfg.test_source not in {"split", "dataset", "none"}:
            raise ValueError(
                "protocol_test_source must be 'split', 'dataset', or 'none'."
            )

        train_dataset_id = params.get("dataset_id")
        if cfg.validation_source == "dataset":
            if not cfg.validation_dataset_id:
                raise ValueError(
                    "Validation source is 'dataset', but no validation dataset "
                    "was selected."
                )
            if cfg.validation_dataset_id == train_dataset_id:
                raise ValueError(
                    "Validation dataset must be different from the selected "
                    "training dataset."
                )

        if cfg.test_source == "dataset":
            if not cfg.test_dataset_id:
                raise ValueError(
                    "Test source is 'dataset', but no test dataset was selected."
                )
            if cfg.test_dataset_id == train_dataset_id:
                raise ValueError(
                    "Test dataset must be different from the selected training dataset."
                )

        if (
            cfg.validation_source == "dataset"
            and cfg.test_source == "dataset"
            and cfg.validation_dataset_id
            and cfg.test_dataset_id
            and cfg.validation_dataset_id == cfg.test_dataset_id
        ):
            raise ValueError("Validation and test datasets must be different.")

        val_from_split = cfg.validation_source == "split"
        test_from_split = cfg.test_source == "split"
        if val_from_split and cfg.split_strategy != "predefined":
            if cfg.validation_size <= 0:
                raise ValueError(
                    "protocol_validation_size must be > 0 when validation "
                    "is split from the selected dataset."
                )
        if test_from_split and cfg.split_strategy != "predefined":
            if cfg.test_size <= 0:
                raise ValueError(
                    "protocol_test_size must be > 0 when test is split from "
                    "the selected dataset. Choose 'No test set' instead."
                )

        if cfg.split_strategy != "predefined":
            split_total = 0.0
            if val_from_split:
                split_total += cfg.validation_size
            if test_from_split:
                split_total += cfg.test_size
            if split_total >= 1.0:
                raise ValueError(
                    "Fractions split from the selected dataset must sum to < 1.0."
                )

        if cfg.split_strategy in {"by_group", "temporal"} and (
            val_from_split or test_from_split
        ):
            if not cfg.group_column:
                raise ValueError(
                    f"{cfg.split_strategy!r} split requires protocol_group_column."
                )

        if cfg.split_strategy == "predefined" and (
            val_from_split or test_from_split
        ):
            if not cfg.split_column:
                raise ValueError("predefined split requires protocol_split_column.")

        cfg.protocol_id = _stable_protocol_id(cfg)
        return cfg

def _bool_param(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default

def _stable_protocol_id(cfg: ProtocolConfig) -> str:
    raw = "|".join(
        str(value)
        for value in (
            cfg.split_strategy,
            cfg.validation_source,
            cfg.test_source,
            cfg.validation_dataset_id,
            cfg.test_dataset_id,
            cfg.group_column,
            cfg.split_column,
            cfg.validation_size,
            cfg.test_size,
            cfg.selection_metric,
            cfg.resolved_mode(),
            cfg.random_state,
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

@dataclass
class DataBinding:
    """Resolved column binding owned by the harness, not the recipe."""

    record_id_column: str
    target_column: Optional[str]
    input_columns: List[str] = field(default_factory=list)
    image_column: Optional[str] = None

@dataclass(frozen=True)
class SourceRevision:
    """Stable identity for the ML-relevant contents of one dataset source.

    The revision is independent of source paths and physical scan order. It
    covers the bounded scan scope, relevant columns, their dtypes, and a
    canonical record-ID-keyed digest of the values used by splitting,
    training, or evaluation.
    """

    schema_version: int
    identity_version: int
    digest_algorithm: str
    dataset_id: str
    backend: str
    scope: str
    row_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    revision_sha256: str
    selected_row_ids_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRevision":
        payload = dict(value or {})
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            identity_version=int(
                payload.get("identity_version", SPLIT_IDENTITY_VERSION)
            ),
            digest_algorithm=str(
                payload.get("digest_algorithm", SPLIT_DIGEST_ALGORITHM)
            ),
            dataset_id=str(payload.get("dataset_id", "")),
            backend=str(payload.get("backend", "unknown")),
            scope=str(payload.get("scope", "full_dataset")),
            row_count=int(payload.get("row_count", 0)),
            columns=[str(column) for column in payload.get("columns") or []],
            dtypes={
                str(column): str(dtype)
                for column, dtype in dict(payload.get("dtypes") or {}).items()
            },
            revision_sha256=str(payload.get("revision_sha256", "")),
            selected_row_ids_sha256=(
                str(payload["selected_row_ids_sha256"])
                if payload.get("selected_row_ids_sha256")
                else None
            ),
        )

@dataclass(frozen=True)
class SplitManifestRef:
    """JSON-safe reference to a durable split-membership table."""

    schema_version: int
    storage: str
    uri: str
    format: str
    created_at: float
    source_dataset_id: str
    protocol_id: str
    record_id_column: str
    target_column: Optional[str]
    row_count: int
    role_counts: Dict[str, int]
    columns: List[str]
    sha256: str
    size_bytes: int
    identity_version: int = SPLIT_IDENTITY_VERSION
    digest_algorithm: str = SPLIT_DIGEST_ALGORITHM
    membership_sha256_by_role: Dict[str, str] = field(default_factory=dict)
    target_sha256_by_role: Dict[str, str] = field(default_factory=dict)
    source_revisions: Dict[str, SourceRevision] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_revisions"] = {
            dataset_id: revision.to_dict()
            for dataset_id, revision in self.source_revisions.items()
        }
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SplitManifestRef":
        payload = dict(value or {})
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            storage=str(payload.get("storage", "local_file")),
            uri=str(payload["uri"]),
            format=str(payload.get("format", "jsonl.gz")),
            created_at=float(payload.get("created_at", 0.0)),
            source_dataset_id=str(payload.get("source_dataset_id", "")),
            protocol_id=str(payload.get("protocol_id", "")),
            record_id_column=str(payload.get("record_id_column", "record_id")),
            target_column=(
                str(payload["target_column"])
                if payload.get("target_column") is not None
                else None
            ),
            row_count=int(payload.get("row_count", 0)),
            role_counts={
                str(role): int(count)
                for role, count in dict(payload.get("role_counts") or {}).items()
            },
            columns=[str(column) for column in payload.get("columns") or []],
            sha256=str(payload.get("sha256", "")),
            size_bytes=int(payload.get("size_bytes", 0)),
            identity_version=int(
                payload.get("identity_version", SPLIT_IDENTITY_VERSION)
            ),
            digest_algorithm=str(
                payload.get("digest_algorithm", SPLIT_DIGEST_ALGORITHM)
            ),
            membership_sha256_by_role={
                str(role): str(digest)
                for role, digest in dict(
                    payload.get("membership_sha256_by_role") or {}
                ).items()
            },
            target_sha256_by_role={
                str(role): str(digest)
                for role, digest in dict(
                    payload.get("target_sha256_by_role") or {}
                ).items()
            },
            source_revisions={
                str(dataset_id): (
                    revision
                    if isinstance(revision, SourceRevision)
                    else SourceRevision.from_dict(revision)
                )
                for dataset_id, revision in dict(
                    payload.get("source_revisions") or {}
                ).items()
            },
        )

    def source_revision(
        self,
        dataset_id: Optional[str],
    ) -> Optional[SourceRevision]:
        if dataset_id is None:
            return None
        return self.source_revisions.get(str(dataset_id))


@dataclass(frozen=True)
class PartitionRef:
    """A partition represented by a role in a durable split manifest.

    It intentionally contains counts and stable semantic identity rather than
    every record ID. The identity fields are independent of manifest paths and
    compressed-file bytes.
    """

    name: str
    role: str
    row_count: int
    manifest: SplitManifestRef
    classes: List[str] = field(default_factory=list)
    dataset_id: Optional[str] = None
    source: str = "split"  # split | dataset | none
    fingerprint: str = ""
    identity_version: int = SPLIT_IDENTITY_VERSION
    digest_algorithm: str = SPLIT_DIGEST_ALGORITHM
    membership_sha256: str = ""
    target_sha256: str = ""
    source_revision_sha256: str = ""

    def __len__(self) -> int:
        return int(self.row_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "row_count": int(self.row_count),
            "manifest": self.manifest.to_dict(),
            "classes": list(self.classes),
            "dataset_id": self.dataset_id,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "identity_version": int(self.identity_version),
            "digest_algorithm": self.digest_algorithm,
            "membership_sha256": self.membership_sha256,
            "target_sha256": self.target_sha256,
            "source_revision_sha256": self.source_revision_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PartitionRef":
        payload = dict(value or {})
        manifest = SplitManifestRef.from_dict(payload["manifest"])
        role = str(payload.get("role", payload.get("name", "partition")))
        dataset_id = (
            str(payload["dataset_id"])
            if payload.get("dataset_id") is not None
            else None
        )
        source_revision = manifest.source_revision(dataset_id)
        return cls(
            name=str(payload.get("name", role)),
            role=role,
            row_count=int(payload.get("row_count", 0)),
            manifest=manifest,
            classes=[str(item) for item in payload.get("classes") or []],
            dataset_id=dataset_id,
            source=str(payload.get("source", "split")),
            fingerprint=str(payload.get("fingerprint", "")),
            identity_version=int(
                payload.get("identity_version", manifest.identity_version)
            ),
            digest_algorithm=str(
                payload.get("digest_algorithm", manifest.digest_algorithm)
            ),
            membership_sha256=str(
                payload.get("membership_sha256")
                or manifest.membership_sha256_by_role.get(role, "")
            ),
            target_sha256=str(
                payload.get("target_sha256")
                or manifest.target_sha256_by_role.get(role, "")
            ),
            source_revision_sha256=str(
                payload.get("source_revision_sha256")
                or (
                    source_revision.revision_sha256
                    if source_revision is not None
                    else ""
                )
            ),
        )


@dataclass
class Partition:
    """Legacy materialised partition retained during the streaming migration."""

    name: str
    record_ids: List[str]
    labels: List[str]
    classes: List[str]
    dataset_id: Optional[str] = None
    source: str = "split"  # split | dataset | none

    def __len__(self) -> int:
        return len(self.record_ids)

PartitionLike = Partition | PartitionRef

@dataclass
class Partitions:
    train: PartitionLike
    val: PartitionLike
    test: Optional[PartitionLike]
    strategy: str
    validation_source: str
    test_source: str
    group_column: Optional[str]
    random_state: int
    protocol_id: str
    target_column: str
    record_id_column: str
    train_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    materialized_split_dataset_ids: Dict[str, str] = field(default_factory=dict)
    split_manifest: Optional[SplitManifestRef] = None
    partition_refs: Dict[str, PartitionRef] = field(default_factory=dict)
    split_generation: Dict[str, Any] = field(default_factory=dict)

    def partition_ref(self, role: str) -> Optional[PartitionRef]:
        """Return the durable reference for a canonical partition role."""

        canonical = _canonical_partition_role(role)
        ref = self.partition_refs.get(canonical)
        if ref is not None:
            return ref

        value = {
            "train": self.train,
            "validation": self.val,
            "test": self.test,
        }[canonical]
        return value if isinstance(value, PartitionRef) else None

    def partition(self, role: str) -> Optional[Partition]:
        """Return the legacy materialised partition when it is still present."""

        canonical = _canonical_partition_role(role)
        value = {
            "train": self.train,
            "validation": self.val,
            "test": self.test,
        }[canonical]
        return value if isinstance(value, Partition) else None

    def attach_manifest(
        self,
        manifest: SplitManifestRef,
        refs: Mapping[str, PartitionRef],
    ) -> None:
        """Attach durable memberships while legacy partitions remain usable."""

        self.split_manifest = manifest
        self.partition_refs = {
            _canonical_partition_role(role): ref for role, ref in refs.items()
        }

def _canonical_partition_role(role: str) -> str:
    value = str(role or "").strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise KeyError(f"Unknown partition role {role!r}.") from exc

@dataclass
class TrainingComponents:
    """What ``configure_training`` returns: recipe policy, not protocol policy."""

    optimizer: Any
    scheduler: Any = None
    criterion: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TargetSpec:
    """Meaning and width of the model output for a run."""

    kind: str = "classification"  # classification | regression
    classes: List[str] = field(default_factory=list)
    n_outputs: int = 1

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_outputs(self) -> int:
        if self.kind == "classification":
            return len(self.classes)
        return int(self.n_outputs)