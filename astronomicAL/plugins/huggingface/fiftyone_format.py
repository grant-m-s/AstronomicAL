from __future__ import annotations

"""Small, dependency-free helpers for native FiftyOne dataset repositories.

FiftyOne's native disk format stores dataset schema in ``metadata.json`` and
sample records in ``samples.json``.  Hugging Face repositories produced by
FiftyOne commonly retain these files.  AstronomicAL does not need to import
FiftyOne itself to recover simple image-classification ground truth from that
format; this module only reads the serialized metadata that the repository
already declares.

The helpers are deliberately structural and conservative:

* a target field is selected only from declared ``Classification`` fields;
* prediction-like field names are never silently treated as ground truth;
* the original sample fields remain available as provenance;
* filepath matching prefers exact/suffix matches and uses basename matching
  only when the basename is unique in the sample manifest.

This module contains no Hub/network code.  ``service.py`` owns acquisition and
cache reuse, while ``importer.py`` decides how recovered annotations are
represented in AstronomicAL.
"""

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

_CLASSIFICATION_SUFFIX = ".labels.Classification"
_PREDICTION_TOKENS = (
    "prediction",
    "predictions",
    "pred",
    "inference",
    "output",
    "logit",
)

@dataclass(frozen=True)
class FiftyOneSchema:
    media_type: str = ""
    ground_truth_field: str = ""
    classification_fields: tuple[str, ...] = ()
    prediction_fields: tuple[str, ...] = ()
    sample_fields: tuple[str, ...] = ()

    @property
    def usable_classification_ground_truth(self) -> bool:
        return bool(self.ground_truth_field)

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "ground_truth_field": self.ground_truth_field,
            "classification_fields": list(self.classification_fields),
            "prediction_fields": list(self.prediction_fields),
            "sample_fields": list(self.sample_fields),
        }

@dataclass(frozen=True)
class FiftyOneClassificationAnnotation:
    filepath: str
    target_label: str = ""
    target_field: str = ""
    predictions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sample_id: str = ""
    tags: tuple[str, ...] = ()

@dataclass
class FiftyOneSampleIndex:
    target_field: str
    prediction_fields: tuple[str, ...]
    by_path: dict[str, FiftyOneClassificationAnnotation] = field(default_factory=dict)
    by_basename: dict[str, FiftyOneClassificationAnnotation] = field(default_factory=dict)
    ambiguous_basenames: set[str] = field(default_factory=set)
    rows: int = 0
    labelled_rows: int = 0

    @property
    def coverage(self) -> float:
        return self.labelled_rows / max(self.rows, 1)

    @property
    def distinct_labels(self) -> tuple[str, ...]:
        return tuple(sorted({item.target_label for item in self.by_path.values() if item.target_label}))

def parse_fiftyone_schema(payload: Mapping[str, Any] | None) -> FiftyOneSchema | None:
    """Return a conservative schema description for a FiftyOne metadata payload."""
    if not isinstance(payload, Mapping):
        return None

    raw_fields = payload.get("sample_fields")
    if not isinstance(raw_fields, list):
        return None

    field_names: list[str] = []
    classification_fields: list[str] = []
    for item in raw_fields:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        field_names.append(name)
        embedded = str(item.get("embedded_doc_type") or "")
        if embedded.endswith(_CLASSIFICATION_SUFFIX) or embedded.endswith("Classification"):
            classification_fields.append(name)

    if not field_names:
        return None

    prediction_fields = tuple(
        name for name in classification_fields if _looks_prediction_like(name)
    )
    target_candidates = [
        name for name in classification_fields if name not in prediction_fields
    ]
    ground_truth_field = _choose_ground_truth_field(target_candidates)

    return FiftyOneSchema(
        media_type=str(payload.get("media_type") or ""),
        ground_truth_field=ground_truth_field,
        classification_fields=tuple(classification_fields),
        prediction_fields=prediction_fields,
        sample_fields=tuple(field_names),
    )

def build_classification_sample_index(
    payload: Any,
    schema: FiftyOneSchema,
) -> FiftyOneSampleIndex:
    """Build a filepath-keyed classification index from ``samples.json``.

    FiftyOne exports have used both a top-level list and an object containing a
    ``samples`` list across versions/workflows, so both are accepted.
    """
    rows = _sample_rows(payload)
    index = FiftyOneSampleIndex(
        target_field=str(schema.ground_truth_field or ""),
        prediction_fields=tuple(schema.prediction_fields),
    )

    basename_candidates: dict[str, list[FiftyOneClassificationAnnotation]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index.rows += 1
        filepath = _normalise_sample_filepath(row.get("filepath"))
        if not filepath:
            continue

        target_label = _classification_label(row.get(schema.ground_truth_field))
        if target_label:
            index.labelled_rows += 1

        predictions: dict[str, Mapping[str, Any]] = {}
        for field_name in schema.prediction_fields:
            value = row.get(field_name)
            label = _classification_label(value)
            confidence = _classification_confidence(value)
            if label or confidence is not None:
                predictions[field_name] = {
                    "label": label,
                    "confidence": confidence,
                }

        annotation = FiftyOneClassificationAnnotation(
            filepath=filepath,
            target_label=target_label,
            target_field=str(schema.ground_truth_field or ""),
            predictions=predictions,
            sample_id=_sample_id(row.get("id") or row.get("_id")),
            tags=_normalise_tags(row.get("tags")),
        )
        for key in _filepath_keys(filepath):
            index.by_path.setdefault(key, annotation)

        basename = PurePosixPath(filepath).name.casefold()
        if basename:
            basename_candidates.setdefault(basename, []).append(annotation)

    for basename, candidates in basename_candidates.items():
        unique_paths = {item.filepath for item in candidates}
        if len(unique_paths) == 1:
            index.by_basename[basename] = candidates[0]
        else:
            index.ambiguous_basenames.add(basename)

    return index

def match_classification_annotation(
    index: FiftyOneSampleIndex | None,
    repo_path: str,
) -> FiftyOneClassificationAnnotation | None:
    if index is None:
        return None
    path = _normalise_sample_filepath(repo_path)
    if not path:
        return None

    for key in _filepath_keys(path):
        found = index.by_path.get(key)
        if found is not None:
            return found

    basename = PurePosixPath(path).name.casefold()
    if not basename or basename in index.ambiguous_basenames:
        return None
    return index.by_basename.get(basename)

def _sample_rows(payload: Any) -> Iterable[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("samples", "records", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return ()

def _classification_label(value: Any) -> str:
    if isinstance(value, Mapping):
        raw = value.get("label")
        if raw is not None:
            return str(raw).strip()
    if isinstance(value, str):
        return value.strip()
    return ""

def _classification_confidence(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("confidence")
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None

def _normalise_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()

def _sample_id(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("$oid", "oid", "id"):
            raw = value.get(key)
            if raw is not None:
                return str(raw)
    return str(value or "")

def _choose_ground_truth_field(candidates: list[str]) -> str:
    if not candidates:
        return ""

    def score(name: str) -> tuple[int, int, str]:
        lowered = name.casefold()
        rank = 0
        if lowered == "ground_truth":
            rank = 100
        elif "ground_truth" in lowered or "groundtruth" in lowered:
            rank = 95
        elif lowered in {"target", "target_label", "label", "labels"}:
            rank = 90
        elif "target" in lowered:
            rank = 80
        elif "label" in lowered:
            rank = 70
        return (-rank, len(name), lowered)

    return sorted(candidates, key=score)[0]

def _looks_prediction_like(name: str) -> bool:
    lowered = str(name or "").casefold()
    tokens = [token for token in lowered.replace("-", "_").split("_") if token]
    return any(token in tokens or token in lowered for token in _PREDICTION_TOKENS)

def _normalise_sample_filepath(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    while text.startswith("./"):
        text = text[2:]
    text = recompact_slashes(text)
    return text

def _filepath_keys(path: str) -> tuple[str, ...]:
    normalised = _normalise_sample_filepath(path)
    if not normalised:
        return ()
    candidates: list[str] = [normalised.casefold()]

    # FiftyOne exports may store absolute source paths even though the Hub copy
    # places media under ``data/``.  A suffix beginning at the data directory is
    # stable across machines and is therefore a safer cross-workspace key.
    lowered = normalised.casefold()
    marker = "/data/"
    pos = lowered.rfind(marker)
    if pos >= 0:
        candidates.append(normalised[pos + 1 :].casefold())
    elif lowered.startswith("data/"):
        candidates.append(lowered)

    return tuple(dict.fromkeys(candidates))

def recompact_slashes(value: str) -> str:
    text = str(value)
    while "//" in text:
        text = text.replace("//", "/")
    return text

__all__ = [
    "FiftyOneSchema",
    "FiftyOneClassificationAnnotation",
    "FiftyOneSampleIndex",
    "parse_fiftyone_schema",
    "build_classification_sample_index",
    "match_classification_annotation",
]
