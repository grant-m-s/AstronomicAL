from __future__ import annotations

"""Inspection-only contracts for structured media annotation repositories.

This module deliberately does not register datasets, decode masks, or resolve media.
It recognises lightweight repository/card evidence and describes the annotation
files that can be acquired safely. Modality-specific runtime behaviour belongs to
plugins such as ``core.video`` and ``vision.segmentation``.
"""

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

_JSON_SUFFIX = ".json"
_SPLIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("train", re.compile(r"(?:^|[_\-/.])train(?:ing)?(?:$|[_\-/.])", re.IGNORECASE)),
    ("validation", re.compile(r"(?:^|[_\-/.])(?:val|valid|validation|dev)(?:$|[_\-/.])", re.IGNORECASE)),
    ("test", re.compile(r"(?:^|[_\-/.])(?:test|testing|eval|evaluation)(?:$|[_\-/.])", re.IGNORECASE)),
)

@dataclass(frozen=True)
class HFStructuredAnnotationFile:
    path: str
    split: str = ""
    variant: str = "base"
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class HFStructuredAnnotationProfile:
    """Provider-neutral description of a structured annotation repository.

    The profile describes what was *observed* from repository metadata and the
    dataset card. It is intentionally not a parsed annotation payload: very large
    JSON files remain external/cache-backed until a modality plugin consumes them.
    """

    format_id: str
    format_label: str
    modalities: tuple[str, ...]
    task_hints: tuple[str, ...]
    annotation_files: tuple[HFStructuredAnnotationFile, ...]
    root_fields: tuple[str, ...] = ()
    external_media: bool = False
    external_media_summary: str = ""
    external_media_urls: tuple[str, ...] = ()
    frame_media_roots_by_split: dict[str, str] = field(default_factory=dict)
    frame_media_fps: float | None = None
    extended_metadata_fields: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    confidence: str = "inferred"

    @property
    def splits(self) -> tuple[str, ...]:
        values = []
        for item in self.annotation_files:
            if item.split and item.split not in values:
                values.append(item.split)
        priority = {"train": 0, "validation": 1, "test": 2}
        values.sort(key=lambda value: (priority.get(value, 10), value))
        return tuple(values)

    @property
    def variants(self) -> tuple[str, ...]:
        values = []
        for item in self.annotation_files:
            if item.variant and item.variant not in values:
                values.append(item.variant)
        return tuple(values)

    @property
    def total_annotation_bytes(self) -> int | None:
        if not self.annotation_files:
            return None
        sizes = [item.size for item in self.annotation_files]
        if any(value is None or int(value) < 0 for value in sizes):
            return None
        return sum(int(value or 0) for value in sizes)

    def files_for(
        self,
        *,
        splits: Sequence[str] | None = None,
        variant: str = "auto",
    ) -> tuple[HFStructuredAnnotationFile, ...]:
        wanted_splits = {str(value).strip() for value in (splits or ()) if str(value).strip()}
        requested_variant = str(variant or "auto").strip().casefold()
        if requested_variant not in {"auto", "base", "extended"}:
            raise ValueError("variant must be one of: auto, base, extended")

        candidates = [
            item
            for item in self.annotation_files
            if not wanted_splits or item.split in wanted_splits
        ]

        # Selection is performed per split so a repository that publishes an
        # extended representation for only some splits never causes another
        # requested split to disappear silently.
        by_split: dict[str, list[HFStructuredAnnotationFile]] = {}
        unsplit: list[HFStructuredAnnotationFile] = []
        for item in candidates:
            if item.split:
                by_split.setdefault(item.split, []).append(item)
            else:
                unsplit.append(item)

        chosen: list[HFStructuredAnnotationFile] = []
        for split in self.splits:
            if wanted_splits and split not in wanted_splits:
                continue
            group = by_split.get(split, [])
            if not group:
                continue
            if requested_variant == "auto":
                preferred = [item for item in group if item.variant == "extended"]
            else:
                preferred = [item for item in group if item.variant == requested_variant]
            chosen.append((preferred or group)[0])

        if requested_variant == "auto":
            chosen.extend(unsplit)
        else:
            preferred_unsplit = [item for item in unsplit if item.variant == requested_variant]
            chosen.extend(preferred_unsplit or unsplit)
        return tuple(chosen)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HFStructuredAnnotationProfile":
        data = dict(value or {})
        files = tuple(
            HFStructuredAnnotationFile(
                path=str(item.get("path") or ""),
                split=str(item.get("split") or ""),
                variant=str(item.get("variant") or "base"),
                size=(int(item["size"]) if item.get("size") is not None else None),
            )
            for item in (data.get("annotation_files") or ())
            if isinstance(item, Mapping) and str(item.get("path") or "").strip()
        )
        return cls(
            format_id=str(data.get("format_id") or ""),
            format_label=str(data.get("format_label") or ""),
            modalities=tuple(str(item) for item in (data.get("modalities") or ())),
            task_hints=tuple(str(item) for item in (data.get("task_hints") or ())),
            annotation_files=files,
            root_fields=tuple(str(item) for item in (data.get("root_fields") or ())),
            external_media=bool(data.get("external_media", False)),
            external_media_summary=str(data.get("external_media_summary") or ""),
            external_media_urls=tuple(str(item) for item in (data.get("external_media_urls") or ())),
            frame_media_roots_by_split={
                str(key): str(value)
                for key, value in dict(data.get("frame_media_roots_by_split") or {}).items()
                if str(key).strip() and str(value).strip()
            },
            frame_media_fps=(
                float(data["frame_media_fps"])
                if data.get("frame_media_fps") not in (None, "")
                else None
            ),
            extended_metadata_fields=tuple(str(item) for item in (data.get("extended_metadata_fields") or ())),
            evidence=tuple(str(item) for item in (data.get("evidence") or ())),
            confidence=str(data.get("confidence") or "inferred"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "format_label": self.format_label,
            "modalities": list(self.modalities),
            "task_hints": list(self.task_hints),
            "annotation_files": [item.to_dict() for item in self.annotation_files],
            "splits": list(self.splits),
            "variants": list(self.variants),
            "total_annotation_bytes": self.total_annotation_bytes,
            "root_fields": list(self.root_fields),
            "external_media": self.external_media,
            "external_media_summary": self.external_media_summary,
            "external_media_urls": list(self.external_media_urls),
            "frame_media_roots_by_split": dict(self.frame_media_roots_by_split),
            "frame_media_fps": self.frame_media_fps,
            "extended_metadata_fields": list(self.extended_metadata_fields),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }

def structured_annotation_candidates(
    *,
    repository_files: Iterable[Mapping[str, Any]],
    declared_data_files_by_split: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return annotation-like JSON candidates from repository metadata only.

    This intentionally uses path/schema conventions rather than repository IDs.
    Declared dataset files are accepted as candidates even when their parent
    directory is not literally named ``annotations``.
    """
    declared = {
        str(split): [str(path) for path in paths if str(path).strip()]
        for split, paths in dict(declared_data_files_by_split or {}).items()
    }
    declared_paths = {path for paths in declared.values() for path in paths}

    rows = [dict(row) for row in repository_files if isinstance(row, Mapping)]
    json_rows = [
        row
        for row in rows
        if str(row.get("path") or "").casefold().endswith(_JSON_SUFFIX)
    ]

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(path: str, row: Mapping[str, Any] | None = None, *, declared_only: bool = False) -> None:
        path = str(path or "").strip()
        if not path or path in seen or not path.casefold().endswith(_JSON_SUFFIX):
            return
        lowered = path.casefold()
        name = PurePosixPath(path).name.casefold()
        stem = PurePosixPath(path).stem.casefold()
        annotation_parent = (
            "/annotation/" in f"/{lowered}"
            or "/annotations/" in f"/{lowered}"
        )
        split_named = bool(
            re.search(
                r"(?:^|[_\-.])(?:train|training|val|valid|validation|dev|test|testing|eval|evaluation)(?:$|[_\-.])",
                stem,
            )
        )
        generic_metadata_name = name in {
            "readme.json",
            "metadata.json",
            "config.json",
            "dataset_info.json",
            "dataset_infos.json",
        }
        if generic_metadata_name:
            return
        if not (
            path in declared_paths
            or "annotation" in name
            or "instance" in name
            or (annotation_parent and (split_named or "ytvis" in stem))
        ):
            return

        payload = dict(row or {})
        payload["path"] = path
        if declared_only:
            # Card-declared data files are authoritative repository paths even when
            # a gated Hub tree cannot enumerate them for the current credentials.
            # Detection still requires independent format/card evidence (or a
            # bounded JSON shape probe), so synthesising this metadata row cannot
            # by itself classify an arbitrary JSON dataset as video segmentation.
            payload.setdefault("size", None)
            payload["declared_data_file"] = True
        seen.add(path)
        candidates.append(payload)

    for row in json_rows:
        add_candidate(str(row.get("path") or ""), row)

    # Hugging Face card YAML can declare exact config/split data files even when
    # repository-tree listing is unavailable (notably gated repositories). Preserve
    # those paths as metadata-only candidates so strong YTVIS/video-segmentation
    # cards can be recognised without downloading or enumerating gated payloads.
    for path in sorted(declared_paths):
        add_candidate(path, declared_only=True)

    return sorted(candidates, key=lambda item: str(item.get("path") or ""))

def _annotation_family_key(path: str) -> str:
    """Return a conservative sibling-family key for split annotation files."""
    value = PurePosixPath(str(path or ""))
    stem = value.stem.casefold()
    stem = re.sub(
        r"(?:^|[_\-.])(?:train|training|val|valid|validation|dev|test|testing|eval|evaluation)(?=$|[_\-.])",
        "_",
        stem,
    )
    stem = re.sub(r"(?:^|[_\-.])(?:ext|extended|base)(?=$|[_\-.])", "_", stem)
    stem = re.sub(r"[_\-.]+", "_", stem).strip("_")
    parent = str(value.parent).casefold().strip(".")
    return f"{parent}/{stem}" if parent else stem

def _probe_matches_ytvis_media_shape(value: Mapping[str, Any]) -> bool:
    """Verify the non-ground-truth portion of a canonical YTVIS document.

    Evaluation/test annotations may intentionally omit ground-truth annotations
    while still retaining the canonical video/category metadata.  This helper is
    deliberately not sufficient by itself to classify a repository as video
    segmentation; callers must pair it with explicit task evidence when the
    annotation array is absent.
    """
    root_fields = {str(item) for item in (value.get("root_fields") or ())}
    if not {"videos", "categories"}.issubset(root_fields):
        return False
    if not bool(value.get("videos_is_array", False)):
        return False
    if not bool(value.get("categories_is_array", False)):
        return False

    video_sampled = bool(value.get("video_sampled", False))
    video_fields = {str(item) for item in (value.get("video_fields") or ())}
    return bool(video_sampled and "file_names" in video_fields)

def _probe_matches_ytvis_shape(value: Mapping[str, Any]) -> bool:
    """Verify a YTVIS document whose annotation field is inspectable."""
    if not _probe_matches_ytvis_media_shape(value):
        return False

    root_fields = {str(item) for item in (value.get("root_fields") or ())}
    if "annotations" not in root_fields:
        return False
    if not bool(value.get("annotations_is_array", False)):
        return False

    annotation_sampled = bool(value.get("annotation_sampled", False))
    if not annotation_sampled:
        # Empty annotation arrays are valid for evaluation metadata. The field is
        # still explicitly present and typed as an array, so the document shape is
        # unambiguous even though no mask row can be sampled.
        return True
    annotation_fields = {str(item) for item in (value.get("annotation_fields") or ())}
    return "video_id" in annotation_fields and (
        "segmentations" in annotation_fields or "segmentation" in annotation_fields
    )

def _probe_matches_ytvis_evaluation_shape(value: Mapping[str, Any]) -> bool:
    """Verify media metadata for a test/evaluation YTVIS document.

    OVIS/YTVIS-style test annotations may publish ``annotations: null`` or omit
    the field because ground truth is withheld.  Such a media-only probe is only
    accepted by the detector when the repository card independently declares a
    video-segmentation task.
    """
    if not _probe_matches_ytvis_media_shape(value):
        return False

    root_fields = {str(item) for item in (value.get("root_fields") or ())}
    annotations_present = bool(value.get("annotations_present", "annotations" in root_fields))
    annotations_is_null = bool(value.get("annotations_is_null", False))
    annotations_is_array = bool(value.get("annotations_is_array", False))

    if annotations_is_array:
        return False
    return bool(not annotations_present or annotations_is_null)

def detect_structured_annotation_profile(
    *,
    card_text: str,
    repository_files: Iterable[Mapping[str, Any]],
    declared_data_files_by_split: Mapping[str, Sequence[str]] | None = None,
    probed_json_by_path: Mapping[str, Mapping[str, Any]] | None = None,
) -> HFStructuredAnnotationProfile | None:
    """Detect a supported structured video-segmentation repository.

    Recognition is format-oriented, never repository-ID-oriented. Strong dataset
    card declarations remain a zero-download fast path. When card wording is too
    sparse, callers may supply bounded JSON structure probes; a probe is accepted
    only when it verifies the canonical YTVIS shape (videos, annotations,
    categories, per-video ``file_names``, and segmentation-bearing annotation
    rows when ground truth is present).
    """

    text = str(card_text or "")
    folded = text.casefold()
    declared = {
        str(split): [str(path) for path in paths if str(path).strip()]
        for split, paths in dict(declared_data_files_by_split or {}).items()
    }
    candidates = structured_annotation_candidates(
        repository_files=repository_files,
        declared_data_files_by_split=declared,
    )
    if not candidates:
        return None

    ytvis_named = bool(re.search(r"\bytvis\b", folded))
    has_video_fields = all(
        token in folded for token in ("videos", "annotations", "segmentations")
    )
    has_video_pair_fields = (
        "video_np_pairs" in folded
        or "video-np" in folded
        or "video np pairs" in folded
    )
    has_masklet_language = "masklet" in folded
    card_declares_video_segmentation_task = bool(
        re.search(r"\bvideo\s+(?:instance|object)\s+segmentation\b", folded)
        or re.search(r"\bvideo\s+segmentation\b", folded)
    )
    card_declares_shape = bool(ytvis_named or has_video_fields)

    probes = {
        str(path): dict(value)
        for path, value in dict(probed_json_by_path or {}).items()
        if isinstance(value, Mapping)
    }
    valid_probe_paths = {
        path for path, value in probes.items() if _probe_matches_ytvis_shape(value)
    }
    evaluation_probe_paths = (
        {
            path
            for path, value in probes.items()
            if _probe_matches_ytvis_evaluation_shape(value)
        }
        if card_declares_video_segmentation_task
        else set()
    )
    accepted_probe_paths = valid_probe_paths | evaluation_probe_paths

    if not card_declares_shape and not accepted_probe_paths:
        return None

    selected_candidates = list(candidates)
    if not card_declares_shape:
        valid_families = {_annotation_family_key(path) for path in accepted_probe_paths}
        selected_candidates = [
            row
            for row in candidates
            if str(row.get("path") or "") in accepted_probe_paths
            or _annotation_family_key(str(row.get("path") or "")) in valid_families
        ]
        if not selected_candidates:
            return None

    files: list[HFStructuredAnnotationFile] = []
    for row in selected_candidates:
        path = str(row.get("path") or "")
        split = _declared_split(path, declared) or infer_split_from_annotation_path(path)
        variant = infer_annotation_variant(path)
        raw_size = row.get("size")
        try:
            size = int(raw_size) if raw_size is not None else None
        except Exception:
            size = None
        files.append(
            HFStructuredAnnotationFile(
                path=path,
                split=split,
                variant=variant,
                size=size,
            )
        )

    evidence: list[str] = []
    if ytvis_named:
        evidence.append("dataset card describes a YTVIS-like annotation format")
    if has_video_fields:
        evidence.append("dataset card declares videos + annotations + segmentations")
    if "video instance segmentation" in folded:
        evidence.append("dataset card describes video instance segmentation")
    if has_video_pair_fields:
        evidence.append("dataset card declares positive/negative video-noun-phrase pairs")
    if has_masklet_language:
        evidence.append("dataset card describes temporal masklets")
    if valid_probe_paths:
        evidence.append(
            "bounded annotation JSON probe verified YTVIS videos + annotations + categories structure"
        )
    if evaluation_probe_paths:
        evidence.append(
            "bounded evaluation JSON probe verified YTVIS video/category metadata while the dataset card declares video segmentation"
        )

    format_label = "YTVIS-like video segmentation"
    if "sa-co/veval" in folded or ("sa-co" in folded and "veval" in folded):
        format_label = "SA-Co/VEval / YTVIS-like video segmentation"

    root_fields = ["info", "videos", "annotations", "categories"]
    if has_video_pair_fields:
        root_fields.append("video_np_pairs")

    task_hints = ["video-object-segmentation"]
    if has_masklet_language or has_video_pair_fields:
        task_hints.append("temporal-masklets")
    if has_video_pair_fields:
        task_hints.append("video-text-grounding")

    extended_fields = tuple(
        field
        for field in (
            "video_num_frames",
            "video_fps",
            "video_creation_datetime",
            "location_id",
            "Kingdom",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
        )
        if field.casefold() in folded
    )

    media_summary = _external_media_summary(text)
    urls = tuple(_external_media_urls(text))
    external_media = bool(
        media_summary
        or re.search(r"\b(?:external|bucket|gcs|google cloud|s3)\b", folded)
        or ("all videos" in folded and "found in" in folded)
    )

    frame_roots, frame_fps = infer_external_frame_media(card_text=text, urls=urls)

    if valid_probe_paths:
        confidence = "verified"
    elif evaluation_probe_paths and card_declares_video_segmentation_task:
        confidence = "declared"
    elif ytvis_named and has_video_fields:
        confidence = "declared"
    else:
        confidence = "inferred"

    return HFStructuredAnnotationProfile(
        format_id="ytvis_like_video_segmentation",
        format_label=format_label,
        modalities=("video", "segmentation"),
        task_hints=tuple(task_hints),
        annotation_files=tuple(files),
        root_fields=tuple(root_fields),
        external_media=external_media,
        external_media_summary=media_summary,
        external_media_urls=urls,
        frame_media_roots_by_split=frame_roots,
        frame_media_fps=frame_fps,
        extended_metadata_fields=extended_fields,
        evidence=tuple(evidence),
        confidence=confidence,
    )

def infer_split_from_annotation_path(path: str) -> str:
    text = str(path or "")
    padded = f"/{text}/"
    for split, pattern in _SPLIT_PATTERNS:
        if pattern.search(padded):
            return split
    return ""

def infer_annotation_variant(path: str) -> str:
    stem = PurePosixPath(str(path or "")).stem.casefold()
    if re.search(r"(?:^|[_\-.])(?:ext|extended)(?:$|[_\-.])", stem):
        return "extended"
    return "base"

def _declared_split(path: str, declared: Mapping[str, Sequence[str]]) -> str:
    for split, paths in declared.items():
        if str(path) in {str(value) for value in paths}:
            return str(split)
    return ""

def infer_external_frame_media(
    *,
    card_text: str,
    urls: Sequence[str] = (),
) -> tuple[dict[str, str], float | None]:
    """Infer public frame-sequence roots from a dataset card, never from repo identity.

    The current acceptance case uses a public Google Cloud Storage bucket and a
    Markdown directory tree containing ``JPEGImages_6fps``. Other repositories
    simply return no inferred root and can use the panel override.
    """
    bucket = _gcs_bucket_from_urls(urls)
    if not bucket:
        bucket = _gcs_bucket_from_urls(_external_media_urls(card_text))
    if not bucket:
        return {}, _frame_fps_from_text(card_text)

    paths = _markdown_tree_paths(card_text)
    candidates = [
        path for path in paths
        if re.search(r"(?:^|/)(?:jpegimages|frames?)[_-]?(?:\d+(?:\.\d+)?)?fps/?$", path, re.IGNORECASE)
        or re.search(r"(?:^|/)jpegimages_\d+(?:\.\d+)?fps/?$", path, re.IGNORECASE)
    ]
    # Prefer unpacked frame directories over sibling tar archives.
    candidates = [path for path in candidates if "_tars" not in path.casefold()]
    roots: dict[str, str] = {}
    detected_fps: float | None = None
    for path in candidates:
        split = infer_split_from_annotation_path(path)
        if not split:
            continue
        clean = path.strip("/")
        roots.setdefault(split, f"https://storage.googleapis.com/{bucket}/{clean}/")
        if detected_fps is None:
            detected_fps = _frame_fps_from_text(PurePosixPath(clean).name)
    return roots, detected_fps or _frame_fps_from_text(card_text)

def _gcs_bucket_from_urls(urls: Sequence[str]) -> str:
    patterns = (
        re.compile(r"https?://console\.cloud\.google\.com/storage/browser/([^/?#]+)", re.I),
        re.compile(r"https?://storage\.googleapis\.com/([^/?#]+)", re.I),
    )
    for raw in urls:
        value = str(raw or "")
        for pattern in patterns:
            match = pattern.search(value)
            if match:
                return match.group(1).strip()
    return ""

def _frame_fps_from_text(text: str) -> float | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*fps\b", str(text or ""), re.I)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None

def _markdown_tree_paths(text: str) -> list[str]:
    """Recover paths from common README tree blocks without inventing hierarchy.

    Markdown cards commonly indent an entire ``tree`` block by four spaces, while
    other renderings start ``├──``/``└──`` at column zero.  The first branch line
    therefore defines a *presentation baseline*; hierarchy is measured relative
    to that baseline, not from absolute leading whitespace.

    This is deliberately conservative when input has already been normalised for
    display.  If indentation has been collapsed, nested branches become siblings
    instead of being guessed as descendants.  That may prevent automatic media
    root inference and require the explicit panel override, but it cannot silently
    manufacture a shorter external URI.
    """
    stack: list[str] = []
    paths: list[str] = []
    branch_baseline: int | None = None

    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        marker = "├──" if "├──" in line else ("└──" if "└──" in line else "")

        if not marker:
            root_candidate = line.strip().lstrip("`").strip().rstrip("`").strip()
            if (
                root_candidate.endswith("/")
                and " " not in root_candidate
                and not re.match(r"^[a-z][a-z0-9+.-]*://", root_candidate, re.I)
            ):
                root_name = root_candidate.rstrip("/")
                if root_name:
                    stack = [root_name]
                    paths.append(root_name)
                    branch_baseline = None
            continue

        prefix, name = line.split(marker, 1)
        name = name.strip().strip("`").rstrip("/")
        if not name:
            continue

        # Tree renderers conventionally use four columns (``│   `` / ``    ``)
        # per level.  Remove the visual vertical bar but keep its occupied width.
        normalized_prefix = prefix.replace("│", " ").expandtabs(4)
        absolute_units = max(0, len(normalized_prefix) // 4)
        if branch_baseline is None:
            branch_baseline = absolute_units
        relative_units = max(0, absolute_units - branch_baseline)

        # A standalone root line anchors branch depth one.  Without a root line,
        # the first branch is itself the root (depth zero).
        depth = relative_units + (1 if stack else 0)

        if depth < len(stack):
            stack = stack[:depth]
        elif depth > len(stack):
            # Missing intermediate indentation is ambiguous. Clamp to the deepest
            # hierarchy actually observed instead of fabricating parent segments.
            depth = len(stack)

        if depth == len(stack):
            stack.append(name)
        else:
            stack[depth] = name
            stack = stack[: depth + 1]
        paths.append("/".join(stack))

    return paths

def _external_media_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in re.findall(r"\[[^\]]*\]\((https?://[^)\s]+)\)", text or ""):
        if match not in urls:
            urls.append(match)
    for match in re.findall(r"(?<!\()\bhttps?://[^\s<>)]+", text or ""):
        cleaned = match.rstrip(".,;:")
        if cleaned not in urls:
            urls.append(cleaned)
    # The dataset-card source itself is not an external media location.
    return [url for url in urls if "huggingface.co/datasets/" not in url.casefold()][:8]

def _external_media_summary(text: str) -> str:
    compact = re.sub(r"[`*_]", "", str(text or ""))
    compact = re.sub(r"\[[^\]]+\]\(([^)]+)\)", lambda match: match.group(0).split("]", 1)[0][1:], compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if not compact:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    for sentence in sentences:
        folded = sentence.casefold()
        if "video" in folded and (
            "found in" in folded
            or "bucket" in folded
            or "external" in folded
            or "storage" in folded
        ):
            return sentence[:600].strip()
    return ""

__all__ = [
    "HFStructuredAnnotationFile",
    "HFStructuredAnnotationProfile",
    "detect_structured_annotation_profile",
    "structured_annotation_candidates",
    "infer_annotation_variant",
    "infer_external_frame_media",
    "infer_split_from_annotation_path",
]
