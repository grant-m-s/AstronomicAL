from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionRequest

from . import service as service_mod
from . import importer as importer_mod
from . import structured_video_importer as structured_video_importer_mod
from . import video_importer as video_importer_mod

HuggingFaceDatasetService = service_mod.HuggingFaceDatasetService
import_hf_image_dataset_as_manifest = importer_mod.import_hf_image_dataset_as_manifest

manifest = PluginManifest(
    id="integrations.huggingface",
    name="Hugging Face Datasets",
    version="0.4.18",
    description=(
        "Inspect Hugging Face datasets through server-side metadata, surface gated "
        "access requirements and authentication state, recognise structured video/segmentation "
        "annotation repositories, cache pinned annotations safely, bridge YTVIS-like frame "
        "sequences into core.video, retain exact manifest-bound local segmentation-source provenance for "
        "vision.segmentation, preserve native image-label semantics, and import supported image splits efficiently."
    ),
    requires=[
        "huggingface_hub>=0.24",
        "datasets>=2.20",
        "pillow>=10",
    ],
    optional_plugins=[
        "core.image",
        "core.video>=0.2.17",
    ],
    capabilities=[
        "integration",
        "datasets",
        "images",
        "search",
        "preview",
        "import",
        "huggingface",
        "gated-access",
        "structured-annotations",
        "video-dataset-inspection",
        "segmentation-dataset-inspection",
        "structured-video-import",
        "repository-video-import",
        "epic-kitchens-action-annotations",
    ],
    tags=[
        "integration",
        "huggingface",
        "datasets",
        "image",
        "manifest",
        "optional",
        "active-learning-ready",
    ],
    metadata={
        "recommended_with": [
            "core.image",
            "core.video",
            "vision.segmentation",
            "core.record_browser",
            "core.annotations",
        ],
    },
)

def register(api) -> None:
    api.register_service(
        key="client",
        factory=create_huggingface_service,
        lazy=True,
        replace=True,
        description="Hugging Face Hub/Datasets client service.",
    )

    api.register_panel(
        id="browser",
        title="Hugging Face Dataset Importer",
        factory=create_browser_panel,
        description=(
            "Search Hugging Face datasets, guide browser-based gated access and token "
            "verification, inspect typed schemas/configs/splits without local construction "
            "when Dataset Viewer metadata is available, recognise/cache structured video "
            "annotation files without flattening them, preview samples, and import supported "
            "image splits into AstronomicAL."
        ),
        category="Dataset Importers",
        icon="cloud-download",
        tags=[
            "huggingface",
            "datasets",
            "image",
            "integration",
            "manifest",
        ],
        uses_services=["integrations.huggingface.client"],
        produces=[
            "dataset.loaded",
            "dataset.mapping.updated",
            "dataset.active.changed",
        ],
        default_layout={"x": 0, "y": 0, "w": 8, "h": 10},
    )

    api.register_action(
        id="cache_structured_annotations",
        title="Cache Hugging Face Structured Annotations",
        handler=cache_structured_annotations_action,
        description=(
            "Detect a structured video/segmentation annotation repository and cache the "
            "selected pinned annotation JSON files without registering a fake image dataset."
        ),
        category="Dataset Importers",
        icon="download",
        tags=["huggingface", "dataset", "video", "segmentation", "annotations", "cache"],
        run_in_job=True,
        outputs=[],
        params_schema={
            "type": "object",
            "required": ["repo_id"],
            "properties": {
                "repo_id": {"type": "string", "minLength": 1},
                "splits": {
                    "type": ["array", "string", "null"],
                    "items": {"type": "string"},
                },
                "variant": {
                    "type": "string",
                    "enum": ["auto", "extended", "base"],
                    "default": "auto",
                },
                "revision": {"type": ["string", "null"]},
                "token": {"type": ["string", "null"]},
                "max_workers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
            },
        },
    )

    api.register_action(
        id="import_structured_video_dataset",
        title="Import Hugging Face Structured Video Dataset",
        handler=import_structured_video_dataset_action,
        description=(
            "Stream video-level metadata from cached YTVIS-like annotations, write compact "
            "Parquet manifests, and register them through the public core.video contract. "
            "Segmentation payloads remain external for vision.segmentation."
        ),
        category="Dataset Importers",
        icon="video",
        tags=["huggingface", "dataset", "video", "ytvis", "frame-sequence", "import"],
        run_in_job=True,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "required": ["repo_id"],
            "properties": {
                "repo_id": {"type": "string", "minLength": 1},
                "splits": {"type": ["array", "string", "null"], "items": {"type": "string"}},
                "variant": {"type": "string", "enum": ["auto", "extended", "base"], "default": "auto"},
                "revision": {"type": ["string", "null"]},
                "token": {"type": ["string", "null"]},
                "dataset_id_base": {"type": ["string", "null"]},
                "dataset_name_base": {"type": ["string", "null"]},
                "media_root_override": {"type": ["string", "null"]},
                "annotation_fps": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "max_videos": {"type": "integer", "minimum": 0, "default": 0},
                "active_split": {"type": ["string", "null"]},
            },
        },
    )

    api.register_action(
        id="import_repository_videos",
        title="Import Hugging Face Repository Videos",
        handler=import_repository_videos_action,
        description=(
            "Cache a bounded selection of revision-pinned repository MP4/MOV-style videos, "
            "optionally join EPIC-KITCHENS-100 action annotations at the same pinned revision, "
            "then register a metadata-only Parquet manifest through core.video."
        ),
        category="Dataset Importers",
        icon="video",
        tags=["huggingface", "video", "repository-files", "import"],
        run_in_job=True,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "required": ["repo_id"],
            "properties": {
                "repo_id": {"type": "string", "minLength": 1},
                "path_prefix": {"type": "string", "default": ""},
                "revision": {"type": ["string", "null"]},
                "token": {"type": ["string", "null"]},
                "dataset_id": {"type": ["string", "null"]},
                "dataset_name": {"type": ["string", "null"]},
                "max_videos": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "max_total_bytes": {"type": "integer", "minimum": 1, "default": 2147483648},
                "max_inventory": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 20000},
                "set_active": {"type": "boolean", "default": True},
                "annotation_mode": {
                    "type": "string",
                    "enum": ["auto", "none", "epic_kitchens_100"],
                    "default": "auto",
                },
            },
        },
    )

    api.register_action(
        id="import_image_dataset",
        title="Import Hugging Face Image Dataset",
        handler=import_image_dataset_action,
        description=(
            "Programmatically import a Hugging Face image dataset split as an "
            "AstronomicAL image manifest dataset."
        ),
        category="Dataset Importers",
        icon="cloud-download",
        tags=["huggingface", "image", "dataset", "import"],
        run_in_job=True,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "properties": {
                "repo_id": {
                    "type": "string",
                    "title": "Hugging Face dataset repo id",
                },
                "config_name": {
                    "type": ["string", "null"],
                    "title": "Config name",
                    "default": None,
                },
                "split": {
                    "type": "string",
                    "title": "Split",
                    "default": "train",
                },
                "dataset_id": {
                    "type": ["string", "null"],
                    "title": "AstronomicAL dataset ID",
                    "default": None,
                },
                "dataset_name": {
                    "type": ["string", "null"],
                    "title": "AstronomicAL dataset name",
                    "default": None,
                },
                "image_column": {
                    "type": ["string", "null"],
                    "title": "Image column",
                    "default": None,
                },
                "label_column": {
                    "type": ["string", "null"],
                    "title": "Label column",
                    "default": None,
                },
                "id_column": {
                    "type": ["string", "null"],
                    "title": "ID column",
                    "default": None,
                },
                "max_rows": {
                    "type": "integer",
                    "title": "Max rows; 0 = all",
                    "default": 0,
                    "minimum": 0,
                },
                "revision": {
                    "type": ["string", "null"],
                    "title": "Pinned dataset revision (commit SHA preferred)",
                    "default": None,
                },
                "token": {
                    "type": ["string", "null"],
                    "title": "HF token",
                    "default": None,
                },
                "trust_remote_code": {
                    "type": "boolean",
                    "title": "Trust remote dataset code",
                    "default": False,
                },
                "write_parquet": {
                    "type": "boolean",
                    "title": "Write manifest to Parquet cache",
                    "default": True,
                },
                "set_active": {
                    "type": "boolean",
                    "title": "Set as active dataset",
                    "default": True,
                },
            },
            "required": ["repo_id"],
        },
    )

def create_huggingface_service(
    context: Any = None,
    manager: Any = None,
    settings: dict[str, Any] | None = None,
):
    settings = settings or {}
    token = settings.get("token")
    cache_dir = settings.get("cache_dir")
    return HuggingFaceDatasetService(token=token, cache_dir=cache_dir)

def create_browser_panel(context: Any, manager: Any = None, **kwargs):
    from . import browser as browser_mod

    controller = browser_mod.HuggingFaceBrowserPanel(context=context, manager=manager)
    return controller.panel(), controller

def import_image_dataset_action(
    context: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})

    repo_id = str(params.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")

    return import_hf_image_dataset_as_manifest(
        context,
        repo_id=repo_id,
        config_name=_empty_to_none(params.get("config_name")),
        split=str(params.get("split") or "train"),
        dataset_id=_empty_to_none(params.get("dataset_id")),
        dataset_name=_empty_to_none(params.get("dataset_name")),
        image_column=_empty_to_none(params.get("image_column")),
        label_column=_empty_to_none(params.get("label_column")),
        id_column=_empty_to_none(params.get("id_column")),
        max_rows=int(params.get("max_rows") or 0),
        revision=_empty_to_none(params.get("revision")),
        token=_empty_to_none(params.get("token")),
        trust_remote_code=bool(params.get("trust_remote_code", False)),
        write_parquet=bool(params.get("write_parquet", True)),
        set_active=bool(params.get("set_active", True)),
        cancel_token=cancel_token,
    )

def import_structured_video_dataset_action(
    context: Any,
    request: ActionRequest,
    manager: Any = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    repo_id = str(params.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")
    raw_splits = params.get("splits")
    if isinstance(raw_splits, str):
        splits = [value.strip() for value in raw_splits.split(",") if value.strip()]
    elif isinstance(raw_splits, (list, tuple)):
        splits = [str(value).strip() for value in raw_splits if str(value).strip()]
    else:
        splits = []
    service = None
    services = getattr(context, "services", None)
    if services is not None:
        try:
            service = services.get("integrations.huggingface.client")
        except Exception:
            service = None
    if service is None:
        service = HuggingFaceDatasetService(token=_empty_to_none(params.get("token")))
    return structured_video_importer_mod.import_structured_video_dataset(
        context,
        manager=manager or getattr(context, "plugins", None),
        service=service,
        repo_id=repo_id,
        splits=splits,
        variant=str(params.get("variant") or "auto"),
        revision=_empty_to_none(params.get("revision")),
        token=_empty_to_none(params.get("token")),
        dataset_id_base=_empty_to_none(params.get("dataset_id_base")),
        dataset_name_base=_empty_to_none(params.get("dataset_name_base")),
        media_root_override=_empty_to_none(params.get("media_root_override")),
        annotation_fps=(float(params["annotation_fps"]) if params.get("annotation_fps") not in (None, "") else None),
        max_videos=max(0, int(params.get("max_videos") or 0)),
        active_split=_empty_to_none(params.get("active_split")),
        cancel_token=cancel_token,
    )

def import_repository_videos_action(
    context: Any,
    request: ActionRequest,
    manager: Any = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    service = None
    services = getattr(context, "services", None)
    if services is not None:
        try:
            service = services.get("integrations.huggingface.client")
        except Exception:
            service = None
    if service is None:
        service = HuggingFaceDatasetService(token=_empty_to_none(params.get("token")))
    return video_importer_mod.import_repository_videos(
        context,
        manager=manager or getattr(context, "plugins", None),
        service=service,
        repo_id=str(params.get("repo_id") or ""),
        path_prefix=str(params.get("path_prefix") or ""),
        revision=_empty_to_none(params.get("revision")),
        token=_empty_to_none(params.get("token")),
        dataset_id=_empty_to_none(params.get("dataset_id")),
        dataset_name=_empty_to_none(params.get("dataset_name")),
        max_videos=int(params.get("max_videos", 10)),
        max_total_bytes=int(params.get("max_total_bytes", video_importer_mod.DEFAULT_MAX_BYTES)),
        max_inventory=int(params.get("max_inventory", video_importer_mod.DEFAULT_MAX_INVENTORY)),
        set_active=bool(params.get("set_active", True)),
        annotation_mode=str(params.get("annotation_mode") or "auto"),
        cancel_token=cancel_token,
    )

def cache_structured_annotations_action(
    context: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    repo_id = str(params.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")
    raw_splits = params.get("splits")
    if isinstance(raw_splits, str):
        splits = [value.strip() for value in raw_splits.split(",") if value.strip()]
    elif isinstance(raw_splits, (list, tuple)):
        splits = [str(value).strip() for value in raw_splits if str(value).strip()]
    else:
        splits = []

    service = None
    services = getattr(context, "services", None)
    if services is not None:
        try:
            service = services.get("integrations.huggingface.client")
        except Exception:
            service = None
    if service is None:
        service = HuggingFaceDatasetService(token=_empty_to_none(params.get("token")))

    return service.cache_structured_annotations(
        repo_id,
        splits=splits or None,
        variant=str(params.get("variant") or "auto"),
        revision=_empty_to_none(params.get("revision")),
        token=_empty_to_none(params.get("token")),
        max_workers=max(1, min(8, int(params.get("max_workers") or 4))),
        cancel_token=cancel_token,
    )

def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in {"", "__default__", "None", "null"}:
        return None
    return value
