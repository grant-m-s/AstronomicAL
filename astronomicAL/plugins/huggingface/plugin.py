from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionRequest

from . import service as service_mod
from . import importer as importer_mod

HuggingFaceDatasetService = service_mod.HuggingFaceDatasetService
import_hf_image_dataset_as_manifest = importer_mod.import_hf_image_dataset_as_manifest


manifest = PluginManifest(
    id="integrations.huggingface",
    name="Hugging Face Datasets",
    version="0.1.0",
    description=(
        "Optional Hugging Face integration for searching, previewing, and "
        "registering image datasets as AstronomicAL manifest datasets."
    ),
    requires=[
        "huggingface_hub>=0.24",
        "datasets>=2.20",
        "pillow>=10",
    ],
    optional_plugins=[
        "core.image",
    ],
    capabilities=[
        "integration",
        "datasets",
        "images",
        "search",
        "preview",
        "import",
        "huggingface",
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
        "recommended_with": ["core.image", "core.record_browser", "core.annotations"],
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
        title="Hugging Face Dataset Browser",
        factory=create_browser_panel,
        description=(
            "Search Hugging Face datasets, inspect configs/splits/features, "
            "preview image samples, and register selected splits as "
            "AstronomicAL image-manifest datasets."
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
        outputs=[],
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


def create_browser_panel(context: Any, **kwargs):
    from . import browser as browser_mod

    controller = browser_mod.HuggingFaceBrowserPanel(context=context)
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
        token=_empty_to_none(params.get("token")),
        trust_remote_code=bool(params.get("trust_remote_code", False)),
        write_parquet=bool(params.get("write_parquet", True)),
        set_active=bool(params.get("set_active", True)),
        cancel_token=cancel_token,
    )


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in {"", "__default__", "None", "null"}:
        return None
    return value