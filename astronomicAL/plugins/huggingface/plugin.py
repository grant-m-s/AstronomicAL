from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionRequest


# ---------------------------------------------------------------------------
# Local-plugin sibling module loader
# ---------------------------------------------------------------------------
# AstronomicAL can load local plugin folders by importing plugin.py as a
# generated standalone module. In that mode, package-relative imports fail.
# This helper imports sibling files explicitly by path so the same plugin works
# as a bundled plugin and as a local plugin folder.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_STEM = "astronomical_integration_huggingface"


def _load_sibling_module(alias: str):
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing

    path = _THIS_DIR / f"{alias}.py"
    if not path.exists():
        raise ModuleNotFoundError(
            f"Cannot find sibling module {alias!r}; expected file {path}"
        )

    unique_name = f"{_PLUGIN_STEM}_{alias}_{abs(hash(str(path.resolve())))}"
    existing_unique = sys.modules.get(unique_name)
    if existing_unique is not None:
        sys.modules[alias] = existing_unique
        return existing_unique

    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


service_mod = _load_sibling_module("service")
importer_mod = _load_sibling_module("importer")

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
    browser_mod = _load_sibling_module("browser")
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