from __future__ import annotations

from typing import Any

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionRequest

from . import assets as assets_mod
from . import manifest as manifest_mod

ImageAssetResolver = assets_mod.ImageAssetResolver
build_image_manifest_dataframe = manifest_mod.build_image_manifest_dataframe
register_image_manifest_dataset = manifest_mod.register_image_manifest_dataset
slugify = manifest_mod.slugify

manifest = PluginManifest(
    id="core.image",
    name="Image Assets",
    version="0.2.4",
    description=(
        "Manifest-backed image dataset support. Provides an image asset resolver, "
        "folder-to-manifest builder, full-panel focused image viewer, and "
        "selection-set gallery for visual inspection and active learning."
    ),
    requires=["pillow>=10"],
    capabilities=[
        "panel",
        "action",
        "datasets",
        "selection",
        "events",
        "artifacts",
        "jobs",
        "services",
        "images",
        "media",
    ],
    tags=["core", "image", "media", "assets", "manifest", "gallery"],
)

RECORD_ID_MAPPING = {
    "semantic_name": "record_id",
    "display_name": "Record ID",
    "description": "Unique identifier for each image record.",
    "aliases": ["record_id", "id", "source_id", "object_id", "image_id"],
}

IMAGE_URI_MAPPING = {
    "semantic_name": "image.uri",
    "display_name": "Image asset",
    "description": (
        "Column containing an image URI, local image path, relative image path, "
        "or HTTP(S) URL."
    ),
    "aliases": [
        "image_uri",
        "image_path",
        "image_url",
        "path",
        "filepath",
        "file_path",
        "filename",
        "url",
        "href",
        "asset_uri",
    ],
}

IMAGE_OPTIONAL_MAPPINGS = [
    {
        "semantic_name": "image.path",
        "display_name": "Image path",
        "description": "Optional explicit local image path column.",
        "aliases": ["image_path", "path", "filepath", "file_path", "filename"],
    },
    {
        "semantic_name": "image.url",
        "display_name": "Image URL",
        "description": "Optional explicit remote image URL column.",
        "aliases": ["image_url", "url", "href"],
    },
    {
        "semantic_name": "image.thumbnail",
        "display_name": "Thumbnail image",
        "description": "Optional smaller thumbnail URI/path used by gallery views.",
        "aliases": ["thumbnail", "thumbnail_path", "thumbnail_uri", "thumb", "thumb_uri"],
    },
    {
        "semantic_name": "target_label",
        "display_name": "Label",
        "description": "Optional class/label column.",
        "aliases": ["target_label", "label", "labels", "class", "classification"],
    },
    {
        "semantic_name": "prediction",
        "display_name": "Prediction",
        "description": "Optional model prediction column used as a gallery badge.",
        "aliases": ["prediction", "predicted_label", "pred", "ml_prediction"],
    },
    {
        "semantic_name": "uncertainty",
        "display_name": "Uncertainty",
        "description": "Optional uncertainty/acquisition score column used as a gallery badge.",
        "aliases": ["uncertainty", "prediction_uncertainty", "entropy", "margin", "least_confidence"],
    },
    {
        "semantic_name": "label_state",
        "display_name": "Label state",
        "description": "Optional active-learning/review state column.",
        "aliases": ["label_state", "al_label_state", "annotation_state", "review_state"],
    },
]

def register(api) -> None:
    api.register_service(
        key="asset_resolver",
        factory=lambda: None,
        lazy=True,
        replace=True,
        description=(
            "Image asset resolver. Installed in on_enable so it can receive the "
            "runtime AppContext."
        ),
    )

    api.register_panel(
        id="manifest_builder",
        title="Image Manifest Builder",
        factory=create_manifest_builder_panel,
        description=(
            "Create an AstronomicAL manifest dataset from a folder of images. "
            "The manifest stores paths/URIs and metadata, not image bytes."
        ),
        category="Images",
        icon="photo",
        tags=["core", "image", "dataset", "manifest"],
        produces=["dataset.loaded", "dataset.active.changed", "dataset.mapping.updated"],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 7},
    )

    api.register_panel(
        id="viewer",
        title="Image Viewer",
        factory=create_image_viewer_panel,
        description=(
            "Preview the currently focused record's image. The image is kept "
            "large at the top of the panel, supports fit modes, and publishes "
            "image.preview artifacts."
        ),
        category="Images",
        icon="photo",
        tags=["core", "image", "selection", "viewer"],
        required_mappings=[RECORD_ID_MAPPING, IMAGE_URI_MAPPING],
        optional_mappings=IMAGE_OPTIONAL_MAPPINGS,
        uses_services=["core.image.asset_resolver"],
        produces=["artifact.created", "image.preview"],
        default_layout={"x": 5, "y": 0, "w": 7, "h": 7},
        default_open_kwargs={"max_size": 2048, "image_height": 360},
        state_version=3,
    )

    api.register_panel(
        id="gallery",
        title="Image Selection Gallery",
        factory=create_image_gallery_panel,
        description=(
            "Scrollable thumbnail gallery for the active selection set. Highlights "
            "the focused record, shows label/prediction/uncertainty badges when "
            "available, and clicking a card updates platform focus."
        ),
        category="Images",
        icon="photo",
        tags=["core", "image", "selection", "gallery", "active-learning"],
        required_mappings=[RECORD_ID_MAPPING, IMAGE_URI_MAPPING],
        optional_mappings=IMAGE_OPTIONAL_MAPPINGS,
        uses_services=["core.image.asset_resolver"],
        produces=["selection.focus.changed"],
        default_layout={"x": 0, "y": 7, "w": 12, "h": 5},
        default_open_kwargs={
            "thumb_size": 112,
            "max_items": 240,
            "show_badges": False,
            "max_in_flight": 4,
            "batch_size": 24,
            "render_interval_ms": 180,
        },
        state_version=6,
    )

    api.register_action(
        id="build_manifest",
        title="Build Image Manifest",
        handler=build_manifest_action,
        description=(
            "Build and register an image manifest dataset from a folder path. "
            "Intended for programmatic use and future action UIs."
        ),
        category="Core",
        icon="folder",
        tags=["core", "image", "dataset", "manifest"],
        run_in_job=True,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "properties": {
                "root": {"type": "string", "title": "Image folder"},
                "dataset_id": {
                    "type": "string",
                    "title": "Dataset ID",
                    "default": "image_dataset",
                },
                "dataset_name": {
                    "type": "string",
                    "title": "Dataset name",
                    "default": "Image Dataset",
                },
                "extensions": {
                    "type": "string",
                    "title": "Extensions",
                    "default": ".jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
                },
                "recursive": {"type": "boolean", "title": "Scan recursively", "default": True},
                "label_from_parent": {
                    "type": "boolean",
                    "title": "Use parent folder as target_label",
                    "default": True,
                },
                "relative_paths": {
                    "type": "boolean",
                    "title": "Store relative image paths",
                    "default": True,
                },
                "write_parquet": {
                    "type": "boolean",
                    "title": "Write manifest to Parquet cache",
                    "default": True,
                },
                "set_active": {"type": "boolean", "title": "Set as active dataset", "default": True},
            },
            "required": ["root"],
        },
    )

    api.register_artifact_viewer(
        artifact_type="image.preview",
        viewer_factory=create_image_preview_artifact_viewer,
        id="preview_viewer",
        title="Image Preview",
        description="Render image.preview artifacts.",
        default=True,
        priority=10,
    )

def on_enable(context: Any, manager: Any = None) -> None:
    services = getattr(context, "services", None)
    if services is not None:
        services.set(
            "core.image.asset_resolver",
            ImageAssetResolver(context),
            replace=True,
            owner="core.image",
        )

def on_disable(context: Any, manager: Any = None) -> None:
    services = getattr(context, "services", None)
    if services is not None:
        try:
            services.remove("core.image.asset_resolver", owner="core.image", dispose=True)
        except Exception:
            pass

def create_manifest_builder_panel(context, **kwargs):
    from . import manifest_panel as manifest_panel_mod

    controller = manifest_panel_mod.ImageManifestBuilderPanel(context=context)
    return controller.panel(), controller

def create_image_viewer_panel(context, **kwargs):
    from . import viewer as viewer_mod

    controller = viewer_mod.ImageViewerPanel(
        context=context,
        max_size=int(kwargs.get("max_size", 2048)),
        image_height=int(kwargs.get("image_height", 360)),
    )
    return controller.panel(), controller

def create_image_gallery_panel(context, **kwargs):
    from . import gallery as gallery_mod

    controller = gallery_mod.ImageSelectionGalleryPanel(
        context=context,
        thumb_size=int(kwargs.get("thumb_size", 112)),
        max_items=int(kwargs.get("max_items", 240)),
        show_badges=bool(kwargs.get("show_badges", False)),
        max_in_flight=int(kwargs.get("max_in_flight", 4)),
        batch_size=int(kwargs.get("batch_size", 24)),
        render_interval_ms=int(kwargs.get("render_interval_ms", 180)),
    )
    return controller.panel(), controller

def build_manifest_action(
    context: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    root = params.get("root")
    if not root:
        raise ValueError("Missing required parameter: root")

    dataset_id = slugify(params.get("dataset_id") or "image_dataset")
    dataset_name = params.get("dataset_name") or dataset_id

    df = build_image_manifest_dataframe(
        root,
        recursive=bool(params.get("recursive", True)),
        extensions=params.get("extensions", ".jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff"),
        label_from_parent=bool(params.get("label_from_parent", True)),
        relative_paths=bool(params.get("relative_paths", True)),
        cancel_token=cancel_token,
    )
    if cancel_token is not None and cancel_token.cancelled():
        return {"cancelled": True, "dataset_id": dataset_id, "rows": 0}

    return register_image_manifest_dataset(
        context,
        df,
        dataset_id=dataset_id,
        name=dataset_name,
        base_path=root,
        set_active=bool(params.get("set_active", True)),
        write_parquet=bool(params.get("write_parquet", True)),
    )

def create_image_preview_artifact_viewer(
    context: Any,
    artifact: Any = None,
    artifact_id: str | None = None,
    **kwargs,
):
    import html

    import panel as pn

    payload = artifact
    if payload is None and artifact_id:
        payload = context.artifacts.get(artifact_id)
    if not isinstance(payload, dict):
        return pn.pane.Alert("Image preview artifact payload is unavailable.", alert_type="warning")

    data_uri = payload.get("data_uri")
    asset = payload.get("asset") or {}
    row_id = html.escape(str(asset.get("row_id", "")))
    dataset_id = html.escape(str(asset.get("dataset_id", "")))
    if not data_uri:
        return pn.pane.Alert("Image preview artifact does not contain data_uri.", alert_type="warning")

    return pn.Column(
        pn.pane.HTML(
            f"""
            <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; overflow:hidden; background:#111;">
              <img src="{data_uri}" alt="{row_id}" style="width:100%; height:100%; object-fit:contain; display:block;" />
            </div>
            """,
            sizing_mode="stretch_both",
            styles={"min-height": "0", "flex": "1 1 auto"},
        ),
        pn.pane.Markdown(
            f"**Dataset:** `{dataset_id}`  \n**Record:** `{row_id}`",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
        styles={
            "height": "100%",
            "width": "100%",
            "display": "flex",
            "flex-direction": "column",
            "min-height": "0",
            "overflow": "hidden",
        },
    )
