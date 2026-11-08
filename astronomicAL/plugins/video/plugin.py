from __future__ import annotations

from typing import Any

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import ActionRequest

from .manifest import (
    build_video_manifest_dataframe,
    register_video_manifest_dataset,
    register_video_manifest_path,
    slugify,
)
from .resolver import VideoAssetResolver
from .temporal import VideoTemporalStateManager
from .overlays import VideoOverlayRegistry

manifest = PluginManifest(
    id="core.video",
    name="Video",
    version="0.2.35",
    description=(
        "Video dataset semantics, encoded-video and frame-sequence resolution, bounded frame "
        "access, aspect-preserving compact responsive playback, rolling-prefetched browser-buffered frame-sequence playback using persistent layered canvas presentation (base media plus an independent provider-neutral overlay canvas with nested-Panel/shadow-root-safe discovery) and an UX-first resident decoded replay cache for ordinary clips, lossless unpremultiplied RGBD browser decoding, standard Bokeh models, and plugin-owned temporal focus "
        "without embedding media bytes in datasets."
    ),
    requires=["pillow>=10"],
    optional_requires=["av>=12"],
    capabilities=[
        "panel",
        "action",
        "datasets",
        "selection",
        "events",
        "jobs",
        "services",
        "video",
        "media",
        "bounded-frame-access",
        "browser-buffered-playback",
        "resident-overlay-frame-cache",
        "overlay-exact-frame-prime",
        "client-overlay-opacity",
        "full-sequence-browser-prefetch",
        "strided-frame-sequences",
        "explicit-frame-sequences",
        "multi-channel-frame-display",
        "client-channel-frame-display",
        "rgbd-unpremultiplied-browser-playback",
        "persistent-canvas-playback",
        "resident-decoded-replay-cache",
        "browser-local-frame-signal",
        "provider-neutral-video-overlays",
        "aspect-preserving-playback",
    ],
    tags=["core", "video", "media", "viewer", "temporal"],
    metadata={
        "maturity": "experimental",
        "stage": "stage4-playback",
    },
)

RECORD_ID_MAPPING = {
    "semantic_name": "record_id",
    "display_name": "Record ID",
    "description": "Stable dataset-row identity for the video record.",
    "aliases": ["record_id", "id", "video_id", "source_id", "object_id"],
}

VIDEO_URI_MAPPING = {
    "semantic_name": "video.uri",
    "display_name": "Video asset",
    "description": (
        "Column containing a video URI, local path, relative path, or HTTP(S) URL."
    ),
    "aliases": [
        "video_uri",
        "video_path",
        "video_url",
        "video",
        "path",
        "filepath",
        "file_path",
        "url",
        "uri",
    ],
}

OPTIONAL_VIDEO_MAPPINGS = [
    {
        "semantic_name": "video.fps",
        "display_name": "Video FPS",
        "description": "Frames per second when known without probing media.",
        "aliases": ["video_fps", "fps", "frame_rate"],
    },
    {
        "semantic_name": "video.frame_count",
        "display_name": "Video frame count",
        "description": "Total frame count when known from source metadata.",
        "aliases": ["video_frame_count", "frame_count", "num_frames"],
    },
    {
        "semantic_name": "video.duration",
        "display_name": "Video duration",
        "description": "Video duration in seconds when known.",
        "aliases": ["video_duration", "duration_seconds", "duration"],
    },
    {
        "semantic_name": "video.width",
        "display_name": "Video width",
        "description": "Video frame width in pixels when known.",
        "aliases": ["video_width", "width"],
    },
    {
        "semantic_name": "video.height",
        "display_name": "Video height",
        "description": "Video frame height in pixels when known.",
        "aliases": ["video_height", "height"],
    },
    {
        "semantic_name": "video.thumbnail_uri",
        "display_name": "Video thumbnail",
        "description": "Optional precomputed thumbnail URI.",
        "aliases": ["video_thumbnail_uri", "thumbnail_uri", "thumbnail_url"],
    },
    {
        "semantic_name": "video.media_kind",
        "display_name": "Video media kind",
        "description": "Optional media representation, e.g. video_file or frame_sequence.",
        "aliases": ["video_media_kind", "media_kind"],
    },
    {
        "semantic_name": "video.frame_uri_template",
        "display_name": "Frame URI template",
        "description": "Optional per-frame URI template for frame-sequence datasets.",
        "aliases": ["video_frame_uri_template", "frame_uri_template"],
    },
    {
        "semantic_name": "video.frame_uris",
        "display_name": "Explicit frame URIs",
        "description": (
            "Optional ordered per-logical-frame URI list for irregular frame sequences "
            "that cannot be represented by one numeric template."
        ),
        "aliases": ["video_frame_uris", "frame_uris"],
    },
    {
        "semantic_name": "video.frame_index_offset",
        "display_name": "Frame index offset",
        "description": "Optional source-frame numbering offset for frame-sequence datasets.",
        "aliases": ["video_frame_index_offset", "frame_index_offset"],
    },
    {
        "semantic_name": "video.frame_index_step",
        "display_name": "Frame index step",
        "description": (
            "Optional positive source-frame numbering stride for frame-sequence datasets; "
            "defaults to 1."
        ),
        "aliases": ["video_frame_index_step", "frame_index_step", "frame_stride"],
    },
]

def register(api) -> None:
    # Runtime-context-aware services are installed in on_enable, matching the
    # existing core.image service lifetime pattern.
    api.register_service(
        key="resolver",
        factory=lambda: None,
        lazy=True,
        replace=True,
        description="Resolve mapped video media and decode bounded frame previews.",
    )
    api.register_service(
        key="temporal_state",
        factory=lambda: None,
        lazy=True,
        replace=True,
        description=(
            "Plugin-owned frame/time focus below the platform dataset-row selection boundary."
        ),
    )
    api.register_service(
        key="overlay_registry",
        factory=lambda: None,
        lazy=True,
        replace=True,
        description=(
            "Provider-neutral catalogue and transient selection state for overlays rendered on the video surface."
        ),
    )

    api.register_panel(
        id="viewer",
        title="Video Viewer",
        factory=create_video_viewer_panel,
        description=(
            "Inspect the focused video record with persistent frame rendering, bounded managed "
            "fallback previews, and browser-buffered HTTP(S) frame-sequence playback. Platform "
            "selection owns the row; core.video owns frame/time state."
        ),
        category="Video",
        icon="video",
        tags=["core", "video", "viewer", "selection", "temporal"],
        required_mappings=[RECORD_ID_MAPPING, VIDEO_URI_MAPPING],
        optional_mappings=OPTIONAL_VIDEO_MAPPINGS,
        uses_services=["core.video.resolver", "core.video.temporal_state", "core.video.overlay_registry"],
        produces=["video.temporal.changed", "video.display.changed", "video.display.info.changed"],
        default_layout={"x": 0, "y": 0, "w": 7, "h": 7},
        default_open_kwargs={"max_size": 1280, "preview_fps": 12.0, "prefetch_count": 6},
        state_version=3,
    )

    api.register_action(
        id="register_manifest",
        title="Register Video Manifest",
        handler=register_manifest_action,
        description=(
            "Register an importer-owned Parquet video manifest through the canonical "
            "core.video semantic contract without materialising it."
        ),
        category="Core",
        icon="database-import",
        tags=["core", "video", "dataset", "manifest", "integration"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
        },
        run_in_job=False,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "required": ["manifest_path", "dataset_id", "columns", "row_count"],
            "properties": {
                "manifest_path": {"type": "string"},
                "dataset_id": {"type": "string"},
                "dataset_name": {"type": ["string", "null"]},
                "columns": {"type": "array", "items": {"type": "string"}},
                "row_count": {"type": "integer", "minimum": 0},
                "mappings": {"type": ["object", "null"]},
                "base_path": {"type": ["string", "null"]},
                "source_format": {"type": "string", "default": "video_manifest"},
                "extra_metadata": {"type": ["object", "null"]},
                "set_active": {"type": "boolean", "default": True},
            },
        },
    )

    api.register_action(
        id="build_manifest",
        title="Build Video Manifest",
        handler=build_manifest_action,
        description=(
            "Build a metadata-only AstronomicAL video dataset from a folder. "
            "The dataset stores paths/URIs, never video bytes."
        ),
        category="Core",
        icon="folder",
        tags=["core", "video", "dataset", "manifest"],
        inputs={
            "dataset": False,
            "selection": "none",
            "columns": "none",
            "numeric_columns": "none",
        },
        run_in_job=True,
        outputs=["dataset.loaded"],
        params_schema={
            "type": "object",
            "properties": {
                "root": {"type": "string", "title": "Video folder"},
                "dataset_id": {
                    "type": "string",
                    "title": "Dataset ID",
                    "default": "video_dataset",
                },
                "dataset_name": {
                    "type": "string",
                    "title": "Dataset name",
                    "default": "Video Dataset",
                },
                "extensions": {
                    "type": "string",
                    "title": "Extensions",
                    "default": ".mp4,.mov,.m4v,.mkv,.webm,.avi,.mpeg,.mpg",
                },
                "recursive": {"type": "boolean", "default": True},
                "label_from_parent": {"type": "boolean", "default": False},
                "relative_paths": {"type": "boolean", "default": True},
                "write_parquet": {"type": "boolean", "default": True},
                "set_active": {"type": "boolean", "default": True},
            },
            "required": ["root"],
        },
    )

def on_enable(context: Any, manager: Any = None) -> None:
    del manager
    services = getattr(context, "services", None)
    if services is None:
        return
    services.set(
        "core.video.resolver",
        VideoAssetResolver(context),
        replace=True,
        owner="core.video",
    )
    services.set(
        "core.video.temporal_state",
        VideoTemporalStateManager(context),
        replace=True,
        owner="core.video",
    )
    services.set(
        "core.video.overlay_registry",
        VideoOverlayRegistry(context),
        replace=True,
        owner="core.video",
    )

def on_disable(context: Any, manager: Any = None) -> None:
    del manager
    services = getattr(context, "services", None)
    if services is None:
        return
    # Temporal state subscribes to EventBus and should be disposed before the
    # resolver is removed.
    for key in ("core.video.overlay_registry", "core.video.temporal_state", "core.video.resolver"):
        try:
            services.remove(key, owner="core.video", dispose=True)
        except Exception:
            pass

def create_video_viewer_panel(
    context,
    manager=None,
    **kwargs,
):
    from .viewer import VideoViewerPanel

    controller = VideoViewerPanel(
        context=context,
        manager=manager,
        max_size=int(kwargs.get("max_size", 1280)),
        preview_fps=float(kwargs.get("preview_fps", 12.0)),
        prefetch_count=int(kwargs.get("prefetch_count", 6)),
    )
    return controller.panel(), controller

def register_manifest_action(
    context: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> dict[str, Any]:
    del cancel_token
    params = dict(getattr(request, "params", {}) or {})
    manifest_path = str(params.get("manifest_path") or "").strip()
    dataset_id = str(params.get("dataset_id") or "").strip()
    if not manifest_path:
        raise ValueError("Missing required parameter: manifest_path")
    if not dataset_id:
        raise ValueError("Missing required parameter: dataset_id")
    return register_video_manifest_path(
        context,
        manifest_path=manifest_path,
        dataset_id=dataset_id,
        name=(str(params.get("dataset_name") or "").strip() or None),
        columns=[str(value) for value in (params.get("columns") or ())],
        row_count=int(params.get("row_count") or 0),
        mappings=dict(params.get("mappings") or {}),
        base_path=(str(params.get("base_path") or "").strip() or None),
        set_active=bool(params.get("set_active", True)),
        source_format=str(params.get("source_format") or "video_manifest"),
        extra_metadata=dict(params.get("extra_metadata") or {}),
    )

def build_manifest_action(
    context: Any,
    request: ActionRequest,
    cancel_token: Any = None,
) -> dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    root = params.get("root")
    if not root:
        raise ValueError("Missing required parameter: root")

    dataset_id = slugify(params.get("dataset_id") or "video_dataset")
    dataset_name = params.get("dataset_name") or dataset_id
    df = build_video_manifest_dataframe(
        root,
        recursive=bool(params.get("recursive", True)),
        extensions=params.get("extensions"),
        label_from_parent=bool(params.get("label_from_parent", False)),
        relative_paths=bool(params.get("relative_paths", True)),
        cancel_token=cancel_token,
    )
    if cancel_token is not None:
        cancelled = getattr(cancel_token, "cancelled", None)
        if callable(cancelled) and cancelled():
            return {"cancelled": True, "dataset_id": dataset_id, "rows": 0}

    return register_video_manifest_dataset(
        context,
        df,
        dataset_id=dataset_id,
        name=dataset_name,
        base_path=root,
        set_active=bool(params.get("set_active", True)),
        write_parquet=bool(params.get("write_parquet", True)),
    )