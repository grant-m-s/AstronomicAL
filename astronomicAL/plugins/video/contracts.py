from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

RECORD_ID_SEMANTIC = "record_id"
VIDEO_URI_SEMANTIC = "video.uri"
VIDEO_PATH_SEMANTIC = "video.path"
VIDEO_URL_SEMANTIC = "video.url"
VIDEO_THUMBNAIL_URI_SEMANTIC = "video.thumbnail_uri"
VIDEO_DURATION_SEMANTIC = "video.duration"
VIDEO_FPS_SEMANTIC = "video.fps"
VIDEO_FRAME_COUNT_SEMANTIC = "video.frame_count"
VIDEO_WIDTH_SEMANTIC = "video.width"
VIDEO_HEIGHT_SEMANTIC = "video.height"
VIDEO_MEDIA_KIND_SEMANTIC = "video.media_kind"
VIDEO_FRAME_URI_TEMPLATE_SEMANTIC = "video.frame_uri_template"
VIDEO_FRAME_URIS_SEMANTIC = "video.frame_uris"
VIDEO_FRAME_INDEX_OFFSET_SEMANTIC = "video.frame_index_offset"
VIDEO_FRAME_INDEX_STEP_SEMANTIC = "video.frame_index_step"

VIDEO_MEDIA_KIND_FILE = "video_file"
VIDEO_MEDIA_KIND_FRAME_SEQUENCE = "frame_sequence"

VIDEO_SOURCE_SEMANTICS = (
    VIDEO_URI_SEMANTIC,
    VIDEO_PATH_SEMANTIC,
    VIDEO_URL_SEMANTIC,
)

VIDEO_METADATA_SEMANTICS = (
    VIDEO_DURATION_SEMANTIC,
    VIDEO_FPS_SEMANTIC,
    VIDEO_FRAME_COUNT_SEMANTIC,
    VIDEO_WIDTH_SEMANTIC,
    VIDEO_HEIGHT_SEMANTIC,
    VIDEO_THUMBNAIL_URI_SEMANTIC,
    VIDEO_MEDIA_KIND_SEMANTIC,
    VIDEO_FRAME_URI_TEMPLATE_SEMANTIC,
    VIDEO_FRAME_URIS_SEMANTIC,
    VIDEO_FRAME_INDEX_OFFSET_SEMANTIC,
    VIDEO_FRAME_INDEX_STEP_SEMANTIC,
)

@dataclass(frozen=True)
class VideoAssetRef:
    """Lightweight reference to video media owned outside the dataset table.

    ``uri`` is the fundamental dataset-level media reference. For ordinary video
    files it identifies the video itself. For ``frame_sequence`` media it is a
    stable sequence/root URI while either ``frame_uri_template`` resolves a regular
    sequence or ``frame_uris`` preserves an explicit logical-frame-to-URI mapping.
    """

    dataset_id: str
    row_id: str
    uri: str
    media_type: str = "video/*"
    source_semantic: str = VIDEO_URI_SEMANTIC
    source_column: str = ""
    media_kind: str = VIDEO_MEDIA_KIND_FILE
    frame_uri_template: str = ""
    frame_uris: tuple[str, ...] = ()
    frame_index_offset: int = 0
    frame_index_step: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class VideoMetadata:
    """Small metadata view used for navigation and bounded frame access."""

    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thumbnail_uri: Optional[str] = None
    source: str = "dataset_mappings"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class DecodedVideoFrame:
    """Decoder-neutral result before the resolver turns it into a UI preview."""

    image_bytes: bytes
    media_type: str
    width: int
    height: int
    requested_frame_index: int
    actual_frame_index: int
    time_seconds: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None

@dataclass(frozen=True)
class VideoFramePreview:
    """Display-ready bounded video-frame preview."""

    asset: VideoAssetRef
    data_uri: str
    frame_index: int
    requested_frame_index: int
    time_seconds: Optional[float]
    width: int
    height: int
    media_type: str
    local_media_path: str
    metadata: VideoMetadata

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["asset"] = self.asset.to_dict()
        result["metadata"] = self.metadata.to_dict()
        return result

@dataclass(frozen=True)
class VideoTemporalFocus:
    """Plugin-owned identity below the platform dataset-row focus boundary."""

    dataset_id: Optional[str] = None
    row_id: Optional[str] = None
    frame_index: int = 0
    time_seconds: Optional[float] = None
    playing: bool = False
    playback_rate: float = 1.0
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    origin: Optional[str] = None
    panel_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
