"""Core video contracts, media resolution, and bounded temporal viewing."""

from .contracts import (
    RECORD_ID_SEMANTIC,
    VIDEO_DURATION_SEMANTIC,
    VIDEO_FPS_SEMANTIC,
    VIDEO_FRAME_COUNT_SEMANTIC,
    VIDEO_FRAME_INDEX_STEP_SEMANTIC,
    VIDEO_FRAME_URIS_SEMANTIC,
    VIDEO_HEIGHT_SEMANTIC,
    VIDEO_PATH_SEMANTIC,
    VIDEO_THUMBNAIL_URI_SEMANTIC,
    VIDEO_URI_SEMANTIC,
    VIDEO_URL_SEMANTIC,
    VIDEO_WIDTH_SEMANTIC,
    VideoAssetRef,
    VideoFramePreview,
    VideoMetadata,
    VideoTemporalFocus,
)

__all__ = [
    "RECORD_ID_SEMANTIC",
    "VIDEO_DURATION_SEMANTIC",
    "VIDEO_FPS_SEMANTIC",
    "VIDEO_FRAME_COUNT_SEMANTIC",
    "VIDEO_FRAME_INDEX_STEP_SEMANTIC",
    "VIDEO_FRAME_URIS_SEMANTIC",
    "VIDEO_HEIGHT_SEMANTIC",
    "VIDEO_PATH_SEMANTIC",
    "VIDEO_THUMBNAIL_URI_SEMANTIC",
    "VIDEO_URI_SEMANTIC",
    "VIDEO_URL_SEMANTIC",
    "VIDEO_WIDTH_SEMANTIC",
    "VideoAssetRef",
    "VideoFramePreview",
    "VideoMetadata",
    "VideoTemporalFocus",
]
