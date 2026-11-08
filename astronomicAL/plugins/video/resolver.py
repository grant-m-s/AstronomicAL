from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
from collections import OrderedDict
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .contracts import (
    RECORD_ID_SEMANTIC,
    VIDEO_DURATION_SEMANTIC,
    VIDEO_FPS_SEMANTIC,
    VIDEO_FRAME_COUNT_SEMANTIC,
    VIDEO_FRAME_INDEX_OFFSET_SEMANTIC,
    VIDEO_FRAME_INDEX_STEP_SEMANTIC,
    VIDEO_FRAME_URI_TEMPLATE_SEMANTIC,
    VIDEO_FRAME_URIS_SEMANTIC,
    VIDEO_HEIGHT_SEMANTIC,
    VIDEO_MEDIA_KIND_FILE,
    VIDEO_MEDIA_KIND_FRAME_SEQUENCE,
    VIDEO_MEDIA_KIND_SEMANTIC,
    VIDEO_METADATA_SEMANTICS,
    VIDEO_PATH_SEMANTIC,
    VIDEO_SOURCE_SEMANTICS,
    VIDEO_THUMBNAIL_URI_SEMANTIC,
    VIDEO_URI_SEMANTIC,
    VIDEO_URL_SEMANTIC,
    VIDEO_WIDTH_SEMANTIC,
    VideoAssetRef,
    VideoFramePreview,
    VideoMetadata,
)
from .decoder import PyAVVideoDecoder

# Remote-cache writes are shared process-wide.  The resolver service is normally a
# singleton, but plugin reloads/tests may temporarily create more than one resolver
# pointing at the same cache directory.  A fixed stripe set avoids an unbounded
# lock-per-frame registry while still coalescing concurrent requests for the same
# cache entry.
_REMOTE_CACHE_LOCKS = tuple(threading.Lock() for _ in range(128))

class VideoAssetResolver:
    """Resolve mapped video rows and provide bounded local frame access.

    Dataset rows retain references only. Ordinary video files are materialised
    into the plugin cache only when a frame/probe request needs them. Frame
    sequences fetch/cache one requested image at a time and therefore do not need
    PyAV or whole-video transfer.
    """

    service_key = "core.video.resolver"

    def __init__(
        self,
        context: Any,
        *,
        cache_dir: str | Path | None = None,
        decoder: Any = None,
    ) -> None:
        self.context = context
        self.cache_dir = Path(cache_dir or ".astronomical_cache/videos").expanduser()
        self.asset_cache_dir = self.cache_dir / "assets"
        self.frame_cache_dir = self.cache_dir / "frames"
        self.asset_cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        self.decoder = decoder or PyAVVideoDecoder()
        # Display interpretation is plugin-owned, transient state keyed by the
        # platform dataset/row identity.  It never changes dataset contents.
        self._display_modes: dict[tuple[str, str], str] = {}
        self._row_image_info: dict[tuple[str, str], dict[str, Any]] = {}
        self._frame_image_info: dict[str, dict[str, Any]] = {}
        # Bounded plugin-owned preview cache shared by Video Viewer and consumers such
        # as vision.segmentation.  It avoids repeating Pillow decode/colour conversion
        # for the same focused frame while keeping heavy media out of platform state.
        self._preview_cache_lock = threading.RLock()
        self._preview_cache: OrderedDict[tuple[Any, ...], VideoFramePreview] = OrderedDict()
        self._preview_cache_limit = 96
        self._preview_key_locks = tuple(threading.Lock() for _ in range(64))

    def decoder_available(self) -> bool:
        available = getattr(self.decoder, "available", None)
        if callable(available):
            try:
                return bool(available())
            except Exception:
                return False
        return True

    def get_display_mode(self, dataset_id: str, row_id: Any) -> str:
        """Return the transient display interpretation for one video row.

        ``auto`` is deliberately conservative for scientific frame sequences: ordinary
        single/RGB images may stay browser-direct, while images with extra channels are
        rendered through Pillow so channel 4 is not accidentally treated as alpha.
        """

        return self._display_modes.get((str(dataset_id), str(row_id)), "auto")

    def set_display_mode(
        self,
        dataset_id: str,
        row_id: Any,
        mode: str,
        *,
        origin: str | None = None,
        panel_id: str | None = None,
    ) -> str:
        selected = _normalise_video_display_mode(mode)
        key = (str(dataset_id), str(row_id))
        previous = self._display_modes.get(key, "auto")
        self._display_modes[key] = selected
        if previous != selected:
            events = getattr(self.context, "events", None)
            publish = getattr(events, "publish", None)
            if callable(publish):
                try:
                    publish(
                        "video.display.changed",
                        {
                            "dataset_id": key[0],
                            "row_id": key[1],
                            "display_mode": selected,
                            "origin": origin,
                            "panel_id": panel_id,
                        },
                    )
                except Exception:
                    pass
        return selected

    def display_info_for_row(self, dataset_id: str, row_id: Any) -> dict[str, Any]:
        return dict(self._row_image_info.get((str(dataset_id), str(row_id)), {}))

    def _preview_cache_get(self, key: tuple[Any, ...]) -> VideoFramePreview | None:
        with self._preview_cache_lock:
            value = self._preview_cache.get(key)
            if value is None:
                return None
            self._preview_cache.move_to_end(key)
            return value

    def _preview_cache_put(self, key: tuple[Any, ...], preview: VideoFramePreview) -> None:
        with self._preview_cache_lock:
            self._preview_cache[key] = preview
            self._preview_cache.move_to_end(key)
            while len(self._preview_cache) > self._preview_cache_limit:
                self._preview_cache.popitem(last=False)

    def browser_direct_display_transform(
        self,
        asset: VideoAssetRef,
        *,
        display_mode: str | None = None,
    ) -> str:
        """Return a browser-safe direct display transform for a frame sequence.

        Direct browser playback is intentionally conservative.  AstronomicAL does
        not reinterpret scientific extra channels with SVG/CSS filters because the
        browser may already have applied alpha/premultiplication semantics before a
        filter runs.  That can corrupt RGBD/RGBA scientific frames.

        ``"source"`` means that the original image can be displayed unchanged.
        An empty string means the requested scientific presentation must use the
        managed resolver, which extracts the requested channels losslessly with
        Pillow and returns an opaque display derivative.
        """

        if asset.media_kind != VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            return ""
        mode = _normalise_video_display_mode(
            display_mode
            if display_mode is not None
            else self.get_display_mode(asset.dataset_id, asset.row_id)
        )
        if mode == "source":
            return "source"

        info = self._row_image_info.get((str(asset.dataset_id), str(asset.row_id)))
        if not info:
            # A cold row has not yet been decoded by the managed resolver.  For
            # automatic display, common JPEG frames have a native browser path; 8-bit true-colour
            # PNGs can instead be checked in the browser before lossless RGB
            # presentation.  Do not speculate about palette, grayscale, 16-bit,
            # or arbitrary formats: the browser falls back to managed decoding.
            if mode == "auto":
                try:
                    frame_uri = self.frame_uri(asset, 0)
                    extension = Path(urllib.parse.urlparse(frame_uri).path).suffix.casefold()
                except (ValueError, IndexError, KeyError):
                    return ""
                if extension in {".jpg", ".jpeg"}:
                    return "source"
                if extension == ".png":
                    return "rgb_unpremultiplied_png"
            return ""
        source_mode = str(info.get("source_image_mode") or "")
        channel_count = int(info.get("source_channel_count") or 0)

        # Auto may stay source-direct when native browser semantics are already the
        # desired scientific presentation.  RGBD datasets are commonly encoded as
        # RGBA PNG even though component 4 is depth rather than transparency.  Those
        # frames can still use browser-clocked playback through the dedicated
        # ``rgb_unpremultiplied`` canvas path: createImageBitmap requests an
        # unpremultiplied decode and WebGL writes RGB with output alpha forced opaque.
        # This preserves the first three source channels without the SVG/CSS filter
        # path which produced scanline corruption on scientific PNGs.
        if mode == "auto":
            if bool(info.get("browser_direct_safe_auto", False)):
                return "source"
            if source_mode == "RGBA" and channel_count >= 4:
                return "rgb_unpremultiplied"
            return ""

        # Explicit RGB uses the source directly for true RGB, or the same
        # unpremultiplied browser canvas for RGBA/RGBD. Unsupported colour models
        # remain managed so core.video never guesses at scientific channel meaning.
        if mode == "rgb":
            if source_mode == "RGB" and channel_count == 3:
                return "source"
            if source_mode == "RGBA" and channel_count >= 4:
                return "rgb_unpremultiplied"
            return ""

        # An explicit first channel of a native single-channel image is identical to
        # its source representation.  Component extraction from RGB/RGBA/LA remains
        # managed; no browser colour-matrix path is used.
        if mode.startswith("channel:"):
            try:
                channel_index = int(mode.split(":", 1)[1])
            except (TypeError, ValueError):
                return ""
            if channel_index == 1 and source_mode in {"1", "L"} and channel_count == 1:
                return "source"
            return ""
        return ""

    def browser_direct_frame_uri(
        self,
        asset: VideoAssetRef,
        frame_index: int,
        *,
        display_mode: str | None = None,
    ) -> str:
        """Return the original HTTP(S) frame URI when browser presentation is supported.

        ``browser_direct_display_transform`` describes how the URI may be presented.
        ``source`` uses native browser pixels; ``rgb_unpremultiplied`` is a core.video
        canvas/WebGL path which preserves source RGB while deliberately ignoring a
        scientific fourth channel. Callers must consult that transform rather than
        assuming that every returned URI is safe as a plain HTML image.
        """

        if asset.media_kind != VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            return ""
        uri = self.frame_uri(asset, frame_index)
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        if not self.browser_direct_display_transform(asset, display_mode=display_mode):
            return ""
        return uri

    def browser_direct_frame_uri_for_row(
        self,
        dataset_id: str,
        row_id: Any,
        frame_index: int,
        *,
        display_mode: str | None = None,
    ) -> str:
        return self.browser_direct_frame_uri(
            self.asset_for_row(dataset_id, row_id),
            frame_index,
            display_mode=display_mode,
        )

    def remember_browser_frame_info(
        self, dataset_id: str, row_id: Any, *, source_mode: str, channel_count: int
    ) -> bool:
        """Record a browser-verified 8-bit PNG channel layout without decoding media again."""
        mode = str(source_mode or "")
        count = int(channel_count or 0)
        if (mode, count) not in {("RGB", 3), ("RGBA", 4)}:
            return False
        key = (str(dataset_id), str(row_id))
        previous = dict(self._row_image_info.get(key) or {})
        updated = {
            **previous, "source_image_mode": mode, "source_channel_count": count,
            "source_image_bands": ["R", "G", "B"] + (["A"] if count == 4 else []),
            "browser_direct_safe_auto": count == 3,
        }
        if previous == updated:
            return False
        self._row_image_info[key] = updated
        events = getattr(self.context, "events", None)
        publish = getattr(events, "publish", None)
        if callable(publish):
            try:
                publish("video.display.info.changed", {
                    "dataset_id": key[0], "row_id": key[1],
                    "source_channel_count": count, "source_image_mode": mode,
                })
            except Exception:
                pass
        return True

    def frame_access_available_for_row(self, dataset_id: str, row_id: Any) -> bool:
        try:
            asset = self.asset_for_row(dataset_id, row_id)
        except Exception:
            return False
        if asset.media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            return bool(asset.frame_uris or asset.frame_uri_template)
        return self.decoder_available()

    def frame_access_description_for_row(self, dataset_id: str, row_id: Any) -> str:
        try:
            asset = self.asset_for_row(dataset_id, row_id)
        except Exception:
            return "Video media mapping unavailable."
        if asset.media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            if asset.frame_uris:
                return "Explicit frame-sequence mapping configured; requested frames are fetched individually."
            if asset.frame_uri_template:
                return "Frame-sequence template configured; requested frames are fetched individually."
            return (
                "Frame-sequence media is missing both `video.frame_uris` and "
                "`video.frame_uri_template` mappings."
            )
        if self.decoder_available():
            return "Video decoder ready."
        return "Install optional `av>=12` for encoded-video frame previews."

    def asset_for_row(self, dataset_id: str, row_id: Any) -> VideoAssetRef:
        columns = self._columns_for_lookup(dataset_id)
        row = self.get_row(dataset_id, row_id, columns=columns or None)
        if row is None or row.empty:
            raise KeyError(f"Could not find row {row_id!r} in dataset {dataset_id!r}.")
        record = row.iloc[0].to_dict()
        uri, semantic, column = self._resolve_uri_from_record(dataset_id, record)
        if not uri:
            raise ValueError(
                "No video asset mapping was found. Map `video.uri` to a URI/path "
                "column (or provide video.path/video.url metadata)."
            )
        metadata = self._metadata_from_record(dataset_id, record)
        media_kind = self._media_kind_from_record(dataset_id, record)
        frame_template = self._frame_template_from_record(dataset_id, record)
        frame_uris = self._frame_uris_from_record(dataset_id, record, base_uri=str(uri))
        frame_offset = self._frame_index_offset_from_record(dataset_id, record)
        frame_step = self._frame_index_step_from_record(dataset_id, record)
        return VideoAssetRef(
            dataset_id=str(dataset_id),
            row_id=str(row_id),
            uri=str(uri),
            media_type=(
                "image/*"
                if media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE
                else self._guess_media_type(str(uri))
            ),
            source_semantic=semantic,
            source_column=column,
            media_kind=media_kind,
            frame_uri_template=frame_template,
            frame_uris=frame_uris,
            frame_index_offset=frame_offset,
            frame_index_step=frame_step,
            metadata=metadata.to_dict(),
        )

    def metadata_for_row(self, dataset_id: str, row_id: Any) -> VideoMetadata:
        columns = self._columns_for_lookup(dataset_id)
        row = self.get_row(dataset_id, row_id, columns=columns or None)
        if row is None or row.empty:
            raise KeyError(f"Could not find row {row_id!r} in dataset {dataset_id!r}.")
        record = row.iloc[0].to_dict()
        metadata = self._metadata_from_record(dataset_id, record)
        kind = self._media_kind_from_record(dataset_id, record)
        extra = dict(metadata.extra or {})
        extra["media_kind"] = kind
        if kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            extra["frame_uri_template"] = self._frame_template_from_record(dataset_id, record)
            explicit = self._frame_uris_from_record(dataset_id, record)
            extra["frame_uri_count"] = len(explicit)
            extra["frame_index_offset"] = self._frame_index_offset_from_record(dataset_id, record)
            extra["frame_index_step"] = self._frame_index_step_from_record(dataset_id, record)
        return VideoMetadata(
            duration_seconds=metadata.duration_seconds,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
            thumbnail_uri=metadata.thumbnail_uri,
            source=metadata.source,
            extra=extra,
        )

    def probe_for_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        materialize_remote: bool = False,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> VideoMetadata:
        asset = self.asset_for_row(dataset_id, row_id)
        mapped = self.metadata_for_row(dataset_id, row_id)
        if asset.media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            # Explicit probe verifies one bounded frame rather than touching the
            # complete sequence. This preserves the same user-intent boundary as
            # encoded-video probing while remaining useful for remote sequences.
            frame_uri = self.frame_uri(asset, 0)
            parsed = urllib.parse.urlparse(frame_uri)
            if parsed.scheme in {"http", "https"} and not materialize_remote:
                raise RuntimeError(
                    "Remote frame-sequence probing requires one frame transfer. Call with "
                    "materialize_remote=True only after explicit user intent."
                )
            local_path = self.materialize_uri(
                frame_uri,
                cache_kind="frame",
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )
            image_meta = self._probe_image(local_path)
            return _merge_video_metadata(
                mapped,
                VideoMetadata(
                    width=image_meta[0],
                    height=image_meta[1],
                    source="frame_sequence_probe",
                ),
                source="dataset_mappings+frame_sequence_probe",
            )

        parsed = urllib.parse.urlparse(asset.uri)
        if parsed.scheme in {"http", "https"} and not materialize_remote:
            raise RuntimeError(
                "Remote video probing may require media transfer. Call with "
                "materialize_remote=True only after explicit user intent."
            )
        local_path = self.materialize_asset(
            asset,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )
        probe = getattr(self.decoder, "probe", None)
        if not callable(probe):
            raise RuntimeError("The active core.video decoder does not support metadata probing.")
        decoded = probe(local_path, cancel_token=cancel_token)
        return _merge_video_metadata(mapped, decoded, source="dataset_mappings+decoder_probe")

    def load_frame_for_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        frame_index: int,
        max_size: int = 1280,
        display_mode: str | None = None,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> VideoFramePreview:
        """Resolve exactly one record and decode/fetch one bounded preview frame."""

        _check_cancelled(cancel_token)
        columns = self._columns_for_lookup(dataset_id)
        row = self.get_row(dataset_id, row_id, columns=columns or None)
        if row is None or row.empty:
            raise KeyError(f"Could not find row {row_id!r} in dataset {dataset_id!r}.")
        record = row.iloc[0].to_dict()
        uri, semantic, column = self._resolve_uri_from_record(dataset_id, record)
        if not uri:
            raise ValueError("The focused row does not contain a mapped video asset.")
        mapped_metadata = self._metadata_from_record(dataset_id, record)
        media_kind = self._media_kind_from_record(dataset_id, record)
        frame_template = self._frame_template_from_record(dataset_id, record)
        frame_uris = self._frame_uris_from_record(dataset_id, record, base_uri=str(uri))
        frame_offset = self._frame_index_offset_from_record(dataset_id, record)
        frame_step = self._frame_index_step_from_record(dataset_id, record)
        asset = VideoAssetRef(
            dataset_id=str(dataset_id),
            row_id=str(row_id),
            uri=str(uri),
            media_type=(
                "image/*"
                if media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE
                else self._guess_media_type(str(uri))
            ),
            source_semantic=semantic,
            source_column=column,
            media_kind=media_kind,
            frame_uri_template=frame_template,
            frame_uris=frame_uris,
            frame_index_offset=frame_offset,
            frame_index_step=frame_step,
            metadata=mapped_metadata.to_dict(),
        )

        requested = max(0, int(frame_index or 0))
        if mapped_metadata.frame_count is not None:
            requested = min(requested, max(0, int(mapped_metadata.frame_count) - 1))

        if asset.media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            if not asset.frame_uris and not asset.frame_uri_template:
                raise ValueError(
                    "Frame-sequence media requires either a `video.frame_uris` or "
                    "`video.frame_uri_template` mapping."
                )
            frame_uri = self.frame_uri(asset, requested)
            selected_display_mode = _normalise_video_display_mode(
                display_mode
                if display_mode is not None
                else self.get_display_mode(dataset_id, row_id)
            )
            preview_cache_key = (
                str(dataset_id),
                str(row_id),
                int(requested),
                int(max_size),
                selected_display_mode,
                str(frame_uri),
            )
            cached_preview = self._preview_cache_get(preview_cache_key)
            if cached_preview is not None:
                return cached_preview
            preview_lock = self._preview_key_locks[hash(preview_cache_key) % len(self._preview_key_locks)]
            with preview_lock:
                # Video Viewer and vision.segmentation commonly request the same
                # managed scientific frame at nearly the same instant. Re-check after
                # acquiring the striped lock so only one caller downloads/decodes it.
                cached_preview = self._preview_cache_get(preview_cache_key)
                if cached_preview is not None:
                    return cached_preview
                local_path = self.materialize_uri(
                    frame_uri,
                    cache_kind="frame",
                    cancel_token=cancel_token,
                    progress_callback=progress_callback,
                )
                image_bytes, media_type, width, height, display_info = self._encode_bounded_image(
                    local_path,
                    max_size=max_size,
                    display_mode=selected_display_mode,
                )
                display_info = dict(display_info)
                display_info["frame_uri"] = frame_uri
                self._frame_image_info[str(frame_uri)] = dict(display_info)
                row_key = (str(dataset_id), str(row_id))
                previous_info = self._row_image_info.get(row_key)
                self._row_image_info[row_key] = dict(display_info)
                if previous_info != display_info:
                    events = getattr(self.context, "events", None)
                    publish = getattr(events, "publish", None)
                    if callable(publish):
                        try:
                            publish(
                                "video.display.info.changed",
                                {
                                    "dataset_id": row_key[0],
                                    "row_id": row_key[1],
                                    "display_mode": selected_display_mode,
                                    "source_channel_count": int(display_info.get("source_channel_count") or 0),
                                    "source_image_mode": str(display_info.get("source_image_mode") or ""),
                                },
                            )
                        except Exception:
                            pass
                merged = _merge_video_metadata(
                    mapped_metadata,
                    VideoMetadata(
                        width=width,
                        height=height,
                        source="frame_sequence",
                        extra={
                            "media_kind": VIDEO_MEDIA_KIND_FRAME_SEQUENCE,
                            "frame_uri": frame_uri,
                            **display_info,
                        },
                    ),
                    source="dataset_mappings+frame_sequence",
                )
                encoded = base64.b64encode(image_bytes).decode("ascii")
                preview = VideoFramePreview(
                    asset=asset,
                    data_uri=f"data:{media_type};base64,{encoded}",
                    frame_index=requested,
                    requested_frame_index=int(frame_index or 0),
                    time_seconds=(
                        requested / mapped_metadata.fps
                        if mapped_metadata.fps is not None and mapped_metadata.fps > 0
                        else None
                    ),
                    width=width,
                    height=height,
                    media_type=media_type,
                    local_media_path=str(local_path),
                    metadata=merged,
                )
                self._preview_cache_put(preview_cache_key, preview)
                return preview
        local_path = self.materialize_asset(
            asset,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )
        _check_cancelled(cancel_token)
        extract = getattr(self.decoder, "extract_frame", None)
        if not callable(extract):
            raise RuntimeError("The active core.video decoder does not support frame extraction.")
        decoded = extract(
            local_path,
            frame_index=requested,
            fps_hint=mapped_metadata.fps,
            frame_count_hint=mapped_metadata.frame_count,
            max_size=max_size,
            cancel_token=cancel_token,
        )
        merged = _merge_video_metadata(
            mapped_metadata,
            VideoMetadata(
                fps=decoded.fps,
                frame_count=decoded.frame_count,
                width=decoded.width,
                height=decoded.height,
                source="decoder_frame",
                extra={"media_kind": VIDEO_MEDIA_KIND_FILE},
            ),
            source="dataset_mappings+decoder_frame",
        )
        encoded = base64.b64encode(decoded.image_bytes).decode("ascii")
        return VideoFramePreview(
            asset=asset,
            data_uri=f"data:{decoded.media_type};base64,{encoded}",
            frame_index=int(decoded.actual_frame_index),
            requested_frame_index=int(decoded.requested_frame_index),
            time_seconds=decoded.time_seconds,
            width=int(decoded.width),
            height=int(decoded.height),
            media_type=str(decoded.media_type),
            local_media_path=str(local_path),
            metadata=merged,
        )

    def frame_uri(self, asset: VideoAssetRef, frame_index: int) -> str:
        if asset.media_kind != VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            raise ValueError("frame_uri() is only valid for frame-sequence assets.")
        logical = max(0, int(frame_index or 0))
        explicit = tuple(str(value) for value in (getattr(asset, "frame_uris", ()) or ()) if str(value).strip())
        if explicit:
            if logical >= len(explicit):
                raise IndexError(
                    f"Logical frame {logical} is outside the explicit frame URI sequence "
                    f"of length {len(explicit)}."
                )
            return explicit[logical]

        template = str(asset.frame_uri_template or "").strip()
        if not template:
            raise ValueError(
                "Frame-sequence asset has neither explicit frame URIs nor a frame URI template."
            )
        step = max(1, int(getattr(asset, "frame_index_step", 1) or 1))
        index = max(0, int(asset.frame_index_offset or 0) + logical * step)
        try:
            return template.format(
                frame_index=index,
                index=index,
                video_uri=asset.uri,
                row_id=asset.row_id,
                dataset_id=asset.dataset_id,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not format video frame URI template {template!r}: {exc}"
            ) from exc

    def materialize_asset(
        self,
        asset: VideoAssetRef,
        *,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> Path:
        if asset.media_kind == VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
            raise ValueError(
                "Frame-sequence assets are materialised one frame at a time; "
                "call load_frame_for_row() rather than materialize_asset()."
            )
        return self.materialize_uri(
            asset.uri,
            cache_kind="asset",
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )

    def materialize_uri(
        self,
        uri: str,
        *,
        cache_kind: str = "asset",
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> Path:
        """Return one local file for a file/http(s) URI.

        Remote cache writes are coalesced by cache entry.  The Video Viewer and
        another capability (for example ``vision.segmentation``) may legitimately
        request the same frame at the same time.  Only one caller is allowed to
        own the deterministic ``.part`` file; waiters re-check and reuse the
        completed cache entry instead of racing the final ``os.replace``.

        Frame images are deliberately downloaded atomically rather than resumed.
        They are small, independently addressable objects and a stale ``.part``
        file can otherwise produce HTTP 416 on layout restore when its byte count
        already equals (or exceeds) the remote object length. Large encoded video
        assets retain resumable Range requests, with one fail-safe retry from byte
        zero when a remote server rejects a stale range.
        """

        _check_cancelled(cancel_token)
        parsed = urllib.parse.urlparse(str(uri))
        if parsed.scheme == "file":
            path = Path(urllib.request.url2pathname(parsed.path)).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Media file does not exist: {path}")
            return path.resolve()
        if parsed.scheme == "":
            path = Path(str(uri)).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Media file does not exist: {path}")
            return path.resolve()
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported media URI scheme {parsed.scheme!r}. "
                "core.video supports file, http, and https references."
            )

        final_path = self._remote_cache_path(str(uri), cache_kind=cache_kind)
        label = "frame" if cache_kind == "frame" else "video media"
        cached = self._cached_remote_path(
            final_path,
            label=label,
            progress_callback=progress_callback,
        )
        if cached is not None:
            return cached

        final_path.parent.mkdir(parents=True, exist_ok=True)
        cache_lock = _remote_cache_lock(final_path)
        acquired = cache_lock.acquire(blocking=False)
        if not acquired:
            _emit_progress(
                progress_callback,
                phase="waiting",
                bytes_done=0,
                bytes_total=0,
                message=f"Waiting for another request to finish caching this {label}…",
            )
            while not acquired:
                _check_cancelled(cancel_token)
                acquired = cache_lock.acquire(timeout=0.1)

        try:
            _check_cancelled(cancel_token)
            # The caller which held the lock may have completed the same transfer
            # while we were waiting.  This second check is the coalescing point.
            cached = self._cached_remote_path(
                final_path,
                label=label,
                progress_callback=progress_callback,
            )
            if cached is not None:
                return cached
            return self._download_remote_uri_locked(
                str(uri),
                final_path=final_path,
                cache_kind=cache_kind,
                label=label,
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )
        finally:
            if acquired:
                cache_lock.release()

    @staticmethod
    def _cached_remote_path(
        final_path: Path,
        *,
        label: str,
        progress_callback: Any = None,
    ) -> Path | None:
        try:
            size = int(final_path.stat().st_size)
        except (FileNotFoundError, OSError):
            return None
        if not final_path.is_file() or size <= 0:
            return None
        _emit_progress(
            progress_callback,
            phase="cached",
            bytes_done=size,
            bytes_total=size,
            message=f"Using cached {label}.",
        )
        return final_path

    def _download_remote_uri_locked(
        self,
        uri: str,
        *,
        final_path: Path,
        cache_kind: str,
        label: str,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> Path:
        """Download one remote cache entry while its process-wide lock is held."""

        part_path = final_path.with_suffix(final_path.suffix + ".part")

        # A frame is a small independently fetchable object. Restarting it from
        # byte zero is cheaper and much safer than carrying a Range resume across
        # application/layout restarts. Encoded video assets may still be large, so
        # they retain resumable downloads.
        resume_allowed = str(cache_kind) != "frame"
        if not resume_allowed and part_path.exists():
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass

        existing = (
            int(part_path.stat().st_size)
            if resume_allowed and part_path.exists()
            else 0
        )
        base_headers = {"User-Agent": "AstronomicAL-core.video/0.2.19"}

        response_context = None
        status = 200
        append = False
        for attempt in range(2):
            headers = dict(base_headers)
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(str(uri), headers=headers)
            try:
                response_context = urllib.request.urlopen(request, timeout=60)
                status = int(getattr(response_context, "status", 200) or 200)
                append = existing > 0 and status == 206
                if existing > 0 and not append:
                    existing = 0
                break
            except urllib.error.HTTPError as exc:
                # A stale/incomplete resume marker is the common cause of 416. If
                # the server tells us that the part is already exactly complete,
                # promote it atomically. Otherwise discard it and retry once with
                # a normal full-object GET.
                if int(exc.code) == 416 and existing > 0 and attempt == 0:
                    total = _unsatisfied_range_total(getattr(exc, "headers", None))
                    try:
                        part_size = int(part_path.stat().st_size)
                    except Exception:
                        part_size = -1
                    if total is not None and total > 0 and part_size == total:
                        os.replace(part_path, final_path)
                        _emit_progress(
                            progress_callback,
                            phase="cached",
                            bytes_done=int(total),
                            bytes_total=int(total),
                            message=f"Recovered completed {label} cache after stale Range state.",
                        )
                        return final_path
                    try:
                        part_path.unlink()
                    except FileNotFoundError:
                        pass
                    existing = 0
                    continue
                raise RuntimeError(
                    f"HTTP {int(exc.code)} while fetching {label} {str(uri)!r}"
                ) from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                raise RuntimeError(
                    f"Could not reach {label} {str(uri)!r}: {reason}"
                ) from exc
        else:
            raise RuntimeError(f"Could not open {label} {str(uri)!r} for download.")

        if response_context is None:
            raise RuntimeError(f"Could not open {label} {str(uri)!r} for download.")

        try:
            with response_context as response:
                remaining = _content_length(response)
                total = None if remaining is None else existing + remaining
                mode = "ab" if append else "wb"
                done = existing
                with part_path.open(mode) as handle:
                    while True:
                        _check_cancelled(cancel_token)
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        _emit_progress(
                            progress_callback,
                            phase="downloading",
                            bytes_done=done,
                            bytes_total=total,
                            message=(
                                f"Caching {label}: {_format_bytes(done)}"
                                + (f" / {_format_bytes(total)}" if total is not None else "")
                            ),
                        )

            _check_cancelled(cancel_token)
            os.replace(part_path, final_path)
        except BaseException:
            # Frame downloads are intentionally non-resumable.  Remove any partial
            # frame before releasing the lock so the next waiter starts from a
            # clean state.  Large encoded assets retain their partial for resume.
            if not resume_allowed:
                try:
                    part_path.unlink()
                except FileNotFoundError:
                    pass
            raise

        size = int(final_path.stat().st_size)
        _emit_progress(
            progress_callback,
            phase="cached",
            bytes_done=size,
            bytes_total=size,
            message=f"{label.title()} cached locally.",
        )
        return final_path

    def get_row(
        self,
        dataset_id: str,
        row_id: Any,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return pd.DataFrame()
        id_column = self.get_mapping(dataset_id, RECORD_ID_SEMANTIC) or "record_id"
        get_row = getattr(datasets, "get_row_by_id", None)
        if callable(get_row):
            try:
                return get_row(dataset_id, row_id, id_column=id_column, columns=columns)
            except Exception:
                if columns is not None:
                    try:
                        return get_row(dataset_id, row_id, id_column=id_column, columns=None)
                    except Exception:
                        pass
        get_source = getattr(datasets, "get_source", None)
        if callable(get_source):
            try:
                source = get_source(dataset_id)
                method = getattr(source, "get_row_by_id", None)
                if callable(method):
                    return method(row_id, id_column=id_column, columns=columns)
            except Exception:
                pass
        raise RuntimeError(
            "core.video requires bounded DatasetManager.get_row_by_id()/DatasetSource access."
        )

    def get_mapping(self, dataset_id: str, semantic_name: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None
        method = getattr(datasets, "get_mapping", None)
        if callable(method):
            try:
                value = method(dataset_id, semantic_name)
            except Exception:
                value = None
            if value:
                return str(value)
        try:
            meta = datasets.get_meta(dataset_id)
        except Exception:
            meta = {}
        mappings = dict(meta.get("column_mappings") or {}) if isinstance(meta, Mapping) else {}
        value = mappings.get(semantic_name)
        return str(value) if value else None

    def _columns_for_lookup(self, dataset_id: str) -> list[str]:
        columns: list[str] = []
        for semantic in (RECORD_ID_SEMANTIC, *VIDEO_SOURCE_SEMANTICS, *VIDEO_METADATA_SEMANTICS):
            mapped = self.get_mapping(dataset_id, semantic)
            if mapped and mapped != "Use Index" and mapped not in columns:
                columns.append(mapped)
        return columns

    def _resolve_uri_from_record(
        self,
        dataset_id: str,
        record: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        for semantic in VIDEO_SOURCE_SEMANTICS:
            column = self.get_mapping(dataset_id, semantic)
            if not column:
                continue
            value = _clean_value(record.get(column))
            if not value:
                continue
            return self._normalise_uri(dataset_id, value, semantic), semantic, column

        for semantic, aliases in (
            (VIDEO_URI_SEMANTIC, ("video_uri", "video", "uri")),
            (VIDEO_PATH_SEMANTIC, ("video_path", "path", "filepath", "file_path")),
            (VIDEO_URL_SEMANTIC, ("video_url", "url")),
        ):
            for column in aliases:
                if column not in record:
                    continue
                value = _clean_value(record.get(column))
                if value:
                    return self._normalise_uri(dataset_id, value, semantic), semantic, column
        return "", VIDEO_URI_SEMANTIC, ""

    def _metadata_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> VideoMetadata:
        def mapped_value(semantic: str, aliases: Sequence[str]) -> Any:
            column = self.get_mapping(dataset_id, semantic)
            if column and column in record:
                value = record.get(column)
                if not _is_missing(value):
                    return value
            for alias in aliases:
                if alias in record and not _is_missing(record.get(alias)):
                    return record.get(alias)
            return None

        thumbnail = mapped_value(
            VIDEO_THUMBNAIL_URI_SEMANTIC,
            ("video_thumbnail_uri", "thumbnail_uri", "thumbnail_url", "thumbnail_path"),
        )
        if thumbnail is not None:
            thumbnail = self._normalise_uri(
                dataset_id,
                str(thumbnail),
                VIDEO_THUMBNAIL_URI_SEMANTIC,
            )
        kind = self._media_kind_from_record(dataset_id, record)
        return VideoMetadata(
            duration_seconds=_nonnegative_float(
                mapped_value(VIDEO_DURATION_SEMANTIC, ("video_duration", "duration_seconds", "duration"))
            ),
            fps=_positive_float(mapped_value(VIDEO_FPS_SEMANTIC, ("video_fps", "fps"))),
            frame_count=_positive_int(
                mapped_value(VIDEO_FRAME_COUNT_SEMANTIC, ("video_frame_count", "frame_count", "num_frames"))
            ),
            width=_positive_int(mapped_value(VIDEO_WIDTH_SEMANTIC, ("video_width", "width"))),
            height=_positive_int(mapped_value(VIDEO_HEIGHT_SEMANTIC, ("video_height", "height"))),
            thumbnail_uri=str(thumbnail) if thumbnail else None,
            source="dataset_mappings",
            extra={"media_kind": kind},
        )

    def _media_kind_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> str:
        value = self._mapped_or_alias_value(
            dataset_id,
            record,
            VIDEO_MEDIA_KIND_SEMANTIC,
            ("video_media_kind", "media_kind"),
        )
        text = str(value or "").strip().casefold().replace("-", "_")
        if text in {"frame_sequence", "frames", "image_sequence", "jpeg_sequence"}:
            return VIDEO_MEDIA_KIND_FRAME_SEQUENCE
        if not text:
            template = self._frame_template_from_record(dataset_id, record)
            explicit = self._frame_uris_from_record(dataset_id, record)
            if template or explicit:
                return VIDEO_MEDIA_KIND_FRAME_SEQUENCE
        return VIDEO_MEDIA_KIND_FILE

    def _frame_template_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> str:
        value = self._mapped_or_alias_value(
            dataset_id,
            record,
            VIDEO_FRAME_URI_TEMPLATE_SEMANTIC,
            ("video_frame_uri_template", "frame_uri_template"),
        )
        text = str(value or "").strip()
        if not text:
            return ""
        # URI templates commonly contain braces; _normalise_uri treats them as
        # ordinary path characters for relative local templates and leaves HTTP
        # schemes untouched.
        return self._normalise_uri(dataset_id, text, VIDEO_FRAME_URI_TEMPLATE_SEMANTIC)

    def _frame_uris_from_record(
        self,
        dataset_id: str,
        record: Mapping[str, Any],
        *,
        base_uri: str = "",
    ) -> tuple[str, ...]:
        value = self._mapped_or_alias_value(
            dataset_id,
            record,
            VIDEO_FRAME_URIS_SEMANTIC,
            ("video_frame_uris", "frame_uris"),
        )
        if value is None:
            return ()

        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ()
            if text.startswith("["):
                try:
                    decoded = json.loads(text)
                except Exception:
                    decoded = None
                if isinstance(decoded, list):
                    value = decoded
                else:
                    value = [text]
            else:
                value = [text]
        elif not isinstance(value, (list, tuple)):
            try:
                value = list(value)
            except Exception:
                return ()

        result: list[str] = []
        for raw in value:
            text = str(raw or "").strip()
            if not text:
                continue
            parsed = urllib.parse.urlparse(text)
            if parsed.scheme:
                result.append(text)
                continue
            base = str(base_uri or "").strip()
            if base:
                base_parsed = urllib.parse.urlparse(base)
                if base_parsed.scheme in {"http", "https", "file"}:
                    result.append(urllib.parse.urljoin(base.rstrip("/") + "/", text))
                    continue
            result.append(self._normalise_uri(dataset_id, text, VIDEO_FRAME_URIS_SEMANTIC))
        return tuple(result)

    def _frame_index_offset_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> int:
        value = self._mapped_or_alias_value(
            dataset_id,
            record,
            VIDEO_FRAME_INDEX_OFFSET_SEMANTIC,
            ("video_frame_index_offset", "frame_index_offset"),
        )
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _frame_index_step_from_record(self, dataset_id: str, record: Mapping[str, Any]) -> int:
        value = self._mapped_or_alias_value(
            dataset_id,
            record,
            VIDEO_FRAME_INDEX_STEP_SEMANTIC,
            ("video_frame_index_step", "frame_index_step", "frame_stride"),
        )
        try:
            return max(1, int(value or 1))
        except Exception:
            return 1

    def _mapped_or_alias_value(
        self,
        dataset_id: str,
        record: Mapping[str, Any],
        semantic: str,
        aliases: Sequence[str],
    ) -> Any:
        column = self.get_mapping(dataset_id, semantic)
        if column and column in record and not _is_missing(record.get(column)):
            return record.get(column)
        for alias in aliases:
            if alias in record and not _is_missing(record.get(alias)):
                return record.get(alias)
        return None

    def _normalise_uri(self, dataset_id: str, value: str, semantic: str) -> str:
        text = str(value).strip()
        parsed = urllib.parse.urlparse(text)
        if parsed.scheme:
            return text
        path = Path(text).expanduser()
        if not path.is_absolute():
            base = self._dataset_base_path(dataset_id)
            if base:
                path = Path(base).expanduser() / path
        if semantic == VIDEO_URL_SEMANTIC and not path.exists():
            return text
        try:
            # ``Path.as_uri`` percent-encodes braces, which would destroy a local
            # frame template. Preserve the template token explicitly.
            if semantic == VIDEO_FRAME_URI_TEMPLATE_SEMANTIC and "{" in text:
                absolute = str(path.resolve()).replace("\\", "/")
                return "file://" + urllib.parse.quote(absolute, safe="/:{}")
            return path.resolve().as_uri()
        except Exception:
            return str(path)

    def _dataset_base_path(self, dataset_id: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None
        try:
            meta = datasets.get_meta(dataset_id)
        except Exception:
            try:
                meta = getattr(datasets.get(dataset_id), "meta", {}) or {}
            except Exception:
                meta = {}
        for key in ("video_base_path", "base_path", "source_dir", "source_path"):
            value = meta.get(key) if isinstance(meta, Mapping) else None
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.is_file():
                return str(path.parent)
            return str(path)
        return None

    def _remote_cache_path(self, uri: str, *, cache_kind: str = "asset") -> Path:
        parsed = urllib.parse.urlparse(uri)
        suffix = Path(parsed.path).suffix
        default_suffix = ".img" if cache_kind == "frame" else ".video"
        if not suffix or len(suffix) > 12 or not re.match(r"^\.[A-Za-z0-9]+$", suffix):
            suffix = default_suffix
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        root = self.frame_cache_dir if cache_kind == "frame" else self.asset_cache_dir
        return root / digest[:2] / f"{digest}{suffix.lower()}"

    @staticmethod
    def _guess_media_type(uri: str) -> str:
        guessed, _ = mimetypes.guess_type(uri)
        return guessed or "video/*"

    @staticmethod
    def _probe_image(path: Path) -> tuple[int, int]:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Frame-sequence access requires Pillow.") from exc
        with Image.open(path) as image:
            return int(image.width), int(image.height)

    @staticmethod
    def _encode_bounded_image(
        path: Path,
        *,
        max_size: int,
        display_mode: str = "auto",
    ) -> tuple[bytes, str, int, int, dict[str, Any]]:
        """Encode one bounded display frame without browser-side channel tricks.

        Ordinary single-channel/RGB images keep the historical compact JPEG preview
        in ``auto`` mode.  Scientific channel extraction (RGBA/RGBD, explicit RGB, or
        explicit component display) is lossless PNG so the displayed RGB values match
        the source channels exactly and the browser never interprets a data channel as
        alpha.  ``source`` preserves native alpha through PNG when present.
        """
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Frame-sequence access requires Pillow.") from exc

        max_size = max(64, int(max_size or 1280))
        requested = _normalise_video_display_mode(display_mode)
        with Image.open(path) as source:
            source.load()
            bands = tuple(str(value) for value in source.getbands())
            channel_count = len(bands)
            source_mode = str(source.mode or "")
            browser_direct_safe_auto = source_mode in {"1", "L", "P", "RGB"}

            applied = requested
            display_bands: list[str] = []
            output_format = "JPEG"
            media_type = "image/jpeg"
            lossless_display = False

            if requested == "source":
                # Preserve source semantics. Alpha-bearing/palette images stay PNG;
                # ordinary RGB is also kept lossless when Source/native is explicit.
                image = source.copy()
                if image.mode not in {"1", "L", "P", "RGB", "RGBA", "LA"}:
                    image = source.convert("RGB")
                output_format = "PNG"
                media_type = "image/png"
                lossless_display = True
                display_bands = list(bands)
            elif requested.startswith("channel:"):
                channel_index = int(requested.split(":", 1)[1]) - 1
                if channel_index < 0 or channel_index >= channel_count:
                    raise ValueError(
                        f"Display channel {channel_index + 1} is unavailable; source has "
                        f"{channel_count} channel(s)."
                    )
                channel = source.getchannel(channel_index)
                try:
                    image = channel.convert("L").convert("RGB")
                finally:
                    if channel is not source:
                        channel.close()
                display_bands = [
                    bands[channel_index] if channel_index < len(bands) else str(channel_index + 1)
                ]
                output_format = "PNG"
                media_type = "image/png"
                lossless_display = True
            else:
                if requested == "rgb" and channel_count < 3:
                    raise ValueError(
                        f"RGB display requires at least 3 source channels; found {channel_count}."
                    )
                if channel_count >= 3:
                    if all(name in bands for name in ("R", "G", "B")):
                        channels = [source.getchannel(name) for name in ("R", "G", "B")]
                    elif source_mode in {"CMYK", "YCbCr", "HSV"}:
                        channels = []
                    else:
                        channels = [source.getchannel(index) for index in range(3)]
                    if channels:
                        try:
                            image = Image.merge("RGB", tuple(channel.convert("L") for channel in channels))
                        finally:
                            for channel in channels:
                                channel.close()
                        display_bands = [
                            bands[index] if index < len(bands) else str(index + 1)
                            for index in range(3)
                        ]
                    else:
                        image = source.convert("RGB")
                        display_bands = ["R", "G", "B"]
                    applied = "rgb"
                    # RGB derived from any source with extra/non-native components is
                    # scientific channel extraction and must remain lossless.
                    if source_mode != "RGB" or channel_count != 3 or requested == "rgb":
                        output_format = "PNG"
                        media_type = "image/png"
                        lossless_display = True
                else:
                    channel = source.getchannel(0) if channel_count else source.convert("L")
                    try:
                        image = channel.convert("L").convert("RGB")
                    finally:
                        if channel is not source:
                            channel.close()
                    display_bands = [bands[0] if bands else "1"]
                    applied = "channel:1"

            try:
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), resample=Image.Resampling.LANCZOS)
                width, height = image.size
                buffer = BytesIO()
                if output_format == "PNG":
                    image.save(buffer, format="PNG", optimize=False, compress_level=3)
                else:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(buffer, format="JPEG", quality=92, optimize=True)
                info = {
                    "source_image_mode": source_mode,
                    "source_image_bands": list(bands),
                    "source_channel_count": int(channel_count),
                    "display_mode_requested": requested,
                    "display_mode_applied": applied,
                    "display_bands": list(display_bands),
                    "browser_direct_safe_auto": bool(browser_direct_safe_auto),
                    "lossless_display_derivative": bool(lossless_display),
                }
                return buffer.getvalue(), media_type, int(width), int(height), info
            finally:
                image.close()

def _normalise_video_display_mode(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    aliases = {
        "native": "source",
        "raw": "source",
        "source/native": "source",
        "automatic": "auto",
    }
    text = aliases.get(text, text)
    if text in {"auto", "source", "rgb"}:
        return text
    match = re.fullmatch(r"channel:(\d+)", text)
    if match and int(match.group(1)) >= 1:
        return f"channel:{int(match.group(1))}"
    raise ValueError(
        "Video display mode must be 'auto', 'source', 'rgb', or 'channel:N'."
    )

def get_video_resolver(context: Any) -> VideoAssetResolver:
    services = getattr(context, "services", None)
    if services is not None:
        try:
            resolver = services.get(VideoAssetResolver.service_key)
        except Exception:
            resolver = None
        if resolver is not None:
            return resolver
    return VideoAssetResolver(context)

def _merge_video_metadata(primary: VideoMetadata, secondary: VideoMetadata, *, source: str) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=(
            primary.duration_seconds
            if primary.duration_seconds is not None
            else secondary.duration_seconds
        ),
        fps=primary.fps if primary.fps is not None else secondary.fps,
        frame_count=(
            primary.frame_count if primary.frame_count is not None else secondary.frame_count
        ),
        width=primary.width if primary.width is not None else secondary.width,
        height=primary.height if primary.height is not None else secondary.height,
        thumbnail_uri=(
            primary.thumbnail_uri
            if primary.thumbnail_uri is not None
            else secondary.thumbnail_uri
        ),
        source=source,
        extra={**dict(secondary.extra or {}), **dict(primary.extra or {})},
    )

def _unsatisfied_range_total(headers: Any) -> Optional[int]:
    """Parse ``Content-Range: bytes */N`` from an HTTP 416 response."""

    if headers is None:
        return None
    try:
        value = headers.get("Content-Range")
    except Exception:
        value = None
    match = re.fullmatch(r"\s*bytes\s+\*/(\d+)\s*", str(value or ""), flags=re.I)
    if match is None:
        return None
    try:
        total = int(match.group(1))
    except Exception:
        return None
    return total if total >= 0 else None

def _content_length(response: Any) -> Optional[int]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = None
    try:
        value = headers.get("Content-Length")
    except Exception:
        pass
    return _positive_int(value)

def _emit_progress(callback: Any, **payload: Any) -> None:
    if not callable(callback):
        return
    try:
        callback(dict(payload))
    except TypeError:
        try:
            callback(**payload)
        except Exception:
            pass
    except Exception:
        pass

def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    number = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{number:.2f} TiB"

def _clean_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}

def _positive_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if result > 0 else None

def _nonnegative_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if result >= 0 else None

def _positive_int(value: Any) -> Optional[int]:
    if _is_missing(value):
        return None
    try:
        result = int(value)
    except Exception:
        try:
            result = int(float(value))
        except Exception:
            return None
    return result if result > 0 else None

def _remote_cache_lock(final_path: Path) -> threading.Lock:
    digest = hashlib.sha256(str(final_path).encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(_REMOTE_CACHE_LOCKS)
    return _REMOTE_CACHE_LOCKS[index]

def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return
    method = getattr(cancel_token, "raise_if_cancelled", None)
    if callable(method):
        method()
        return
    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled) and cancelled():
        raise RuntimeError("Video media operation cancelled.")
    if isinstance(cancelled, bool) and cancelled:
        raise RuntimeError("Video media operation cancelled.")
