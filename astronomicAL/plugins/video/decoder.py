from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from .contracts import DecodedVideoFrame, VideoMetadata


class PyAVVideoDecoder:
    """Optional PyAV-backed metadata and bounded frame decoder.

    PyAV is imported only when decoding is requested so ``core.video`` can be
    enabled in lightweight AstronomicAL installations. Callers can inspect
    mapped video metadata and media references without the decoder installed.
    """

    def available(self) -> bool:
        try:
            import av  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    def probe(self, path: str | Path, *, cancel_token: Any = None) -> VideoMetadata:
        _check_cancelled(cancel_token)
        av = self._require_av()
        path = str(Path(path).expanduser())

        with av.open(path) as container:
            stream = self._video_stream(container)
            fps = _stream_fps(stream)
            frame_count = _positive_int(getattr(stream, "frames", None))
            width = _positive_int(getattr(stream, "width", None))
            height = _positive_int(getattr(stream, "height", None))
            duration = _stream_duration_seconds(stream)
            if duration is None:
                duration = _container_duration_seconds(container, av)
            if frame_count is None and duration is not None and fps is not None:
                frame_count = max(1, int(round(float(duration) * float(fps))))

            return VideoMetadata(
                duration_seconds=duration,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                source="decoder_probe",
            )

    def extract_frame(
        self,
        path: str | Path,
        *,
        frame_index: int,
        fps_hint: Optional[float] = None,
        frame_count_hint: Optional[int] = None,
        max_size: int = 1280,
        image_format: str = "JPEG",
        cancel_token: Any = None,
    ) -> DecodedVideoFrame:
        """Decode a single bounded preview, seeking rather than decoding a video eagerly."""

        _check_cancelled(cancel_token)
        av = self._require_av()
        requested = max(0, int(frame_index or 0))
        max_size = max(64, int(max_size or 1280))
        save_format = "JPEG" if str(image_format).upper() in {"JPG", "JPEG"} else str(image_format).upper()

        with av.open(str(Path(path).expanduser())) as container:
            stream = self._video_stream(container)
            fps = _positive_float(fps_hint) or _stream_fps(stream)
            frame_count = _positive_int(frame_count_hint) or _positive_int(getattr(stream, "frames", None))
            if frame_count is not None:
                requested = min(requested, max(0, frame_count - 1))

            target_time = None if fps is None else float(requested) / float(fps)
            if target_time is not None and target_time > 0:
                self._seek_near_time(container, stream, target_time)

            chosen = None
            chosen_time = None
            chosen_index = requested
            decoded_after_seek = 0
            linear_target = requested if target_time is None else None
            half_interval = None if fps is None else 0.5 / float(fps)

            for frame in container.decode(stream):
                _check_cancelled(cancel_token)
                decoded_after_seek += 1
                frame_time = _frame_time_seconds(frame, stream)

                if target_time is None:
                    if decoded_after_seek - 1 < int(linear_target or 0):
                        if decoded_after_seek > 5000:
                            raise RuntimeError(
                                "Video frame lookup requires FPS/timestamp metadata for deep random access."
                            )
                        continue
                    chosen = frame
                    chosen_time = frame_time
                    chosen_index = requested
                    break

                chosen = frame
                chosen_time = frame_time
                if frame_time is not None and fps is not None:
                    chosen_index = max(0, int(round(frame_time * fps)))
                    if frame_time + float(half_interval or 0.0) >= target_time:
                        break
                elif decoded_after_seek >= 1:
                    # A decoder without timestamps has already honored the seek.
                    chosen_index = requested
                    break

                if decoded_after_seek >= 600:
                    # Do not turn one preview request into an unbounded decode.
                    break

            if chosen is None:
                raise RuntimeError(f"Could not decode video frame {requested}.")

            image = chosen.to_image()
            try:
                if save_format == "JPEG" and getattr(image, "mode", "") not in {"RGB", "L"}:
                    image = image.convert("RGB")
                image.thumbnail((max_size, max_size))
                width, height = image.size
                buffer = BytesIO()
                image.save(buffer, format=save_format)
            finally:
                close = getattr(image, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

            media_type = "image/jpeg" if save_format == "JPEG" else f"image/{save_format.lower()}"
            return DecodedVideoFrame(
                image_bytes=buffer.getvalue(),
                media_type=media_type,
                width=int(width),
                height=int(height),
                requested_frame_index=requested,
                actual_frame_index=int(chosen_index),
                time_seconds=chosen_time if chosen_time is not None else target_time,
                fps=fps,
                frame_count=frame_count,
            )

    @staticmethod
    def _video_stream(container: Any) -> Any:
        streams = getattr(container, "streams", None)
        videos = list(getattr(streams, "video", ()) or ())
        if not videos:
            raise RuntimeError("The selected media does not contain a video stream.")
        return videos[0]

    @staticmethod
    def _seek_near_time(container: Any, stream: Any, target_time: float) -> None:
        time_base = _positive_float(getattr(stream, "time_base", None))
        if time_base is None:
            return
        # Backward keyframe seek. The subsequent bounded decode advances to the
        # first frame at/near the requested presentation time.
        try:
            timestamp = max(0, int(float(target_time) / float(time_base)))
            container.seek(timestamp, stream=stream, backward=True, any_frame=False)
        except Exception:
            return

    @staticmethod
    def _require_av() -> Any:
        try:
            import av
        except Exception as exc:
            raise RuntimeError(
                "Video frame decoding requires the optional dependency `av>=12`. "
                "Install PyAV in the AstronomicAL environment to enable frame previews."
            ) from exc
        try:
            from PIL import Image  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Video frame preview encoding requires Pillow (`pillow>=10`)."
            ) from exc
        return av


def _stream_fps(stream: Any) -> Optional[float]:
    for name in ("average_rate", "base_rate", "guessed_rate"):
        value = _positive_float(getattr(stream, name, None))
        if value is not None:
            return value
    return None


def _stream_duration_seconds(stream: Any) -> Optional[float]:
    duration = getattr(stream, "duration", None)
    time_base = _positive_float(getattr(stream, "time_base", None))
    if duration is None or time_base is None:
        return None
    try:
        value = float(duration) * float(time_base)
    except Exception:
        return None
    return value if value >= 0 else None


def _container_duration_seconds(container: Any, av_module: Any) -> Optional[float]:
    duration = getattr(container, "duration", None)
    if duration is None:
        return None
    scale = getattr(av_module, "time_base", None)
    try:
        scale_value = float(scale)
    except Exception:
        scale_value = 1_000_000.0
    if scale_value <= 0:
        return None
    try:
        return max(0.0, float(duration) / scale_value)
    except Exception:
        return None


def _frame_time_seconds(frame: Any, stream: Any) -> Optional[float]:
    value = getattr(frame, "time", None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    pts = getattr(frame, "pts", None)
    time_base = _positive_float(getattr(stream, "time_base", None))
    if pts is None or time_base is None:
        return None
    try:
        return float(pts) * float(time_base)
    except Exception:
        return None


def _positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if result <= 0:
        return None
    return result


def _positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except Exception:
        return None
    if result <= 0:
        return None
    return result


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
