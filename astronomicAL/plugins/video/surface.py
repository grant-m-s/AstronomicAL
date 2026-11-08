from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import html
import re
import string
import urllib.parse
from typing import Any, Callable, Mapping, Optional

import panel as pn
from bokeh.models import ColumnDataSource, CustomJS, Div

from .contracts import VIDEO_MEDIA_KIND_FRAME_SEQUENCE, VideoAssetRef

FRAME_TOKEN = "__ASTRONOMICAL_FRAME_INDEX__"
_TRANSPARENT_PIXEL = (
    "data:image/gif;base64,"
    "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
)

# Bokeh ``Div`` renders its markup inside an internal ``.bk-clearfix`` element
# whose default inline style is ``display:inline-block`` and whose auto width is
# ``fit-content``.  Inline HTML styles on the nested frame cannot reliably
# override that host sizing boundary.  Apply the sizing contract through the
# model's shadow-root stylesheet instead so the internal markup element really
# fills the Bokeh layout box.
_VIDEO_SURFACE_STYLESHEET = r"""
:host {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  height: auto !important;
  overflow: visible !important;
  background: transparent !important;
  box-sizing: border-box !important;
}
.bk-clearfix {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  line-height: 0 !important;
  box-sizing: border-box !important;
}
.astronomical-video-frame-shell {
  display: grid !important;
  align-items: start !important;
  justify-items: center !important;
  width: 100% !important;
  max-width: 2560px !important;
  min-width: 0 !important;
  height: auto !important;
  margin: 0 auto !important;
  padding: 0 !important;
  overflow: visible !important;
  background: transparent !important;
  line-height: 0 !important;
  box-sizing: border-box !important;
  opacity: 1 !important;
  filter: none !important;
  mix-blend-mode: normal !important;
  isolation: isolate !important;
}
.astronomical-video-frame,
.astronomical-video-canvas,
.astronomical-video-overlay {
  grid-area: 1 / 1 !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  height: auto !important;
  margin: 0 auto !important;
  padding: 0 !important;
  border: 0 !important;
  object-fit: contain !important;
  object-position: center center !important;
  image-rendering: auto !important;
  backface-visibility: hidden !important;
  filter: none !important;
  mix-blend-mode: normal !important;
}
.astronomical-video-frame,
.astronomical-video-canvas {
  display: block !important;
}
.astronomical-video-frame {
  opacity: 1 !important;
  z-index: 0 !important;
}
.astronomical-video-canvas {
  z-index: 1 !important;
}
.astronomical-video-overlay {
  /* Direct overlays sit above the canonical base canvas, but their visibility is
     still owned entirely by the browser presenter. Do not make display !important:
     that would defeat element.style.display='none' when a frame has no overlay. */
  pointer-events: none !important;
  z-index: 2 !important;
}
.astronomical-video-message {
  width: 100% !important;
  min-height: 32px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: #111 !important;
  color: #c9d1d9 !important;
  box-sizing: border-box !important;
  padding: 12px !important;
  text-align: center !important;
  line-height: 1.35 !important;
}
"""

@dataclass(frozen=True)
class BrowserFrameSequencePattern:
    """Browser-safe representation of a numeric frame URI template.

    Regular sequences use ``uri_pattern`` with ``FRAME_TOKEN`` plus offset/step.
    Irregular sequences use ``frame_uris`` as an exact ordered logical-frame URI list.
    Both representations keep the browser clock in contiguous annotation-frame identity.
    """

    uri_pattern: str = ""
    frame_uris: tuple[str, ...] = ()
    frame_digits: int = 0
    frame_index_offset: int = 0
    frame_index_step: int = 1
    display_transform: str = "source"
    frame_token: str = FRAME_TOKEN

    def uri_for(self, frame_index: int) -> str:
        logical = max(0, int(frame_index or 0))
        if self.frame_uris:
            if logical >= len(self.frame_uris):
                raise IndexError(
                    f"Logical frame {logical} is outside explicit sequence length "
                    f"{len(self.frame_uris)}."
                )
            return str(self.frame_uris[logical])
        step = max(1, int(self.frame_index_step or 1))
        physical = max(0, int(self.frame_index_offset or 0) + logical * step)
        text = str(physical)
        if self.frame_digits > 0:
            text = text.zfill(int(self.frame_digits))
        return self.uri_pattern.replace(self.frame_token, text)

def browser_frame_sequence_pattern(asset: VideoAssetRef) -> Optional[BrowserFrameSequencePattern]:
    """Return a conservative browser-direct pattern for HTTP(S) frame sequences.

    The renderer intentionally understands only the small, explicit formatting
    vocabulary already supported by ``core.video``. Unsupported or ambiguous
    templates fail closed and stay on the managed resolver/JobManager path.
    """

    if str(getattr(asset, "media_kind", "") or "") != VIDEO_MEDIA_KIND_FRAME_SEQUENCE:
        return None

    explicit = tuple(
        str(value)
        for value in (getattr(asset, "frame_uris", ()) or ())
        if str(value).strip()
    )
    if explicit:
        for uri in explicit:
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
                return None
        return BrowserFrameSequencePattern(frame_uris=explicit)

    template = str(getattr(asset, "frame_uri_template", "") or "").strip()
    if not template:
        return None

    context = {
        "video_uri": str(getattr(asset, "uri", "") or ""),
        "row_id": str(getattr(asset, "row_id", "") or ""),
        "dataset_id": str(getattr(asset, "dataset_id", "") or ""),
    }
    parts: list[str] = []
    digits: Optional[int] = None
    frame_fields = 0

    try:
        parsed_fields = list(string.Formatter().parse(template))
    except ValueError:
        return None

    for literal, field_name, format_spec, conversion in parsed_fields:
        parts.append(literal)
        if field_name is None:
            continue
        field_name = str(field_name)
        format_spec = str(format_spec or "")
        conversion = str(conversion or "")

        if field_name in {"frame_index", "index"}:
            if conversion:
                return None
            if format_spec in {"", "d"}:
                field_digits = 0
            else:
                match = re.fullmatch(r"0(\d+)d", format_spec)
                if match is None:
                    return None
                field_digits = int(match.group(1))
            if digits is None:
                digits = field_digits
            elif digits != field_digits:
                return None
            frame_fields += 1
            parts.append(FRAME_TOKEN)
            continue

        if field_name not in context:
            return None
        value: Any = context[field_name]
        if conversion == "r":
            value = repr(value)
        elif conversion == "s" or not conversion:
            value = str(value)
        else:
            return None
        try:
            parts.append(format(value, format_spec) if format_spec else str(value))
        except Exception:
            return None

    if frame_fields <= 0:
        return None

    pattern = "".join(parts)
    probe_uri = pattern.replace(FRAME_TOKEN, "0")
    parsed = urllib.parse.urlparse(probe_uri)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None

    return BrowserFrameSequencePattern(
        uri_pattern=pattern,
        frame_digits=int(digits or 0),
        frame_index_offset=int(getattr(asset, "frame_index_offset", 0) or 0),
        frame_index_step=max(1, int(getattr(asset, "frame_index_step", 1) or 1)),
    )

@dataclass(frozen=True)
class SurfaceEvent:
    name: str
    old: Any
    new: Any

def _image_html(
    source: str,
    width: int,
    height: int,
    *,
    previous_source: str = "",
    overlay_source: str = "",
    overlay_opacity: float = 0.45,
    control_id: str = "",
    frame_index: int = 0,
) -> str:
    """Return a responsive double-buffered image shell.

    Managed playback used to replace the only visible ``<img>`` before the next data
    URI had decoded, exposing the shell's black background for a frame. Keeping the
    previous decoded image underneath the incoming image removes that black flash while
    preserving the same standard-Bokeh/HTML presentation contract.
    """

    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    escaped = html.escape(str(source or _TRANSPARENT_PIXEL), quote=True)
    previous = str(previous_source or "").strip()
    layers: list[str] = []
    if previous and previous != str(source or ""):
        layers.append(
            f'<img class="astronomical-video-frame" src="{html.escape(previous, quote=True)}" '
            f'width="{safe_width}" height="{safe_height}" loading="eager" decoding="async" '
            'draggable="false" aria-hidden="true" />'
        )
    layers.append(
        f'<img class="astronomical-video-frame" data-astronomical-current-frame="1" src="{escaped}" '
        f'width="{safe_width}" height="{safe_height}" loading="eager" decoding="async" '
        'draggable="false" />'
    )
    overlay = str(overlay_source or "").strip()
    if overlay:
        layers.append(
            f'<img class="astronomical-video-overlay" src="{html.escape(overlay, quote=True)}" '
            f'width="{safe_width}" height="{safe_height}" loading="eager" decoding="async" '
            f'draggable="false" aria-hidden="true" style="opacity:{min(1.0, max(0.0, float(overlay_opacity or 0.45))):.3f};pointer-events:none;" />'
        )
    shell_id = f'astronomical-video-managed-{_surface_id_suffix(control_id)}'
    return (
        f'<div id="{html.escape(shell_id, quote=True)}" '
        f'class="astronomical-video-frame-shell" data-astronomical-frame-index="{max(0, int(frame_index))}">'
        + "".join(layers) + "</div>"
    )

def _surface_id_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "surface")) or "surface"

def _surface_overlay_id(control_id: str) -> str:
    return f"astronomical-video-overlay-{_surface_id_suffix(control_id)}"

def _surface_canvas_id(control_id: str) -> str:
    return f"astronomical-video-canvas-{_surface_id_suffix(control_id)}"

def _direct_surface_html(
    fallback_source: str,
    width: int,
    height: int,
    *,
    control_id: str,
) -> str:
    """Return the persistent direct-playback canvas over the last managed frame.

    The canvas starts transparent. Browser playback decodes/draws its first frame off
    the visible hot path and reveals the canvas only after that draw succeeds, so
    promotion from the validated managed preview cannot flash black.
    """

    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    fallback = html.escape(str(fallback_source or _TRANSPARENT_PIXEL), quote=True)
    canvas_id = html.escape(_surface_canvas_id(control_id), quote=True)
    overlay_id = html.escape(_surface_overlay_id(control_id), quote=True)
    return (
        '<div class="astronomical-video-frame-shell">'
        f'<img class="astronomical-video-frame" src="{fallback}" '
        f'width="{safe_width}" height="{safe_height}" loading="eager" decoding="async" '
        'draggable="false" aria-hidden="true" />'
        f'<canvas id="{canvas_id}" class="astronomical-video-canvas" '
        f'width="{safe_width}" height="{safe_height}" '
        'style="opacity:0!important;transition:none;pointer-events:none;"></canvas>'
        f'<canvas id="{overlay_id}" class="astronomical-video-overlay" '
        f'width="{safe_width}" height="{safe_height}" '
        'aria-hidden="true" style="display:none;opacity:0.45;pointer-events:none;"></canvas>'
        '</div>'
    )

def _message_html(message: str) -> str:
    text = html.escape(str(message or ""))
    return f'<div class="astronomical-video-message">{text}</div>'

class VideoFrameSurface:
    """Persistent responsive frame surface using only pre-registered Bokeh models.

    The visible media is a standard Bokeh ``Div``. Managed frames use ordinary
    browser ``<img>`` elements; browser-buffered frame sequences promote to one
    persistent base HTML ``<canvas>`` plus a separate presentation-only overlay ``<canvas>`` inside that same Div. This deliberately avoids a
    Bokeh plotting canvas for raster frames:
    draggable workspace grids may constrain plot width and height independently,
    which can distort ``ImageURL`` glyphs even when match-aspect hints are set.

    The Bokeh model itself uses ``scale_width`` plus the mapped/native frame
    ``aspect_ratio``.  Width therefore drives height: narrow workspace cells make
    the frame genuinely smaller, while wide cells make it proportionally taller.
    The viewer column may scroll vertically if the dashboard cell is shorter than
    that natural aspect-derived height.  The shadow-root stylesheet also overrides
    MarkupView's internal fit-content element, so the nested ``<img>`` receives the
    full model box and ``object-fit: contain`` can do its job reliably.

    Direct HTTP(S) frame-sequence playback uses a browser-side buffering-aware
    ``requestAnimationFrame`` clock, a decoded sliding window, and a rate-aware rolling
    network prefetch window. The Div markup is installed once and decoded frames are
    painted into the persistent canvas, so frame changes do not rebuild the DOM or
    expose the black shell while the next image is decoding. Scientific RGBA/RGBD
    presentation uses an unpremultiplied ImageBitmap + WebGL path rather than CSS/SVG
    colour filters. No custom Bokeh model is introduced.
    """

    def __init__(
        self,
        *,
        sizing_mode: str = "stretch_width",
        min_height: int = 320,
        margin: Any = (0, 0, 8, 0),
        prefetch_count: int = 6,
        message: str = "No video selected",
    ) -> None:
        del sizing_mode
        min_height = max(24, int(min_height or 24))
        self._state: dict[str, Any] = {
            "mode": "server",
            "uri_pattern": "",
            "frame_uris": [],
            "frame_token": FRAME_TOKEN,
            "frame_digits": 0,
            "frame_index_offset": 0,
            "frame_index_step": 1,
            "display_transform": "source",
            "dataset_id": "",
            "row_id": "",
            "frame_index": 0,
            "frame_count": 0,
            "fps": 0.0,
            "playback_rate": 1.0,
            "playing": False,
            "client_clock": False,
            "prefetch_count": max(0, min(128, int(prefetch_count or 0))),
            "decoded_lookahead": max(256, min(1024, int(prefetch_count or 0) * 8)),
            "generation": 0,
            "control_revision": 0,
            "bootstrap_tick": 0,
            "prepare_serial": 0,
            "message": str(message or ""),
            "native_width": 960,
            "native_height": 540,
        }
        # Overlay presentation is deliberately transported independently from playback
        # controls. Updating a mask/detection overlay must never re-run the browser
        # playback state machine, re-anchor its clock, or revive a paused surface.
        self._overlay_state: dict[str, Any] = {
            "overlay_frames": {},
            "overlay_opacity": 0.45,
            "overlay_generation": 0,
        }
        self._source = ""
        self._error = ""
        self._error_frame = -1
        self._source_info: dict[str, Any] = {}
        self._listeners: dict[str, dict[int, Callable[[SurfaceEvent], None]]] = {
            "frame_index": {},
            "playing": {},
            "error": {},
            "image_info": {},
            "cache_prepare": {},
        }
        self._listener_seq = 0
        self._first_frame_watchdog: Any = None
        self._first_frame_watchdog_started = 0.0
        self._first_frame_watchdog_pulses = 0
        self._first_frame_painted = False

        self._control_source = ColumnDataSource(data=self._control_data())
        self._overlay_source = ColumnDataSource(data=self._overlay_data())
        # Browser-side stage timings are transported separately from frame/temporal
        # feedback: telemetry must never advance playback or alter record focus.
        self._timing_source = ColumnDataSource(data={
            "stage": [""], "dataset_id": [""], "row_id": [""],
            "frame_index": [-1], "elapsed_ms": [0.0], "detail": [""],
        })
        self._timing_source.on_change("data", self._on_timing_data)
        self._feedback_source = ColumnDataSource(
            data={
                "frame_index": [0],
                "playing": [False],
                "control_revision": [0],
                "error": [""],
                "error_frame": [-1],
                "source_image_mode": [""],
                "source_channel_count": [0],
                "painted": [False],
            }
        )
        self._display = Div(
            text=_message_html(message),
            sizing_mode="stretch_width",
            min_height=min_height,
            max_width=2560,
            stylesheets=[_VIDEO_SURFACE_STYLESHEET],
            styles={
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "height": "auto",
                "overflow": "visible",
                "background": "transparent",
                "box-sizing": "border-box",
                "display": "block",
            },
        )
        # Keep transport models in the ordinary Bokeh reference graph without
        # storing live models in ``tags`` (which can produce duplicate-reference
        # warnings when dynamic plugin panels patch the document).
        self._display.js_on_change(
            "visible",
            CustomJS(
                args={
                    "control": self._control_source,
                    "overlay": self._overlay_source,
                    "feedback": self._feedback_source,
                    "metrics": self._timing_source,
                },
                code="void control; void overlay; void feedback; void metrics;",
            ),
        )
        self._control_source.js_on_change(
            "data",
            CustomJS(
                args={
                    "control": self._control_source,
                    "overlay": self._overlay_source,
                    "display": self._display,
                    "feedback": self._feedback_source,
                    "metrics": self._timing_source,
                },
                code=_DIRECT_SEQUENCE_JS,
            ),
        )
        # Encoded videos are decoded server-side into ordinary <img> frames.
        # Publish the same browser-local presentation signal as direct frame sequences
        # so provider-neutral tools can bind to the actual visible image shell.
        managed_presentation = CustomJS(
            args={"control": self._control_source, "display": self._display},
            code=_MANAGED_FRAME_PRESENTATION_JS,
        )
        self._display.js_on_change("text", managed_presentation)
        self._control_source.js_on_change("data", managed_presentation)
        self._overlay_source.js_on_change(
            "data",
            CustomJS(
                args={
                    "control": self._control_source,
                    "overlay": self._overlay_source,
                    "display": self._display,
                },
                code=_DIRECT_OVERLAY_JS,
            ),
        )
        self._feedback_source.on_change("data", self._on_feedback_data)

        self._bokeh_pane = pn.pane.Bokeh(
            self._display,
            sizing_mode="stretch_width",
            margin=0,
            max_width=2560,
            align="center",
            styles={
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "height": "auto",
                "overflow": "visible",
                "box-sizing": "border-box",
            },
        )
        self.view = pn.Row(
            self._bokeh_pane,
            sizing_mode="stretch_width",
            min_height=min_height,
            margin=margin,
            align="center",
            styles={
                "width": "100%",
                "max-width": "100%",
                "min-width": "0",
                "height": "auto",
                "overflow": "visible",
                "box-sizing": "border-box",
                "justify-content": "center",
                "align-items": "flex-start",
            },
        )

    # ------------------------------------------------------------------ listeners
    def on_frame_changed(self, callback: Callable[[SurfaceEvent], None]) -> tuple[str, int]:
        return self._add_listener("frame_index", callback)

    def on_playing_changed(self, callback: Callable[[SurfaceEvent], None]) -> tuple[str, int]:
        return self._add_listener("playing", callback)

    def on_error(self, callback: Callable[[SurfaceEvent], None]) -> tuple[str, int]:
        return self._add_listener("error", callback)

    def on_image_info_changed(self, callback: Callable[[SurfaceEvent], None]) -> tuple[str, int]:
        return self._add_listener("image_info", callback)

    def remove_listener(self, token: tuple[str, int]) -> None:
        try:
            name, listener_id = token
        except Exception:
            return
        self._listeners.get(str(name), {}).pop(int(listener_id), None)

    def _add_listener(
        self,
        name: str,
        callback: Callable[[SurfaceEvent], None],
    ) -> tuple[str, int]:
        self._listener_seq += 1
        token = (name, self._listener_seq)
        self._listeners[name][self._listener_seq] = callback
        return token

    def _notify(self, name: str, old: Any, new: Any) -> None:
        if old == new:
            return
        event = SurfaceEvent(name=name, old=old, new=new)
        for callback in list(self._listeners.get(name, {}).values()):
            try:
                callback(event)
            except Exception:
                pass

    # ------------------------------------------------------------------ public state
    @property
    def mode(self) -> str:
        return str(self._state["mode"])

    @mode.setter
    def mode(self, value: str) -> None:
        value = str(value or "server")
        if value not in {"server", "direct_sequence"}:
            value = "server"
        self._set_control(mode=value)

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        self._source = str(value or "")
        if self.mode == "server":
            self._render_server_source()

    @property
    def uri_pattern(self) -> str:
        return str(self._state["uri_pattern"])

    @uri_pattern.setter
    def uri_pattern(self, value: str) -> None:
        self._set_control(uri_pattern=str(value or ""))

    @property
    def frame_uris(self) -> tuple[str, ...]:
        value = self._state.get("frame_uris") or []
        return tuple(str(item) for item in value if str(item).strip())

    @frame_uris.setter
    def frame_uris(self, value: Any) -> None:
        values = [str(item) for item in (value or []) if str(item).strip()]
        self._set_control(frame_uris=values)

    @property
    def frame_token(self) -> str:
        return str(self._state["frame_token"])

    @frame_token.setter
    def frame_token(self, value: str) -> None:
        self._set_control(frame_token=str(value or FRAME_TOKEN))

    @property
    def frame_digits(self) -> int:
        return int(self._state["frame_digits"])

    @frame_digits.setter
    def frame_digits(self, value: int) -> None:
        self._set_control(frame_digits=max(0, min(16, int(value or 0))))

    @property
    def frame_index_offset(self) -> int:
        return int(self._state["frame_index_offset"])

    @frame_index_offset.setter
    def frame_index_offset(self, value: int) -> None:
        self._set_control(frame_index_offset=int(value or 0))

    @property
    def frame_index_step(self) -> int:
        return int(self._state["frame_index_step"])

    @frame_index_step.setter
    def frame_index_step(self, value: int) -> None:
        self._set_control(frame_index_step=max(1, int(value or 1)))

    @property
    def frame_index(self) -> int:
        return int(self._state["frame_index"])

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        self._set_control(frame_index=max(0, int(value or 0)))

    @property
    def frame_count(self) -> int:
        return int(self._state["frame_count"])

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        self._set_control(frame_count=max(0, int(value or 0)))

    @property
    def fps(self) -> float:
        return float(self._state["fps"])

    @fps.setter
    def fps(self, value: float) -> None:
        self._set_control(fps=max(0.0, float(value or 0.0)))

    @property
    def playback_rate(self) -> float:
        return float(self._state["playback_rate"])

    @playback_rate.setter
    def playback_rate(self, value: float) -> None:
        self._set_control(playback_rate=max(0.01, float(value or 1.0)))

    @property
    def playing(self) -> bool:
        return bool(self._state["playing"])

    @playing.setter
    def playing(self, value: bool) -> None:
        self._set_control(playing=bool(value))

    @property
    def client_clock(self) -> bool:
        return bool(self._state["client_clock"])

    @client_clock.setter
    def client_clock(self, value: bool) -> None:
        self._set_control(client_clock=bool(value))

    @property
    def prefetch_count(self) -> int:
        return int(self._state["prefetch_count"])

    @prefetch_count.setter
    def prefetch_count(self, value: int) -> None:
        configured = max(0, min(128, int(value or 0)))
        self._set_control(
            prefetch_count=configured,
            decoded_lookahead=max(256, min(1024, configured * 8)),
        )

    @property
    def decoded_lookahead(self) -> int:
        """Internal decoded-frame window used by browser playback.

        This is intentionally separate from ``prefetch_count``. The latter is
        caller configuration and remains stable for panel state/API compatibility;
        decoded look-ahead may grow independently to keep higher-rate playback
        smooth without silently rewriting the caller's setting.
        """
        return int(self._state["decoded_lookahead"])

    @property
    def generation(self) -> int:
        return int(self._state["generation"])

    @generation.setter
    def generation(self, value: int) -> None:
        self._set_control(generation=max(0, int(value or 0)))

    @property
    def message(self) -> str:
        return str(self._state["message"])

    @message.setter
    def message(self, value: str) -> None:
        text = str(value or "")
        self._set_control(message=text)
        if self.mode == "server" and not self._source:
            self._display.text = _message_html(text)

    @property
    def error(self) -> str:
        return self._error

    @error.setter
    def error(self, value: str) -> None:
        old = self._error
        self._error = str(value or "")
        self._notify("error", old, self._error)

    @property
    def error_frame(self) -> int:
        return int(self._error_frame)

    @error_frame.setter
    def error_frame(self, value: int) -> None:
        self._error_frame = int(value)

    # ------------------------------------------------------------------ rendering API
    def reset(self, message: str = "No video selected") -> None:
        self._stop_first_frame_watchdog()
        self._source = ""
        self._source_info = {}
        self._error = ""
        self._error_frame = -1
        next_generation = self.generation + 1
        text = str(message or "")
        self._display.text = _message_html(text)
        self._set_control(
            mode="server",
            dataset_id="",
            row_id="",
            uri_pattern="",
            frame_uris=[],
            frame_index=0,
            frame_count=0,
            fps=0.0,
            playing=False,
            client_clock=False,
            generation=next_generation,
            prepare_serial=0,
            message=text,
        )
        self._set_overlay_control(
            overlay_frames={},
            overlay_generation=int(self._overlay_state.get("overlay_generation") or 0) + 1,
        )
        self._set_feedback(frame_index=0, playing=False, error="", error_frame=-1)

    def show_server_frame(
        self,
        source: str,
        *,
        frame_index: int,
        dataset_id: str = "",
        row_id: str = "",
    ) -> None:
        self._stop_first_frame_watchdog()
        previous = self._source
        self._source = str(source or "")
        self._error = ""
        self._error_frame = -1
        self._display.text = _image_html(
            self._source or _TRANSPARENT_PIXEL,
            int(self._state["native_width"]),
            int(self._state["native_height"]),
            previous_source=previous,
            overlay_source=str(
                (self._overlay_state.get("overlay_frames") or {}).get(str(max(0, int(frame_index or 0))))
                or (self._overlay_state.get("overlay_frames") or {}).get(max(0, int(frame_index or 0)))
                or ""
            ),
            overlay_opacity=float(self._overlay_state.get("overlay_opacity") or 0.45),
            control_id=self._control_source.id,
            frame_index=max(0, int(frame_index or 0)),
        )
        self._set_control(
            mode="server",
            dataset_id=str(dataset_id or ""),
            row_id=str(row_id or ""),
            frame_index=max(0, int(frame_index or 0)),
            client_clock=False,
            message="",
        )
        self._set_feedback(
            frame_index=max(0, int(frame_index or 0)),
            playing=self.playing,
            error="",
            error_frame=-1,
        )

    def configure_direct_sequence(
        self,
        pattern: BrowserFrameSequencePattern,
        *,
        frame_index: int,
        frame_count: Optional[int],
        fps: Optional[float],
        playback_rate: float,
        playing: bool,
        prefetch_count: int,
        dataset_id: str = "",
        row_id: str = "",
    ) -> None:
        self._error = ""
        self._error_frame = -1
        first_source = self._source or _TRANSPARENT_PIXEL
        if pattern.display_transform == "source":
            try:
                first_source = pattern.uri_for(frame_index)
            except (ValueError, IndexError, KeyError):
                pass
        self._display.text = _direct_surface_html(
            first_source,
            int(self._state["native_width"]),
            int(self._state["native_height"]),
            control_id=self._control_source.id,
        )
        self._set_control(
            mode="direct_sequence",
            uri_pattern=str(pattern.uri_pattern),
            frame_uris=list(pattern.frame_uris),
            frame_token=str(pattern.frame_token),
            frame_digits=max(0, int(pattern.frame_digits or 0)),
            frame_index_offset=int(pattern.frame_index_offset or 0),
            frame_index_step=max(1, int(pattern.frame_index_step or 1)),
            display_transform=str(pattern.display_transform or "source"),
            dataset_id=str(dataset_id or ""),
            row_id=str(row_id or ""),
            frame_index=max(0, int(frame_index or 0)),
            frame_count=max(0, int(frame_count or 0)),
            fps=max(0.0, float(fps or 0.0)),
            playback_rate=max(0.01, float(playback_rate or 1.0)),
            playing=bool(playing),
            client_clock=True,
            prefetch_count=max(0, min(128, int(prefetch_count or 0))),
            decoded_lookahead=max(256, min(1024, int(prefetch_count or 0) * 8)),
            message="",
        )
        self._set_feedback(
            frame_index=max(0, int(frame_index or 0)),
            playing=bool(playing),
            error="",
            error_frame=-1,
        )
        self._start_first_frame_watchdog()

    def prepare_browser_cache(self) -> None:
        """Ask the browser to cache the next bounded frame window while paused.

        This changes only the preparation request marker, never the video frame,
        playback revision, current focus or overlay identity.
        """
        if self.mode != "direct_sequence" or self.playing:
            return
        self._state["prepare_serial"] = int(self._state.get("prepare_serial") or 0) + 1
        self._control_source.data = self._control_data()

    def on_cache_prepare(self, callback: Callable[[SurfaceEvent], None]) -> tuple[str, int]:
        return self._add_listener("cache_prepare", callback)

    def set_overlay_frames(self, frames: Mapping[int, str] | dict[int, str], *, opacity: float = 0.45) -> None:
        """Replace the resident overlay cache without touching playback controls.

        Browser-direct playback owns its requestAnimationFrame clock inside
        ``_DIRECT_SEQUENCE_JS``. Overlay providers are allowed to finish asynchronously,
        including after Pause, so overlay delivery has a separate ColumnDataSource and
        can never replay stale ``frame_index``/``playing`` control values.

        ``core.video`` callers normally use :meth:`merge_overlay_frames` for rolling
        prefetch. Full replacement remains the explicit source/layer invalidation path.
        """

        values = {
            str(max(0, int(frame))): str(uri)
            for frame, uri in dict(frames or {}).items()
            if str(uri or "")
        }
        self._set_overlay_control(
            overlay_frames=values,
            overlay_opacity=min(1.0, max(0.0, float(opacity or 0.45))),
            overlay_generation=int(self._overlay_state.get("overlay_generation") or 0) + 1,
        )
        if self.mode == "server" and self._source:
            self._render_server_source()

    def merge_overlay_frames(
        self,
        frames: Mapping[int, str] | dict[int, str],
        *,
        opacity: float | None = None,
    ) -> None:
        """Merge newly prepared overlay frames into the browser-resident cache.

        Rolling overlay preparation must never discard still-useful frames merely
        because a producer job extended the forward window.  This mirrors the former
        Segmentation Viewer resident replay cache while keeping ``core.video`` agnostic
        to the overlay semantics.
        """

        current = {
            str(max(0, int(frame))): str(uri)
            for frame, uri in dict(self._overlay_state.get("overlay_frames") or {}).items()
            if str(uri or "")
        }
        for frame, uri in dict(frames or {}).items():
            key = str(max(0, int(frame)))
            value = str(uri or "")
            if value:
                current[key] = value
            else:
                current.pop(key, None)
        selected_opacity = (
            float(self._overlay_state.get("overlay_opacity") or 0.45)
            if opacity is None
            else min(1.0, max(0.0, float(opacity)))
        )
        self._set_overlay_control(
            overlay_frames=current,
            overlay_opacity=selected_opacity,
            overlay_generation=int(self._overlay_state.get("overlay_generation") or 0) + 1,
        )
        if self.mode == "server" and self._source:
            self._render_server_source()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Apply overlay opacity immediately without re-rendering provider pixels."""

        selected = min(1.0, max(0.0, float(opacity)))
        self._set_overlay_control(
            overlay_opacity=selected,
            overlay_generation=int(self._overlay_state.get("overlay_generation") or 0) + 1,
        )
        if self.mode == "server" and self._source:
            self._render_server_source()

    def clear_overlay_frames(self) -> None:
        self.set_overlay_frames(
            {},
            opacity=float(self._overlay_state.get("overlay_opacity") or 0.45),
        )

    def set_dimensions(self, width: Optional[int], height: Optional[int]) -> None:
        try:
            safe_width = int(width or 0)
            safe_height = int(height or 0)
        except Exception:
            return
        if safe_width <= 0 or safe_height <= 0:
            return
        aspect_ratio = float(safe_width) / float(safe_height)
        self._display.aspect_ratio = aspect_ratio
        self._set_control(native_width=safe_width, native_height=safe_height)
        if self.mode == "server" and self._source:
            self._render_server_source()

    # ------------------------------------------------------------------ internals
    def _render_server_source(self) -> None:
        frame = max(0, int(self._state.get("frame_index") or 0))
        self._display.text = _image_html(
            self._source or _TRANSPARENT_PIXEL,
            int(self._state["native_width"]),
            int(self._state["native_height"]),
            overlay_source=str(
                (self._overlay_state.get("overlay_frames") or {}).get(str(frame))
                or (self._overlay_state.get("overlay_frames") or {}).get(frame)
                or ""
            ),
            overlay_opacity=float(self._overlay_state.get("overlay_opacity") or 0.45),
            control_id=self._control_source.id,
            frame_index=frame,
        )

    def _control_data(self) -> dict[str, list[Any]]:
        return {key: [value] for key, value in self._state.items()}

    def _overlay_data(self) -> dict[str, list[Any]]:
        return {key: [value] for key, value in self._overlay_state.items()}

    def _set_control(self, **changes: Any) -> None:
        changed = False
        for key, value in changes.items():
            if key not in self._state or key == "control_revision":
                continue
            if self._state[key] != value:
                self._state[key] = value
                changed = True
        if changed:
            # Every server-originated playback-control mutation starts a new revision.
            # Browser feedback echoes the revision that produced its presentation; a
            # late frame from an older paused seek can therefore never rewind current
            # temporal state after the user has already moved on.
            self._state["control_revision"] = int(self._state.get("control_revision") or 0) + 1
            self._control_source.data = self._control_data()

    def _set_overlay_control(self, **changes: Any) -> None:
        changed = False
        for key, value in changes.items():
            if key not in self._overlay_state:
                continue
            if self._overlay_state[key] != value:
                self._overlay_state[key] = value
                changed = True
        if changed:
            self._overlay_source.data = self._overlay_data()

    def _set_feedback(
        self,
        *,
        frame_index: int,
        playing: bool,
        error: str,
        error_frame: int,
    ) -> None:
        self._feedback_source.data = {
            "frame_index": [max(0, int(frame_index or 0))],
            "playing": [bool(playing)],
            "control_revision": [int(self._state.get("control_revision") or 0)],
            "error": [str(error or "")],
            "error_frame": [int(error_frame)],
            "source_image_mode": [""],
            "source_channel_count": [0],
            "painted": [False],
        }

    def _on_timing_data(self, _attr: str, _old: Any, new: Any) -> None:
        values = dict(new or {})
        def first(name: str, default: Any = "") -> Any:
            column = values.get(name) or [default]
            return column[0]
        stage = str(first("stage"))
        if stage in {"cache_prepare_progress", "cache_prepare_done", "cache_prepare_cancelled", "cache_prepare_unavailable"}:
            if (str(first("dataset_id")) != str(self._state.get("dataset_id") or "")
                    or str(first("row_id")) != str(self._state.get("row_id") or "")):
                return
            detail = str(first("detail"))
            counts = {}
            for piece in detail.split():
                key, separator, value = piece.partition("=")
                if separator and value.isdigit():
                    counts[key] = int(value)
            self._notify("cache_prepare", None, {"stage": stage, **counts})
        if stage:
            print(
                f"[AL_VIDEO_TIMING][browser] stage={stage!r} "
                f"dataset={first('dataset_id')!r} row={first('row_id')!r} "
                f"frame={first('frame_index', -1)} "
                f"elapsed_ms={float(first('elapsed_ms', 0.0)):.1f} "
                f"detail={str(first('detail'))[:720 if stage == 'playback_window' else 180]!r}", flush=True,
            )

    def _stop_first_frame_watchdog(self) -> None:
        callback = self._first_frame_watchdog
        self._first_frame_watchdog = None
        if callback is not None:
            try:
                callback.stop()
            except Exception:
                pass

    def _start_first_frame_watchdog(self) -> None:
        self._stop_first_frame_watchdog()
        self._first_frame_painted = False
        self._first_frame_watchdog_pulses = 0
        self._first_frame_watchdog_started = perf_counter()
        print(
            f"[AL_VIDEO_TIMING][server] stage=direct_configured "
            f"dataset={self._state.get('dataset_id')!r} "
            f"row={self._state.get('row_id')!r} "
            f"frame={self._state.get('frame_index')}", flush=True,
        )
        try:
            # The first control-data change may precede Bokeh/Panel mounting the
            # restored workspace. Periodically rebroadcast unchanged playback
            # controls until the browser confirms that a base frame was painted.
            # This is not a seek: the control revision and playback clock stay put.
            self._first_frame_watchdog = pn.state.add_periodic_callback(
                self._retry_first_frame, period=400, start=True,
            )
        except Exception as exc:
            print(f"[AL_VIDEO_TIMING][server] stage=watchdog_unavailable error={exc!s}", flush=True)

    def _retry_first_frame(self) -> None:
        if self._first_frame_painted or self.mode != "direct_sequence":
            self._stop_first_frame_watchdog()
            return
        age = perf_counter() - self._first_frame_watchdog_started
        if age > 30:
            print(
                f"[AL_VIDEO_TIMING][server] stage=first_frame_watchdog_timeout "
                f"elapsed_ms={age * 1000:.1f} "
                f"retry_pulses={self._first_frame_watchdog_pulses}", flush=True,
            )
            self._stop_first_frame_watchdog()
            previous_error = self._error
            self._error = "Browser did not acknowledge a displayed first frame within 30 seconds."
            self._error_frame = int(self._state.get("frame_index") or 0)
            self._notify("error", previous_error, self._error)
            return
        self._first_frame_watchdog_pulses += 1
        self._state["bootstrap_tick"] = self._first_frame_watchdog_pulses
        # Changing only the bootstrap counter causes a post-mount CustomJS
        # callback without invalidating the current frame/clock revision.
        self._control_source.data = self._control_data()
        if self._first_frame_watchdog_pulses in {1, 5, 15, 35}:
            print(
                f"[AL_VIDEO_TIMING][server] stage=bootstrap_retry "
                f"elapsed_ms={age * 1000:.1f} "
                f"attempt={self._first_frame_watchdog_pulses}", flush=True,
            )

    def dispose(self) -> None:
        self._stop_first_frame_watchdog()

    def _on_feedback_data(self, _attr: str, _old: Any, new: Any) -> None:
        data = dict(new or {})
        current_revision = int(self._state.get("control_revision") or 0)
        try:
            feedback_revision = int((data.get("control_revision") or [current_revision])[0])
        except Exception:
            feedback_revision = current_revision
        if feedback_revision != current_revision:
            # The browser may finish decoding a frame from an older seek after a newer
            # slider/Next/Previous command has already been accepted by Python. Keep
            # the server-side transport revision authoritative and ignore that stale
            # presentation feedback instead of rubber-banding shared temporal focus.
            return

        if bool((data.get("painted") or [False])[0]):
            if not self._first_frame_painted:
                elapsed_ms = (perf_counter() - self._first_frame_watchdog_started) * 1000
                print(
                    f"[AL_VIDEO_TIMING][server] stage=first_frame_ack "
                    f"dataset={self._state.get('dataset_id')!r} "
                    f"row={self._state.get('row_id')!r} elapsed_ms={elapsed_ms:.1f} "
                    f"retry_pulses={self._first_frame_watchdog_pulses}", flush=True,
                )
            self._first_frame_painted = True
            self._stop_first_frame_watchdog()
        old_frame = int(self._state["frame_index"])
        old_playing = bool(self._state["playing"])
        old_error = self._error

        try:
            frame = max(0, int((data.get("frame_index") or [old_frame])[0]))
        except Exception:
            frame = old_frame
        try:
            playing = bool((data.get("playing") or [old_playing])[0])
        except Exception:
            playing = old_playing
        try:
            error = str((data.get("error") or [""])[0] or "")
        except Exception:
            error = ""
        try:
            error_frame = int((data.get("error_frame") or [-1])[0])
        except Exception:
            error_frame = -1

        mode = str((data.get("source_image_mode") or [""])[0] or "")
        try:
            count = int((data.get("source_channel_count") or [0])[0])
        except (TypeError, ValueError):
            count = 0
        if not error and (mode, count) in {("RGB", 3), ("RGBA", 4)}:
            info = {"source_image_mode": mode, "source_channel_count": count}
            if info != self._source_info:
                previous_info = self._source_info
                self._source_info = info
                self._notify("image_info", previous_info, info)

        self._state["frame_index"] = frame
        self._state["playing"] = playing
        self._error = error
        self._error_frame = error_frame

        self._notify("frame_index", old_frame, frame)
        self._notify("playing", old_playing, playing)
        self._notify("error", old_error, error)

_DIRECT_OVERLAY_JS = r"""
// Mask delivery is independent of the video clock. Only the base-video presenter
// knows which frame is actually on screen; never infer that from the slider or an
// uninitialised lastPresented=-1 after restoring a workspace.
const key = `astronomical-video-${control.id}`;
const registry = window.__astronomicalVideoSurfaces || Object.create(null);
const state = registry[key] || null;
const field = (name, fallback) => {
  const col = control.data[name];
  return col && col.length && col[0] != null ? col[0] : fallback;
};
const generation = Number(field('generation', 0));
const mode = String(field('mode', 'server'));
if (
  state && state.generation === generation && mode === 'direct_sequence' &&
  Number.isInteger(state.lastPresented) && state.lastPresented >= 0 &&
  state.canvas && state.canvas.isConnected &&
  typeof state.paintOverlay === 'function'
) {
  // Re-read the newest overlay mapping using the canonical browser frame index.
  state.paintOverlay(state.lastPresented);
  if (typeof state.prefetchOverlays === 'function') state.prefetchOverlays(state.lastPresented);
} else {
  // An overlay job can finish before the first browser frame, or after the surface
  // has been replaced. Keep the mask invisible until a matching base is painted.
  const canvas = state && state.overlayCanvas;
  if (canvas && canvas.isConnected) {
    canvas.dataset.overlaySerial = String(Number(canvas.dataset.overlaySerial || 0) + 1);
    canvas.dataset.overlayPendingUri = '';
    canvas.dataset.overlayPendingFrame = '';
    canvas.dataset.overlayFrame = '';
    canvas.style.display = 'none';
  }
}
"""

_MANAGED_FRAME_PRESENTATION_JS = r"""
// Encoded MP4/MOV videos are served as decoded image previews instead of the
// browser-direct frame-sequence canvas.  Publish the same browser presentation
// contract for the displayed image so other plugins do not need an MP4-specific
// interaction implementation.  The Workbench treats `detail.canvas` as an image
// coordinate surface; this property name is kept for existing consumers.
const first = (name, fallback) => {
  const column = control.data[name];
  if (!column || !column.length) return fallback;
  return column[0] == null ? fallback : column[0];
};
const safeId = (value) => String(value || '').replace(/[^a-zA-Z0-9_-]/g, '-');
const shellId = `astronomical-video-managed-${safeId(control.id)}`;
const key = `astronomical-video-${control.id}`;
const states = window.__astronomicalManagedVideoPresentations ||
  (window.__astronomicalManagedVideoPresentations = Object.create(null));
const state = states[key] || (states[key] = {image: null, target: '', frame: -1, playing: null});

const findShell = () => {
  try {
    const view = globalThis.Bokeh?.index?.[display.id];
    const root = view ? (view.shadow_el || view.el) : null;
    const shell = root?.querySelector?.(`#${shellId}`);
    if (shell) return shell;
  } catch (_err) {}
  // Bokeh can embed the Div below another root/shadow root. Search only during
  // frame presentation, never in a repeated playback polling loop.
  const visit = (root) => {
    if (!root?.querySelector) return null;
    const found = root.querySelector(`#${shellId}`);
    if (found) return found;
    for (const element of root.querySelectorAll('*')) {
      if (element.shadowRoot) {
        const nested = visit(element.shadowRoot);
        if (nested) return nested;
      }
    }
    return null;
  };
  try { return visit(document); } catch (_err) { return null; }
};

const present = (attempt = 0) => {
  const mode = String(first('mode', '') || '');
  const datasetId = String(first('dataset_id', '') || '');
  const rowId = String(first('row_id', '') || '');
  if (mode !== 'server' || !datasetId || !rowId) return;
  const frame = Math.max(0, Math.floor(Number(first('frame_index', 0) || 0)));
  const playing = Boolean(first('playing', false));
  const shell = findShell();
  const image = shell?.querySelector?.('img.astronomical-video-frame[data-astronomical-current-frame="1"]');
  const correctFrame = shell?.getAttribute?.('data-astronomical-frame-index') === String(frame);
  if (!image?.isConnected || !correctFrame || !image.complete || !image.naturalWidth) {
    if (attempt < 24) requestAnimationFrame(() => present(attempt + 1));
    return;
  }
  const target = `${datasetId}\u0000${rowId}`;
  if (state.image === image && state.target === target && state.frame === frame && state.playing === playing) return;
  state.image = image;
  state.target = target;
  state.frame = frame;
  state.playing = playing;
  const sequence = Number(window.__astronomicalVideoPresentationSequence || 0) + 1;
  window.__astronomicalVideoPresentationSequence = sequence;
  const detail = {
    surface_key: key,
    presentation_sequence: sequence,
    dataset_id: datasetId,
    row_id: rowId,
    frame_index: frame,
    playing,
    display_transform: String(first('display_transform', 'source') || 'source'),
    canvas: image,
  };
  const last = window.__astronomicalVideoLastFrames ||
    (window.__astronomicalVideoLastFrames = Object.create(null));
  last[target] = detail;
  window.dispatchEvent(new CustomEvent('astronomical-video-frame', {detail}));
};
// Allow Bokeh to commit the Div.text DOM patch before resolving the new image.
requestAnimationFrame(() => present());
"""

_DIRECT_SEQUENCE_JS = r"""
const first = (name, fallback) => {
  const column = control.data[name];
  if (!column || column.length === 0) return fallback;
  const value = column[0];
  return value == null ? fallback : value;
};

const overlayFirst = (name, fallback) => {
  const column = overlay.data[name];
  if (!column || column.length === 0) return fallback;
  const value = column[0];
  return value == null ? fallback : value;
};

const escapeHtml = (value) => String(value == null ? '' : value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#39;');

const mode = String(first('mode', 'server'));
const generation = Number(first('generation', 0));
const controlRevision = Math.max(0, Math.floor(Number(first('control_revision', 0))));
const clientClock = Boolean(first('client_clock', false));
const playing = Boolean(first('playing', false));
const fps = Math.max(0, Number(first('fps', 0)));
const rate = Math.max(0.01, Number(first('playback_rate', 1)));
const frameCount = Math.max(0, Math.floor(Number(first('frame_count', 0))));
const prefetchCount = Math.max(0, Math.min(128, Math.floor(Number(first('prefetch_count', 0)))));
const decodedLookahead = Math.max(256, Math.min(1024, Math.floor(Number(first('decoded_lookahead', 256)))));
const configuredFrame = Math.max(0, Math.floor(Number(first('frame_index', 0))));
const pattern = String(first('uri_pattern', ''));
const rawFrameUris = first('frame_uris', []);
const frameUris = Array.isArray(rawFrameUris) ? rawFrameUris.map((value) => String(value || '')) : [];
const token = String(first('frame_token', '__ASTRONOMICAL_FRAME_INDEX__'));
const digits = Math.max(0, Math.floor(Number(first('frame_digits', 0))));
const offset = Math.floor(Number(first('frame_index_offset', 0)));
const step = Math.max(1, Math.floor(Number(first('frame_index_step', 1))));
const displayTransform = String(first('display_transform', 'source') || 'source');
const datasetId = String(first('dataset_id', '') || '');
const rowId = String(first('row_id', '') || '');
const message = String(first('message', ''));
const nativeWidth = Math.max(1, Math.floor(Number(first('native_width', 1)) || 1));
const nativeHeight = Math.max(1, Math.floor(Number(first('native_height', 1)) || 1));
const safeId = (value) => String(value || '').replaceAll(/[^a-zA-Z0-9_-]/g, '-');
const canvasId = `astronomical-video-canvas-${safeId(control.id)}`;
const overlayId = `astronomical-video-overlay-${safeId(control.id)}`;
const key = `astronomical-video-${control.id}`;
const registry = window.__astronomicalVideoSurfaces ||
  (window.__astronomicalVideoSurfaces = Object.create(null));
let state = registry[key];
if (!state) {
  state = registry[key] = {
    generation: -1,
    raf: null,
    cache: new Map(),
    readyFrames: new Set(),
    warm: new Set(),
    warmPending: new Map(),
    warmQueue: [],
    warmActive: 0,
    warmSignature: '',
    warmSerial: 0,
    warmWindowStart: -1,
    warmWindowEnd: -1,
    decodeQueue: [],
    decodeQueued: new Set(),
    decodeActive: 0,
    decodeSerial: 0,
    bufferPrimed: false,
    presentationSerial: 0,
    playbackSerial: 0,
    presentationPending: false,
    anchorFrame: 0,
    anchorMs: performance.now(),
    lastPresented: -1,
    configuredFrame: -1,
    lastPlaying: null,
    sourceImageMode: '',
    sourceChannelCount: 0,
    displayTransform: '',
    canvas: null,
    ctx2d: null,
    glState: null,
    overlayCanvas: null,
    overlayImages: new Map(),
    overlayQueue: [],
    overlayQueued: new Set(),
    overlayActive: 0,
    overlayStats: {paired: 0, blanks: 0, waits: 0, waitMs: 0, timeouts: 0},
    playbackPerf: null,
    diskCachePromise: null,
    diskWritePending: 0,
    diskWritesSincePrune: 0,
    diskPrunePending: false,
    prepareSerial: -1,
    prepareAborter: null,
    diskStats: {hits: 0, misses: 0, writes: 0, errors: 0},
  };
}

// A model configured before a saved layout is attached can miss its original
// data-change callback. The Python first-frame watchdog sends a bootstrap tick
// after mounting; timings below identify when callback, transfer, decode and
// first paint actually happen (without affecting the playback control revision).
if (!state.timingStart) state.timingStart = performance.now();
if (!state.timingStages) state.timingStages = new Set();
const markTiming = (stage, frame = configuredFrame, detail = '', once = true) => {
  if (once && state.timingStages.has(stage)) return;
  if (once) state.timingStages.add(stage);
  const elapsed = Math.max(0, performance.now() - state.timingStart);
  if (typeof metrics !== 'undefined' && metrics?.change?.emit) {
    metrics.data = {
      stage: [stage], dataset_id: [datasetId], row_id: [rowId],
      frame_index: [frame], elapsed_ms: [elapsed], detail: [String(detail || '')],
    };
    metrics.change.emit();
  }
  if (typeof console !== 'undefined' && console.debug) {
    console.debug('[AL_VIDEO_TIMING]', stage, 'frame=', frame,
      'elapsed_ms=', elapsed.toFixed(1), String(detail || ''));
  }
};
markTiming('js_callback', configuredFrame, `bootstrap=${first('bootstrap_tick', 0)}`);

const cancelClock = () => {
  if (state.raf != null) {
    cancelAnimationFrame(state.raf);
    state.raf = null;
  }
};

const closeCachedEntry = (promise) => {
  if (!promise || typeof promise.then !== 'function') return;
  promise.then((entry) => {
    try {
      if (entry && entry.bitmap && typeof entry.bitmap.close === 'function') entry.bitmap.close();
    } catch (_err) {}
  }).catch(() => {});
};

const clearDecodedCache = () => {
  for (const promise of state.cache.values()) closeCachedEntry(promise);
  state.cache.clear();
  state.readyFrames.clear();
};

const invalidatePresentation = () => {
  state.presentationSerial += 1;
  state.playbackSerial += 1;
  state.presentationPending = false;
};

const resetDrawingState = () => {
  state.canvas = null;
  state.ctx2d = null;
  state.glState = null;
  state.overlayCanvas = null;
};

if (state.generation !== generation) {
  try {state.prepareAborter?.abort();} catch (_err) {}
  state.prepareAborter = null;
  cancelClock();
  state.generation = generation;
  state.timingStart = performance.now();
  state.timingStages = new Set();
  state.playbackPerf = null;
  state.prepareSerial = -1;
  markTiming('generation_start', configuredFrame);
  clearDecodedCache();
  state.warm.clear();
  state.warmPending = new Map();
  state.warmQueue = [];
  state.warmActive = 0;
  state.warmSignature = '';
  state.warmSerial += 1;
  state.warmWindowStart = -1;
  state.warmWindowEnd = -1;
  state.decodeQueue = [];
  state.decodeQueued = new Set();
  state.decodeActive = 0;
  state.decodeSerial += 1;
  state.bufferPrimed = false;
  invalidatePresentation();
  state.lastPresented = -1;
  state.paintOverlay = null;
  state.overlayImages = new Map();
  state.overlayQueue = [];
  state.overlayQueued = new Set();
  state.overlayActive = 0;
  state.overlayStats = {paired: 0, blanks: 0, waits: 0, waitMs: 0, timeouts: 0};
  state.configuredFrame = -1;
  state.lastPlaying = null;
  state.sourceImageMode = '';
  state.sourceChannelCount = 0;
  state.displayTransform = '';
  resetDrawingState();
}

if (state.displayTransform !== displayTransform) {
  cancelClock();
  clearDecodedCache();
  state.warm.clear();
  state.warmPending = new Map();
  state.warmQueue = [];
  state.warmSerial += 1;
  state.decodeQueue = [];
  state.decodeQueued = new Set();
  state.decodeActive = 0;
  state.decodeSerial += 1;
  state.bufferPrimed = false;
  invalidatePresentation();
  state.lastPresented = -1;
  state.paintOverlay = null;
  state.overlayImages = new Map();
  state.overlayQueue = [];
  state.overlayQueued = new Set();
  state.overlayActive = 0;
  state.overlayStats = {paired: 0, blanks: 0, waits: 0, waitMs: 0, timeouts: 0};
  state.sourceImageMode = '';
  state.sourceChannelCount = 0;
  state.displayTransform = displayTransform;
  resetDrawingState();
}

// A play/pause transition is a hard presentation boundary. Pause invalidates an
// unresolved image/bitmap promise before it can publish another playing frame.
if (state.lastPlaying !== playing) {
  if (playing) {
    try {state.prepareAborter?.abort();} catch (_err) {}
    state.playRequestedAt = performance.now();
    // A new play segment must not include the user's paused interval in FPS.
    state.playbackPerf = null;
    markTiming('play_requested', configuredFrame, `last_presented=${state.lastPresented}`, false);
  } else {
    state.playbackPerf = null;
  }
  cancelClock();
  invalidatePresentation();
  state.lastPlaying = playing;
  state.anchorFrame = configuredFrame;
  state.anchorMs = performance.now();
}

if (mode !== 'direct_sequence' || !clientClock || ((!pattern || !token) && frameUris.length === 0)) {
  cancelClock();
  invalidatePresentation();
  state.paintOverlay = null;
  if (message) {
    display.text = `<div class="astronomical-video-message">${escapeHtml(message)}</div>`;
  }
  return;
}

// Only explicit core.video presentation contracts are accepted. Scientific RGBA
// is never corrected with CSS/SVG filters; rgb_unpremultiplied uses a decoded
// ImageBitmap plus WebGL and forces output alpha opaque in the fragment shader.
if (!['source', 'rgb_unpremultiplied', 'rgb_unpremultiplied_png'].includes(displayTransform)) {
  cancelClock();
  invalidatePresentation();
  return;
}

const clampFrame = (value) => {
  let frame = Math.max(0, Math.floor(Number(value || 0)));
  if (frameCount > 0) frame = Math.min(frame, frameCount - 1);
  return frame;
};

const frameUri = (logicalFrame) => {
  const logical = clampFrame(logicalFrame);
  if (frameUris.length > 0) {
    if (logical >= frameUris.length) return '';
    return String(frameUris[logical] || '');
  }
  const physical = Math.max(0, offset + logical * step);
  let text = String(physical);
  if (digits > 0) text = text.padStart(digits, '0');
  return pattern.split(token).join(text);
};

const findCanvas = () => {
  if (state.canvas && presentationCanvasIsLive(state.canvas)) return state.canvas;
  try {
    const bokeh = globalThis.Bokeh;
    const view = bokeh && bokeh.index ? bokeh.index[display.id] : null;
    const root = view ? (view.shadow_el || view.el) : null;
    const canvas = root && typeof root.querySelector === 'function'
      ? root.querySelector(`#${canvasId}`)
      : null;
    if (canvas) {
      state.canvas = canvas;
      return canvas;
    }
  } catch (_err) {}

  // Defensive fallback for embedding arrangements where the Div is not a global
  // Bokeh root. Traverse open shadow roots only during initial surface discovery.
  const visit = (root) => {
    if (!root || typeof root.querySelector !== 'function') return null;
    const direct = root.querySelector(`#${canvasId}`);
    if (direct) return direct;
    for (const element of root.querySelectorAll('*')) {
      if (element.shadowRoot) {
        const nested = visit(element.shadowRoot);
        if (nested) return nested;
      }
    }
    return null;
  };
  try {
    const canvas = visit(document);
    if (canvas) state.canvas = canvas;
    return canvas;
  } catch (_err) {
    return null;
  }
};

const findOverlay = () => {
  if (state.overlayCanvas && state.overlayCanvas.isConnected) return state.overlayCanvas;
  try {
    const bokeh = globalThis.Bokeh;
    const view = bokeh && bokeh.index ? bokeh.index[display.id] : null;
    const root = view ? (view.shadow_el || view.el) : null;
    const canvas = root && typeof root.querySelector === 'function'
      ? root.querySelector(`#${overlayId}`)
      : null;
    if (canvas) {
      state.overlayCanvas = canvas;
      return canvas;
    }
  } catch (_err) {}

  // ``display`` is commonly nested inside Panel's Bokeh pane rather than being a
  // global Bokeh root. The base video canvas already falls back to traversing open
  // shadow roots for exactly that reason. Overlay discovery must use the same
  // embedding-safe path or the provider can return valid frames that are never painted.
  const visit = (root) => {
    if (!root || typeof root.querySelector !== 'function') return null;
    const direct = root.querySelector(`#${overlayId}`);
    if (direct) return direct;
    for (const element of root.querySelectorAll('*')) {
      if (element.shadowRoot) {
        const nested = visit(element.shadowRoot);
        if (nested) return nested;
      }
    }
    return null;
  };
  try {
    const canvas = visit(document);
    if (canvas) state.overlayCanvas = canvas;
    return canvas;
  } catch (_err) {
    return null;
  }
};

// Decode masks ahead of their corresponding base frames. A Bokeh overlay update
// may arrive independently of the browser clock; keep the cache keyed by the
// exact URI, not merely the frame number, so provider/layer changes are safe.
const overlayUriFor = (frame) => {
  const frames = overlayFirst('overlay_frames', {}) || {};
  return String(frames[String(frame)] || frames[frame] || '');
};
const trimOverlayImages = () => {
  const maxBytes = 64 * 1024 * 1024;
  const maxEntries = 24;
  let bytes = 0;
  for (const entry of state.overlayImages.values()) bytes += entry.bytes || 0;
  // The current frame and its nearest successors must survive an LRU trim.
  const protectedUris = new Set();
  for (let i = 0; i <= 2; i += 1) {
    const uri = overlayUriFor(Math.max(0, state.lastPresented) + i);
    if (uri) protectedUris.add(uri);
  }
  for (const [uri, entry] of state.overlayImages) {
    if (bytes <= maxBytes && state.overlayImages.size <= maxEntries) break;
    if (protectedUris.has(uri) || !entry.image) continue;
    state.overlayImages.delete(uri);
    bytes -= entry.bytes || 0;
  }
};
const loadOverlayImage = (uri) => {
  if (!uri) return Promise.resolve(null);
  const existing = state.overlayImages.get(uri);
  if (existing) {
    // Refresh LRU order, including inflight requests.
    state.overlayImages.delete(uri);
    state.overlayImages.set(uri, existing);
    return existing.promise;
  }
  const entry = {image: null, bytes: 0, promise: null};
  state.overlayImages.set(uri, entry);
  entry.promise = new Promise((resolve) => {
    const image = new Image();
    image.loading = 'eager';
    image.decoding = 'async';
    image.onload = () => {
      const decode = typeof image.decode === 'function' ? image.decode() : Promise.resolve();
      Promise.resolve(decode).then(() => {
        if (state.overlayImages.get(uri) !== entry) return resolve(null);
        entry.image = image;
        entry.bytes = Math.max(1, Number(image.naturalWidth || image.width || nativeWidth)) *
          Math.max(1, Number(image.naturalHeight || image.height || nativeHeight)) * 4;
        trimOverlayImages();
        resolve(image);
      }).catch(() => resolve(null));
    };
    image.onerror = () => resolve(null);
    image.src = uri;
  });
  return entry.promise;
};
const pumpOverlayQueue = () => {
  while (state.overlayActive < 3 && state.overlayQueue.length) {
    const uri = state.overlayQueue.shift();
    state.overlayQueued.delete(uri);
    if (state.overlayImages.has(uri)) continue;
    state.overlayActive += 1;
    void loadOverlayImage(uri).finally(() => {
      state.overlayActive = Math.max(0, state.overlayActive - 1);
      pumpOverlayQueue();
    });
  }
};
const prefetchOverlays = (frame) => {
  if (state.lastPresented < 0) return; // never compete with initial base paint
  // Scale lookahead to the approximate decoded RGBA memory footprint.
  const perFrameBytes = Math.max(1, nativeWidth * nativeHeight * 4);
  const lookahead = Math.max(3, Math.min(16, Math.floor(48 * 1024 * 1024 / perFrameBytes)));
  const needed = [];
  for (let i = 1; i <= lookahead; i += 1) {
    const logical = frame + i;
    if (frameCount && logical >= frameCount) break;
    const uri = overlayUriFor(logical);
    if (uri) needed.push(uri);
  }
  const wanted = new Set(needed);
  state.overlayQueue = state.overlayQueue.filter(uri => wanted.has(uri));
  state.overlayQueued = new Set(state.overlayQueue);
  for (const uri of needed) {
    if (!state.overlayImages.has(uri) && !state.overlayQueued.has(uri)) {
      state.overlayQueued.add(uri);
      state.overlayQueue.push(uri);
    }
  }
  pumpOverlayQueue();
};
const waitForMatchingOverlay = async (logical, isPlaying) => {
  // Keep the existing base+mask pair visible while a known next mask decodes.
  // Never hold the initial frame or a manual seek on overlay work.
  if (!isPlaying || state.lastPresented < 0) return;
  const uri = overlayUriFor(logical);
  if (!uri) return;
  if (state.overlayImages.get(uri)?.image) return;
  const startedAt = performance.now();
  state.overlayStats.waits += 1;
  let timeout;
  try {
    const outcome = await Promise.race([
      loadOverlayImage(uri).then(() => 'decoded'),
      new Promise(resolve => { timeout = setTimeout(() => resolve('timeout'), 750); }),
    ]);
    if (outcome === 'timeout') state.overlayStats.timeouts += 1;
  } finally {
    state.overlayStats.waitMs += performance.now() - startedAt;
    if (timeout != null && typeof clearTimeout === 'function') clearTimeout(timeout);
  }
};
const applyOverlay = (frame) => {
  const canvas = findOverlay();
  if (!canvas) return;
  const uri = overlayUriFor(frame);
  const target = String(frame);
  const nextSerial = () => {
    const serial = Math.max(0, Number(canvas.dataset.overlaySerial || 0)) + 1;
    canvas.dataset.overlaySerial = String(serial);
    return serial;
  };
  const hideOverlay = () => {
    nextSerial();
    canvas.dataset.overlayUri = '';
    canvas.dataset.overlayFrame = '';
    canvas.dataset.overlayPendingUri = '';
    canvas.dataset.overlayPendingFrame = '';
    canvas.style.display = 'none';
    // Clearing is required when a mask is genuinely absent, not on every frame.
    const ctx = canvas.getContext('2d', {alpha: true});
    if (ctx) ctx.clearRect(0, 0, Number(canvas.width || 0), Number(canvas.height || 0));
  };

  if (state.lastPresented !== frame || !state.canvas || !state.canvas.isConnected || !uri) {
    hideOverlay();
    return;
  }
  canvas.style.setProperty('z-index', '2', 'important');
  canvas.style.opacity = String(Math.max(0, Math.min(1,
    Number(overlayFirst('overlay_opacity', 0.45)) || 0.45)));
  if (canvas.dataset.overlayUri === uri && canvas.dataset.overlayFrame === target) {
    canvas.style.display = 'block';
    prefetchOverlays(frame);
    return;
  }
  const ready = state.overlayImages.get(uri)?.image;
  if (ready) {
    // This is synchronous with the base-frame draw, so the browser can composite
    // the matching pair in the same paint without an intermediate blank mask.
    const width = Math.max(1, Number(ready.naturalWidth || ready.width || nativeWidth));
    const height = Math.max(1, Number(ready.naturalHeight || ready.height || nativeHeight));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext('2d', {alpha: true});
    if (!ctx) return hideOverlay();
    nextSerial();
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(ready, 0, 0, width, height);
    canvas.dataset.overlayUri = uri;
    canvas.dataset.overlayFrame = target;
    canvas.dataset.overlayPendingUri = '';
    canvas.dataset.overlayPendingFrame = '';
    canvas.style.display = 'block';
    state.overlayStats.paired += 1;
    prefetchOverlays(frame);
    return;
  }
  if (canvas.dataset.overlayPendingUri === uri && canvas.dataset.overlayPendingFrame === target)
    return;
  // No matching decoded mask is available. Do not show a different frame's mask.
  state.overlayStats.blanks += 1;
  hideOverlay();
  const serial = Number(canvas.dataset.overlaySerial || 0);
  canvas.dataset.overlayPendingUri = uri;
  canvas.dataset.overlayPendingFrame = target;
  void loadOverlayImage(uri).then((image) => {
    if (!image || Number(canvas.dataset.overlaySerial || 0) !== serial ||
        state.lastPresented !== frame || !state.canvas?.isConnected ||
        overlayUriFor(frame) !== uri) return;
    applyOverlay(frame);
  });
  prefetchOverlays(frame);
};
// Keep the overlay-only transport on exactly the same presentation path.
state.paintOverlay = applyOverlay;
state.prefetchOverlays = prefetchOverlays;

const ensureCanvas = (isCurrent = () => true) => new Promise((resolve, reject) => {
  let attempts = 0;
  const probe = () => {
    if (!isCurrent()) {
      reject(new Error('Video presentation superseded before its canvas was attached.'));
      return;
    }
    const canvas = findCanvas();
    if (canvas) {
      markTiming('canvas_attached', configuredFrame, `probes=${attempts}`);
      resolve(canvas);
      return;
    }
    attempts += 1;
    if (attempts > 120) {
      reject(new Error('Could not locate the video canvas after layout restoration.'));
      return;
    }
    // Panel's saved-layout restoration can attach the Div after its data callback.
    // Retry without traversing every Bokeh/shadow root on every animation frame.
    setTimeout(probe, Math.min(200, 40 + attempts * 5));
  };
  probe();
});

const revealCanvas = (canvas) => {
  if (!canvas) return;
  // The source canvas is the canonical base presentation. Make its visible opacity
  // explicit at the same priority as the initial hidden state so workspace/theme
  // styles cannot wash it out while browser consumers (e.g. Segmentation Viewer)
  // copy the raw canvas pixels at full strength.
  canvas.style.setProperty('opacity', '1', 'important');
  canvas.style.setProperty('filter', 'none', 'important');
  canvas.style.setProperty('mix-blend-mode', 'normal', 'important');
  canvas.style.setProperty('z-index', '1', 'important');

  // The managed first-frame image exists only as a no-flash hand-off surface while
  // the first browser frame decodes. Once the persistent canvas has drawn, retire
  // that fallback explicitly. Leaving two transformed/grid media layers visible lets
  // browser compositor ordering vary and can make the fallback look frozen even while
  // the canonical canvas is advancing (downstream viewers still see the new canvas).
  try {
    const shell = canvas.parentElement;
    if (shell && typeof shell.querySelectorAll === 'function') {
      for (const fallback of shell.querySelectorAll('.astronomical-video-frame')) {
        fallback.style.visibility = 'hidden';
        fallback.style.pointerEvents = 'none';
      }
    }
  } catch (_err) {}
};

const drawSourceImage = async (img, isCurrent) => {
  const canvas = await ensureCanvas(isCurrent);
  if (!isCurrent()) return null;
  const width = Math.max(1, Number(img.naturalWidth || img.width || nativeWidth));
  const height = Math.max(1, Number(img.naturalHeight || img.height || nativeHeight));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  if (!state.ctx2d) {
    state.ctx2d = canvas.getContext('2d', {alpha: false, desynchronized: true});
  }
  if (!state.ctx2d) throw new Error('Browser 2-D canvas playback is unavailable.');
  state.ctx2d.drawImage(img, 0, 0, width, height);
  revealCanvas(canvas);
  return canvas;
};

const compileShader = (gl, type, source) => {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'unknown shader compilation error';
    gl.deleteShader(shader);
    throw new Error(`WebGL shader compilation failed: ${message}`);
  }
  return shader;
};

const ensureWebGL = async (isCurrent) => {
  const canvas = await ensureCanvas(isCurrent);
  if (!isCurrent()) return null;
  if (state.glState && state.glState.canvas === canvas) return state.glState;
  const gl = canvas.getContext('webgl', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  }) || canvas.getContext('experimental-webgl', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  if (!gl) throw new Error('WebGL is required for lossless browser RGBD playback.');

  const vertex = compileShader(gl, gl.VERTEX_SHADER, `
    attribute vec2 a_position;
    attribute vec2 a_texcoord;
    varying vec2 v_texcoord;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
      v_texcoord = a_texcoord;
    }
  `);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    uniform sampler2D u_texture;
    varying vec2 v_texcoord;
    void main() {
      vec4 sample = texture2D(u_texture, v_texcoord);
      gl_FragColor = vec4(sample.rgb, 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || 'unknown program link error';
    gl.deleteProgram(program);
    throw new Error(`WebGL program link failed: ${message}`);
  }

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const texcoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]),
    gl.STATIC_DRAW,
  );

  state.glState = {canvas, gl, program, positionBuffer, texcoordBuffer};
  return state.glState;
};

const drawUnpremultipliedRGB = async (bitmap, isCurrent) => {
  const webgl = await ensureWebGL(isCurrent);
  if (!webgl || !isCurrent()) return null;
  const {canvas, gl, program, positionBuffer, texcoordBuffer} = webgl;
  const width = Math.max(1, Number(bitmap.width || nativeWidth));
  const height = Math.max(1, Number(bitmap.height || nativeHeight));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  gl.viewport(0, 0, width, height);
  gl.useProgram(program);

  const position = gl.getAttribLocation(program, 'a_position');
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const texcoord = gl.getAttribLocation(program, 'a_texcoord');
  gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
  gl.enableVertexAttribArray(texcoord);
  gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0);

  gl.disable(gl.BLEND);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  try {
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  } catch (_err) {}

  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  const sampler = gl.getUniformLocation(program, 'u_texture');
  gl.uniform1i(sampler, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.deleteTexture(texture);
  revealCanvas(canvas);
  return canvas;
};

const presentationCanvasIsLive = (canvas) => {
  if (!canvas || !canvas.isConnected || !Number(canvas.width) || !Number(canvas.height)) return false;
  try {
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0) || !(rect.height > 0)) return false;
    let node = canvas;
    while (node) {
      if (node.nodeType === 1) {
        const style = globalThis.getComputedStyle ? globalThis.getComputedStyle(node) : null;
        if (style && (style.display === 'none' || style.visibility === 'hidden')) return false;
      }
      if (node.parentElement) {
        node = node.parentElement;
        continue;
      }
      const root = typeof node.getRootNode === 'function' ? node.getRootNode() : null;
      node = root && root.host ? root.host : null;
    }
  } catch (_err) {
    return Boolean(canvas.isConnected);
  }
  return true;
};

const publish = (frame, isPlaying, error = '', errorFrame = -1, canvas = null) => {
  const logical = clampFrame(frame);
  feedback.data = {
    frame_index: [logical],
    playing: [Boolean(isPlaying)],
    control_revision: [controlRevision],
    error: [String(error || '')],
    error_frame: [Number(errorFrame)],
    source_image_mode: [state.sourceImageMode || ''],
    source_channel_count: [state.sourceChannelCount || 0],
    painted: [Boolean(!error && state.lastPresented === logical &&
      presentationCanvasIsLive(canvas || findCanvas()))],
  };
  feedback.change.emit();

  // Browser-local presentation signal for dependent modality plugins. Only a live,
  // connected presentation surface is allowed to publish globally. Mapping-gate or
  // workspace panel replacement can leave an older requestAnimationFrame closure
  // alive briefly; without this guard it can compete with the visible Video Viewer
  // for the same dataset/row and make downstream overlays follow stale frames.
  try {
    const presentedCanvas = canvas || findCanvas();
    if (!presentationCanvasIsLive(presentedCanvas)) return;
    const sequence = Number(window.__astronomicalVideoPresentationSequence || 0) + 1;
    window.__astronomicalVideoPresentationSequence = sequence;
    const detail = {
      surface_key: key,
      presentation_sequence: sequence,
      presented_at: performance.now(),
      dataset_id: datasetId,
      row_id: rowId,
      frame_index: logical,
      playing: Boolean(isPlaying),
      control_revision: controlRevision,
      display_transform: displayTransform,
      canvas: presentedCanvas,
    };
    const lastFrames = window.__astronomicalVideoLastFrames ||
      (window.__astronomicalVideoLastFrames = Object.create(null));
    if (datasetId || rowId) lastFrames[`${datasetId}\u0000${rowId}`] = detail;
    window.dispatchEvent(new CustomEvent('astronomical-video-frame', {detail}));
  } catch (_err) {}
};

const loadSourceImage = (logical, uri) => {
  const loadStartedAt = performance.now();
  const img = new Image();
  if (logical === configuredFrame && state.lastPresented < 0)
    markTiming('image_request_start', logical);
  img.loading = 'eager';
  img.decoding = 'async';
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      if (logical === configuredFrame && state.lastPresented < 0)
        markTiming('image_decoded', logical);
      state.warm.add(logical);
      resolve({
        kind: 'image', img, uri, frame: logical,
        loadMs: performance.now() - loadStartedAt,
        transferMs: null, decodeMs: null,
      });
    };
    const fail = () => {
      if (settled) return;
      settled = true;
      reject(new Error(`Could not load browser video frame ${logical}: ${uri}`));
    };
    img.onerror = fail;
    img.onload = () => {
      if (typeof img.decode !== 'function') {
        finish();
        return;
      }
      img.decode().then(finish).catch(finish);
    };
    img.src = uri;
  });
};

// Persistent, optional browser-side frame storage. The managed Python resolver
// already caches its own downloads; browser-direct scientific PNGs bypass that
// resolver. Cache Storage closes that gap without a Bokeh request per frame.
// Every entry is scoped by its full source URL + dataset/row identity, capped by
// age and a shared byte budget, and stored only after a successful image decode.
// The HTTP cache remains the fallback when Cache Storage is unavailable.
const diskCacheName = 'astronomical-video-frames-v1';
const diskCacheMaxAgeMs = 72 * 60 * 60 * 1000;
const diskCacheMaxBytes = 512 * 1024 * 1024;
const diskCacheMaxEntries = 320;
const diskCacheEnabled = () => (
  typeof caches !== 'undefined' && typeof caches.open === 'function' &&
  typeof Request === 'function' && typeof Response === 'function' &&
  typeof TextEncoder === 'function' &&
  typeof crypto !== 'undefined' && Boolean(crypto.subtle?.digest) &&
  typeof location !== 'undefined' && /^https?:$/.test(String(location.protocol))
);
const diskCacheHandle = async () => {
  if (!diskCacheEnabled()) return null;
  if (!state.diskCachePromise)
    state.diskCachePromise = caches.open(diskCacheName).catch(() => null);
  return state.diskCachePromise;
};
const diskCacheRequest = async (uri) => {
  if (!diskCacheEnabled()) return null;
  // A synthetic same-origin key avoids retaining signed URLs or user tokens in
  // the browser's Cache Storage request list. The full URI (including a pinned
  // revision when supplied by the importer) is part of the SHA-256 digest.
  const identity = `${datasetId}\u0000${rowId}\u0000${uri}`;
  const bytes = new TextEncoder().encode(identity);
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hex = Array.from(hash, value => value.toString(16).padStart(2, '0')).join('');
  return new Request(`${location.origin}/__astronomical_video_frame_cache__/${hex}`);
};
const diskCacheRead = async (uri) => {
  try {
    const cache = await diskCacheHandle();
    if (!cache) return null;
    const request = await diskCacheRequest(uri);
    if (!request) return null;
    const response = await cache.match(request);
    if (!response) { state.diskStats.misses += 1; return null; }
    const saved = Number(response.headers.get('x-astronomical-saved-ms') || 0);
    if (!saved || Date.now() - saved > diskCacheMaxAgeMs || saved > Date.now() + 60000) {
      cache.delete(request).catch(() => {});
      state.diskStats.misses += 1;
      return null;
    }
    const blob = await response.blob();
    if (!blob || blob.size <= 0) {
      cache.delete(request).catch(() => {});
      state.diskStats.misses += 1;
      return null;
    }
    state.diskStats.hits += 1;
    return {blob, request};
  } catch (_err) {
    state.diskStats.errors += 1;
    return null;
  }
};
const pruneDiskCache = async (cache) => {
  const keys = await cache.keys();
  if (keys.length === 0) return;
  const sizes = [];
  let total = 0;
  for (const request of keys) {
    const response = await cache.match(request);
    const size = Math.max(0, Number(response?.headers?.get('x-astronomical-bytes') || 0));
    sizes.push(size);
    total += size;
    // Release the main thread periodically while checking browser disk usage.
    if (sizes.length % 16 === 0)
      await new Promise(resolve => setTimeout(resolve, 0));
  }
  let remaining = keys.length;
  // Cache.keys() returns insertion order; FIFO eviction avoids reading the
  // stored image bodies or maintaining a second persistent metadata database.
  for (let i = 0; i < keys.length &&
       (remaining > diskCacheMaxEntries || total > diskCacheMaxBytes); i += 1) {
    await cache.delete(keys[i]);
    total -= sizes[i];
    remaining -= 1;
  }
};
const diskCacheWrite = async (uri, blob, headers) => {
  if (!blob || blob.size <= 0 || blob.size > 16 * 1024 * 1024 ||
      state.diskWritePending >= 2 || !diskCacheEnabled()) return false;
  const cacheControl = String(headers?.get?.('cache-control') || '').toLowerCase();
  if (cacheControl.includes('no-store') || cacheControl.includes('private')) return false;
  state.diskWritePending += 1;
  try {
    const cache = await diskCacheHandle();
    const request = await diskCacheRequest(uri);
    if (!cache || !request) return false;
    const makeResponse = () => new Response(blob, {headers: {
      'Content-Type': blob.type || 'image/png',
      'x-astronomical-saved-ms': String(Date.now()),
      'x-astronomical-bytes': String(blob.size),
    }});
    try {
      await cache.put(request, makeResponse());
    } catch (err) {
      // A browser may impose a smaller quota than our own byte budget. Evict
      // some older frames and retry once; never block playback on this path.
      if (String(err?.name || '').includes('QuotaExceeded')) {
        const entries = await cache.keys();
        const toRemove = Math.max(1, Math.ceil(entries.length / 4));
        for (const old of entries.slice(0, toRemove)) await cache.delete(old);
        await cache.put(request, makeResponse());
      } else throw err;
    }
    state.diskStats.writes += 1;
    state.diskWritesSincePrune += 1;
    if (state.diskWritesSincePrune >= 24 && !state.diskPrunePending) {
      state.diskWritesSincePrune = 0;
      state.diskPrunePending = true;
      // Storage housekeeping must not occupy either of the two image-write
      // slots or block the display path while playback is underway.
      setTimeout(() => {
        pruneDiskCache(cache).catch(() => {state.diskStats.errors += 1;})
          .finally(() => {state.diskPrunePending = false;});
      }, 250);
    }
    return true;
  } catch (_err) {
    state.diskStats.errors += 1;
    return false;
  } finally {
    state.diskWritePending = Math.max(0, state.diskWritePending - 1);
  }
};

// Explicit preparation is bounded to 60 frames (at most twelve seconds at 5
// FPS). Fetching while paused avoids competing with an active playback clock.
// Store original image bytes: RGBD depth/alpha must not be converted to a JPEG
// that could accidentally be consumed as the original scientific image.
const prepareFrameWindow = async (serial) => {
  const identity = state.generation;
  const aborter = typeof AbortController === 'function' ? new AbortController() : null;
  state.prepareAborter = aborter;
  const isCurrent = () => state.prepareSerial === serial &&
    state.generation === identity && !state.lastPlaying &&
    state.configuredFrame === configuredFrame;
  if (!diskCacheEnabled() || frameCount < 1 ||
      !['rgb_unpremultiplied', 'rgb_unpremultiplied_png'].includes(displayTransform)) {
    markTiming('cache_prepare_unavailable', configuredFrame,
      'done=0 total=0 cached=0 fetched=0 failed=0', false);
    return;
  }
  // Give the foreground first frame priority over explicitly requested caching.
  for (let attempts = 0; attempts < 150 && state.lastPresented < 0 && isCurrent(); attempts += 1)
    await new Promise(resolve => setTimeout(resolve, 100));
  if (!isCurrent()) return;
  const start = Math.max(0, state.lastPresented);
  const total = Math.max(0, Math.min(60, frameCount - start - 1));
  let next = start + 1, done = 0, cached = 0, fetched = 0, failed = 0;
  let lastReport = performance.now();
  const progress = (stage) => markTiming(stage, start + done,
    `done=${done} total=${total} cached=${cached} fetched=${fetched} failed=${failed}`, false);
  const worker = async () => {
    while (next <= start + total && isCurrent()) {
      const frame = next++;
      const uri = frameUri(frame);
      if (!uri) {failed++; done++; continue;}
      try {
        const local = await diskCacheRead(uri);
        if (local) {cached++;} else {
          const response = await fetch(uri, {mode: 'cors', credentials: 'omit',
            cache: 'force-cache', ...(aborter ? {signal: aborter.signal} : {})});
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const blob = await response.blob();
          if (!isCurrent()) return;
          if (displayTransform === 'rgb_unpremultiplied_png') {
            const header = new Uint8Array(await blob.slice(0, 26).arrayBuffer());
            const signature = [137, 80, 78, 71, 13, 10, 26, 10];
            if (header.length < 26 || !signature.every((byte,i) => header[i] === byte) ||
                String.fromCharCode(...header.slice(12,16)) !== 'IHDR' ||
                header[24] !== 8 || ![2,6].includes(header[25]))
              throw new Error('Only supported 8-bit RGB/RGBA PNGs may be prepared.');
          }
          if (await diskCacheWrite(uri, blob, response.headers)) fetched++;
          else failed++;
        }
      } catch (_err) {failed++;}
      done++;
      if (performance.now() - lastReport >= 2000) {
        lastReport = performance.now();
        progress('cache_prepare_progress');
      }
    }
  };
  await Promise.all([worker(), worker()]);
  if (state.prepareAborter === aborter) state.prepareAborter = null;
  progress(isCurrent() ? 'cache_prepare_done' : 'cache_prepare_cancelled');
};

const loadUnpremultipliedBitmap = async (logical, uri) => {
  const loadStartedAt = performance.now();
  if (typeof createImageBitmap !== 'function') {
    throw new Error('createImageBitmap is unavailable for scientific RGBD playback.');
  }
  let response;
  let diskHit = false;
  let storedRequest = null;
  if (logical === configuredFrame && state.lastPresented < 0)
    markTiming('fetch_start', logical, displayTransform);
  const stored = await diskCacheRead(uri);
  let headersAt;
  let blob;
  if (stored) {
    ({blob, request: storedRequest} = stored);
    diskHit = true;
    headersAt = performance.now();
    if (logical === configuredFrame && state.lastPresented < 0)
      markTiming('disk_cache_hit', logical, `bytes=${blob.size}`);
  } else {
    try {
      response = await fetch(uri, {mode: 'cors', credentials: 'omit', cache: 'force-cache'});
    } catch (err) {
      throw new Error(`CORS fetch failed for scientific video frame ${logical}: ${err}`);
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while loading scientific video frame ${logical}.`);
    }
    headersAt = performance.now();
    if (logical === configuredFrame && state.lastPresented < 0)
      markTiming('fetch_headers', logical, `status=${response.status || 200}`);
    blob = await response.blob();
  }
  const bytesReadyAt = performance.now();
  if (logical === configuredFrame && state.lastPresented < 0)
    markTiming('blob_ready', logical, `bytes=${blob.size || 'unknown'} disk=${diskHit}`);
  if (displayTransform === 'rgb_unpremultiplied_png') {
    // PNG signature + IHDR: only 8-bit RGB (2) or RGBA (6) may take the
    // cold-start path. Palette/greyscale/16-bit inputs retain the managed
    // decoder's scientific-channel semantics rather than guessing in WebGL.
    const header = new Uint8Array(await blob.slice(0, 26).arrayBuffer());
    const signature = [137, 80, 78, 71, 13, 10, 26, 10];
    const png = header.length >= 26 && signature.every((byte, i) => header[i] === byte);
    const ihdr = png && String.fromCharCode(...header.slice(12, 16)) === 'IHDR';
    if (!ihdr || header[24] !== 8 || ![2, 6].includes(header[25])) {
      throw new Error('PNG channel format requires managed frame decoding.');
    }
    state.sourceImageMode = header[25] === 6 ? 'RGBA' : 'RGB';
    state.sourceChannelCount = header[25] === 6 ? 4 : 3;
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, {
      premultiplyAlpha: 'none',
      colorSpaceConversion: 'none',
      imageOrientation: 'flipY',
    });
  } catch (err) {
    throw new Error(`Could not decode scientific video frame ${logical} without alpha premultiplication: ${err}`);
  }
  const decodedAt = performance.now();
  if (logical === configuredFrame && state.lastPresented < 0)
    markTiming('bitmap_decoded', logical);
  state.warm.add(logical);
  if (!diskHit) void diskCacheWrite(uri, blob, response?.headers);
  return {
    kind: 'bitmap', bitmap, uri, frame: logical, diskHit,
    loadMs: decodedAt - loadStartedAt,
    headersMs: headersAt - loadStartedAt,
    transferMs: diskHit ? 0 : bytesReadyAt - headersAt,
    decodeMs: decodedAt - bytesReadyAt,
    bytes: blob.size || 0,
  };
};

const ensureCached = (frame) => {
  const logical = clampFrame(frame);
  const existing = state.cache.get(logical);
  if (existing) return existing;
  const uri = frameUri(logical);
  if (!uri) return Promise.reject(new Error(`No frame URI is available for logical frame ${logical}`));
  const promise = (
    (displayTransform === 'rgb_unpremultiplied' || displayTransform === 'rgb_unpremultiplied_png')
      ? loadUnpremultipliedBitmap(logical, uri)
      : loadSourceImage(logical, uri)
  ).then((entry) => {
    if (state.cache.get(logical) === promise) state.readyFrames.add(logical);
    return entry;
  }).catch((err) => {
    if (state.cache.get(logical) === promise) state.cache.delete(logical);
    state.readyFrames.delete(logical);
    throw err;
  });
  state.cache.set(logical, promise);
  return promise;
};

const warmFrame = (frame, serial) => {
  const logical = clampFrame(frame);
  if (serial !== state.warmSerial) return Promise.resolve();
  if (state.warm.has(logical) || state.cache.has(logical)) return Promise.resolve();
  const pending = state.warmPending.get(logical);
  if (pending) return pending;
  const uri = frameUri(logical);
  if (!uri) return Promise.resolve(false);

  let promise;
  if (displayTransform === 'rgb_unpremultiplied' || displayTransform === 'rgb_unpremultiplied_png') {
    promise = fetch(uri, {mode: 'cors', credentials: 'omit', cache: 'force-cache'})
      .then((response) => {
        if (!response.ok) return false;
        return response.arrayBuffer().then(() => true);
      })
      .catch(() => false);
  } else {
    const img = new Image();
    img.loading = 'eager';
    img.decoding = 'async';
    promise = new Promise((resolve) => {
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
      img.src = uri;
    });
  }
  promise = promise.then((ok) => {
    if (ok && serial === state.warmSerial) state.warm.add(logical);
    return ok;
  }).finally(() => {
    if (state.warmPending.get(logical) === promise) state.warmPending.delete(logical);
  });
  state.warmPending.set(logical, promise);
  return promise;
};

const pumpWarmQueue = (serial) => {
  if (serial !== state.warmSerial) return;
  const concurrency = 2;
  while (state.warmActive < concurrency && state.warmQueue.length > 0) {
    const frame = state.warmQueue.shift();
    if (state.warm.has(frame) || state.cache.has(frame) || state.warmPending.has(frame)) continue;
    state.warmActive += 1;
    warmFrame(frame, serial).finally(() => {
      if (serial !== state.warmSerial) return;
      state.warmActive = Math.max(0, state.warmActive - 1);
      pumpWarmQueue(serial);
    });
  }
};

const scheduleNetworkWarm = (center) => {
  if (frameCount <= 0 || fps <= 0) return;
  // Scientific PNG/RGBD already uses the decoded-frame prefetch queue. This
  // separate warmer downloaded whole images into throwaway ArrayBuffers,
  // competing with frames that the playback clock actually needs. Do not
  // duplicate these network requests; continue decoded-frame prefetch instead.
  if (displayTransform === 'rgb_unpremultiplied' ||
      displayTransform === 'rgb_unpremultiplied_png') return;
  const signature = `${generation}|${pattern}|${token}|${digits}|${offset}|${step}|${displayTransform}|${frameCount}`;
  if (state.warmSignature !== signature) {
    state.warmSignature = signature;
    state.warmSerial += 1;
    state.warmQueue = [];
    state.warmActive = 0;
    state.warmWindowStart = -1;
    state.warmWindowEnd = -1;
  }
  const serial = state.warmSerial;
  const logicalCenter = clampFrame(center);
  const aheadFrames = Math.max(12, Math.min(120, Math.ceil(fps * rate * 8.0)));
  const targetAhead = Math.max(aheadFrames, Math.min(180, Math.ceil(aheadFrames * 1.25)));
  const behindFrames = Math.max(4, Math.min(32, Math.ceil(fps * 2.0)));
  const desiredStart = Math.max(0, logicalCenter - behindFrames);
  const desiredEnd = Math.min(frameCount - 1, logicalCenter + targetAhead);
  const remaining = state.warmWindowEnd >= logicalCenter
    ? state.warmWindowEnd - logicalCenter
    : -1;
  const refillThreshold = Math.max(16, Math.ceil(aheadFrames * 0.85));
  if (
    state.warmWindowStart >= 0 &&
    logicalCenter >= state.warmWindowStart &&
    logicalCenter <= state.warmWindowEnd &&
    remaining > refillThreshold
  ) return;

  state.warmWindowStart = desiredStart;
  state.warmWindowEnd = desiredEnd;
  const queue = [];
  for (let frame = logicalCenter + 1; frame <= desiredEnd; frame += 1) {
    if (!state.warm.has(frame) && !state.cache.has(frame) && !state.warmPending.has(frame)) queue.push(frame);
  }
  for (let frame = logicalCenter; frame >= desiredStart; frame -= 1) {
    if (!state.warm.has(frame) && !state.cache.has(frame) && !state.warmPending.has(frame)) queue.push(frame);
  }
  state.warmQueue = queue;
  pumpWarmQueue(serial);
};

// Bound decoded resident memory by estimated pixels, not just frame count.
const decodedFrameBudget = () => {
  const width = Math.max(1, Number((state.canvas && state.canvas.width) || nativeWidth));
  const height = Math.max(1, Number((state.canvas && state.canvas.height) || nativeHeight));
  return Math.max(4, Math.min(96, Math.floor((256 * 1024 * 1024) / (width * height * 4))));
};
const decodedTarget = () => Math.max(
  2,
  Math.min(48, Math.floor(decodedFrameBudget() / 2), decodedLookahead,
    Math.ceil(Math.max(1.0, fps * rate) * 2.0)),
);

const pruneCache = (center) => {
  const target = decodedTarget();
  // UX-first replay cache: for ordinary research clips keep every decoded frame in
  // the current browser surface. A completed first pass can then replay/seek at 2×
  // without paying image/RGBD decode cost again. Very long clips still use a large
  // sliding window so browser memory remains finite.
  const residentDecodeLimit = decodedFrameBudget();
  if (frameCount > 0 && frameCount <= residentDecodeLimit) return;
  const behind = Math.min(Math.floor(residentDecodeLimit / 3),
    Math.max(1, Math.ceil(Math.max(1.0, fps * rate) * 2.0)));
  const minimum = Math.max(0, center - behind);
  const maximum = frameCount > 0
    ? Math.min(frameCount - 1, center + target)
    : center + Math.max(prefetchCount, target);
  const hardLimit = residentDecodeLimit;
  if (state.cache.size <= hardLimit) return;
  const keep = new Map();
  for (let frame = minimum; frame <= maximum; frame += 1) {
    if (state.cache.has(frame)) keep.set(frame, state.cache.get(frame));
  }
  for (const [frame, promise] of state.cache.entries()) {
    if (!keep.has(frame)) {
      state.readyFrames.delete(frame);
      closeCachedEntry(promise);
    }
  }
  state.cache = keep;
  registry[key].cache = keep;
};

const pumpDecodeQueue = () => {
  const serial = state.decodeSerial;
  // Keep a few browser connections available for a newly selected/seeked frame.
  while (state.decodeActive < 4 && state.decodeQueue.length > 0) {
    const frame = state.decodeQueue.shift();
    state.decodeQueued.delete(frame);
    if (state.cache.has(frame)) continue;
    state.decodeActive += 1;
    ensureCached(frame).catch(() => {}).finally(() => {
      if (serial !== state.decodeSerial) return;
      state.decodeActive = Math.max(0, state.decodeActive - 1);
      pumpDecodeQueue();
    });
  }
};

const prefetchDecoded = (frame) => {
  const logical = clampFrame(frame);
  const target = decodedTarget();
  for (let distance = 1; distance <= target; distance += 1) {
    const candidate = logical + distance;
    if (frameCount > 0 && candidate >= frameCount) break;
    if (state.cache.has(candidate) || state.decodeQueued.has(candidate)) continue;
    state.decodeQueued.add(candidate);
    state.decodeQueue.push(candidate);
  }
  pruneCache(logical);
  pumpDecodeQueue();
};

const drawEntry = async (entry, isCurrent) => {
  if (entry.kind === 'bitmap') return drawUnpremultipliedRGB(entry.bitmap, isCurrent);
  return drawSourceImage(entry.img, isCurrent);
};

// Steady-state playback telemetry: aggregate in the browser and send at most
// one sample per two seconds. Per-frame Bokeh traffic would itself cause stalls.
// Timestamp the successful base-canvas paint, not the slider or requested frame.
const recordPlaybackPaint = (logical, entry, requestAt, cacheHit) => {
  const now = performance.now();
  const expectedMs = 1000 / Math.max(0.01, fps * rate);
  let perf = state.playbackPerf;
  if (!perf) {
    perf = state.playbackPerf = {
      start: now, lastPaintAt: null, frames: 0, intervals: 0,
      gapSum: 0, maxGap: 0, late: 0, waitSum: 0, maxWait: 0,
      cacheHits: 0, loads: 0, loadSum: 0, transferSum: 0,
      decodeSum: 0, decodeSamples: 0, pendingTicks: 0,
    };
  }
  const waitMs = Math.max(0, now - requestAt);
  if (perf.lastPaintAt !== null) {
    const gap = Math.max(0, now - perf.lastPaintAt);
    perf.intervals += 1;
    perf.gapSum += gap;
    perf.maxGap = Math.max(perf.maxGap, gap);
    if (gap > expectedMs * 1.5) perf.late += 1;
  }
  perf.lastPaintAt = now;
  perf.frames += 1;
  perf.waitSum += waitMs;
  perf.maxWait = Math.max(perf.maxWait, waitMs);
  if (cacheHit) perf.cacheHits += 1;
  // Timing for each distinct load is attached to its decoded entry. A warm
  // decoded frame counts as a cache hit; do not mistake its old fetch duration
  // for the time this frame spent waiting on the playback critical path.
  if (entry && Number.isFinite(entry.loadMs) && !cacheHit) {
    perf.loads += 1;
    perf.loadSum += entry.loadMs;
    if (Number.isFinite(entry.transferMs)) perf.transferSum += entry.transferMs;
    if (Number.isFinite(entry.decodeMs)) {
      perf.decodeSum += entry.decodeMs;
      perf.decodeSamples += 1;
    }
  }
  const windowMs = now - perf.start;
  if (windowMs < 2000) return;
  const observedFps = perf.gapSum > 0 ? 1000 * perf.intervals / perf.gapSum : 0;
  const detail = [
    `fps=${observedFps.toFixed(2)}`,
    `target=${(fps * rate).toFixed(2)}`,
    `late=${perf.late}/${perf.intervals}`,
    `max_gap_ms=${perf.maxGap.toFixed(1)}`,
    `avg_gap_ms=${(perf.intervals ? perf.gapSum / perf.intervals : 0).toFixed(1)}`,
    `avg_wait_ms=${(perf.waitSum / perf.frames).toFixed(1)}`,
    `max_wait_ms=${perf.maxWait.toFixed(1)}`,
    `cache_hits=${perf.cacheHits}/${perf.frames}`,
    `loads=${perf.loads}`,
    `avg_load_ms=${(perf.loads ? perf.loadSum / perf.loads : 0).toFixed(1)}`,
    `avg_transfer_ms=${(perf.loads ? perf.transferSum / perf.loads : 0).toFixed(1)}`,
    `avg_decode_ms=${(perf.decodeSamples ? perf.decodeSum / perf.decodeSamples : 0).toFixed(1)}`,
    `pending_ticks=${perf.pendingTicks}`,
    `decoding=${state.decodeActive}`,
    `warm=${state.warmActive}`,
    `decode_queue=${state.decodeQueue.length}`,
    `warm_queue=${state.warmQueue.length}`,
    `cached=${state.cache.size}`,
    `disk_hits=${state.diskStats.hits}`,
    `disk_misses=${state.diskStats.misses}`,
    `disk_writes=${state.diskStats.writes}`,
    `disk_errors=${state.diskStats.errors}`,
    `mask_paired=${state.overlayStats.paired}`,
    `mask_blank=${state.overlayStats.blanks}`,
    `mask_waits=${state.overlayStats.waits}`,
    `mask_wait_ms=${state.overlayStats.waitMs.toFixed(1)}`,
    `mask_timeouts=${state.overlayStats.timeouts}`,
  ].join(' ');
  markTiming('playback_window', logical, detail, false);
  state.playbackPerf = {...perf,
    start: now, frames: 0, intervals: 0, gapSum: 0, maxGap: 0,
    late: 0, waitSum: 0, maxWait: 0, cacheHits: 0, loads: 0,
    loadSum: 0, transferSum: 0, decodeSum: 0, decodeSamples: 0,
    pendingTicks: 0,
  };
};

const present = (frame, isPlaying) => {
  const logical = clampFrame(frame);
  if (state.lastPresented === logical && state.canvas &&
      presentationCanvasIsLive(state.canvas)) {
    applyOverlay(logical);
    prefetchDecoded(logical);
    return Promise.resolve(true);
  }
  if (state.presentationPending) return Promise.resolve(false);

  const requestAt = performance.now();
  // A queued promise is not a cache hit until its bitmap is decoded.
  const cacheHit = state.readyFrames.has(logical);
  const serial = ++state.presentationSerial;
  const expectedGeneration = state.generation;
  const expectedPlaybackSerial = state.playbackSerial;
  state.presentationPending = true;
  return ensureCached(logical).then(async (entry) => {
    if (
      serial !== state.presentationSerial ||
      expectedGeneration !== state.generation ||
      expectedPlaybackSerial !== state.playbackSerial
    ) return false;
    const isCurrent = () => (
      serial === state.presentationSerial &&
      expectedGeneration === state.generation &&
      expectedPlaybackSerial === state.playbackSerial
    );
    await waitForMatchingOverlay(logical, isPlaying);
    if (!isCurrent()) return false;
    const canvas = await drawEntry(entry, isCurrent);
    if (!canvas ||
      serial !== state.presentationSerial ||
      expectedGeneration !== state.generation ||
      expectedPlaybackSerial !== state.playbackSerial
    ) return false;
    const firstPaint = state.lastPresented < 0;
    const previousFrame = state.lastPresented;
    state.lastPresented = logical;
    if (isPlaying && previousFrame >= 0 && logical !== previousFrame)
      recordPlaybackPaint(logical, entry, requestAt, cacheHit);
    if (firstPaint) markTiming('first_frame_painted', logical, `transform=${displayTransform}`);
    if (isPlaying && previousFrame >= 0 && logical !== previousFrame && state.playRequestedAt) {
      markTiming('first_motion', logical,
        `since_play_ms=${(performance.now() - state.playRequestedAt).toFixed(1)}`);
      state.playRequestedAt = null;
    }
    applyOverlay(logical);
    state.presentationPending = false;
    state.anchorFrame = logical;
    state.anchorMs = performance.now();
    const atEnd = frameCount > 0 && logical >= frameCount - 1;
    publish(logical, Boolean(isPlaying) && Boolean(state.lastPlaying) && !atEnd, '', -1, canvas);
    prefetchDecoded(logical);
    if (state.bufferPrimed && state.lastPlaying) scheduleNetworkWarm(logical);
    if (atEnd) cancelClock();
    return true;
  }).catch((err) => {
    if (
      serial !== state.presentationSerial ||
      expectedGeneration !== state.generation ||
      expectedPlaybackSerial !== state.playbackSerial
    ) return false;
    state.presentationPending = false;
    cancelClock();
    const text = err && err.message ? err.message : String(err || 'Browser frame load failed');
    markTiming('frame_error', logical, text.replace(/https?:\/\/[^\s]+/g, '[frame URL]'), false);
    publish(logical, false, text, logical, findCanvas());
    return false;
  });
};

// A server-originated seek/scrub invalidates unresolved presentation immediately.
if (state.configuredFrame !== configuredFrame) {
  state.configuredFrame = configuredFrame;
  invalidatePresentation();
  state.decodeQueue = [];
  state.decodeQueued.clear();
  state.decodeActive = 0;
  state.decodeSerial += 1;
}

cancelClock();
state.anchorFrame = configuredFrame;
state.anchorMs = performance.now();

const startClock = () => {
  if (!playing || fps <= 0 || state.generation !== generation) return;
  const clockSerial = state.playbackSerial;
  const tick = (now) => {
    if (state.generation !== generation || state.playbackSerial !== clockSerial || !state.lastPlaying) {
      state.raf = null;
      return;
    }
    // A replaced/hidden plugin panel must relinquish its browser clock. Otherwise an
    // orphaned surface can keep decoding frames and broadcasting the same dataset/row
    // after MappingGatedPanel has installed a newer visible Video Viewer.
    if (state.canvas && !presentationCanvasIsLive(state.canvas)) {
      cancelClock();
      invalidatePresentation();
      return;
    }
    if (state.presentationPending) {
      if (state.playbackPerf) state.playbackPerf.pendingTicks += 1;
      state.raf = requestAnimationFrame(tick);
      return;
    }
    if (frameCount > 0 && state.lastPresented >= frameCount - 1) {
      publish(state.lastPresented, false, '', -1, findCanvas());
      state.raf = null;
      return;
    }

    const elapsedSeconds = Math.max(0, (now - state.anchorMs) / 1000.0);
    const due = elapsedSeconds * fps * rate >= 1.0;
    if (due) {
      // Never catch up by jumping over source frames. Advance one prepared source
      // frame at a time; if the next bitmap is not ready, the previous canvas pixels
      // remain visible. The renderer therefore never clears to black and never runs
      // a catch-up burst when I/O resumes.
      const nextFrame = clampFrame(Math.max(state.lastPresented, state.anchorFrame) + 1);
      present(nextFrame, true);
    }
    state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
};

// The selected frame is the only request on the first-paint critical path.
// Subsequent decoded/network prefetch starts after that paint and never gates
// the playback clock. The next-frame clock already buffers on a missing frame.
const requestedPrepare = Math.max(0, Math.floor(Number(first('prepare_serial', 0))));
if (requestedPrepare > state.prepareSerial) {
  state.prepareSerial = requestedPrepare;
  if (requestedPrepare > 0) void prepareFrameWindow(requestedPrepare);
}

const startupSerial = state.playbackSerial;
const initial = present(configuredFrame, playing);
initial.then((painted) => {
  if (!painted || state.generation !== generation || state.playbackSerial !== startupSerial) return;
  if (playing && fps > 0 && state.lastPlaying) {
    state.bufferPrimed = true;
    state.anchorFrame = state.lastPresented >= 0 ? state.lastPresented : configuredFrame;
    state.anchorMs = performance.now();
    markTiming('play_clock_started', state.lastPresented);
    startClock();
  }
  requestAnimationFrame(() => {
    if (state.generation === generation && state.playbackSerial === startupSerial) {
      scheduleNetworkWarm(state.lastPresented >= 0 ? state.lastPresented : configuredFrame);
    }
  });
}).catch(() => {});
"""
