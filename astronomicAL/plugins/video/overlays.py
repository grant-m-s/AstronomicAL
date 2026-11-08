from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping, Optional, Protocol, Sequence

OVERLAY_CATALOG_CHANGED_TOPIC = "video.overlay.catalog.changed"
OVERLAY_SELECTION_CHANGED_TOPIC = "video.overlay.selection.changed"

@dataclass(frozen=True)
class VideoOverlaySource:
    """Small provider-neutral descriptor for one renderable video overlay source.

    ``core.video`` deliberately does not know whether a source represents masks,
    detections, tracks, imported annotations, or another modality. Providers own
    those semantics and return only JSON-safe display metadata here.
    """

    source_id: str
    provider_id: str
    label: str
    kind: str = "overlay"
    description: str = ""
    prepared: bool = True
    live: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": str(self.source_id),
            "provider_id": str(self.provider_id),
            "label": str(self.label),
            "kind": str(self.kind or "overlay"),
            "description": str(self.description or ""),
            "prepared": bool(self.prepared),
            "live": bool(self.live),
            "metadata": dict(self.metadata or {}),
        }

@dataclass(frozen=True)
class VideoOverlayLayer:
    layer_id: str
    label: str
    enabled_by_default: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": str(self.layer_id),
            "label": str(self.label),
            "enabled_by_default": bool(self.enabled_by_default),
            "metadata": dict(self.metadata or {}),
        }

class VideoOverlayProvider(Protocol):
    provider_id: str

    def list_sources(self, dataset_id: str, row_id: Any) -> Sequence[VideoOverlaySource | Mapping[str, Any]]:
        ...

    def list_layers(
        self,
        source_id: str,
        dataset_id: str,
        row_id: Any,
    ) -> Sequence[VideoOverlayLayer | Mapping[str, Any]]:
        ...

    def prepare_source(
        self,
        source_id: str,
        dataset_id: str,
        row_id: Any,
        *,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> Mapping[str, Any] | None:
        ...

    def render_window(
        self,
        source_id: str,
        dataset_id: str,
        row_id: Any,
        *,
        frame_start: int,
        frame_end: int,
        layer_ids: Sequence[str] = (),
        opacity: float = 0.45,
        cancel_token: Any = None,
    ) -> Mapping[str, Any]:
        ...

class VideoOverlayRegistry:
    """Live registry for provider-neutral overlays on the core.video surface.

    Providers are plugin-owned services. The registry keeps only descriptors,
    provider handles and the user's transient per-row source choice. Large overlay
    payloads never enter EventBus or workspace state.
    """

    service_key = "core.video.overlay_registry"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._lock = RLock()
        self._providers: dict[str, Any] = {}
        self._selected: dict[tuple[str, str], str] = {}

    def register_provider(self, provider: Any, *, replace: bool = False) -> None:
        provider_id = str(getattr(provider, "provider_id", "") or "").strip()
        if not provider_id:
            raise ValueError("video overlay providers require a non-empty provider_id")
        with self._lock:
            if provider_id in self._providers and not replace:
                raise ValueError(f"video overlay provider {provider_id!r} is already registered")
            self._providers[provider_id] = provider
        self.notify_catalog_changed(provider_id=provider_id)

    def unregister_provider(self, provider_id: str) -> None:
        provider_id = str(provider_id or "").strip()
        with self._lock:
            self._providers.pop(provider_id, None)
        self.notify_catalog_changed(provider_id=provider_id)

    def provider(self, provider_id: str) -> Any | None:
        with self._lock:
            return self._providers.get(str(provider_id or ""))

    def list_sources(self, dataset_id: str, row_id: Any) -> list[dict[str, Any]]:
        dataset_id = str(dataset_id or "")
        row_text = str(row_id or "")
        result: list[dict[str, Any]] = []
        with self._lock:
            providers = list(self._providers.items())
        for provider_id, provider in providers:
            method = getattr(provider, "list_sources", None)
            if not callable(method):
                continue
            try:
                values = method(dataset_id, row_id) or ()
            except Exception:
                continue
            for raw in values:
                if isinstance(raw, VideoOverlaySource):
                    value = raw.to_dict()
                elif isinstance(raw, Mapping):
                    value = dict(raw)
                else:
                    continue
                source_id = str(value.get("source_id") or "").strip()
                if not source_id:
                    continue
                value["source_id"] = source_id
                value["provider_id"] = str(value.get("provider_id") or provider_id)
                value["label"] = str(value.get("label") or source_id)
                value["kind"] = str(value.get("kind") or "overlay")
                value["dataset_id"] = dataset_id
                value["row_id"] = row_text
                result.append(value)
        result.sort(key=lambda value: (str(value.get("label") or "").casefold(), str(value.get("source_id") or "")))
        return result

    def list_layers(self, source_id: str, dataset_id: str, row_id: Any) -> list[dict[str, Any]]:
        source = self.resolve_source(source_id, dataset_id, row_id)
        if source is None:
            return []
        provider = self.provider(str(source.get("provider_id") or ""))
        method = getattr(provider, "list_layers", None)
        if not callable(method):
            return []
        try:
            values = method(str(source_id), str(dataset_id), row_id) or ()
        except Exception:
            return []
        result: list[dict[str, Any]] = []
        for raw in values:
            if isinstance(raw, VideoOverlayLayer):
                value = raw.to_dict()
            elif isinstance(raw, Mapping):
                value = dict(raw)
            else:
                continue
            layer_id = str(value.get("layer_id") or "").strip()
            if not layer_id:
                continue
            value["layer_id"] = layer_id
            value["label"] = str(value.get("label") or layer_id)
            value["enabled_by_default"] = bool(value.get("enabled_by_default", True))
            result.append(value)
        return result

    def resolve_source(self, source_id: str, dataset_id: str, row_id: Any) -> dict[str, Any] | None:
        source_id = str(source_id or "").strip()
        if not source_id:
            return None
        for source in self.list_sources(dataset_id, row_id):
            if str(source.get("source_id") or "") == source_id:
                return source
        return None

    def prepare_source(
        self,
        source_id: str,
        dataset_id: str,
        row_id: Any,
        *,
        cancel_token: Any = None,
        progress_callback: Any = None,
    ) -> Mapping[str, Any] | None:
        source = self.resolve_source(source_id, dataset_id, row_id)
        if source is None:
            raise KeyError(f"video overlay source {source_id!r} is unavailable")
        provider = self.provider(str(source.get("provider_id") or ""))
        method = getattr(provider, "prepare_source", None)
        if not callable(method):
            return None
        return method(
            str(source_id),
            str(dataset_id),
            row_id,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )

    def render_window(
        self,
        source_id: str,
        dataset_id: str,
        row_id: Any,
        *,
        frame_start: int,
        frame_end: int,
        layer_ids: Sequence[str] = (),
        opacity: float = 0.45,
        cancel_token: Any = None,
    ) -> Mapping[str, Any]:
        source = self.resolve_source(source_id, dataset_id, row_id)
        if source is None:
            raise KeyError(f"video overlay source {source_id!r} is unavailable")
        provider = self.provider(str(source.get("provider_id") or ""))
        method = getattr(provider, "render_window", None)
        if not callable(method):
            raise RuntimeError(f"video overlay provider {source.get('provider_id')!r} cannot render frames")
        return method(
            str(source_id),
            str(dataset_id),
            row_id,
            frame_start=max(0, int(frame_start or 0)),
            frame_end=max(0, int(frame_end or 0)),
            layer_ids=tuple(str(value) for value in (layer_ids or ()) if str(value)),
            opacity=min(1.0, max(0.0, float(opacity or 0.45))),
            cancel_token=cancel_token,
        )

    def selected_source(self, dataset_id: str, row_id: Any) -> str:
        key = (str(dataset_id or ""), str(row_id or ""))
        with self._lock:
            return str(self._selected.get(key) or "")

    def select_source(
        self,
        dataset_id: str,
        row_id: Any,
        source_id: Optional[str],
        *,
        origin: str = "",
    ) -> str:
        dataset_id = str(dataset_id or "")
        row_text = str(row_id or "")
        source_id = str(source_id or "").strip()
        if source_id and self.resolve_source(source_id, dataset_id, row_text) is None:
            # Live providers can advertise a source a few milliseconds after the
            # caller announces it. Preserve the requested transient choice and let
            # the viewer resolve it on the next catalogue event.
            pass
        key = (dataset_id, row_text)
        with self._lock:
            previous = str(self._selected.get(key) or "")
            if source_id:
                self._selected[key] = source_id
            else:
                self._selected.pop(key, None)
        if previous != source_id:
            self._publish(
                OVERLAY_SELECTION_CHANGED_TOPIC,
                {
                    "dataset_id": dataset_id,
                    "row_id": row_text,
                    "source_id": source_id or None,
                    "previous_source_id": previous or None,
                    "origin": str(origin or ""),
                },
            )
        return source_id

    def notify_catalog_changed(
        self,
        *,
        dataset_id: str = "",
        row_id: Any = "",
        provider_id: str = "",
        origin: str = "",
        source_id: str = "",
        frame_index: Optional[int] = None,
        content_changed: bool = False,
    ) -> None:
        """Publish lightweight catalogue/content invalidation metadata.

        ``content_changed`` is intentionally optional and frame-bounded. Providers can
        replace pixels in an already-selected transient source without forcing the
        Video Viewer to discard every resident overlay frame.
        """

        self._publish(
            OVERLAY_CATALOG_CHANGED_TOPIC,
            {
                "dataset_id": str(dataset_id or "") or None,
                "row_id": str(row_id or "") or None,
                "provider_id": str(provider_id or "") or None,
                "origin": str(origin or ""),
                "source_id": str(source_id or "") or None,
                "frame_index": (
                    None if frame_index is None else max(0, int(frame_index))
                ),
                "content_changed": bool(content_changed),
            },
        )

    def _publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        events = getattr(self.context, "events", None)
        publish = getattr(events, "publish", None)
        if not callable(publish):
            return
        try:
            publish(topic, dict(payload))
        except TypeError:
            try:
                publish(topic=topic, payload=dict(payload))
            except Exception:
                pass
        except Exception:
            pass

def create_video_overlay_registry(context: Any, **_: Any) -> VideoOverlayRegistry:
    return VideoOverlayRegistry(context)

def get_video_overlay_registry(context: Any) -> VideoOverlayRegistry | None:
    services = getattr(context, "services", None)
    if services is None:
        return None
    for name in ("get", "require"):
        method = getattr(services, name, None)
        if not callable(method):
            continue
        try:
            value = method(VideoOverlayRegistry.service_key)
        except Exception:
            continue
        if value is not None:
            return value
    return None

__all__ = [
    "OVERLAY_CATALOG_CHANGED_TOPIC",
    "OVERLAY_SELECTION_CHANGED_TOPIC",
    "VideoOverlayLayer",
    "VideoOverlayProvider",
    "VideoOverlayRegistry",
    "VideoOverlaySource",
    "create_video_overlay_registry",
    "get_video_overlay_registry",
]