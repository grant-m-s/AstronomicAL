from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

try:
    from packaging.version import Version
except Exception:
    Version = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MarketplaceBrowseEntry:
    source_id: str
    plugin_id: str
    name: str
    version: str
    description: str
    status: str
    tags: tuple[str, ...] = ()
    license: Optional[str] = None
    installed_version: Optional[str] = None
    update_version: Optional[str] = None


@dataclass(frozen=True)
class MarketplaceInstalledEntry:
    source_id: str
    plugin_id: str
    name: str
    version: str
    status: str
    update_version: Optional[str] = None


@dataclass(frozen=True)
class MarketplaceUpdateEntry:
    source_id: str
    plugin_id: str
    installed_version: str
    status: str
    target_version: Optional[str] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class MarketplaceViewSnapshot:
    source_id: Optional[str]
    source_name: Optional[str]
    catalogue_loaded: bool
    source_status: Optional[str]
    source_error: Optional[str]
    browse: tuple[MarketplaceBrowseEntry, ...]
    installed: tuple[MarketplaceInstalledEntry, ...]
    updates: tuple[MarketplaceUpdateEntry, ...]


def collect_marketplace_view(
    context: Any,
    *,
    source_id: str | None = None,
    query: str = "",
) -> MarketplaceViewSnapshot:
    marketplace = getattr(context, "marketplace", None)
    if marketplace is None:
        return MarketplaceViewSnapshot(
            source_id=None,
            source_name=None,
            catalogue_loaded=False,
            source_status=None,
            source_error="MarketplaceService is not available on AppContext.",
            browse=(),
            installed=(),
            updates=(),
        )

    try:
        sources = list(marketplace.sources() or [])
    except Exception as exc:
        return MarketplaceViewSnapshot(
            source_id=None,
            source_name=None,
            catalogue_loaded=False,
            source_status=None,
            source_error=str(exc),
            browse=(),
            installed=(),
            updates=(),
        )

    source_ids = [str(getattr(source, "id", "") or "") for source in sources]
    source_ids = [value for value in source_ids if value]
    selected = str(source_id or "").strip()
    if selected not in source_ids:
        selected = source_ids[0] if source_ids else ""

    if not selected:
        return MarketplaceViewSnapshot(
            source_id=None,
            source_name=None,
            catalogue_loaded=False,
            source_status=None,
            source_error=None,
            browse=(),
            installed=(),
            updates=(),
        )

    state = _source_state(marketplace, selected)
    try:
        catalogue = marketplace.catalogue(selected)
    except Exception as exc:
        return MarketplaceViewSnapshot(
            source_id=selected,
            source_name=selected,
            catalogue_loaded=False,
            source_status=getattr(state, "last_status", None),
            source_error=str(exc),
            browse=(),
            installed=(),
            updates=(),
        )

    source_name = (
        str(getattr(getattr(catalogue, "marketplace", None), "name", "") or selected)
        if catalogue is not None
        else selected
    )

    installed_records = _installed_records(context)
    installed_by_id = {
        str(getattr(record, "id", "") or ""): record
        for record in installed_records
        if str(getattr(record, "id", "") or "")
    }
    update_infos = _update_infos(context)
    updates_by_id = {
        str(getattr(info, "plugin_id", "") or ""): info
        for info in update_infos
        if str(getattr(info, "plugin_id", "") or "")
    }

    browse: list[MarketplaceBrowseEntry] = []
    if catalogue is not None:
        for plugin in list(catalogue.list_plugins() or []):
            release = latest_release(getattr(plugin, "releases", ()), include_yanked=False)
            if release is None:
                release = latest_release(getattr(plugin, "releases", ()), include_yanked=True)
            if release is None:
                continue

            manifest = release.manifest
            plugin_id = str(getattr(plugin, "id", "") or manifest.id)
            record = installed_by_id.get(plugin_id)
            installed_version = (
                str(getattr(record, "version", "") or "") or None
                if record is not None
                else None
            )
            installed_source = (
                str(getattr(record, "source", "") or "")
                if record is not None
                else ""
            )
            installed_source_id = (
                str(getattr(record, "source_id", "") or "")
                if record is not None
                else ""
            )
            update = updates_by_id.get(plugin_id)
            update_version = _optional_text(getattr(update, "target_version", None))
            update_status = _status_value(getattr(update, "status", None))

            if record is None:
                status = "Available"
            elif installed_source == "marketplace" and installed_source_id == selected:
                if update_status == "current_release_yanked":
                    status = "Installed release withdrawn"
                elif update_version:
                    status = "Update available"
                else:
                    status = "Installed"
            else:
                status = "Installed elsewhere"

            entry = MarketplaceBrowseEntry(
                source_id=selected,
                plugin_id=plugin_id,
                name=str(getattr(manifest, "name", "") or plugin_id),
                version=str(getattr(release, "version", "") or ""),
                description=str(getattr(manifest, "description", "") or ""),
                status=status,
                tags=tuple(str(value) for value in (getattr(manifest, "tags", ()) or ())),
                license=_optional_text(getattr(plugin, "license", None)),
                installed_version=installed_version,
                update_version=update_version,
            )
            if _matches_query(entry, query):
                browse.append(entry)

    browse.sort(key=lambda item: (item.name.lower(), item.plugin_id.lower()))

    installed: list[MarketplaceInstalledEntry] = []
    for record in installed_records:
        if str(getattr(record, "source", "") or "") != "marketplace":
            continue
        if str(getattr(record, "source_id", "") or "") != selected:
            continue
        plugin_id = str(getattr(record, "id", "") or "")
        update = updates_by_id.get(plugin_id)
        update_version = _optional_text(getattr(update, "target_version", None))
        status = _status_value(getattr(update, "status", None)) or "installed"
        manifest = getattr(record, "manifest", {}) or {}
        name = str(manifest.get("name", "") or getattr(record, "name", "") or plugin_id)
        installed.append(
            MarketplaceInstalledEntry(
                source_id=selected,
                plugin_id=plugin_id,
                name=name,
                version=str(getattr(record, "version", "") or ""),
                status=status,
                update_version=update_version,
            )
        )
    installed.sort(key=lambda item: (item.name.lower(), item.plugin_id.lower()))

    updates: list[MarketplaceUpdateEntry] = []
    for info in update_infos:
        if str(getattr(info, "source_id", "") or "") != selected:
            continue
        status = _status_value(getattr(info, "status", None)) or "unknown"
        if status == "up_to_date":
            continue
        updates.append(
            MarketplaceUpdateEntry(
                source_id=selected,
                plugin_id=str(getattr(info, "plugin_id", "") or ""),
                installed_version=str(getattr(info, "installed_version", "") or ""),
                status=status,
                target_version=_optional_text(getattr(info, "target_version", None)),
                detail=_optional_text(getattr(info, "detail", None)),
            )
        )
    updates.sort(key=lambda item: item.plugin_id.lower())

    return MarketplaceViewSnapshot(
        source_id=selected,
        source_name=source_name,
        catalogue_loaded=catalogue is not None,
        source_status=_optional_text(getattr(state, "last_status", None)),
        source_error=_optional_text(getattr(state, "error", None)),
        browse=tuple(browse),
        installed=tuple(installed),
        updates=tuple(updates),
    )


def latest_release(
    releases: Iterable[Any],
    *,
    include_yanked: bool = False,
) -> Any | None:
    candidates: list[tuple[Any, str, Any]] = []
    for release in releases:
        if not include_yanked and bool(getattr(release, "yanked", False)):
            continue
        version_text = str(getattr(release, "version", "") or "").strip()
        if not version_text:
            continue
        if Version is None:
            key: Any = version_text
        else:
            try:
                key = Version(version_text)
            except Exception:
                continue
        candidates.append((key, str(getattr(release, "url", "") or ""), release))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _source_state(marketplace: Any, source_id: str) -> Any | None:
    try:
        states = list(marketplace.states() or [])
    except Exception:
        return None
    for state in states:
        if str(getattr(state, "source_id", "") or "") == source_id:
            return state
    return None


def _installed_records(context: Any) -> list[Any]:
    store = getattr(context, "installed_plugins", None)
    list_records = getattr(store, "list", None)
    if not callable(list_records):
        return []
    try:
        return list(list_records() or [])
    except Exception:
        return []


def _update_infos(context: Any) -> list[Any]:
    updates = getattr(context, "marketplace_updates", None)
    list_updates = getattr(updates, "list_updates", None)
    if not callable(list_updates):
        return []
    try:
        return list(list_updates() or [])
    except Exception:
        return []


def _matches_query(entry: MarketplaceBrowseEntry, query: str) -> bool:
    query = str(query or "").strip().lower()
    if not query:
        return True
    values = [
        entry.plugin_id,
        entry.name,
        entry.description,
        entry.status,
        entry.license or "",
        *entry.tags,
    ]
    return query in " ".join(values).lower()


def _status_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None