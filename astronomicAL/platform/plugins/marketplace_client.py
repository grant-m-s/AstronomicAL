from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import time
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .marketplace import MarketplaceCatalogue, MarketplaceCatalogueError

DEFAULT_MARKETPLACE_TIMEOUT_SECONDS = 10.0
DEFAULT_MARKETPLACE_MAX_CATALOG_BYTES = 5 * 1024 * 1024
DEFAULT_MARKETPLACE_USER_AGENT = "AstronomicAL-Marketplace/1"
_MARKETPLACE_SOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

class MarketplaceClientError(RuntimeError):
    """Base error for marketplace retrieval and cache operations."""

class MarketplaceFetchError(MarketplaceClientError):
    """Raised when a marketplace catalogue cannot be retrieved."""

class MarketplaceCacheError(MarketplaceClientError):
    """Raised when a cached marketplace catalogue cannot be read or written."""

@dataclass(frozen=True)
class MarketplaceSource:
    """Configured remote marketplace catalogue,"""

    id: str
    url: str
    enabled: bool = True

    def __post_init__(self) -> None:
        source_id = str(self.id or "").strip()
        url = str(self.url or "").strip()

        if not source_id or not _MARKETPLACE_SOURCE_ID_RE.fullmatch(source_id):
            raise ValueError(
                "Marketplace source id must contain only letters, numbers, "
                "'.', '_' or '-'."
            )
        if not url:
            raise ValueError("Marketplace source URL is required.")

        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError(
                "Marketplace source URL must be an absolute HTTPS URL."
            )

        object.__setattr__(self, "id", source_id)
        object.__setattr__(self, "url", url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "enabled": self.enabled,
        }

@dataclass(frozen=True)
class MarketplaceCacheMetadata:
    """HTTP/cache metadata associated with a cached catalogue,"""

    source_id: str
    source_url: str
    fetched_at: float
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketplaceCacheMetadata":
        if not isinstance(data, Mapping):
            raise MarketplaceCacheError(
                "Marketplace cache metadata must be a JSON object."
            )

        try:
            fetched_at = float(data.get("fetched_at", 0.0))
        except (TypeError, ValueError) as exc:
            raise MarketplaceCacheError(
                "Marketplace cache fetched_at must be numeric."
            ) from exc

        return cls(
            source_id=str(data.get("source_id", "") or "").strip(),
            source_url=str(data.get("source_url", "") or "").strip(),
            fetched_at=fetched_at,
            etag=_optional_header(data.get("etag")),
            last_modified=_optional_header(data.get("last_modified")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "etag": self.etag,
            "last_modified": self.last_modified,
        }

@dataclass(frozen=True)
class MarketplaceHttpResponse:
    """Transport-neutral response used by MarketplaceClient."""

    status: int
    body: bytes
    headers: Mapping[str, str]

@dataclass(frozen=True)
class MarketplaceRefreshResult:
    """Result of loading or refreshing one marketplace source."""

    catalogue: MarketplaceCatalogue
    source: MarketplaceSource
    status: str
    fetched_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def from_cache(self) -> bool:
        return self.status in {"cached", "not_modified", "stale_cache"}

    @property
    def refreshed(self) -> bool:
        return self.status == "updated"

MarketplaceFetcher = Callable[
    [MarketplaceSource, Mapping[str, str], float, int],
    MarketplaceHttpResponse,
]

class MarketplaceCache:
    """Last-known-good on-disk cache for marketplace catalogues."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def source_dir(self, source: MarketplaceSource) -> Path:
        return self.root / source.id

    def catalogue_path(self, source: MarketplaceSource) -> Path:
        return self.source_dir(source) / "catalogue.json"

    def metadata_path(self, source: MarketplaceSource) -> Path:
        return self.source_dir(source) / "metadata.json"

    def load_catalogue(self, source: MarketplaceSource) -> Optional[MarketplaceCatalogue]:
        path = self.catalogue_path(source)
        if not path.exists():
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            catalogue = MarketplaceCatalogue.from_dict(data)
        except (OSError, UnicodeError, json.JSONDecodeError, MarketplaceCatalogueError) as exc:
            raise MarketplaceCacheError(
                f"Could not read cached marketplace catalogue {path}: {exc}"
            ) from exc

        _validate_catalogue_source(catalogue, source)
        return catalogue

    def load_metadata(
        self,
        source: MarketplaceSource,
    ) -> Optional[MarketplaceCacheMetadata]:
        path = self.metadata_path(source)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = MarketplaceCacheMetadata.from_dict(data)
        except (OSError, UnicodeError, json.JSONDecodeError, MarketplaceCacheError) as exc:
            raise MarketplaceCacheError(
                f"Could not read marketplace cache metadata {path}: {exc}"
            ) from exc

        if metadata.source_id != source.id:
            return None
        return metadata

    def store(
        self,
        source: MarketplaceSource,
        *,
        body: bytes,
        catalogue: MarketplaceCatalogue,
        metadata: MarketplaceCacheMetadata,
    ) -> None:
        _validate_catalogue_source(catalogue, source)

        source_dir = self.source_dir(source)
        catalogue_path = self.catalogue_path(source)
        metadata_path = self.metadata_path(source)

        try:
            source_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(catalogue_path, body)
            _atomic_write_text(
                metadata_path,
                json.dumps(
                    metadata.to_dict(),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        except OSError as exc:
            raise MarketplaceCacheError(
                f"Could not write marketplace cache under {source_dir}: {exc}"
            ) from exc

class MarketplaceClient:
    """Read-through cached client for one remote marketplace catalogue,

    Refreshing a catalogue never imports or executes plugin runtime code.
    Downloading/installing .alplugin archives is intentionally outside this class.
    """

    def __init__(
        self,
        *,
        source: MarketplaceSource,
        cache: MarketplaceCache,
        fetcher: Optional[MarketplaceFetcher] = None,
        timeout: float = DEFAULT_MARKETPLACE_TIMEOUT_SECONDS,
        max_catalogue_bytes: int = DEFAULT_MARKETPLACE_MAX_CATALOG_BYTES,
        user_agent: str = DEFAULT_MARKETPLACE_USER_AGENT,
    ) -> None:
        if float(timeout) <= 0:
            raise ValueError("Marketplace timeout must be greater than zero.")
        if int(max_catalogue_bytes) <= 0:
            raise ValueError("Marketplace max_catalogue_bytes must be greater than zero.")

        self.source = source
        self.cache = cache
        self.timeout = float(timeout)
        self.max_catalogue_bytes = int(max_catalogue_bytes)
        self.user_agent = str(user_agent or DEFAULT_MARKETPLACE_USER_AGENT)
        self._fetcher = fetcher or _fetch_https

    def load_cached(self) -> Optional[MarketplaceCatalogue]:
        return self.cache.load_catalogue(self.source)

    def refresh(
        self,
        *,
        allow_stale_cache: bool = True,
    ) -> MarketplaceRefreshResult:
        """Refresh the configured source and return a validated catalogue,

        When ``allow_stale_cache`` is true, a valid last-known-good catalogue is
        returned on network or validation failure with status ``stale_cache``.
        """

        cached: Optional[MarketplaceCatalogue]
        cache_error: Optional[Exception] = None
        try:
            cached = self.cache.load_catalogue(self.source)
        except MarketplaceCacheError as exc:
            cached = None
            cache_error = exc

        metadata: Optional[MarketplaceCacheMetadata]
        try:
            metadata = self.cache.load_metadata(self.source)
        except MarketplaceCacheError:
            metadata = None

        headers = self._request_headers(metadata)
        try:
            response = self._fetch(headers)
            if response.status == 304:
                if cached is None:
                    response = self._fetch(self._request_headers(None))
                else:
                    return MarketplaceRefreshResult(
                        catalogue=cached,
                        source=self.source,
                        status="not_modified",
                        fetched_at=(
                            metadata.fetched_at
                            if metadata is not None
                            else None
                        ),
                    )

            if response.status != 200:
                raise MarketplaceFetchError(
                    f"Marketplace returned unexpected HTTP status {response.status}."
                )

            catalogue = self._parse_response(response.body)
            now = time.time()
            new_metadata = MarketplaceCacheMetadata(
                source_id=self.source.id,
                source_url=self.source.url,
                fetched_at=now,
                etag=_header(response.headers, "etag"),
                last_modified=_header(response.headers, "last-modified"),
            )
            self.cache.store(
                self.source,
                body=response.body,
                catalogue=catalogue,
                metadata=new_metadata,
            )
            return MarketplaceRefreshResult(
                catalogue=catalogue,
                source=self.source,
                status="updated",
                fetched_at=now,
            )
        except (MarketplaceClientError, MarketplaceCatalogueError, UnicodeError, json.JSONDecodeError) as exc:
            if allow_stale_cache and cached is not None:
                return MarketplaceRefreshResult(
                    catalogue=cached,
                    source=self.source,
                    status="stale_cache",
                    fetched_at=(
                        metadata.fetched_at
                        if metadata is not None
                        else None
                    ),
                    error=str(exc),
                )

            if cache_error is not None:
                raise MarketplaceFetchError(
                    f"Marketplace refresh failed and cached catalogue is invalid: "
                    f"{cache_error}; refresh error: {exc}"
                ) from exc

            if isinstance(exc, MarketplaceClientError):
                raise
            raise MarketplaceFetchError(
                f"Marketplace refresh failed: {exc}"
            ) from exc

    def _request_headers(
        self,
        metadata: Optional[MarketplaceCacheMetadata],
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        # Validators are only reusable while the configured source URL is unchanged.
        if metadata is not None and metadata.source_url == self.source.url:
            if metadata.etag:
                headers["If-None-Match"] = metadata.etag
            if metadata.last_modified:
                headers["If-Modified-Since"] = metadata.last_modified

        return headers

    def _fetch(self, headers: Mapping[str, str]) -> MarketplaceHttpResponse:
        try:
            response = self._fetcher(
                self.source,
                headers,
                self.timeout,
                self.max_catalogue_bytes,
            )
        except MarketplaceClientError:
            raise
        except Exception as exc:
            raise MarketplaceFetchError(
                f"Could not fetch marketplace {self.source.id!r}: {exc}"
            ) from exc

        if not isinstance(response, MarketplaceHttpResponse):
            raise MarketplaceFetchError(
                "Marketplace fetcher returned an invalid response object."
            )

        if len(response.body) > self.max_catalogue_bytes:
            raise MarketplaceFetchError(
                f"Marketplace catalogue exceeds the "
                f"{self.max_catalogue_bytes}-byte size limit."
            )
        return response

    def _parse_response(self, body: bytes) -> MarketplaceCatalogue:
        if not body:
            raise MarketplaceFetchError("Marketplace returned an empty catalogue,")

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MarketplaceFetchError(
                "Marketplace catalogue must be UTF-8 encoded."
            ) from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MarketplaceFetchError(
                f"Marketplace catalogue is not valid JSON: {exc}"
            ) from exc

        try:
            catalogue = MarketplaceCatalogue.from_dict(data)
        except MarketplaceCatalogueError as exc:
            raise MarketplaceFetchError(
                f"Marketplace catalogue failed validation: {exc}"
            ) from exc

        _validate_catalogue_source(catalogue, self.source)
        return catalogue

class _HttpsOnlyRedirectHandler(HTTPRedirectHandler):
    """Reject catalogue redirects that leave HTTPS."""

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        parsed = urlparse(newurl)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise MarketplaceFetchError(
                f"Marketplace catalogue redirect is not HTTPS: {newurl!r}."
            )
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def _fetch_https(
    source: MarketplaceSource,
    headers: Mapping[str, str],
    timeout: float,
    max_catalogue_bytes: int,
) -> MarketplaceHttpResponse:
    request = Request(source.url, headers=dict(headers), method="GET")
    opener = build_opener(_HttpsOnlyRedirectHandler())

    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            parsed_final_url = urlparse(final_url)
            if (
                parsed_final_url.scheme.lower() != "https"
                or not parsed_final_url.netloc
            ):
                raise MarketplaceFetchError(
                    f"Marketplace catalogue final URL is not HTTPS: {final_url!r}."
                )

            status = int(getattr(response, "status", response.getcode()))
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_catalogue_bytes:
                    raise MarketplaceFetchError(
                        f"Marketplace catalogue declares {declared_size} bytes, "
                        f"above the {max_catalogue_bytes}-byte size limit."
                    )

            body = response.read(max_catalogue_bytes + 1)
            return MarketplaceHttpResponse(
                status=status,
                body=body,
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        if exc.code == 304:
            return MarketplaceHttpResponse(
                status=304,
                body=b"",
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        raise MarketplaceFetchError(
            f"Marketplace request failed with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise MarketplaceFetchError(
            f"Marketplace request failed: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise MarketplaceFetchError("Marketplace request timed out.") from exc

def _validate_catalogue_source(
    catalogue: MarketplaceCatalogue,
    source: MarketplaceSource,
) -> None:
    if catalogue.marketplace.id != source.id:
        raise MarketplaceCatalogueError(
            f"Marketplace source {source.id!r} returned catalogue id "
            f"{catalogue.marketplace.id!r}."
        )

def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return _optional_header(value)
    return None

def _optional_header(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None

def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))