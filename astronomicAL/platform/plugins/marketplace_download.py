from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import os
import tempfile
from typing import Callable, Iterable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from .manifest import PluginManifest
from .marketplace import MarketplaceRelease


DEFAULT_PLUGIN_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PLUGIN_DOWNLOAD_BYTES = 512 * 1024 * 1024
DEFAULT_PLUGIN_DOWNLOAD_USER_AGENT = "AstronomicAL-Marketplace/1"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class MarketplaceDownloadError(RuntimeError):
    """Base error for marketplace package download/verification."""


class MarketplacePackageVerificationError(MarketplaceDownloadError):
    """Raised when downloaded package bytes do not match marketplace metadata."""


@dataclass(frozen=True)
class MarketplaceDownloadResponse:
    """Transport-neutral streamed response used by MarketplacePackageDownloader."""

    status: int
    headers: Mapping[str, str]
    chunks: Iterable[bytes]


@dataclass
class VerifiedMarketplacePackage:
    """Verified local archive ready to be handed to PluginInstaller."""

    path: Path
    release: MarketplaceRelease
    sha256: str
    manifest: PluginManifest
    byte_count: int
    _deleted: bool = False

    def cleanup(self) -> None:
        if self._deleted:
            return
        self._deleted = True
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "VerifiedMarketplacePackage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


MarketplacePackageInspector = Callable[[Path], PluginManifest]
MarketplacePackageFetcher = Callable[
    [str, Mapping[str, str], float, int],
    MarketplaceDownloadResponse,
]


class MarketplacePackageDownloader:
    """Download and verify one marketplace .alplugin release.

    This class intentionally does not install packages and does not implement
    archive parsing itself. ``inspect_package`` must use AstronomicAL's existing
    static .alplugin package inspection path and return the archive manifest.
    """

    def __init__(
        self,
        *,
        staging_dir: str | Path,
        inspect_package: MarketplacePackageInspector,
        fetcher: Optional[MarketplacePackageFetcher] = None,
        timeout: float = DEFAULT_PLUGIN_DOWNLOAD_TIMEOUT_SECONDS,
        max_download_bytes: int = DEFAULT_MAX_PLUGIN_DOWNLOAD_BYTES,
        user_agent: str = DEFAULT_PLUGIN_DOWNLOAD_USER_AGENT,
    ) -> None:
        if not callable(inspect_package):
            raise TypeError("inspect_package must be callable.")
        if float(timeout) <= 0:
            raise ValueError("Marketplace package timeout must be greater than zero.")
        if int(max_download_bytes) <= 0:
            raise ValueError(
                "Marketplace package max_download_bytes must be greater than zero."
            )

        self.staging_dir = Path(staging_dir).expanduser()
        self.inspect_package = inspect_package
        self.timeout = float(timeout)
        self.max_download_bytes = int(max_download_bytes)
        self.user_agent = str(
            user_agent or DEFAULT_PLUGIN_DOWNLOAD_USER_AGENT
        )
        self._fetcher = fetcher or _fetch_https_stream

    def download_and_verify(
        self,
        release: MarketplaceRelease,
        *,
        expected_plugin_id: Optional[str] = None,
    ) -> VerifiedMarketplacePackage:
        """Download a release to staging and verify all marketplace invariants.

        The returned archive remains a temporary staging file owned by the
        caller. Use it as a context manager or call ``cleanup()`` after the
        installer has consumed it.
        """

        self._validate_release_url(release.url)
        if release.size is not None and release.size > self.max_download_bytes:
            raise MarketplaceDownloadError(
                f"Marketplace package declares {release.size} bytes, above the "
                f"{self.max_download_bytes}-byte download limit."
            )

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix=".marketplace-",
            suffix=".alplugin",
            dir=self.staging_dir,
        )
        os.close(fd)
        path = Path(raw_path)

        try:
            sha256, byte_count = self._download_to_path(
                release,
                path,
            )
            self._verify_download_metadata(
                release,
                sha256=sha256,
                byte_count=byte_count,
            )

            try:
                manifest = self.inspect_package(path)
            except Exception as exc:
                raise MarketplacePackageVerificationError(
                    f"Downloaded package failed static .alplugin inspection: {exc}"
                ) from exc

            if not isinstance(manifest, PluginManifest):
                raise MarketplacePackageVerificationError(
                    "Package inspector must return PluginManifest."
                )

            self._verify_manifest(
                release,
                manifest=manifest,
                expected_plugin_id=expected_plugin_id,
            )

            return VerifiedMarketplacePackage(
                path=path,
                release=release,
                sha256=sha256,
                manifest=manifest,
                byte_count=byte_count,
            )
        except Exception:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _download_to_path(
        self,
        release: MarketplaceRelease,
        path: Path,
    ) -> tuple[str, int]:
        headers = {
            "Accept": "application/octet-stream",
            "User-Agent": self.user_agent,
        }

        try:
            response = self._fetcher(
                release.url,
                headers,
                self.timeout,
                self.max_download_bytes,
            )
        except MarketplaceDownloadError:
            raise
        except Exception as exc:
            raise MarketplaceDownloadError(
                f"Could not download marketplace package: {exc}"
            ) from exc

        if not isinstance(response, MarketplaceDownloadResponse):
            raise MarketplaceDownloadError(
                "Marketplace package fetcher returned an invalid response object."
            )
        if response.status != 200:
            raise MarketplaceDownloadError(
                f"Marketplace package returned unexpected HTTP status "
                f"{response.status}."
            )

        declared_length = _content_length(response.headers)
        if (
            declared_length is not None
            and declared_length > self.max_download_bytes
        ):
            raise MarketplaceDownloadError(
                f"Marketplace package declares {declared_length} bytes, above the "
                f"{self.max_download_bytes}-byte download limit."
            )

        digest = hashlib.sha256()
        byte_count = 0

        chunks = response.chunks
        try:
            with path.open("wb") as handle:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise MarketplaceDownloadError(
                            "Marketplace package fetcher yielded a non-bytes chunk."
                        )
                    if not chunk:
                        continue

                    byte_count += len(chunk)
                    if byte_count > self.max_download_bytes:
                        raise MarketplaceDownloadError(
                            f"Marketplace package exceeds the "
                            f"{self.max_download_bytes}-byte download limit."
                        )

                    handle.write(chunk)
                    digest.update(chunk)

                handle.flush()
                os.fsync(handle.fileno())
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()

        if declared_length is not None and byte_count != declared_length:
            raise MarketplaceDownloadError(
                "Marketplace package download length does not match HTTP "
                f"Content-Length: {byte_count} != {declared_length}."
            )

        return digest.hexdigest(), byte_count

    @staticmethod
    def _verify_download_metadata(
        release: MarketplaceRelease,
        *,
        sha256: str,
        byte_count: int,
    ) -> None:
        if not hmac.compare_digest(
            sha256.lower(),
            release.sha256.lower(),
        ):
            raise MarketplacePackageVerificationError(
                "Downloaded marketplace package SHA-256 does not match the "
                f"catalogue release: {sha256} != {release.sha256}."
            )

        if release.size is not None and byte_count != release.size:
            raise MarketplacePackageVerificationError(
                "Downloaded marketplace package size does not match the "
                f"catalogue release: {byte_count} != {release.size}."
            )

    @staticmethod
    def _verify_manifest(
        release: MarketplaceRelease,
        *,
        manifest: PluginManifest,
        expected_plugin_id: Optional[str],
    ) -> None:
        expected_id = (
            str(expected_plugin_id).strip()
            if expected_plugin_id is not None
            else release.manifest.id
        )

        if manifest.id != expected_id:
            raise MarketplacePackageVerificationError(
                "Downloaded package plugin id does not match the requested "
                f"plugin: {manifest.id!r} != {expected_id!r}."
            )

        if manifest.id != release.manifest.id:
            raise MarketplacePackageVerificationError(
                "Downloaded package plugin id does not match the marketplace "
                f"release: {manifest.id!r} != {release.manifest.id!r}."
            )

        if manifest.version != release.version:
            raise MarketplacePackageVerificationError(
                "Downloaded package version does not match the marketplace "
                f"release: {manifest.version!r} != {release.version!r}."
            )

        if manifest.to_dict() != release.manifest.to_dict():
            raise MarketplacePackageVerificationError(
                "Downloaded package manifest does not match the static manifest "
                "snapshot published in the marketplace catalogue,"
            )

    @staticmethod
    def _validate_release_url(url: str) -> None:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise MarketplaceDownloadError(
                "Marketplace package release URL must be an absolute HTTPS URL."
            )


class _HttpsOnlyRedirectHandler(HTTPRedirectHandler):
    """Reject redirects that leave HTTPS while allowing normal CDN redirects."""

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
            raise MarketplaceDownloadError(
                f"Marketplace package redirect is not HTTPS: {newurl!r}."
            )
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def _fetch_https_stream(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_download_bytes: int,
) -> MarketplaceDownloadResponse:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise MarketplaceDownloadError(
            "Marketplace package release URL must be an absolute HTTPS URL."
        )

    request = Request(url, headers=dict(headers), method="GET")
    opener = build_opener(_HttpsOnlyRedirectHandler())

    try:
        response = opener.open(request, timeout=timeout)
    except MarketplaceDownloadError:
        raise
    except HTTPError as exc:
        raise MarketplaceDownloadError(
            f"Marketplace package request failed with HTTP "
            f"{exc.code}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise MarketplaceDownloadError(
            f"Marketplace package request failed: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise MarketplaceDownloadError(
            "Marketplace package request timed out."
        ) from exc

    try:
        final_url = response.geturl()
        final_parsed = urlparse(final_url)
        if (
            final_parsed.scheme.lower() != "https"
            or not final_parsed.netloc
        ):
            raise MarketplaceDownloadError(
                f"Marketplace package final URL is not HTTPS: {final_url!r}."
            )

        status = int(getattr(response, "status", response.getcode()))
        response_headers = dict(response.headers.items())
        declared_length = _content_length(response_headers)
        if (
            declared_length is not None
            and declared_length > max_download_bytes
        ):
            raise MarketplaceDownloadError(
                f"Marketplace package declares {declared_length} bytes, "
                f"above the {max_download_bytes}-byte download limit."
            )
    except Exception:
        response.close()
        raise

    def iter_chunks():
        byte_count = 0
        try:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                byte_count += len(chunk)
                if byte_count > max_download_bytes:
                    raise MarketplaceDownloadError(
                        f"Marketplace package exceeds the "
                        f"{max_download_bytes}-byte download limit."
                    )
                yield chunk
        finally:
            response.close()

    return MarketplaceDownloadResponse(
        status=status,
        headers=response_headers,
        chunks=iter_chunks(),
    )

def _content_length(headers: Mapping[str, str]) -> Optional[int]:
    for key, value in headers.items():
        if str(key).lower() != "content-length":
            continue
        try:
            length = int(str(value).strip())
        except ValueError:
            return None
        return length if length >= 0 else None
    return None