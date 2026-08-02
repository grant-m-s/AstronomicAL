from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Dict, Iterable, Tuple
import zipfile

from .errors import PluginPackageError
from .manifest import PluginManifest, coerce_manifest


ALPLUGIN_SUFFIX = ".alplugin"
MANIFEST_NAME = "astronomical-plugin.json"
ENTRYPOINT_NAME = "plugin.py"

MAX_ARCHIVE_SIZE = 64 * 1024 * 1024
MAX_MEMBER_COUNT = 2000
MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_UNCOMPRESSED_SIZE = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250.0
MAX_MANIFEST_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PluginPackageInspection:
    """Static, non-executing description of an .alplugin archive."""

    archive_path: Path
    manifest: PluginManifest
    sha256: str
    members: Tuple[str, ...]
    file_count: int
    total_uncompressed_size: int


def inspect_plugin_package(
    archive_path: str | Path,
) -> PluginPackageInspection:
    """Validate an .alplugin ZIP without importing plugin runtime code."""

    path = Path(archive_path).expanduser()
    if path.suffix.lower() != ALPLUGIN_SUFFIX:
        raise PluginPackageError(
            f"Plugin package must use the {ALPLUGIN_SUFFIX} extension: {path.name}"
        )
    if not path.is_file():
        raise PluginPackageError(f"Plugin package does not exist: {path}")

    archive_size = path.stat().st_size
    if archive_size > MAX_ARCHIVE_SIZE:
        raise PluginPackageError(
            f"Plugin package is too large ({archive_size} bytes; "
            f"maximum is {MAX_ARCHIVE_SIZE})."
        )

    sha256 = _sha256_file(path)

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            validated = _validate_members(infos)

            names = {name for _, name in validated}
            if MANIFEST_NAME not in names:
                raise PluginPackageError(
                    f"Plugin package must contain {MANIFEST_NAME!r} at the archive root."
                )
            if ENTRYPOINT_NAME not in names:
                raise PluginPackageError(
                    f"Plugin package must contain {ENTRYPOINT_NAME!r} at the archive root."
                )

            manifest_info = next(
                info for info, name in validated if name == MANIFEST_NAME
            )
            if manifest_info.file_size > MAX_MANIFEST_SIZE:
                raise PluginPackageError(
                    f"{MANIFEST_NAME} exceeds the maximum manifest size."
                )

            try:
                manifest_bytes = archive.read(manifest_info)
            except Exception as exc:
                raise PluginPackageError(
                    f"Could not read {MANIFEST_NAME}: {exc}"
                ) from exc

    except PluginPackageError:
        raise
    except zipfile.BadZipFile as exc:
        raise PluginPackageError(f"Invalid .alplugin ZIP archive: {exc}") from exc
    except Exception as exc:
        raise PluginPackageError(f"Could not inspect plugin package: {exc}") from exc

    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PluginPackageError(
            f"{MANIFEST_NAME} must be UTF-8 encoded."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PluginPackageError(
            f"{MANIFEST_NAME} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(manifest_data, dict):
        raise PluginPackageError(f"{MANIFEST_NAME} root must be a JSON object.")

    manifest_data = manifest_data.get(
        "plugin",
        manifest_data.get("astronomical", manifest_data),
    )
    try:
        manifest = coerce_manifest(manifest_data)
    except Exception as exc:
        raise PluginPackageError(
            f"Invalid plugin manifest in {MANIFEST_NAME}: {exc}"
        ) from exc

    file_names = tuple(
        name
        for info, name in validated
        if not info.is_dir()
    )
    total_uncompressed = sum(
        info.file_size
        for info, _ in validated
        if not info.is_dir()
    )

    return PluginPackageInspection(
        archive_path=path,
        manifest=manifest,
        sha256=sha256,
        members=file_names,
        file_count=len(file_names),
        total_uncompressed_size=total_uncompressed,
    )


def extract_plugin_package(
    inspection: PluginPackageInspection,
    destination: str | Path,
) -> Path:
    """Safely extract a previously inspected package into a new directory."""

    if not isinstance(inspection, PluginPackageInspection):
        raise TypeError("inspection must be a PluginPackageInspection.")

    destination_path = Path(destination).expanduser()
    if destination_path.exists():
        raise PluginPackageError(
            f"Plugin staging destination already exists: {destination_path}"
        )

    current_sha256 = _sha256_file(inspection.archive_path)
    if current_sha256 != inspection.sha256:
        raise PluginPackageError(
            "Plugin package changed after inspection; refusing to extract it."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.mkdir()

    try:
        with zipfile.ZipFile(inspection.archive_path, "r") as archive:
            validated = _validate_members(archive.infolist())

            for info, relative_name in validated:
                target = destination_path / PurePosixPath(relative_name)

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)

        manifest_path = destination_path / MANIFEST_NAME
        entrypoint_path = destination_path / ENTRYPOINT_NAME
        if not manifest_path.is_file() or not entrypoint_path.is_file():
            raise PluginPackageError(
                "Extracted plugin package is missing its manifest or plugin.py."
            )

        return destination_path

    except Exception:
        shutil.rmtree(destination_path, ignore_errors=True)
        raise


def _validate_members(
    infos: Iterable[zipfile.ZipInfo],
) -> list[tuple[zipfile.ZipInfo, str]]:
    infos = list(infos)
    if len(infos) > MAX_MEMBER_COUNT:
        raise PluginPackageError(
            f"Plugin package contains too many entries ({len(infos)}; "
            f"maximum is {MAX_MEMBER_COUNT})."
        )

    validated: list[tuple[zipfile.ZipInfo, str]] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    total_uncompressed = 0

    for info in infos:
        if info.flag_bits & 0x1:
            raise PluginPackageError(
                f"Encrypted ZIP entries are not supported: {info.filename!r}"
            )

        normalised = _normalise_member_name(info.filename, is_dir=info.is_dir())

        folded = normalised.casefold()
        if normalised in seen or folded in seen_casefolded:
            raise PluginPackageError(
                f"Plugin package contains a duplicate path: {normalised!r}"
            )
        seen.add(normalised)
        seen_casefolded.add(folded)

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode:
            file_type = stat.S_IFMT(unix_mode)
            if file_type == stat.S_IFLNK:
                raise PluginPackageError(
                    f"Plugin packages may not contain symbolic links: {normalised!r}"
                )
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise PluginPackageError(
                    f"Plugin package contains a non-regular file: {normalised!r}"
                )

        if not info.is_dir():
            if info.file_size > MAX_MEMBER_SIZE:
                raise PluginPackageError(
                    f"Plugin package member is too large: {normalised!r}"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
                raise PluginPackageError(
                    "Plugin package expands beyond the maximum allowed size."
                )

            if info.file_size >= 1024 * 1024 and info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise PluginPackageError(
                        f"Suspicious compression ratio for {normalised!r}."
                    )

        validated.append((info, normalised))

    return validated


def _normalise_member_name(name: str, *, is_dir: bool) -> str:
    if not isinstance(name, str) or not name:
        raise PluginPackageError("Plugin package contains an empty ZIP path.")
    if "\x00" in name:
        raise PluginPackageError("Plugin package contains a NUL byte in a path.")
    if "\\" in name:
        raise PluginPackageError(
            f"Plugin package paths must use '/' separators: {name!r}"
        )
    if name.startswith("/"):
        raise PluginPackageError(
            f"Plugin package contains an absolute path: {name!r}"
        )

    raw = name[:-1] if is_dir and name.endswith("/") else name
    if not raw:
        raise PluginPackageError("Plugin package contains an invalid root directory entry.")

    raw_parts = raw.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise PluginPackageError(
            f"Plugin package contains an unsafe path: {name!r}"
        )
    if any(":" in part for part in raw_parts):
        raise PluginPackageError(
            f"Plugin package contains a platform-unsafe ':' path component: {name!r}"
        )

    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or ".." in path.parts:
        raise PluginPackageError(
            f"Plugin package contains an unsafe path: {name!r}"
        )

    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()