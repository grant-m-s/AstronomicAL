from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles common numpy values."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def default_layout_directory(context: Any | None = None) -> Path:
    """
    Canonical user layout directory.

    context.config.layout_directory can override this, but by default layouts
    live beside user-installed plugins under ~/.astronomical/.
    """

    if context is not None:
        config = getattr(context, "config", None)
        configured = getattr(config, "layout_directory", None)
        if configured:
            return Path(configured).expanduser()

    return Path.home() / ".astronomical" / "layouts"


def _default_workspace_path(context: Any | None = None) -> Path:
    """
    Compatibility path for older save paths.

    New UI save actions should prefer timestamped/named saves, but keeping this
    helper avoids breaking any older code that still calls save_workspace()
    without a path.
    """

    if context is not None:
        config = getattr(context, "config", None)
        layout_file = getattr(config, "layout_file", None)
        if layout_file:
            return Path(layout_file).expanduser()

    return default_layout_directory(context) / "workspace.json"


def sanitize_layout_name(name: str) -> str:
    """
    Return a safe layout filename.

    Keeps letters, numbers, spaces, hyphens, underscores, and dots. Path
    separators and other special characters are converted to underscores.
    """

    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Layout name cannot be empty.")

    raw = raw.replace("\\", "/").split("/")[-1]
    raw = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw).strip(" ._")

    if not raw:
        raise ValueError("Layout name does not contain any valid filename characters.")

    if not raw.lower().endswith(".json"):
        raw = f"{raw}.json"

    return raw


def list_workspace_layouts(
    *,
    context: Any | None = None,
    directory: str | Path | None = None,
) -> list[Path]:
    """
    Return saved layout JSON files, newest first.
    """

    root = Path(directory).expanduser() if directory is not None else default_layout_directory(context)
    if not root.exists():
        return []

    paths = [path for path in root.glob("*.json") if path.is_file()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths


def save_workspace(
    context: Any,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Save the current plugin workspace through context.persistence.
    """

    if context is None:
        raise ValueError("save_workspace requires a non-null context.")

    persistence = getattr(context, "persistence", None)
    if persistence is None:
        raise RuntimeError("context.persistence is not configured.")

    workspace = getattr(context, "workspace", None)
    if workspace is not None and hasattr(workspace, "register_existing"):
        workspace.register_existing()

    target = Path(path).expanduser() if path is not None else _default_workspace_path(context)
    snapshot = persistence.save(target)

    panel_count = len(snapshot.get("workspace", {}).get("panels", []))
    grid_keys = snapshot.get("workspace", {}).get("grid", {}).get("keys", [])

    print(
        f"[save_workspace] saved {panel_count} persistent panels "
        f"with grid_keys={grid_keys} to {target}"
    )

    return snapshot


def save_workspace_timestamped(
    context: Any,
    directory: str | Path | None = None,
    prefix: str = "layout",
) -> Path:
    """
    Quick-save a timestamped workspace snapshot.

    This intentionally creates a new file every time.
    """

    root = Path(directory).expanduser() if directory is not None else default_layout_directory(context)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = root / f"{prefix}_{timestamp}.json"

    save_workspace(context, path)
    return path


def save_workspace_as(
    context: Any,
    name: str,
    directory: str | Path | None = None,
) -> Path:
    """
    Save the current workspace using a user-provided layout filename.
    """

    root = Path(directory).expanduser() if directory is not None else default_layout_directory(context)
    root.mkdir(parents=True, exist_ok=True)

    filename = sanitize_layout_name(name)
    path = root / filename

    save_workspace(context, path)
    return path


def load_workspace_file(path: str | Path) -> dict[str, Any]:
    """
    Small helper for reading a workspace JSON file directly.
    """

    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    if not isinstance(snapshot, dict):
        raise TypeError("Workspace JSON must contain an object at the top level.")

    return snapshot


def load_workspace_path(
    context: Any,
    path: str | Path,
    *,
    reconcile: bool = True,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """
    Load a workspace layout file into the running application.

    By default this uses the new reconcile flow rather than destructive
    clear-and-restore.
    """

    if context is None:
        raise ValueError("load_workspace_path requires a non-null context.")

    persistence = getattr(context, "persistence", None)
    if persistence is None:
        raise RuntimeError("context.persistence is not configured.")

    snapshot = persistence.load(path)

    if reconcile and hasattr(persistence, "reconcile"):
        return persistence.reconcile(snapshot, strict=strict)

    return persistence.restore(snapshot, strict=strict)


def save_dataframe_to_fits(
    df,
    filename: str | Path,
    overwrite: bool = True,
) -> None:
    """
    Export a dataframe to a FITS file.

    Kept here because header export still uses this helper for labelled data.
    """

    if len(df.columns) > 999:
        raise ValueError(
            "FITS files only allow up to 999 columns; "
            f"dataframe contains {len(df.columns)} columns."
        )

    filename = Path(filename).expanduser()
    filename.parent.mkdir(parents=True, exist_ok=True)

    table = Table.from_pandas(df)
    table.write(filename, overwrite=overwrite)