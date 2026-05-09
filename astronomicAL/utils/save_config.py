from __future__ import annotations

import json
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


def _default_workspace_path(context: Any | None = None) -> Path:
    if context is not None:
        config = getattr(context, "config", None)
        layout_file = getattr(config, "layout_file", None)
        if layout_file:
            return Path(layout_file).expanduser()

    return Path("configs/workspace.json")


def save_workspace(
    context: Any,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Save the current plugin workspace.

    This saves the platform workspace snapshot through context.persistence.
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
    directory: str | Path = "configs",
    prefix: str = "workspace",
) -> Path:
    """
    Save a timestamped workspace snapshot.

    Useful if you want the button to create a new file every time rather than
    overwrite config.layout_file.
    """
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{prefix}_{timestamp}.json"

    save_workspace(context, path)
    return path


def load_workspace_file(path: str | Path) -> dict[str, Any]:
    """
    Small helper for reading a workspace JSON file directly.
    """
    path = Path(path).expanduser()

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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