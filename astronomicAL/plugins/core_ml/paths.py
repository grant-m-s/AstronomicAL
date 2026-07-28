from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

def _safe_path_part(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")

    return text or fallback

def ml_artifact_root(
    context: Any,
    params: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Stable local root for ML sidecar files.

    Do NOT use run.work_dir for durable trained models. run.work_dir is temp.

    Priority:
        1. params["ml_artifact_dir"]
        2. params["artifact_dir"]
        3. env ASTRONOMICAL_ML_ARTIFACT_DIR
        4. context persistence-ish directory, if discoverable
        5. ~/.astronomical/ml_artifacts
    """

    params = dict(params or {})

    explicit = (
        params.get("ml_artifact_dir")
        or params.get("artifact_dir")
        or os.environ.get("ASTRONOMICAL_ML_ARTIFACT_DIR")
    )

    if explicit:
        root = Path(str(explicit)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root

    persistence = getattr(context, "persistence", None)

    for attr in (
        "root_dir",
        "base_dir",
        "workspace_dir",
        "artifact_dir",
        "path",
    ):
        value = getattr(persistence, attr, None)
        if value:
            root = Path(str(value)).expanduser() / "ml_artifacts"
            root.mkdir(parents=True, exist_ok=True)
            return root

    root = Path.home() / ".astronomical" / "ml_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root

def ml_run_artifact_dir(
    run: Any,
    *,
    kind: str,
) -> Path:
    """Stable directory for files belonging to one ML recipe run."""

    recipe_id = _safe_path_part(
        getattr(run, "recipe_id", None) or run.params.get("recipe_id"),
        fallback="recipe",
    )
    run_id = _safe_path_part(
        getattr(run, "run_id", None) or run.params.get("run_id"),
        fallback=uuid.uuid4().hex,
    )
    kind = _safe_path_part(kind, fallback="artifact")

    root = ml_artifact_root(
        getattr(run, "context", None),
        getattr(run, "params", {}),
    )

    path = root / recipe_id / run_id / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
