from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .artifacts import file_sha256, verify_file_ref
from .paths import ml_artifact_root, ml_run_artifact_dir
from .runtime import publish, put_artifact
from .serialization import json_safe

RESUME_SCHEMA_VERSION = 1
RUNTIME_RESUME_OVERRIDE_KEYS = {
    "device",
    "ml_artifact_dir",
    "artifact_dir",
    "prediction_output_dir",
    "save_predictions",
    "keep_work_dir",
    "keep_failed_work_dir",
    "resume_checkpoint_artifact_id",
    "resume_manifest_path",
    "trust_external_checkpoint",
}

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _assert_trusted_path(
    *,
    context: Any,
    path: str | Path,
    trusted_root: Optional[str | Path],
    allow_external: bool,
    label: str,
) -> Path:
    resolved = Path(str(path)).expanduser().resolve()

    if allow_external:
        return resolved

    root = (
        Path(str(trusted_root)).expanduser().resolve()
        if trusted_root
        else ml_artifact_root(context).resolve()
    )

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Refusing to load {label} outside the trusted ML "
            f"artifact root {root}: {resolved}. Set the explicit "
            "trust flag only for files from a trusted source."
        ) from exc

    return resolved


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def save_resume_checkpoint(
    *,
    run: Any,
    state: Mapping[str, Any],
    framework: str,
    completed_epoch: int,
) -> Tuple[Dict[str, Any], Path]:
    """Persist the full state needed to continue a managed recipe exactly.

    Torch checkpoints use torch.save so tensor/RNG/optimizer state is preserved.
    Sklearn checkpoints use joblib so fitted estimators and preprocessors remain
    intact. The adjacent JSON manifest is intentionally inspectable/importable.
    """

    framework = str(framework or "").lower()
    checkpoint_dir = ml_run_artifact_dir(run, kind="checkpoints")
    stem = f"paused-epoch-{int(completed_epoch):06d}"

    if framework == "sklearn":
        import joblib

        path = checkpoint_dir / f"{stem}.joblib"
        tmp = checkpoint_dir / f"{stem}.joblib.tmp"
        joblib.dump(dict(state), tmp)
        os.replace(tmp, path)
        fmt = "sklearn_resume_checkpoint"
    elif framework == "torch":
        import torch

        path = checkpoint_dir / f"{stem}.pt"
        tmp = checkpoint_dir / f"{stem}.pt.tmp"
        torch.save(dict(state), tmp)
        os.replace(tmp, path)
        fmt = "torch_resume_checkpoint"
    else:
        raise ValueError(f"Pause/resume is not supported for framework {framework!r}.")

    created_at = time.time()
    sha256 = file_sha256(path)
    ref = {
        "storage": "local_file",
        "uri": str(path),
        "path": str(path),
        "format": fmt,
        "framework": framework,
        "created_at": created_at,
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
        "metadata": {
            "schema_version": RESUME_SCHEMA_VERSION,
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
            "run_id": getattr(run, "run_id", None),
            "recipe_id": getattr(run, "recipe_id", None),
            "recipe_version": getattr(run, "recipe_version", None),
            "dataset_id": getattr(run, "dataset_id", None),
            "completed_epoch": int(completed_epoch),
            "resumable": True,
        },
    }

    manifest_path = checkpoint_dir / f"{stem}.resume_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "artifact_type": "ml.resume_checkpoint",
            "schema_version": RESUME_SCHEMA_VERSION,
            "status": "paused",
            "run_id": getattr(run, "run_id", None),
            "recipe_id": getattr(run, "recipe_id", None),
            "recipe_version": getattr(run, "recipe_version", None),
            "dataset_id": getattr(run, "dataset_id", None),
            "completed_epoch": int(completed_epoch),
            "created_at": created_at,
            "resume_ref": ref,
            "params": json_safe(getattr(run, "params", {}) or {}),
            "training_log_artifact_id": getattr(run, "training_log_artifact_id", None),
        },
    )
    return ref, manifest_path


def load_resume_checkpoint(
    *,
    context: Any,
    artifact_id: Optional[str] = None,
    manifest_path: Optional[str | Path] = None,
    sidecar_path: Optional[str | Path] = None,
    trusted_root: Optional[str | Path] = None,
    allow_external: bool = False,
) -> Dict[str, Any]:
    """Load a resumable checkpoint from an artifact, manifest, or direct sidecar."""

    ref: Dict[str, Any] = {}
    manifest: Dict[str, Any] = {}
    manifest_file: Optional[Path] = None

    if artifact_id:
        payload = context.artifacts.get(str(artifact_id))
        if not isinstance(payload, Mapping):
            raise TypeError(f"Resume artifact {artifact_id!r} does not contain a mapping payload.")
        manifest = dict(payload)
        ref = dict(payload.get("resume_ref") or payload.get("model_ref") or {})

    if manifest_path:
        manifest_file = _assert_trusted_path(
            context=context,
            path=manifest_path,
            trusted_root=trusted_root,
            allow_external=allow_external,
            label="resume manifest",
        )
        if not manifest_file.exists():
            raise FileNotFoundError(f"Resume manifest not found: {manifest_file}")
        loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Resume manifest {manifest_file} is not a JSON object.")
        manifest = dict(loaded)
        ref = dict(manifest.get("resume_ref") or manifest.get("model_ref") or ref)

    if sidecar_path:
        path = _assert_trusted_path(
            context=context,
            path=sidecar_path,
            trusted_root=trusted_root,
            allow_external=allow_external,
            label="resume checkpoint",
        )
        ref = {
            "storage": "local_file",
            "uri": str(path),
            "path": str(path),
            "format": (
                "sklearn_resume_checkpoint"
                if path.suffix == ".joblib"
                else "torch_resume_checkpoint"
            ),
        }

    if not ref:
        raise ValueError("No resume checkpoint reference was provided.")

    path = _resolve_ref_path(
        ref,
        relative_to=manifest_file.parent if manifest_file else None,
    )

    if not artifact_id:
        path = _assert_trusted_path(
            context=context,
            path=path,
            trusted_root=trusted_root,
            allow_external=allow_external,
            label="resume checkpoint",
        )

    verify_file_ref(ref, path)

    fmt = str(ref.get("format") or "").lower()

    if fmt == "sklearn_resume_checkpoint":
        import joblib

        state = joblib.load(path)
    elif fmt == "torch_resume_checkpoint":
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported resume checkpoint format {fmt!r}.")

    if not isinstance(state, Mapping):
        raise TypeError(f"Resume checkpoint {path} did not contain a mapping payload.")

    result = dict(state)
    result["_resume_ref"] = {**ref, "uri": str(path), "path": str(path)}
    result["_resume_manifest"] = manifest
    result["_resume_manifest_path"] = str(manifest_file) if manifest_file else None
    result["_resume_artifact_id"] = str(artifact_id) if artifact_id else None
    return result


def merge_resume_params(saved: Mapping[str, Any], requested: Mapping[str, Any]) -> Dict[str, Any]:
    """Use saved hyperparameters verbatim, allowing only execution-location overrides."""

    merged = dict(saved or {})
    for key in RUNTIME_RESUME_OVERRIDE_KEYS:
        if key in requested and requested.get(key) not in (None, ""):
            merged[key] = requested.get(key)
    return merged


def write_paused_model_manifest(path: Path, model_payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, model_payload)


def import_model_manifest(
    *,
    context: Any,
    manifest_path: str | Path,
    trusted_root: Optional[str | Path] = None,
    allow_external: bool = False,
) -> Dict[str, Any]:
    """Register a trusted on-disk model manifest in the ArtifactStore."""

    path = _assert_trusted_path(
        context=context,
        path=manifest_path,
        trusted_root=trusted_root,
        allow_external=allow_external,
        label="model manifest",
    )
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Model manifest {path} is not a JSON object.")

    payload = dict(loaded.get("model_payload") or loaded)
    if str(payload.get("artifact_type") or "ml.model") not in {"", "ml.model"}:
        raise ValueError(f"Manifest {path} is not an ml.model manifest.")

    payload["artifact_type"] = "ml.model"
    payload["imported_from_manifest"] = str(path)
    _resolve_payload_paths(payload, relative_to=path.parent)

    model_ref = payload.get("model_ref")
    if not isinstance(model_ref, Mapping):
        raise ValueError(f"Model manifest {path} is missing model_ref.")

    model_path = _resolve_ref_path(
        model_ref,
        relative_to=path.parent,
    )

    model_path = _assert_trusted_path(
        context=context,
        path=model_path,
        trusted_root=trusted_root,
        allow_external=allow_external,
        label="model sidecar",
    )

    verify_file_ref(model_ref, model_path)

    artifact_id = put_artifact(
        context,
        "ml.model",
        payload,
        dataset_id=payload.get("dataset_id") or payload.get("train_dataset_id"),
        params=payload.get("params") or {},
        persist=True,
        required=True,
    )
    publish(
        context,
        "ml.model.saved",
        {
            "artifact_id": artifact_id,
            "model_artifact_id": artifact_id,
            "manifest_path": str(path),
            "model_path": str(model_path),
            "imported": True,
        },
    )
    return {
        "status": "loaded",
        "artifact_id": artifact_id,
        "model_artifact_id": artifact_id,
        "manifest_path": str(path),
        "model_path": str(model_path),
        "payload": json_safe(payload),
    }


def discover_model_manifests(context: Any, params: Optional[Mapping[str, Any]] = None) -> list[str]:
    root = ml_artifact_root(context, params)
    patterns = ("**/*model_manifest.json",)
    found = {str(path.resolve()) for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    return sorted(found)


def discover_resume_manifests(context: Any, params: Optional[Mapping[str, Any]] = None) -> list[str]:
    root = ml_artifact_root(context, params)
    return sorted(str(path.resolve()) for path in root.glob("**/*.resume_manifest.json") if path.is_file())


def storage_locations(context: Any, params: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    root = ml_artifact_root(context, params)
    prediction_dir = Path(str((params or {}).get("prediction_output_dir") or root / "predictions")).expanduser()
    return {
        "artifact_root": str(root.resolve()),
        "models": str(root.resolve()),
        "checkpoints": str(root.resolve()),
        "predictions": str(prediction_dir.resolve()),
        "training_logs": str((root / "training_logs").resolve()),
    }


def _resolve_ref_path(ref: Mapping[str, Any], *, relative_to: Optional[Path]) -> Path:
    raw = ref.get("uri") or ref.get("path")
    if not raw:
        raise ValueError("File reference is missing uri/path.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Referenced file not found: {path}")
    return path


def _resolve_payload_paths(payload: Dict[str, Any], *, relative_to: Path) -> None:
    for key in ("model_ref", "resume_ref", "prediction_ref"):
        ref = payload.get(key)
        if not isinstance(ref, Mapping):
            continue
        mutable = dict(ref)
        raw = mutable.get("uri") or mutable.get("path")
        if raw:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = (relative_to / path).resolve()
            mutable["uri"] = str(path)
            mutable["path"] = str(path)
        payload[key] = mutable

    files = payload.get("files")
    if isinstance(files, Mapping):
        resolved = {}
        for key, value in files.items():
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (relative_to / path).resolve()
            resolved[str(key)] = str(path)
        payload["files"] = resolved


def load_model_manifest_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    from .runtime import coerce_action_request

    req = coerce_action_request(request)
    params = dict(req.params or {})
    path = params.get("manifest_path") or params.get("model_manifest_path")
    if not path:
        raise ValueError("load_model_manifest requires params.manifest_path.")

    return import_model_manifest(
        context=context,
        manifest_path=str(path),
        trusted_root=(
            params.get("ml_artifact_dir")
            or params.get("artifact_dir")
        ),
        allow_external=_as_bool(
            params.get("trust_external_files")
        ),
    )
