# astronomicAL/plugins/core_ml/recipe_profiles.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
import time
import uuid


PROFILE_ARTIFACT_TYPE = "ml.recipe_profile"

PROTOCOL_KEYS = (
    "protocol_split_strategy",
    "protocol_validation_source",
    "protocol_validation_dataset_id",
    "protocol_test_source",
    "protocol_test_dataset_id",
    "protocol_group_column",
    "protocol_split_column",
    "protocol_validation_size",
    "protocol_test_size",
    "protocol_selection_metric",
    "protocol_random_state",
)

RUN_ONLY_KEYS = {
    "run_id",
    "dataset_id",
    "selection",
    "profile_id",
    "profile_artifact_id",
    "recipe_profile_id",
    "recipe_profile_artifact_id",
}


@dataclass(frozen=True)
class RecipeProfile:
    profile_id: str
    name: str
    recipe_id: str
    recipe_version: str = ""
    recipe_title: str = ""
    execution_mode: str = "freeform"
    task: str = ""
    modality: str = ""
    default_dataset_id: str = ""
    recipe_params: Dict[str, Any] = field(default_factory=dict)
    protocol_params: Dict[str, Any] = field(default_factory=dict)
    binding_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "name": self.name,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "recipe_title": self.recipe_title,
            "execution_mode": self.execution_mode,
            "task": self.task,
            "modality": self.modality,
            "default_dataset_id": self.default_dataset_id,
            "recipe_params": dict(self.recipe_params or {}),
            "protocol_params": dict(self.protocol_params or {}),
            "binding_params": dict(self.binding_params or {}),
            "notes": self.notes,
            "tags": list(self.tags or []),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_run_params(self, *, dataset_id: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        params.update(self.recipe_params or {})
        params.update(self.protocol_params or {})
        params.update(self.binding_params or {})
        params["recipe_id"] = self.recipe_id
        params["recipe_profile_id"] = self.profile_id
        params["dataset_id"] = dataset_id or self.default_dataset_id
        params.update(dict(overrides or {}))
        return params


def save_recipe_profile_action(context: Any, request: Any, *, cancel_token: Any = None) -> Dict[str, Any]:
    params = dict(getattr(request, "params", {}) or {})
    store = context.services.get("core.ml.recipe_profile_store")
    artifact_id = store.save(params)
    payload = store.get(params.get("profile_id") or artifact_id)
    return {
        "status": "complete",
        "artifact_id": artifact_id,
        "profile_id": payload.get("profile_id"),
        "profile": payload,
    }



class RecipeProfileStore:
    def __init__(self, context: Any) -> None:
        self.context = context

    def list(self, *, recipe_id: Optional[str] = None) -> List[Dict[str, Any]]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            return []

        refs = artifacts.find(type=PROFILE_ARTIFACT_TYPE)
        latest_by_profile: Dict[str, Dict[str, Any]] = {}

        for ref in refs:
            try:
                payload = artifacts.get(ref.artifact_id)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue
            if recipe_id and payload.get("recipe_id") != recipe_id:
                continue

            profile_id = str(payload.get("profile_id") or ref.artifact_id)
            existing = latest_by_profile.get(profile_id)
            if existing is None or float(payload.get("updated_at") or 0) > float(existing.get("updated_at") or 0):
                item = dict(payload)
                item["artifact_id"] = ref.artifact_id
                latest_by_profile[profile_id] = item

        return sorted(
            latest_by_profile.values(),
            key=lambda item: str(item.get("name") or item.get("profile_id") or "").lower(),
        )

    def get(self, profile_id_or_artifact_id: str) -> Dict[str, Any]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            raise RuntimeError("Artifact store is not available.")

        key = str(profile_id_or_artifact_id or "").strip()
        if not key:
            raise ValueError("Missing recipe profile id.")

        try:
            payload = artifacts.get(key)
            if isinstance(payload, dict) and payload.get("recipe_id"):
                payload = dict(payload)
                payload["artifact_id"] = key
                return payload
        except Exception:
            pass

        matches = [
            item for item in self.list()
            if str(item.get("profile_id") or "") == key
        ]
        if not matches:
            raise KeyError(f"Unknown recipe profile: {key}")

        return matches[0]

    def save(self, payload: Mapping[str, Any]) -> str:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            raise RuntimeError("Artifact store is not available.")

        now = time.time()
        profile_id = str(payload.get("profile_id") or uuid.uuid4().hex)
        body = dict(payload)
        body.setdefault("schema_version", 1)
        body["profile_id"] = profile_id
        body["created_at"] = float(body.get("created_at") or now)
        body["updated_at"] = now

        artifact_id = artifacts.put(
            PROFILE_ARTIFACT_TYPE,
            body,
            dataset_id=str(body.get("default_dataset_id") or body.get("dataset_id") or "default"),
            params={
                "profile_id": profile_id,
                "recipe_id": body.get("recipe_id"),
            },
            persist=True,
        )

        events = getattr(self.context, "events", None)
        publish = getattr(events, "publish", None)
        if callable(publish):
            publish("ml.recipe_profile.saved", {
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "recipe_id": body.get("recipe_id"),
                "name": body.get("name"),
            })
            publish("ml.recipe_profiles.changed", {"profile_id": profile_id, "artifact_id": artifact_id})

        return artifact_id


def create_recipe_profile_store(context: Any) -> RecipeProfileStore:
    return RecipeProfileStore(context)


def split_profile_params(params: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    recipe_params: Dict[str, Any] = {}
    protocol_params: Dict[str, Any] = {}
    binding_params: Dict[str, Any] = {}

    for key, value in dict(params or {}).items():
        if key in RUN_ONLY_KEYS:
            continue
        if key in PROTOCOL_KEYS:
            protocol_params[key] = value
        elif key in {
            "record_id_column",
            "target_column",
            "label_column",
            "class_column",
            "image_column",
            "image_path_column",
            "image_uri_column",
            "feature_columns",
            "input_columns",
            "auto_feature_columns",
        }:
            binding_params[key] = value
        else:
            recipe_params[key] = value

    return recipe_params, protocol_params, binding_params