from __future__ import annotations

import importlib.util
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .data.dataset_access import infer_column_bindings
from .resource_estimates import RecipeDataAccess

def schema_defaults(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract default values from a JSON-schema-like parameter schema."""
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    defaults: Dict[str, Any] = {}
    for name, spec in properties.items():
        if isinstance(spec, Mapping) and "default" in spec:
            defaults[str(name)] = spec["default"]
    return defaults

def validate_required_params(schema: Mapping[str, Any], params: Mapping[str, Any]) -> None:
    required = list(schema.get("required", []) or []) if isinstance(schema, Mapping) else []
    missing = []
    for key in required:
        value = params.get(key)
        if value is None or value == "":
            missing.append(str(key))
    if missing:
        raise ValueError(f"Missing required recipe parameter(s): {', '.join(missing)}")

def infer_recipe_params(context: Any, dataset_id: str, recipe_spec: Any) -> Dict[str, Any]:
    """Return inferred values only for parameters exposed by the recipe schema."""
    bindings = infer_column_bindings(context, dataset_id)
    schema = getattr(recipe_spec, "params_schema", None) or {}
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}

    inferred: Dict[str, Any] = {}
    for key in (
        "record_id_column",
        "image_column",
        "target_column",
        "mask_column",
        "feature_columns",
    ):
        if key in properties and key in bindings:
            inferred[key] = bindings[key]

    alias_map = {
        "label_column": "target_column",
        "class_column": "target_column",
        "image_path_column": "image_column",
        "image_uri_column": "image_column",
        "features": "feature_columns",
    }
    for alias, source_key in alias_map.items():
        if alias in properties and source_key in bindings:
            inferred[alias] = bindings[source_key]

    return inferred

def is_empty_param_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}

class RecipeUnavailableError(RuntimeError):
    """Raised when a registered recipe cannot run in the current environment."""

@dataclass(frozen=True)
class RecipeAvailability:
    recipe_id: str
    available: bool
    required_imports: List[str]
    missing_imports: List[str]
    checked_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def reason(self) -> str:
        if self.available:
            return ""
        return "Missing Python import(s): " + ", ".join(self.missing_imports)

@dataclass(frozen=True)
class RecipeSpec:
    recipe_cls: type
    id: str
    title: str
    version: str
    task: str
    modality: str
    framework: str
    complexity: str
    author: str
    description: str
    tags: List[str]
    required_mappings: List[str]
    optional_mappings: List[str]
    produces: List[str]
    params_schema: Dict[str, Any]
    required_imports: List[str]
    data_access: RecipeDataAccess

    @classmethod
    def from_recipe_cls(cls, recipe_cls: type) -> "RecipeSpec":
        if not isinstance(recipe_cls, type):
            raise TypeError(f"RecipeSpec expects a recipe class, got {recipe_cls!r}.")

        def g(name: str, default: Any) -> Any:
            return getattr(recipe_cls, name, default)

        rid = str(g("id", "") or "").strip()
        if not rid:
            raise ValueError(f"Recipe {recipe_cls.__name__} has no `id`; cannot register it.")

        return cls(
            recipe_cls=recipe_cls,
            id=rid,
            title=str(g("title", "") or rid),
            version=str(g("version", "0.0.0")),
            task=str(g("task", "custom")),
            modality=str(g("modality", "custom")),
            framework=str(g("framework", "") or ""),
            complexity=str(g("complexity", "expert")),
            author=str(g("author", "")),
            description=str(g("description", "")),
            tags=list(g("tags", []) or []),
            required_mappings=list(g("required_mappings", []) or []),
            optional_mappings=list(g("optional_mappings", []) or []),
            produces=list(g("produces", []) or []),
            params_schema=dict(g("params_schema", {"type": "object", "properties": {}}) or {}),
            required_imports=[str(value).strip() for value in (g("required_imports", []) or []) if str(value).strip()],
            data_access=RecipeDataAccess.coerce(
                g("data_access", None),
                framework=str(g("framework", "") or ""),
            ),
        )

class MLRecipeRegistry:
    """In-memory registry of recipe specs, keyed by recipe id.

    Recipes remain registered even when optional frameworks are not installed,
    so profiles and persisted runs can still be inspected. UI callers should use
    ``list_available()`` and execution callers should use ``require_available()``.
    """

    def __init__(self, recipes: Optional[Sequence[type]] = None) -> None:
        self._specs: Dict[str, RecipeSpec] = {}
        self._availability: Dict[str, RecipeAvailability] = {}
        self._unavailable_notice_consumed = False
        if recipes:
            self.register_many(recipes)

    def register(self, recipe_cls: type, *, replace: bool = True) -> RecipeSpec:
        from .recipe_base import ManagedMLRecipe

        if not isinstance(recipe_cls, type) or not issubclass(recipe_cls, ManagedMLRecipe):
            raise TypeError(f"{recipe_cls!r} must be a ManagedMLRecipe subclass.")

        spec = RecipeSpec.from_recipe_cls(recipe_cls)
        if spec.id in self._specs and not replace:
            raise ValueError(f"Recipe id {spec.id!r} is already registered.")

        self._specs[spec.id] = spec
        self._availability[spec.id] = self._check_availability(spec)
        return spec

    def register_many(self, recipe_classes: Sequence[type]) -> None:
        for recipe_cls in recipe_classes:
            self.register(recipe_cls)

    def unregister(self, recipe_id: str) -> None:
        key = str(recipe_id)
        self._specs.pop(key, None)
        self._availability.pop(key, None)

    def get(self, recipe_id: str) -> RecipeSpec:
        try:
            return self._specs[str(recipe_id)]
        except KeyError:
            raise KeyError(
                f"No recipe registered with id {recipe_id!r}. Known recipe ids: {sorted(self._specs)}"
            ) from None

    def require_available(self, recipe_id: str) -> RecipeSpec:
        spec = self.get(recipe_id)
        availability = self.availability(spec.id)
        if not availability.available:
            raise RecipeUnavailableError(
                f"Recipe {spec.title!r} is unavailable. {availability.reason}. "
                "Install the missing optional dependency bundle and refresh the recipe list."
            )
        return spec

    def availability(self, recipe_id: str) -> RecipeAvailability:
        key = str(recipe_id)
        if key not in self._specs:
            self.get(key)
        if key not in self._availability:
            self._availability[key] = self._check_availability(self._specs[key])
        return self._availability[key]

    def refresh_availability(self) -> Dict[str, RecipeAvailability]:
        self._availability = {
            recipe_id: self._check_availability(spec)
            for recipe_id, spec in self._specs.items()
        }
        return dict(self._availability)

    def list(self) -> List[RecipeSpec]:
        return list(self._specs.values())

    def list_available(self) -> List[RecipeSpec]:
        return [spec for spec in self._specs.values() if self.availability(spec.id).available]

    def list_unavailable(self) -> List[tuple[RecipeSpec, RecipeAvailability]]:
        return [
            (spec, self.availability(spec.id))
            for spec in self._specs.values()
            if not self.availability(spec.id).available
        ]

    def consume_unavailable_notice(self) -> List[tuple[RecipeSpec, RecipeAvailability]]:
        """Return unavailable recipes once per registry/application lifetime."""
        if self._unavailable_notice_consumed:
            return []
        unavailable = self.list_unavailable()
        if unavailable:
            self._unavailable_notice_consumed = True
        return unavailable

    def ids(self) -> List[str]:
        return list(self._specs.keys())

    def __contains__(self, recipe_id: object) -> bool:
        return str(recipe_id) in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    @staticmethod
    def _check_availability(spec: RecipeSpec) -> RecipeAvailability:
        missing = [name for name in spec.required_imports if not _import_available(name)]
        return RecipeAvailability(
            recipe_id=spec.id,
            available=not missing,
            required_imports=list(spec.required_imports),
            missing_imports=missing,
            checked_at=time.time(),
        )

def _import_available(import_name: str) -> bool:
    """Check import presence without importing heavy ML frameworks."""
    name = str(import_name or "").strip()
    if not name:
        return True
    root = name.split(".", 1)[0]
    try:
        importlib.invalidate_caches()
        return importlib.util.find_spec(root) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False