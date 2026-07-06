from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .data.dataset_access import infer_column_bindings

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

def infer_recipe_params(
    context: Any,
    dataset_id: str,
    recipe_spec: Any,
) -> Dict[str, Any]:
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

    # Common aliases used by some recipe/action schemas.
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
    execution_mode: str

    @classmethod
    def from_recipe_cls(cls, recipe_cls: type) -> "RecipeSpec":
        if not isinstance(recipe_cls, type):
            raise TypeError(
                f"RecipeSpec expects a recipe class, got {recipe_cls!r}."
            )

        def g(name: str, default: Any) -> Any:
            return getattr(recipe_cls, name, default)

        rid = str(g("id", "") or "").strip()
        if not rid:
            raise ValueError(
                f"Recipe {recipe_cls.__name__} has no `id`; cannot register it."
            )

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
            params_schema=dict(
                g("params_schema", {"type": "object", "properties": {}}) or {}
            ),
            execution_mode=str(g("execution_mode", "freeform") or "freeform"),
        )

class MLRecipeRegistry:
    """In-memory registry of recipe specs, keyed by recipe id.

    Registered as the `core.ml.recipe_registry` service by the plugin's
    register(). `list()` preserves registration order, so the first recipe the
    plugin registers becomes the launcher's default selection.
    """

    def __init__(self, recipes: Optional[Sequence[type]] = None) -> None:
        self._specs: Dict[str, RecipeSpec] = {}
        if recipes:
            self.register_many(recipes)

    def register(self, recipe_cls: type, *, replace: bool = True) -> RecipeSpec:
        spec = RecipeSpec.from_recipe_cls(recipe_cls)
        if spec.id in self._specs and not replace:
            raise ValueError(f"Recipe id {spec.id!r} is already registered.")
        self._specs[spec.id] = spec
        return spec

    def register_many(self, recipe_classes: Sequence[type]) -> None:
        for recipe_cls in recipe_classes:
            self.register(recipe_cls)

    def unregister(self, recipe_id: str) -> None:
        self._specs.pop(str(recipe_id), None)

    def get(self, recipe_id: str) -> RecipeSpec:
        try:
            return self._specs[str(recipe_id)]
        except KeyError:
            raise KeyError(
                f"No recipe registered with id {recipe_id!r}. "
                f"Known recipe ids: {sorted(self._specs)}"
            ) from None

    def list(self) -> List[RecipeSpec]:
        return list(self._specs.values())

    def ids(self) -> List[str]:
        return list(self._specs.keys())

    def __contains__(self, recipe_id: object) -> bool:
        return str(recipe_id) in self._specs

    def __len__(self) -> int:
        return len(self._specs)
