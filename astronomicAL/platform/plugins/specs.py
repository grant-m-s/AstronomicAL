from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from astronomicAL.platform.mapping_requirements import MappingRequirementLike

class PluginStatus(str, Enum):
    DISCOVERED = "discovered"
    DISABLED = "disabled"
    ENABLED = "enabled"
    ERROR = "error"


class SelectionRequirement(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class ColumnRequirement(str, Enum):
    NONE = "none"
    ONE = "one"
    MANY = "many"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        return cls(ok=True, warnings=warnings or [])

    @classmethod
    def failure(
        cls,
        errors: Sequence[str],
        *,
        missing_dependencies: Optional[Sequence[str]] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> "ValidationResult":
        return cls(
            ok=False,
            errors=list(errors),
            warnings=list(warnings or []),
            missing_dependencies=list(missing_dependencies or []),
        )


@dataclass(frozen=True)
class InputSpec:
    """Declarative inputs for a plugin action.

    A UI layer can use this to auto-render dataset selectors, column selectors,
    row-selection toggles, artifact pickers, parameter forms, and semantic
    column-mapping requests.
    """

    dataset: bool = True
    selection: str = SelectionRequirement.OPTIONAL.value
    numeric_columns: str = ColumnRequirement.NONE.value
    columns: str = ColumnRequirement.OPTIONAL.value
    required_mappings: List[MappingRequirementLike] = field(default_factory=list)
    optional_mappings: List[MappingRequirementLike] = field(default_factory=list)
    accepts_artifact_types: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_choice("selection", self.selection, SelectionRequirement)
        _validate_choice("numeric_columns", self.numeric_columns, ColumnRequirement)
        _validate_choice("columns", self.columns, ColumnRequirement)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputSpec":
        return cls(**data)


@dataclass(frozen=True)
class OutputSpec:
    type: str
    optional: bool = False
    description: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "OutputSpec":
        if isinstance(value, OutputSpec):
            return value
        if isinstance(value, str):
            return cls(type=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Cannot convert {value!r} to OutputSpec")


@dataclass
class ActionRequest:
    """Runtime request passed to registered action handlers."""

    dataset_id: Optional[str] = None
    row_ids: Optional[List[Any]] = None
    columns: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    artifact_id: Optional[str] = None
    origin: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ActionRequest":
        if not data:
            return cls()
        return cls(**data)


@dataclass
class ArtifactResult:
    type: str
    payload: Any
    dataset_id: Optional[str] = None
    row_ids: Optional[List[Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    publish: bool = True
    artifact_id: Optional[str] = None


@dataclass
class DatasetResult:
    id: str
    dataframe: Any
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    set_active: bool = False


@dataclass
class EventResult:
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    value: Any = None
    artifacts: List[ArtifactResult] = field(default_factory=list)
    datasets: List[DatasetResult] = field(default_factory=list)
    events: List[EventResult] = field(default_factory=list)


@dataclass
class ProcessedActionResult:
    """Result after the platform has stored datasets/artifacts and published events.

    ``value`` is the plugin's direct return value when one exists. ``artifact_ids``
    and ``dataset_ids`` are the platform identifiers produced while processing the
    action result. ``raw`` preserves the handler's original return value for
    debugging or advanced callers.
    """

    raw: Any = None
    value: Any = None
    artifact_ids: List[str] = field(default_factory=list)
    dataset_ids: List[str] = field(default_factory=list)
    events: List[EventResult] = field(default_factory=list)
    result: Optional[ActionResult] = None

    def legacy_return(self) -> Any:
        """Return the pre-processed API shape used by early callers."""

        if self.value is not None:
            return self.value
        if self.artifact_ids:
            return self.artifact_ids[0] if len(self.artifact_ids) == 1 else list(self.artifact_ids)
        if self.dataset_ids:
            return self.dataset_ids[0] if len(self.dataset_ids) == 1 else list(self.dataset_ids)
        if self.result is not None:
            return self.result
        return self.raw


@dataclass
class PluginRegistration:
    plugin_id: str
    id: str
    title: str
    description: str = ""
    category: Optional[str] = None
    icon: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class PanelRegistration(PluginRegistration):
    factory: Callable[..., Any] = None  # type: ignore[assignment]

    required_mappings: List[MappingRequirementLike] = field(default_factory=list)
    optional_mappings: List[MappingRequirementLike] = field(default_factory=list)

    uses_services: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)

    default_layout: Optional[Dict[str, Any]] = None
    default_open_kwargs: Dict[str, Any] = field(default_factory=dict)

    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)

    state_version: int = 1
    persist_layout: bool = True
    persist_state: bool = True
    restore_policy: str = "best_effort"

@dataclass
class CreatedPanel:
    view: Any
    controller: Any
    registration: PanelRegistration
    instance_id: str
    title: str

@dataclass
class ActionRegistration(PluginRegistration):
    handler: Callable[..., Any] = None  # type: ignore[assignment]
    inputs: InputSpec = field(default_factory=InputSpec)
    outputs: List[OutputSpec] = field(default_factory=list)
    params_schema: Dict[str, Any] = field(default_factory=dict)
    settings_schema: Dict[str, Any] = field(default_factory=dict)
    run_in_job: bool = True
    key_fn: Optional[Callable[[ActionRequest], str]] = None
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)


@dataclass
class WorkflowRegistration(PluginRegistration):
    builder: Callable[..., Any] = None  # type: ignore[assignment]
    settings_schema: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)


@dataclass
class ServiceRegistration:
    plugin_id: str
    key: str
    factory: Callable[..., Any]
    lazy: bool = True
    replace: bool = False
    description: str = ""
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)


@dataclass
class ArtifactViewerRegistration:
    plugin_id: str
    artifact_type: str
    viewer_factory: Callable[..., Any]
    id: Optional[str] = None
    title: Optional[str] = None
    description: str = ""
    priority: int = 100
    default: bool = False
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    id: str
    name: str
    version: str
    status: PluginStatus
    description: str = ""
    source: str = ""
    path: Optional[str] = None
    error: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    optional_requires: List[str] = field(default_factory=list)
    requires_plugins: List[str] = field(default_factory=list)
    panels: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    workflows: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    artifact_viewers: List[str] = field(default_factory=list)


def _validate_choice(name: str, value: str, enum_cls: type[Enum]) -> None:
    allowed = {item.value for item in enum_cls}
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")