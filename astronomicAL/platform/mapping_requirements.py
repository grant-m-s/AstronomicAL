from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from astronomicAL.utils.debug import mapping_debug_print

_COMMON_ALIASES: Dict[str, List[str]] = {
    "record_id": [
        "source_id",
        "sourceid",
        "object_id",
        "objid",
        "id",
        "ID",
        "row_id",
        "name",
    ],
    "coords.ra": [
        "ra",
        "RA",
        "raj2000",
        "ra_j2000",
        "ra_deg",
        "right_ascension",
        "alpha",
    ],
    "coords.dec": [
        "dec",
        "DEC",
        "dej2000",
        "dec_j2000",
        "dec_deg",
        "declination",
        "delta",
    ],
    "target_label": [
        "label",
        "labels",
        "class",
        "classification",
        "target",
        "y",
    ],
}


@dataclass(frozen=True)
class MappingRequirement:
    """Declarative semantic-column requirement.

    Plugin authors should describe the semantic column they need, not
    manually call MappingAlertController or DatasetManager.

    ``required=True`` blocks panel creation until mapped.
    ``required=False`` requests the mapping but allows the panel to open.
    """

    semantic_name: str
    display_name: Optional[str] = None
    description: str = ""
    required: bool = True
    config_key: Optional[str] = None
    candidates: Optional[List[str]] = None
    suggested: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    allow_index: bool = False

    @classmethod
    def from_any(
        cls,
        value: MappingRequirementLike,
        *,
        required_default: bool,
    ) -> "MappingRequirement":
        if isinstance(value, MappingRequirement):
            if value.required == required_default:
                return value
            return cls(
                semantic_name=value.semantic_name,
                display_name=value.display_name,
                description=value.description,
                required=required_default,
                config_key=value.config_key,
                candidates=list(value.candidates) if value.candidates is not None else None,
                suggested=value.suggested,
                aliases=list(value.aliases),
                allow_index=value.allow_index,
            )

        if isinstance(value, str):
            return cls(
                semantic_name=value,
                display_name=_default_display_name(value),
                description="",
                required=required_default,
                aliases=_default_aliases(value),
                allow_index=value in {"record_id", "id", "id_col", "row_id"},
            )

        if isinstance(value, Mapping):
            data = dict(value)
            if "semantic_name" not in data:
                raise ValueError("Mapping requirement dictionaries need 'semantic_name'.")
            data.setdefault("required", required_default)
            data.setdefault("display_name", _default_display_name(str(data["semantic_name"])))
            data.setdefault("aliases", _default_aliases(str(data["semantic_name"])))
            data.setdefault(
                "allow_index",
                str(data["semantic_name"]) in {"record_id", "id", "id_col", "row_id"},
            )
            return cls(**data)

        raise TypeError(f"Cannot convert {value!r} to MappingRequirement.")

    def payload(
        self,
        *,
        context: Any,
        dataset_id: str,
        source: str,
        panel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        columns = list_dataset_columns(context, dataset_id)

        if self.candidates is None:
            candidates = list(columns)
            if self.allow_index and "Use Index" not in candidates:
                candidates = ["Use Index"] + candidates
        else:
            candidates = list(self.candidates)

        suggested = self.suggested
        if suggested is None:
            suggested = guess_column(columns, self.aliases)
            if suggested is None and self.allow_index:
                suggested = "Use Index"

        payload = {
            "source": source,
            "panel_id": panel_id,
            "dataset_id": dataset_id,
            "semantic_name": self.semantic_name,
            "display_name": self.display_name or _default_display_name(self.semantic_name),
            "description": self.description,
            "required": self.required,
            "config_key": self.config_key,
            "candidates": candidates,
            "suggested": suggested,
        }
        return payload


MappingRequirementLike = Union[str, Mapping[str, Any], MappingRequirement]


@dataclass(frozen=True)
class MappingResolution:
    dataset_id: str
    required: List[MappingRequirement]
    optional: List[MappingRequirement]
    missing_required: List[MappingRequirement]
    missing_optional: List[MappingRequirement]

    @property
    def missing_any(self) -> bool:
        return bool(self.missing_required or self.missing_optional)

    @property
    def can_open(self) -> bool:
        return not self.missing_required


def coerce_mapping_requirements(
    values: Optional[Sequence[MappingRequirementLike]],
    *,
    required_default: bool,
) -> List[MappingRequirement]:
    return [
        MappingRequirement.from_any(value, required_default=required_default)
        for value in (values or [])
    ]


def dataset_exists(context: Any, dataset_id: Optional[str]) -> bool:
    if context is None or dataset_id is None:
        return False

    datasets = getattr(context, "datasets", None)

    if datasets is None:
        return False

    try:
        has_dataset = getattr(datasets, "has_dataset", None)
        if callable(has_dataset):
            return bool(has_dataset(dataset_id))
    except Exception:
        pass

    try:
        if hasattr(datasets, "_datasets"):
            return str(dataset_id) in datasets._datasets
    except Exception:
        pass

    try:
        list_columns = getattr(datasets, "list_columns", None)
        if callable(list_columns):
            list_columns(dataset_id)
            return True
    except Exception:
        pass

    # Last resort only. Avoid this path for the new source-backed datasets.
    try:
        get_source = getattr(datasets, "get_source", None)
        if callable(get_source):
            get_source(dataset_id)
            return True
    except Exception:
        pass

    return False

def _mapping_column_is_valid(
    *,
    context: Any,
    dataset_id: str,
    column_name: Optional[str],
    allow_index: bool = False,
) -> bool:
    if column_name is None:
        return False

    column_name = str(column_name)

    if allow_index and column_name == "Use Index":
        return True

    columns = list_dataset_columns(context, dataset_id)

    if not columns:
        return False

    return column_name in {str(col) for col in columns}


def resolve_mapping_requirements(
    *,
    context: Any,
    dataset_id: Optional[str],
    required_mappings: Optional[Sequence[MappingRequirementLike]] = None,
    optional_mappings: Optional[Sequence[MappingRequirementLike]] = None,
) -> MappingResolution:
    resolved_dataset_id = dataset_id or active_dataset_id(context) or "main"

    required = coerce_mapping_requirements(
        required_mappings,
        required_default=True,
    )
    optional = coerce_mapping_requirements(
        optional_mappings,
        required_default=False,
    )

    # Important for workspace restore:
    # if the dataset is not loaded yet, mappings cannot be considered valid.
    # Panels should wait until the dataset exists, then re-check saved mappings.
    if not dataset_exists(context, resolved_dataset_id):
        mapping_debug_print(
            "resolve requirements: dataset not loaded",
            {
                "dataset_id": resolved_dataset_id,
                "required": [req.semantic_name for req in required],
                "optional": [req.semantic_name for req in optional],
            },
        )
        return MappingResolution(
            dataset_id=resolved_dataset_id,
            required=required,
            optional=optional,
            missing_required=required,
            missing_optional=optional,
        )

    missing_required = []
    for req in required:
        column_name = get_mapping(context, resolved_dataset_id, req.semantic_name)

        if column_name is None:
            missing_required.append(req)
            continue

        if not _mapping_column_is_valid(
            context=context,
            dataset_id=resolved_dataset_id,
            column_name=column_name,
            allow_index=req.allow_index,
        ):
            missing_required.append(req)

    missing_optional = []
    for req in optional:
        column_name = get_mapping(context, resolved_dataset_id, req.semantic_name)

        if column_name is None:
            missing_optional.append(req)
            continue

        if not _mapping_column_is_valid(
            context=context,
            dataset_id=resolved_dataset_id,
            column_name=column_name,
            allow_index=req.allow_index,
        ):
            missing_required.append(req)

    mapping_debug_print(
        "resolve requirements",
        {
            "dataset_id": resolved_dataset_id,
            "required": [req.semantic_name for req in required],
            "optional": [req.semantic_name for req in optional],
            "missing_required": [req.semantic_name for req in missing_required],
            "missing_optional": [req.semantic_name for req in missing_optional],
        },
    )

    return MappingResolution(
        dataset_id=resolved_dataset_id,
        required=required,
        optional=optional,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )


def publish_mapping_requests(
    *,
    context: Any,
    dataset_id: str,
    source: str,
    panel_id: Optional[str],
    requirements: Iterable[MappingRequirement],
    sent_keys: Optional[set[Tuple[str, str, str]]] = None,
) -> None:
    events = getattr(context, "events", None)
    if events is None:
        return

    for requirement in requirements:
        key = (dataset_id, source, requirement.semantic_name)
        if sent_keys is not None and key in sent_keys:
            continue

        events.publish(
            "mapping.requested",
            requirement.payload(
                context=context,
                dataset_id=dataset_id,
                source=source,
                panel_id=panel_id,
            ),
        )

        if sent_keys is not None:
            sent_keys.add(key)


def active_dataset_id(context: Any) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    try:
        active_id = getattr(datasets, "active_id", None)
        if callable(active_id):
            value = active_id()
            return str(value) if value else None
        if active_id:
            return str(active_id)
    except Exception:
        pass

    try:
        value = getattr(datasets, "active_dataset_id", None)
        return str(value) if value else None
    except Exception:
        return None


def get_mapping(context: Any, dataset_id: str, semantic_name: str) -> Optional[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return None

    try:
        get_mapping_fn = getattr(datasets, "get_mapping", None)
        if callable(get_mapping_fn):
            value = get_mapping_fn(dataset_id, semantic_name)
            return str(value) if value is not None else None
    except Exception:
        pass

    try:
        get_mappings_fn = getattr(datasets, "get_mappings", None)
        if callable(get_mappings_fn):
            mappings = get_mappings_fn(dataset_id)
            if isinstance(mappings, Mapping):
                value = mappings.get(semantic_name)
                return str(value) if value is not None else None
    except Exception:
        pass

    return None


def list_dataset_columns(context: Any, dataset_id: str) -> List[str]:
    datasets = getattr(context, "datasets", None)
    if datasets is None:
        return []

    try:
        list_columns_fn = getattr(datasets, "list_columns", None)
        if callable(list_columns_fn):
            return [str(col) for col in list_columns_fn(dataset_id)]
    except Exception:
        pass

    try:
        get_df_fn = getattr(datasets, "get_df", None)
        if callable(get_df_fn):
            df = get_df_fn(dataset_id)
        else:
            df = None
        if df is not None and hasattr(df, "columns"):
            return [str(col) for col in df.columns]
    except Exception:
        pass

    return []


def guess_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    if not columns:
        return None

    exact = {str(col).lower(): str(col) for col in columns}
    for alias in aliases:
        match = exact.get(str(alias).lower())
        if match is not None:
            return match

    normalised_columns = {_normalise_name(str(col)): str(col) for col in columns}
    for alias in aliases:
        match = normalised_columns.get(_normalise_name(str(alias)))
        if match is not None:
            return match

    return None


def _default_aliases(semantic_name: str) -> List[str]:
    aliases = list(_COMMON_ALIASES.get(semantic_name, []))
    tail = semantic_name.split(".")[-1]
    if tail not in aliases:
        aliases.append(tail)
    return aliases


def _default_display_name(semantic_name: str) -> str:
    if semantic_name == "record_id":
        return "ID column"
    if semantic_name == "coords.ra":
        return "RA column"
    if semantic_name == "coords.dec":
        return "DEC column"
    if semantic_name == "target_label":
        return "Label column"
    return semantic_name.replace("_", " ").replace(".", " / ").title()


def _normalise_name(value: str) -> str:
    return (
        value.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
    )