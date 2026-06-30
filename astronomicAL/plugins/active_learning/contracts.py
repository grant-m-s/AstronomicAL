from __future__ import annotations

"""Recipe-driven data contracts for the Active Learning plugin.

This module deliberately adapts the *current* ``core.ml`` RecipeSpec rather
than requiring every recipe to adopt a new base class immediately. A future
core.ml release can expose the same contract natively; the AL plugin only needs
``RecipeInputContract.from_recipe_spec`` to keep working.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd

IMAGE_SEMANTICS = ("image.path", "image.uri", "image", "cutout.path")
MASK_SEMANTICS = ("mask.path", "mask", "segmentation.mask")
RECORD_ID_SEMANTICS = ("record_id", "id", "row_id")
TARGET_SEMANTICS = ("target_label", "label", "class")

IMAGE_PARAM_NAMES = (
    "image_column",
    "image_path_column",
    "image_uri_column",
)
MASK_PARAM_NAMES = ("mask_column", "mask_path_column")
FEATURE_PARAM_NAMES = (
    "feature_columns",
    "input_columns",
    "features",
    "x_columns",
)
RECORD_ID_PARAM_NAMES = ("record_id_column", "id_column")
TARGET_PARAM_NAMES = ("target_column", "label_column", "class_column")

IMAGE_ALIASES = (
    "image_path",
    "image_uri",
    "image_url",
    "image",
    "filepath",
    "file_path",
    "path",
    "filename",
    "cutout_path",
)
MASK_ALIASES = ("mask_path", "mask", "segmentation_mask")
RECORD_ID_ALIASES = (
    "record_id",
    "row_id",
    "object_id",
    "source_id",
    "id",
    "ID",
)
TARGET_ALIASES = (
    "target_label",
    "label",
    "labels",
    "class_label",
    "class",
    "target",
    "y",
)

def _stable_strings(values: Iterable[Any]) -> List[str]:
    return list(
        dict.fromkeys(
            text
            for value in values or []
            if value is not None
            for text in [str(value).strip()]
            if text and text != "Use Index"
        )
    )

def _normalise_column_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [
            part.strip()
            for line in value.splitlines()
            for part in line.split(",")
        ]
    elif isinstance(value, Mapping):
        value = value.keys()
    elif not isinstance(value, Iterable):
        value = [value]

    return _stable_strings(value)

def _properties(spec: Any) -> Mapping[str, Any]:
    schema = getattr(spec, "params_schema", {}) or {}

    if not isinstance(schema, Mapping):
        return {}

    properties = schema.get("properties", {}) or {}
    return properties if isinstance(properties, Mapping) else {}

@dataclass(frozen=True)
class RecipeInputContract:
    """The data-facing requirements that the AL panel derives from a recipe."""

    schema_version: int = 1
    task: str = "classification"
    modality: str = "tabular"
    needs_features: bool = False
    needs_image: bool = False
    needs_mask: bool = False
    feature_param: str = "feature_columns"
    image_param: str = "image_column"
    mask_param: str = "mask_column"
    record_id_param: str = "record_id_column"
    target_param: str = "target_column"
    required_mappings: Tuple[str, ...] = field(default_factory=tuple)
    optional_mappings: Tuple[str, ...] = field(default_factory=tuple)
    supports_external_validation: bool = True
    supports_external_test: bool = True
    supports_internal_validation_split: bool = True
    supports_internal_test_split: bool = True
    minimum_samples: int = 2
    minimum_classes: int = 2
    minimum_samples_per_class: int = 1

    @classmethod
    def from_recipe_spec(cls, spec: Any) -> "RecipeInputContract":
        if spec is None:
            return cls()

        task = str(
            getattr(spec, "task", "classification")
            or "classification"
        ).lower()

        modality = str(
            getattr(spec, "modality", "tabular")
            or "tabular"
        ).lower()

        properties = _properties(spec)
        property_names = {
            str(name).lower()
            for name in properties
        }

        required_mappings = tuple(
            _stable_strings(
                getattr(spec, "required_mappings", [])
                or []
            )
        )

        optional_mappings = tuple(
            _stable_strings(
                getattr(spec, "optional_mappings", [])
                or []
            )
        )

        all_mappings = (
            set(required_mappings)
            | set(optional_mappings)
        )

        has_image_param = bool(
            property_names.intersection(
                IMAGE_PARAM_NAMES
            )
        )

        has_mask_param = bool(
            property_names.intersection(
                MASK_PARAM_NAMES
            )
        )

        has_feature_param = bool(
            property_names.intersection(
                FEATURE_PARAM_NAMES
            )
        )

        image_modality = any(
            token in modality
            for token in (
                "image",
                "vision",
                "pixel",
            )
        )

        tabular_modality = any(
            token in modality
            for token in (
                "tabular",
                "table",
                "catalog",
            )
        )

        segmentation_task = (
            "segment" in task
            or "segment" in modality
        )

        needs_image = bool(
            image_modality
            or has_image_param
            or all_mappings.intersection(
                IMAGE_SEMANTICS
            )
        )

        needs_mask = bool(
            segmentation_task
            or has_mask_param
            or all_mappings.intersection(
                MASK_SEMANTICS
            )
        )

        needs_features = bool(
            tabular_modality
            or has_feature_param
            or (
                not needs_image
                and not needs_mask
            )
        )

        def first_present(
            names: Sequence[str],
            fallback: str,
        ) -> str:
            for name in names:
                if name in property_names:
                    return name

            return fallback

        recipe_cls = getattr(
            spec,
            "recipe_cls",
            None,
        )

        declared_contract = getattr(
            spec,
            "input_contract",
            None,
        ) or getattr(
            recipe_cls,
            "input_contract",
            None,
        ) or {}

        if not isinstance(declared_contract, Mapping):
            declared_contract = {}

        def declared_value(
            name: str,
            default: Any,
        ) -> Any:
            if name in declared_contract:
                return declared_contract[name]
            return getattr(
                recipe_cls,
                name,
                getattr(spec, name, default),
            )

        def capability(
            name: str,
            default: bool = True,
        ) -> bool:
            return bool(declared_value(name, default))

        classification = "class" in task

        return cls(
            task=task,
            modality=modality,
            needs_features=needs_features,
            needs_image=needs_image,
            needs_mask=needs_mask,
            feature_param=first_present(
                FEATURE_PARAM_NAMES,
                "feature_columns",
            ),
            image_param=first_present(
                IMAGE_PARAM_NAMES,
                "image_column",
            ),
            mask_param=first_present(
                MASK_PARAM_NAMES,
                "mask_column",
            ),
            record_id_param=first_present(
                RECORD_ID_PARAM_NAMES,
                "record_id_column",
            ),
            target_param=first_present(
                TARGET_PARAM_NAMES,
                "target_column",
            ),
            required_mappings=required_mappings,
            optional_mappings=optional_mappings,
            supports_external_validation=capability(
                "supports_external_validation"
            ),
            supports_external_test=capability(
                "supports_external_test"
            ),
            supports_internal_validation_split=capability(
                "supports_internal_validation_split"
            ),
            supports_internal_test_split=capability(
                "supports_internal_test_split"
            ),
            minimum_samples=max(
                1,
                int(declared_value("minimum_samples", 2)),
            ),
            minimum_classes=max(
                1,
                int(
                    declared_value(
                        "minimum_classes",
                        2 if classification else 1,
                    )
                ),
            ),
            minimum_samples_per_class=max(
                1,
                int(
                    declared_value(
                        "minimum_samples_per_class",
                        1,
                    )
                ),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)

        payload["required_mappings"] = list(
            self.required_mappings
        )

        payload["optional_mappings"] = list(
            self.optional_mappings
        )

        return payload

@dataclass
class DatasetRoleBinding:
    role: str
    dataset_id: str
    record_id_column: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = field(
        default_factory=list
    )
    image_column: Optional[str] = None
    mask_column: Optional[str] = None
    resolution_sources: Dict[str, str] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "dataset_id": self.dataset_id,
            "record_id_column": (
                self.record_id_column
            ),
            "target_column": self.target_column,
            "feature_columns": list(
                self.feature_columns
            ),
            "image_column": self.image_column,
            "mask_column": self.mask_column,
            "resolution_sources": dict(
                self.resolution_sources
            ),
        }

@dataclass
class PreflightReport:
    contract: Dict[str, Any]
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                "Active Learning data preflight failed:\n- "
                + "\n- ".join(self.errors)
            )

def dataset_columns(
    context: Any,
    dataset_id: str,
) -> List[str]:
    dataset_id = str(
        dataset_id or ""
    ).strip()

    if not dataset_id:
        return []

    try:
        return [
            str(column)
            for column in (
                context.datasets.list_columns(
                    dataset_id
                )
            )
        ]
    except Exception:
        pass

    try:
        return [
            str(column)
            for column in (
                context.datasets.get_df(
                    dataset_id
                ).columns
            )
        ]
    except Exception:
        return []

def dataset_mappings(
    context: Any,
    dataset_id: str,
) -> Dict[str, str]:
    try:
        return {
            str(key): str(value)
            for key, value in dict(
                context.datasets.get_mappings(
                    dataset_id
                )
                or {}
            ).items()
            if value is not None
        }
    except Exception:
        return {}

def dataset_row_count(
    context: Any,
    dataset_id: str,
) -> Optional[int]:
    datasets = getattr(
        context,
        "datasets",
        None,
    )

    source = None

    if datasets is not None:
        getter = getattr(
            datasets,
            "get_source",
            None,
        )

        if callable(getter):
            try:
                source = getter(
                    dataset_id
                )
            except Exception:
                source = None

    for owner in (
        source,
        datasets,
    ):
        if owner is None:
            continue

        for name in (
            "row_count",
            "count_rows",
            "count",
        ):
            value = getattr(
                owner,
                name,
                None,
            )

            if callable(value):
                for args in (
                    (),
                    (dataset_id,),
                ):
                    try:
                        result = value(*args)
                    except TypeError:
                        continue
                    except Exception:
                        result = None

                    if result is not None:
                        try:
                            return int(result)
                        except Exception:
                            pass

            elif value is not None:
                try:
                    return int(value)
                except Exception:
                    pass

    columns = dataset_columns(
        context,
        dataset_id,
    )

    projected = columns[:1]

    try:
        if projected:
            try:
                frame = (
                    context.datasets.get_df(
                        dataset_id,
                        columns=projected,
                    )
                )
            except TypeError:
                frame = (
                    context.datasets.get_df(
                        dataset_id
                    )
                )
        else:
            frame = (
                context.datasets.get_df(
                    dataset_id
                )
            )

        return int(len(frame))

    except Exception:
        return None

def _mapping_value(
    mappings: Mapping[str, str],
    semantics: Sequence[str],
) -> Optional[str]:
    for semantic in semantics:
        value = mappings.get(semantic)

        if value:
            return str(value)

    return None

def _first_existing(
    columns: Sequence[str],
    aliases: Sequence[str],
) -> Optional[str]:
    by_lower = {
        str(column).lower(): str(column)
        for column in columns
    }

    for alias in aliases:
        value = by_lower.get(
            str(alias).lower()
        )

        if value:
            return value

    return None

def _resolve_single_column(
    *,
    explicit: Any,
    mappings: Mapping[str, str],
    semantics: Sequence[str],
    columns: Sequence[str],
    aliases: Sequence[str],
) -> Tuple[Optional[str], str]:
    explicit_text = str(
        explicit or ""
    ).strip()

    if (
        explicit_text
        and explicit_text in columns
    ):
        return explicit_text, "explicit"

    mapped = _mapping_value(
        mappings,
        semantics,
    )

    if (
        mapped
        and mapped in columns
    ):
        semantic = next(
            (
                item
                for item in semantics
                if mappings.get(item) == mapped
            ),
            semantics[0],
        )

        return (
            mapped,
            f"mapping:{semantic}",
        )

    guessed = _first_existing(
        columns,
        aliases,
    )

    if guessed:
        return (
            guessed,
            "column-name inference",
        )

    return None, "unresolved"

def resolve_dataset_role_binding(
    context: Any,
    *,
    role: str,
    dataset_id: str,
    recipe_contract: RecipeInputContract,
    explicit: Optional[
        Mapping[str, Any]
    ] = None,
    pool_binding: Optional[
        DatasetRoleBinding
    ] = None,
) -> DatasetRoleBinding:
    explicit = dict(
        explicit or {}
    )

    columns = dataset_columns(
        context,
        dataset_id,
    )

    mappings = dataset_mappings(
        context,
        dataset_id,
    )

    record_id, record_source = (
        _resolve_single_column(
            explicit=explicit.get(
                "record_id_column"
            ),
            mappings=mappings,
            semantics=RECORD_ID_SEMANTICS,
            columns=columns,
            aliases=RECORD_ID_ALIASES,
        )
    )

    image, image_source = (
        _resolve_single_column(
            explicit=explicit.get(
                "image_column"
            ),
            mappings=mappings,
            semantics=IMAGE_SEMANTICS,
            columns=columns,
            aliases=IMAGE_ALIASES,
        )
    )

    mask, mask_source = (
        _resolve_single_column(
            explicit=explicit.get(
                "mask_column"
            ),
            mappings=mappings,
            semantics=MASK_SEMANTICS,
            columns=columns,
            aliases=MASK_ALIASES,
        )
    )

    target_explicit = explicit.get(
        "target_column"
    )

    target, target_source = (
        _resolve_single_column(
            explicit=target_explicit,
            mappings=mappings,
            semantics=TARGET_SEMANTICS,
            columns=columns,
            aliases=TARGET_ALIASES,
        )
    )

    requested_features = (
        _normalise_column_list(
            explicit.get(
                "feature_columns"
            )
        )
    )

    if requested_features:
        features = [
            column
            for column in requested_features
            if column in columns
        ]

        feature_source = "explicit"

    elif (
        pool_binding is not None
        and role != "pool"
        and pool_binding.feature_columns
    ):
        # External datasets frequently use the same
        # feature names. Keep only those that physically
        # exist and report missing values in preflight.
        features = [
            column
            for column in (
                pool_binding.feature_columns
            )
            if column in columns
        ]

        feature_source = "pool contract"

    else:
        excluded = {
            value
            for value in (
                record_id,
                target,
                image,
                mask,
            )
            if value
        }

        features = [
            column
            for column in columns
            if column not in excluded
        ]

        feature_source = (
            "all non-binding columns"
        )

    return DatasetRoleBinding(
        role=str(role),
        dataset_id=str(dataset_id),
        record_id_column=record_id,
        target_column=target,
        feature_columns=features,
        image_column=image,
        mask_column=mask,
        resolution_sources={
            "record_id_column": (
                record_source
            ),
            "target_column": target_source,
            "feature_columns": (
                feature_source
            ),
            "image_column": image_source,
            "mask_column": mask_source,
        },
    )

def _projected_frame(
    context: Any,
    dataset_id: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    clean = _stable_strings(
        columns
    )

    try:
        return context.datasets.get_df(
            dataset_id,
            columns=clean,
        )

    except TypeError:
        frame = (
            context.datasets.get_df(
                dataset_id
            )
        )

        return frame.loc[
            :,
            [
                column
                for column in clean
                if column in frame.columns
            ],
        ]

def _non_null_count(
    context: Any,
    dataset_id: str,
    column: Optional[str],
) -> Optional[int]:
    if not column:
        return None

    try:
        frame = _projected_frame(
            context,
            dataset_id,
            [column],
        )

        if column not in frame.columns:
            return None

        return int(
            frame[column].notna().sum()
        )

    except Exception:
        return None

def _column_profile(
    context: Any,
    dataset_id: str,
    column: Optional[str],
    *,
    include_values: bool = False,
) -> Dict[str, Any]:
    if not column:
        return {}

    try:
        frame = _projected_frame(context, dataset_id, [column])
    except Exception as exc:
        return {"error": str(exc)}

    if column not in frame.columns:
        return {"error": f"Column {column!r} is missing."}

    series = frame[column]
    non_null = series.dropna()
    as_text = non_null.astype(str)
    result: Dict[str, Any] = {
        "column": column,
        "rows": int(len(series)),
        "non_null": int(non_null.shape[0]),
        "null": int(series.isna().sum()),
        "unique": int(as_text.nunique(dropna=True)),
        "duplicates": int(as_text.duplicated(keep=False).sum()),
    }

    if include_values:
        counts = as_text.value_counts(dropna=False).head(100)
        result["value_counts"] = {
            str(key): int(value)
            for key, value in counts.items()
        }
        result["values_truncated"] = bool(as_text.nunique(dropna=True) > 100)

    return result


def _is_remote_reference(
    value: str,
) -> bool:
    parsed = urlparse(value)

    return parsed.scheme.lower() in {
        "http",
        "https",
        "s3",
        "gs",
        "hf",
    }

def inspect_image_column(
    context: Any,
    *,
    dataset_id: str,
    image_column: str,
    sample_size: int = 8,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Inspect a small local sample.

    Remote URIs are reported but are not fetched.
    """

    result: Dict[str, Any] = {
        "requested_sample_size": int(
            max(
                0,
                sample_size,
            )
        ),
        "inspected": 0,
        "readable": 0,
        "unreadable": 0,
        "remote_skipped": 0,
        "missing": 0,
        "sizes": {},
        "modes": {},
        "formats": {},
        "errors": [],
    }

    if (
        sample_size <= 0
        or not image_column
    ):
        return result

    try:
        from PIL import Image
    except Exception:
        result["errors"].append(
            "Pillow is not installed; image "
            "dimensions were not inspected."
        )

        return result

    try:
        frame = _projected_frame(
            context,
            dataset_id,
            [image_column],
        )
    except Exception as exc:
        result["errors"].append(
            "Could not read image column: "
            f"{exc}"
        )

        return result

    if image_column not in frame.columns:
        result["errors"].append(
            f"Image column {image_column!r} "
            "is missing."
        )

        return result

    size_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()

    attempted = 0

    for raw_value in (
        frame[image_column]
        .dropna()
        .tolist()
    ):
        if attempted >= sample_size:
            break

        attempted += 1
        result["inspected"] = attempted

        if cancel_token is not None:
            for name in (
                "raise_if_cancelled",
                "throw_if_cancelled",
                "check_cancelled",
            ):
                method = getattr(
                    cancel_token,
                    name,
                    None,
                )

                if callable(method):
                    method()
                    break

        value = str(
            raw_value
        ).strip()

        if not value:
            result["missing"] += 1
            continue

        if _is_remote_reference(value):
            result["remote_skipped"] += 1
            continue

        path = Path(value).expanduser()

        if not path.exists():
            result["missing"] += 1
            result["unreadable"] += 1
            continue

        try:
            with Image.open(path) as image:
                width, height = image.size

                size_counter[
                    f"{width}×{height}"
                ] += 1

                mode_counter[
                    str(
                        image.mode
                        or "unknown"
                    )
                ] += 1

                format_counter[
                    str(
                        image.format
                        or path.suffix.lstrip(".")
                        or "unknown"
                    )
                ] += 1

                result["readable"] += 1

        except Exception as exc:
            result["unreadable"] += 1

            if len(result["errors"]) < 5:
                result["errors"].append(
                    f"{path}: {exc}"
                )

    result["sizes"] = dict(
        size_counter
    )

    result["modes"] = dict(
        mode_counter
    )

    result["formats"] = dict(
        format_counter
    )

    return result

def profile_dataset_role(
    context: Any,
    *,
    binding: DatasetRoleBinding,
    recipe_contract: RecipeInputContract,
    inspect_images: bool = False,
    include_column_counts: bool = False,
    image_sample_size: int = 8,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    columns = dataset_columns(
        context,
        binding.dataset_id,
    )

    row_count = dataset_row_count(
        context,
        binding.dataset_id,
    )

    profile: Dict[str, Any] = {
        "role": binding.role,
        "dataset_id": binding.dataset_id,
        "shape": (
            [
                row_count,
                len(columns),
            ]
            if row_count is not None
            else [
                None,
                len(columns),
            ]
        ),
        "row_count": row_count,
        "column_count": len(columns),
        "binding": binding.to_dict(),
        "columns": list(columns),
        "mappings": dataset_mappings(context, binding.dataset_id),
        "non_null": {},
        "record_id_profile": {},
        "target_profile": {},
        "image_profile": None,
    }

    if include_column_counts:
        profile["record_id_profile"] = _column_profile(
            context,
            binding.dataset_id,
            binding.record_id_column,
        )
        profile["target_profile"] = _column_profile(
            context,
            binding.dataset_id,
            binding.target_column,
            include_values=True,
        )

        for key, column in (
            (
                "record_id",
                binding.record_id_column,
            ),
            (
                "target",
                binding.target_column,
            ),
            (
                "image",
                binding.image_column,
            ),
            (
                "mask",
                binding.mask_column,
            ),
        ):
            if column:
                profile["non_null"][key] = (
                    _non_null_count(
                        context,
                        binding.dataset_id,
                        column,
                    )
                )

    if (
        inspect_images
        and recipe_contract.needs_image
        and binding.image_column
    ):
        profile["image_profile"] = (
            inspect_image_column(
                context,
                dataset_id=(
                    binding.dataset_id
                ),
                image_column=(
                    binding.image_column
                ),
                sample_size=(
                    image_sample_size
                ),
                cancel_token=cancel_token,
            )
        )

    return profile

def _validate_role(
    *,
    role: str,
    binding: DatasetRoleBinding,
    profile: Mapping[str, Any],
    recipe_contract: RecipeInputContract,
    pool_binding: Optional[
        DatasetRoleBinding
    ],
    require_target: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    columns = set(
        dataset_columns_from_profile(
            profile
        )
    )

    if not binding.record_id_column:
        errors.append(
            f"{role}: no record-ID column "
            "or mapping could be resolved."
        )

    record_profile = dict(profile.get("record_id_profile") or {})
    if binding.record_id_column and record_profile:
        if int(record_profile.get("null") or 0) > 0:
            errors.append(
                f"{role}: record-ID column {binding.record_id_column!r} contains "
                f"{int(record_profile.get('null') or 0):,} null value(s)."
            )
        if int(record_profile.get("duplicates") or 0) > 0:
            errors.append(
                f"{role}: record-ID column {binding.record_id_column!r} is not unique; "
                f"{int(record_profile.get('duplicates') or 0):,} row(s) participate in duplicates."
            )

    mappings = dict(profile.get("mappings") or {})
    available_columns = set(str(column) for column in profile.get("columns") or [])
    semantic_aliases = {
        "record_id": binding.record_id_column,
        "target_label": binding.target_column,
        "image.path": binding.image_column,
        "image.uri": binding.image_column,
        "image": binding.image_column,
        "mask.path": binding.mask_column,
        "mask": binding.mask_column,
    }
    for semantic in recipe_contract.required_mappings:
        resolved = semantic_aliases.get(str(semantic)) or mappings.get(str(semantic))
        if not resolved:
            errors.append(f"{role}: required semantic mapping {semantic!r} is unresolved.")
        elif (
            str(resolved) != "Use Index"
            and str(resolved) not in available_columns
            and not (
                role == "pool"
                and str(semantic) == "target_label"
                and binding.resolution_sources.get("target_column") == "AL annotation output"
            )
        ):
            errors.append(
                f"{role}: required semantic mapping {semantic!r} points to missing "
                f"column {resolved!r}."
            )

    if (
        recipe_contract.needs_image
        and not binding.image_column
    ):
        errors.append(
            f"{role}: recipe requires an "
            "image input, but no image "
            "column was resolved."
        )

    if (
        recipe_contract.needs_mask
        and not binding.mask_column
    ):
        errors.append(
            f"{role}: recipe requires a "
            "mask input, but no mask "
            "column was resolved."
        )

    if (
        recipe_contract.needs_features
        and not binding.feature_columns
    ):
        errors.append(
            f"{role}: recipe requires "
            "feature columns, but none "
            "are selected/resolved."
        )

    if (
        require_target
        and not binding.target_column
    ):
        errors.append(
            f"{role}: external evaluation "
            "dataset has no target/label "
            "column."
        )

    if (
        pool_binding is not None
        and role != "pool"
        and recipe_contract.needs_features
    ):
        missing = [
            column
            for column in (
                pool_binding.feature_columns
            )
            if column not in (
                binding.feature_columns
            )
        ]

        if missing:
            errors.append(
                f"{role}: missing training "
                "feature column(s): "
                + ", ".join(
                    missing[:12]
                )
            )

    row_count = profile.get(
        "row_count"
    )

    if row_count == 0:
        errors.append(
            f"{role}: dataset is empty."
        )

    elif row_count is None:
        warnings.append(
            f"{role}: row count could "
            "not be determined."
        )

    image_non_null = (
        profile.get("non_null")
        or {}
    ).get("image")

    if (
        recipe_contract.needs_image
        and row_count
        and image_non_null is not None
    ):
        if image_non_null == 0:
            errors.append(
                f"{role}: image column "
                "contains no non-null values."
            )

        elif image_non_null < row_count:
            warnings.append(
                f"{role}: image column is "
                f"null for "
                f"{row_count - image_non_null:,} "
                "row(s)."
            )

    del columns

    return errors, warnings

def dataset_columns_from_profile(
    profile: Mapping[str, Any],
) -> List[str]:
    # The profile intentionally does not persist
    # every column name. Bindings are enough for
    # validation and keep session artifacts compact.
    binding = (
        profile.get("binding")
        or {}
    )

    return _stable_strings(
        [
            binding.get(
                "record_id_column"
            ),
            binding.get(
                "target_column"
            ),
            binding.get(
                "image_column"
            ),
            binding.get(
                "mask_column"
            ),
            *(
                binding.get(
                    "feature_columns"
                )
                or []
            ),
        ]
    )

def build_session_contract(
    context: Any,
    *,
    recipe_spec: Any,
    pool_dataset_id: str,
    validation_dataset_id: str = "",
    test_dataset_id: str = "",
    target_column: str = "al_label",
    feature_columns: Optional[
        Sequence[str]
    ] = None,
    image_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    recipe_params: Optional[
        Mapping[str, Any]
    ] = None,
    protocol_params: Optional[
        Mapping[str, Any]
    ] = None,
    labelled_count: int = 0,
    label_counts: Optional[Mapping[str, int]] = None,
    inspect_images: bool = False,
    include_column_counts: bool = False,
    image_sample_size: int = 8,
    cancel_token: Any = None,
) -> PreflightReport:
    recipe_params = dict(
        recipe_params or {}
    )

    protocol_params = dict(
        protocol_params or {}
    )
    label_counts = {
        str(label): int(count)
        for label, count in dict(label_counts or {}).items()
        if str(label).strip() and int(count) > 0
    }

    recipe_contract = (
        RecipeInputContract.from_recipe_spec(
            recipe_spec
        )
    )

    pool_explicit = {
        "target_column": target_column,
        "feature_columns": list(
            feature_columns or []
        ),
        "image_column": image_column,
        "mask_column": mask_column,
        "record_id_column": (
            recipe_params.get(
                "record_id_column"
            )
        ),
    }

    pool_binding = (
        resolve_dataset_role_binding(
            context,
            role="pool",
            dataset_id=pool_dataset_id,
            recipe_contract=(
                recipe_contract
            ),
            explicit=pool_explicit,
        )
    )

    # The AL annotation target can be created
    # later, so preserve the requested name even
    # when the source pool does not physically
    # contain it.
    if target_column:
        pool_binding.target_column = str(
            target_column
        )

        pool_binding.resolution_sources[
            "target_column"
        ] = (
            "existing pool column"
            if str(target_column)
            in dataset_columns(
                context,
                pool_dataset_id,
            )
            else "AL annotation output"
        )

    bindings: Dict[
        str,
        DatasetRoleBinding,
    ] = {
        "pool": pool_binding,
    }

    if validation_dataset_id:
        bindings["validation"] = (
            resolve_dataset_role_binding(
                context,
                role="validation",
                dataset_id=(
                    validation_dataset_id
                ),
                recipe_contract=(
                    recipe_contract
                ),
                explicit={
                    "feature_columns": (
                        pool_binding.feature_columns
                    ),
                    "image_column": (
                        image_column
                    ),
                    "mask_column": (
                        mask_column
                    ),
                    "target_column": (
                        target_column
                    ),
                },
                pool_binding=pool_binding,
            )
        )

    if test_dataset_id:
        bindings["test"] = (
            resolve_dataset_role_binding(
                context,
                role="test",
                dataset_id=test_dataset_id,
                recipe_contract=(
                    recipe_contract
                ),
                explicit={
                    "feature_columns": (
                        pool_binding.feature_columns
                    ),
                    "image_column": (
                        image_column
                    ),
                    "mask_column": (
                        mask_column
                    ),
                    "target_column": (
                        target_column
                    ),
                },
                pool_binding=pool_binding,
            )
        )

    profiles = {
        role: profile_dataset_role(
            context,
            binding=binding,
            recipe_contract=(
                recipe_contract
            ),
            inspect_images=(
                inspect_images
            ),
            include_column_counts=(
                include_column_counts
            ),
            image_sample_size=(
                image_sample_size
            ),
            cancel_token=cancel_token,
        )
        for role, binding in (
            bindings.items()
        )
    }

    errors: List[str] = []
    warnings: List[str] = []

    for role, binding in (
        bindings.items()
    ):
        role_errors, role_warnings = (
            _validate_role(
                role=role,
                binding=binding,
                profile=profiles[role],
                recipe_contract=(
                    recipe_contract
                ),
                pool_binding=(
                    pool_binding
                ),
                require_target=(
                    role
                    in {
                        "validation",
                        "test",
                    }
                ),
            )
        )

        errors.extend(
            role_errors
        )

        warnings.extend(
            role_warnings
        )

    ids = [
        value
        for value in (
            pool_dataset_id,
            validation_dataset_id,
            test_dataset_id,
        )
        if value
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "Pool, validation, and test "
            "roles must use distinct "
            "dataset IDs."
        )

    validation_source = str(
        protocol_params.get(
            "protocol_validation_source"
        )
        or "split"
    )

    test_source = str(
        protocol_params.get(
            "protocol_test_source"
        )
        or "split"
    )

    validation_fraction = float(
        protocol_params.get(
            "protocol_validation_size"
        )
        or 0.0
    )

    test_fraction = float(
        protocol_params.get(
            "protocol_test_size"
        )
        or 0.0
    )

    if (
        validation_source == "split"
        and not (
            recipe_contract
            .supports_internal_validation_split
        )
    ):
        errors.append(
            "Selected recipe does not "
            "support an internal "
            "validation split."
        )

    if (
        validation_source == "dataset"
        and not (
            recipe_contract
            .supports_external_validation
        )
    ):
        errors.append(
            "Selected recipe does not "
            "support an external "
            "validation dataset."
        )

    if (
        test_source == "split"
        and test_fraction > 0
        and not (
            recipe_contract
            .supports_internal_test_split
        )
    ):
        errors.append(
            "Selected recipe does not "
            "support an internal test split."
        )

    if (
        test_source == "dataset"
        and not (
            recipe_contract
            .supports_external_test
        )
    ):
        errors.append(
            "Selected recipe does not "
            "support an external "
            "test dataset."
        )

    if (
        validation_fraction < 0
        or test_fraction < 0
    ):
        errors.append(
            "Validation and test "
            "fractions cannot be negative."
        )

    if (
        validation_source == "split"
        and test_source == "split"
    ):
        if (
            validation_fraction
            + test_fraction
            >= 1.0
        ):
            errors.append(
                "Validation fraction + "
                "test fraction must be "
                "less than 1.0."
            )

    labelled_count = max(
        0,
        int(labelled_count or 0),
    )

    if labelled_count:
        if labelled_count < recipe_contract.minimum_samples:
            errors.append(
                "Training requires at least "
                f"{recipe_contract.minimum_samples} verified sample(s); "
                f"only {labelled_count} are available."
            )

        represented_classes = len(label_counts)
        if represented_classes < recipe_contract.minimum_classes:
            errors.append(
                "Training requires at least "
                f"{recipe_contract.minimum_classes} represented class(es); "
                f"only {represented_classes} are present."
            )

        underfilled = {
            label: count
            for label, count in label_counts.items()
            if count < recipe_contract.minimum_samples_per_class
        }
        if underfilled:
            errors.append(
                "Each represented class requires at least "
                f"{recipe_contract.minimum_samples_per_class} verified sample(s). "
                "Underfilled: "
                + ", ".join(f"{label}={count}" for label, count in sorted(underfilled.items()))
            )

    split_estimates: Dict[
        str,
        Optional[int],
    ] = {
        "labelled_total": (
            labelled_count
        ),
        "train": None,
        "validation": None,
        "test": None,
    }

    if labelled_count:
        validation_rows = (
            int(
                round(
                    labelled_count
                    * validation_fraction
                )
            )
            if validation_source == "split"
            else (
                profiles.get(
                    "validation",
                    {},
                ).get(
                    "row_count"
                )
            )
        )

        test_rows = (
            int(
                round(
                    labelled_count
                    * test_fraction
                )
            )
            if test_source == "split"
            else (
                profiles.get(
                    "test",
                    {},
                ).get(
                    "row_count"
                )
            )
        )

        internal_val = (
            validation_rows
            if validation_source == "split"
            else 0
        )

        internal_test = (
            test_rows
            if test_source == "split"
            else 0
        )

        split_estimates.update(
            {
                "train": max(
                    0,
                    labelled_count
                    - int(
                        internal_val
                        or 0
                    )
                    - int(
                        internal_test
                        or 0
                    ),
                ),
                "validation": (
                    validation_rows
                ),
                "test": test_rows,
            }
        )

    declared_classes = _normalise_column_list(
        recipe_params.get("class_labels")
        or recipe_params.get("classes")
        or recipe_params.get("known_classes")
        or recipe_params.get("label_options")
        or []
    )
    if declared_classes:
        declared = set(declared_classes)
        for role in ("validation", "test"):
            target_profile = dict((profiles.get(role) or {}).get("target_profile") or {})
            observed = set(str(value) for value in (target_profile.get("value_counts") or {}))
            unexpected = sorted(observed - declared)
            if unexpected:
                errors.append(
                    f"{role}: target contains class value(s) outside the declared class set: "
                    + ", ".join(unexpected[:12])
                )

    recipe_payload = {
        "id": str(
            getattr(
                recipe_spec,
                "id",
                "",
            )
            or ""
        ),
        "version": str(
            getattr(
                recipe_spec,
                "version",
                "",
            )
            or ""
        ),
        "title": str(
            getattr(
                recipe_spec,
                "title",
                "",
            )
            or ""
        ),
        "task": str(
            getattr(
                recipe_spec,
                "task",
                "",
            )
            or ""
        ),
        "modality": str(
            getattr(
                recipe_spec,
                "modality",
                "",
            )
            or ""
        ),
        "framework": str(
            getattr(
                recipe_spec,
                "framework",
                "",
            )
            or ""
        ),
        "execution_mode": str(
            getattr(
                recipe_spec,
                "execution_mode",
                "",
            )
            or ""
        ),
        "input_contract": (
            recipe_contract.to_dict()
        ),
        "params": dict(
            recipe_params
        ),
    }

    contract = {
        "schema_version": 1,
        "recipe": recipe_payload,
        "bindings": {
            role: binding.to_dict()
            for role, binding in (
                bindings.items()
            )
        },
        "profiles": profiles,
        "protocol": dict(
            protocol_params
        ),
        "split_estimates": (
            split_estimates
        ),
        "label_counts": dict(label_counts),
        "preflight": {
            "ok": not errors,
            "errors": list(errors),
            "warnings": list(
                warnings
            ),
        },
    }

    return PreflightReport(
        contract=contract,
        errors=errors,
        warnings=warnings,
    )
