from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class RecipeDataAccessMode(str, Enum):
    """How a managed recipe consumes its training partition."""

    STREAMING = "streaming"
    INCREMENTAL = "incremental"
    EXTERNAL_MEMORY = "external_memory"
    MATERIALIZED = "materialized"


_MODE_LABELS = {
    RecipeDataAccessMode.STREAMING: "Streaming",
    RecipeDataAccessMode.INCREMENTAL: "Incremental",
    RecipeDataAccessMode.EXTERNAL_MEMORY: "External memory",
    RecipeDataAccessMode.MATERIALIZED: "Materialised",
}


@dataclass(frozen=True)
class RecipeDataAccess:
    """Recipe-level data-access contract used by routing and preflight checks.

    The contract is intentionally framework-neutral. A third-party recipe can
    declare one of these modes without teaching the launcher about its estimator
    class or implementation details.
    """

    mode: RecipeDataAccessMode
    description: str = ""
    requires_batch_scan: bool = False
    requires_numeric_features: bool = False
    supports_partial_fit: bool = False
    uses_external_memory: bool = False
    materializes_training_partition: bool = False
    memory_multiplier: float = 1.0
    working_set_multiplier: float = 3.0
    disk_multiplier: float = 0.0
    fixed_overhead_bytes: int = 256 * 1024**2
    warning_bytes: int = 2 * 1024**3
    blocking_bytes: int = 8 * 1024**3
    allow_memory_override: bool = True
    batch_size_param: str = "stream_source_batch_size"
    default_batch_size: int = 8192

    @property
    def label(self) -> str:
        return _MODE_LABELS[self.mode]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["label"] = self.label
        return value

    @classmethod
    def coerce(
        cls,
        value: Any,
        *,
        framework: str = "",
    ) -> "RecipeDataAccess":
        if isinstance(value, cls):
            return value

        if isinstance(value, RecipeDataAccessMode):
            return defaults_for_mode(value)

        if isinstance(value, str) and value.strip():
            return defaults_for_mode(RecipeDataAccessMode(value.strip().lower()))

        if isinstance(value, Mapping):
            raw = dict(value)
            raw.pop("label", None)
            mode = RecipeDataAccessMode(
                str(raw.pop("mode", RecipeDataAccessMode.MATERIALIZED.value)).lower()
            )
            defaults = defaults_for_mode(mode).to_dict()
            defaults.pop("mode", None)
            defaults.pop("label", None)
            defaults.update(raw)
            return cls(mode=mode, **defaults)

        framework = str(framework or "").strip().lower()
        if framework == "torch":
            return defaults_for_mode(RecipeDataAccessMode.STREAMING)
        if framework in {"sklearn", "scikit-learn", "xgboost"}:
            return defaults_for_mode(RecipeDataAccessMode.MATERIALIZED)
        return defaults_for_mode(RecipeDataAccessMode.MATERIALIZED)


def defaults_for_mode(mode: RecipeDataAccessMode | str) -> RecipeDataAccess:
    if not isinstance(mode, RecipeDataAccessMode):
        mode = RecipeDataAccessMode(str(mode).strip().lower())
    if mode == RecipeDataAccessMode.STREAMING:
        return RecipeDataAccess(
            mode=mode,
            description="Reads bounded source batches during each training epoch.",
            requires_batch_scan=True,
            memory_multiplier=0.0,
            working_set_multiplier=4.0,
            disk_multiplier=1.15,
            blocking_bytes=0,
        )
    if mode == RecipeDataAccessMode.INCREMENTAL:
        return RecipeDataAccess(
            mode=mode,
            description="Updates an incremental estimator with bounded batches.",
            requires_batch_scan=True,
            supports_partial_fit=True,
            memory_multiplier=0.0,
            working_set_multiplier=4.0,
            disk_multiplier=1.15,
            blocking_bytes=0,
        )
    if mode == RecipeDataAccessMode.EXTERNAL_MEMORY:
        return RecipeDataAccess(
            mode=mode,
            description="Builds disk-backed training matrices and keeps the full feature matrix out of process memory.",
            requires_batch_scan=True,
            uses_external_memory=True,
            memory_multiplier=0.0,
            working_set_multiplier=5.0,
            disk_multiplier=2.2,
            fixed_overhead_bytes=512 * 1024**2,
            blocking_bytes=0,
        )
    return RecipeDataAccess(
        mode=RecipeDataAccessMode.MATERIALIZED,
        description="Materialises the selected columns and encoded training matrix in process memory.",
        materializes_training_partition=True,
        memory_multiplier=5.0,
        working_set_multiplier=0.0,
        disk_multiplier=1.15,
    )


@dataclass(frozen=True)
class RecipeResourceEstimate:
    dataset_id: str
    access: RecipeDataAccess
    row_count: Optional[int]
    estimated_training_rows: Optional[int]
    feature_columns: List[str]
    non_numeric_feature_columns: List[str]
    estimated_source_bytes: Optional[int]
    estimated_peak_memory_bytes: Optional[int]
    estimated_disk_bytes: Optional[int]
    available_memory_bytes: Optional[int]
    batch_size: Optional[int]
    source_supports_batch_scan: Optional[bool]
    status: str
    blocked: bool
    override_applied: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def access_mode(self) -> str:
        return self.access.mode.value

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["access"] = self.access.to_dict()
        value["access_mode"] = self.access_mode
        return value

    def summary(self) -> str:
        rows = format_count(self.row_count) if self.row_count is not None else "unknown rows"
        peak = format_bytes(self.estimated_peak_memory_bytes)
        disk = format_bytes(self.estimated_disk_bytes)
        if self.access.mode == RecipeDataAccessMode.MATERIALIZED:
            detail = f"Estimated peak memory: **{peak}**."
        elif self.access.mode == RecipeDataAccessMode.EXTERNAL_MEMORY:
            detail = f"Estimated bounded working memory: **{peak}**; temporary disk: **{disk}**."
        else:
            detail = f"Estimated bounded working memory: **{peak}**."
        return f"**{self.access.label}** data access for {rows}. {detail}"


class RecipeResourceLimitError(ValueError):
    """Raised when a recipe preflight blocks execution."""

    def __init__(self, estimate: RecipeResourceEstimate):
        self.estimate = estimate
        details = estimate.errors or estimate.warnings or [estimate.summary()]
        super().__init__("Recipe resource preflight blocked the run: " + " ".join(details))


def estimate_recipe_resources(
    *,
    context: Any,
    dataset_id: str,
    recipe_spec: Any,
    params: Optional[Mapping[str, Any]] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> RecipeResourceEstimate:
    """Estimate recipe memory/disk requirements without materialising a dataset."""

    from .data.dataset_access import (
        dataset_capabilities,
        dataset_dtypes,
        dataset_row_count,
        list_dataset_columns,
        mapped_column,
    )
    from .feature_columns import parse_column_list

    params = dict(params or {})
    access = RecipeDataAccess.coerce(
        getattr(recipe_spec, "data_access", None),
        framework=str(getattr(recipe_spec, "framework", "") or ""),
    )

    columns = [str(value) for value in (feature_columns or []) if value]
    if not columns:
        columns = parse_column_list(
            params.get("feature_columns")
            or params.get("input_columns")
            or params.get("features")
        )
    if not columns and _truthy(params.get("auto_feature_columns")):
        excluded = {
            str(value)
            for value in (
                params.get("record_id_column") or mapped_column(context, dataset_id, "record_id"),
                params.get("target_column") or mapped_column(context, dataset_id, "target_label"),
                params.get("image_column") or mapped_column(context, dataset_id, "image.uri"),
            )
            if value
        }
        columns = [
            column
            for column in list_dataset_columns(context, dataset_id)
            if column not in excluded
        ]

    row_count = dataset_row_count(context, dataset_id)
    estimated_training_rows = _estimated_training_rows(row_count, params)
    dtypes = dataset_dtypes(context, dataset_id)
    widths = [_dtype_width(dtypes.get(column)) for column in columns]
    bytes_per_row = max(1, sum(widths) + 16)
    source_rows_for_memory = row_count if access.materializes_training_partition else estimated_training_rows
    source_bytes = (
        int(source_rows_for_memory) * bytes_per_row
        if source_rows_for_memory is not None
        else None
    )

    batch_size = None
    if access.mode != RecipeDataAccessMode.MATERIALIZED:
        batch_size = _positive_int(
            params.get(access.batch_size_param),
            default=access.default_batch_size,
        )

    peak_memory = None
    if access.mode == RecipeDataAccessMode.MATERIALIZED:
        if source_bytes is not None:
            peak_memory = int(source_bytes * max(1.0, float(access.memory_multiplier)))
            peak_memory += int(access.fixed_overhead_bytes)
    elif batch_size is not None:
        peak_memory = int(batch_size * bytes_per_row * max(1.0, float(access.working_set_multiplier)))
        peak_memory += int(access.fixed_overhead_bytes)

    disk_bytes = None
    if estimated_training_rows is not None and access.disk_multiplier > 0:
        disk_bytes = int(
            estimated_training_rows
            * bytes_per_row
            * max(0.0, float(access.disk_multiplier))
        )

    available_memory = _available_memory_bytes()
    capabilities = dataset_capabilities(context, dataset_id)
    supports_batch_scan = _supports_batch_scan(capabilities)

    non_numeric = [
        column
        for column in columns
        if dtypes.get(column) and not _is_numeric_dtype(dtypes[column])
    ]
    warnings: List[str] = []
    errors: List[str] = []

    if not columns and str(getattr(recipe_spec, "modality", "")).lower() == "tabular":
        warnings.append("No feature columns are currently selected, so the estimate excludes model inputs.")

    if access.requires_batch_scan and supports_batch_scan is False:
        errors.append(
            "This recipe requires DatasetSource batch scanning, but the selected dataset source does not advertise that capability."
        )

    if access.requires_numeric_features and non_numeric:
        errors.append(
            "This recipe currently supports numeric feature columns only. Unsupported columns: "
            + ", ".join(f"`{column}`" for column in non_numeric)
            + "."
        )

    memory_block = False
    if access.mode == RecipeDataAccessMode.MATERIALIZED:
        if peak_memory is None:
            warnings.append(
                "The dataset row count is unavailable, so the materialised-memory requirement cannot be estimated reliably."
            )
        else:
            if access.blocking_bytes > 0 and peak_memory >= access.blocking_bytes:
                memory_block = True
                errors.append(
                    f"Estimated peak memory {format_bytes(peak_memory)} exceeds the recipe's "
                    f"{format_bytes(access.blocking_bytes)} materialisation limit."
                )
            elif access.warning_bytes > 0 and peak_memory >= access.warning_bytes:
                warnings.append(
                    f"Estimated peak memory is {format_bytes(peak_memory)} for this materialised recipe."
                )

            if available_memory:
                ratio = peak_memory / max(1, available_memory)
                if ratio >= 0.8:
                    memory_block = True
                    errors.append(
                        f"Estimated peak memory is {ratio:.0%} of currently available system memory "
                        f"({format_bytes(available_memory)})."
                    )
                elif ratio >= 0.5:
                    warnings.append(
                        f"Estimated peak memory is {ratio:.0%} of currently available system memory."
                    )

    override_requested = _truthy(params.get("allow_unsafe_materialization"))
    override_applied = bool(
        memory_block
        and override_requested
        and access.allow_memory_override
        and not [error for error in errors if "numeric feature" in error or "batch scanning" in error]
    )
    if override_applied:
        errors = [
            error
            for error in errors
            if "Estimated peak memory" not in error
        ]
        warnings.append(
            "The materialisation safety limit was explicitly overridden for this run. The process may exhaust memory."
        )
        memory_block = False

    blocked = bool(errors)
    status = "blocked" if blocked else "warning" if warnings else "safe"
    return RecipeResourceEstimate(
        dataset_id=str(dataset_id),
        access=access,
        row_count=row_count,
        estimated_training_rows=estimated_training_rows,
        feature_columns=list(columns),
        non_numeric_feature_columns=non_numeric,
        estimated_source_bytes=source_bytes,
        estimated_peak_memory_bytes=peak_memory,
        estimated_disk_bytes=disk_bytes,
        available_memory_bytes=available_memory,
        batch_size=batch_size,
        source_supports_batch_scan=supports_batch_scan,
        status=status,
        blocked=blocked,
        override_applied=override_applied,
        warnings=warnings,
        errors=errors,
    )


def ensure_recipe_resources_allowed(estimate: RecipeResourceEstimate) -> None:
    if estimate.blocked:
        raise RecipeResourceLimitError(estimate)


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    amount = float(max(0, value))
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PiB"


def format_count(value: int) -> str:
    return f"{int(value):,} rows"


def _estimated_training_rows(row_count: Optional[int], params: Mapping[str, Any]) -> Optional[int]:
    explicit_ids = (
        params.get("training_row_ids")
        or params.get("selected_row_ids")
        or params.get("row_ids")
    )
    if explicit_ids is not None and not isinstance(explicit_ids, str):
        try:
            return len(explicit_ids)
        except Exception:
            pass
    if row_count is None:
        return None

    validation_fraction = 0.0
    test_fraction = 0.0
    if str(params.get("protocol_validation_source") or "split") == "split":
        validation_fraction = _fraction(params.get("protocol_validation_size"), 0.1)
    if str(params.get("protocol_test_source") or "split") == "split":
        test_fraction = _fraction(params.get("protocol_test_size"), 0.2)
    train_fraction = max(0.0, min(1.0, 1.0 - validation_fraction - test_fraction))
    return max(0, int(round(int(row_count) * train_fraction)))


def _fraction(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    return max(1, result)


def _dtype_width(dtype: Any) -> int:
    text = str(dtype or "").lower()
    if "bool" in text:
        return 1
    if any(token in text for token in ("int8", "uint8")):
        return 1
    if any(token in text for token in ("int16", "uint16", "float16")):
        return 2
    if any(token in text for token in ("int32", "uint32", "float32")):
        return 4
    if any(token in text for token in ("int64", "uint64", "float64", "double", "datetime", "timedelta")):
        return 8
    if any(token in text for token in ("decimal", "numeric")):
        return 16
    if any(token in text for token in ("category", "string", "object", "utf", "char")):
        return 64
    return 32


def _is_numeric_dtype(dtype: Any) -> bool:
    text = str(dtype or "").lower()
    return any(
        token in text
        for token in ("int", "uint", "float", "double", "decimal", "numeric", "bool")
    )


def _supports_batch_scan(capabilities: Any) -> Optional[bool]:
    if capabilities is None:
        return None
    names = (
        "batch_scan",
        "batch_scanning",
        "scan_batches",
        "supports_batch_scan",
        "iter_batches",
    )
    if isinstance(capabilities, Mapping):
        for name in names:
            if name in capabilities:
                return bool(capabilities[name])
        return None
    for name in names:
        if hasattr(capabilities, name):
            return bool(getattr(capabilities, name))
    return None


def _available_memory_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
