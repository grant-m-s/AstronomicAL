from __future__ import annotations

from typing import Any, Mapping

from .dataset_access import infer_column_bindings, list_dataset_columns
from ..protocol import DataBinding
from ..feature_columns import default_feature_columns, feature_columns_from_params

def build_data_binding(context: Any, dataset_id: str, spec: Any, params: Mapping[str, Any]):
    """Resolve and validate dataset columns for a managed recipe run."""
    inferred = infer_column_bindings(context, dataset_id, spec, params=params)
    columns = list_dataset_columns(context, dataset_id)

    record_id_column = (
        params.get("record_id_column") or params.get("id_column")
        or inferred.get("record_id_column") or inferred.get("record_id")
    )
    if not record_id_column and "id" in columns:
        record_id_column = "id"

    target_column = (
        params.get("target_column") or params.get("label_column")
        or params.get("target") or params.get("label")
        or inferred.get("target_column") or inferred.get("target_label")
    )
    image_column = (
        params.get("image_column") or params.get("image_path_column")
        or params.get("image_uri_column") or inferred.get("image_column")
        or inferred.get("image_path") or inferred.get("image_uri")
    )

    input_columns = feature_columns_from_params(params)

    record_id_column = str(record_id_column) if record_id_column else ""
    target_column = str(target_column) if target_column else None
    image_column = str(image_column) if image_column else None

    recipe_cls = getattr(spec, "recipe_cls", None)
    execution_mode = str(getattr(recipe_cls, "execution_mode", None) or getattr(spec, "execution_mode", "freeform") or "freeform")
    task = str(getattr(spec, "task", "") or "").lower()
    modality = str(getattr(spec, "modality", "") or "").lower()

    if modality == "tabular" and not input_columns and _truthy(params.get("auto_feature_columns")):
        input_columns = default_feature_columns(
            context, dataset_id, record_id_column=record_id_column,
            target_column=target_column, image_column=image_column, params=params,
        )

    if image_column and image_column not in input_columns:
        input_columns.append(image_column)

    binding = DataBinding(
        record_id_column=record_id_column,
        target_column=target_column,
        input_columns=[str(column) for column in input_columns if column],
        image_column=image_column,
    )

    if execution_mode == "managed":
        _validate_managed_binding(binding=binding, task=task, modality=modality)

    missing_columns = []
    for column in (binding.record_id_column, binding.target_column, binding.image_column, *binding.input_columns):
        if column and column not in columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError("Resolved recipe column(s) are not present in the dataset: " + ", ".join(sorted(set(missing_columns))))

    return binding

def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

def _validate_managed_binding(*, binding: Any, task: str, modality: str) -> None:
    if not binding.record_id_column:
        raise ValueError("Managed recipes require a record-id column. Set `record_id_column`, map `record_id`, or add an id column.")
    if task in {"classification", "regression"} and not binding.target_column:
        raise ValueError(f"Managed {task} recipes require a target column. Set `target_column`, map `target_label`, or add a matching target/label column to the dataset.")
    if modality == "image" and not binding.image_column:
        raise ValueError("Managed image recipes require an image column. Set `image_column`, map `image.path`/`image.uri`, or add a matching image path column to the dataset.")
    if modality == "tabular" and not binding.input_columns:
        raise ValueError("Managed tabular recipes require input feature columns. Choose `feature_columns` in the launcher/AL panel, or enable `auto_feature_columns`.")
