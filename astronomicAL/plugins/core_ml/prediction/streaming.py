from __future__ import annotations

import time
import traceback
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetScan,
    DuckDBParquetDatasetSource,
    PandasDatasetSource,
)

from .. import artifacts as artifact_utils
from .. import contracts as contract_utils
from ..runtime import (
    MLPredictionExecutionError,
    check_cancelled,
    cleanup_ml_runtime,
    coerce_action_request,
    is_out_of_memory_error,
    publish as publish_event,
    put_artifact,
    request_dataset_id,
)
from ..serialization import json_safe
from . import action as legacy
from .join_source import PredictionColumnSource
from .online_metrics import OnlineEvaluation
from .panel_bridge import install_prediction_panel_bridge
from .provenance import StreamingProvenanceIndex
from .storage import (
    PredictionTableRef,
    PredictionTableWriter,
    delete_prediction_ref,
    prediction_output_root,
)

def predict_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:
    """Run inference/evaluation using bounded source batches."""

    started_at = time.time()
    request = coerce_action_request(request)
    params = dict(request.params or {})
    dataset_id = request_dataset_id(context, request, params)
    model_artifact_id = str(
        params.get("model_artifact_id") or request.artifact_id or ""
    ).strip()
    predictor = None
    failure = None
    try:
        if not dataset_id:
            raise ValueError("Prediction requires a dataset.")
        if not model_artifact_id:
            raise ValueError("Prediction requires model_artifact_id.")
        model_payload = context.artifacts.get(model_artifact_id)
        if not isinstance(model_payload, Mapping):
            raise TypeError(
                f"Artifact {model_artifact_id!r} is not an ml.model payload."
            )

        persist_existing = getattr(
            artifact_utils,
            "persist_existing_model_artifact",
            None,
        )
        if callable(persist_existing):
            # Already-durable artifacts may still use legacy class aliases or
            # retain the AL class universe only in checkpoint params. Promote
            # that metadata before validating the prediction output contract.
            model_payload = persist_existing(
                context=context,
                artifact_id=model_artifact_id,
            )

        compatibility = contract_utils.validate_model_for_dataset(
            context=context,
            model_artifact_id=model_artifact_id,
            dataset_id=str(dataset_id),
            target_column=params.get("target_column"),
            image_column=params.get("image_column"),
            feature_column_mapping=(
                params.get("feature_column_mapping")
                or params.get("feature_mapping")
            ),
            require_target_compatible=bool(
                params.get("require_target_compatible", False)
            ),
        )
        if not compatibility.can_predict:
            return json_safe(
                {
                    "ok": False,
                    "status": compatibility.status,
                    "dataset_id": dataset_id,
                    "model_artifact_id": model_artifact_id,
                    "compatibility_report": compatibility.to_dict(),
                    "errors": list(compatibility.errors),
                    "warnings": list(compatibility.warnings),
                }
            )

        contract = contract_utils.ensure_model_contract(
            context=context,
            model_artifact_id=model_artifact_id,
            model_payload=model_payload,
            persist=True,
        )
        predictor = legacy.make_predictor(
            context=context,
            dataset_id=str(dataset_id),
            model_artifact_id=model_artifact_id,
            model_payload=model_payload,
            compatibility=compatibility,
            request=request,
            scope=params.get("scope"),
            cancel_token=cancel_token,
            framework=contract.get("framework") or model_payload.get("framework"),
            modality=(
                contract.get("modality")
                or model_payload.get("modality")
                or "tabular"
            ),
        )
        return _run_streaming_prediction(predictor)
    except MLPredictionExecutionError:
        raise
    except Exception as exc:
        failure = exc
        failure_payload = {
            "ok": False,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id or None,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "out_of_memory": is_out_of_memory_error(exc),
            "elapsed_seconds": time.time() - started_at,
            "traceback": traceback.format_exc(limit=25),
        }
        raise MLPredictionExecutionError(
            str(exc),
            failure_payload=json_safe(failure_payload),
        ) from None
    finally:
        if predictor is not None:
            for attribute in (
                "_model",
                "model",
                "_recipe",
                "recipe",
                "_recipe_run",
                "_pre",
                "_label_encoder",
                "_transform",
                "_bundle",
                "_checkpoint",
            ):
                try:
                    setattr(predictor, attribute, None)
                except Exception:
                    pass
        cleanup_ml_runtime(
            reason="prediction failed" if failure is not None else "prediction finished",
            aggressive=failure is not None or is_out_of_memory_error(failure or ""),
        )

def _run_streaming_prediction(predictor: Any) -> Dict[str, Any]:
    check_cancelled(predictor.cancel_token)
    predictor.reconstruct()
    source = predictor.context.datasets.get_source(predictor.dataset_id)
    columns = list(dict.fromkeys(
        str(column)
        for column in predictor.read_columns()
        if column and not _uses_index(column)
    ))
    record_id_column = predictor.binding.get("record_id_column")
    if record_id_column and not _uses_index(record_id_column) and record_id_column not in columns:
        columns.append(str(record_id_column))
    if (
        predictor.scope == legacy.SCOPE_EVALUATION
        and predictor.target_column
        and predictor.target_column not in columns
    ):
        columns.append(str(predictor.target_column))

    params = dict(predictor.params or {})
    source_batch_size = max(
        1,
        int(params.get("prediction_source_batch_size") or 8192),
    )
    inline_limit = max(
        0,
        int(params.get("prediction_inline_limit") or 1000),
    )
    save_predictions = bool(params.get("save_predictions", True))
    requested_ids = list(getattr(predictor.request, "row_ids", None) or [])
    limit = _row_limit(params)
    if limit is not None:
        requested_ids = requested_ids[:limit] if requested_ids else requested_ids

    estimated_count = (
        len(requested_ids)
        if requested_ids
        else _bounded_row_count(source, limit)
    )
    if not save_predictions and (
        estimated_count is None or estimated_count > inline_limit
    ):
        raise ValueError(
            "save_predictions=False is limited to bounded runs no larger than "
            f"prediction_inline_limit ({inline_limit})."
        )

    output_root = prediction_output_root(predictor.context, params)
    writer: Optional[PredictionTableWriter] = None
    prediction_ref: Optional[PredictionTableRef] = None
    if save_predictions:
        writer = PredictionTableWriter(
            root=output_root,
            run_id=predictor.run_id,
            dataset_id=predictor.dataset_id,
            model_artifact_id=predictor.model_artifact_id,
            storage_format=str(params.get("prediction_storage_format") or "auto"),
            columns=_prediction_table_columns(predictor),
            column_types=_prediction_table_column_types(predictor),
        )

    inline_records: list[Dict[str, Any]] = []
    inline_rows: list[Dict[str, Any]] = []
    preview_rows: list[Dict[str, Any]] = []
    failed_rows: list[Dict[str, Any]] = []
    total_count = 0
    input_row_count = 0
    table_columns: list[str] = []
    provenance_counts: Dict[str, int] = defaultdict(int)
    evaluator = OnlineEvaluation(predictor.task)

    try:
        with StreamingProvenanceIndex(
            predictor.context,
            predictor.model_payload,
        ) as provenance:
            for batch in _iter_prediction_batches(
                source=source,
                columns=columns,
                record_id_column=predictor.binding.get("record_id_column"),
                requested_ids=requested_ids,
                batch_size=source_batch_size,
                limit=limit,
            ):
                check_cancelled(predictor.cancel_token)
                frame = batch.frame
                if frame.empty:
                    continue
                input_row_count += int(len(frame))
                predictor.failed_rows = []
                records = list(predictor.predict_records(frame) or [])
                predictor._apply_abstention(records)
                failed_rows.extend(list(predictor.failed_rows or []))
                predictor.failed_rows = []
                _attach_truth(
                    records,
                    frame,
                    record_id_column=predictor.binding.get("record_id_column"),
                    target_column=predictor.target_column,
                    scope=predictor.scope,
                )
                row_ids = [
                    contract_utils.safe_record_id(
                        record.get("record_id", record.get("row_id"))
                    )
                    for record in records
                ]
                classifications = provenance.classify_many(
                    predictor.dataset_id,
                    row_ids,
                )
                for record, role in zip(records, classifications):
                    record["data_provenance"] = role
                    provenance_counts[role] += 1

                evaluator.update_records(records)
                table_rows = contract_utils.prediction_table_rows(
                    records=records,
                    row_ids=row_ids,
                    output_schema=predictor.output_schema,
                    model_artifact_id=predictor.model_artifact_id,
                    prediction_run_id=predictor.run_id,
                )
                for table_row, role in zip(table_rows, classifications):
                    table_row["data_provenance"] = role
                if table_rows and not table_columns:
                    table_columns = list(table_rows[0].keys())
                if writer is not None:
                    writer.write_rows(table_rows)
                if len(preview_rows) < 25:
                    preview_rows.extend(table_rows[: 25 - len(preview_rows)])
                remaining = inline_limit - len(inline_records)
                if remaining > 0:
                    inline_records.extend(records[:remaining])
                    inline_rows.extend(table_rows[:remaining])
                total_count += len(records)

            _validate_prediction_counts(
                predictor=predictor,
                input_row_count=input_row_count,
                prediction_count=total_count,
                estimated_count=estimated_count,
                failed_rows=failed_rows,
            )
            if writer is not None:
                prediction_ref = writer.finalize()
                writer = None
                table_columns = list(prediction_ref.columns)

            evaluation = evaluator.finalize(
                scope=predictor.scope,
                selection_metric=(
                    predictor.model_payload.get("selection_metric")
                    or params.get("selection_metric")
                ),
            )
            evaluation_report_id = _write_evaluation_report(
                predictor,
                evaluation=evaluation,
                provenance=provenance.summary(provenance_counts),
            )
            payload = _build_payload(
                predictor,
                inline_records=inline_records,
                inline_rows=inline_rows,
                preview_rows=preview_rows,
                total_count=total_count,
                table_columns=table_columns,
                prediction_ref=prediction_ref,
                failed_rows=failed_rows,
                provenance=provenance.summary(provenance_counts),
                evaluation=evaluation,
                evaluation_report_id=evaluation_report_id,
            )

        artifact_type = getattr(
            getattr(artifact_utils, "ARTIFACTS", None),
            "PREDICTIONS",
            "ml.predictions",
        )
        artifact_id = put_artifact(
            predictor.context,
            artifact_type,
            payload,
            dataset_id=predictor.dataset_id,
            row_ids=[row.get("record_id") for row in inline_rows],
            params={"model_artifact_id": predictor.model_artifact_id, **params},
            persist=True,
            required=True,
        )
        attached_columns: list[str] = []
        append_columns = params.get(
            "append_prediction_columns",
            params.get("register_prediction_dataset", True),
        )
        if bool(append_columns):
            attached_columns = _register_prediction_dataset(
                predictor,
                payload=payload,
                artifact_id=artifact_id,
                prediction_ref=prediction_ref,
                inline_rows=inline_rows,
                total_count=total_count,
            )
        updated_dataset_id = predictor.dataset_id if attached_columns else None
        _publish_prediction_events(
            predictor,
            artifact_id=artifact_id,
            updated_dataset_id=updated_dataset_id,
            attached_columns=attached_columns,
            total_count=total_count,
            provenance=payload["provenance"],
        )
        install_prediction_panel_bridge()
        checkpoint_sha = legacy._checkpoint_sha256(predictor.model_payload)
        return json_safe(
            {
                "ok": True,
                "scope": predictor.scope,
                "artifact_id": artifact_id,
                "evaluation_report_artifact_id": evaluation_report_id,
                "derived_dataset_id": None,
                "updated_dataset_id": updated_dataset_id,
                "prediction_columns_attached": bool(attached_columns),
                "prediction_columns": attached_columns,
                "dataset_id": predictor.dataset_id,
                "model_artifact_id": predictor.model_artifact_id,
                "count": total_count,
                "provenance": payload["provenance"],
                "evaluation": evaluation,
                "recipe_version_check": predictor.recipe_version_check,
                "checkpoint_sha256": checkpoint_sha,
                "prediction_ref": (
                    prediction_ref.to_dict() if prediction_ref is not None else None
                ),
                "audit_gaps": legacy.AUDIT_GAPS,
                "recommended_color_columns": payload.get("visualisation", {}).get(
                    "recommended_color_columns",
                    [],
                ),
                "prediction_preview": preview_rows,
                "prediction_table_columns": table_columns,
                "failed_image_row_count": len(failed_rows),
                "failed_image_rows": failed_rows[:25],
            }
        )
    except Exception:
        if writer is not None:
            writer.abort()
        delete_prediction_ref(prediction_ref)
        raise

def _iter_prediction_batches(
    *,
    source: Any,
    columns: Sequence[str],
    record_id_column: Optional[str],
    requested_ids: Sequence[Any],
    batch_size: int,
    limit: Optional[int],
) -> Iterator[DatasetBatch]:
    if requested_ids:
        if not record_id_column or _uses_index(record_id_column):
            raise ValueError(
                "Selected-row streaming prediction requires a mapped record_id column."
            )
        selected_columns = list(dict.fromkeys([*columns, str(record_id_column)]))
        row_offset = 0
        for batch_index, start in enumerate(range(0, len(requested_ids), batch_size)):
            batch_ids = requested_ids[start : start + batch_size]
            frame = source.get_rows_by_ids(
                batch_ids,
                id_column=str(record_id_column),
                columns=selected_columns,
            )
            if not frame.empty:
                order = {str(value): index for index, value in enumerate(batch_ids)}
                frame = frame.assign(
                    __prediction_order=frame[str(record_id_column)].map(
                        lambda value: order.get(str(value), len(order))
                    )
                ).sort_values("__prediction_order", kind="stable")
                frame = frame.drop(columns=["__prediction_order"]).reset_index(drop=True)
            yield DatasetBatch(
                frame=frame,
                batch_index=batch_index,
                row_offset=row_offset,
            )
            row_offset += len(frame)
        return

    scan = DatasetScan(
        columns=tuple(columns),
        batch_size=int(batch_size),
        limit=limit,
    )
    for batch in _iter_source_scan_batches(source, scan):
        frame = batch.frame
        if not record_id_column or _uses_index(record_id_column):
            frame = frame.copy()
            frame.index = range(batch.row_offset, batch.row_offset + len(frame))
        yield DatasetBatch(
            frame=frame,
            batch_index=batch.batch_index,
            row_offset=batch.row_offset,
        )

def _iter_source_scan_batches(
    source: Any,
    scan: DatasetScan,
) -> Iterator[DatasetBatch]:
    """Scan a source in bounded batches across old and new source contracts.

    New sources provide ``iter_batches`` directly. Older composed sources such
    as ``LazyPredictionJoinSource`` may only expose positional row access. The
    positional adapter keeps memory bounded and deliberately does not fall back
    to ``to_pandas``/``get_df``.
    """

    iterator_factory = getattr(source, "iter_batches", None)
    if callable(iterator_factory):
        yielded = False
        try:
            for batch in iterator_factory(scan):
                yielded = True
                yield batch
            return
        except NotImplementedError:
            if yielded:
                raise

    yield from _iter_positional_source_batches(source, scan)

def _iter_positional_source_batches(
    source: Any,
    scan: DatasetScan,
) -> Iterator[DatasetBatch]:
    getter = getattr(source, "get_row_by_position", None)
    if not callable(getter):
        raise TypeError(
            f"Dataset source {type(source).__name__!r} supports neither "
            "iter_batches() nor get_row_by_position(); bounded prediction "
            "cannot scan it safely."
        )

    count_method = getattr(source, "row_count", None)
    total_rows: Optional[int] = None
    if callable(count_method):
        try:
            value = count_method()
        except Exception:
            value = None
        if value is not None:
            total_rows = max(0, int(value))
    if scan.limit is not None:
        requested_limit = max(0, int(scan.limit))
        total_rows = (
            requested_limit
            if total_rows is None
            else min(total_rows, requested_limit)
        )

    columns = list(scan.columns) if scan.columns else None
    batch_size = max(1, int(scan.batch_size))
    position = 0
    row_offset = 0
    batch_index = 0

    while total_rows is None or position < total_rows:
        rows: list[pd.DataFrame] = []
        reached_end = False
        batch_start = position

        for _ in range(batch_size):
            if total_rows is not None and position >= total_rows:
                break

            current_position = position
            position += 1
            row = getter(current_position, columns=columns)
            if row is None or row.empty:
                if total_rows is None:
                    reached_end = True
                    break
                raise RuntimeError(
                    f"Dataset source {type(source).__name__!r} returned no row "
                    f"at position {current_position}, although row_count() "
                    f"reported {total_rows} row(s)."
                )
            if len(row) != 1:
                raise RuntimeError(
                    f"Dataset source {type(source).__name__!r} returned "
                    f"{len(row)} rows for position {current_position}; exactly "
                    "one row is required."
                )
            rows.append(row)

        if rows:
            frame = pd.concat(rows, axis=0, copy=False)
            yield DatasetBatch(
                frame=frame,
                batch_index=batch_index,
                row_offset=row_offset,
            )
            row_offset += len(frame)
            batch_index += 1

        if reached_end or not rows:
            break
        if position == batch_start:
            raise RuntimeError(
                f"Dataset source {type(source).__name__!r} made no progress "
                "during bounded positional scanning."
            )

def _attach_truth(
    records: Sequence[Dict[str, Any]],
    frame: pd.DataFrame,
    *,
    record_id_column: Optional[str],
    target_column: Optional[str],
    scope: str,
) -> None:
    if scope != legacy.SCOPE_EVALUATION or not target_column:
        return
    if target_column not in frame.columns:
        return
    if record_id_column and not _uses_index(record_id_column) and record_id_column in frame.columns:
        truth = {
            contract_utils.safe_record_id(row_id): value
            for row_id, value in zip(frame[record_id_column], frame[target_column])
        }
    else:
        truth = {
            contract_utils.safe_record_id(row_id): value
            for row_id, value in zip(frame.index, frame[target_column])
        }
    for record in records:
        row_id = contract_utils.safe_record_id(
            record.get("record_id", record.get("row_id"))
        )
        if row_id in truth:
            record["y_true"] = truth[row_id]

def _build_payload(
    predictor: Any,
    *,
    inline_records: Sequence[Mapping[str, Any]],
    inline_rows: Sequence[Mapping[str, Any]],
    preview_rows: Sequence[Mapping[str, Any]],
    total_count: int,
    table_columns: Sequence[str],
    prediction_ref: Optional[PredictionTableRef],
    failed_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    evaluation: Optional[Mapping[str, Any]],
    evaluation_report_id: Optional[str],
) -> Dict[str, Any]:
    row_ids = [
        contract_utils.safe_record_id(
            record.get("record_id", record.get("row_id"))
        )
        for record in inline_records
    ]
    input_binding = {**predictor.binding, "transform": predictor.transform_desc}
    if failed_rows:
        input_binding["failed_image_rows"] = list(failed_rows[:100])
        input_binding["failed_image_row_count"] = len(failed_rows)
    payload = contract_utils.build_predictions_payload(
        context=predictor.context,
        run_id=predictor.run_id,
        dataset_id=predictor.dataset_id,
        model_artifact_id=predictor.model_artifact_id,
        model_payload=predictor.model_payload,
        records=inline_records,
        row_ids=row_ids,
        input_binding=input_binding,
        compatibility_report=predictor.compatibility,
        params=predictor.params,
        prediction_scope=predictor.scope,
    )
    checkpoint_sha = legacy._checkpoint_sha256(predictor.model_payload)
    payload["row_count"] = int(total_count)
    payload["row_ids_inline_complete"] = total_count <= len(row_ids)
    payload["records_inline_complete"] = total_count <= len(inline_records)
    payload["prediction_ref"] = (
        prediction_ref.to_dict() if prediction_ref is not None else None
    )
    payload["prediction_table"] = {
        "join_key": "record_id",
        "rows": list(inline_rows) if total_count <= len(inline_rows) else [],
        "preview": list(preview_rows),
        "columns": list(table_columns),
        "row_count": int(total_count),
        "inline_complete": total_count <= len(inline_rows),
        "storage": prediction_ref.to_dict() if prediction_ref is not None else None,
    }
    payload["provenance"] = dict(provenance)
    visualisation = payload.setdefault("visualisation", {})
    recommended = list(visualisation.get("recommended_color_columns") or [])
    if "data_provenance" not in recommended:
        recommended.insert(0, "data_provenance")
    visualisation["recommended_color_columns"] = recommended
    payload["reproducibility"] = legacy._reproducibility_manifest(
        model_payload=predictor.model_payload,
        transform_desc=predictor.transform_desc,
        checkpoint_sha256=checkpoint_sha,
        recipe_version_check=predictor.recipe_version_check,
    )
    payload["recipe_version_check"] = predictor.recipe_version_check
    payload["scope"] = predictor.scope
    payload["audit_gaps"] = legacy.AUDIT_GAPS
    if evaluation is not None:
        payload["evaluation"] = dict(evaluation)
        payload["evaluation_report_artifact_id"] = evaluation_report_id
    return json_safe(payload)

def _write_evaluation_report(
    predictor: Any,
    *,
    evaluation: Optional[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Optional[str]:
    if evaluation is None:
        return None
    artifact_type = getattr(
        getattr(artifact_utils, "ARTIFACTS", None),
        "EVALUATION_REPORT",
        "ml.evaluation_report",
    )
    payload = {
        "artifact_type": "ml.evaluation_report",
        "schema_version": legacy.EVALUATION_SCHEMA_VERSION,
        "run_id": predictor.run_id,
        "dataset_id": predictor.dataset_id,
        "model_artifact_id": predictor.model_artifact_id,
        "evaluation": dict(evaluation),
        "provenance": dict(provenance),
        "created_at": time.time(),
    }
    return put_artifact(
        predictor.context,
        artifact_type,
        payload,
        dataset_id=predictor.dataset_id,
        params={"model_artifact_id": predictor.model_artifact_id},
        persist=True,
        required=True,
    )

def _register_prediction_dataset(
    predictor: Any,
    *,
    payload: Mapping[str, Any],
    artifact_id: str,
    prediction_ref: Optional[PredictionTableRef],
    inline_rows: Sequence[Mapping[str, Any]],
    total_count: int,
) -> list[str]:
    """Replace prediction columns on the existing dataset ID."""

    datasets = getattr(predictor.context, "datasets", None)
    if datasets is None:
        return []
    get_source = getattr(datasets, "get_source", None)
    ensure_source = getattr(datasets, "ensure_source_registered", None)
    upsert_overlay = getattr(datasets, "upsert_column_overlay", None)
    get_dataset = getattr(datasets, "get", None)
    if not callable(get_source):
        return []

    prediction_source = None
    prediction_columns: list[str] = []
    if prediction_ref is not None and prediction_ref.parquet_parts:
        prediction_columns = list(prediction_ref.columns)
        prediction_source = DuckDBParquetDatasetSource(
            prediction_ref.parquet_parts,
            dataset_name=f"Predictions for {predictor.dataset_id}",
            columns_hint=prediction_columns,
            row_count_hint=int(total_count),
        )
    elif total_count <= len(inline_rows):
        prediction_frame = pd.DataFrame.from_records(list(inline_rows))
        prediction_columns = [
            str(column) for column in prediction_frame.columns
        ]
        prediction_source = PandasDatasetSource(prediction_frame)

    if prediction_source is None:
        raise RuntimeError(
            "Appending prediction columns requires Parquet prediction output. "
            "Install DuckDB, PyArrow, or Fastparquet, or run a bounded "
            "prediction small enough to remain inline."
        )

    attached_columns = [
        column for column in prediction_columns if column != "record_id"
    ]
    storage = (
        prediction_ref.to_dict() if prediction_ref is not None else None
    )
    if callable(upsert_overlay):
        upsert_overlay(
            predictor.dataset_id,
            overlay_name="core.ml.predictions",
            overlay_source=prediction_source,
            base_record_id_column=str(
                predictor.binding.get("record_id_column") or ""
            ),
            overlay_record_id_column="record_id",
            columns=attached_columns,
            preserve_base_on_missing=False,
            origin="core.ml.predict",
            prediction_columns=attached_columns,
            predictions_artifact_id=artifact_id,
            model_artifact_id=predictor.model_artifact_id,
            prediction_run_id=predictor.run_id,
            prediction_row_count=int(total_count),
            prediction_storage=storage,
            prediction_origin="core.ml.predict",
        )
    else:
        if not callable(ensure_source):
            return []
        base_source = get_source(predictor.dataset_id)
        joined_source = PredictionColumnSource(
            base_source=base_source,
            prediction_source=prediction_source,
            base_record_id_column=predictor.binding.get(
                "record_id_column"
            ),
            prediction_record_id_column="record_id",
            prediction_columns=prediction_columns,
        )
        dataset = (
            get_dataset(predictor.dataset_id)
            if callable(get_dataset)
            else None
        )
        name = getattr(dataset, "name", None) or str(
            predictor.dataset_id
        )
        ensure_source(
            predictor.dataset_id,
            joined_source,
            name=name,
            prediction_columns=attached_columns,
            predictions_artifact_id=artifact_id,
            model_artifact_id=predictor.model_artifact_id,
            prediction_run_id=predictor.run_id,
            prediction_row_count=int(total_count),
            prediction_storage=storage,
            prediction_origin="core.ml.predict",
        )
        event_payload = {
            "dataset_id": predictor.dataset_id,
            "updated_dataset_id": predictor.dataset_id,
            "predictions_artifact_id": artifact_id,
            "model_artifact_id": predictor.model_artifact_id,
            "prediction_run_id": predictor.run_id,
            "prediction_columns": attached_columns,
            "added_columns": attached_columns,
            "changed_columns": attached_columns,
            "change": "column.added",
            "schema_changed": True,
            "row_count": int(total_count),
            "origin": "core.ml.predict",
        }
        publish_event(
            predictor.context,
            "dataset.columns.changed",
            event_payload,
        )
        publish_event(
            predictor.context,
            "dataset.columns.updated",
            event_payload,
        )
        publish_event(
            predictor.context,
            "dataset.updated",
            event_payload,
        )

    publish_event(
        predictor.context,
        "ml.predictions.attached",
        {
            "dataset_id": predictor.dataset_id,
            "updated_dataset_id": predictor.dataset_id,
            "predictions_artifact_id": artifact_id,
            "model_artifact_id": predictor.model_artifact_id,
            "prediction_run_id": predictor.run_id,
            "prediction_columns": attached_columns,
            "row_count": int(total_count),
            "origin": "core.ml.predict",
        },
    )
    return attached_columns

def _prediction_table_columns(predictor: Any) -> list[str]:
    task = str(predictor.task or "classification").strip().lower()
    is_regression = task in {"regression", "regressor", "regress"}

    columns = [
        "record_id",
        "predicted_label",
        "prediction",
        "prediction_run_id",
        "model_artifact_id",
    ]
    if predictor.scope == legacy.SCOPE_EVALUATION:
        columns.extend(["true_label", "is_correct"])
    columns.extend(
        [
            "prediction_confidence",
            "confidence_source",
            "confidence_semantics",
            "least_confidence",
            "margin",
            "margin_uncertainty",
            "entropy",
            "active_learning_score",
        ]
    )

    # Probability columns are classification-only.  Saved profiles and older
    # artifacts may retain stale class aliases, but they must never alter the
    # physical schema of a regression prediction stream.
    if not is_regression:
        probability_columns = (
            predictor.output_schema.get("probability_columns") or {}
        )
        if isinstance(probability_columns, Mapping) and probability_columns:
            columns.extend(
                str(value) for value in probability_columns.values()
            )
        else:
            for class_name in (
                predictor.output_schema.get("classes")
                or predictor.output_schema.get("class_order")
                or []
            ):
                token = "".join(
                    character if character.isalnum() else "_"
                    for character in str(class_name)
                ).strip("_")
                columns.append(f"prob_{token or 'class'}")

    columns.append("data_provenance")
    return list(dict.fromkeys(columns))

def _prediction_table_column_types(predictor: Any) -> Dict[str, str]:
    task = str(predictor.task or "classification").lower()
    types: Dict[str, str] = {
        "record_id": "string",
        "prediction_run_id": "string",
        "model_artifact_id": "string",
        "confidence_source": "string",
        "confidence_semantics": "string",
        "data_provenance": "string",
        "is_correct": "boolean",
        "prediction_confidence": "float",
        "least_confidence": "float",
        "margin": "float",
        "margin_uncertainty": "float",
        "entropy": "float",
        "active_learning_score": "float",
    }
    if task == "regression":
        types.update(
            {
                "predicted_label": "float",
                "prediction": "float",
                "true_label": "float",
            }
        )
    else:
        types.update(
            {
                "predicted_label": "string",
                "prediction": "string",
                "true_label": "string",
            }
        )
    probability_columns = predictor.output_schema.get("probability_columns") or {}
    if isinstance(probability_columns, Mapping):
        for column in probability_columns.values():
            types[str(column)] = "float"
    for column in _prediction_table_columns(predictor):
        if column.startswith("prob_"):
            types[column] = "float"
    return types

def _publish_prediction_events(
    predictor: Any,
    *,
    artifact_id: str,
    updated_dataset_id: Optional[str],
    attached_columns: Sequence[str],
    total_count: int,
    provenance: Mapping[str, Any],
) -> None:
    publish_event(
        predictor.context,
        "ml.predictions.created",
        {
            "artifact_id": artifact_id,
            "dataset_id": predictor.dataset_id,
            "derived_dataset_id": None,
            "updated_dataset_id": updated_dataset_id,
            "prediction_columns_attached": bool(attached_columns),
            "prediction_columns": list(attached_columns),
            "model_artifact_id": predictor.model_artifact_id,
            "scope": predictor.scope,
            "provenance_verified": bool(provenance.get("verified")),
            "count": int(total_count),
        },
    )
    getter = getattr(getattr(predictor.context, "services", None), "get", None)
    if callable(getter):
        try:
            catalog = getter("core.ml.trained_model_catalog")
            refresh = getattr(catalog, "refresh", None)
            if callable(refresh):
                refresh()
        except Exception:
            pass

def _row_limit(params: Mapping[str, Any]) -> Optional[int]:
    """Return a positive prediction limit; zero means unlimited in the UI."""

    for key in ("max_rows", "row_limit"):
        value = params.get(key)
        if value in (None, ""):
            continue
        resolved = int(value)
        return resolved if resolved > 0 else None
    return None

def _validate_prediction_counts(
    *,
    predictor: Any,
    input_row_count: int,
    prediction_count: int,
    estimated_count: Optional[int],
    failed_rows: Sequence[Mapping[str, Any]],
) -> None:
    if input_row_count <= 0:
        if estimated_count is not None and estimated_count > 0:
            raise RuntimeError(
                f"Dataset {predictor.dataset_id!r} reports {estimated_count} row(s), "
                "but its streaming source yielded no rows for prediction."
            )
        return
    if prediction_count > 0:
        return

    detail = ""
    if failed_rows:
        first = dict(failed_rows[0])
        reason = (
            first.get("error")
            or first.get("message")
            or first.get("reason")
        )
        if reason:
            detail = f" First failure: {reason}"
    raise RuntimeError(
        f"Prediction read {input_row_count} row(s) from dataset "
        f"{predictor.dataset_id!r} but produced no predictions. "
        f"Failed input rows: {len(failed_rows)}.{detail}"
    )

def _bounded_row_count(source: Any, limit: Optional[int]) -> Optional[int]:
    try:
        count = source.row_count()
    except Exception:
        count = None
    if count is None:
        return limit
    return min(int(count), limit) if limit is not None else int(count)

def _uses_index(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "use index",
        "use_index",
        "__index__",
        "index",
    }