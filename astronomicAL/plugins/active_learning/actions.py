from __future__ import annotations

import json
import random
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from astronomicAL.platform.plugins.specs import ActionRequest

from . import state as al_state
from . import contracts as al_contracts
from . import strategies as _strategies_module

create_default_strategy_registry = _strategies_module.create_default_strategy_registry

ORIGIN = "core.active_learning"

def _column_like_recipe_params(params: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for key, value in dict(params or {}).items():
        if value in (None, "", [], {}):
            continue

        key_text = str(key)
        key_lower = key_text.lower()

        looks_like_column = key_lower.endswith("_column") or key_lower in {
            "target",
            "label",
            "label_column",
            "target_column",
            "image_column",
            "image_path_column",
            "image_uri_column",
            "record_id_column",
            "id_column",
            "group_column",
            "split_column",
            "mask_column",
        }

        if looks_like_column:
            out[key_text] = str(value)

    return out

def _feature_columns_from_recipe_params(params: Mapping[str, Any]) -> List[str]:
    values = (
        params.get("feature_columns")
        or params.get("input_columns")
        or params.get("features")
        or params.get("x_columns")
        or []
    )

    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values = parsed
            elif isinstance(parsed, str):
                values = [parsed]
            else:
                values = [text]
        except Exception:
            values = [
                part.strip()
                for chunk in text.splitlines()
                for part in chunk.split(",")
                if part.strip()
            ]

    elif isinstance(values, Mapping):
        values = values.keys()

    elif isinstance(values, (int, float)):
        values = [values]

    columns: List[str] = []

    for value in values or []:
        if value is None:
            continue
        column = str(value).strip()

        if not column or column == "Use Index":
            continue

        if column not in columns:
            columns.append(column)

    return columns

def _existing_dataset_columns(context: Any, dataset_id: str) -> set[str]:
    try:
        return {str(col) for col in context.datasets.list_columns(dataset_id)}

    except Exception:
        try:
            return {str(col) for col in context.datasets.get_df(dataset_id).columns}

        except Exception:
            return set()

def _resolve_training_required_columns(
    context: Any,
    *,
    dataset_id: str,
    recipe_params: Mapping[str, Any],
    target_column: str,
    id_column: Optional[str],
) -> List[str]:
    columns = _existing_dataset_columns(context, dataset_id)
    mappings = _dataset_mappings(context, dataset_id)

    needed: List[str] = []

    def add(column: Any) -> None:
        if column is None:
            return

        column = str(column).strip()

        if not column or column == "Use Index":
            return

        if column not in needed:
            needed.append(column)

    add(id_column)

    # Existing label column may be useful for class inference, but the AL
    # target column itself can also be newly created later.
    if target_column in columns:
        add(target_column)

    # Carry mapped source columns that recipes may infer from.
    for semantic in (
        "record_id",
        "target_label",
        "image.path",
        "image.uri",
        "image",
        "cutout.path",
        "mask.path",
        "mask",
    ):
        add(mappings.get(semantic))

    # Carry explicit recipe/protocol single-column params.
    for column in _column_like_recipe_params(recipe_params).values():
        add(column)

    # Carry explicit tabular model input features. The derived AL training
    # dataframe must physically contain every selected feature column.
    for column in _feature_columns_from_recipe_params(recipe_params):
        add(column)

    return needed

def _filter_mappings_to_existing_columns(
    mappings: Mapping[str, Any],
    columns: Iterable[str],
) -> Dict[str, str]:
    existing = {str(col) for col in columns}

    filtered: Dict[str, str] = {}

    for semantic, column in dict(mappings or {}).items():
        if column is None:
            continue

        text = str(column)

        if text == "Use Index" or text in existing:
            filtered[str(semantic)] = text

    return filtered

def start_session_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _resolve_dataset_id(context, request, params)
    recipe_id = str(params.get("recipe_id") or "").strip()

    if not recipe_id:
        raise ValueError("start_session requires recipe_id.")

    label_options = al_state.parse_label_options(params.get("label_options"))
    target_column = str(params.get("target_column") or "al_label")

    if not label_options:
        label_options = _infer_label_options_from_dataset_column(
            context,
            dataset_id=dataset_id,
            column=target_column,
        )

    if not label_options:
        raise ValueError(
            "Active Learning currently supports classification sessions and "
            "requires a complete label_options/class list."
        )

    seed = int(params.get("seed", 42))
    initial_k = max(0, int(params.get("initial_k", params.get("k", 20))))
    make_selection = bool(params.get("make_selection", True))

    preflight = _build_session_preflight(
        context,
        dataset_id=dataset_id,
        recipe_id=recipe_id,
        params=params,
        labelled_count=0,
        label_counts={},
        cancel_token=cancel_token,
    )
    preflight.raise_for_errors()

    _check_cancelled(cancel_token)
    selected_row_ids = _sample_dataset_row_ids(
        context,
        dataset_id=dataset_id,
        k=initial_k,
        seed=seed,
    )

    session = al_state.create_session(
        dataset_id=dataset_id,
        pool_dataset_id=str(params.get("pool_dataset_id") or dataset_id),
        validation_dataset_id=str(params.get("validation_dataset_id") or ""),
        test_dataset_id=str(params.get("test_dataset_id") or ""),
        recipe_id=recipe_id,
        al_protocol=str(params.get("al_protocol") or "review"),
        label_options=label_options,
        seed=seed,
        target_column=target_column,
        contract=preflight.contract,
    )

    records = [
        {
            "row_id": row_id,
            "selection_rank": idx,
            "rank": idx,
            "informativeness_score": None,
            "active_learning_strategy": "initial_random",
            "source": "initial_random",
        }
        for idx, row_id in enumerate(selected_row_ids, start=1)
    ]

    batch_payload = _batch_payload(
        dataset_id=dataset_id,
        session=session,
        strategy_id="initial_random",
        records=records,
        params=params,
        predictions_artifact_id=None,
        kind="initial_random",
        rank_stats={
            "strategy_id": "initial_random",
            "ranked_count": len(records),
            "requested_k": initial_k,
            "seed": seed,
        },
    )

    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=dataset_id,
        row_ids=selected_row_ids,
        params={
            "session_id": session["session_id"],
            "strategy_id": "initial_random",
            "seed": seed,
            "k": initial_k,
        },
    )

    session = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id="initial_random",
        row_ids=selected_row_ids,
        predictions_artifact_id=None,
        kind="initial_random",
    )
    session_artifact_id = _put_session(context, session)

    if make_selection and selected_row_ids:
        _set_ranked_selection(
            context,
            dataset_id=dataset_id,
            row_ids=selected_row_ids,
            session_artifact_id=session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id="initial_random",
            origin=f"{ORIGIN}.start_session",
        )

    _publish(
        context,
        "al.session.created",
        {
            "session_artifact_id": session_artifact_id,
            "session_id": session["session_id"],
            "dataset_id": dataset_id,
            "initial_count": len(selected_row_ids),
            "contract": preflight.contract,
        },
    )

    return {
        "ok": True,
        "session_artifact_id": session_artifact_id,
        "session_id": session["session_id"],
        "batch_artifact_id": batch_artifact_id,
        "dataset_id": dataset_id,
        "row_ids": selected_row_ids,
        "counts": al_state.counts(session),
        "contract": preflight.contract,
        "preview": records[:25],
    }

def profile_data_contract_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Resolve and inspect the recipe/data contract used by the AL panel."""

    request = _coerce_request(request)
    params = dict(request.params or {})
    dataset_id = _resolve_dataset_id(context, request, params)

    recipe_id = str(
        params.get("recipe_id")
        or ""
    ).strip()

    if not recipe_id:
        raise ValueError(
            "profile_data_contract requires recipe_id."
        )

    registry = None
    services = getattr(context, "services", None,)

    if services is not None:
        try:
            registry = services.get(
                "core.ml.recipe_registry"
            )
        except Exception:
            registry = None

    if registry is None:
        raise RuntimeError(
            "Enable core.ml before profiling an AL recipe contract."
        )

    try:
        recipe_spec = registry.get(recipe_id)
    except Exception as exc:
        raise ValueError(
            f"Unknown core.ml recipe: {recipe_id!r}."
        ) from exc

    report = al_contracts.build_session_contract(
        context,
        recipe_spec=recipe_spec,
        pool_dataset_id=dataset_id,
        validation_dataset_id=str(params.get("validation_dataset_id") or ""),
        test_dataset_id=str(params.get("test_dataset_id") or ""),
        target_column=str(params.get("target_column") or "al_label"),
        feature_columns=(params.get("feature_columns") or []),
        image_column=params.get("image_column"),
        mask_column=params.get("mask_column"),
        recipe_params=(params.get("recipe_params") or {}),
        protocol_params=(
            params.get("protocol_params")
            or params
        ),
        labelled_count=int(
            params.get("labelled_count")
            or 0
        ),
        inspect_images=bool(
            params.get("inspect_images", True)
        ),
        include_column_counts=True,
        image_sample_size=max(
            0,
            int(
                params.get(
                    "image_sample_size",
                    8,
                )
            ),
        ),
        cancel_token=cancel_token,
    )

    return {
        "ok": report.ok,
        "contract": report.contract,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
    }

def query_batch_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    request = _coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()

    if not session_artifact_id:
        raise ValueError("query_batch requires session_artifact_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    dataset_id = _session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("query_batch could not determine the AL session pool dataset.")

    predictions_artifact_id = str(
        params.get("predictions_artifact_id")
        or request.artifact_id
        or al_state.latest_reference(session, "predictions_artifact_id")
        or ""
    ).strip()
    if not predictions_artifact_id:
        raise ValueError(
            "No pool prediction is available. Train with auto_predict enabled or "
            "supply predictions_artifact_id."
        )

    predictions_payload = context.artifacts.get(predictions_artifact_id)
    if not isinstance(predictions_payload, Mapping):
        raise TypeError(f"{predictions_artifact_id!r} is not an ml.predictions payload.")

    predictions_dataset_id = str(predictions_payload.get("dataset_id") or "").strip()
    if not predictions_dataset_id:
        raise ValueError("The ml.predictions artifact does not include dataset_id.")

    _assert_dataset_ids_match(
        dataset_id,
        predictions_dataset_id,
        action="query_batch predictions",
    )

    strategy_id = str(params.get("strategy_id") or params.get("strategy") or "least_confidence")
    k = max(1, int(params.get("k", 200)))
    seed = int(params.get("seed", session.get("seed", 42)))
    make_selection = bool(params.get("make_selection", True))
    records = list(predictions_payload.get("records") or [])

    if not records:
        raise ValueError("Prediction artifact contains no records to rank.")

    exclude = _session_query_exclude_row_ids(session, params)
    _check_cancelled(cancel_token)
    registry = _get_strategy_registry(context)
    ranked = registry.rank(
        records,
        strategy_id=strategy_id,
        k=k,
        exclude_row_ids=exclude,
        predictions_payload=predictions_payload,
        seed=seed,
        params=params,
    )
    _check_cancelled(cancel_token)

    row_ids = [str(row["row_id"]) for row in ranked]
    rank_stats = dict(getattr(registry, "last_rank_stats", {}) or {})
    rank_stats.setdefault("excluded_count", len(exclude))
    rank_stats.setdefault("exclude_row_ids_count", len(exclude))
    rank_stats.setdefault("prediction_record_count", len(records))
    rank_stats.setdefault("returned_count", len(row_ids))
    rank_stats.setdefault("query_dataset_id", dataset_id)
    rank_stats.setdefault("predictions_dataset_id", predictions_dataset_id)

    batch_payload = _batch_payload(
        dataset_id=dataset_id,
        session=session,
        strategy_id=strategy_id,
        records=ranked,
        params=params,
        predictions_artifact_id=predictions_artifact_id,
        kind="query",
        rank_stats=rank_stats,
    )
    batch_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_BATCH,
        batch_payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={
            "session_id": session["session_id"],
            "strategy_id": strategy_id,
            "predictions_artifact_id": predictions_artifact_id,
            "k": k,
            "excluded_count": len(exclude),
            "prediction_record_count": len(records),
            "predictions_dataset_id": predictions_dataset_id,
        },
    )

    session = al_state.with_last_batch(
        session,
        batch_artifact_id=batch_artifact_id,
        strategy_id=strategy_id,
        row_ids=row_ids,
        predictions_artifact_id=predictions_artifact_id,
        kind="query",
    )
    new_session_artifact_id = _put_session(
        context,
        session,
        previous_artifact_id=session_artifact_id,
    )

    if make_selection and row_ids:
        _set_ranked_selection(
            context,
            dataset_id=dataset_id,
            row_ids=row_ids,
            session_artifact_id=new_session_artifact_id,
            batch_artifact_id=batch_artifact_id,
            strategy_id=strategy_id,
            origin=f"{ORIGIN}.query_batch",
        )

    _publish(
        context,
        "al.query_batch.created",
        {
            "session_artifact_id": new_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": session["session_id"],
            "dataset_id": dataset_id,
            "predictions_dataset_id": predictions_dataset_id,
            "predictions_artifact_id": predictions_artifact_id,
            "batch_artifact_id": batch_artifact_id,
            "strategy_id": strategy_id,
            "count": len(row_ids),
            "rank_stats": rank_stats,
        },
    )

    return {
        "ok": True,
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": session["session_id"],
        "dataset_id": dataset_id,
        "predictions_dataset_id": predictions_dataset_id,
        "predictions_artifact_id": predictions_artifact_id,
        "batch_artifact_id": batch_artifact_id,
        "strategy_id": strategy_id,
        "row_ids": row_ids,
        "rank_stats": rank_stats,
        "counts": al_state.counts(session),
        "latest": dict(session.get("latest") or {}),
        "preview": ranked[:25],
    }

def record_label_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    request = _coerce_request(request)
    params = dict(request.params or {})

    session_artifact_id = str(
        params.get("session_artifact_id")
        or ""
    ).strip()

    if not session_artifact_id:
        raise ValueError(
            "record_label requires session_artifact_id."
        )

    session = al_state.coerce_session(
        context.artifacts.get(
            session_artifact_id
        )
    )

    session_dataset_id = str(
        session.get("dataset_id")
        or ""
    ).strip()

    focus_dataset_id, focus_row_id = (
        _focused_row_ref(context)
    )

    row_id = params.get("row_id")

    if row_id is None:
        row_id = focus_row_id

        if focus_dataset_id:
            _assert_dataset_ids_match(
                session_dataset_id,
                focus_dataset_id,
                action="record_label focus",
            )

    else:
        request_dataset_id = str(
            params.get("dataset_id")
            or request.dataset_id
            or ""
        ).strip()

        if request_dataset_id:
            _assert_dataset_ids_match(
                session_dataset_id,
                request_dataset_id,
                action="record_label request",
            )

    if row_id is None:
        raise ValueError(
            "No row_id supplied and no active selection focus exists."
        )

    label = params.get("label")

    if label is None:
        raise ValueError(
            "record_label requires label."
        )

    _check_cancelled(cancel_token)

    source = str(
        params.get("source")
        or request.origin
        or "manual"
    )

    updated = al_state.record_label(
        session,
        row_id=row_id,
        label=label,
        source=source,
    )

    new_session_artifact_id = _put_session(
        context,
        updated,
        previous_artifact_id=session_artifact_id,
    )

    label_value = al_state.normalise_label(
        label
    )

    _publish(
        context,
        "al.label.recorded",
        {
            "session_artifact_id": new_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": updated["session_id"],
            "dataset_id": updated["dataset_id"],
            "row_id": str(row_id),
            "label": label_value,
            "status": (
                "unsure"
                if label_value == al_state.UNSURE_LABEL
                else "verified"
            ),
            "counts": al_state.counts(updated),
        },
    )

    return {
        "ok": True,
        "session_artifact_id": new_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated["session_id"],
        "dataset_id": updated["dataset_id"],
        "row_id": str(row_id),
        "label": label_value,
        "display_label": al_state.display_label(label_value),
        "counts": al_state.counts(updated),
    }

def train_from_session_action(
    context: Any,
    request: Any,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    """Train one AL round and, by default, predict and acquire the next batch."""

    request = _coerce_request(request)
    params = dict(request.params or {})
    session_artifact_id = str(params.get("session_artifact_id") or "").strip()
    recipe_id = str(params.get("recipe_id") or "").strip()

    if not session_artifact_id:
        raise ValueError("train_from_session requires session_artifact_id.")
    if not recipe_id:
        raise ValueError("train_from_session requires recipe_id.")

    session = al_state.coerce_session(context.artifacts.get(session_artifact_id))
    dataset_id = _session_pool_dataset_id(session)
    if not dataset_id:
        raise ValueError("Could not determine source/pool dataset_id.")

    request_dataset_id = str(params.get("dataset_id") or request.dataset_id or "").strip()
    if request_dataset_id:
        _assert_dataset_ids_match(dataset_id, request_dataset_id, action="train_from_session")

    labelled_items = al_state.labelled_training_items(session)
    if not labelled_items:
        raise ValueError("No verified labels are available for training.")

    seed = int(params.get("seed", session.get("seed", 42)))
    target_column = str(params.get("target_column") or session.get("target_column") or "al_label")
    label_counts = _label_counts(labelled_items)

    preflight = _build_session_preflight(
        context,
        dataset_id=dataset_id,
        recipe_id=recipe_id,
        params={**params, "target_column": target_column},
        session=session,
        labelled_count=len(labelled_items),
        label_counts=label_counts,
        cancel_token=cancel_token,
    )
    preflight.raise_for_errors()
    session_contract = preflight.contract

    task = str((session_contract.get("recipe") or {}).get("task") or "classification").lower()
    if "class" not in task:
        raise ValueError(
            "core.active_learning currently implements classification annotation and "
            f"acquisition only; selected recipe task is {task!r}."
        )

    class_labels = _resolve_al_class_labels(
        context,
        params=params,
        session=session,
        dataset_id=dataset_id,
        target_column=target_column,
    )
    if not class_labels:
        raise ValueError("Active-learning classification requires the complete class list.")

    round_index = int(session.get("round", 0)) + 1
    _check_cancelled(cancel_token)
    _publish(
        context,
        "al.round.training_started",
        {
            "session_artifact_id": session_artifact_id,
            "session_id": session["session_id"],
            "dataset_id": dataset_id,
            "round": round_index,
            "labelled_count": len(labelled_items),
            "label_counts": label_counts,
            "seed": seed,
            "recipe_id": recipe_id,
        },
    )

    contract_recipe_params = dict((session_contract.get("recipe") or {}).get("params") or {})
    recipe_params = contract_recipe_params
    recipe_params.update(dict(params.get("recipe_params") or {}))
    recipe_params.update(_al_protocol_params(params, session))

    train_dataset_id = str(params.get("train_dataset_id") or "").strip()
    if not train_dataset_id:
        train_dataset_id = (
            f"{dataset_id}__al_train_r{round_index}_{uuid.uuid4().hex[:6]}"
            .replace(":", "_")
            .replace("/", "_")
        )

    id_column = _resolve_record_id_column(context, dataset_id)
    required_columns = _resolve_training_required_columns(
        context,
        dataset_id=dataset_id,
        recipe_params=recipe_params,
        target_column=target_column,
        id_column=id_column,
    )
    train_df, id_column = _training_dataframe(
        context,
        dataset_id=dataset_id,
        labelled_items=labelled_items,
        target_column=target_column,
        required_columns=required_columns,
    )

    mappings = _dataset_mappings(context, dataset_id)
    if id_column:
        mappings["record_id"] = id_column
    mappings["target_label"] = target_column
    mappings = _filter_mappings_to_existing_columns(mappings, train_df.columns)

    if id_column and id_column in train_df.columns:
        recipe_params.setdefault("record_id_column", id_column)
    recipe_params.setdefault("target_column", target_column)
    recipe_params.setdefault("label_column", target_column)

    pool_binding = dict((session_contract.get("bindings") or {}).get("pool") or {})
    image_column = str(pool_binding.get("image_column") or recipe_params.get("image_column") or "").strip()
    if image_column:
        recipe_params["image_column"] = image_column
        if image_column not in train_df.columns:
            raise ValueError(
                f"Derived AL training dataset does not contain image column {image_column!r}."
            )

    feature_columns = list(pool_binding.get("feature_columns") or _feature_columns_from_recipe_params(recipe_params))
    if feature_columns:
        missing_features = [column for column in feature_columns if column not in train_df.columns]
        if missing_features:
            raise ValueError(
                "Derived AL training dataset is missing selected feature columns: "
                + ", ".join(missing_features)
            )
        recipe_params["feature_columns"] = feature_columns
        recipe_params["input_columns"] = feature_columns
        recipe_params["features"] = feature_columns

    context.datasets.register(
        train_dataset_id,
        train_df,
        name=f"AL training round {round_index}",
        column_mappings=mappings,
        al_session_id=session["session_id"],
        al_session_artifact_id=session_artifact_id,
        al_round=round_index,
        source_dataset_id=dataset_id,
        pool_dataset_id=dataset_id,
        label_column=target_column,
    )
    _publish(
        context,
        "dataset.registered",
        {
            "dataset_id": train_dataset_id,
            "source_dataset_id": dataset_id,
            "origin": f"{ORIGIN}.train_from_session",
            "kind": "active_learning_training_dataset",
            "session_id": session["session_id"],
            "session_artifact_id": session_artifact_id,
            "round": round_index,
        },
    )

    training_artifact_payload = {
        "schema_version": 2,
        "session_id": session["session_id"],
        "session_artifact_id": session_artifact_id,
        "source_dataset_id": dataset_id,
        "pool_dataset_id": dataset_id,
        "training_dataset_id": train_dataset_id,
        "validation_dataset_id": str(params.get("validation_dataset_id") or session.get("validation_dataset_id") or ""),
        "test_dataset_id": str(params.get("test_dataset_id") or session.get("test_dataset_id") or ""),
        "target_column": target_column,
        "record_id_column": id_column,
        "class_labels": list(class_labels),
        "classes": list(class_labels),
        "round": round_index,
        "seed": seed,
        "row_ids": [item["row_id"] for item in labelled_items],
        "labels": labelled_items,
        "label_counts": label_counts,
        "counts": al_state.counts(session),
        "session_contract": session_contract,
    }
    training_artifact_id = context.artifacts.put(
        al_state.ARTIFACT_TRAINING_SET,
        training_artifact_payload,
        dataset_id=train_dataset_id,
        row_ids=[item["row_id"] for item in labelled_items],
        params={
            "session_id": session["session_id"],
            "round": round_index,
            "target_column": target_column,
        },
    )

    recipe_params.update(
        {
            "dataset_id": train_dataset_id,
            "recipe_id": recipe_id,
            "target_column": target_column,
            "label_column": target_column,
            "al_session_id": session["session_id"],
            "al_session_artifact_id": session_artifact_id,
            "al_training_artifact_id": training_artifact_id,
            "al_round": round_index,
            "label_options": list(class_labels),
            "class_labels": list(class_labels),
            "classes": list(class_labels),
            "known_classes": list(class_labels),
            "target_classes": list(class_labels),
            "evaluation_scope": "all_known_classes",
            "warm_start": False,
            "reset_model": True,
            "reset_model_each_round": True,
            "initialise_from_scratch": True,
            "initialization_seed": seed,
            "initialisation_seed": seed,
            "seed": seed,
            "random_seed": seed,
            "numpy_seed": seed,
            "torch_seed": seed,
            "run_id": recipe_params.get("run_id") or uuid.uuid4().hex,
        }
    )

    _seed_everything(seed)
    _check_cancelled(cancel_token)
    ml_request = ActionRequest(
        dataset_id=train_dataset_id,
        row_ids=None,
        columns=[],
        params=recipe_params,
        artifact_id=None,
        origin=f"{ORIGIN}.train_from_session",
    )

    try:
        ml_result = _run_core_ml_recipe_action(context, ml_request, cancel_token=cancel_token)
    except Exception as exc:
        _publish(
            context,
            "al.round.training_failed",
            {
                "session_artifact_id": session_artifact_id,
                "session_id": session["session_id"],
                "dataset_id": dataset_id,
                "training_dataset_id": train_dataset_id,
                "round": round_index,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    updated_session = al_state.with_completed_training_round(
        session,
        training_dataset_id=train_dataset_id,
        training_artifact_id=training_artifact_id,
        ml_result=ml_result,
    )
    updated_session = al_state.with_contract(
        updated_session,
        contract=session_contract,
        event="training_contract_committed",
    )
    updated_session["recipe_id"] = recipe_id
    updated_session["label_options"] = list(class_labels)
    updated_session["class_labels"] = list(class_labels)
    updated_session["validation_dataset_id"] = training_artifact_payload["validation_dataset_id"]
    updated_session["test_dataset_id"] = training_artifact_payload["test_dataset_id"]

    training_session_artifact_id = _put_session(
        context,
        updated_session,
        previous_artifact_id=session_artifact_id,
    )
    final_session_artifact_id = training_session_artifact_id
    prediction_session_artifact_id: Optional[str] = None
    prediction_result: Dict[str, Any] = {}
    query_result: Dict[str, Any] = {}
    workflow_errors: List[Dict[str, str]] = []

    auto_predict = bool(params.get("auto_predict", True))
    auto_query = bool(params.get("auto_query", True))

    if auto_predict:
        try:
            model_artifact_id = al_state.latest_reference(updated_session, "model_artifact_id")
            if not model_artifact_id:
                raise ValueError("The training result did not provide model_artifact_id.")

            prediction_params: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "model_artifact_id": model_artifact_id,
                "scope": "inference",
                "require_target_compatible": False,
                "register_prediction_dataset": True,
                "run_id": uuid.uuid4().hex,
            }
            if image_column:
                prediction_params["image_column"] = image_column
            if feature_columns:
                prediction_params["feature_column_mapping"] = {
                    str(column): str(column) for column in feature_columns
                }

            prediction_request = ActionRequest(
                dataset_id=dataset_id,
                row_ids=None,
                columns=[],
                params=prediction_params,
                artifact_id=model_artifact_id,
                origin=f"{ORIGIN}.auto_predict",
            )
            prediction_result = _run_core_ml_predict_action(
                context,
                prediction_request,
                cancel_token=cancel_token,
            )
            updated_session = al_state.with_prediction_result(
                updated_session,
                prediction_result=prediction_result,
            )
            prediction_session_artifact_id = _put_session(
                context,
                updated_session,
                previous_artifact_id=training_session_artifact_id,
            )
            final_session_artifact_id = prediction_session_artifact_id
            _publish(
                context,
                "al.round.pool_prediction_finished",
                {
                    "session_artifact_id": prediction_session_artifact_id,
                    "session_id": updated_session["session_id"],
                    "round": round_index,
                    "dataset_id": dataset_id,
                    "model_artifact_id": model_artifact_id,
                    "predictions_artifact_id": al_state.latest_reference(
                        updated_session, "predictions_artifact_id"
                    ),
                },
            )

            if auto_query:
                query_request = ActionRequest(
                    artifact_id=al_state.latest_reference(updated_session, "predictions_artifact_id"),
                    params={
                        "session_artifact_id": prediction_session_artifact_id,
                        "strategy_id": str(
                            params.get("query_strategy_id")
                            or params.get("strategy_id")
                            or "least_confidence"
                        ),
                        "k": max(1, int(params.get("query_k", params.get("k", 200)))),
                        "seed": seed,
                        "make_selection": bool(params.get("make_selection", True)),
                    },
                    origin=f"{ORIGIN}.auto_query",
                )
                query_result = query_batch_action(
                    context,
                    query_request,
                    cancel_token=cancel_token,
                )
                final_session_artifact_id = str(query_result["session_artifact_id"])

        except Exception as exc:
            stage = "query" if prediction_result else "prediction"
            workflow_errors.append({"stage": stage, "error": str(exc)})
            _publish(
                context,
                f"al.round.{stage}_failed",
                {
                    "session_artifact_id": final_session_artifact_id,
                    "session_id": updated_session["session_id"],
                    "round": round_index,
                    "dataset_id": dataset_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    try:
        final_session = al_state.coerce_session(
            context.artifacts.get(final_session_artifact_id)
        )
    except Exception:
        final_session = updated_session

    _publish(
        context,
        "al.round.training_finished",
        {
            "session_artifact_id": final_session_artifact_id,
            "training_session_artifact_id": training_session_artifact_id,
            "prediction_session_artifact_id": prediction_session_artifact_id,
            "previous_session_artifact_id": session_artifact_id,
            "session_id": updated_session["session_id"],
            "source_dataset_id": dataset_id,
            "training_dataset_id": train_dataset_id,
            "training_artifact_id": training_artifact_id,
            "round": round_index,
            "recipe_id": recipe_id,
            "seed": seed,
            "ml_result": _compact_ml_result(ml_result),
            "prediction_result": _compact_ml_result(prediction_result),
            "query_result": _compact_ml_result(query_result),
            "workflow_errors": workflow_errors,
        },
    )

    return {
        "ok": True,
        "workflow_status": "complete" if not workflow_errors else "partial",
        "workflow_errors": workflow_errors,
        "session_artifact_id": final_session_artifact_id,
        "training_session_artifact_id": training_session_artifact_id,
        "prediction_session_artifact_id": prediction_session_artifact_id,
        "previous_session_artifact_id": session_artifact_id,
        "session_id": updated_session["session_id"],
        "source_dataset_id": dataset_id,
        "training_dataset_id": train_dataset_id,
        "training_artifact_id": training_artifact_id,
        "round": round_index,
        "seed": seed,
        "recipe_id": recipe_id,
        "labelled_count": len(labelled_items),
        "label_counts": label_counts,
        "ml_result": ml_result,
        "prediction_result": prediction_result,
        "query_result": query_result,
        "counts": al_state.counts(final_session),
        "latest": dict(final_session.get("latest") or {}),
    }

def _batch_payload(
    *,
    dataset_id: str,
    session: Mapping[str, Any],
    strategy_id: str,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    predictions_artifact_id: Optional[str],
    kind: str,
    rank_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "dataset_id": dataset_id,
        "session_id": session.get("session_id"),
        "round": int(session.get("round", 0)),
        "predictions_artifact_id": predictions_artifact_id,
        "strategy": strategy_id,
        "strategy_id": strategy_id,
        "records": [
            dict(record)
            for record in records
        ],
        "row_ids": [
            str(record.get("row_id"))
            for record in records
            if record.get("row_id") is not None
        ],
        "count": len(records),
        "ordered_by": (
            "informativeness_desc"
            if kind == "query"
            else "random"
        ),
        "params": dict(params or {}),
        "rank_stats": dict(rank_stats or {}),
        "session_counts": al_state.counts(session),
    }

def _training_dataframe(
    context: Any,
    *,
    dataset_id: str,
    labelled_items: Sequence[Mapping[str, Any]],
    target_column: str,
    required_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Materialise verified AL labels into a derived training dataframe.

    Image recipes require source columns such as image_path to exist physically
    in the derived dataframe. Copying only semantic mappings is insufficient.
    """

    id_column = _resolve_record_id_column(
        context,
        dataset_id,
    )

    row_ids = [
        str(item["row_id"])
        for item in labelled_items
    ]

    row_id_set = set(row_ids)

    labels = {
        str(item["row_id"]): al_state.display_label(
            item["label"]
        )
        for item in labelled_items
    }

    required: List[str] = []

    def add_required(column: Any) -> None:
        if column is None:
            return

        text = str(column).strip()

        if not text or text == "Use Index":
            return

        if text not in required:
            required.append(text)

    add_required(id_column)

    for column in required_columns or []:
        add_required(column)

    source_columns = _existing_dataset_columns(
        context,
        dataset_id,
    )

    if target_column in source_columns:
        add_required(target_column)

    missing_from_source = [
        column
        for column in required
        if (
            column != target_column
            and source_columns
            and column not in source_columns
        )
    ]

    if missing_from_source:
        raise ValueError(
            "Active Learning could not find required recipe column(s) in the "
            f"source dataset {dataset_id!r}: {', '.join(missing_from_source)}. "
            f"Available columns: {sorted(source_columns)}"
        )

    def missing_required(
        frame: pd.DataFrame,
    ) -> List[str]:
        return [
            column
            for column in required
            if (
                column != target_column
                and column not in frame.columns
            )
        ]

    def get_df_compat(
        columns: Optional[
            Sequence[str]
        ] = None,
    ) -> pd.DataFrame:
        """Call DatasetManager.get_df across old and new signatures."""

        if columns:
            cols = list(
                dict.fromkeys(
                    str(col)
                    for col in columns
                    if col
                )
            )

            try:
                return context.datasets.get_df(
                    dataset_id,
                    columns=cols,
                )

            except TypeError:
                return context.datasets.get_df(
                    dataset_id
                )

            except Exception:
                return context.datasets.get_df(
                    dataset_id
                )

        return context.datasets.get_df(
            dataset_id
        )

    df: pd.DataFrame

    if id_column:
        try:
            df = context.datasets.get_rows_by_ids(
                dataset_id,
                row_ids,
                id_column=id_column,
            )

            df = df.copy()

        except Exception:
            df = pd.DataFrame()

        if (
            df.empty
            or missing_required(df)
        ):
            source_df = get_df_compat(
                required
            )

            if id_column not in source_df.columns:
                raise ValueError(
                    f"record_id mapping points to missing column: {id_column}"
                )

            df = source_df[
                source_df[id_column]
                .astype(str)
                .isin(row_id_set)
            ].copy()

    else:
        source_df = get_df_compat(
            required
            or None
        )

        matching_index = [
            idx
            for idx in source_df.index
            if str(idx) in row_id_set
        ]

        df = source_df.loc[
            matching_index
        ].copy()

    if df.empty:
        raise ValueError(
            "Could not materialise any labelled rows for training."
        )

    missing_after_materialise = missing_required(
        df
    )

    if missing_after_materialise:
        raise ValueError(
            "Active Learning could not materialise required recipe column(s) "
            f"from source dataset {dataset_id!r}: "
            f"{', '.join(missing_after_materialise)}. "
            f"Materialised columns: {list(df.columns)}"
        )

    if (
        id_column
        and id_column in df.columns
    ):
        df[target_column] = (
            df[id_column]
            .astype(str)
            .map(labels)
        )

    else:
        df[target_column] = [
            labels.get(str(idx))
            for idx in df.index
        ]

    df = df[
        df[target_column].notna()
    ].copy()

    if df.empty:
        raise ValueError(
            "Training dataframe has no rows after applying labels."
        )

    return df, id_column

def _set_ranked_selection(
    context: Any,
    *,
    dataset_id: str,
    row_ids: Sequence[Any],
    session_artifact_id: str,
    batch_artifact_id: str,
    strategy_id: str,
    origin: str,
) -> None:
    selection = getattr(
        context,
        "selection",
        None,
    )

    if (
        selection is None
        or not hasattr(
            selection,
            "set_selection_set",
        )
    ):
        return

    ordered_row_ids = [
        str(row_id)
        for row_id in row_ids
    ]

    selection.set_selection_set(
        dataset_id=dataset_id,
        row_ids=ordered_row_ids,
        origin=origin,
        mode="replace",
        metadata={
            "ordered": True,
            "order": "informativeness_desc",
            "strategy_id": strategy_id,
            "session_artifact_id": session_artifact_id,
            "active_learning_batch_artifact_id": batch_artifact_id,
            "count": len(ordered_row_ids),
            "note": (
                "Rows are intentionally ordered. Selection/record-browser "
                "next navigation should follow this order."
            ),
        },
        create_artifact=True,
        update_focus_policy="first",
    )

def _put_session(
    context: Any,
    session: Mapping[str, Any],
    *,
    previous_artifact_id: Optional[str] = None,
) -> str:
    session_payload = al_state.with_revision(
        session,
        previous_session_artifact_id=previous_artifact_id,
    )

    artifact_id = context.artifacts.put(
        al_state.ARTIFACT_SESSION,
        session_payload,
        dataset_id=str(
            session_payload.get("dataset_id")
            or "default"
        ),
        row_ids=list(
            session_payload.get("ignored_row_ids")
            or []
        ),
        params={
            "session_id": session_payload.get("session_id"),
            "round": session_payload.get("round"),
            "revision": session_payload.get("revision"),
            "previous_session_artifact_id": previous_artifact_id,
        },
    )

    _publish(
        context,
        "al.session.saved",
        {
            "session_artifact_id": artifact_id,
            "previous_session_artifact_id": previous_artifact_id,
            "session_id": session_payload.get("session_id"),
            "dataset_id": session_payload.get("dataset_id"),
            "round": session_payload.get("round"),
            "revision": session_payload.get("revision"),
            "counts": al_state.counts(session_payload),
        },
    )

    return artifact_id

def _assert_dataset_ids_match(
    expected: str,
    actual: str,
    *,
    action: str,
) -> None:
    expected = str(
        expected
        or ""
    ).strip()

    actual = str(
        actual
        or ""
    ).strip()

    if (
        expected
        and actual
        and expected != actual
    ):
        raise ValueError(
            f"{action}: dataset mismatch. Expected {expected!r}, got {actual!r}."
        )

def _assert_session_dataset(
    session: Mapping[str, Any],
    dataset_id: str,
    *,
    action: str,
) -> None:
    _assert_dataset_ids_match(
        str(
            session.get("dataset_id")
            or ""
        ),
        str(
            dataset_id
            or ""
        ),
        action=action,
    )

def _session_pool_dataset_id(
    session: Mapping[str, Any],
) -> str:
    """Return the original query pool, not a derived round-training dataset."""

    return str(
        session.get("pool_dataset_id")
        or session.get("dataset_id")
        or ""
    ).strip()

def _session_query_exclude_row_ids(
    session: Mapping[str, Any],
    params: Mapping[str, Any],
) -> set[str]:
    """Rows already queued, reviewed, trained, deferred, or excluded."""

    exclude = set(
        al_state.row_ids_in_states(
            session,
            states=tuple(al_state.TERMINAL_QUERY_STATES),
        )
    )

    values = params.get("exclude_row_ids") or []
    if isinstance(values, Mapping):
        values = values.keys()
    elif isinstance(values, (str, bytes, bytearray, int, float)):
        values = [values]

    for row_id in values:
        if row_id is not None and str(row_id).strip():
            exclude.add(str(row_id).strip())

    return exclude

def _focused_row_ref(
    context: Any,
) -> Tuple[Optional[str], Optional[str]]:
    selection = getattr(
        context,
        "selection",
        None,
    )

    if (
        selection is None
        or not hasattr(
            selection,
            "get_focus",
        )
    ):
        return None, None

    focus = selection.get_focus()

    if focus is None:
        return None, None

    dataset_id = getattr(
        focus,
        "dataset_id",
        None,
    )

    row_id = getattr(
        focus,
        "row_id",
        None,
    )

    return (
        None
        if dataset_id is None
        else str(dataset_id),
        None
        if row_id is None
        else str(row_id),
    )

def _focused_row_id(
    context: Any,
) -> Optional[str]:
    _, row_id = _focused_row_ref(
        context
    )

    return row_id

def _sample_dataset_row_ids(
    context: Any,
    *,
    dataset_id: str,
    k: int,
    seed: int,
) -> List[str]:
    k = max(
        0,
        int(k or 0),
    )

    if k == 0:
        return []

    datasets = getattr(
        context,
        "datasets",
        None,
    )

    source = (
        getattr(
            datasets,
            "get_source",
            lambda *_: None,
        )(dataset_id)
        if datasets
        else None
    )

    for owner in (
        datasets,
        source,
    ):
        for method_name in (
            "sample_row_ids",
            "sample_ids",
        ):
            method = getattr(
                owner,
                method_name,
                None,
            )

            if callable(method):
                try:
                    values = method(
                        dataset_id=dataset_id,
                        k=k,
                        seed=seed,
                    )

                except TypeError:
                    try:
                        values = method(
                            k=k,
                            seed=seed,
                        )

                    except TypeError:
                        values = method(
                            k,
                            seed,
                        )

                return [
                    str(value)
                    for value in values
                ]

    pool_row_ids = _dataset_row_ids(
        context,
        dataset_id,
    )

    rng = random.Random(seed)

    return rng.sample(
        pool_row_ids,
        min(
            k,
            len(pool_row_ids),
        ),
    )

def _al_protocol_params(
    params: Mapping[str, Any],
    session: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key in (
        "al_protocol",
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
    ):
        if key in params:
            out[key] = params[key]

    validation_dataset_id = str(
        params.get("validation_dataset_id")
        or session.get("validation_dataset_id")
        or ""
    ).strip()

    test_dataset_id = str(
        params.get("test_dataset_id")
        or session.get("test_dataset_id")
        or ""
    ).strip()

    if validation_dataset_id:
        out.setdefault(
            "protocol_validation_source",
            "dataset",
        )

        out.setdefault(
            "protocol_validation_dataset_id",
            validation_dataset_id,
        )

    if test_dataset_id:
        out.setdefault(
            "protocol_test_source",
            "dataset",
        )

        out.setdefault(
            "protocol_test_dataset_id",
            test_dataset_id,
        )

    out.setdefault(
        "al_protocol",
        str(
            params.get("al_protocol")
            or session.get("al_protocol")
            or "review"
        ),
    )

    return out

def _run_core_ml_recipe_action(
    context: Any,
    request: ActionRequest,
    *,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    return _call_registered_action(
        context,
        "core.ml.run_ml_recipe",
        request,
        cancel_token=cancel_token,
    )


def _run_core_ml_predict_action(
    context: Any,
    request: ActionRequest,
    *,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    return _call_registered_action(
        context,
        "core.ml.predict",
        request,
        cancel_token=cancel_token,
    )


def _call_registered_action(
    context: Any,
    action_id: str,
    request: ActionRequest,
    *,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    manager = getattr(context, "plugins", None)
    if manager is None or not hasattr(manager, "get_action"):
        raise RuntimeError(f"{action_id} requires the plugin manager.")

    try:
        registration = manager.get_action(action_id)
    except Exception as exc:
        raise RuntimeError(f"Required action {action_id!r} is not registered.") from exc

    handler = getattr(registration, "handler", None)
    if not callable(handler):
        raise RuntimeError(f"Registered action {action_id!r} has no callable handler.")

    raw = handler(context, request, cancel_token=cancel_token)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"ok": True, "result": raw}


def _resolve_recipe_spec(context: Any, recipe_id: str) -> Any:
    services = getattr(context, "services", None)
    if services is None:
        raise RuntimeError("core.ml.recipe_registry is unavailable.")
    registry = services.get("core.ml.recipe_registry")
    try:
        return registry.get(recipe_id)
    except Exception as exc:
        raise ValueError(f"Unknown core.ml recipe: {recipe_id!r}.") from exc


def _build_session_preflight(
    context: Any,
    *,
    dataset_id: str,
    recipe_id: str,
    params: Mapping[str, Any],
    labelled_count: int,
    label_counts: Mapping[str, int],
    session: Optional[Mapping[str, Any]] = None,
    cancel_token: Any = None,
) -> al_contracts.PreflightReport:
    session = dict(session or {})
    recipe_params = dict(params.get("recipe_params") or {})
    class_labels = list(
        (session.get("class_labels") if session else None)
        or (session.get("label_options") if session else None)
        or params.get("class_labels")
        or params.get("label_options")
        or []
    )
    if class_labels:
        recipe_params.setdefault("class_labels", class_labels)
        recipe_params.setdefault("label_options", class_labels)
    supplied_contract = dict(params.get("session_contract") or params.get("contract") or {})
    supplied_binding = dict((supplied_contract.get("bindings") or {}).get("pool") or {})

    feature_columns = (
        params.get("feature_columns")
        or supplied_binding.get("feature_columns")
        or _feature_columns_from_recipe_params(recipe_params)
        or []
    )
    image_column = params.get("image_column") or supplied_binding.get("image_column")
    mask_column = params.get("mask_column") or supplied_binding.get("mask_column")
    protocol_params = dict(params.get("protocol_params") or {})
    protocol_params.update(_al_protocol_params(params, session))

    return al_contracts.build_session_contract(
        context,
        recipe_spec=_resolve_recipe_spec(context, recipe_id),
        pool_dataset_id=dataset_id,
        validation_dataset_id=str(
            params.get("validation_dataset_id")
            or session.get("validation_dataset_id")
            or ""
        ),
        test_dataset_id=str(
            params.get("test_dataset_id")
            or session.get("test_dataset_id")
            or ""
        ),
        target_column=str(
            params.get("target_column")
            or session.get("target_column")
            or "al_label"
        ),
        feature_columns=feature_columns,
        image_column=str(image_column or "") or None,
        mask_column=str(mask_column or "") or None,
        recipe_params=recipe_params,
        protocol_params=protocol_params,
        labelled_count=labelled_count,
        label_counts=label_counts,
        inspect_images=bool(params.get("inspect_images", False)),
        include_column_counts=True,
        image_sample_size=max(0, int(params.get("image_sample_size", 8))),
        cancel_token=cancel_token,
    )


def _label_counts(labelled_items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in labelled_items:
        label = al_state.normalise_label(item.get("label"))
        if not label or label == al_state.UNSURE_LABEL:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts

def _dataset_row_ids(
    context: Any,
    dataset_id: str,
) -> List[str]:
    id_column = _resolve_record_id_column(
        context,
        dataset_id,
    )

    if id_column:
        df = context.datasets.get_df(
            dataset_id,
            columns=[id_column],
        )

        if id_column not in df.columns:
            raise ValueError(
                f"record_id mapping points to missing column: {id_column}"
            )

        return [
            str(value)
            for value in (
                df[id_column]
                .dropna()
                .tolist()
            )
        ]

    df = context.datasets.get_df(
        dataset_id
    )

    return [
        str(idx)
        for idx in df.index.tolist()
    ]

def _resolve_dataset_id(
    context: Any,
    request: ActionRequest,
    params: Mapping[str, Any],
) -> str:
    dataset_id = str(
        params.get("dataset_id")
        or request.dataset_id
        or ""
    ).strip()

    if dataset_id:
        return dataset_id

    return str(
        context.datasets.active_id()
    )

def _resolve_record_id_column(
    context: Any,
    dataset_id: str,
) -> Optional[str]:
    datasets = getattr(
        context,
        "datasets",
        None,
    )

    if datasets is None:
        return None

    for semantic_name in (
        "record_id",
        "id",
        "row_id",
    ):
        try:
            column = datasets.get_mapping(
                dataset_id,
                semantic_name,
            )

        except Exception:
            column = None

        if column:
            return str(column)

    try:
        columns = {
            str(col)
            for col in datasets.list_columns(
                dataset_id
            )
        }

    except Exception:
        columns = set()

    for candidate in (
        "record_id",
        "id",
        "ID",
        "source_id",
        "object_id",
        "row_id",
    ):
        if candidate in columns:
            return candidate

    return None

def _dataset_mappings(
    context: Any,
    dataset_id: str,
) -> Dict[str, str]:
    try:
        return dict(
            context.datasets.get_mappings(
                dataset_id
            )
            or {}
        )

    except Exception:
        return {}

def _get_strategy_registry(
    context: Any,
):
    services = getattr(
        context,
        "services",
        None,
    )

    if services is not None:
        try:
            registry = services.get(
                "core.active_learning.query_strategy_registry"
            )

            if registry is not None:
                return registry

        except Exception:
            pass

    return create_default_strategy_registry()

def _resolve_al_class_labels(
    context: Any,
    *,
    params: Mapping[str, Any],
    session: Mapping[str, Any],
    dataset_id: str,
    target_column: str,
) -> List[str]:
    """Resolve the complete class universe for an active-learning run."""

    candidates: List[Any] = [
        params.get("class_labels"),
        params.get("classes"),
        params.get("known_classes"),
        params.get("target_classes"),
        params.get("label_options"),
        session.get("class_labels"),
        session.get("classes"),
        session.get("known_classes"),
        session.get("label_options"),
    ]

    labels: List[str] = []

    for value in candidates:
        labels.extend(
            al_state.parse_label_options(
                value
            )
        )

    if labels:
        return list(
            dict.fromkeys(labels)
        )

    try:
        labels = _infer_label_options_from_dataset_column(
            context,
            dataset_id=dataset_id,
            column=target_column,
        )

    except Exception:
        labels = []

    return list(
        dict.fromkeys(
            al_state.parse_label_options(
                labels
            )
        )
    )

def _infer_label_options(
    predictions_payload: Mapping[str, Any],
) -> List[str]:
    labels: List[str] = []

    for key in (
        "labels",
        "class_labels",
        "classes",
    ):
        value = predictions_payload.get(key)

        if (
            isinstance(value, Sequence)
            and not isinstance(
                value,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            )
        ):
            labels.extend(
                str(item)
                for item in value
                if item is not None
            )

    for record in (
        predictions_payload.get("records")
        or []
    ):
        if not isinstance(record, Mapping):
            continue

        for key in (
            "predicted_label",
            "label",
            "target",
        ):
            value = record.get(key)

            if value is not None:
                labels.append(
                    str(value)
                )

        probabilities = (
            record.get("probabilities")
            or record.get("class_probabilities")
        )

        if isinstance(
            probabilities,
            Mapping,
        ):
            labels.extend(
                str(item)
                for item in probabilities.keys()
            )

        if len(labels) >= 50:
            break

    return al_state.parse_label_options(
        labels
    )

def _infer_label_options_from_dataset_column(
    context: Any,
    *,
    dataset_id: str,
    column: str,
    max_values: int = 200,
) -> List[str]:
    dataset_id = str(
        dataset_id
        or ""
    ).strip()

    column = str(
        column
        or ""
    ).strip()

    if (
        not dataset_id
        or not column
        or column == "al_label"
    ):
        return []

    values: List[str] = []

    try:
        source = None

        get_source = getattr(
            context.datasets,
            "get_source",
            None,
        )

        if callable(get_source):
            source = get_source(
                dataset_id
            )

        for method_name in (
            "unique_values",
            "distinct_values",
            "get_column_values",
        ):
            method = (
                getattr(
                    source,
                    method_name,
                    None,
                )
                if source is not None
                else None
            )

            if callable(method):
                try:
                    raw = method(
                        column,
                        limit=max_values,
                    )

                except TypeError:
                    raw = method(column)

                return al_state.parse_label_options(
                    raw
                )

    except Exception:
        pass

    try:
        try:
            df = context.datasets.get_df(
                dataset_id,
                columns=[column],
            )

        except TypeError:
            df = context.datasets.get_df(
                dataset_id
            )

        if column not in df.columns:
            return []

        for value in (
            df[column]
            .dropna()
            .tolist()
        ):
            text = str(value).strip()

            if not text:
                continue

            if text not in values:
                values.append(text)

            if len(values) >= max_values:
                break

    except Exception:
        return []

    return al_state.parse_label_options(
        values
    )

def _coerce_request(
    request: Any,
) -> ActionRequest:
    if isinstance(
        request,
        ActionRequest,
    ):
        return request

    if isinstance(
        request,
        dict,
    ):
        return ActionRequest.from_dict(
            request
        )

    return ActionRequest(
        dataset_id=getattr(
            request,
            "dataset_id",
            None,
        ),
        row_ids=getattr(
            request,
            "row_ids",
            None,
        ),
        columns=list(
            getattr(
                request,
                "columns",
                [],
            )
            or []
        ),
        params=dict(
            getattr(
                request,
                "params",
                {},
            )
            or {}
        ),
        artifact_id=getattr(
            request,
            "artifact_id",
            None,
        ),
        origin=getattr(
            request,
            "origin",
            None,
        ),
    )

def _check_cancelled(
    cancel_token: Any,
) -> None:
    if cancel_token is None:
        return

    for attr in (
        "raise_if_cancelled",
        "throw_if_cancelled",
        "check_cancelled",
    ):
        method = getattr(
            cancel_token,
            attr,
            None,
        )

        if callable(method):
            method()
            return

    for attr in (
        "cancelled",
        "is_cancelled",
        "cancel_requested",
    ):
        value = getattr(
            cancel_token,
            attr,
            None,
        )

        try:
            cancelled = (
                value()
                if callable(value)
                else bool(value)
            )

        except Exception:
            cancelled = False

        if cancelled:
            raise RuntimeError(
                "Active-learning action cancelled."
            )

def _seed_everything(
    seed: int,
) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)

    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(
                seed
            )

        if (
            hasattr(torch, "backends")
            and hasattr(
                torch.backends,
                "cudnn",
            )
        ):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    except Exception:
        pass

def _compact_ml_result(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key, value in dict(
        result
        or {}
    ).items():
        if key in {
            "traceback",
            "records",
            "payload",
        }:
            continue

        if (
            isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
            or value is None
        ):
            out[key] = value

        elif isinstance(
            value,
            Mapping,
        ):
            out[key] = {
                str(item_key): item_value
                for item_key, item_value in value.items()
                if (
                    isinstance(
                        item_value,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    )
                    or item_value is None
                )
            }

        elif (
            isinstance(
                value,
                Sequence,
            )
            and not isinstance(
                value,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            )
        ):
            out[key] = list(
                value[:20]
            )

        else:
            out[key] = str(value)

    return out

def _publish(
    context: Any,
    topic: str,
    payload: Mapping[str, Any],
) -> None:
    events = getattr(
        context,
        "events",
        None,
    )

    publish = getattr(
        events,
        "publish",
        None,
    )

    if callable(publish):
        publish(
            topic,
            dict(payload),
        )
