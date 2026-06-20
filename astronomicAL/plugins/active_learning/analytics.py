from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import state as al_state

PREFERRED_X_COLUMNS = (
    "umap_x",
    "tsne_x",
    "pca_x",
    "embedding_x",
    "x",
    "X",
    "ra",
    "RA",
    "coords.ra",
)

PREFERRED_Y_COLUMNS = (
    "umap_y",
    "tsne_y",
    "pca_y",
    "embedding_y",
    "y",
    "Y",
    "dec",
    "DEC",
    "coords.dec",
)

METRIC_PRIORITY = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "f1_macro",
    "f1",
    "auroc",
    "auc",
    "r2",
    "val_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "loss",
    "val_loss",
    "validation_loss",
    "test_loss",
    "rmse",
    "mae",
)


def dataset_columns(context: Any, dataset_id: Any) -> List[str]:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return []

    try:
        return [str(col) for col in context.datasets.list_columns(dataset_id)]
    except Exception:
        pass

    try:
        return [str(col) for col in context.datasets.get_df(dataset_id).columns]
    except Exception:
        return []


def guess_xy_columns(context: Any, dataset_id: Any) -> Tuple[Optional[str], Optional[str]]:
    columns = dataset_columns(context, dataset_id)
    if not columns:
        return None, None

    column_set = set(columns)

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in column_set:
                return candidate
        return None

    x_col = pick(PREFERRED_X_COLUMNS)
    y_col = pick(PREFERRED_Y_COLUMNS)

    if not x_col:
        x_col = _mapped_column(context, dataset_id, "coords.ra") or _first_numeric_column(
            context,
            dataset_id,
            columns,
        )

    if not y_col:
        y_col = _mapped_column(context, dataset_id, "coords.dec") or _first_numeric_column(
            context,
            dataset_id,
            [col for col in columns if col != x_col],
        )

    if x_col == y_col:
        remaining = [col for col in columns if col != x_col]
        y_col = _first_numeric_column(context, dataset_id, remaining) or (
            remaining[0] if remaining else None
        )

    return x_col, y_col


def query_batches_for_session(context: Any, session: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return []

    refs = _find_artifacts(context, al_state.ARTIFACT_BATCH)
    batches: List[Dict[str, Any]] = []

    for ref in refs:
        artifact_id = _artifact_id(ref)
        if not artifact_id:
            continue

        try:
            payload = context.artifacts.get(artifact_id)
        except Exception:
            continue

        if not isinstance(payload, Mapping):
            continue

        if str(payload.get("session_id") or "") != session_id:
            continue

        item = dict(payload)
        item["artifact_id"] = artifact_id
        item.setdefault("timestamp", _payload_timestamp(payload, ref))
        item.setdefault("round", _safe_int(payload.get("round"), 0))
        item.setdefault(
            "strategy_id",
            payload.get("strategy_id") or payload.get("strategy") or "unknown",
        )
        item.setdefault("kind", payload.get("kind") or "query")
        batches.append(item)

    batches.sort(
        key=lambda payload: (
            _safe_float(payload.get("timestamp"), 0.0),
            _safe_int(payload.get("round"), 0),
            str(payload.get("artifact_id") or ""),
        )
    )

    for index, payload in enumerate(batches):
        payload["frame_index"] = index
        payload["frame_count"] = len(batches)

    return batches


def training_rounds_for_session(context: Any, session: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return []

    refs = _find_artifacts(context, al_state.ARTIFACT_TRAINING_SET)
    rounds: List[Dict[str, Any]] = []

    for ref in refs:
        artifact_id = _artifact_id(ref)
        if not artifact_id:
            continue

        try:
            payload = context.artifacts.get(artifact_id)
        except Exception:
            continue

        if not isinstance(payload, Mapping):
            continue

        if str(payload.get("session_id") or "") != session_id:
            continue

        item = dict(payload)
        item["artifact_id"] = artifact_id
        item.setdefault("timestamp", _payload_timestamp(payload, ref))
        item.setdefault("round", _safe_int(payload.get("round"), 0))
        item.setdefault("trained_count", len(list(payload.get("row_ids") or [])))
        rounds.append(item)

    rounds.sort(
        key=lambda payload: (
            _safe_int(payload.get("round"), 0),
            _safe_float(payload.get("timestamp"), 0.0),
            str(payload.get("artifact_id") or ""),
        )
    )
    return rounds


def performance_dataframe(context: Any, session: Mapping[str, Any]) -> pd.DataFrame:
    training_rounds = training_rounds_for_session(context, session)
    history = list(session.get("history") or [])

    history_by_training_artifact: Dict[str, Mapping[str, Any]] = {}
    history_by_round: Dict[int, Mapping[str, Any]] = {}

    for event in history:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("event") or "") != "training_round_completed":
            continue

        training_artifact_id = str(event.get("training_artifact_id") or "")
        if training_artifact_id:
            history_by_training_artifact[training_artifact_id] = event

        history_by_round[_safe_int(event.get("round"), 0)] = event

    rows: List[Dict[str, Any]] = []

    for training in training_rounds:
        round_index = _safe_int(training.get("round"), 0)
        event = history_by_training_artifact.get(str(training.get("artifact_id") or ""))
        if event is None:
            event = history_by_round.get(round_index, {})

        metric_name, metric_value = _metric_from_training_event(context, event)

        rows.append(
            {
                "round": round_index,
                "trained_count": _safe_int(
                    training.get("trained_count"),
                    len(list(training.get("row_ids") or [])),
                ),
                "metric_name": metric_name or "metric",
                "metric_value": metric_value,
                "training_artifact_id": training.get("artifact_id"),
                "training_dataset_id": training.get("training_dataset_id"),
                "timestamp": _safe_float(training.get("timestamp"), 0.0),
            }
        )

    return pd.DataFrame(rows)


def informativeness_dataframe(context: Any, session: Mapping[str, Any]) -> pd.DataFrame:
    batches = query_batches_for_session(context, session)
    rows: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(batches):
        records = [
            record
            for record in list(batch.get("records") or [])
            if isinstance(record, Mapping)
        ]

        scores = [
            _safe_float(_record_score(record), math.nan)
            for record in records
        ]
        scores = [score for score in scores if math.isfinite(score)]

        if scores:
            series = pd.Series(scores, dtype="float64")
            mean_score = float(series.mean())
            median_score = float(series.median())
            max_score = float(series.max())
            p90_score = float(series.quantile(0.9))
        else:
            mean_score = median_score = max_score = p90_score = math.nan

        rows.append(
            {
                "batch_index": batch_index + 1,
                "round": _safe_int(batch.get("round"), 0),
                "strategy_id": str(
                    batch.get("strategy_id")
                    or batch.get("strategy")
                    or "unknown"
                ),
                "kind": str(batch.get("kind") or "query"),
                "count": len(records),
                "mean_informativeness": mean_score,
                "median_informativeness": median_score,
                "max_informativeness": max_score,
                "p90_informativeness": p90_score,
                "batch_artifact_id": batch.get("artifact_id"),
                "timestamp": _safe_float(batch.get("timestamp"), 0.0),
            }
        )

    return pd.DataFrame(rows)


def training_scatter_dataframe(
    context: Any,
    session: Mapping[str, Any],
    *,
    frame_index: int,
    x_column: str,
    y_column: str,
    max_points: int = 100_000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    batches = query_batches_for_session(context, session)
    training_rounds = training_rounds_for_session(context, session)

    if batches:
        frame_index = max(0, min(int(frame_index or 0), len(batches) - 1))
        frame = dict(batches[frame_index])
    else:
        frame_index = 0
        frame = {
            "frame_index": 0,
            "frame_count": 0,
            "round": _safe_int(session.get("round"), 0),
            "strategy_id": "none",
            "kind": "session",
            "artifact_id": None,
            "records": [],
        }

    frame_round = _safe_int(
        frame.get("round"),
        _safe_int(session.get("round"), 0),
    )

    trained_row_ids: List[str] = []
    trained_round_by_row: Dict[str, int] = {}

    for training in training_rounds:
        training_round = _safe_int(training.get("round"), 0)
        if training_round > frame_round:
            continue

        for row_id in training.get("row_ids") or []:
            row_id = str(row_id)
            if row_id not in trained_round_by_row:
                trained_round_by_row[row_id] = training_round
                trained_row_ids.append(row_id)

    # Fallback before a first al.training_set artifact exists.
    if not trained_row_ids:
        for entry in dict(session.get("labels") or {}).values():
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("status") or "") != "verified":
                continue

            row_id = str(entry.get("row_id") or "").strip()
            if not row_id:
                continue

            label_round = _safe_int(entry.get("round"), 0)
            if label_round <= frame_round:
                trained_round_by_row[row_id] = label_round
                trained_row_ids.append(row_id)

    trained_row_ids = list(dict.fromkeys(trained_row_ids))[:max_points]

    score_by_row: Dict[str, float] = {}
    strategy_by_row: Dict[str, str] = {}
    source_batch_by_row: Dict[str, str] = {}
    queried_round_by_row: Dict[str, int] = {}

    for batch in batches[: frame_index + 1]:
        strategy = str(
            batch.get("strategy_id")
            or batch.get("strategy")
            or "unknown"
        )
        batch_id = str(batch.get("artifact_id") or "")
        batch_round = _safe_int(batch.get("round"), 0)

        for record in batch.get("records") or []:
            if not isinstance(record, Mapping):
                continue

            row_id = str(record.get("row_id") or "").strip()
            if not row_id:
                continue

            score = _safe_float(_record_score(record), math.nan)
            if math.isfinite(score):
                score_by_row[row_id] = score

            strategy_by_row[row_id] = str(
                record.get("active_learning_strategy")
                or strategy
            )
            source_batch_by_row[row_id] = batch_id
            queried_round_by_row[row_id] = batch_round

    dataset_id = str(
        session.get("pool_dataset_id")
        or session.get("dataset_id")
        or frame.get("dataset_id")
        or ""
    ).strip()

    coords = _read_rows_for_ids(
        context,
        dataset_id=dataset_id,
        row_ids=trained_row_ids,
        columns=[x_column, y_column],
    )

    if coords.empty:
        return pd.DataFrame(), frame

    rows: List[Dict[str, Any]] = []

    for _, row in coords.iterrows():
        row_id = str(row.get("row_id") or row.get("__row_id") or "").strip()
        if not row_id:
            continue

        x_value = _safe_float(row.get(x_column), math.nan)
        y_value = _safe_float(row.get(y_column), math.nan)

        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue

        score = score_by_row.get(row_id, math.nan)

        rows.append(
            {
                "row_id": row_id,
                "x": x_value,
                "y": y_value,
                "informativeness_score": score,
                "strategy_id": strategy_by_row.get(row_id, "initial_random"),
                "source_batch_artifact_id": source_batch_by_row.get(row_id, ""),
                "queried_round": queried_round_by_row.get(row_id),
                "trained_round": trained_round_by_row.get(row_id),
                "has_score": math.isfinite(score),
            }
        )

    return pd.DataFrame(rows), frame


def make_empty_figure(message: str = "No active-learning analytics data available yet.") -> Any:
    from bokeh.plotting import figure

    p = figure(
        height=360,
        sizing_mode="stretch_width",
        toolbar_location="above",
        title=message,
    )
    p.grid.visible = False
    p.axis.visible = False
    return p


def make_training_scatter_figure(
    df: pd.DataFrame,
    *,
    frame: Mapping[str, Any],
    x_column: str,
    y_column: str,
) -> Any:
    from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    if df is None or df.empty:
        return make_empty_figure("No trained points to show for this AL round/frame.")

    frame_number = int(frame.get("frame_index", 0)) + 1
    frame_count = int(frame.get("frame_count", 0) or frame_number)
    strategy = str(frame.get("strategy_id") or "unknown")
    round_index = _safe_int(frame.get("round"), 0)

    p = figure(
        height=430,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        title=(
            f"AL trained-point map — frame {frame_number}/{frame_count}, "
            f"round {round_index}, strategy {strategy}"
        ),
        x_axis_label=str(x_column),
        y_axis_label=str(y_column),
    )

    scored = df[df["has_score"] == True].copy()  # noqa: E712
    missing = df[df["has_score"] != True].copy()  # noqa: E712

    renderers = []

    if not scored.empty:
        low = float(scored["informativeness_score"].min())
        high = float(scored["informativeness_score"].max())

        if not math.isfinite(low):
            low = 0.0
        if not math.isfinite(high):
            high = low + 1.0
        if high <= low:
            high = low + 1e-9

        mapper = LinearColorMapper(
            palette=Viridis256,
            low=low,
            high=high,
        )
        source = ColumnDataSource(scored)

        renderer = p.scatter(
            x="x",
            y="y",
            source=source,
            size=7,
            alpha=0.82,
            line_alpha=0.25,
            fill_color={"field": "informativeness_score", "transform": mapper},
            legend_label="Scored query points",
        )
        renderers.append(renderer)

        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=6),
            label_standoff=8,
            title="Query strategy value",
        )
        p.add_layout(color_bar, "right")

    if not missing.empty:
        source = ColumnDataSource(missing)

        renderer = p.scatter(
            x="x",
            y="y",
            source=source,
            size=6,
            alpha=0.45,
            line_alpha=0.15,
            color="#9e9e9e",
            legend_label="No query score / initial random",
        )
        renderers.append(renderer)

    if renderers:
        p.add_tools(
            HoverTool(
                renderers=renderers,
                tooltips=[
                    ("row_id", "@row_id"),
                    ("strategy", "@strategy_id"),
                    ("query value", "@informativeness_score"),
                    ("queried round", "@queried_round"),
                    ("trained round", "@trained_round"),
                    (str(x_column), "@x"),
                    (str(y_column), "@y"),
                ],
            )
        )

    try:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    except Exception:
        pass

    return p


def make_performance_figure(df: pd.DataFrame) -> Any:
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    if df is None or df.empty:
        return make_empty_figure("No completed training rounds to plot yet.")

    metric_rows = df[pd.to_numeric(df.get("metric_value"), errors="coerce").notna()].copy()
    if metric_rows.empty:
        return make_empty_figure(
            "Training rounds exist, but no numeric performance metric was found in the ML result."
        )

    metric_name = str(metric_rows.iloc[-1].get("metric_name") or "metric")
    source = ColumnDataSource(metric_rows)

    p = figure(
        height=280,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        title=f"Images trained on vs performance ({metric_name})",
        x_axis_label="Number of trained images",
        y_axis_label=metric_name,
    )

    line = p.line(
        x="trained_count",
        y="metric_value",
        source=source,
        line_width=2,
    )
    points = p.scatter(
        x="trained_count",
        y="metric_value",
        source=source,
        size=8,
    )

    p.add_tools(
        HoverTool(
            renderers=[line, points],
            tooltips=[
                ("round", "@round"),
                ("trained", "@trained_count"),
                ("metric", "@metric_name"),
                ("value", "@metric_value"),
                ("training artifact", "@training_artifact_id"),
            ],
        )
    )

    return p


def make_informativeness_figure(df: pd.DataFrame) -> Any:
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    if df is None or df.empty:
        return make_empty_figure("No query batches to plot yet.")

    source = ColumnDataSource(df)

    p = figure(
        height=280,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        title="Query informativeness over time",
        x_axis_label="Query batch",
        y_axis_label="Query strategy value",
    )

    renderers = []

    for key, label in (
        ("mean_informativeness", "Mean"),
        ("median_informativeness", "Median"),
        ("p90_informativeness", "P90"),
        ("max_informativeness", "Max"),
    ):
        if key in df.columns and pd.to_numeric(df[key], errors="coerce").notna().any():
            renderers.append(
                p.line(
                    x="batch_index",
                    y=key,
                    source=source,
                    line_width=2,
                    legend_label=label,
                )
            )
            renderers.append(
                p.scatter(
                    x="batch_index",
                    y=key,
                    source=source,
                    size=6,
                    legend_label=label,
                )
            )

    if renderers:
        p.add_tools(
            HoverTool(
                renderers=renderers,
                tooltips=[
                    ("batch", "@batch_index"),
                    ("round", "@round"),
                    ("strategy", "@strategy_id"),
                    ("count", "@count"),
                    ("mean", "@mean_informativeness"),
                    ("median", "@median_informativeness"),
                    ("p90", "@p90_informativeness"),
                    ("max", "@max_informativeness"),
                ],
            )
        )

    try:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    except Exception:
        pass

    return p


def _read_rows_for_ids(
    context: Any,
    *,
    dataset_id: str,
    row_ids: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    row_ids = [str(row_id) for row_id in row_ids if row_id is not None]
    if not row_ids:
        return pd.DataFrame()

    id_column = _record_id_column(context, dataset_id)
    wanted_columns = [str(col) for col in columns if col]

    if id_column:
        wanted_columns = list(dict.fromkeys([id_column, *wanted_columns]))

    row_id_set = set(row_ids)

    if id_column:
        try:
            df = context.datasets.get_rows_by_ids(
                dataset_id,
                row_ids,
                id_column=id_column,
            )
            df = df.copy()
            if id_column in df.columns and all(col in df.columns for col in columns):
                df["__row_id"] = df[id_column].astype(str)
                return df
        except Exception:
            pass

    try:
        try:
            df = context.datasets.get_df(dataset_id, columns=wanted_columns or None)
        except TypeError:
            df = context.datasets.get_df(dataset_id)
    except Exception:
        return pd.DataFrame()

    df = df.copy()

    if id_column and id_column in df.columns:
        df["__row_id"] = df[id_column].astype(str)
        return df[df["__row_id"].isin(row_id_set)].copy()

    index_lookup = {str(idx): idx for idx in df.index}
    real_indices = [
        index_lookup[row_id]
        for row_id in row_ids
        if row_id in index_lookup
    ]

    if not real_indices:
        return pd.DataFrame()

    out = df.loc[real_indices].copy()
    out["__row_id"] = [str(idx) for idx in out.index]
    return out


def _record_id_column(context: Any, dataset_id: str) -> Optional[str]:
    for semantic in ("record_id", "id", "row_id"):
        column = _mapped_column(context, dataset_id, semantic)
        if column:
            return column

    columns = dataset_columns(context, dataset_id)
    for candidate in ("record_id", "id", "ID", "source_id", "object_id", "row_id"):
        if candidate in columns:
            return candidate

    return None


def _mapped_column(context: Any, dataset_id: Any, semantic: str) -> Optional[str]:
    try:
        value = context.datasets.get_mapping(dataset_id, semantic)
    except Exception:
        value = None

    value = str(value or "").strip()
    return value or None


def _first_numeric_column(
    context: Any,
    dataset_id: Any,
    columns: Sequence[str],
) -> Optional[str]:
    if not columns:
        return None

    try:
        df = context.datasets.get_df(dataset_id, columns=list(columns[:12]))
    except TypeError:
        try:
            df = context.datasets.get_df(dataset_id)
        except Exception:
            return columns[0] if columns else None
    except Exception:
        return columns[0] if columns else None

    for column in columns:
        if column not in df.columns:
            continue

        try:
            series = pd.to_numeric(df[column].dropna().head(50), errors="coerce")
            if series.notna().any():
                return str(column)
        except Exception:
            continue

    return columns[0] if columns else None


def _find_artifacts(context: Any, artifact_type: str) -> List[Any]:
    try:
        return list(context.artifacts.find(type=artifact_type) or [])
    except Exception:
        return []


def _artifact_id(ref: Any) -> str:
    if isinstance(ref, Mapping):
        return str(ref.get("artifact_id") or ref.get("id") or "")
    return str(getattr(ref, "artifact_id", None) or getattr(ref, "id", None) or "")


def _payload_timestamp(payload: Mapping[str, Any], ref: Any = None) -> float:
    for key in ("timestamp", "updated_at", "created_at"):
        value = payload.get(key)
        parsed = _safe_float(value, math.nan)
        if math.isfinite(parsed):
            return parsed

    if ref is not None:
        for key in ("created_at", "updated_at", "timestamp"):
            parsed = _safe_float(getattr(ref, key, None), math.nan)
            if math.isfinite(parsed):
                return parsed

    return 0.0


def _metric_from_training_event(
    context: Any,
    event: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    ml_result = event.get("ml_result") if isinstance(event, Mapping) else None
    if not isinstance(ml_result, Mapping):
        ml_result = {}

    metric = _extract_metric(ml_result)
    if metric[0] is not None:
        return metric

    for key in (
        "evaluation_report_artifact_id",
        "evaluation_artifact_id",
        "evaluation_report_id",
        "eval_artifact_id",
        "report_artifact_id",
    ):
        artifact_id = str(ml_result.get(key) or "").strip()
        if not artifact_id:
            continue

        try:
            payload = context.artifacts.get(artifact_id)
        except Exception:
            continue

        if isinstance(payload, Mapping):
            metric = _extract_metric(payload)
            if metric[0] is not None:
                return metric

    return None, None


def _extract_metric(payload: Mapping[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    candidates: List[Tuple[str, Any]] = []

    def visit(prefix: str, value: Any, depth: int = 0) -> None:
        if depth > 4:
            return

        if isinstance(value, Mapping):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(name, child, depth + 1)
            return

        parsed = _safe_float(value, math.nan)
        if math.isfinite(parsed):
            candidates.append((prefix, parsed))

    visit("", payload)

    if not candidates:
        return None, None

    lower_to_metric = {
        name.lower().replace(" ", "_"): (name, value)
        for name, value in candidates
    }

    for preferred in METRIC_PRIORITY:
        for name, value in lower_to_metric.values():
            normalized = name.lower().replace(" ", "_")
            if normalized.endswith(preferred) or normalized == preferred:
                return name.split(".")[-1], value

    name, value = candidates[0]
    return name.split(".")[-1], value


def _record_score(record: Mapping[str, Any]) -> Any:
    for key in (
        "informativeness_score",
        "active_learning_score",
        "uncertainty_score",
        "score",
        "entropy",
        "least_confidence",
        "margin_uncertainty",
    ):
        if key in record:
            return record.get(key)

    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        parsed = float(value)
        if math.isnan(parsed):
            return default

        return parsed
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default