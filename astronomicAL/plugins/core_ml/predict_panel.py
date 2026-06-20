from __future__ import annotations

import html as _html
import importlib.util
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import panel as pn

from astronomicAL.platform.plugins.specs import ActionRequest

# Platform parquet-cache helpers (same ones core.table_tools uses to add
# columns to a dataset). Guarded so the panel still imports if they move.
try:
    from astronomicAL.platform.parquet_cache import (
        default_cache_dir_for_context,
        normalise_dataset_id,
        replace_dataset_with_dataframe_parquet,
    )
except Exception:  # pragma: no cover
    default_cache_dir_for_context = None
    normalise_dataset_id = None
    replace_dataset_with_dataframe_parquet = None


_PROV_META = {
    "novel": ("New data", "the real test — never seen by the model", True),
    "test": ("Held-out test", "the real test — held out during training", True),
    "validation": ("Tuning set", "used to tune the model — slightly optimistic", False),
    "train": ("Training data", "the model learned from these — reference only", False),
    "unknown": ("Unverified", "couldn't confirm whether the model saw these", False),
}
_PROV_ORDER = ["novel", "test", "validation", "train", "unknown"]

# Prediction-table columns we attach, mapped to friendly dataset column names.
_ATTACH_FRIENDLY = {
    "predicted_label": "pred_label",
    "prediction_confidence": "pred_confidence",
    "data_provenance": "pred_provenance",
    "entropy": "pred_entropy",
    "least_confidence": "pred_least_confidence",
    "margin_uncertainty": "pred_margin_uncertainty",
    "true_label": "pred_true_label",
    "is_correct": "pred_correct",
}
_ATTACH_PRIORITY = list(_ATTACH_FRIENDLY.keys())


# =============================================================================
# DuckDB relation helpers (adapted from core.table_tools) + a lazy join source
# =============================================================================

def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _quote_sql_string(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _is_duckdb_relation_source(source: Any) -> bool:
    if source is None:
        return False
    return (callable(getattr(source, "_connect", None))
            and callable(getattr(source, "_relation_sql", None))
            and callable(getattr(source, "_path_argument", None)))


def _duckdb_relation_params(source: Any) -> list:
    if source is None:
        return []
    seen: set = set()

    def _walk(src: Any) -> list:
        if src is None:
            return []
        oid = id(src)
        if oid in seen:
            raise RuntimeError("Cycle in DuckDB source chain.")
        seen.add(oid)
        base = getattr(src, "base_source", None)
        if base is not None:
            params = _walk(base)
            where_params = getattr(src, "where_params", None)
            if where_params:
                params.extend(list(where_params))
            seen.discard(oid)
            return params
        path_argument = getattr(src, "_path_argument", None)
        seen.discard(oid)
        return [path_argument()] if callable(path_argument) else []

    return _walk(source)


def _duckdb_relation_query_sql(source: Any) -> str:
    relation_sql = str(source._relation_sql()).strip()
    lower = relation_sql.lower()
    if lower.startswith("select ") or lower.startswith("with "):
        return relation_sql
    return f"SELECT * FROM {relation_sql}"


def _duckdb_relation_from_sql(source: Any, *, alias: str = "base") -> str:
    return f"({_duckdb_relation_query_sql(source)}) AS {_quote_identifier(alias)}"


def _duckdb_self_from_sql(source: Any, *, alias: str = "src") -> str:
    return f"({source._relation_sql()}) AS {_quote_identifier(alias)}"


class LazyPredictionJoinSource:
    """Lazy LEFT JOIN of a predictions parquet onto a DuckDB/Parquet source.

    Adds prediction columns to a dataset without copying its rows: the join is
    pushed down to DuckDB and only the requested rows/columns are ever read.
    Same duck-typed source protocol as core.table_tools' lazy sources.
    """

    backend_name = "duckdb_parquet_prediction_join"

    def __init__(self, *, base_source, predictions_path, id_column, join_key,
                 column_plan, base_columns, dataset_name=None, row_count_hint=None):
        self.base_source = base_source
        self.predictions_path = str(predictions_path)
        self.id_column = str(id_column)
        self.join_key = str(join_key)
        self.column_plan = list(column_plan)  # [(src, final), ...]
        self.dataset_name = dataset_name
        self._columns_cache = [str(c) for c in base_columns] + [f for _s, f in self.column_plan]
        self._row_count_cache = int(row_count_hint) if row_count_hint is not None else None

    def _connect(self):
        return self.base_source._connect()

    def _path_argument(self):
        return self.base_source._path_argument()

    def _relation_params(self) -> list:
        return _duckdb_relation_params(self.base_source)

    def _relation_sql(self) -> str:
        proj = ", ".join(f"preds.{_quote_identifier(src)} AS {_quote_identifier(final)}"
                         for src, final in self.column_plan)
        return (f"SELECT base.*, {proj} "
                f"FROM {_duckdb_relation_from_sql(self.base_source, alias='base')} "
                f"LEFT JOIN read_parquet({_quote_sql_string(self.predictions_path)}) AS preds "
                f"ON CAST(base.{_quote_identifier(self.id_column)} AS VARCHAR) "
                f"= CAST(preds.{_quote_identifier(self.join_key)} AS VARCHAR)")

    def _select_sql(self, *, columns=None) -> str:
        if not columns:
            return "*"
        return ", ".join(_quote_identifier(c) for c in columns)

    def columns(self) -> list:
        return list(self._columns_cache)

    def dtypes(self) -> dict:
        con = self._connect()
        try:
            df = con.execute(f"DESCRIBE SELECT * FROM {_duckdb_self_from_sql(self)} LIMIT 0",
                             self._relation_params()).df()
        finally:
            con.close()
        return {str(r["column_name"]): str(r["column_type"]) for _, r in df.iterrows()}

    def row_count(self) -> Optional[int]:
        if self._row_count_cache is not None:
            return int(self._row_count_cache)
        con = self._connect()
        try:
            res = con.execute(f"SELECT COUNT(*) FROM {_duckdb_self_from_sql(self)}",
                              self._relation_params()).fetchone()
        finally:
            con.close()
        self._row_count_cache = int(res[0]) if res else 0
        return self._row_count_cache

    def to_pandas(self, *, columns=None, limit=None, where_sql=None, params=None) -> pd.DataFrame:
        sql = f"SELECT {self._select_sql(columns=columns)} FROM {_duckdb_self_from_sql(self)}"
        sql_params = self._relation_params()
        if where_sql:
            sql += f" WHERE ({where_sql})"
            if params:
                sql_params.extend(list(params))
        if limit is not None:
            sql += " LIMIT ?"
            sql_params.append(int(limit))
        con = self._connect()
        try:
            return con.execute(sql, sql_params).df()
        finally:
            con.close()

    def head(self, n: int = 5, *, columns=None) -> pd.DataFrame:
        return self.to_pandas(columns=columns, limit=n)

    def get_row_by_position(self, position: int, *, columns=None) -> pd.DataFrame:
        if position < 0:
            return pd.DataFrame(columns=self.columns() if columns is None else columns)
        sql = (f"SELECT {self._select_sql(columns=columns)} "
               f"FROM {_duckdb_self_from_sql(self)} LIMIT 1 OFFSET ?")
        con = self._connect()
        try:
            return con.execute(sql, [*self._relation_params(), int(position)]).df()
        finally:
            con.close()

    def get_row_by_id(self, row_id, *, id_column, columns=None) -> pd.DataFrame:
        if id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=self.columns() if columns is None else columns)
        return self.to_pandas(columns=columns, limit=1,
                              where_sql=f"CAST({_quote_identifier(id_column)} AS VARCHAR) = ?",
                              params=[str(row_id)])

    def get_rows_by_ids(self, row_ids, *, id_column, columns=None) -> pd.DataFrame:
        ids = [str(r) for r in (row_ids or [])]
        if not ids or id_column == "Use Index" or id_column not in self.columns():
            return pd.DataFrame(columns=list(columns or self.columns()))
        ordered, seen = [], set()
        for r in ids:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        sel = list(columns or [])
        if id_column not in sel:
            sel.insert(0, id_column)
        sel = [c for c in sel if c in set(self.columns())] or [id_column]
        select_sql = ", ".join(f"src.{_quote_identifier(c)}" for c in sel)
        values_sql = ", ".join(["(?, ?)"] * len(ordered))
        values_params: list = []
        for order, rid in enumerate(ordered):
            values_params.extend([order, rid])
        sql = ("WITH requested(__o, __id) AS "
               f"(VALUES {values_sql}) SELECT {select_sql} "
               f"FROM {_duckdb_self_from_sql(self, alias='src')} JOIN requested "
               f"ON CAST(src.{_quote_identifier(id_column)} AS VARCHAR) = requested.__id "
               "ORDER BY requested.__o")
        con = self._connect()
        try:
            return con.execute(sql, [*values_params, *self._relation_params()]).df()
        finally:
            con.close()

    def find_position_by_id(self, row_id, *, id_column) -> Optional[int]:
        if id_column == "Use Index" or id_column not in self.columns():
            return None
        sql = ("SELECT rn FROM (SELECT ROW_NUMBER() OVER () - 1 AS rn, "
               f"{_quote_identifier(id_column)} AS rid "
               f"FROM {_duckdb_self_from_sql(self)}) AS n WHERE CAST(rid AS VARCHAR) = ? LIMIT 1")
        con = self._connect()
        try:
            res = con.execute(sql, [*self._relation_params(), str(row_id)]).fetchone()
        finally:
            con.close()
        return int(res[0]) if res else None

    def metadata(self) -> dict:
        return {"backend": self.backend_name,
                "base_backend": getattr(self.base_source, "backend_name", "unknown"),
                "dataset_name": self.dataset_name,
                "predictions_path": self.predictions_path,
                "added_columns": [f for _s, f in self.column_plan],
                "materialized": False}


# =============================================================================
# Panel
# =============================================================================

class MLPredictPanel:
    """Run a trained model on a dataset and attach its predictions as columns."""

    def __init__(self, *, context: Any, restore_state: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self._disposed = False
        self._subscriptions: List[Any] = []
        self._active_job_handle = None
        self._active_job_key = None
        self._active_job_id = None
        self._last_predictions_artifact_id: Optional[str] = None

        # --- primary choices ------------------------------------------------
        self.model = pn.widgets.Select(name="Model", options={}, sizing_mode="stretch_width")
        self.model_line = pn.pane.Markdown("", sizing_mode="stretch_width", margin=(2, 0, 0, 0))
        self.dataset = pn.widgets.Select(name="Data to run on", options=[], sizing_mode="stretch_width")
        self.run_on = pn.widgets.Select(name="Rows", options=["All rows", "Selected points only"],
                                        value="All rows", sizing_mode="stretch_width")
        self.check_accuracy = pn.widgets.Checkbox(
            name="I have the correct answers and want to measure how accurate it is",
            value=False, sizing_mode="stretch_width")
        self.target = pn.widgets.Select(name="Which column has the correct answers?", options=[],
                                        visible=False, sizing_mode="stretch_width")

        # --- advanced -------------------------------------------------------
        self.image_column = pn.widgets.Select(name="Image column (if not detected)", options=[],
                                               sizing_mode="stretch_width")
        self.skip_bad_images = pn.widgets.Checkbox(name="Skip images that can't be opened", value=True,
                                                   sizing_mode="stretch_width")
        self.device = pn.widgets.Select(name="Run on", options=["auto", "cpu", "cuda", "mps"], value="auto",
                                        sizing_mode="stretch_width")
        self.batch_size = pn.widgets.IntInput(name="Batch size", value=64, start=1, sizing_mode="stretch_width")
        self.max_rows = pn.widgets.IntInput(name="Limit number of rows (0 = all)", value=0, start=0,
                                            sizing_mode="stretch_width")
        self.decision_threshold = pn.widgets.FloatInput(
            name="Leave unlabelled if less sure than (0 = always label)",
            value=0.0, start=0.0, end=1.0, step=0.05, sizing_mode="stretch_width")
        self.require_target_compatible = pn.widgets.Checkbox(
            name="Only run if my answers use the same categories as the model", value=False,
            sizing_mode="stretch_width")
        self.attach_columns = pn.widgets.Checkbox(
            name="Add the prediction columns to this dataset", value=True, sizing_mode="stretch_width")

        # --- actions / status ----------------------------------------------
        self.run_button = pn.widgets.Button(name="Run", button_type="primary", height=42,
                                             sizing_mode="stretch_width")
        self.cancel_button = pn.widgets.Button(name="Cancel", button_type="warning", disabled=True, height=42,
                                               sizing_mode="stretch_width")
        self.refresh_button = pn.widgets.Button(name="Refresh models", button_type="light", height=32,
                                                sizing_mode="stretch_width")
        self.status = pn.pane.Alert("Pick a model and your data to begin.", alert_type="info",
                                    sizing_mode="stretch_width", margin=(8, 0, 0, 0))

        # --- results --------------------------------------------------------
        self.trust = pn.pane.Alert("", alert_type="success", sizing_mode="stretch_width", visible=False)
        self.metrics_pane = pn.pane.HTML("", sizing_mode="stretch_width", visible=False, margin=(0, 0, 6, 0))
        self.distribution_pane = pn.pane.HTML("", sizing_mode="stretch_width", visible=False, margin=(0, 0, 6, 0))
        self.result_summary = pn.pane.Markdown("Run a prediction to see results here.",
                                               sizing_mode="stretch_width")
        self.preview = pn.widgets.Tabulator(pd.DataFrame(), height=200, sizing_mode="stretch_width", disabled=True)
        self.technical = pn.pane.Markdown("Run a prediction first.", sizing_mode="stretch_width")
        self.result_json = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width", height=240)

        # --- wiring ---------------------------------------------------------
        self.refresh_button.on_click(lambda *_: self.refresh(reason="manual refresh"))
        self.run_button.on_click(lambda *_: self.predict())
        self.cancel_button.on_click(lambda *_: self.cancel())
        self.model.param.watch(lambda *_: self._on_model_change(), "value")
        self.dataset.param.watch(lambda *_: self._on_dataset_change(), "value")
        self.check_accuracy.param.watch(lambda *_: self._on_accuracy_toggle(), "value")
        self.target.param.watch(lambda *_: self.validate(), "value")
        self.image_column.param.watch(lambda *_: self.validate(), "value")
        self.require_target_compatible.param.watch(lambda *_: self.validate(), "value")

        self._subscribe_to_events()
        self.refresh(reason="panel opened")
        if restore_state:
            self.restore_state(restore_state)

    # ---------------------------------------------------------------- layout
    def panel(self):
        advanced = self._spaced(
            self.image_column, self.skip_bad_images, self.device, self.batch_size,
            self.max_rows, self.decision_threshold, self.require_target_compatible, self.attach_columns)

        results = self._spaced(
            self.trust, self.metrics_pane, self.distribution_pane, self.result_summary,
            pn.pane.HTML("<div style='font-size:12px;font-weight:600;'>Preview</div>", height=18),
            self.preview,
            pn.Accordion(("Technical details (for ML / audit)",
                          self._spaced(self.technical, self.result_json)),
                         active=[], sizing_mode="stretch_width"))

        content = pn.Column(
            self._header(),
            self._section("1.  Pick a model", self.model, self.model_line),
            self._section("2.  Pick your data", self.dataset, self.run_on),
            self._section("3.  Check accuracy (optional)", self.check_accuracy, self.target),
            pn.Accordion(("More options", advanced), active=[], sizing_mode="stretch_width", margin=(0, 0, 16, 0)),
            pn.GridBox(self.run_button, self.cancel_button, ncols=2, sizing_mode="stretch_width",
                       margin=(4, 0, 8, 0)),
            self.refresh_button, self.status,
            self._section("Results", results),
            sizing_mode="stretch_width", margin=(0, 0, 20, 0))

        return pn.Column(content, sizing_mode="stretch_both", scroll=True,
                         styles={"padding": "10px", "background": "white"})

    def _header(self):
        return pn.pane.HTML(
            """
            <div style="font-size:19px;font-weight:700;margin:0 0 4px;">Run a model on your data</div>
            <div style="font-size:13px;color:#444;margin:0 0 10px;">
              Pick a trained model, choose your data, and press Run. The predictions
              are added as new columns on your dataset so you can plot them.
            </div>
            """, sizing_mode="stretch_width", margin=(0, 0, 6, 0))

    def _spaced(self, *objects, gap: int = 8):
        items = []
        for obj in objects:
            if obj is None:
                continue
            try:
                obj.margin = (gap, 0, gap, 0)
            except Exception:
                pass
            items.append(obj)
        return pn.Column(*items, sizing_mode="stretch_width", margin=(0, 0, 0, 0))

    def _section(self, title: str, *objects):
        items = [pn.pane.HTML(
            f"<div style='font-size:14px;font-weight:700;color:#20242a;'>{title}</div>",
            height=22, margin=(0, 0, 8, 0), sizing_mode="stretch_width")]
        for obj in objects:
            if obj is None:
                continue
            try:
                obj.margin = (8, 0, 8, 0)
            except Exception:
                pass
            items.append(obj)
        return pn.Column(*items, sizing_mode="stretch_width", margin=(0, 0, 16, 0),
                         styles={"padding": "14px", "border": "1px solid #d9dee8",
                                 "border-radius": "8px", "background": "white"})

    # ----------------------------------------------------------- state I/O
    def get_state(self) -> Dict[str, Any]:
        return {
            "model": self.model.value, "dataset": self.dataset.value, "run_on": self.run_on.value,
            "check_accuracy": bool(self.check_accuracy.value), "target": self.target.value,
            "image_column": self.image_column.value, "device": self.device.value,
            "batch_size": int(self.batch_size.value or 64), "max_rows": int(self.max_rows.value or 0),
            "decision_threshold": float(self.decision_threshold.value or 0.0),
            "skip_bad_images": bool(self.skip_bad_images.value),
            "require_target_compatible": bool(self.require_target_compatible.value),
            "attach_columns": bool(self.attach_columns.value),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.refresh(reason="restore state")
        if state.get("dataset") in self.dataset.options:
            self.dataset.value = state["dataset"]
            self._load_columns()
        if state.get("model") in self._model_values():
            self.model.value = state["model"]
        if state.get("run_on") in self.run_on.options:
            self.run_on.value = state["run_on"]
        self.check_accuracy.value = bool(state.get("check_accuracy", self.check_accuracy.value))
        if state.get("target") in self.target.options:
            self.target.value = state["target"]
        if state.get("image_column") in self.image_column.options:
            self.image_column.value = state["image_column"]
        if state.get("device") in self.device.options:
            self.device.value = state["device"]
        for key, widget, cast in (("batch_size", self.batch_size, int), ("max_rows", self.max_rows, int),
                                  ("decision_threshold", self.decision_threshold, float)):
            try:
                widget.value = cast(state.get(key, widget.value) or 0)
            except Exception:
                pass
        for key, widget in (("skip_bad_images", self.skip_bad_images),
                            ("require_target_compatible", self.require_target_compatible),
                            ("attach_columns", self.attach_columns)):
            widget.value = bool(state.get(key, widget.value))
        self._on_accuracy_toggle()
        self.validate()

    def dispose(self) -> None:
        self._disposed = True
        self.cancel()
        unsubscribe = getattr(getattr(self.context, "events", None), "unsubscribe", None)
        if callable(unsubscribe):
            for sub in list(self._subscriptions):
                try:
                    unsubscribe(sub)
                except Exception:
                    pass
        self._subscriptions.clear()

    # ----------------------------------------------------------- refresh / validate
    def refresh(self, *, reason: str = "refresh", select_model_artifact_id: Optional[str] = None) -> None:
        if self._disposed:
            return
        previous = self.model.value
        self._load_datasets()
        self._load_columns()
        self._load_models()
        values = self._model_values()
        if select_model_artifact_id and select_model_artifact_id in values:
            self.model.value = select_model_artifact_id
        elif previous in values:
            self.model.value = previous
        elif (opts := getattr(self.model, "options", None)):
            if isinstance(opts, dict) and opts:
                self.model.value = next(iter(opts.values()))
            elif isinstance(opts, (list, tuple)) and opts:
                self.model.value = opts[0]
        self._update_model_line()
        self.validate()

    def validate(self) -> None:
        if self._disposed:
            return
        dataset_id, model_id = self._dataset_id(), self._model_artifact_id()
        if not model_id or not dataset_id:
            self.run_button.disabled = True
            self.status.alert_type = "info"
            self.status.object = "Pick a model and your data to begin."
            return
        catalog = self._catalog()
        if catalog is None:
            self.run_button.disabled = True
            self.status.alert_type = "danger"
            self.status.object = "Models aren't available right now. Try Refresh."
            return
        try:
            report = catalog.compatibility(
                model_id, dataset_id, target_column=self._target_column_param(),
                image_column=self._image_column_param(),
                require_target_compatible=(
                    bool(self.require_target_compatible.value) if self._is_evaluation_mode() else False))
        except Exception as exc:
            self.run_button.disabled = True
            self.status.alert_type = "danger"
            self.status.object = f"Couldn't check this model against the data ({exc})."
            return
        self._last_compat_report = report
        if str(report.get("status")) == "incompatible":
            self.run_button.disabled = True
            self.status.alert_type = "warning"
            self.status.object = self._plain_block_reason(report)
        else:
            self.run_button.disabled = False
            self.status.alert_type = "success"
            self.status.object = ("Ready — Run will label your data and measure accuracy."
                                  if self._is_evaluation_mode() else "Ready — Run will label your data.")

    # ----------------------------------------------------------- run
    def predict(self) -> None:
        dataset_id, model_id = self._dataset_id(), self._model_artifact_id()
        if not dataset_id or not model_id:
            self.status.alert_type = "danger"
            self.status.object = "Pick both a model and your data."
            return
        self.validate()
        if self.run_button.disabled:
            return

        params = {
            "dataset_id": dataset_id, "model_artifact_id": model_id,
            "scope": "evaluation" if self._is_evaluation_mode() else "inference",
            "target_column": self._target_column_param(), "image_column": self._image_column_param(),
            # The panel attaches columns to the live dataset itself, so the
            # action must NOT spin up a separate predictions dataset.
            "register_prediction_dataset": False,
            "require_target_compatible": (
                bool(self.require_target_compatible.value) if self._is_evaluation_mode() else False),
            "device": str(self.device.value or "auto"),
            "image_batch_size": int(self.batch_size.value or 64), "batch_size": int(self.batch_size.value or 64),
            "skip_bad_images": bool(self.skip_bad_images.value), "max_rows": int(self.max_rows.value or 0),
            "run_id": uuid.uuid4().hex,
        }
        threshold = float(self.decision_threshold.value or 0.0)
        if threshold > 0.0:
            params["decision_threshold"] = threshold

        request = ActionRequest(dataset_id=dataset_id, row_ids=self._row_ids_for_scope(), params=params,
                                artifact_id=model_id, origin="core.ml.predict_panel")
        self._set_running(True)
        self.status.alert_type = "info"
        self.status.object = "Working…"

        from . import prediction

        submit = getattr(getattr(self.context, "jobs", None), "submit", None)
        key = f"core.ml.predict:{dataset_id}:{model_id}"
        if callable(submit):
            self._active_job_key = key
            submitted = submit(prediction.predict_action, title="Run model on data", key=key,
                               on_done=self._on_predict_done, on_error=self._on_predict_error,
                               context=self.context, request=request)
            self._active_job_handle = submitted
            self._active_job_id = getattr(submitted, "job_id", None) or getattr(submitted, "id", None)
            return
        try:
            self._on_predict_done(prediction.predict_action(context=self.context, request=request))
        except Exception as exc:
            self._on_predict_error(exc)

    def cancel(self) -> None:
        jobs = getattr(self.context, "jobs", None)
        for method_name, value in (("cancel", self._active_job_id), ("cancel", self._active_job_key),
                                   ("cancel_job", self._active_job_id), ("cancel_job", self._active_job_key),
                                   ("cancel_by_key", self._active_job_key)):
            if not value:
                continue
            method = getattr(jobs, method_name, None)
            if callable(method):
                try:
                    method(value)
                    break
                except TypeError:
                    continue
                except Exception:
                    break
        self._set_running(False)

    # ----------------------------------------------------------- results
    def _on_predict_done(self, result: Any) -> None:
        self._set_running(False)
        if not isinstance(result, dict):
            self._fail("Something went wrong", "The model didn't return a usable result.", result)
            return
        if not result.get("ok", True):
            reason = self._plain_block_reason(result.get("compatibility_report") or {}) \
                or "This model can't run on this data."
            self._fail("Couldn't run", reason, result)
            return

        artifact_id = result.get("artifact_id")
        self._last_predictions_artifact_id = str(artifact_id) if artifact_id else None
        rows, failed_count, _failed = self._fetch_table(artifact_id, result)
        try:
            self.preview.value = pd.DataFrame(rows[:200])
        except Exception:
            self.preview.value = pd.DataFrame()

        # Attach prediction columns to the dataset that was predicted on.
        added, attach_error = self._attach_predictions(result, rows)

        self.metrics_pane.object = self._metrics_html(result)
        self.metrics_pane.visible = bool(self.metrics_pane.object)
        self.distribution_pane.object = self._distribution_html(rows)
        self.distribution_pane.visible = bool(self.distribution_pane.object)

        verdict = self._trust_verdict(result)
        if verdict:
            self._show_trust(*verdict)
        else:
            self.trust.visible = False

        self.result_summary.object = self._render_summary_plain(result, failed_count, added, attach_error)
        self.technical.object = self._render_technical(result)
        self.result_json.object = result

        warned = bool((verdict and verdict[0] != "success") or attach_error)
        self.status.alert_type = "warning" if warned else "success"
        self.status.object = "Done — see the note above the results." if warned else "Done."
        self.refresh(reason="prediction complete")

    def _on_predict_error(self, error: Any) -> None:
        self._set_running(False)
        self._fail("Couldn't run", str(error), {"error": str(error)})

    def _fail(self, title: str, detail: str, result: Any) -> None:
        self._show_trust("danger", title, detail)
        self.metrics_pane.visible = False
        self.distribution_pane.visible = False
        self.result_summary.object = detail
        self.result_json.object = result if isinstance(result, (dict, list)) else {"result": str(result)}
        self.status.alert_type = "danger"
        self.status.object = title + "."

    # ---- attach prediction columns to the dataset --------------------------
    def _attach_predictions(self, result: Mapping[str, Any], rows: List[Mapping[str, Any]]):
        """Add prediction columns to the predicted dataset. Returns
        (added_column_names, error_message)."""
        if not bool(self.attach_columns.value):
            return [], None
        if not rows:
            return [], "there were no predictions to attach"
        dataset_id = str(result.get("dataset_id") or self._dataset_id() or "")
        if not dataset_id:
            return [], "couldn't tell which dataset to update"

        plan = self._attach_plan(rows, dataset_id)
        if not plan:
            return [], "there were no prediction columns to attach"

        id_column = self._resolve_id_column(dataset_id)
        source = self._base_source(dataset_id)
        added = [final for _src, final in plan]

        try:
            preds_path = self._write_predictions_parquet(rows, plan, dataset_id)
        except Exception as exc:
            return [], f"couldn't stage the predictions ({exc})"

        # Preferred: lazy DuckDB join (no row copy).
        if _is_duckdb_relation_source(source) and id_column and id_column != "Use Index":
            try:
                self._attach_lazy(dataset_id, source, preds_path, plan, id_column)
                self._emit_dataset_updated(dataset_id, added)
                return added, None
            except Exception:
                pass  # fall back to materialise

        # Fallback: materialise + replace (pandas datasets, or if lazy failed).
        try:
            self._attach_materialise(dataset_id, source, rows, plan, id_column)
            self._emit_dataset_updated(dataset_id, added)
            return added, None
        except Exception as exc:
            return [], f"couldn't add the columns ({exc})"

    def _attach_plan(self, rows, dataset_id) -> List[Tuple[str, str]]:
        base_cols = set(self._columns())
        keys: set = set()
        for r in rows[:200]:
            keys.update(r.keys())

        src_cols = [k for k in _ATTACH_PRIORITY if k in keys]
        src_cols += sorted(k for k in keys if k.startswith("prob_"))
        seen: set = set()
        src_cols = [k for k in src_cols if not (k in seen or seen.add(k))]

        used = set(base_cols)
        run8 = (self._last_predictions_artifact_id or uuid.uuid4().hex)[-8:]
        plan: List[Tuple[str, str]] = []
        for src in src_cols:
            final = _ATTACH_FRIENDLY.get(src, f"pred_{src}")
            if final in used:
                final = f"{final}_{run8}"
            base_final, i = final, 2
            while final in used:
                final = f"{base_final}_{i}"
                i += 1
            used.add(final)
            plan.append((src, final))
        return plan

    def _write_predictions_parquet(self, rows, plan, dataset_id) -> Path:
        cache_dir = self._cache_dir()
        run8 = (self._last_predictions_artifact_id or uuid.uuid4().hex)[-8:]
        safe = normalise_dataset_id(dataset_id) if callable(normalise_dataset_id) \
            else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_id)).strip("._-") or "dataset"
        path = cache_dir / f"predictions_attach_{safe}_{run8}.parquet"
        records = [{"record_id": r.get("record_id"), **{src: r.get(src) for src, _f in plan}} for r in rows]
        pd.DataFrame(records).to_parquet(path, index=False)
        return path

    def _attach_lazy(self, dataset_id, source, preds_path, plan, id_column) -> None:
        base_columns = self._columns()
        row_count = self._row_count(dataset_id)
        join_source = LazyPredictionJoinSource(
            base_source=source, predictions_path=preds_path, id_column=id_column,
            join_key="record_id", column_plan=plan, base_columns=base_columns,
            dataset_name=self._dataset_name(dataset_id), row_count_hint=row_count)
        # Validate by actually querying before we replace the live source.
        join_source.columns()
        join_source.head(1)
        meta = self._dataset_meta(dataset_id)
        meta.update({"backend": join_source.backend_name,
                     "columns": base_columns + [f for _s, f in plan],
                     "added_columns": [f for _s, f in plan],
                     "row_count": row_count, "rows": row_count,
                     "created_by": "core.ml.predict_panel", "materialized": False})
        self.context.datasets.register_source(dataset_id, join_source,
                                               name=self._dataset_name(dataset_id), **meta)

    def _attach_materialise(self, dataset_id, source, rows, plan, id_column) -> None:
        if replace_dataset_with_dataframe_parquet is None or default_cache_dir_for_context is None:
            raise RuntimeError("the platform parquet-cache helpers aren't available")
        if source is None or not hasattr(source, "to_pandas"):
            raise RuntimeError("couldn't read the dataset")
        df = source.to_pandas()
        if id_column and id_column != "Use Index" and id_column in df.columns:
            key = df[id_column].astype(str)
        else:
            key = pd.Series(df.index.astype(str), index=df.index)
        for src, final in plan:
            value_by_id = {str(r.get("record_id")): r.get(src) for r in rows}
            df[final] = key.map(value_by_id)
        meta = self._dataset_meta(dataset_id)
        meta.update({"added_columns": [f for _s, f in plan],
                     "created_by": "core.ml.predict_panel", "materialized": True})
        replace_dataset_with_dataframe_parquet(
            self.context, dataset_id=dataset_id, df=df, name=self._dataset_name(dataset_id),
            cache_dir=self._cache_dir(), **meta)

    def _emit_dataset_updated(self, dataset_id, added_columns) -> None:
        publish = getattr(getattr(self.context, "events", None), "publish", None)
        if callable(publish):
            try:
                publish("dataset.updated", {
                    "dataset_id": dataset_id, "change": "column.added",
                    "added_columns": list(added_columns), "changed_columns": list(added_columns),
                    "schema_changed": True, "origin": "core.ml.predict_panel"})
            except Exception:
                pass

    def _resolve_id_column(self, dataset_id) -> Optional[str]:
        get_mapping = getattr(getattr(self.context, "datasets", None), "get_mapping", None)
        if callable(get_mapping):
            try:
                value = get_mapping(dataset_id, "record_id")
                if value:
                    return str(value)
            except Exception:
                pass
        binding = dict((self._last_compat_report or {}).get("resolved_input_binding") or {}) \
            if hasattr(self, "_last_compat_report") else {}
        return binding.get("record_id_column")

    def _base_source(self, dataset_id):
        get_source = getattr(getattr(self.context, "datasets", None), "get_source", None)
        if callable(get_source):
            try:
                return get_source(dataset_id)
            except Exception:
                return None
        return None

    def _cache_dir(self) -> Path:
        if callable(default_cache_dir_for_context):
            try:
                d = default_cache_dir_for_context(self.context)
                d.mkdir(parents=True, exist_ok=True)
                return d
            except Exception:
                pass
        d = Path.cwd() / ".astronomical" / "ml_predictions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _dataset_name(self, dataset_id) -> str:
        get = getattr(getattr(self.context, "datasets", None), "get", None)
        if callable(get):
            try:
                return str(get(dataset_id).name)
            except Exception:
                pass
        return str(dataset_id)

    def _dataset_meta(self, dataset_id) -> Dict[str, Any]:
        get_meta = getattr(getattr(self.context, "datasets", None), "get_meta", None)
        if callable(get_meta):
            try:
                return dict(get_meta(dataset_id) or {})
            except Exception:
                pass
        return {}

    def _row_count(self, dataset_id) -> Optional[int]:
        row_count = getattr(getattr(self.context, "datasets", None), "row_count", None)
        if callable(row_count):
            try:
                v = row_count(dataset_id)
                return int(v) if v is not None else None
            except Exception:
                pass
        return None

    # ---- visible numbers ---------------------------------------------------
    def _metrics_html(self, result: Mapping[str, Any]) -> str:
        by = (result.get("evaluation") or {}).get("metrics_by_provenance") or {}
        if not by:
            return ""
        cards = []
        for key in _PROV_ORDER:
            if key not in by:
                continue
            label, note, reportable = _PROV_META.get(key, (key, "", False))
            m = by[key]
            acc = m.get("accuracy")
            acc_str = f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "—"
            f1 = m.get("f1_macro")
            f1_str = f" · F1 {f1:.2f}" if isinstance(f1, (int, float)) else ""
            n = int(m.get("n") or 0)
            accent = "#1a7f37" if reportable else "#8a8f98"
            bg = "#f1f9f3" if reportable else "#f6f7f9"
            cards.append(
                f"<div style='flex:1;min-width:150px;border:1px solid #e1e5ec;border-left:4px solid {accent};"
                f"border-radius:8px;background:{bg};padding:12px;'>"
                f"<div style='font-size:30px;font-weight:800;color:{accent};line-height:1;'>{acc_str}</div>"
                f"<div style='font-size:13px;font-weight:700;margin-top:4px;color:#20242a;'>{label}</div>"
                f"<div style='font-size:11px;color:#666;margin-top:2px;'>{n:,} rows{f1_str}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:4px;'>{note}</div></div>")
        if not cards:
            return ""
        footnote = self._overfit_note(by)
        foot = f"<div style='font-size:12px;color:#b06b00;margin-top:8px;'>{footnote}</div>" if footnote else ""
        return ("<div style='font-size:14px;font-weight:700;margin:0 0 8px;'>How accurate it was</div>"
                "<div style='display:flex;flex-wrap:wrap;gap:10px;'>" + "".join(cards) + "</div>" + foot)

    def _overfit_note(self, by: Mapping[str, Any]) -> str:
        train = by.get("train")
        unseen = by.get("test") or by.get("novel")
        if not train or not unseen:
            return ""
        ta, ua = train.get("accuracy"), unseen.get("accuracy")
        if isinstance(ta, (int, float)) and isinstance(ua, (int, float)) and (ta - ua) >= 0.15:
            return "⚠ Much better on training data than on new data — the model may be overfitting."
        return ""

    def _distribution_html(self, rows: List[Mapping[str, Any]]) -> str:
        values, abstained = [], 0
        for r in rows:
            v = r.get("predicted_label", r.get("prediction"))
            if v is None or (isinstance(v, float) and pd.isna(v)):
                abstained += 1
                continue
            values.append(v)
        if not values:
            return ""
        task = str((self._selected_model_descriptor() or {}).get("task") or "").lower()
        total = len(values)
        extra = f" · {abstained:,} left unlabelled" if abstained else ""

        if task == "regression" and all(self._is_number(v) for v in values):
            nums = sorted(float(v) for v in values)
            lo, hi = nums[0], nums[-1]
            bins = 8
            width = (hi - lo) / bins if hi > lo else 1.0
            counts = [0] * bins
            for v in nums:
                idx = min(bins - 1, int((v - lo) / width)) if hi > lo else 0
                counts[idx] += 1
            maxc = max(counts) or 1
            bars = "".join(self._bar(f"{lo + i * width:.2f}–{lo + (i + 1) * width:.2f}", counts[i], total, maxc)
                           for i in range(bins))
            title = "Predicted values"
        else:
            counts = Counter(str(v) for v in values)
            ordered = counts.most_common(12)
            maxc = max((c for _, c in ordered), default=1)
            bars = "".join(self._bar(lbl, c, total, maxc) for lbl, c in ordered)
            more = len(counts) - len(ordered)
            if more > 0:
                bars += f"<div style='font-size:11px;color:#888;margin-top:4px;'>+{more} more categories</div>"
            title = "What it predicted"

        return (f"<div style='font-size:14px;font-weight:700;margin:0 0 6px;'>{title}</div>"
                f"<div style='font-size:11px;color:#888;margin:0 0 8px;'>{total:,} labelled{extra}</div>" + bars)

    def _bar(self, label, count, total, maxcount) -> str:
        pct = (count / total * 100) if total else 0
        width = (count / maxcount * 100) if maxcount else 0
        safe = _html.escape(str(label))
        return ("<div style='display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;'>"
                f"<div style='width:130px;text-align:right;color:#333;overflow:hidden;text-overflow:ellipsis;"
                f"white-space:nowrap;' title='{safe}'>{safe}</div>"
                "<div style='flex:1;background:#eef1f5;border-radius:4px;overflow:hidden;'>"
                f"<div style='width:{width:.0f}%;background:#3b6fb0;height:16px;'></div></div>"
                f"<div style='width:96px;color:#555;'>{count:,} ({pct:.0f}%)</div></div>")

    def _is_number(self, v) -> bool:
        try:
            float(v)
            return True
        except Exception:
            return False

    def _trust_verdict(self, result: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
        evaluation = result.get("evaluation") or {}
        prov = result.get("provenance") or {}
        rvc = result.get("recipe_version_check") or {}
        has_eval = bool(evaluation.get("metrics_by_provenance"))
        issues = []
        if rvc and not rvc.get("match"):
            issues.append("this model was built with a different version of the method than is installed now")
        if has_eval and not prov.get("verified"):
            issues.append("we couldn't confirm which rows were new to the model, so treat the green figures as a rough guide")
        if issues:
            return ("warning", "Heads up", "Because " + "; and ".join(issues) + ".")
        if has_eval:
            return ("success", "Measured on unseen data",
                    "The green figures above come only from data the model had never seen before.")
        return None

    def _render_summary_plain(self, result, failed_count, added_columns, attach_error) -> str:
        count = int(result.get("count") or 0)
        dataset_id = str(result.get("dataset_id") or self._dataset_id() or "your dataset")
        lines = [f"**Labelled {count:,} rows.**"]
        if failed_count:
            lines.append(f"Skipped **{failed_count}** image(s) that couldn't be opened.")
        if added_columns:
            lines.append(f"Added **{len(added_columns)}** prediction column(s) to **{dataset_id}**.")
            label_col = next((c for c in added_columns if c.endswith("label")), added_columns[0])
            conf_col = next((c for c in added_columns if "confidence" in c), None)
            tip = f"Open a plot and colour points by **{label_col}**"
            if conf_col:
                tip += f" (or **{conf_col}** to spot the unsure ones)"
            lines.append(tip + ".")
        elif attach_error:
            lines.append(f"_Couldn't add prediction columns: {attach_error}._")
        elif not bool(self.attach_columns.value):
            lines.append("_Column-adding is turned off (see More options)._")
        return "  \n".join(lines)

    def _render_technical(self, result: Mapping[str, Any]) -> str:
        prov = result.get("provenance") or {}
        lines: List[str] = [f"**Scope:** `{result.get('scope')}`",
                            "**Provenance verified:** " + ("yes" if prov.get("verified") else "NO")]
        if prov.get("split_spec_artifact_id"):
            lines.append(f"- split record: `{prov['split_spec_artifact_id']}`")
        counts = prov.get("counts") or {}
        if counts:
            lines.append("- rows by origin: " + ", ".join(
                f"{_PROV_META.get(k, (k,))[0]} {v}" for k, v in counts.items()))
        for key, m in ((result.get("evaluation") or {}).get("metrics_by_provenance") or {}).items():
            lines.append(f"- {_PROV_META.get(key, (key,))[0]}: n={m.get('n')}, "
                         f"acc {self._fmt(m.get('accuracy'))}, F1 {self._fmt(m.get('f1_macro'))}")
        rvc = result.get("recipe_version_check")
        if rvc:
            lines.append(f"**Method version:** trained `{rvc.get('trained_version')}`, "
                         f"installed `{rvc.get('registered_version')}` "
                         f"({'match' if rvc.get('match') else 'MISMATCH'})")
        if result.get("checkpoint_sha256"):
            lines.append(f"**Model fingerprint:** `{str(result['checkpoint_sha256'])[:16]}…`")
        return "  \n".join(lines)

    def _fmt(self, value) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return "—"

    def _show_trust(self, alert_type, title, sub) -> None:
        self.trust.alert_type = alert_type
        self.trust.object = f"**{title}.** {sub}"
        self.trust.visible = True

    def _fetch_table(self, artifact_id, result):
        rows = list(result.get("prediction_preview") or [])
        failed_count = int(result.get("failed_image_row_count") or 0)
        failed_rows = list(result.get("failed_image_rows") or [])
        if (not rows or not failed_count) and artifact_id:
            payload = None
            try:
                payload = self.context.artifacts.get(artifact_id)
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                rows = list((payload.get("prediction_table") or {}).get("rows") or []) or rows
                ib = payload.get("input_binding") or {}
                transform = ib.get("transform") or {}
                if not failed_count:
                    failed_count = int(transform.get("failed_rows") or ib.get("failed_image_row_count") or 0)
                if not failed_rows:
                    failed_rows = list(ib.get("failed_image_rows") or [])
        return rows, failed_count, failed_rows

    # ----------------------------------------------------------- plain helpers
    def _plain_block_reason(self, report: Mapping[str, Any]) -> str:
        needs = [str(n).lower() for n in (report.get("needs_mapping") or [])]
        errors = [str(e).lower() for e in (report.get("errors") or [])]
        if any("image" in n for n in needs) or any("image" in e for e in errors):
            return "This data doesn't have an image column the model needs. Set one under “More options”."
        if any("feature" in e or "column" in e for e in errors):
            return "This data is missing some columns the model needs to make predictions."
        if errors:
            return "This model can't run on this data. Open “Technical details” to see why."
        return "This model can't run on this data."

    def _update_model_line(self) -> None:
        d = self._selected_model_descriptor()
        if not d:
            self.model_line.object = "" if self._model_artifact_id() else "_No trained models yet — train one first._"
            return
        out = d.get("output_summary") or {}
        classes = out.get("classes") or []
        task = str(d.get("task") or "")
        if task == "regression":
            predicts = "a number"
        elif classes and len(classes) <= 4:
            predicts = ", ".join(str(c) for c in classes)
        elif classes:
            predicts = f"one of {len(classes)} categories"
        else:
            predicts = "a category"
        kind = "from images" if str(d.get("modality") or "").lower() == "image" else "from your data"
        self.model_line.object = f"Predicts **{predicts}** {kind}."

    # ----------------------------------------------------------- change handlers
    def _on_dataset_change(self) -> None:
        self._load_columns()
        self.validate()

    def _on_model_change(self) -> None:
        self._update_model_line()
        self.validate()

    def _on_accuracy_toggle(self) -> None:
        evaluate = self._is_evaluation_mode()
        self.target.visible = evaluate
        if evaluate and self.target.value in (None, ""):
            self.target.value = self._guess_target(self._columns())
        self.validate()

    def _is_evaluation_mode(self) -> bool:
        return bool(self.check_accuracy.value)

    def _target_column_param(self) -> Optional[str]:
        if not self._is_evaluation_mode():
            return None
        return str(self.target.value) if self.target.value else None

    def _image_column_param(self) -> Optional[str]:
        return str(self.image_column.value) if self.image_column.value else None

    def _row_ids_for_scope(self) -> Optional[List[str]]:
        if str(self.run_on.value) == "Selected points only":
            return self._active_selection_row_ids()
        return None

    # ----------------------------------------------------------- loaders
    def _columns(self) -> List[str]:
        dataset_id = self._dataset_id()
        if not dataset_id:
            return []
        try:
            return [str(c) for c in self.context.datasets.list_columns(dataset_id)]
        except Exception:
            try:
                return [str(c) for c in self.context.datasets.get_source(dataset_id).columns()]
            except Exception:
                return []

    def _load_datasets(self) -> None:
        try:
            options = list(self.context.datasets.list_ids())
        except Exception:
            options = []
        current = self.dataset.value
        self.dataset.options = options
        if current in options:
            self.dataset.value = current
        elif options:
            try:
                active = self.context.datasets.active_id()
            except Exception:
                active = None
            self.dataset.value = active if active in options else options[0]
        else:
            self.dataset.value = None

    def _load_columns(self) -> None:
        columns = self._columns()
        self.target.options = [None] + columns
        self.image_column.options = [None] + columns
        if self.target.value not in self.target.options:
            self.target.value = self._guess_target(columns) if self._is_evaluation_mode() else None
        if self.image_column.value not in self.image_column.options:
            self.image_column.value = self._guess_image_column(columns)

    def _load_models(self) -> None:
        catalog = self._catalog()
        options: Dict[str, str] = {}
        if catalog is not None:
            try:
                catalog.refresh()
                options = catalog.as_options()
            except Exception:
                options = {}
        current = self.model.value
        self.model.options = options
        values = set(options.values())
        if current in values:
            self.model.value = current
        elif options:
            self.model.value = next(iter(options.values()))
        else:
            self.model.value = None

    # ----------------------------------------------------------- events
    def _subscribe_to_events(self) -> None:
        subscribe = getattr(getattr(self.context, "events", None), "subscribe", None)
        if not callable(subscribe):
            return
        for topic in ["ml.run.finished", "ml.recipe_run.finished", "ml.model.saved",
                      "artifact.created", "artifact.updated", "workspace.restored",
                      "dataset.loaded", "dataset.active.changed", "dataset.mapping.updated"]:
            try:
                sub = subscribe(topic, self._on_platform_event, owner_label="ML Predictor", owner_kind="panel")
            except TypeError:
                try:
                    sub = subscribe(topic, self._on_platform_event)
                except Exception:
                    continue
            except Exception:
                continue
            self._subscriptions.append(sub)

    def _on_platform_event(self, *args: Any, **kwargs: Any) -> None:
        if self._disposed:
            return
        topic, payload = self._normalise_event_args(*args, **kwargs)
        select_model_id = None
        if topic in {"ml.run.finished", "ml.recipe_run.finished"}:
            ids = payload.get("artifact_ids") or {}
            select_model_id = (ids.get("model") or ids.get("model_artifact_id")
                               or payload.get("model_artifact_id") or payload.get("artifact_id"))
        elif topic in {"artifact.created", "artifact.updated"}:
            atype = payload.get("type") or payload.get("artifact_type")
            if atype not in {"ml.model", "ml.model_definition", None}:
                return
            select_model_id = payload.get("artifact_id") if atype == "ml.model" else None
        elif topic == "ml.model.saved":
            select_model_id = payload.get("artifact_id") or payload.get("model_artifact_id")

        def update():
            if not self._disposed:
                self.refresh(reason=topic, select_model_artifact_id=select_model_id)

        self._schedule_ui_update(update)

    def _normalise_event_args(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        topic = str(kwargs.get("topic") or kwargs.get("event") or "")
        payload = kwargs.get("payload")
        if len(args) >= 2:
            topic, payload = str(args[0]), args[1]
        elif len(args) == 1:
            if isinstance(args[0], dict):
                payload = args[0]
                topic = str(payload.get("topic") or payload.get("event") or "")
            else:
                topic = str(args[0])
        if not isinstance(payload, dict):
            payload = {}
        return topic, payload

    def _schedule_ui_update(self, callback) -> None:
        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(callback)
                return
        except Exception:
            pass
        callback()

    # ----------------------------------------------------------- selection
    def _active_selection_row_ids(self) -> List[str]:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return []
        for name in ("active_row_ids", "selected_row_ids", "get_selected_row_ids",
                     "current_selection_ids", "get_current_selection"):
            method = getattr(selection, name, None)
            if callable(method):
                try:
                    ids = self._coerce_row_ids(method())
                except Exception:
                    continue
                if ids:
                    return ids
        for name in ("row_ids", "selected_ids", "selection", "current_selection"):
            ids = self._coerce_row_ids(getattr(selection, name, None))
            if ids:
                return ids
        return []

    def _coerce_row_ids(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            for key in ("row_ids", "record_ids", "ids"):
                if key in value:
                    return self._coerce_row_ids(value[key])
            one = value.get("record_id") or value.get("row_id") or value.get("id")
            return [str(one)] if one is not None else []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    # ----------------------------------------------------------- catalog
    def _catalog(self) -> Any:
        get = getattr(getattr(self.context, "services", None), "get", None)
        if not callable(get):
            return None
        try:
            return get("core.ml.trained_model_catalog")
        except Exception:
            return None

    def _selected_model_descriptor(self) -> Dict[str, Any]:
        catalog = self._catalog()
        model_id = self._model_artifact_id()
        if catalog is None or not model_id:
            return {}
        try:
            descriptor = catalog.get(model_id)
        except Exception:
            return {}
        if hasattr(descriptor, "to_dict") and callable(descriptor.to_dict):
            try:
                return dict(descriptor.to_dict())
            except Exception:
                return {}
        return dict(descriptor) if isinstance(descriptor, Mapping) else {}

    # ----------------------------------------------------------- misc
    def _dataset_id(self) -> Optional[str]:
        return str(self.dataset.value) if self.dataset.value else None

    def _model_artifact_id(self) -> Optional[str]:
        return str(self.model.value) if self.model.value else None

    def _model_values(self) -> set:
        if isinstance(self.model.options, dict):
            return set(self.model.options.values())
        return set(self.model.options or [])

    def _set_running(self, running: bool) -> None:
        self.run_button.disabled = bool(running)
        self.cancel_button.disabled = not bool(running)
        self.refresh_button.disabled = bool(running)
        self.run_button.name = "Working…" if running else "Run"

    def _guess_target(self, columns: List[str]) -> Optional[str]:
        dataset_id = self._dataset_id()
        if dataset_id:
            try:
                mapped = self.context.datasets.get_mapping(dataset_id, "target_label")
                if mapped in columns:
                    return mapped
            except Exception:
                pass
        lowered = {c.lower(): c for c in columns}
        for cand in ("target_label", "target", "label", "class", "classification", "y"):
            if cand in lowered:
                return lowered[cand]
        return None

    def _guess_image_column(self, columns: List[str]) -> Optional[str]:
        dataset_id = self._dataset_id()
        if dataset_id:
            for semantic in ("image.path", "image.uri", "image.url"):
                try:
                    mapped = self.context.datasets.get_mapping(dataset_id, semantic)
                    if mapped in columns:
                        return mapped
                except Exception:
                    pass
        lowered = {c.lower(): c for c in columns}
        for cand in ("image", "image_path", "image_uri", "image_url", "img", "path", "file", "filename", "cutout"):
            if cand in lowered:
                return lowered[cand]
        for column in columns:
            if any(t in column.lower() for t in ("image", "img", "path", "uri", "url", "cutout", "jpg", "png")):
                return column
        return None


def create_predict_panel(context: Any, **kwargs: Any):
    controller = MLPredictPanel(context=context, restore_state=kwargs.get("restore_state"))
    return controller.panel(), controller