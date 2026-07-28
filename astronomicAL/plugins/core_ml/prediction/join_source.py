from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from astronomicAL.platform.dataset_sources import (
    DatasetBatch,
    DatasetCapabilities,
    DatasetScan,
    DatasetSource,
)


class PredictionColumnSource(DatasetSource):
    """Left-join the latest prediction columns onto an existing dataset source.

    The source keeps the original dataset row count and order. Prediction rows
    are looked up in bounded batches by record ID, so replacing a dataset with
    this wrapper does not materialise either side of the join.
    """

    backend_name = "prediction_join"

    def __init__(
        self,
        *,
        base_source: DatasetSource,
        prediction_source: DatasetSource,
        base_record_id_column: Optional[str],
        prediction_record_id_column: str = "record_id",
        prediction_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.base_source = unwrap_prediction_source(base_source)
        self.prediction_source = prediction_source
        self.base_record_id_column = _normalise_id_column(base_record_id_column)
        self.prediction_record_id_column = str(prediction_record_id_column)

        base_columns = [str(column) for column in self.base_source.columns()]
        available_prediction_columns = [
            str(column) for column in self.prediction_source.columns()
        ]
        if self.prediction_record_id_column not in available_prediction_columns:
            raise KeyError(
                "Prediction source is missing record ID column "
                f"{self.prediction_record_id_column!r}."
            )
        selected = (
            [str(column) for column in prediction_columns]
            if prediction_columns is not None
            else available_prediction_columns
        )
        missing = [
            column
            for column in selected
            if column not in available_prediction_columns
        ]
        if missing:
            raise KeyError(f"Unknown prediction columns: {missing!r}")
        self._prediction_columns = [
            column
            for column in dict.fromkeys(selected)
            if column != self.prediction_record_id_column
        ]
        self._base_columns = base_columns
        self._all_columns = list(base_columns)
        for column in self._prediction_columns:
            if column not in self._all_columns:
                self._all_columns.append(column)

        if self.base_record_id_column is not None:
            if self.base_record_id_column not in base_columns:
                raise KeyError(
                    "Base dataset is missing mapped record ID column "
                    f"{self.base_record_id_column!r}."
                )

    def columns(self) -> list[str]:
        return list(self._all_columns)

    def row_count(self) -> Optional[int]:
        return self.base_source.row_count()

    def dtypes(self) -> dict[str, str]:
        dtypes = dict(self.base_source.dtypes())
        prediction_dtypes = dict(self.prediction_source.dtypes())
        for column in self._prediction_columns:
            dtypes[column] = prediction_dtypes.get(column, "object")
        return dtypes

    def capabilities(self) -> DatasetCapabilities:
        base = self.base_source.capabilities()
        return DatasetCapabilities(
            batch_scan=bool(base.batch_scan),
            batch_lookup_by_id=bool(base.batch_lookup_by_id),
            filtered_scan=bool(base.filtered_scan),
            ordered_scan=bool(base.ordered_scan),
        )

    def iter_batches(self, scan: DatasetScan) -> Iterator[DatasetBatch]:
        requested = self._validate_columns(scan.columns)
        base_columns = self._base_columns_for(requested)
        base_scan = DatasetScan(
            columns=tuple(base_columns),
            batch_size=int(scan.batch_size),
            where_sql=scan.where_sql,
            params=tuple(scan.params or ()),
            limit=scan.limit,
        )
        iterator = getattr(self.base_source, "iter_batches", None)
        if not callable(iterator):
            yield from self._iter_positional_batches(scan, requested)
            return
        try:
            for batch in iterator(base_scan):
                yield DatasetBatch(
                    frame=self._join_frame(
                        batch.frame,
                        row_offset=batch.row_offset,
                        requested=requested,
                    ),
                    batch_index=batch.batch_index,
                    row_offset=batch.row_offset,
                )
        except NotImplementedError:
            yield from self._iter_positional_batches(scan, requested)

    def to_pandas(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        batches = [
            batch.frame
            for batch in self.iter_batches(
                DatasetScan(
                    columns=tuple(requested) if requested is not None else None,
                    batch_size=8192,
                    where_sql=where_sql,
                    params=tuple(params or ()),
                    limit=limit,
                )
            )
        ]
        if not batches:
            return pd.DataFrame(columns=requested or self.columns())
        return pd.concat(batches, ignore_index=True)

    def head(
        self,
        n: int = 5,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.to_pandas(columns=columns, limit=max(0, int(n)))

    def get_row_by_position(
        self,
        position: int,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        if position < 0:
            return pd.DataFrame(columns=requested or self.columns())
        frame = self.base_source.get_row_by_position(
            int(position),
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(
            frame,
            row_offset=int(position),
            requested=requested,
        )

    def get_row_by_id(
        self,
        row_id: Any,
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frame = self.base_source.get_row_by_id(
            row_id,
            id_column=id_column,
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(frame, row_offset=0, requested=requested)

    def get_rows_by_ids(
        self,
        row_ids: Sequence[Any],
        *,
        id_column: str,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        requested = self._validate_columns(columns)
        frame = self.base_source.get_rows_by_ids(
            row_ids,
            id_column=id_column,
            columns=self._base_columns_for(requested),
        )
        return self._join_frame(frame, row_offset=0, requested=requested)

    def find_position_by_id(self, row_id: Any, *, id_column: str) -> Optional[int]:
        return self.base_source.find_position_by_id(row_id, id_column=id_column)

    def count_where(
        self,
        *,
        where_sql: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[int]:
        if where_sql and any(column in where_sql for column in self._prediction_columns):
            raise NotImplementedError(
                "Filtering a prediction-joined dataset by prediction columns is "
                "not yet supported by the generic join source."
            )
        return self.base_source.count_where(where_sql=where_sql, params=params)

    def sample_points(
        self,
        *,
        x_col: str,
        y_col: str,
        record_id_col: Optional[str] = None,
        limit: int = 10_000,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        if x_col not in self._all_columns or y_col not in self._all_columns:
            raise KeyError(f"Unknown sample columns: {x_col!r}, {y_col!r}")
        limit = max(1, int(limit))
        selected_columns = [x_col, y_col]
        if record_id_col and record_id_col != "Use Index":
            selected_columns.append(str(record_id_col))
        selected_columns = list(dict.fromkeys(selected_columns))

        rng = np.random.default_rng(int(seed))
        reservoir: list[tuple[float, float, Any]] = []
        eligible = 0
        for batch in self.iter_batches(
            DatasetScan(columns=tuple(selected_columns), batch_size=8192)
        ):
            frame = batch.frame
            x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
            if record_id_col and record_id_col != "Use Index" and record_id_col in frame:
                row_ids = frame[record_id_col].tolist()
            else:
                row_ids = list(range(batch.row_offset, batch.row_offset + len(frame)))
            mask = np.isfinite(x) & np.isfinite(y)
            if log_x:
                mask &= x > 0
            if log_y:
                mask &= y > 0
            if x_range is not None:
                lo, hi = sorted((float(x_range[0]), float(x_range[1])))
                mask &= (x >= lo) & (x <= hi)
            if y_range is not None:
                lo, hi = sorted((float(y_range[0]), float(y_range[1])))
                mask &= (y >= lo) & (y <= hi)
            for index in np.flatnonzero(mask):
                eligible += 1
                item = (float(x[index]), float(y[index]), row_ids[index])
                if len(reservoir) < limit:
                    reservoir.append(item)
                else:
                    replacement = int(rng.integers(0, eligible))
                    if replacement < limit:
                        reservoir[replacement] = item
        output = pd.DataFrame(reservoir, columns=["__x", "__y", "__row_id"])
        return {
            "frame": output,
            "row_count": int(eligible),
            "sampled_from": int(eligible),
            "backend": self.backend_name,
        }

    def aggregate_2d(
        self,
        *,
        x_col: str,
        y_col: str,
        bins: int,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        if x_col not in self._all_columns or y_col not in self._all_columns:
            raise KeyError(f"Unknown aggregate columns: {x_col!r}, {y_col!r}")
        bins = max(5, min(500, int(bins)))
        raw_x_range, raw_y_range, row_count = self._aggregate_ranges(
            x_col=x_col,
            y_col=y_col,
            x_range=x_range,
            y_range=y_range,
            log_x=log_x,
            log_y=log_y,
        )
        raw_x0, raw_x1 = raw_x_range
        raw_y0, raw_y1 = raw_y_range
        plot_x0, plot_x1 = (
            (float(np.log10(raw_x0)), float(np.log10(raw_x1)))
            if log_x
            else raw_x_range
        )
        plot_y0, plot_y1 = (
            (float(np.log10(raw_y0)), float(np.log10(raw_y1)))
            if log_y
            else raw_y_range
        )
        counts = np.zeros((bins, bins), dtype="float64")
        for batch in self.iter_batches(
            DatasetScan(columns=(x_col, y_col), batch_size=8192)
        ):
            x, y = _finite_xy(
                batch.frame,
                x_col=x_col,
                y_col=y_col,
                x_range=raw_x_range,
                y_range=raw_y_range,
                log_x=log_x,
                log_y=log_y,
            )
            if len(x) == 0:
                continue
            if log_x:
                x = np.log10(x)
            if log_y:
                y = np.log10(y)
            batch_counts, _, _ = np.histogram2d(
                y,
                x,
                bins=[bins, bins],
                range=[[plot_y0, plot_y1], [plot_x0, plot_x1]],
            )
            counts += batch_counts
        return {
            "counts": counts,
            "x_edges": np.linspace(plot_x0, plot_x1, bins + 1),
            "y_edges": np.linspace(plot_y0, plot_y1, bins + 1),
            "raw_x_range": raw_x_range,
            "raw_y_range": raw_y_range,
            "plot_x_range": (plot_x0, plot_x1),
            "plot_y_range": (plot_y0, plot_y1),
            "row_count": int(row_count),
            "backend": self.backend_name,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_backend": getattr(self.base_source, "backend_name", "unknown"),
            "prediction_backend": getattr(
                self.prediction_source, "backend_name", "unknown"
            ),
            "prediction_columns": list(self._prediction_columns),
            "capabilities": self.capabilities().to_dict(),
        }

    def _validate_columns(
        self, columns: Optional[Sequence[str]]
    ) -> Optional[list[str]]:
        if columns is None:
            return None
        requested = [str(column) for column in columns]
        missing = [column for column in requested if column not in self._all_columns]
        if missing:
            raise KeyError(f"Unknown dataset columns: {missing!r}")
        return requested

    def _base_columns_for(self, requested: Optional[Sequence[str]]) -> list[str]:
        if requested is None:
            selected = list(self._base_columns)
        else:
            selected = [
                column
                for column in requested
                if column in self._base_columns
                and column not in self._prediction_columns
            ]
        if self.base_record_id_column is not None:
            selected.append(self.base_record_id_column)
        return list(dict.fromkeys(selected))

    def _join_frame(
        self,
        frame: pd.DataFrame,
        *,
        row_offset: int,
        requested: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        output_columns = list(requested) if requested is not None else self.columns()
        if frame is None or frame.empty:
            return pd.DataFrame(columns=output_columns)
        base = frame.copy()
        ids = self._record_ids(base, row_offset=row_offset)
        wanted_prediction_columns = [
            column for column in output_columns if column in self._prediction_columns
        ]
        if wanted_prediction_columns:
            prediction_frame = self.prediction_source.get_rows_by_ids(
                [str(value) for value in ids],
                id_column=self.prediction_record_id_column,
                columns=[self.prediction_record_id_column, *wanted_prediction_columns],
            )
            if prediction_frame is not None and not prediction_frame.empty:
                keys = prediction_frame[self.prediction_record_id_column].map(str)
                if keys.duplicated().any():
                    prediction_frame = prediction_frame.loc[
                        ~keys.duplicated(keep="last")
                    ].copy()
                    keys = prediction_frame[self.prediction_record_id_column].map(str)
                prediction_frame.index = keys
                for column in wanted_prediction_columns:
                    base[column] = [
                        prediction_frame.at[str(row_id), column]
                        if str(row_id) in prediction_frame.index
                        else None
                        for row_id in ids
                    ]
            else:
                for column in wanted_prediction_columns:
                    base[column] = None
        for column in output_columns:
            if column not in base.columns:
                base[column] = None
        return base.loc[:, output_columns].reset_index(drop=True)

    def _record_ids(self, frame: pd.DataFrame, *, row_offset: int) -> list[Any]:
        if self.base_record_id_column is not None:
            return frame[self.base_record_id_column].tolist()
        return list(range(row_offset, row_offset + len(frame)))

    def _iter_positional_batches(
        self,
        scan: DatasetScan,
        requested: Optional[Sequence[str]],
    ) -> Iterator[DatasetBatch]:
        if scan.where_sql is not None or scan.params:
            raise NotImplementedError(
                "Legacy positional prediction joins do not support filtered scans."
            )
        total = self.row_count()
        if total is None:
            raise NotImplementedError(
                "The wrapped source needs row_count() or iter_batches()."
            )
        stop = int(total)
        if scan.limit is not None:
            stop = min(stop, int(scan.limit))
        batch_size = int(scan.batch_size)
        for batch_index, start in enumerate(range(0, stop, batch_size)):
            frames = [
                self.base_source.get_row_by_position(
                    position,
                    columns=self._base_columns_for(requested),
                )
                for position in range(start, min(start + batch_size, stop))
            ]
            non_empty = [frame for frame in frames if frame is not None and not frame.empty]
            base = (
                pd.concat(non_empty, ignore_index=True)
                if non_empty
                else pd.DataFrame(columns=self._base_columns_for(requested))
            )
            yield DatasetBatch(
                frame=self._join_frame(base, row_offset=start, requested=requested),
                batch_index=batch_index,
                row_offset=start,
            )

    def _aggregate_ranges(
        self,
        *,
        x_col: str,
        y_col: str,
        x_range: Optional[tuple[float, float]],
        y_range: Optional[tuple[float, float]],
        log_x: bool,
        log_y: bool,
    ) -> tuple[tuple[float, float], tuple[float, float], int]:
        x_min = x_max = y_min = y_max = None
        row_count = 0
        requested_x = (
            tuple(sorted((float(x_range[0]), float(x_range[1]))))
            if x_range is not None
            else None
        )
        requested_y = (
            tuple(sorted((float(y_range[0]), float(y_range[1]))))
            if y_range is not None
            else None
        )
        for batch in self.iter_batches(
            DatasetScan(columns=(x_col, y_col), batch_size=8192)
        ):
            x, y = _finite_xy(
                batch.frame,
                x_col=x_col,
                y_col=y_col,
                x_range=requested_x,
                y_range=requested_y,
                log_x=log_x,
                log_y=log_y,
            )
            if len(x) == 0:
                continue
            row_count += len(x)
            x_min = float(np.min(x)) if x_min is None else min(x_min, float(np.min(x)))
            x_max = float(np.max(x)) if x_max is None else max(x_max, float(np.max(x)))
            y_min = float(np.min(y)) if y_min is None else min(y_min, float(np.min(y)))
            y_max = float(np.max(y)) if y_max is None else max(y_max, float(np.max(y)))
        if row_count == 0 or x_min is None or y_min is None:
            raise ValueError("No finite X/Y rows available for 2D aggregation")
        resolved_x = requested_x or (x_min, x_max)
        resolved_y = requested_y or (y_min, y_max)
        if resolved_x[1] <= resolved_x[0] or resolved_y[1] <= resolved_y[0]:
            raise ValueError("Invalid aggregate range")
        return resolved_x, resolved_y, row_count


def unwrap_prediction_source(source: DatasetSource) -> DatasetSource:
    while isinstance(source, PredictionColumnSource):
        source = source.base_source
    return source


def _normalise_id_column(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"use index", "use_index", "__index__", "index"}:
        return None
    return text


def _finite_xy(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_range: Optional[tuple[float, float]],
    y_range: Optional[tuple[float, float]],
    log_x: bool,
    log_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if log_x:
        mask &= x > 0
    if log_y:
        mask &= y > 0
    if x_range is not None:
        mask &= (x >= x_range[0]) & (x <= x_range[1])
    if y_range is not None:
        mask &= (y >= y_range[0]) & (y <= y_range[1])
    return x[mask], y[mask]
