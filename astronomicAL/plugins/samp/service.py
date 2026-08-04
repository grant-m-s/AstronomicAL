from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pandas.api.types as pdt
from astropy.samp import SAMPIntegratedClient
from astropy.table import Table
from astropy.io import fits
from astropy.io.votable import tree as votable_tree

TableListener = Callable[[dict[str, Any]], None]

SUPPORTED_RECEIVE_MTYPES = ("table.load.votable", "table.load.fits")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return value.strip("-") or "table"


def _safe_suffix_from_url(url: str, *, mtype: str | None = None) -> str:
    parsed = urlparse(str(url))
    suffix = Path(unquote(parsed.path or "")).suffix.lower()
    if suffix in {".vot", ".votable", ".xml", ".fits", ".fit"}:
        return suffix
    if str(mtype or "").lower() == "table.load.fits":
        return ".fits"
    return ".vot"


class SAMPBridge:
    """Long-lived SAMP bridge for AstronomicAL.

    Large-table principle: SAMP callbacks mirror the referenced table bytes locally,
    but do not parse the table into pandas. Parsing/export work is done explicitly by
    panel jobs.
    """

    def __init__(
        self,
        *,
        client_name: str = "AstronomicAL",
        description: str = "AstronomicAL SAMP bridge",
        icon_url: str | None = None,
        on_table_received: TableListener | None = None,
        url_factory: Callable[[Path], str] | None = None,
    ) -> None:
        self.client_name = client_name
        self.description = description
        self.icon_url = icon_url
        self.url_factory = url_factory
        self.client: SAMPIntegratedClient | None = None
        self._started = False
        self.last_error: str | None = None
        self.last_connected_at: float | None = None
        self.last_message_at: float | None = None
        self._tmpdir = tempfile.TemporaryDirectory(prefix="astronomical-samp-")
        self._base = Path(self._tmpdir.name)
        self._listeners: dict[str, TableListener] = {}
        if on_table_received is not None:
            self.add_table_listener(on_table_received)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the running SAMP hub.

        This method raises when no hub is available so explicit send operations can
        report a hard failure. Panel startup/polling should use ``try_start()``.
        """
        if self._started and self.client is not None:
            self.last_error = None
            return

        metadata = {"samp.name": self.client_name, "samp.description.text": self.description}
        if self.icon_url:
            metadata["samp.icon.url"] = self.icon_url

        client = SAMPIntegratedClient(name=self.client_name, description=self.description, metadata=metadata, callable=True)
        try:
            client.connect()
            for mtype in SUPPORTED_RECEIVE_MTYPES:
                client.bind_receive_message(mtype, self._receive_table_message)
        except Exception as exc:
            self.last_error = str(exc)
            try:
                client.disconnect()
            except Exception:
                pass
            self.client = None
            self._started = False
            raise

        self.client = client
        self._started = True
        self.last_error = None
        self.last_connected_at = time.time()

    def try_start(self) -> bool:
        """Best-effort connect used by panels while waiting for a hub."""
        if self.started:
            self.last_error = None
            return True
        try:
            self.start()
            return True
        except Exception:
            return False

    def connection_status(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "last_error": self.last_error,
            "last_connected_at": self.last_connected_at,
            "last_message_at": self.last_message_at,
            "activity_seen": self.activity_seen,
            "base_dir": str(self._base),
        }

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None
                self._started = False

    def close(self) -> None:
        self.stop()
        self._listeners.clear()
        self._tmpdir.cleanup()

    dispose = close

    @property
    def started(self) -> bool:
        if not (self._started and self.client is not None):
            return False
        return True

    @property
    def activity_seen(self) -> bool:
        return bool(self.started or self.last_connected_at is not None or self.last_message_at is not None)

    # ------------------------------------------------------------------
    # Listeners and clients
    # ------------------------------------------------------------------

    def add_table_listener(self, listener: TableListener) -> str:
        token = uuid.uuid4().hex
        self._listeners[token] = listener
        return token

    def remove_table_listener(self, token: str) -> None:
        self._listeners.pop(token, None)

    def clear_table_listeners(self) -> None:
        self._listeners.clear()

    def list_clients(self, *, ensure_started: bool = True, raise_on_error: bool = False) -> list[dict[str, Any]]:
        if ensure_started and not self.started:
            if raise_on_error:
                self.start()
            elif not self.try_start():
                return []
        if self.client is None:
            return []

        rows: list[dict[str, Any]] = []
        for client_id in self.client.get_registered_clients():
            if client_id == "hub":
                continue
            meta = self.client.get_metadata(client_id) or {}
            rows.append({"id": client_id, "name": meta.get("samp.name", client_id), "metadata": meta})
        return rows

    def find_topcat_clients(self) -> list[dict[str, Any]]:
        return [row for row in self.list_clients() if str(row["name"]).lower() == "topcat"]

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_dataframe(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        target_mode: str = "topcat",
        target_client_id: str | None = None,
        export_format: str = "fits",
    ) -> dict[str, Any]:
        """Export a dataframe to a temporary table file and send it through SAMP.

        ``export_format``:
          - ``fits``: fastest common path for TOPCAT/DS9; sends ``table.load.fits``.
          - ``votable_binary``: compact VOTable; sends ``table.load.votable``.
          - ``votable_tabledata``: XML TABLEDATA compatibility; slowest path.
        """
        self.start()
        assert self.client is not None

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        export_df, coercions = self._make_table_safe(df)
        timings["coerce_seconds"] = round(time.perf_counter() - t0, 3)

        if str(export_format or "").strip().lower() in {"original", "original_file", "same"}:
            export_format = "fits"

        t1 = time.perf_counter()
        out_path, mtype, resolved_format = self._write_table(export_df, table_name=table_name, export_format=export_format)
        timings["write_seconds"] = round(time.perf_counter() - t1, 3)
        url = self._path_to_url(out_path)
        params = {"url": url, "name": table_name.strip() or "AstronomicAL table"}

        t2 = time.perf_counter()
        target_ids = self._resolve_target_ids(target_mode=target_mode, target_client_id=target_client_id)
        if target_ids is None:
            self.client.enotify_all(mtype, **params)
            delivered_to: str | list[str] = "all"
        else:
            for cid in target_ids:
                self.client.enotify(cid, mtype, **params)
            delivered_to = target_ids
        timings["notify_seconds"] = round(time.perf_counter() - t2, 3)
        timings["total_seconds"] = round(time.perf_counter() - t0, 3)

        return {
            "table_name": params["name"],
            "path": str(out_path),
            "url": url,
            "mtype": mtype,
            "export_format": resolved_format,
            "row_count": int(len(export_df)),
            "column_count": int(len(export_df.columns)),
            "delivered_to": delivered_to,
            "coercions": coercions,
            "columns": [str(col) for col in export_df.columns],
            "timings": timings,
        }


    def send_table_file(
        self,
        path: str | Path,
        *,
        mtype: str,
        table_name: str,
        target_mode: str = "topcat",
        target_client_id: str | None = None,
        row_count: int | None = None,
        columns: list[str] | None = None,
        export_format: str = "original_file",
    ) -> dict[str, Any]:
        """Send an existing local SAMP-loadable table file without reserialising it."""
        self.start()
        assert self.client is not None
        t0 = time.perf_counter()
        table_path = Path(path).expanduser().resolve()
        if not table_path.exists():
            raise FileNotFoundError(f"SAMP table file does not exist: {table_path}")
        url = self._path_to_url(table_path)
        params = {"url": url, "name": table_name.strip() or table_path.stem or "AstronomicAL table"}
        target_ids = self._resolve_target_ids(target_mode=target_mode, target_client_id=target_client_id)
        t_notify = time.perf_counter()
        if target_ids is None:
            self.client.enotify_all(mtype, **params)
            delivered_to: str | list[str] = "all"
        else:
            for cid in target_ids:
                self.client.enotify(cid, mtype, **params)
            delivered_to = target_ids
        timings = {
            "coerce_seconds": 0.0,
            "write_seconds": 0.0,
            "notify_seconds": round(time.perf_counter() - t_notify, 3),
            "total_seconds": round(time.perf_counter() - t0, 3),
        }
        return {
            "table_name": params["name"],
            "path": str(table_path),
            "url": url,
            "mtype": mtype,
            "export_format": export_format,
            "row_count": int(row_count) if row_count is not None else 0,
            "column_count": len(columns or []),
            "delivered_to": delivered_to,
            "coercions": [],
            "columns": list(columns or []),
            "timings": timings,
            "direct_file": True,
        }

    def _write_table(self, export_df: pd.DataFrame, *, table_name: str, export_format: str) -> tuple[Path, str, str]:
        fmt = str(export_format or "fits").strip().lower()
        table = Table.from_pandas(export_df.reset_index(drop=True))

        if fmt == "fits":
            out_path = self._base / f"{_slugify(table_name)}-{uuid.uuid4().hex}.fits"
            table.write(out_path, format="fits", overwrite=True)
            return out_path, "table.load.fits", "fits"

        if fmt in {"votable_binary", "binary", "votable"}:
            out_path = self._base / f"{_slugify(table_name)}-{uuid.uuid4().hex}.vot"
            votable = votable_tree.VOTableFile.from_table(table)
            votable.to_xml(str(out_path), tabledata_format="binary")
            return out_path, "table.load.votable", "votable_binary"

        if fmt in {"votable_tabledata", "tabledata", "xml"}:
            out_path = self._base / f"{_slugify(table_name)}-{uuid.uuid4().hex}.vot"
            votable = votable_tree.VOTableFile.from_table(table)
            votable.to_xml(str(out_path), tabledata_format="tabledata")
            return out_path, "table.load.votable", "votable_tabledata"

        raise RuntimeError(f"Unknown SAMP export format: {export_format!r}")

    def _path_to_url(self, path: Path) -> str:
        if self.url_factory is not None:
            return self.url_factory(path)
        return path.resolve().as_uri()

    def _resolve_target_ids(self, *, target_mode: str, target_client_id: str | None) -> list[str] | None:
        assert self.client is not None
        registered = [cid for cid in self.client.get_registered_clients() if cid != "hub"]

        if target_mode == "all":
            return None
        if target_mode == "client":
            if not target_client_id:
                raise RuntimeError("No target SAMP client selected.")
            if target_client_id not in registered:
                raise RuntimeError(f"Selected SAMP client {target_client_id!r} is not connected.")
            return [target_client_id]
        if target_mode == "topcat":
            topcat_ids: list[str] = []
            for cid in registered:
                meta = self.client.get_metadata(cid) or {}
                if str(meta.get("samp.name", "")).lower() == "topcat":
                    topcat_ids.append(cid)
            if not topcat_ids:
                raise RuntimeError("No TOPCAT client found on the SAMP hub.")
            return topcat_ids
        raise RuntimeError(f"Unknown target_mode: {target_mode}")

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def _receive_table_message(
        self,
        private_key: str,
        sender_id: str,
        msg_id: str | None,
        mtype: str,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> None:
        try:
            payload = self._build_incoming_payload(sender_id=sender_id, msg_id=msg_id, mtype=mtype, params=params, extra=extra)
            self._notify_table_listeners(payload)
            if msg_id is not None:
                assert self.client is not None
                self.client.reply(msg_id, {"samp.status": "samp.ok", "samp.result": {}})
        except Exception as exc:
            if msg_id is not None and self.client is not None:
                self.client.reply(msg_id, {"samp.status": "samp.error", "samp.error": {"samp.errortxt": str(exc)}})
            raise

    def _build_incoming_payload(
        self,
        *,
        sender_id: str,
        msg_id: str | None,
        mtype: str,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        url = params.get("url")
        if not url:
            raise RuntimeError("Incoming SAMP table message did not include a 'url' parameter.")

        self.last_error = None
        self.last_message_at = time.time()
        if self.client is not None:
            self._started = True
        local_path = self._mirror_table_url(str(url), name=params.get("name") or "incoming-samp-table", mtype=mtype)
        return {
            "sender_id": sender_id,
            "msg_id": msg_id,
            "is_call": msg_id is not None,
            "mtype": mtype,
            "params": dict(params or {}),
            "extra": dict(extra or {}),
            "name": params.get("name") or "Incoming SAMP table",
            "url": str(url),
            "local_path": str(local_path),
            "local_url": local_path.resolve().as_uri(),
            "row_count": None,
            "column_count": None,
            "columns": [],
        }

    def _mirror_table_url(self, url: str, *, name: str, mtype: str | None = None) -> Path:
        suffix = _safe_suffix_from_url(url, mtype=mtype)
        out_path = self._base / f"incoming-{_slugify(name)}-{uuid.uuid4().hex}{suffix}"
        parsed = urlparse(str(url))

        if parsed.scheme in {"", "file"}:
            raw_path = unquote(parsed.path if parsed.scheme == "file" else url)
            if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
                raw_path = raw_path[1:]
            src_path = Path(raw_path).expanduser()
            shutil.copyfile(src_path, out_path)
            return out_path

        with urlopen(url) as response, out_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        return out_path

    def inspect_table(self, url_or_path: str, *, preview_rows: int = 20, preview_columns: list[str] | None = None) -> dict[str, Any]:
        path = self._local_path_from_urlish(url_or_path)
        if path is not None and path.suffix.lower() in {".fits", ".fit", ".fts"}:
            return self._inspect_fits_table(path, preview_rows=preview_rows, preview_columns=preview_columns)

        table = Table.read(url_or_path)
        columns = [str(col) for col in table.colnames]
        selected_columns = [col for col in (preview_columns or []) if col in columns]
        preview_table = table[selected_columns] if selected_columns else table
        preview_df = preview_table[: max(1, int(preview_rows or 20))].to_pandas()
        return {"row_count": int(len(table)), "column_count": int(len(columns)), "columns": columns, "preview": preview_df}

    def read_table_dataframe(self, url_or_path: str) -> pd.DataFrame:
        return Table.read(url_or_path).to_pandas()

    def convert_table_to_parquet(
        self,
        url_or_path: str,
        parquet_path: str | Path,
        *,
        chunk_rows: int = 65536,
    ) -> dict[str, Any]:
        """Convert a SAMP table reference to Parquet.

        FITS binary tables use a chunked writer so import does not require a
        full-table pandas DataFrame. VOTable/XML falls back to Astropy's full
        parser because robust streaming VOTable support is format dependent.
        """
        path = self._local_path_from_urlish(url_or_path)
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if path is not None and path.suffix.lower() in {".fits", ".fit", ".fts"}:
            try:
                return self._fits_to_parquet(path, parquet_path, chunk_rows=chunk_rows)
            except Exception:
                try:
                    if parquet_path.exists():
                        parquet_path.unlink()
                except Exception:
                    pass
                # Fall through to the generic Astropy path as a correctness fallback.
        df = self.read_table_dataframe(str(path or url_or_path))
        df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="zstd")
        return {
            "parquet_path": str(parquet_path),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": [str(col) for col in df.columns],
            "conversion": "astropy_full_dataframe",
        }

    def _local_path_from_urlish(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value)
        parsed = urlparse(text)
        if parsed.scheme == "file":
            raw_path = unquote(parsed.path)
            if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
                raw_path = raw_path[1:]
            return Path(raw_path).expanduser()
        if parsed.scheme in {"http", "https"}:
            return None
        return Path(text).expanduser()

    def _first_fits_table_hdu(self, hdul):
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and getattr(hdu, "columns", None) is not None:
                names = list(getattr(hdu.columns, "names", []) or [])
                if names:
                    return hdu
        raise RuntimeError("No table HDU found in FITS file.")

    def _coerce_fits_values(self, values):
        arr = np.asarray(values)
        if arr.dtype.kind == "S":
            return np.char.decode(arr, "utf-8", errors="ignore")
        if arr.dtype.kind == "O":
            return [None if value is None else str(value) for value in arr]
        if arr.ndim > 1:
            return [item.tolist() for item in arr]
        return arr

    def _inspect_fits_table(self, path: Path, *, preview_rows: int, preview_columns: list[str] | None = None) -> dict[str, Any]:
        with fits.open(path, memmap=True) as hdul:
            hdu = self._first_fits_table_hdu(hdul)
            data = hdu.data
            columns = [str(col) for col in hdu.columns.names]
            n_rows = int(len(data))
            selected = [col for col in (preview_columns or []) if col in columns] or columns
            stop = min(max(1, int(preview_rows or 20)), n_rows)
            frame_data = {col: self._coerce_fits_values(data[col][:stop]) for col in selected}
            preview_df = pd.DataFrame(frame_data)
        return {"row_count": n_rows, "column_count": len(columns), "columns": columns, "preview": preview_df}

    def _fits_to_parquet(self, fits_path: Path, parquet_path: Path, *, chunk_rows: int = 65536) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        rows_written = 0
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                hdu = self._first_fits_table_hdu(hdul)
                data = hdu.data
                columns = [str(col) for col in hdu.columns.names]
                row_count = int(len(data))
                chunk_rows = max(1, int(chunk_rows or 65536))
                for start in range(0, row_count, chunk_rows):
                    stop = min(start + chunk_rows, row_count)
                    chunk = {col: self._coerce_fits_values(data[col][start:stop]) for col in columns}
                    chunk_df = pd.DataFrame(chunk)
                    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows_written += len(chunk_df)
        finally:
            if writer is not None:
                writer.close()
        return {
            "parquet_path": str(parquet_path),
            "row_count": int(rows_written),
            "column_count": len(columns),
            "columns": columns,
            "conversion": "fits_chunked_pyarrow",
        }

    def _notify_table_listeners(self, payload: dict[str, Any]) -> None:
        errors: list[str] = []
        for token, listener in list(self._listeners.items()):
            try:
                listener(payload)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{token}: {exc}")
        if errors:
            raise RuntimeError("One or more SAMP table listeners failed: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # DataFrame -> table compatibility
    # ------------------------------------------------------------------

    def _make_table_safe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        out = df.copy(deep=False)
        coercions: list[dict[str, Any]] = []
        numeric_object_types = {
            int,
            float,
            bool,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.bool_,
        }

        for col in out.columns:
            s = out[col]
            dt = s.dtype
            if dt == np.int8:
                out[col] = s.astype(np.int16)
                coercions.append({"column": col, "from": "int8", "to": "int16"})
            elif dt == np.uint16:
                out[col] = s.astype(np.int32)
                coercions.append({"column": col, "from": "uint16", "to": "int32"})
            elif dt == np.uint32:
                out[col] = s.astype(np.int64)
                coercions.append({"column": col, "from": "uint32", "to": "int64"})
            elif dt == np.uint64:
                mx = s.max(skipna=True)
                if pd.isna(mx) or mx <= np.iinfo(np.int64).max:
                    out[col] = s.astype(np.int64)
                    coercions.append({"column": col, "from": "uint64", "to": "int64"})
                else:
                    raise ValueError(f"Column {col!r} is uint64 with values above int64 range; cannot safely represent it in FITS/VOTable.")
            elif isinstance(dt, pd.CategoricalDtype):
                out[col] = s.astype("string")
                coercions.append({"column": col, "from": "category", "to": "string"})
            elif pdt.is_object_dtype(dt):
                non_null = s.dropna()
                if non_null.empty:
                    continue
                sample_types = {type(v) for v in non_null.iloc[:100]}
                if sample_types <= {bytes, bytearray, np.bytes_}:
                    out[col] = s.map(lambda v: None if pd.isna(v) else bytes(v).decode("utf-8", errors="ignore")).astype("string")
                    coercions.append({"column": col, "from": "object(bytes)", "to": "string"})
                elif sample_types <= {str}:
                    out[col] = s.astype("string")
                    coercions.append({"column": col, "from": "object(str)", "to": "string"})
                elif sample_types <= numeric_object_types:
                    out[col] = pd.to_numeric(s, errors="raise")
                    coercions.append({"column": col, "from": "object(numeric)", "to": str(out[col].dtype)})
                else:
                    raise ValueError(f"Column {col!r} has mixed object values {sample_types}; cannot safely export without explicit conversion.")
        return out, coercions


# Compatibility alias for older code/tests.
SAMPBridge._make_votable_safe = SAMPBridge._make_table_safe


def create_samp_bridge(context=None, **kwargs) -> SAMPBridge:
    return SAMPBridge()
