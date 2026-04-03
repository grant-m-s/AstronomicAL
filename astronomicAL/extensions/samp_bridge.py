from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas.api.types as pdt
from astropy.samp import SAMPIntegratedClient
from astropy.table import Table


TableListener = Callable[[dict[str, Any]], None]


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return value.strip("-") or "table"


class SAMPBridge:
    """
    Persistent SAMP bridge for AstronomicAL.

    Responsibilities
    ----------------
    - stay connected to the SAMP hub so other tools see AstronomicAL as a client
    - send pandas DataFrames as VOTables
    - receive incoming VOTables from other SAMP clients
    - notify registered listeners when a table is received

    Notes
    -----
    - This class is designed as a long-lived runtime service.
    - By default it sends local file:// URLs. If you need remote/browser-safe URLs,
      pass a url_factory that turns a local Path into an externally reachable URL.
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

        self._tmpdir = tempfile.TemporaryDirectory(prefix="astronomical-samp-")
        self._base = Path(self._tmpdir.name)

        self._listeners: dict[str, TableListener] = {}
        if on_table_received is not None:
            self.add_table_listener(on_table_received)

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the SAMP hub and register receive handlers."""
        if self._started and self.client is not None:
            return

        metadata = {
            "samp.name": self.client_name,
            "samp.description.text": self.description,
        }
        if self.icon_url:
            metadata["samp.icon.url"] = self.icon_url

        self.client = SAMPIntegratedClient(
            name=self.client_name,
            description=self.description,
            metadata=metadata,
            callable=True,
        )
        self.client.connect()

        # One handler for both notifications and calls. For notifications,
        # Astropy passes msg_id=None.
        self.client.bind_receive_message(
            "table.load.votable",
            self._receive_table_message,
        )

        self._started = True

    def stop(self) -> None:
        """Disconnect from the SAMP hub."""
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None
                self._started = False

    def close(self) -> None:
        """Full teardown for service shutdown."""
        self.stop()
        self._listeners.clear()
        self._tmpdir.cleanup()

    @property
    def started(self) -> bool:
        return self._started and self.client is not None

    # -------------------------------------------------------------------------
    # listener registration
    # -------------------------------------------------------------------------

    def add_table_listener(self, listener: TableListener) -> str:
        """
        Register a callback for incoming tables.

        Returns a token that can be passed to remove_table_listener().
        """
        token = uuid.uuid4().hex
        self._listeners[token] = listener
        return token

    def remove_table_listener(self, token: str) -> None:
        self._listeners.pop(token, None)

    def clear_table_listeners(self) -> None:
        self._listeners.clear()

    # -------------------------------------------------------------------------
    # client discovery
    # -------------------------------------------------------------------------

    def list_clients(self, *, ensure_started: bool = True) -> list[dict[str, Any]]:
        """
        Return currently registered SAMP clients.

        Each entry includes:
        - id
        - name
        - metadata
        """
        if ensure_started:
            self.start()

        if self.client is None:
            return []

        rows: list[dict[str, Any]] = []
        for client_id in self.client.get_registered_clients():
            if client_id == "hub":
                continue

            meta = self.client.get_metadata(client_id) or {}
            rows.append(
                {
                    "id": client_id,
                    "name": meta.get("samp.name", client_id),
                    "metadata": meta,
                }
            )
        return rows

    def find_topcat_clients(self) -> list[dict[str, Any]]:
        """Return connected clients whose samp.name is topcat."""
        out = []
        for row in self.list_clients():
            if str(row["name"]).lower() == "topcat":
                out.append(row)
        return out

    # -------------------------------------------------------------------------
    # outgoing table send
    # -------------------------------------------------------------------------

    def send_dataframe(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        target_mode: str = "topcat",
        target_client_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Export a dataframe to a temporary VOTable and send it via SAMP.

        target_mode:
          - "topcat" : send to all TOPCAT clients
          - "all"    : broadcast to all listening clients
          - "client" : send to one specific client id
        """
        self.start()
        assert self.client is not None

        export_df, coercions = self._make_votable_safe(df)
        out_path = self._write_votable(export_df, table_name=table_name)
        url = self._path_to_url(out_path)
        params = {
            "url": url,
            "name": table_name.strip() or "AstronomicAL table",
        }

        target_ids = self._resolve_target_ids(
            target_mode=target_mode,
            target_client_id=target_client_id,
        )

        if target_ids is None:
            self.client.enotify_all("table.load.votable", **params)
            delivered_to: str | list[str] = "all"
        else:
            for cid in target_ids:
                self.client.enotify(cid, "table.load.votable", **params)
            delivered_to = target_ids

        return {
            "table_name": params["name"],
            "path": str(out_path),
            "url": url,
            "row_count": len(export_df),
            "column_count": len(export_df.columns),
            "delivered_to": delivered_to,
            "coercions": coercions,
            "columns": list(export_df.columns),
        }

    def _write_votable(self, export_df: pd.DataFrame, *, table_name: str) -> Path:
        out_path = self._base / f"{_slugify(table_name)}-{uuid.uuid4().hex}.vot"
        table = Table.from_pandas(export_df.reset_index(drop=True))
        table.write(out_path, format="votable", overwrite=True)
        return out_path

    def _path_to_url(self, path: Path) -> str:
        if self.url_factory is not None:
            return self.url_factory(path)
        return path.resolve().as_uri()

    def _resolve_target_ids(
        self,
        *,
        target_mode: str,
        target_client_id: str | None,
    ) -> list[str] | None:
        """
        Return:
          - None for broadcast
          - a list of recipient client ids otherwise
        """
        assert self.client is not None

        registered = [cid for cid in self.client.get_registered_clients() if cid != "hub"]

        if target_mode == "all":
            return None

        if target_mode == "client":
            if not target_client_id:
                raise RuntimeError("No target SAMP client selected.")
            if target_client_id not in registered:
                raise RuntimeError(f"Selected SAMP client '{target_client_id}' is not connected.")
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

    # -------------------------------------------------------------------------
    # incoming table receive
    # -------------------------------------------------------------------------

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
            payload = self._read_incoming_table(
                sender_id=sender_id,
                msg_id=msg_id,
                mtype=mtype,
                params=params,
                extra=extra,
            )
            self._notify_table_listeners(payload)

            if msg_id is not None:
                assert self.client is not None
                self.client.reply(
                    msg_id,
                    {
                        "samp.status": "samp.ok",
                        "samp.result": {},
                    },
                )
        except Exception as exc:
            if msg_id is not None and self.client is not None:
                self.client.reply(
                    msg_id,
                    {
                        "samp.status": "samp.error",
                        "samp.error": {
                            "samp.errortxt": str(exc),
                        },
                    },
                )
            raise

    def _read_incoming_table(
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

        table = Table.read(url)
        df = table.to_pandas()

        return {
            "sender_id": sender_id,
            "msg_id": msg_id,
            "is_call": msg_id is not None,
            "mtype": mtype,
            "params": params,
            "extra": extra,
            "dataframe": df,
            "name": params.get("name") or "Incoming SAMP table",
            "url": url,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
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

    # -------------------------------------------------------------------------
    # dataframe -> VOTable compatibility
    # -------------------------------------------------------------------------

    def _make_votable_safe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Return a shallow-copy export dataframe with columns coerced where needed
        for VOTable compatibility.

        Keeps the in-memory AstronomicAL dataset untouched.
        """
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

            # VOTable has no signed 8-bit integer primitive.
            if dt == np.int8:
                out[col] = s.astype(np.int16)
                coercions.append({"column": col, "from": "int8", "to": "int16"})

            # Keep unsigned types conservative for broad compatibility.
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
                    raise ValueError(
                        f"Column '{col}' is uint64 with values above int64 range; "
                        "cannot safely represent it in standard VOTable."
                    )

            elif isinstance(dt, pd.CategoricalDtype):
                out[col] = s.astype("string")
                coercions.append({"column": col, "from": "category", "to": "string"})

            elif pdt.is_object_dtype(dt):
                non_null = s.dropna()
                if non_null.empty:
                    continue

                sample_types = {type(v) for v in non_null.iloc[:100]}

                if sample_types <= {bytes, bytearray, np.bytes_}:
                    out[col] = s.map(
                        lambda v: None if pd.isna(v) else bytes(v).decode("utf-8", errors="ignore")
                    ).astype("string")
                    coercions.append({"column": col, "from": "object(bytes)", "to": "string"})

                elif sample_types <= {str}:
                    out[col] = s.astype("string")
                    coercions.append({"column": col, "from": "object(str)", "to": "string"})

                elif sample_types <= numeric_object_types:
                    out[col] = pd.to_numeric(s, errors="raise")
                    coercions.append({"column": col, "from": "object(numeric)", "to": str(out[col].dtype)})

                else:
                    raise ValueError(
                        f"Column '{col}' has mixed object values {sample_types}; "
                        "cannot safely export to VOTable without explicit conversion."
                    )

        return out, coercions