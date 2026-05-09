from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from astronomicAL.utils.debug import (
    compact_list,
    dataset_debug_print,
    mapping_debug_print,
    summarize_pending_restore_items,
)

@dataclass
class Dataset:
    dataset_id: str
    name: str
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

def _json_safe_metadata(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(_json_safe_metadata(key)): _json_safe_metadata(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe_metadata(item) for item in value]

    return str(value)


class DatasetManager:
    """
    Manages datasets. Phase 1 can still run single dataset, but we keep the API multi-ready.

    Added for mapping POC:
    - ensure_registered()
    - list_columns()
    - get_meta()
    - get_mapping() / set_mapping()
    - get_mappings()
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, Dataset] = {}
        self._active_id: Optional[str] = None
        self._pending_restore_items = []
        self._pending_active_id = None

    def register(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        if name is None:
            name = dataset_id

        meta = dict(meta)
        meta.setdefault("column_mappings", {})

        self._datasets[dataset_id] = Dataset(
            dataset_id=dataset_id,
            name=name,
            df=df,
            meta=meta,
        )

        if self._active_id is None:
            self._active_id = dataset_id

        self._ensure_pending_restore_state()

        dataset_debug_print(
            "register",
            {
                "dataset_id": dataset_id,
                "column_count": len(df.columns),
                "columns_preview": compact_list([str(col) for col in df.columns]),
                "pending_restore_items": summarize_pending_restore_items(
                    self._pending_restore_items
                ),
            },
        )

        self._apply_pending_restore_to_dataset(dataset_id)

        dataset_debug_print(
            "register after pending restore",
            {
                "dataset_id": dataset_id,
                "mappings": self.get_mappings(dataset_id)
                if hasattr(self, "get_mappings")
                else self._datasets[dataset_id].meta.get("column_mappings"),
            },
        )

    def ensure_registered(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        """
        Register the dataset if missing; otherwise update the existing dataset in place.
        """
        if dataset_id not in self._datasets:
            self.register(dataset_id, df, name=name, **meta)
            return

        ds = self._datasets[dataset_id]
        ds.df = df

        if name is not None:
            ds.name = name

        if meta:
            ds.meta.update(meta)

        ds.meta.setdefault("column_mappings", {})

    def list_ids(self) -> list[str]:
        return list(self._datasets.keys())

    def active_id(self) -> str:
        if self._active_id is None:
            raise RuntimeError("No active dataset set.")
        return self._active_id

    def set_active(self, dataset_id: str) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")

        self._active_id = dataset_id

        apply_pending = getattr(self, "_apply_pending_restore_to_dataset", None)
        if callable(apply_pending):
            apply_pending(dataset_id)

    def get(self, dataset_id: Optional[str] = None) -> Dataset:
        if dataset_id is None:
            dataset_id = self.active_id()

        if dataset_id not in self._datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")

        return self._datasets[dataset_id]

    def get_df(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        return self.get(dataset_id).df

    def get_meta(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        return self.get(dataset_id).meta

    def list_columns(self, dataset_id: Optional[str] = None) -> list[str]:
        return list(self.get_df(dataset_id).columns)

    def _ensure_pending_restore_state(self) -> None:
        if not hasattr(self, "_pending_restore_items"):
            self._pending_restore_items = []

        if not hasattr(self, "_pending_active_id"):
            self._pending_active_id = None

    def _dataset_columns(self, dataset_id: str) -> list[str]:
        try:
            df = self.get_df(dataset_id)
            return [str(col) for col in df.columns]
        except Exception:
            return []

    def _dataset_meta(self, dataset_id: str) -> dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return {}

        return dict(getattr(dataset, "meta", {}) or {})

    def _dataset_name(self, dataset_id: str) -> str | None:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return None

        return getattr(dataset, "name", None)

    @staticmethod
    def _source_from_meta(meta: dict[str, Any]) -> str | None:
        for key in (
            "source",
            "source_path",
            "path",
            "filename",
            "file",
            "filepath",
            "url",
        ):
            value = meta.get(key)
            if value:
                return str(value)

        return None

    def _dataset_source(self, dataset_id: str) -> str | None:
        return self._source_from_meta(self._dataset_meta(dataset_id))

    @staticmethod
    def _normalise_mapping_column(value: Any) -> str | None:
        if value is None:
            return None

        value = str(value)

        if value.lower() in {"use index", "use_index", "__index__", "index"}:
            return "Use Index"

        return value


    def _mapping_columns_exist(self, mappings: dict[str, Any], columns: list[str]) -> bool:
        if not mappings:
            return True

        column_set = {str(col) for col in columns}

        for semantic_name, column_name in mappings.items():
            column_name = self._normalise_mapping_column(column_name)

            if column_name is None:
                mapping_debug_print(
                    "restored mapping invalid: None column",
                    {
                        "semantic_name": semantic_name,
                        "column_name": column_name,
                    },
                )
                return False

            if column_name == "Use Index":
                continue

            if column_name not in column_set:
                mapping_debug_print(
                    "restored mapping invalid: missing column",
                    {
                        "semantic_name": semantic_name,
                        "column_name": column_name,
                        "available_column_count": len(column_set),
                        "available_columns_preview": compact_list(sorted(column_set)),
                    },
                )
                return False

        return True

    def _pending_item_matches_dataset(
        self,
        item: dict[str, Any],
        dataset_id: str,
    ) -> bool:
        """
        Decide whether a pending restored mapping item belongs to a newly
        registered dataset.

        Matching order:
        1. Exact dataset id
        2. Same source/path metadata
        3. Same dataset name
        4. If there is only one pending item, allow it for the first dataset
        as long as mapped columns exist.
        """
        item_id = item.get("id")
        if item_id and str(item_id) == str(dataset_id):
            dataset_debug_print(
                "pending restore match by dataset_id",
                {
                    "item_id": item_id,
                    "dataset_id": dataset_id,
                },
            )
            return True

        item_meta = dict(item.get("meta") or {})
        item_source = self._source_from_meta(item_meta)
        dataset_source = self._dataset_source(dataset_id)

        if item_source and dataset_source and str(item_source) == str(dataset_source):
            dataset_debug_print(
                "pending restore match by source",
                {
                    "item_source": item_source,
                    "dataset_source": dataset_source,
                },
            )
            return True

        item_name = item.get("name")
        dataset_name = self._dataset_name(dataset_id)

        if item_name and dataset_name and str(item_name) == str(dataset_name):
            dataset_debug_print(
                "pending restore match by name",
                {
                    "item_name": item_name,
                    "dataset_name": dataset_name,
                },
            )
            return True

        self._ensure_pending_restore_state()

        mappings = dict(item.get("mappings") or {})
        columns = self._dataset_columns(dataset_id)

        if len(self._pending_restore_items) == 1:
            valid = self._mapping_columns_exist(mappings, columns)
            dataset_debug_print(
                "pending restore single-item fallback",
                {
                    "dataset_id": dataset_id,
                    "item_id": item_id,
                    "mappings": mappings,
                    "column_count": len(columns),
                    "columns_preview": compact_list(columns),
                    "valid": valid,
                },
            )
            return valid

        dataset_debug_print(
            "pending restore did not match",
            {
                "dataset_id": dataset_id,
                "item_id": item_id,
                "item_name": item_name,
                "dataset_name": dataset_name,
                "item_source": item_source,
                "dataset_source": dataset_source,
                "pending_count": len(self._pending_restore_items),
                "mappings": mappings,
                "column_count": len(columns),
                "columns_preview": compact_list(columns),
            },
        )

        return False

    def _publish_mapping_restored(self, dataset_id: str, mappings: dict[str, Any]) -> None:
        """
        Best-effort event publication.

        DatasetManager may or may not have a direct events reference depending
        on your current constructor. This keeps the method safe either way.
        """
        events = (
            getattr(self, "events", None)
            or getattr(self, "_events", None)
            or getattr(self, "event_bus", None)
            or getattr(self, "_event_bus", None)
        )

        if events is None:
            return

        try:
            events.publish(
                "dataset.mapping_updated",
                {
                    "dataset_id": dataset_id,
                    "mappings": dict(mappings),
                    "origin": "workspace.restore",
                },
            )
        except Exception:
            pass

    def _apply_restore_item_to_dataset(
        self,
        dataset_id: str,
        item: dict[str, Any],
    ) -> bool:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return False

        meta = dict(item.get("meta") or {})
        mappings = dict(item.get("mappings") or {})
        columns = self._dataset_columns(dataset_id)
        column_set = {str(col) for col in columns}

        valid_mappings = {}
        invalid_mappings = {}

        for semantic_key, column_name in mappings.items():
            column_name = self._normalise_mapping_column(column_name)

            if column_name is None:
                continue

            if column_name == "Use Index" or column_name in column_set:
                valid_mappings[str(semantic_key)] = str(column_name)
            else:
                invalid_mappings[str(semantic_key)] = str(column_name)

        dataset_meta = dict(getattr(dataset, "meta", {}) or {})
        dataset_meta.update(meta)

        existing_mappings = dict(dataset_meta.get("column_mappings", {}) or {})
        existing_mappings.update(valid_mappings)

        dataset_meta["column_mappings"] = existing_mappings

        if invalid_mappings:
            dataset_meta["invalid_restored_column_mappings"] = invalid_mappings
        else:
            dataset_meta.pop("invalid_restored_column_mappings", None)

        dataset.meta = dataset_meta

        if item.get("name") is not None:
            try:
                dataset.name = item["name"]
            except Exception:
                pass

        mapping_debug_print(
            "restored mappings",
            {
                "dataset_id": dataset_id,
                "valid": valid_mappings,
                "invalid": invalid_mappings,
            },
        )

        self._publish_mapping_restored(dataset_id, valid_mappings)

        return True

    def _apply_pending_restore_to_dataset(self, dataset_id: str) -> None:
        self._ensure_pending_restore_state()

        if not self._pending_restore_items:
            return

        remaining = []

        for item in self._pending_restore_items:
            if self._pending_item_matches_dataset(item, dataset_id):
                self._apply_restore_item_to_dataset(dataset_id, item)
            else:
                remaining.append(item)

        self._pending_restore_items = remaining

        if (
            self._pending_active_id is not None
            and str(self._pending_active_id) == str(dataset_id)
        ):
            try:
                self.set_active(dataset_id)
            except Exception:
                pass
            self._pending_active_id = None

    def snapshot(self) -> dict[str, Any]:
        items = []

        for dataset_id, dataset in self._datasets.items():
            meta = dict(getattr(dataset, "meta", {}) or {})
            mappings = dict(meta.pop("column_mappings", {}) or {})

            try:
                df = self.get_df(dataset_id)
                columns = [str(col) for col in df.columns]
            except Exception:
                columns = []

            items.append(
                {
                    "id": dataset_id,
                    "name": getattr(dataset, "name", dataset_id),
                    "meta": _json_safe_metadata(meta),
                    "mappings": _json_safe_metadata(mappings),
                    "columns": columns,
                }
            )

        return {
            "active_id": self._active_id,
            "items": items,
        }

    def restore_metadata_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Restore dataset metadata/mappings from a workspace snapshot.

        If datasets are not loaded yet, queue mapping information. If matching
        datasets are already loaded, apply immediately.
        """
        self._ensure_pending_restore_state()

        if not snapshot:
            dataset_debug_print("restore_metadata_snapshot skipped: empty snapshot")
            return

        self._pending_active_id = snapshot.get("active_id")
        items = snapshot.get("items", []) or []

        dataset_debug_print(
            "restore_metadata_snapshot start",
            {
                "active_id": self._pending_active_id,
                "items_count": len(items),
                "existing_datasets": list(self._datasets.keys()),
            },
        )

        for item in items:
            if not isinstance(item, dict):
                continue

            dataset_debug_print(
                "restore_metadata_snapshot item",
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "mappings": dict(item.get("mappings") or {}),
                },
            )

            dataset_id = item.get("id")
            applied = False

            if dataset_id and str(dataset_id) in self._datasets:
                dataset_debug_print(
                    "restore_metadata_snapshot applying immediately",
                    dataset_id,
                )
                applied = self._apply_restore_item_to_dataset(str(dataset_id), item)

            if not applied:
                dataset_debug_print(
                    "restore_metadata_snapshot queueing",
                    {
                        "id": item.get("id"),
                        "mappings": dict(item.get("mappings") or {}),
                    },
                )
                self._pending_restore_items.append(dict(item))

        for dataset_id in list(self._datasets.keys()):
            dataset_debug_print(
                "restore_metadata_snapshot trying pending restore",
                dataset_id,
            )
            self._apply_pending_restore_to_dataset(dataset_id)

        active_id = snapshot.get("active_id")
        if active_id and str(active_id) in self._datasets:
            try:
                self.set_active(str(active_id))
            except Exception:
                pass

        dataset_debug_print(
            "restore_metadata_snapshot complete",
            {
                "pending": summarize_pending_restore_items(self._pending_restore_items),
            },
        )

    def get_mappings(self, dataset_id: Optional[str] = None) -> Dict[str, str]:
        ds = self.get(dataset_id)
        ds.meta.setdefault("column_mappings", {})
        return ds.meta["column_mappings"]

    def get_mapping(
        self,
        dataset_id: Optional[str],
        semantic_name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        mappings = self.get_mappings(dataset_id)
        return mappings.get(semantic_name, default)

    def set_mapping(
        self,
        dataset_id: Optional[str],
        semantic_name: str,
        column_name: str,
    ) -> None:
        mappings = self.get_mappings(dataset_id)
        mappings[semantic_name] = column_name

    def has_mapping(self, dataset_id: Optional[str], semantic_name: str) -> bool:
        mappings = self.get_mappings(dataset_id)
        return semantic_name in mappings

    def snapshot(self) -> dict[str, Any]:
        items = []

        for dataset_id, dataset in self._datasets.items():
            meta = dict(dataset.meta or {})
            mappings = dict(meta.pop("column_mappings", {}) or {})

            items.append(
                {
                    "id": dataset_id,
                    "name": dataset.name,
                    "meta": _json_safe_metadata(meta),
                    "mappings": _json_safe_metadata(mappings),
                }
            )

        return {
            "active_id": self._active_id,
            "items": items,
        }

    def restore_metadata_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return

        for item in snapshot.get("items", []) or []:
            if not isinstance(item, dict):
                continue

            dataset_id = item.get("id")
            if not dataset_id:
                continue

            if dataset_id not in self._datasets:
                continue

            dataset = self._datasets[dataset_id]

            if item.get("name") is not None:
                dataset.name = item["name"]

            meta = dict(item.get("meta") or {})
            mappings = dict(item.get("mappings") or {})

            dataset.meta.update(meta)
            dataset.meta["column_mappings"] = mappings

        active_id = snapshot.get("active_id")
        if active_id and active_id in self._datasets:
            self.set_active(active_id)