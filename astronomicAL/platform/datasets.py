from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Dataset:
    dataset_id: str
    name: str
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


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