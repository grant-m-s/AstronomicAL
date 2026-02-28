# astronomicAL/platform/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Dataset:
    dataset_id: str
    name: str
    df: pd.DataFrame
    meta: Dict[str, Any]


class DatasetManager:
    """
    Manages datasets. Phase 1 can still run single dataset, but we keep the API multi-ready.

    - register()
    - set_active()
    - get_df()
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, Dataset] = {}
        self._active_id: Optional[str] = None

    def register(self, dataset_id: str, df: pd.DataFrame, *, name: Optional[str] = None, **meta: Any) -> None:
        if name is None:
            name = dataset_id
        self._datasets[dataset_id] = Dataset(dataset_id=dataset_id, name=name, df=df, meta=dict(meta))
        if self._active_id is None:
            self._active_id = dataset_id

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