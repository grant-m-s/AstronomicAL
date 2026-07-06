from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

def resolve_selection_mode(metric: str, mode: str = "auto") -> str:
    if mode in ("max", "min"):
        return mode
    m = str(metric or "").lower()
    return "min" if any(t in m for t in ("loss", "error", "mae", "mse", "rmse")) else "max"

@dataclass
class ProtocolConfig:
    """Experiment protocol enforced by the harness.

    split_strategy controls only how the selected/main dataset is split when
    validation_source or test_source is "split".

    validation_source:
        split   -> create validation from selected dataset
        dataset -> use protocol_validation_dataset_id

    test_source:
        split   -> create test from selected dataset
        dataset -> use protocol_test_dataset_id
        none    -> no test set
    """

    split_strategy: str = "random"  # random | by_group | temporal | predefined

    validation_source: str = "split"  # split | dataset
    test_source: str = "split"        # split | dataset | none

    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None

    group_column: Optional[str] = None
    split_column: Optional[str] = None

    validation_size: float = 0.1
    test_size: float = 0.2

    selection_metric: str = "val_accuracy"
    selection_mode: str = "auto"
    random_state: int = 42
    protocol_id: str = ""

    def resolved_mode(self) -> str:
        return resolve_selection_mode(
            self.selection_metric,
            self.selection_mode,
        )

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ProtocolConfig":
        def num(key, default):
            try:
                return float(params.get(key, default))
            except Exception:
                return default

        cfg = cls(
            split_strategy=str(
                params.get("protocol_split_strategy", "random")
            ),
            validation_source=str(
                params.get("protocol_validation_source", "split")
            ),
            test_source=str(
                params.get("protocol_test_source", "split")
            ),
            validation_dataset_id=(
                params.get("protocol_validation_dataset_id") or None
            ),
            test_dataset_id=(
                params.get("protocol_test_dataset_id") or None
            ),
            group_column=(
                params.get("protocol_group_column") or None
            ),
            split_column=(
                params.get("protocol_split_column") or None
            ),
            validation_size=num("protocol_validation_size", 0.1),
            test_size=num("protocol_test_size", 0.2),
            selection_metric=str(
                params.get("protocol_selection_metric", "val_accuracy")
            ),
            selection_mode=str(
                params.get("protocol_selection_mode", "auto")
            ),
            random_state=int(params.get("protocol_random_state", 42)),
        )

        allowed_split_strategies = {
            "random",
            "by_group",
            "temporal",
            "predefined",
        }
        if cfg.split_strategy not in allowed_split_strategies:
            raise ValueError(
                f"Unknown protocol_split_strategy {cfg.split_strategy!r}. "
                f"Expected one of {sorted(allowed_split_strategies)}."
            )

        if cfg.validation_source not in {"split", "dataset"}:
            raise ValueError(
                "protocol_validation_source must be 'split' or 'dataset'."
            )

        if cfg.test_source not in {"split", "dataset", "none"}:
            raise ValueError(
                "protocol_test_source must be 'split', 'dataset', or 'none'."
            )

        train_dataset_id = params.get("dataset_id")

        if cfg.validation_source == "dataset":
            if not cfg.validation_dataset_id:
                raise ValueError(
                    "Validation source is 'dataset', but no validation dataset "
                    "was selected."
                )

            if cfg.validation_dataset_id == train_dataset_id:
                raise ValueError(
                    "Validation dataset must be different from the selected "
                    "training dataset."
                )

        if cfg.test_source == "dataset":
            if not cfg.test_dataset_id:
                raise ValueError(
                    "Test source is 'dataset', but no test dataset was selected."
                )

            if cfg.test_dataset_id == train_dataset_id:
                raise ValueError(
                    "Test dataset must be different from the selected training dataset."
                )

        if (
            cfg.validation_source == "dataset"
            and cfg.test_source == "dataset"
            and cfg.validation_dataset_id
            and cfg.test_dataset_id
            and cfg.validation_dataset_id == cfg.test_dataset_id
        ):
            raise ValueError(
                "Validation and test datasets must be different."
            )

        val_from_split = cfg.validation_source == "split"
        test_from_split = cfg.test_source == "split"

        if val_from_split and cfg.split_strategy != "predefined":
            if cfg.validation_size <= 0:
                raise ValueError(
                    "protocol_validation_size must be > 0 when validation "
                    "is split from the selected dataset."
                )

        if test_from_split and cfg.split_strategy != "predefined":
            if cfg.test_size <= 0:
                raise ValueError(
                    "protocol_test_size must be > 0 when test is split from "
                    "the selected dataset. Choose 'No test set' instead."
                )

        if cfg.split_strategy != "predefined":
            split_total = 0.0
            if val_from_split:
                split_total += cfg.validation_size
            if test_from_split:
                split_total += cfg.test_size

            if split_total >= 1.0:
                raise ValueError(
                    "Fractions split from the selected dataset must sum to < 1.0."
                )

        if cfg.split_strategy in {"by_group", "temporal"} and (
            val_from_split or test_from_split
        ):
            if not cfg.group_column:
                raise ValueError(
                    f"{cfg.split_strategy!r} split requires protocol_group_column."
                )

        if cfg.split_strategy == "predefined" and (
            val_from_split or test_from_split
        ):
            if not cfg.split_column:
                raise ValueError(
                    "predefined split requires protocol_split_column."
                )

        cfg.protocol_id = _stable_protocol_id(cfg)
        return cfg

def _stable_protocol_id(cfg: ProtocolConfig) -> str:
    import hashlib

    raw = "|".join(
        str(x)
        for x in (
            cfg.split_strategy,
            cfg.validation_source,
            cfg.test_source,
            cfg.validation_dataset_id,
            cfg.test_dataset_id,
            cfg.group_column,
            cfg.split_column,
            cfg.validation_size,
            cfg.test_size,
            cfg.selection_metric,
            cfg.resolved_mode(),
            cfg.random_state,
        )
    )

    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

@dataclass
class DataBinding:
    """Resolved column binding. The harness resolves this (mappings/inference);
    the recipe only READS it inside load_sample. Data binding is not a recipe
    internal."""
    record_id_column: str
    target_column: Optional[str]
    input_columns: List[str] = field(default_factory=list)
    image_column: Optional[str] = None

@dataclass
class Partition:
    name: str
    record_ids: List[str]
    labels: List[str]
    classes: List[str]
    dataset_id: Optional[str] = None
    source: str = "split"  # split | dataset | none

    def __len__(self) -> int:
        return len(self.record_ids)

@dataclass
@dataclass
class Partitions:
    train: Partition
    val: Partition
    test: Optional[Partition]
    strategy: str
    validation_source: str
    test_source: str
    group_column: Optional[str]
    random_state: int
    protocol_id: str
    target_column: str
    record_id_column: str
    train_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None

@dataclass
class TrainingComponents:
    """What configure_training returns — all the expert's, none of it protocol."""
    optimizer: Any
    scheduler: Any = None
    criterion: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TargetSpec:
    """What 'the model output' means for a run.

    The harness derives this once (via _target_spec) and threads it everywhere
    that used to assume a class count. Classification carries the class list;
    regression carries the number of continuous outputs. New task kinds add a
    new `kind` plus a matching harness, without touching the protocol flow.
    """

    kind: str = "classification"        # classification | regression
    classes: List[str] = field(default_factory=list)
    n_outputs: int = 1

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_outputs(self) -> int:
        """Width of the model's output layer."""
        if self.kind == "classification":
            return len(self.classes)
        return int(self.n_outputs)
