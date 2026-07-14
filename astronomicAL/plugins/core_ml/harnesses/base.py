from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..data.dataset_access import get_dataset_frame
from ..protocol import DataBinding, Partition, Partitions, ProtocolConfig, TargetSpec
from ..runtime import check_cancelled, put_artifact
from ..serialization import json_safe
from ..split_datasets import materialize_split_datasets

class RunHarness:
    """Owns partitioning, selection, test evaluation, and the audit artifacts.

    The recipe is handed ONLY the train loader and a report_epoch() callback.
    It never receives val/test loaders and never computes the selection metric,
    so 'test used in training' and 'best epoch chosen on test' are not
    expressible by a recipe, correct or malicious.

    Subclasses implement the framework/modality specifics:
        _make_loader, _evaluate, _snapshot, _restore,
        _assert_output_dim, _write_model_artifact
    """

    def __init__(self, run, recipe: "MLRecipe"):
        self.run = run
        self.recipe = recipe
        self.protocol: ProtocolConfig = getattr(run, "protocol", None) or ProtocolConfig()
        self.binding: DataBinding = getattr(run, "binding", None)
        self._frame = None
        self._partition_frames: Dict[str, Any] = {}
        self._val_loader = None
        self._best_score = None
        self._best_epoch = None
        self._best_state = None
        self._history: List[Dict[str, Any]] = []

    # ---- the ONE primitive the recipe calls each epoch ---------------------
    def report_epoch(
        self,
        epoch: int,
        model,
        *,
        train_metrics: Dict[str, Any],
    ):
        """Called by managed recipes once per epoch.

        The recipe reports training metrics only. The harness evaluates validation,
        chooses the best epoch, snapshots best weights, and logs a merged curve row.
        """

        self.run.check_cancelled()

        val_metrics = self._evaluate(model, self._val_loader)[0]

        row: Dict[str, Any] = {"epoch": int(epoch)}

        row.update(
            {
                f"train_{key}": value
                for key, value in dict(train_metrics or {}).items()
            }
        )

        row.update(
            {
                f"val_{key}": value
                for key, value in dict(val_metrics or {}).items()
            }
        )

        self._history.append(row)

        score = row.get(self.protocol.selection_metric)

        if score is not None:
            try:
                score_float = float(score)
            except Exception:
                score_float = None

            if score_float is not None and self._is_better(score_float):
                self._best_score = score_float
                self._best_epoch = int(epoch)
                self._best_state = self._snapshot(model)

        self.run.log(
            message=f"epoch {epoch}",
            status="running",
            step=int(epoch),
            metrics=row,
            extra={
                "phase": "epoch",
                "selection_metric": self.protocol.selection_metric,
                "selection_mode": self.protocol.resolved_mode(),
                "best_epoch_so_far": self._best_epoch,
                "best_score_so_far": self._best_score,
            },
        )

        return val_metrics

# ---- task abstraction (overridable; default classification) ------------
    def _task_kind(self) -> str:
        task = str(
            getattr(self.recipe, "task", "")
            or self.run.params.get("task", "")
            or ""
        ).lower()
        if task in {"regression", "regressor"}:
            return "regression"
        return "classification"

    def _target_spec(self, parts: Partitions) -> TargetSpec:
        """Derive what the output means from the partitions + task kind.

        Classification reads the class list discovered during partitioning;
        regression reports the number of continuous outputs (default 1, override
        via params['n_outputs']). Subclasses override for exotic outputs.
        """
        if self._task_kind() == "regression":
            return TargetSpec(
                kind="regression",
                classes=[],
                n_outputs=int(self.run.params.get("n_outputs", 1) or 1),
            )
        return TargetSpec(
            kind="classification",
            classes=list(parts.train.classes),
        )

    def _build_model(self, parts: Partitions, target: TargetSpec):
        """Call the recipe's build_model with task-correct kwargs.

        New recipes may accept `target=`; legacy recipes accept `num_classes=`
        (interpreted as output width). The output-dim check then verifies the
        head matches `target` regardless of which signature was used.
        """
        recipe = self.recipe
        try:
            return recipe.build_model(self.run, target=target)
        except TypeError:
            return recipe.build_model(self.run, num_classes=target.num_outputs)

# ---- the protocol flow (NOT overridable by recipes) --------------------
    def execute(self) -> Dict[str, Any]:
        recipe, run = self.recipe, self.run

        parts = self._partition()                               # PROTOCOL
        split_spec_id = self._write_split_spec(parts)           # AUDIT: row-ids on disk

        target = self._target_spec(parts)                       # what 'output' means
        model = self._build_model(parts, target)
        components = recipe.configure_training(run, model)

        train_loader = self._make_loader(parts.train, train=True)
        self._val_loader = self._make_loader(parts.val, train=False)

        self._assert_output_dim(model, target, train_loader)    # kuangliu-10 trap

        # Expert's loop. It only sees train_loader + report_epoch(harness).
        recipe.fit(run, model=model, components=components,
                   train_loader=train_loader, harness=self)

        if self._best_state is None:
            raise RuntimeError(
                "Recipe completed without calling harness.report_epoch(...). "
                "A managed recipe must report at least one epoch so the harness "
                "can select on the validation partition.")
        self._restore(model, self._best_state)                  # best-epoch weights

        test_metrics, test_records = {}, []
        if parts.test is not None and len(parts.test) > 0:       # test LAST, ONCE
            test_loader = self._make_loader(parts.test, train=False)
            test_metrics, test_records = self._evaluate(
                model, test_loader, return_records=True)

        model_artifact_id = self._write_model_artifact(
            model,
            parts,
            target,
            split_spec_artifact_id=split_spec_id,
        )

        eval_id = self._write_evaluation_report(
            parts=parts,
            split_spec_id=split_spec_id,
            test_metrics=test_metrics,
            model_artifact_id=model_artifact_id,
        )

        predictions_id = None
        if test_records:
            predictions_id = self._write_predictions(
                test_records,
                parts,
                model_artifact_id,
            )

        result = {
            "status": "complete",
            "model_artifact_id": model_artifact_id,
            "evaluation_report_artifact_id": eval_id,
            "predictions_artifact_id": predictions_id,
            "split_spec_artifact_id": split_spec_id,
            "source_dataset_id": self.run.dataset_id,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
            "best_epoch": self._best_epoch,
            "best_score": self._best_score,
            "selection_metric": self.protocol.selection_metric,
            "selection_mode": self.protocol.resolved_mode(),
            "protocol_id": self.protocol.protocol_id,
            "task_kind": target.kind,
            "test_metrics": test_metrics,
            "history": self._history,
        }

        if getattr(self.run, "logger", None) is not None:
            self.run.logger.update_summary(**result)

        return result

    def _load_partition_frame(
        self,
        dataset_id: str,
        *,
        columns: Sequence[str],
        role: str,
    ):
        b = self.binding

        if not dataset_id:
            raise ValueError(f"{role} dataset id is missing.")

        df = get_dataset_frame(
            self.run.context,
            dataset_id,
            columns=list(columns),
        )

        required = [b.record_id_column]

        if b.target_column:
            required.append(b.target_column)

        missing = [
            column
            for column in required
            if column and column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{role} dataset {dataset_id!r} is missing required column(s): "
                + ", ".join(missing)
            )

        if b.target_column:
            df = df.dropna(subset=[b.target_column])

        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"{role} dataset {dataset_id!r} has no usable rows after "
                "dropping missing targets."
            )

        duplicate_count = int(
            df[b.record_id_column].astype(str).duplicated().sum()
        )

        if duplicate_count:
            raise ValueError(
                f"{role} dataset {dataset_id!r} has {duplicate_count} duplicate "
                f"record IDs in column {b.record_id_column!r}."
            )

        return df

    def _partition_from_frame(
        self,
        *,
        name: str,
        frame,
        dataset_id: str,
        classes: List[str],
        source: str,
    ) -> Partition:
        b = self.binding

        record_ids = frame[b.record_id_column].astype(str).tolist()

        labels = (
            frame[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in record_ids]
        )

        return Partition(
            name=name,
            record_ids=record_ids,
            labels=labels,
            classes=list(classes),
            dataset_id=dataset_id,
            source=source,
        )

    def _base_split_indices(
        self,
        df,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        p = self.protocol

        if p.split_strategy == "predefined":
            return self._predefined_base_indices(
                df,
                need_val=need_val,
                need_test=need_test,
            )

        if p.split_strategy == "by_group":
            return self._grouped_base_indices(
                df,
                labels,
                need_val=need_val,
                need_test=need_test,
            )

        if p.split_strategy == "temporal":
            return self._temporal_base_indices(
                df,
                need_val=need_val,
                need_test=need_test,
            )

        return self._random_base_indices(
            labels,
            need_val=need_val,
            need_test=need_test,
        )

    def _random_base_indices(
        self,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np
        from sklearn.model_selection import train_test_split

        p = self.protocol
        idx = np.arange(len(labels))
        labels_arr = np.asarray(labels)

        rest = idx
        test = np.array([], dtype=int)

        if need_test:
            strat = labels_arr if _stratifiable(labels_arr) else None
            rest, test = train_test_split(
                idx,
                test_size=p.test_size,
                random_state=p.random_state,
                stratify=strat,
            )

        val = np.array([], dtype=int)

        if need_val:
            rest_labels = labels_arr[rest]
            strat_rest = rest_labels if _stratifiable(rest_labels) else None

            if need_test:
                rel_val = p.validation_size / max(
                    1e-9,
                    1.0 - p.test_size,
                )
            else:
                rel_val = p.validation_size

            rest, val = train_test_split(
                rest,
                test_size=rel_val,
                random_state=p.random_state,
                stratify=strat_rest,
            )

        train = rest

        return list(train), list(val), list(test)

    def _grouped_base_indices(
        self,
        df,
        labels: Sequence[str],
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np
        from sklearn.model_selection import GroupShuffleSplit

        p = self.protocol

        if not p.group_column or p.group_column not in df.columns:
            raise ValueError(
                "Grouped split requires a valid protocol_group_column."
            )

        groups = df[p.group_column].astype(str).to_numpy()
        idx = np.arange(len(df))

        rest = idx
        test = np.array([], dtype=int)

        if need_test:
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=p.test_size,
                random_state=p.random_state,
            )
            rest_local, test_local = next(
                gss.split(idx, labels, groups)
            )
            rest = idx[rest_local]
            test = idx[test_local]

        val = np.array([], dtype=int)

        if need_val:
            if need_test:
                rel_val = p.validation_size / max(
                    1e-9,
                    1.0 - p.test_size,
                )
            else:
                rel_val = p.validation_size

            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=rel_val,
                random_state=p.random_state,
            )
            tr_local, va_local = next(
                gss2.split(
                    rest,
                    [labels[i] for i in rest],
                    groups[rest],
                )
            )
            train = rest[tr_local]
            val = rest[va_local]
        else:
            train = rest

        return list(train), list(val), list(test)

    def _temporal_base_indices(
        self,
        df,
        *,
        need_val: bool,
        need_test: bool,
    ):
        import numpy as np

        p = self.protocol

        if not p.group_column or p.group_column not in df.columns:
            raise ValueError(
                "Temporal split requires protocol_group_column as the time column."
            )

        order = np.argsort(
            df[p.group_column].to_numpy(),
            kind="stable",
        )

        n = len(order)
        n_test = int(round(n * p.test_size)) if need_test else 0
        n_val = int(round(n * p.validation_size)) if need_val else 0

        if n_val <= 0 and need_val:
            raise ValueError("Temporal validation split is empty.")

        if n_test <= 0 and need_test:
            raise ValueError("Temporal test split is empty.")

        test = (
            order[n - n_test:]
            if n_test
            else np.array([], dtype=int)
        )

        val_end = n - n_test
        val_start = val_end - n_val

        val = (
            order[val_start:val_end]
            if n_val
            else np.array([], dtype=int)
        )

        train = order[:val_start]

        return list(train), list(val), list(test)

    def _predefined_base_indices(
        self,
        df,
        *,
        need_val: bool,
        need_test: bool,
    ):
        p = self.protocol

        if not p.split_column or p.split_column not in df.columns:
            raise ValueError(
                "Predefined split requires a valid protocol_split_column."
            )

        col = df[p.split_column].astype(str).str.lower()

        train = df.index[col.isin(["train", "training"])].tolist()

        val = (
            df.index[col.isin(["val", "valid", "validation"])].tolist()
            if need_val
            else []
        )

        test = (
            df.index[col.isin(["test", "testing", "holdout"])].tolist()
            if need_test
            else []
        )

        if not train:
            raise ValueError(
                "Predefined split column has no train/training rows."
            )

        if need_val and not val:
            raise ValueError(
                "Predefined split column has no val/valid/validation rows."
            )

        if need_test and not test:
            raise ValueError(
                "Predefined split column has no test/testing/holdout rows."
            )

        return train, val, test

    def _external_dataset_partitions(
        self,
        columns: Sequence[str],
    ) -> Partitions:
        b = self.binding
        p = self.protocol

        if not p.validation_dataset_id:
            raise ValueError("External split strategy requires a validation dataset.")

        train_dataset_id = self.run.dataset_id
        val_dataset_id = p.validation_dataset_id
        test_dataset_id = p.test_dataset_id

        train_df = self._load_partition_frame(
            train_dataset_id,
            columns=columns,
            role="training",
        )
        val_df = self._load_partition_frame(
            val_dataset_id,
            columns=columns,
            role="validation",
        )

        test_df = None
        if test_dataset_id:
            test_df = self._load_partition_frame(
                test_dataset_id,
                columns=columns,
                role="test",
            )

        if self._task_kind() == "regression":
            classes: List[str] = []
        else:
            classes = self._classification_classes(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=train_dataset_id,
            classes=classes,
            source="selected",
        )
        val_p = self._partition_from_frame(
            name="val",
            frame=val_df,
            dataset_id=val_dataset_id,
            classes=classes,
            source="dataset",
        )

        test_p = None
        if test_df is not None:
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=test_dataset_id,
                classes=classes,
                source="dataset",
            )

        self._assert_disjoint(train_p, val_p, test_p)

        if self._task_kind() != "regression":
            self._assert_label_subset(classes, val_p, test_p)

        if len(val_p) == 0:
            raise ValueError("External validation dataset is empty.")

        self._frame = train_df
        self._partition_frames = {
            "train": train_df,
            "val": val_df,
        }
        if test_df is not None:
            self._partition_frames["test"] = test_df

        return Partitions(
            train=train_p,
            val=val_p,
            test=test_p,
            strategy=p.split_strategy,
            validation_source=p.validation_source,
            test_source=p.test_source,
            group_column=None,
            random_state=p.random_state,
            protocol_id=p.protocol_id,
            target_column=b.target_column or "",
            record_id_column=b.record_id_column,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=val_dataset_id,
            test_dataset_id=test_dataset_id,
            materialized_split_dataset_ids={},
        )

# ---- partitioning (modality-agnostic) ----------------------------------
    def _partition(self) -> Partitions:
        b = self.binding
        p = self.protocol
        regression = self._task_kind() == "regression"

        cols = [b.record_id_column]
        if b.target_column:
            cols.append(b.target_column)
        cols += [c for c in b.input_columns if c]
        if p.group_column:
            cols.append(p.group_column)
        if p.split_column:
            cols.append(p.split_column)
        cols = list(dict.fromkeys([c for c in cols if c]))

        base_df = self._load_partition_frame(
            self.run.dataset_id,
            columns=cols,
            role="selected training",
        )

        base_labels = (
            base_df[b.target_column].astype(str).tolist()
            if b.target_column
            else ["" for _ in range(len(base_df))]
        )

        val_from_split = p.validation_source == "split"
        test_from_split = p.test_source == "split"

        train_idx, val_idx, test_idx = self._base_split_indices(
            base_df,
            base_labels,
            need_val=val_from_split,
            need_test=test_from_split,
        )

        train_df = base_df.iloc[train_idx].reset_index(drop=True)

        if p.validation_source == "split":
            val_df = base_df.iloc[val_idx].reset_index(drop=True)
            val_dataset_id = self.run.dataset_id
            val_source = "split"
        else:
            val_df = self._load_partition_frame(
                p.validation_dataset_id,
                columns=cols,
                role="validation",
            )
            val_dataset_id = p.validation_dataset_id
            val_source = "dataset"

        test_df = None
        test_dataset_id = None
        test_source = "none"

        if p.test_source == "split":
            test_df = base_df.iloc[test_idx].reset_index(drop=True)
            test_dataset_id = self.run.dataset_id
            test_source = "split"
        elif p.test_source == "dataset":
            test_df = self._load_partition_frame(
                p.test_dataset_id,
                columns=cols,
                role="test",
            )
            test_dataset_id = p.test_dataset_id
            test_source = "dataset"

        if regression:
            classes: List[str] = []
        else:
            classes = self._classification_classes(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )

        train_p = self._partition_from_frame(
            name="train",
            frame=train_df,
            dataset_id=self.run.dataset_id,
            classes=classes,
            source="selected",
        )

        val_p = self._partition_from_frame(
            name="val",
            frame=val_df,
            dataset_id=val_dataset_id,
            classes=classes,
            source=val_source,
        )

        test_p = None
        if test_df is not None:
            test_p = self._partition_from_frame(
                name="test",
                frame=test_df,
                dataset_id=test_dataset_id,
                classes=classes,
                source=test_source,
            )

        self._assert_disjoint(train_p, val_p, test_p)

        if not regression:
            self._assert_label_subset(classes, val_p, test_p)

        if len(val_p) == 0:
            raise ValueError("Validation partition is empty.")

        split_dataset_ids: Dict[str, str] = {}

        # Only materialise partitions that were actually split from the selected
        # source dataset. External validation/test datasets are already explicit
        # platform datasets.
        split_partitions = {}
        split_frames = {}

        if val_from_split or test_from_split:
            split_partitions["train"] = train_p
            split_frames["train"] = train_df

        if val_from_split:
            split_partitions["validation"] = val_p
            split_frames["validation"] = val_df

        if test_from_split and test_p is not None and test_df is not None:
            split_partitions["test"] = test_p
            split_frames["test"] = test_df

        if split_partitions:
            split_dataset_ids = materialize_split_datasets(
                context=self.run.context,
                source_dataset_id=self.run.dataset_id,
                run_id=self.run.run_id,
                recipe_id=self.run.recipe_id,
                recipe_version=self.run.recipe_version,
                protocol=p,
                binding=b,
                params=self.run.params,
                partitions=split_partitions,
                fallback_frames=split_frames,
            )

            if split_dataset_ids.get("train"):
                train_p.dataset_id = split_dataset_ids["train"]
                train_p.source = "materialized_split"

            if split_dataset_ids.get("validation"):
                val_p.dataset_id = split_dataset_ids["validation"]
                val_p.source = "materialized_split"

            if test_p is not None and split_dataset_ids.get("test"):
                test_p.dataset_id = split_dataset_ids["test"]
                test_p.source = "materialized_split"

            self.run.params["train_dataset_id"] = train_p.dataset_id
            self.run.params["validation_dataset_id"] = val_p.dataset_id
            self.run.params["test_dataset_id"] = (
                test_p.dataset_id if test_p is not None else None
            )
            self.run.params["split_dataset_ids"] = dict(split_dataset_ids)


        self._frame = train_df
        self._partition_frames = {
            "train": train_df,
            "val": val_df,
        }
        if test_df is not None:
            self._partition_frames["test"] = test_df

        return Partitions(
            train=train_p,
            val=val_p,
            test=test_p,
            strategy=p.split_strategy,
            validation_source=p.validation_source,
            test_source=p.test_source,
            group_column=p.group_column,
            random_state=p.random_state,
            protocol_id=p.protocol_id,
            target_column=b.target_column or "",
            record_id_column=b.record_id_column,
            train_dataset_id=train_p.dataset_id,
            validation_dataset_id=val_p.dataset_id,
            test_dataset_id=test_p.dataset_id if test_p is not None else None,
            materialized_split_dataset_ids=split_dataset_ids,
        )

    def _random_indices(self, idx, labels, p):
        from sklearn.model_selection import train_test_split
        import numpy as np
        labels = np.asarray(labels)
        strat = labels if _stratifiable(labels) else None
        rest, test = (idx, np.array([], int))
        if p.test_size > 0:
            rest, test = train_test_split(
                idx, test_size=p.test_size, random_state=p.random_state, stratify=strat)
        strat_rest = labels[rest] if (strat is not None) else None
        if strat_rest is not None and not _stratifiable(strat_rest):
            strat_rest = None
        rel_val = p.validation_size / max(1e-9, 1.0 - p.test_size)
        train, val = train_test_split(
            rest, test_size=rel_val, random_state=p.random_state, stratify=strat_rest)
        return list(train), list(val), list(test)

    def _grouped_indices(self, df, labels, p):
        # Same group never spans partitions — the astronomy cutout/object case.
        from sklearn.model_selection import GroupShuffleSplit
        import numpy as np
        if not p.group_column or p.group_column not in df.columns:
            raise ValueError("by_group split requires a valid protocol_group_column.")
        groups = df[p.group_column].astype(str).to_numpy()
        idx = np.arange(len(df))
        rest, test = idx, np.array([], int)
        if p.test_size > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=p.test_size,
                                    random_state=p.random_state)
            rest, test = next(gss.split(idx, labels, groups))
        rel_val = p.validation_size / max(1e-9, 1.0 - p.test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val,
                                 random_state=p.random_state)
        tr_local, va_local = next(gss2.split(rest, [labels[i] for i in rest],
                                             groups[rest]))
        return list(rest[tr_local]), list(rest[va_local]), list(test)

    def _temporal_indices(self, df, p):
        import numpy as np
        if not p.group_column or p.group_column not in df.columns:
            raise ValueError("temporal split requires protocol_group_column (a time column).")
        order = np.argsort(df[p.group_column].to_numpy(), kind="stable")
        n = len(order)
        n_test = int(round(n * p.test_size))
        n_val = int(round(n * p.validation_size))
        test = order[n - n_test:] if n_test else np.array([], int)
        val = order[n - n_test - n_val:n - n_test] if n_val else np.array([], int)
        train = order[:n - n_test - n_val]
        return list(train), list(val), list(test)

    def _predefined_indices(self, df, p):
        if not p.split_column or p.split_column not in df.columns:
            raise ValueError("predefined split requires a valid protocol_split_column.")
        col = df[p.split_column].astype(str).str.lower()
        tr = df.index[col.isin(["train", "training"])].tolist()
        va = df.index[col.isin(["val", "valid", "validation"])].tolist()
        te = df.index[col.isin(["test", "testing", "holdout"])].tolist()
        return tr, va, te

    def _assert_disjoint(self, train_p, val_p, test_p):
        s_tr, s_va = set(train_p.record_ids), set(val_p.record_ids)
        bad = s_tr & s_va
        if bad:
            raise RuntimeError(f"train/val share {len(bad)} record-ids (leakage).")
        if test_p is not None:
            s_te = set(test_p.record_ids)
            if s_tr & s_te or s_va & s_te:
                raise RuntimeError("test partition overlaps train/val (leakage).")

    def _is_active_learning_run(self) -> bool:
        params = dict(getattr(self.run, "params", {}) or {})
        if params.get("al_session_id") or params.get("al_session_artifact_id"):
            return True

        protocol = str(params.get("al_protocol") or "").strip().lower()
        return protocol in {"review", "active_learning", "active-learning", "al", "benchmark"}

    def _normalise_class_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("\n", ",").split(",")]
        elif isinstance(value, Mapping):
            parts = [str(key).strip() for key in value.keys()]
        elif isinstance(value, Iterable):
            parts = [str(item).strip() for item in value]
        else:
            parts = [str(value).strip()]

        out: List[str] = []
        for item in parts:
            if not item:
                continue
            if item not in out:
                out.append(item)
        return out

    def _labels_from_frame(self, frame) -> List[str]:
        b = self.binding
        if frame is None or not b.target_column or b.target_column not in frame.columns:
            return []

        values: List[str] = []
        try:
            raw = frame[b.target_column].dropna().astype(str).tolist()
        except Exception:
            raw = []

        for value in raw:
            value = str(value).strip()
            if value and value not in values:
                values.append(value)
        return values

    def _declared_class_universe(self) -> List[str]:
        """Return the run-declared class universe, preserving user order.

        Active-learning runs should pass this from the AL session's label_options.
        The harness also accepts common aliases because recipes/plugins may use
        different names.
        """
        params = dict(getattr(self.run, "params", {}) or {})

        for key in (
            "label_options",
            "class_labels",
            "classes",
            "class_names",
            "known_classes",
            "target_classes",
        ):
            labels = self._normalise_class_list(params.get(key))
            if labels:
                return labels

        return []

    def _classification_classes(
        self,
        *,
        train_df,
        val_df=None,
        test_df=None,
    ) -> List[str]:
        """Resolve model class universe for classification.

        Non-AL default:
            Use classes present in the training partition.

        AL default:
            Prefer declared label_options/classes from the AL session. If those
            are missing, include labels from train/val/test so evaluation can
            cover every known label instead of crashing on labels absent from
            the initial training subset.
        """
        train_labels = self._labels_from_frame(train_df)
        val_labels = self._labels_from_frame(val_df)
        test_labels = self._labels_from_frame(test_df)

        declared = self._declared_class_universe()
        is_al = self._is_active_learning_run()

        if declared:
            classes = list(declared)
            source = "declared"
        elif is_al:
            classes = list(dict.fromkeys([*train_labels, *val_labels, *test_labels]))
            source = "active_learning_eval_partitions"
        else:
            classes = sorted(set(train_labels))
            source = "training_partition"

        class_set = set(str(label) for label in classes)

        missing_train_labels = sorted(set(str(label) for label in train_labels) - class_set)
        if missing_train_labels:
            raise ValueError(
                "Training labels contain value(s) outside the model class universe: "
                f"{missing_train_labels}. Known classes are {classes}."
            )

        if len(classes) < 2:
            raise ValueError(
                f"Classification requires at least two known classes. Resolved {classes} "
                f"from source={source!r}."
            )

        train_seen = sorted(set(str(label) for label in train_labels))
        val_seen = sorted(set(str(label) for label in val_labels))
        test_seen = sorted(set(str(label) for label in test_labels))

        self._class_universe_info = {
            "source": source,
            "active_learning": bool(is_al),
            "classes": list(classes),
            "train_seen_classes": train_seen,
            "validation_seen_classes": val_seen,
            "test_seen_classes": test_seen,
            "classes_without_training_examples": [
                label for label in classes if str(label) not in set(train_seen)
            ],
            "evaluation_includes_classes_absent_from_training": bool(
                (set(val_seen) | set(test_seen)) - set(train_seen)
            ),
        }

        return classes

    def _assert_label_subset(self, classes, val_p, test_p):
        cset = set(str(label) for label in classes)

        for part in (val_p, test_p):
            if part is None:
                continue

            extra = sorted(set(str(label) for label in part.labels) - cset)
            if extra:
                raise ValueError(
                    f"{part.name} contains labels that are not in the model class "
                    f"universe: {extra}. Known classes are {sorted(cset)}. "
                    "For active learning, pass the full class list through "
                    "label_options/classes/class_labels when training."
                )

    def _is_better(self, score: float) -> bool:
        if self._best_score is None:
            return True
        return (score > self._best_score
                if self.protocol.resolved_mode() == "max"
                else score < self._best_score)

    # ---- audit artifacts (protocol-owned) ----------------------------------
    def _write_split_spec(self, parts: Partitions) -> Optional[str]:
        return self.run.put_artifact(
            "ml.split_spec",
            {
                "schema_version": 3,
                "run_id": self.run.run_id,
                "dataset_id": self.run.dataset_id,
                "source_dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
                "partition_dataset_ids": {
                    "train": parts.train_dataset_id,
                    "validation": parts.validation_dataset_id,
                    "test": parts.test_dataset_id,
                },
                "partition_sources": {
                    "train": parts.train.source,
                    "validation": parts.val.source,
                    "test": parts.test.source if parts.test else "none",
                },
                "protocol_id": parts.protocol_id,
                "split_strategy": parts.strategy,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
                "group_column": parts.group_column,
                "random_state": parts.random_state,
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "train_row_ids": parts.train.record_ids,
                "validation_row_ids": parts.val.record_ids,
                "test_row_ids": parts.test.record_ids if parts.test else [],
                "classes": parts.train.classes,
                "class_universe": json_safe(
                    dict(getattr(self, "_class_universe_info", {}) or {})
                ),
                "isolation": {
                    "partitions_disjoint": True,
                    "test_used_in_training": False,
                    "test_used_in_selection": False,
                    "selection_partition": "validation",
                    "explicit_split_datasets": bool(parts.materialized_split_dataset_ids),
                },
            },
        )

    def _write_evaluation_report(
        self,
        *,
        parts,
        split_spec_id,
        test_metrics,
        model_artifact_id,
    ) -> Optional[str]:
        return self.run.put_artifact(
            "ml.evaluation_report",
            {
                "schema_version": 2,
                "run_id": self.run.run_id,
                "dataset_id": self.run.dataset_id,
                "train_dataset_id": parts.train_dataset_id,
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
                "model_artifact_id": model_artifact_id,
                "split_spec_artifact_id": split_spec_id,
                "protocol_id": parts.protocol_id,
                "selection_metric": self.protocol.selection_metric,
                "selected_on_partition": "validation",
                "best_epoch": self._best_epoch,
                "best_score": self._best_score,
                "history": self._history,
                "metrics": test_metrics,
                "reported_metrics": test_metrics,
                "reported_partition": "test" if parts.test else None,
                "classes": parts.train.classes,
                "class_universe": json_safe(
                    dict(getattr(self, "_class_universe_info", {}) or {})
                ),
                "evaluation_scope": "all_known_classes",
                "validity": {
                    "metric_partition": "test" if parts.test else "none",
                    "selection_partition": "validation",
                    "selection_touched_reported_partition": False,
                },
            },
        )

    def _write_predictions(self, records, parts, model_artifact_id) -> Optional[str]:
        return self.run.put_artifact("ml.predictions", {
            "schema_version": 2,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "model_artifact_id": model_artifact_id,
            "protocol_id": parts.protocol_id,
            "prediction_scope": "test",
            "record_id_column": parts.record_id_column,
            "class_names": parts.train.classes,
            "records": records,
        }, row_ids=[r["record_id"] for r in records])

# ---- modality/framework specifics (subclasses implement) ---------------
    def _make_loader(self, partition: Partition, *, train: bool): raise NotImplementedError
    def _evaluate(self, model, loader, *, return_records: bool = False): raise NotImplementedError
    def _snapshot(self, model): raise NotImplementedError
    def _restore(self, model, state): raise NotImplementedError
    def _assert_output_dim(self, model, target: TargetSpec, train_loader): raise NotImplementedError
    def _write_model_artifact(self, model, parts, target: TargetSpec, *, split_spec_artifact_id=None) -> Optional[str]: raise NotImplementedError

def _stratifiable(labels) -> bool:
    import numpy as np
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return len(vals) > 1 and counts.min() >= 2
