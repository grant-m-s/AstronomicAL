from __future__ import annotations

import json
import os
import re
import time

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..paths import ml_run_artifact_dir
from ..protocol import Partition, Partitions, TargetSpec, _stable_protocol_id
from ..runtime import put_artifact
from ..serialization import json_safe
from ..normalization import (
    apply_train_split_image_normalization,
    should_compute_train_split_normalization,
)
from .base import RunHarness

class TorchRegressionHarness(RunHarness):

    # val-partition metrics this harness emits. A stray classification default
    # (e.g. val_accuracy) is coerced to val_loss so epoch selection can't
    # silently no-op on a metric the rows never contain.
    _REGRESSION_VAL_METRICS = (
        "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2",
    )
    _CLASSIFICATION_TOKENS = (
        "accuracy", "f1", "auc", "precision", "recall", "balanced",
    )

    def __init__(self, run, recipe: "MLRecipe"):
        super().__init__(run, recipe)
        if self.binding is None or not self.binding.target_column:
            raise ValueError(
                "Regression runs require a target column. Set `target_column`, "
                "map `target_label`, or add a numeric target column to the "
                "dataset. (The runner only auto-requires a target for "
                "classification, so regression must assert it here.)"
            )
        self._coerce_selection_metric()

    # ---- selection-metric sanity ------------------------------------------
    def _coerce_selection_metric(self) -> None:
        metric = str(self.protocol.selection_metric or "").strip()
        lowered = metric.lower()

        needs_default = (
            not metric
            or any(tok in lowered for tok in self._CLASSIFICATION_TOKENS)
        )
        if not needs_default:
            return

        self.protocol.selection_metric = "val_loss"
        self.protocol.selection_mode = "min"            # val_loss must minimise
        # Provenance must stay truthful: protocol_id is a hash of the config.
        self.protocol.protocol_id = _stable_protocol_id(self.protocol)

        self.run.log(
            message=(
                f"Regression run: selection metric {metric or '(unset)'!r} is not "
                "a regression metric; selecting best epoch on 'val_loss' (min)."
            ),
            status="running",
            extra={"phase": "protocol", "selection_metric": "val_loss"},
        )

    # ---- task identity (drives base _target_spec / regression branches) ----
    def _task_kind(self) -> str:
        return "regression"

    def _n_outputs(self) -> int:
        try:
            return max(1, int(self.run.params.get("n_outputs", 1) or 1))
        except Exception:
            return 1

    # ---- torch plumbing (parallels TorchClassificationHarness) -------------
    @property
    def device(self):
        return self._device()

    def _device(self):
        import torch
        req = str(self.run.params.get("device", "auto")).lower()
        if req == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _ensure_train_split_normalization(self, partition: Partition) -> None:
        """Compute image mean/std once, after splitting, using train rows only."""

        if getattr(self, "_train_split_normalization_done", False):
            return

        if not should_compute_train_split_normalization(self.run.params):
            self._train_split_normalization_done = True
            return

        train_partition = self._resolve_train_partition(partition)
        if train_partition is None:
            return

        info = apply_train_split_image_normalization(
            context=self.run.context,
            params=self.run.params,
            source_dataset_id=self.run.dataset_id,
            train_dataset_id=(
                getattr(train_partition, "dataset_id", None)
                or self.run.params.get("train_dataset_id")
            ),
            train_row_ids=list(train_partition.record_ids),
            record_id_column=self.binding.record_id_column,
            image_column=(
                self.binding.image_column
                or self.run.params.get("image_column")
                or self.run.params.get("image_path_column")
            ),
            cancel_token=getattr(self.run, "cancel_token", None),
        )

        self._train_split_normalization_info = info
        self._train_split_normalization_done = True

        if info:
            self.run.log(
                message="Calculated image mean/std from training split.",
                status="running",
                extra={
                    "phase": "normalization",
                    "normalization": info,
                },
            )

    def _resolve_train_partition(self, partition: Partition) -> Optional[Partition]:
        if getattr(partition, "name", None) == "train":
            return partition

        parts = (
            getattr(self, "_parts", None)
            or getattr(self, "_partitions", None)
            or getattr(self, "partitions", None)
        )

        train_partition = getattr(parts, "train", None)
        if train_partition is not None:
            return train_partition

        return None

    def _transform_metadata(self) -> Dict[str, Any]:
        params = dict(self.run.params or {})
        transform_meta: Dict[str, Any] = {}

        image_size = (
            params.get("image_size")
            or params.get("input_size")
            or params.get("resize")
        )

        if image_size is not None and image_size != "":
            try:
                transform_meta["image_size"] = int(image_size)
            except Exception:
                transform_meta["image_size"] = image_size

        mean = self._vector_param(
            params,
            (
                "normalization_mean",
                "normalization_means",
                "image_mean",
                "normalize_mean",
                "mean",
            ),
        )

        std = self._vector_param(
            params,
            (
                "normalization_std",
                "normalization_stds",
                "image_std",
                "normalize_std",
                "std",
            ),
        )

        normalization = params.get("normalization")
        if isinstance(normalization, Mapping):
            transform_meta["normalization"] = dict(normalization)

        if mean or std:
            existing = dict(transform_meta.get("normalization") or {})
            if mean:
                existing["mean"] = mean
                transform_meta["mean"] = mean
            if std:
                existing["std"] = std
                transform_meta["std"] = std
            transform_meta["normalization"] = existing

        computed = params.get("computed_normalization")
        if isinstance(computed, Mapping):
            transform_meta["computed_normalization"] = json_safe(dict(computed))

        return json_safe(transform_meta)

    def _vector_param(
        self,
        params: Mapping[str, Any],
        keys: Sequence[str],
    ) -> List[float]:
        for key in keys:
            value = params.get(key)
            if value is None or value == "":
                continue

            parsed = self._parse_vector(value)
            if parsed:
                return parsed

        normalization = params.get("normalization")
        if isinstance(normalization, Mapping):
            for key in keys:
                value = normalization.get(key)
                if value is None or value == "":
                    continue

                parsed = self._parse_vector(value)
                if parsed:
                    return parsed

        return []

    def _parse_vector(self, value: Any) -> List[float]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []

            try:
                value = json.loads(text)
            except Exception:
                value = [
                    part.strip()
                    for chunk in text.splitlines()
                    for part in chunk.split(",")
                    if part.strip()
                ]

        if isinstance(value, Mapping):
            return []

        try:
            return [float(item) for item in value]
        except Exception:
            try:
                return [float(value)]
            except Exception:
                return []

    def _frame_for(self, partition: Partition):
        b = self.binding
        frame = (
            self._partition_frames.get(partition.name)
            if hasattr(self, "_partition_frames")
            else None
        )
        if frame is None:
            frame = self._frame
        wanted = set(str(record_id) for record_id in partition.record_ids)
        return frame[frame[b.record_id_column].astype(str).isin(wanted)]

    def _snapshot(self, model):
        m = model.module if hasattr(model, "module") else model
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def _restore(self, model, state):
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(state)

    # ---- target reading ----------------------------------------------------
    def _read_target(self, value, n_outputs: int):
        import numpy as np
        if isinstance(value, (list, tuple, np.ndarray)):
            vec = [float(v) for v in np.asarray(value).reshape(-1)]
        elif isinstance(value, str) and any(s in value for s in (",", ";")):
            vec = [float(p) for p in re.split(r"[,;]", value) if p.strip() != ""]
        else:
            vec = [float(value)]
        if len(vec) != n_outputs:
            raise ValueError(
                f"Target has {len(vec)} value(s) but the run expects "
                f"n_outputs={n_outputs}. For multi-output regression, store a "
                f"vector per row in {self.binding.target_column!r} (list or "
                f"comma-separated) and set params['n_outputs']."
            )
        return vec

    # ---- modality/framework leaf methods -----------------------------------
    def _make_loader(self, partition: Partition, *, train: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        recipe, run, b = self.recipe, self.run, self.binding

        self._ensure_train_split_normalization(partition)

        frame = self._frame_for(partition).reset_index(drop=True)
        transform = recipe.train_transform(run) if train else recipe.eval_transform(run)
        n_outputs = self._n_outputs()
        read_target = self._read_target
        target_col = b.target_column

        class _DS(Dataset):
            def __len__(self):
                return len(frame)

            def __getitem__(self, i):
                row = frame.iloc[i]
                x = recipe.load_sample(run, row)

                if transform is not None:
                    x = transform(x)

                shape = tuple(getattr(x, "shape", ()) or ())
                if len(shape) >= 3:
                    h, w = int(shape[-2]), int(shape[-1])
                    if h <= 0 or w <= 0:
                        raise ValueError(
                            f"{recipe.id} produced an invalid image tensor shape {shape} "
                            f"for row {row.get(b.record_id_column)!r}."
                        )

                vec = read_target(row[target_col], n_outputs) if target_col else [0.0] * n_outputs
                y = torch.tensor(vec, dtype=torch.float32)
                return x, y, str(row[b.record_id_column])

        def collate_with_image_size_hint(batch):
            try:
                from torch.utils.data._utils.collate import default_collate
                return default_collate(batch)
            except RuntimeError as exc:
                text = str(exc)
                if "stack expects each tensor to be equal size" not in text:
                    raise

                shapes = []
                row_ids = []
                for x, _y, rid in batch[:8]:
                    shapes.append(tuple(getattr(x, "shape", ()) or ()))
                    row_ids.append(str(rid))

                raise ValueError(
                    "Image tensors in this batch have different shapes, so PyTorch "
                    "cannot stack them. Image recipes must enforce a fixed output "
                    "size in both train_transform() and eval_transform(). "
                    f"First batch shapes: {shapes}; row_ids: {row_ids}"
                ) from exc

        return DataLoader(
            _DS(),
            batch_size=int(run.params.get("batch_size", 128)),
            shuffle=bool(train),
            num_workers=int(run.params.get("num_workers", 0)),
            pin_memory=str(self._device()).startswith("cuda"),
            collate_fn=collate_with_image_size_hint,
        )

    def _assert_output_dim(self, model, target: TargetSpec, train_loader):
        # Regression analogue of the kuangliu wrong-width-head trap: catch a head
        # that emits the wrong number of continuous outputs before training.
        import torch
        expected = int(target.num_outputs)
        device = self._device()
        model.to(device)
        model.eval()
        x, _y, _ids = next(iter(train_loader))
        if hasattr(x, "to"):
            x = x.to(device)
        with torch.no_grad():
            out = self.recipe.eval_forward(model, x)
        out_dim = 1 if out.dim() == 1 else int(out.shape[1])
        if out_dim != expected:
            raise ValueError(
                f"Model produces {out_dim} output(s) but the run expects "
                f"{expected} (regression, n_outputs={expected}). Give the model a "
                f"final layer with {expected} unit(s), or pass the output width "
                f"through your build_model."
            )

    def _evaluate(self, model, loader, *, return_records: bool = False):
        import numpy as np
        import torch

        device = self._device()
        model.to(device)
        model.eval()

        preds_all, true_all, ids_all = [], [], []
        with torch.no_grad():
            for x, y, ids in loader:
                if hasattr(x, "to"):
                    x = x.to(device)
                out = self.recipe.eval_forward(model, x)
                out = out.detach().cpu().float().numpy()
                if out.ndim == 1:
                    out = out.reshape(-1, 1)
                yb = (
                    y.detach().cpu().float().numpy()
                    if hasattr(y, "detach")
                    else np.asarray(y, dtype=float)
                )
                if yb.ndim == 1:
                    yb = yb.reshape(-1, 1)
                preds_all.append(out)
                true_all.append(yb)
                ids_all.extend(str(i) for i in ids)

        if not preds_all:
            return (
                {"loss": None, "mse": None, "rmse": None, "mae": None, "r2": None},
                [],
            )

        preds = np.concatenate(preds_all, axis=0)
        true = np.concatenate(true_all, axis=0)
        metrics = self._regression_metrics(true, preds)

        records: List[Dict[str, Any]] = []
        if return_records:
            single = preds.shape[1] == 1
            for rid, p_row, t_row in zip(ids_all, preds, true):
                pred_val = float(p_row[0]) if single else [float(v) for v in p_row]
                true_val = float(t_row[0]) if single else [float(v) for v in t_row]
                rec = {"record_id": rid, "prediction": pred_val, "y_true": true_val}
                if single:
                    rec["abs_error"] = abs(pred_val - true_val)
                records.append(rec)

        return metrics, records

    def _regression_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        import numpy as np
        yt = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
        yp = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
        diff = yp - yt
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(diff)))
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(yt, yp, multioutput="uniform_average"))
        except Exception:
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((yt - yt.mean(axis=0)) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        # 'loss' == mse so selecting on val_loss (min) is well defined and
        # comparable regardless of the recipe's training criterion.
        return {"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def _write_model_artifact(
        self,
        model,
        parts: Partitions,
        target: TargetSpec,
        *,
        split_spec_artifact_id: Optional[str] = None,
    ) -> Optional[str]:
        import torch

        n_outputs = int(target.num_outputs)

        model_dir = ml_run_artifact_dir(self.run, kind="model")
        checkpoint_path = model_dir / "model.pt"
        tmp_checkpoint_path = model_dir / "model.pt.tmp"
        manifest_path = model_dir / "model_manifest.json"

        m = model.module if hasattr(model, "module") else model

        architecture = str(self.run.params.get("architecture") or "custom")
        custom_model_import = str(self.run.params.get("custom_model_import") or "").strip()

        # Only record transform hints the recipe actually supplied. If automatic
        # mean/std was enabled, these values were calculated from the training
        # split before transforms/loaders were built.
        transform_meta = self._transform_metadata()

        target_columns = [parts.target_column] if parts.target_column else []

        checkpoint_payload = {
            "schema_version": 2,
            "state_dict": m.state_dict(),
            "class_names": [],
            "num_classes": int(n_outputs),      # output width; kept for loader parity
            "num_outputs": int(n_outputs),
            "params": dict(self.run.params),
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "run_id": self.run.run_id,
            "source_dataset_id": self.run.dataset_id,
            "dataset_id": parts.train_dataset_id or self.run.dataset_id,
            "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
            "protocol_id": parts.protocol_id,
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "architecture": architecture,
            "custom_model_import": custom_model_import,
            "epoch": self._best_epoch,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
            },
            "transform": transform_meta,
        }

        # Atomic-ish write: temp then replace.
        torch.save(checkpoint_payload, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)

        manifest_payload = {
            "schema_version": 2,
            "kind": "torch_regressor",
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "run_id": self.run.run_id,
            "source_dataset_id": self.run.dataset_id,
            "dataset_id": parts.train_dataset_id or self.run.dataset_id,
            "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "protocol_id": parts.protocol_id,
            "split_spec_artifact_id": split_spec_artifact_id,
            "created_at": time.time(),
            "class_names": [],
            "num_classes": int(n_outputs),
            "num_outputs": int(n_outputs),
            "target_columns": target_columns,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "files": {
                "checkpoint": str(checkpoint_path),
                "manifest": str(manifest_path),
            },
            "model_ref": {
                "storage": "local_file",
                "uri": str(checkpoint_path),
                "path": str(checkpoint_path),
                "format": "torch_checkpoint",
                "framework": "torch",
                "metadata": {
                    "class_names": [],
                    "num_classes": int(n_outputs),
                    "num_outputs": int(n_outputs),
                    "task": "regression",
                    "recipe_id": self.run.recipe_id,
                    "recipe_version": self.run.recipe_version,
                    "run_id": self.run.run_id,
                    "architecture": architecture,
                    "custom_model_import": custom_model_import,
                    **transform_meta,
                },
            },
            "input_contract": {
                "record_id_column": parts.record_id_column,
                "target_column": parts.target_column,
                "image_column": self.binding.image_column,
                "input_columns": list(self.binding.input_columns or []),
            },
            "protocol": {
                "protocol_id": parts.protocol_id,
                "split_strategy": parts.strategy,
                "group_column": parts.group_column,
                "random_state": parts.random_state,
                "selection_metric": self.protocol.selection_metric,
                "selection_mode": self.protocol.resolved_mode(),
                "validation_dataset_id": parts.validation_dataset_id,
                "test_dataset_id": parts.test_dataset_id,
                "validation_source": parts.validation_source,
                "test_source": parts.test_source,
            },
            "model_title": getattr(self.recipe, "title", self.run.recipe_id),
            "architecture": architecture,
            "custom_model_import": custom_model_import,
            "transform": transform_meta,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
            },
        }

        manifest_path.write_text(
            json.dumps(json_safe(manifest_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return self.run.put_artifact(
            "ml.model",
            manifest_payload,
            params=self.run.params,
        )

def _is_torch_regression(
    framework="", task="", modality="", run=None, recipe=None, **_kwargs
) -> bool:
    return (
        str(framework).lower() == "torch"
        and str(task).lower() in {"regression", "regressor"}
    )
