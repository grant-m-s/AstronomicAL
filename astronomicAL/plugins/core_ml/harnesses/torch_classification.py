from __future__ import annotations

import json
import os
import time

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..paths import ml_run_artifact_dir
from ..protocol import Partition, Partitions, TargetSpec
from ..runtime import put_artifact
from ..serialization import json_safe
from .base import RunHarness

class TorchClassificationHarness(RunHarness):

    @property
    def device(self):
        return self._device()

    def _device(self):
        import torch
        req = str(self.run.params.get("device", "auto")).lower()
        if req == "cpu":
            return torch.device("cpu")
        if req == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        return frame[
            frame[b.record_id_column].astype(str).isin(wanted)
        ]

    def _make_loader(self, partition: Partition, *, train: bool):
        from torch.utils.data import DataLoader, Dataset

        recipe, run, b = self.recipe, self.run, self.binding
        frame = self._frame_for(partition).reset_index(drop=True)
        transform = recipe.train_transform(run) if train else recipe.eval_transform(run)
        class_to_idx = {c: i for i, c in enumerate(partition.classes)}
        self._eval_classes = list(partition.classes)

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

                y = class_to_idx[str(row[b.target_column])] if b.target_column else -1
                return x, int(y), str(row[b.record_id_column])

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

    def _assert_output_dim(self, model, target, train_loader):
        # Catches the kuangliu ResNet18()-ignores-num_classes trap loudly,
        # before training a wrong-width head.
        import torch
        expected = int(target.num_outputs)
        model.eval()
        device = self._device()
        model.to(device)
        x, _y, _ids = next(iter(train_loader))
        with torch.no_grad():
            logits = self.recipe.eval_forward(model, x.to(device))
        out = int(logits.shape[1])
        if out != expected:
            raise ValueError(
                f"Model produces {out} outputs but the run expects {expected} "
                f"({target.kind}). The architecture is ignoring the output width "
                f"(common with hard-coded CIFAR-10 model factories). Pass the "
                f"output width through, or wrap the final layer.")

    def _evaluate(self, model, loader, *, return_records: bool = False):
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score

        device = self._device()
        model.to(device)
        model.eval()
        crit = torch.nn.CrossEntropyLoss()
        loss_sum = seen = 0
        y_true, y_pred = [], []
        records = []
        classes = None
        with torch.no_grad():
            for x, y, ids in loader:
                x = x.to(device)
                y = y.to(device).long()
                logits = self.recipe.eval_forward(model, x)
                loss_sum += float(crit(logits, y)) * int(y.size(0))
                seen += int(y.size(0))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.tolist())
                if return_records:
                    if classes is None:
                        classes = self._eval_classes
                    for rid, p_idx, prob in zip(ids, preds, probs):
                        prob = [float(v) for v in prob]
                        top = float(max(prob)) if prob else None
                        records.append({
                            "record_id": str(rid),
                            "prediction": classes[int(p_idx)],
                            "confidence": top,
                            "uncertainty": None if top is None else 1.0 - top,
                            "probabilities": prob,
                        })
        metrics = {
            "loss": (loss_sum / seen) if seen else None,
            "accuracy": float(accuracy_score(y_true, y_pred)) if seen else None,
            "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)) if seen else None,
        }
        return metrics, records

    def _snapshot(self, model):
        m = model.module if hasattr(model, "module") else model
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def _restore(self, model, state):
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(state)

    def _write_model_artifact(
        self,
        model,
        parts: Partitions,
        target: TargetSpec,
        *,
        split_spec_artifact_id: Optional[str] = None,
    ) -> Optional[str]:
        import torch

        num_classes = int(target.num_outputs)

        model_dir = ml_run_artifact_dir(
            self.run,
            kind="model",
        )

        checkpoint_path = model_dir / "model.pt"
        tmp_checkpoint_path = model_dir / "model.pt.tmp"
        manifest_path = model_dir / "model_manifest.json"

        m = model.module if hasattr(model, "module") else model

        architecture = str(self.run.params.get("architecture") or "resnet18")
        custom_model_import = str(self.run.params.get("custom_model_import") or "").strip()

        if architecture in {"resnet18", "resnet34"}:
            architecture_ref = f"torchvision.{architecture}"
        else:
            architecture_ref = architecture

        checkpoint_payload = {
            "schema_version": 2,
            "state_dict": m.state_dict(),
            "class_names": list(parts.train.classes),
            "num_classes": int(num_classes),
            "params": dict(self.run.params),
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
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
            "architecture": architecture_ref,
            "custom_model_import": custom_model_import,
            "epoch": self._best_epoch,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
            },
            "transform": {
                "image_size": 32,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010],
            },
        }

        # Atomic-ish write: write temp, then replace final checkpoint.
        torch.save(checkpoint_payload, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)

        manifest_payload = {
            "schema_version": 2,
            "kind": "torch_classifier",
            "framework": "torch",
            "task": self.recipe.task,
            "modality": self.recipe.modality,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "protocol_id": parts.protocol_id,
            "split_spec_artifact_id": split_spec_artifact_id,
            "created_at": time.time(),
            "class_names": list(parts.train.classes),
            "num_classes": int(num_classes),
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
                    "class_names": list(parts.train.classes),
                    "num_classes": int(num_classes),
                    "recipe_id": self.run.recipe_id,
                    "recipe_version": self.run.recipe_version,
                    "run_id": self.run.run_id,
                    "architecture": architecture_ref,
                    "custom_model_import": custom_model_import,
                    "image_size": 32,
                    "normalization": {
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2023, 0.1994, 0.2010],
                    },
                }
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
            "architecture": architecture_ref,
            "custom_model_import": custom_model_import,
            "metrics": {
                "best_score": self._best_score,
                "selection_metric": self.protocol.selection_metric,
                "best_epoch": self._best_epoch,
            },
        }

        manifest_path.write_text(
            json.dumps(
                json_safe(manifest_payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        self._eval_classes = list(parts.train.classes)

        return self.run.put_artifact(
            "ml.model",
            manifest_payload,
            params=self.run.params,
        )
