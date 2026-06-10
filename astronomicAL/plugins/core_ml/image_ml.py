from __future__ import annotations

import copy
import io
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ImageTrainConfig:
    dataset_id: str
    image_column: str
    target_column: str
    model_id: str
    validation_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    run_id: Optional[str] = None
    training_log_artifact_id: Optional[str] = None


def train_image_model(
    *,
    context: Any,
    registry: Any,
    config: ImageTrainConfig,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    _check_cancelled(cancel_token)

    run_id = config.run_id or uuid.uuid4().hex
    started = time.time()

    model_spec = registry.get_model(config.model_id)

    if getattr(model_spec, "modality", None) != "image":
        raise ValueError(
            f"Model `{model_spec.title}` has modality "
            f"`{getattr(model_spec, 'modality', 'unknown')}`. "
            "Use an image model definition."
        )

    if getattr(model_spec, "framework", None) != "torch":
        raise ValueError("The initial image backend only supports torch image models.")

    if model_spec.task != "classification":
        raise ValueError("The initial image backend only supports image classification.")

    params = dict(getattr(model_spec, "default_params", {}) or {})
    template_id = str((getattr(model_spec, "metadata", {}) or {}).get("template_id", ""))

    if "resnet" not in template_id:
        raise ValueError(
            f"Image model template `{template_id}` is not trainable yet. "
            "Start with ResNet-18 or ResNet-50 classifier definitions."
        )

    _publish_training_log_update(
        context,
        artifact_id=config.training_log_artifact_id,
        run_id=run_id,
        status="running",
        message="Loading image dataset...",
        epochs=[],
        best_epoch=None,
    )

    record_id_column = _mapped_column(context, config.dataset_id, "record_id")
    columns = [config.image_column, config.target_column]

    if record_id_column and record_id_column not in columns:
        columns.append(record_id_column)

    df = context.datasets.get_df(config.dataset_id, columns=columns)
    df = df.dropna(subset=[config.target_column])
    df = df[df[config.image_column].map(_has_image_value)].copy()

    if df.empty:
        raise ValueError("No usable rows remain after dropping missing targets/images.")

    row_ids = _row_ids(context, config.dataset_id, df)

    split = _split_image_dataframe(
        df=df,
        row_ids=row_ids,
        target_column=config.target_column,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
        stratify=config.stratify,
    )

    tuning_result = _optuna_tune_resnet_classifier(
        context=context,
        config=config,
        model_spec=model_spec,
        base_params=params,
        template_id=template_id,
        split=split,
        run_id=run_id,
        cancel_token=cancel_token,
    )

    if tuning_result.get("enabled"):
        params = {
            **params,
            **dict(tuning_result.get("best_params") or {}),
        }

    result = _train_resnet_classifier(
        context=context,
        config=config,
        model_spec=model_spec,
        params=params,
        template_id=template_id,
        split=split,
        run_id=run_id,
        cancel_token=cancel_token,
    )

    result.setdefault("extra", {})["tuning"] = tuning_result
    result.setdefault("model_metadata", {})["tuning"] = tuning_result

    finished = time.time()

    image_spec_payload = {
        "run_id": run_id,
        "modality": "image",
        "dataset_id": config.dataset_id,
        "image_column": config.image_column,
        "target_column": config.target_column,
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "params": params,
        "tuning": tuning_result,
    }

    split_payload = {
        "run_id": run_id,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "stratify": bool(config.stratify),
        "train_row_ids": split["rows_train"],
        "validation_row_ids": split["rows_val"],
        "test_row_ids": split["rows_test"],
    }

    model_payload = {
        "run_id": run_id,
        "model": result["model"],
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "modality": "image",
        "task": "classification",
        "dataset_id": config.dataset_id,
        "image_column": config.image_column,
        "target_column": config.target_column,
        "created_at": finished,
        "metadata": result.get("model_metadata", {}),
    }

    evaluation_payload = {
        "run_id": run_id,
        "task": "classification",
        "modality": "image",
        "metrics": result["metrics"],
        "extra": result.get("extra", {}),
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
    }

    predictions_payload = {
        "run_id": run_id,
        "task": "classification",
        "modality": "image",
        "records": result["prediction_records"],
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
    }

    training_log_payload = {
        "run_id": run_id,
        "status": "finished",
        "message": "Image training complete.",
        "task": "classification",
        "modality": "image",
        "model_id": model_spec.id,
        "model_title": model_spec.title,
        "framework": model_spec.framework,
        "dataset_id": config.dataset_id,
        "optimize_metric": str(params.get("optimize_metric", "val_loss")),
        "best_epoch": result.get("best_epoch"),
        "epochs": result.get("training_log", []),
        "finished_at": finished,
        "updated_at": finished,
        "tuning": tuning_result,
        "tuning_trials": tuning_result.get("trials", []),
    }

    training_log_artifact_id = config.training_log_artifact_id

    if training_log_artifact_id:
        _update_training_log_artifact(context, training_log_artifact_id, training_log_payload)
    else:
        training_log_artifact_id = _put(
            context,
            "ml.training_log",
            training_log_payload,
            config.dataset_id,
        )

    artifact_ids = {
        "image_spec": _put(context, "ml.image_spec", image_spec_payload, config.dataset_id),
        "split_spec": _put(context, "ml.split_spec", split_payload, config.dataset_id),
        "model": _put(context, "ml.model", model_payload, config.dataset_id),
        "evaluation": _put(context, "ml.evaluation_report", evaluation_payload, config.dataset_id),
        "predictions": _put(
            context,
            "ml.predictions",
            predictions_payload,
            config.dataset_id,
            split["rows_test"],
        ),
        "training_log": training_log_artifact_id,
    }

    run_payload = {
        "run_id": run_id,
        "status": "finished",
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": finished - started,
        "config": asdict(config),
        "metrics": result["metrics"],
        "artifact_ids": dict(artifact_ids),
    }

    artifact_ids["run"] = _put(context, "ml.run", run_payload, config.dataset_id)

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": training_log_artifact_id,
            "run_id": run_id,
            "status": "finished",
            "message": "Image training complete.",
        },
    )

    _publish(
        context,
        "ml.run.finished",
        {
            "run_id": run_id,
            "artifact_ids": artifact_ids,
            "training_log_artifact_id": training_log_artifact_id,
        },
    )

    return {
        "run_id": run_id,
        "metrics": result["metrics"],
        "extra": result.get("extra", {}),
        "prediction_preview": result["prediction_records"][:25],
        "artifact_ids": artifact_ids,
    }


def _train_resnet_classifier(
    *,
    context: Any,
    config: ImageTrainConfig,
    model_spec: Any,
    params: Dict[str, Any],
    template_id: str,
    split: Dict[str, Any],
    run_id: str,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    model = None

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
        from torch.utils.data import DataLoader
        from torchvision import transforms

        torch.manual_seed(int(config.random_state))

        image_size = int(params.get("image_size", 224))
        batch_size = int(params.get("batch_size", 32))
        epochs = int(params.get("epochs", 20))
        learning_rate = float(params.get("learning_rate", 3e-4))
        weight_decay = float(params.get("weight_decay", 1e-4))
        patience = int(params.get("patience", 8))
        optimize_metric = str(params.get("optimize_metric", "val_loss"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        label_encoder = LabelEncoder()
        all_labels = pd.concat(
            [
                split["train_df"][config.target_column],
                split["val_df"][config.target_column],
                split["test_df"][config.target_column],
            ],
            ignore_index=True,
        )
        label_encoder.fit(all_labels)

        class_names = [str(c) for c in label_encoder.classes_]
        num_classes = len(class_names)

        if num_classes < 2:
            raise ValueError("Image classification needs at least two classes.")

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        train_dataset = _ImageClassificationDataset(
            df=split["train_df"],
            rows=split["rows_train"],
            image_column=config.image_column,
            target_column=config.target_column,
            label_encoder=label_encoder,
            transform=transform,
        )
        val_dataset = _ImageClassificationDataset(
            df=split["val_df"],
            rows=split["rows_val"],
            image_column=config.image_column,
            target_column=config.target_column,
            label_encoder=label_encoder,
            transform=transform,
        )
        test_dataset = _ImageClassificationDataset(
            df=split["test_df"],
            rows=split["rows_test"],
            image_column=config.image_column,
            target_column=config.target_column,
            label_encoder=label_encoder,
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        _publish_training_log_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            status="running",
            message="Building ResNet model...",
            epochs=[],
            best_epoch=None,
        )

        model, pretrained_loaded = _build_resnet_model(
            template_id=template_id,
            num_classes=num_classes,
            params=params,
            context=context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_score = None
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        training_log: List[Dict[str, Any]] = []

        _publish_training_log_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            status="running",
            message="Image training initialised. Waiting for first epoch...",
            epochs=training_log,
            best_epoch=best_epoch,
        )

        for epoch in range(1, epochs + 1):
            _check_cancelled(cancel_token)

            model.train()
            train_losses: List[float] = []
            train_correct = 0
            train_total = 0

            for images, targets, _row_ids in train_loader:
                _check_cancelled(cancel_token)

                images = images.to(device, non_blocking=False)
                targets = targets.to(device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().item()))

                preds = torch.argmax(logits, dim=1)
                train_correct += int((preds == targets).sum().detach().cpu().item())
                train_total += int(targets.shape[0])

                del images, targets, logits, loss, preds

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_accuracy = float(train_correct / train_total) if train_total else float("nan")

            val_eval = _evaluate_image_classifier(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                label_encoder=label_encoder,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_eval["loss"],
                "val_accuracy": val_eval["metrics"]["accuracy"],
                "val_f1_macro": val_eval["metrics"]["f1_macro"],
            }
            training_log.append(row)

            score = _score_for_optimisation(row, optimize_metric)
            if _is_better(score, best_score, optimize_metric):
                best_score = score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            _publish_training_log_update(
                context,
                artifact_id=config.training_log_artifact_id,
                run_id=run_id,
                status="running",
                message=f"Finished epoch {epoch}.",
                epochs=training_log,
                best_epoch=best_epoch,
            )

            if patience > 0 and epochs_without_improvement >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        _publish_training_log_update(
            context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
            status="running",
            message="Evaluating best image model on validation/test sets...",
            epochs=training_log,
            best_epoch=best_epoch,
        )

        val_eval = _evaluate_image_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            label_encoder=label_encoder,
        )
        test_eval = _evaluate_image_classifier(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            label_encoder=label_encoder,
        )

        metrics = _prefix_metrics("val", val_eval["metrics"])
        metrics.update(_prefix_metrics("test", test_eval["metrics"]))
        metrics["val_loss"] = float(val_eval["loss"])
        metrics["test_loss"] = float(test_eval["loss"])
        metrics["best_epoch"] = float(best_epoch)

        return {
            "model": {
                "torch_model": model,
                "label_encoder": label_encoder,
                "class_names": class_names,
                "image_size": image_size,
                "template_id": template_id,
            },
            "metrics": metrics,
            "extra": {
                "validation": val_eval.get("extra", {}),
                "test": test_eval.get("extra", {}),
            },
            "prediction_records": test_eval["prediction_records"],
            "training_log": training_log,
            "best_epoch": best_epoch,
            "model_metadata": {
                "framework": "torch",
                "modality": "image",
                "template_id": template_id,
                "image_size": image_size,
                "class_names": class_names,
                "device": str(device),
                "pretrained_requested": bool(params.get("pretrained", True)),
                "pretrained_loaded": bool(pretrained_loaded),
            },
        }

    except Exception as exc:
        if _is_cancel_exception(exc):
            _publish_training_log_update(
                context,
                artifact_id=config.training_log_artifact_id,
                run_id=run_id,
                status="cancelled",
                message="Image training cancelled. Releasing GPU memory...",
                epochs=None,
                best_epoch=None,
            )
        raise

    finally:
        _release_torch_cuda(model)


class _ImageClassificationDataset:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        rows: List[Any],
        image_column: str,
        target_column: str,
        label_encoder: Any,
        transform: Any,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.rows = list(rows)
        self.image_column = image_column
        self.target_column = target_column
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        image = _load_image(self.df.iloc[index][self.image_column])
        image = self.transform(image)

        label = self.df.iloc[index][self.target_column]
        target = int(self.label_encoder.transform([label])[0])

        return image, target, self.rows[index]


def _build_resnet_model(
    *,
    template_id: str,
    num_classes: int,
    params: Dict[str, Any],
    context: Any,
    artifact_id: Optional[str],
    run_id: str,
):
    import torch.nn as nn
    from torchvision import models

    pretrained = bool(params.get("pretrained", True))
    freeze_backbone = bool(params.get("freeze_backbone", False))

    arch = "resnet50" if "resnet50" in template_id else "resnet18"

    def make(pretrained_flag: bool):
        if arch == "resnet50":
            weights_cls = getattr(models, "ResNet50_Weights", None)
            weights = weights_cls.DEFAULT if pretrained_flag and weights_cls is not None else None
            try:
                return models.resnet50(weights=weights)
            except TypeError:
                return models.resnet50(pretrained=pretrained_flag)

        weights_cls = getattr(models, "ResNet18_Weights", None)
        weights = weights_cls.DEFAULT if pretrained_flag and weights_cls is not None else None
        try:
            return models.resnet18(weights=weights)
        except TypeError:
            return models.resnet18(pretrained=pretrained_flag)

    pretrained_loaded = False

    try:
        model = make(pretrained)
        pretrained_loaded = pretrained
    except Exception as exc:
        if not pretrained:
            raise

        _publish_training_log_update(
            context,
            artifact_id=artifact_id,
            run_id=run_id,
            status="running",
            message=(
                f"Could not load pretrained weights for {arch}: {exc}. "
                "Continuing with random initialisation."
            ),
            epochs=None,
            best_epoch=None,
        )
        model = make(False)
        pretrained_loaded = False

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model, pretrained_loaded


def _evaluate_image_classifier(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    label_encoder: Any,
) -> Dict[str, Any]:
    import torch
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()

    losses: List[float] = []
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    y_proba_rows: List[np.ndarray] = []
    row_ids: List[Any] = []

    with torch.no_grad():
        for images, targets, batch_rows in loader:

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)

            losses.append(float(loss.detach().cpu().item()))
            y_true_idx.extend([int(x) for x in targets.detach().cpu().numpy().tolist()])
            y_pred_idx.extend([int(x) for x in preds.detach().cpu().numpy().tolist()])
            y_proba_rows.extend([row for row in proba.detach().cpu().numpy()])
            row_ids.extend([_json_value(x) for x in batch_rows])

    classes = [str(c) for c in label_encoder.classes_]
    y_true = label_encoder.inverse_transform(np.asarray(y_true_idx, dtype=int))
    y_pred = label_encoder.inverse_transform(np.asarray(y_pred_idx, dtype=int))
    y_proba = np.asarray(y_proba_rows, dtype=float)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        if y_proba.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
    except Exception:
        pass

    extra = {
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    prediction_records = []
    for i, row_id in enumerate(row_ids):
        record = {
            "row_id": _json_value(row_id),
            "y_true": _json_value(y_true[i]),
            "y_pred": _json_value(y_pred[i]),
            "confidence": float(np.max(y_proba[i])),
        }

        for j, label in enumerate(classes):
            record[f"proba_{label}"] = float(y_proba[i, j])

        prediction_records.append(record)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "metrics": metrics,
        "extra": extra,
        "prediction_records": prediction_records,
    }


def _split_image_dataframe(
    *,
    df: pd.DataFrame,
    row_ids: List[Any],
    target_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
    stratify: bool,
) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split

    y = df[target_column]
    stratify_full = _stratify_or_none(y, enabled=stratify)

    remaining_df, test_df, remaining_rows, test_rows = train_test_split(
        df,
        row_ids,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify_full,
    )

    validation_fraction_of_remaining = float(validation_size) / (1.0 - float(test_size))
    stratify_remaining = _stratify_or_none(
        remaining_df[target_column],
        enabled=stratify,
    )

    train_df, val_df, train_rows, val_rows = train_test_split(
        remaining_df,
        remaining_rows,
        test_size=validation_fraction_of_remaining,
        random_state=int(random_state),
        stratify=stratify_remaining,
    )

    return {
        "train_df": train_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "rows_train": [_json_value(x) for x in train_rows],
        "rows_val": [_json_value(x) for x in val_rows],
        "rows_test": [_json_value(x) for x in test_rows],
    }


def _load_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, np.ndarray):
        arr = value
        if arr.dtype.kind == "f" and np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.asarray(arr, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _load_image(value["bytes"])
        if value.get("path") is not None:
            return _load_image(value["path"])

    if isinstance(value, (str, Path)):
        path_or_url = str(value).strip()

        if path_or_url.startswith(("http://", "https://")):
            with urllib.request.urlopen(path_or_url, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")

        return Image.open(path_or_url).convert("RGB")

    raise ValueError(f"Unsupported image value type: {type(value)!r}")


def _has_image_value(value: Any) -> bool:
    if value is None:
        return False

    try:
        if isinstance(value, float) and np.isnan(value):
            return False
    except Exception:
        pass

    return True


def _stratify_or_none(y: pd.Series, *, enabled: bool):
    if not enabled:
        return None

    counts = y.value_counts(dropna=False)
    if len(counts) <= 1:
        return None
    if counts.min() < 2:
        return None

    return y


def _mapped_column(context: Any, dataset_id: str, mapping_name: str) -> Optional[str]:
    try:
        return context.datasets.get_mapping(dataset_id, mapping_name)
    except Exception:
        return None


def _row_ids(context: Any, dataset_id: str, df: pd.DataFrame) -> List[Any]:
    record_id_column = _mapped_column(context, dataset_id, "record_id")

    if record_id_column and record_id_column in df.columns:
        return [_json_value(v) for v in df[record_id_column].tolist()]

    return [_json_value(v) for v in df.index.tolist()]


def _json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _put(
    context: Any,
    artifact_type: str,
    payload: Dict[str, Any],
    dataset_id: Optional[str],
    row_ids: Optional[List[Any]] = None,
) -> Optional[str]:
    artifacts = getattr(context, "artifacts", None)
    put = getattr(artifacts, "put", None)

    if not callable(put):
        return None

    return put(
        artifact_type,
        payload,
        dataset_id=dataset_id,
        row_ids=row_ids,
        params={"run_id": payload.get("run_id")},
    )


def _publish(context: Any, topic: str, payload: Dict[str, Any]) -> None:
    events = getattr(context, "events", None)
    publish = getattr(events, "publish", None)

    if callable(publish):
        publish(topic, payload)


def _update_training_log_artifact(
    context: Any,
    artifact_id: Optional[str],
    updates: Dict[str, Any],
) -> None:
    if not artifact_id:
        return

    artifacts = getattr(context, "artifacts", None)
    get = getattr(artifacts, "get", None)

    if not callable(get):
        return

    try:
        payload = get(artifact_id)
    except Exception:
        return

    if isinstance(payload, dict):
        payload.update(updates)


def _publish_training_log_update(
    context: Any,
    *,
    artifact_id: Optional[str],
    run_id: str,
    status: str,
    message: str,
    epochs: Optional[List[Dict[str, Any]]] = None,
    best_epoch: Optional[int] = None,
) -> None:
    if not artifact_id:
        return

    now = time.time()

    updates: Dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at": now,
    }

    if epochs is not None:
        updates["epochs"] = list(epochs)
        updates["last_epoch"] = epochs[-1].get("epoch") if epochs else None

    if best_epoch is not None:
        updates["best_epoch"] = best_epoch

    _update_training_log_artifact(context, artifact_id, updates)

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "status": status,
            "message": message,
            "best_epoch": best_epoch,
            "last_epoch": updates.get("last_epoch"),
        },
    )

def _tuning_config(model_spec: Any) -> Dict[str, Any]:
    metadata = getattr(model_spec, "metadata", {}) or {}
    tuning = metadata.get("tuning") or {}

    if not isinstance(tuning, dict):
        return {}

    if not tuning.get("enabled"):
        return {}

    if tuning.get("backend") != "optuna":
        return {}

    if not tuning.get("search_space"):
        return {}

    return tuning


def _optuna_direction(metric: str) -> str:
    metric = metric.lower()

    if (
        "loss" in metric
        or "error" in metric
        or "mae" in metric
        or "mse" in metric
        or "rmse" in metric
    ):
        return "minimize"

    return "maximize"


def _make_optuna_sampler(name: str, random_state: int):
    import optuna

    if name == "random":
        return optuna.samplers.RandomSampler(seed=int(random_state))

    return optuna.samplers.TPESampler(seed=int(random_state))


def _sample_optuna_params(trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}

    for name, spec in search_space.items():
        kind = spec.get("type")

        if kind == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            log = bool(spec.get("log", False))

            if high < low:
                low, high = high, low

            if log:
                sampled[name] = trial.suggest_int(name, max(1, low), max(1, high), log=True)
            else:
                step = int(spec.get("step", 1) or 1)
                sampled[name] = trial.suggest_int(name, low, high, step=max(1, step))

        elif kind == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))

            if high < low:
                low, high = high, low

            if log:
                low = max(low, 1e-12)
                high = max(high, low * 10.0)

            sampled[name] = trial.suggest_float(name, low, high, log=log)

        elif kind == "categorical":
            choices = list(spec.get("choices", []))
            if choices:
                sampled[name] = trial.suggest_categorical(name, choices)

    return sampled


def _extract_objective_value(metrics: Dict[str, float], metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])

    aliases = {
        "val_loss": ["loss"],
        "val_accuracy": ["accuracy"],
        "val_balanced_accuracy": ["balanced_accuracy"],
        "val_f1_macro": ["f1_macro"],
        "val_roc_auc": ["roc_auc", "val_roc_auc_ovr_macro", "roc_auc_ovr_macro"],
    }

    for alias in aliases.get(metric, []):
        if alias in metrics:
            return float(metrics[alias])

    raise ValueError(
        f"Optuna metric `{metric}` was not produced by image evaluation. "
        f"Available metrics: {sorted(metrics.keys())}"
    )


def _publish_tuning_update(
    context: Any,
    *,
    artifact_id: Optional[str],
    run_id: str,
    message: str,
    trials: List[Dict[str, Any]],
    best_params: Optional[Dict[str, Any]] = None,
    best_value: Optional[float] = None,
) -> None:
    if not artifact_id:
        return

    try:
        payload = context.artifacts.get(artifact_id)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        payload.update(
            {
                "status": "tuning",
                "message": message,
                "tuning_trials": list(trials),
                "tuning_best_params": dict(best_params or {}),
                "tuning_best_value": best_value,
                "updated_at": time.time(),
            }
        )

    _publish(
        context,
        "ml.training_log.updated",
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "status": "tuning",
            "message": message,
        },
    )


def _optuna_tune_resnet_classifier(
    *,
    context: Any,
    config: ImageTrainConfig,
    model_spec: Any,
    base_params: Dict[str, Any],
    template_id: str,
    split: Dict[str, Any],
    run_id: str,
    cancel_token: Any = None,
) -> Dict[str, Any]:
    tuning = _tuning_config(model_spec)

    if not tuning:
        return {
            "enabled": False,
            "backend": "optuna",
            "best_params": {},
            "best_value": None,
            "trials": [],
        }

    import optuna

    metric = str(tuning.get("metric", "val_f1_macro"))
    direction = _optuna_direction(metric)
    search_space = dict(tuning.get("search_space") or {})
    n_trials = int(tuning.get("n_trials", 10) or 10)
    timeout_seconds = tuning.get("timeout_seconds")
    sampler_name = str(tuning.get("sampler", "tpe"))

    # Image tuning is expensive. Unless explicitly configured later, use short
    # trial training and full training only for the final selected params.
    base_epochs = int(base_params.get("epochs", 20))
    trial_epochs = int(tuning.get("trial_epochs") or min(3, max(1, base_epochs)))

    sampler = _make_optuna_sampler(sampler_name, config.random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)

    trial_records: List[Dict[str, Any]] = []

    _publish_tuning_update(
        context,
        artifact_id=config.training_log_artifact_id,
        run_id=run_id,
        message=(
            f"Starting Optuna image tuning: {n_trials} trials, "
            f"{trial_epochs} epoch(s) per trial, metric `{metric}`."
        ),
        trials=trial_records,
    )

    def objective(trial: Any) -> float:
        _check_cancelled(cancel_token)

        trial_params = _sample_optuna_params(trial, search_space)
        params = {
            **base_params,
            **trial_params,
            "epochs": trial_epochs,
        }

        try:
            metrics = _train_resnet_trial(
                context=context,
                config=config,
                params=params,
                template_id=template_id,
                split=split,
                run_id=run_id,
                trial_number=int(trial.number),
                objective_metric=metric,
                cancel_token=cancel_token,
            )

            value = _extract_objective_value(metrics, metric)

            record = {
                "number": int(trial.number),
                "state": "complete",
                "value": float(value),
                "metric": metric,
                "params": dict(trial_params),
                "metrics": dict(metrics),
            }
            trial_records.append(record)

            _publish_tuning_update(
                context,
                artifact_id=config.training_log_artifact_id,
                run_id=run_id,
                message=f"Finished Optuna image trial {trial.number + 1}/{n_trials}.",
                trials=trial_records,
            )

            return float(value)

        except Exception as exc:
            if _is_cancel_exception(exc):
                raise

            record = {
                "number": int(trial.number),
                "state": "failed",
                "value": None,
                "metric": metric,
                "params": dict(trial_params),
                "error": str(exc),
            }
            trial_records.append(record)

            _publish_tuning_update(
                context,
                artifact_id=config.training_log_artifact_id,
                run_id=run_id,
                message=f"Optuna image trial {trial.number + 1}/{n_trials} failed: {exc}",
                trials=trial_records,
            )

            raise optuna.TrialPruned()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
    )

    completed = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state.name == "COMPLETE"
    ]

    if not completed:
        raise RuntimeError("Optuna image tuning did not complete any valid trials.")

    best_params = dict(study.best_trial.params)
    best_value = float(study.best_value)

    _publish_tuning_update(
        context,
        artifact_id=config.training_log_artifact_id,
        run_id=run_id,
        message=f"Optuna image tuning complete. Best {metric}={best_value:.6g}.",
        trials=trial_records,
        best_params=best_params,
        best_value=best_value,
    )

    return {
        "enabled": True,
        "backend": "optuna",
        "metric": metric,
        "direction": direction,
        "n_trials": n_trials,
        "trial_epochs": trial_epochs,
        "best_params": best_params,
        "best_value": best_value,
        "trials": trial_records,
    }

def _train_resnet_trial(
    *,
    context: Any,
    config: ImageTrainConfig,
    params: Dict[str, Any],
    template_id: str,
    split: Dict[str, Any],
    run_id: str,
    trial_number: int,
    objective_metric: str,
    cancel_token: Any = None,
) -> Dict[str, float]:
    model = None

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
        from torch.utils.data import DataLoader
        from torchvision import transforms

        _check_cancelled(cancel_token)

        torch.manual_seed(int(config.random_state) + int(trial_number))

        image_size = int(params.get("image_size", 224))
        batch_size = int(params.get("batch_size", 32))
        epochs = int(params.get("epochs", 3))
        learning_rate = float(params.get("learning_rate", 3e-4))
        weight_decay = float(params.get("weight_decay", 1e-4))
        optimize_metric = str(params.get("optimize_metric", objective_metric))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        label_encoder = LabelEncoder()
        all_labels = pd.concat(
            [
                split["train_df"][config.target_column],
                split["val_df"][config.target_column],
                split["test_df"][config.target_column],
            ],
            ignore_index=True,
        )
        label_encoder.fit(all_labels)

        num_classes = int(len(label_encoder.classes_))

        if num_classes < 2:
            raise ValueError("Image classification needs at least two classes.")

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        train_dataset = _ImageClassificationDataset(
            df=split["train_df"],
            rows=split["rows_train"],
            image_column=config.image_column,
            target_column=config.target_column,
            label_encoder=label_encoder,
            transform=transform,
        )
        val_dataset = _ImageClassificationDataset(
            df=split["val_df"],
            rows=split["rows_val"],
            image_column=config.image_column,
            target_column=config.target_column,
            label_encoder=label_encoder,
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model, _pretrained_loaded = _build_resnet_model(
            template_id=template_id,
            num_classes=num_classes,
            params=params,
            context=context,
            artifact_id=config.training_log_artifact_id,
            run_id=run_id,
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_score = None
        best_metrics: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            _check_cancelled(cancel_token)

            model.train()
            train_losses: List[float] = []
            train_correct = 0
            train_total = 0

            for images, targets, _row_ids in train_loader:
                _check_cancelled(cancel_token)

                images = images.to(device, non_blocking=False)
                targets = targets.to(device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().item()))

                preds = torch.argmax(logits, dim=1)
                train_correct += int((preds == targets).sum().detach().cpu().item())
                train_total += int(targets.shape[0])

                del images, targets, logits, loss, preds

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_accuracy = float(train_correct / train_total) if train_total else float("nan")

            val_eval = _evaluate_image_classifier(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                label_encoder=label_encoder,
            )

            row = {
                "val_loss": float(val_eval["loss"]),
                "val_accuracy": float(val_eval["metrics"]["accuracy"]),
                "val_balanced_accuracy": float(val_eval["metrics"].get("balanced_accuracy", float("nan"))),
                "val_f1_macro": float(val_eval["metrics"]["f1_macro"]),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "trial_epoch": float(epoch),
            }

            if "roc_auc" in val_eval["metrics"]:
                row["val_roc_auc"] = float(val_eval["metrics"]["roc_auc"])
            if "roc_auc_ovr_macro" in val_eval["metrics"]:
                row["val_roc_auc_ovr_macro"] = float(val_eval["metrics"]["roc_auc_ovr_macro"])

            score = _score_for_optimisation(row, optimize_metric)

            if _is_better(score, best_score, optimize_metric):
                best_score = score
                best_metrics = dict(row)

        if not best_metrics:
            raise RuntimeError("Trial produced no validation metrics.")

        return best_metrics

    finally:
        _release_torch_cuda(model)

def _check_cancelled(cancel_token: Any) -> None:
    if cancel_token is None:
        return

    is_cancelled = getattr(cancel_token, "is_cancelled", None)
    if callable(is_cancelled) and is_cancelled():
        raise RuntimeError("Training cancelled.")

    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled):
        if cancelled():
            raise RuntimeError("Training cancelled.")
        return

    if isinstance(cancelled, bool) and cancelled:
        raise RuntimeError("Training cancelled.")


def _score_for_optimisation(row: Dict[str, Any], metric: str) -> Optional[float]:
    value = row.get(metric)
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _is_better(new_score: Optional[float], best_score: Optional[float], metric: str) -> bool:
    if new_score is None:
        return False
    if best_score is None:
        return True

    lowered = metric.lower()
    minimise = "loss" in lowered or "error" in lowered or "mae" in lowered or "rmse" in lowered

    if minimise:
        return new_score < best_score

    return new_score > best_score


def _prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}

def _release_torch_cuda(model: Any = None) -> None:
    """Best-effort torch cleanup for completion, error, or cancellation."""

    try:
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass

        import gc
        import torch

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

def _is_cancel_exception(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "cancel" in text or "cancelled" in text or "canceled" in text