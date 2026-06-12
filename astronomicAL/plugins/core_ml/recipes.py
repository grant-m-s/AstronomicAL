from __future__ import annotations

import importlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .recipe_registry import (
    MLRecipe,
    MLRunContext,
    get_dataset_frame,
    infer_recipe_params,
    json_safe,
    mapped_column,
)


def _import_object(path: str):
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


class ExternalPythonRecipe(MLRecipe):
    """Bridge recipe for expert-owned code.

    The imported object may be:
    - an MLRecipe subclass,
    - an MLRecipe instance,
    - a callable accepting MLRunContext and returning a dict.
    """

    id = "core.ml.external_python_recipe"
    title = "External Python ML recipe"
    version = "0.1.0"
    task = "custom"
    modality = "custom"
    complexity = "expert"
    author = "AstronomicAL"
    description = (
        "Run a recipe implemented in an installed/local Python module. "
        "Use this for expert training loops that cannot be represented as panel widgets."
    )
    tags = ["expert", "python", "extension", "recipe"]
    required_mappings: List[str] = []
    optional_mappings = ["record_id", "target_label", "image.path", "image.uri"]
    produces = ["ml.run", "ml.training_log"]

    params_schema = {
        "type": "object",
        "required": ["import_path"],
        "properties": {
            "import_path": {
                "type": "string",
                "title": "Import path",
                "description": "Dotted path to an MLRecipe subclass/instance or callable.",
            },
            "kwargs_json": {
                "type": "string",
                "title": "Extra kwargs JSON",
                "default": "{}",
                "description": "Optional JSON object merged into run.params before execution.",
            },
        },
    }

    def run(self, run: MLRunContext) -> Dict[str, Any]:
        import_path = str(run.params.get("import_path") or "").strip()
        if not import_path:
            raise ValueError("ExternalPythonRecipe requires import_path.")

        kwargs_json = str(run.params.get("kwargs_json") or "{}").strip() or "{}"
        try:
            extra = json.loads(kwargs_json)
        except Exception as exc:
            raise ValueError(f"kwargs_json is not valid JSON: {exc}") from exc

        if not isinstance(extra, dict):
            raise ValueError("kwargs_json must decode to a JSON object.")

        run.params.update(extra)

        obj = _import_object(import_path)

        run.log(message=f"Loaded external recipe object `{import_path}`.")

        if isinstance(obj, type) and issubclass(obj, MLRecipe):
            return obj().run(run)

        if isinstance(obj, MLRecipe):
            return obj.run(run)

        if callable(obj):
            result = obj(run)
            return dict(result or {})

        raise TypeError(
            "import_path must point to an MLRecipe subclass, MLRecipe instance, "
            "or callable accepting MLRunContext."
        )


class CIFARStyleImageClassifierRecipe(MLRecipe):
    """Torch image classifier recipe with CIFAR-style transforms and scheduler.

    This is not a visual model builder. It is a code-backed recipe that exposes
    a small typed surface to the panel while keeping the training loop in Python.
    """

    id = "core.ml.cifar_style_image_classifier"
    title = "CIFAR-style Torch image classifier"
    version = "0.1.0"
    task = "classification"
    modality = "image"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Train an image classifier using a real PyTorch loop with configurable "
        "transforms, SGD/AdamW, schedulers, AMP, checkpointing, and prediction artifacts. "
        "Use custom_model_import to point at kuangliu/pytorch-cifar-style model code."
    )
    tags = ["torch", "image", "classification", "cifar", "transforms", "scheduler"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label", "image.path", "image.uri"]
    produces = [
        "ml.image_spec",
        "ml.split_spec",
        "ml.model",
        "ml.evaluation_report",
        "ml.predictions",
        "ml.training_log",
        "ml.run",
    ]

    params_schema = {
        "type": "object",
        "required": ["image_column", "target_column"],
        "properties": {
            "image_column": {
                "type": "string",
                "title": "Image path/URI column",
                "default": "",
                "description": "Blank means infer from dataset mappings/columns.",
            },
            "target_column": {
                "type": "string",
                "title": "Label column",
                "default": "",
                "description": "Blank means infer from dataset mappings/columns.",
            },
            "record_id_column": {
                "type": "string",
                "title": "Record ID column",
                "default": "",
                "description": "Blank means infer from dataset mappings/columns.",
            },
            "validation_source": {
                "type": "string",
                "title": "Validation source",
                "enum": ["split", "dataset"],
                "default": "split",
                "description": (
                    "How to provide validation data. "
                    "`split` reserves validation_size from the selected training dataset. "
                    "`dataset` uses validation_dataset_id."
                ),
            },
            "validation_size": {
                "type": "number",
                "title": "Validation split",
                "default": 0.1,
                "minimum": 0.01,
                "maximum": 0.5,
                "description": (
                    "Fraction of the selected training dataset to reserve for validation. "
                    "Used only when Validation source = split. Must be > 0 and <= 0.5."
                ),
            },
            "validation_dataset_id": {
                "type": "string",
                "title": "Validation dataset",
                "default": "",
                "x-widget": "dataset_select",
                "description": "Required when Validation source = dataset.",
            },
            "test_source": {
                "type": "string",
                "title": "Test source",
                "enum": ["none", "split", "dataset"],
                "default": "none",
                "description": (
                    "Optional test data. `none` skips test evaluation. "
                    "`split` reserves test_size from the selected training dataset. "
                    "`dataset` uses test_dataset_id."
                ),
            },
            "test_size": {
                "type": "number",
                "title": "Test split",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 0.3,
                "description": (
                    "Fraction of the selected training dataset to reserve for test. "
                    "Used only when Test source = split. Must be > 0 and <= 0.3."
                ),
            },
            "test_dataset_id": {
                "type": "string",
                "title": "Test dataset",
                "default": "",
                "x-widget": "dataset_select",
                "description": "Required when Test source = dataset.",
            },
            "architecture": {
                "type": "string",
                "title": "Architecture",
                "enum": [
                    "torchvision.resnet18",
                    "torchvision.resnet34",
                    "torchvision.mobilenet_v3_small",
                    "custom_import",
                ],
                "default": "torchvision.resnet18",
            },
            "custom_model_import": {
                "type": "string",
                "title": "Custom model import path",
                "default": "",
                "description": (
                    "Dotted callable path, e.g. recipes.pytorch_cifar.models.ResNet18. "
                    "The callable should return an nn.Module."
                ),
            },
            "image_size": {
                "type": "integer",
                "title": "Image size",
                "default": 32,
                "minimum": 8,
            },
            "epochs": {
                "type": "integer",
                "title": "Epochs",
                "default": 200,
                "minimum": 1,
            },
            "batch_size": {
                "type": "integer",
                "title": "Batch size",
                "default": 128,
                "minimum": 1,
            },
            "num_workers": {
                "type": "integer",
                "title": "DataLoader workers",
                "default": 2,
                "minimum": 0,
                        },
            "learning_rate": {
                "type": "number",
                "title": "Learning rate",
                "default": 0.1,
                "minimum": 0.0,
            },
            "optimize_metric": {
                "type": "string",
                "title": "Optimise metric",
                "default": "val_accuracy",
                "description": "Metric used by the curves panel to identify the best epoch.",
            },
            "optimizer": {
                "type": "string",
                "title": "Optimizer",
                "enum": ["sgd", "adamw"],
                "default": "sgd",
            },
            "momentum": {
                "type": "number",
                "title": "SGD momentum",
                "default": 0.9,
            },
            "weight_decay": {
                "type": "number",
                "title": "Weight decay",
                "default": 0.0005,
                "minimum": 0.0,
            },
            "scheduler": {
                "type": "string",
                "title": "Scheduler",
                "enum": ["cosine", "step", "none"],
                "default": "cosine",
            },
            "cosine_t_max": {
                "type": "integer",
                "title": "Cosine T_max",
                "default": 200,
                "minimum": 1,
            },
            "step_size": {
                "type": "integer",
                "title": "StepLR step size",
                "default": 60,
                "minimum": 1,
            },
            "gamma": {
                "type": "number",
                "title": "StepLR gamma",
                "default": 0.2,
            },
            "augmentation_preset": {
                "type": "string",
                "title": "Augmentation preset",
                "enum": ["cifar_standard", "none"],
                "default": "cifar_standard",
            },
            "normalization": {
                "type": "string",
                "title": "Normalization",
                "enum": ["cifar10", "imagenet", "none"],
                "default": "cifar10",
            },
            "random_state": {
                "type": "integer",
                "title": "Random seed",
                "default": 42,
            },
            "device": {
                "type": "string",
                "title": "Device",
                "enum": ["auto", "cpu", "cuda", "mps"],
                "default": "auto",
            },
            "amp": {
                "type": "boolean",
                "title": "Use automatic mixed precision when available",
                "default": False,
                "description": (
                    "Disabled by default because AMP can produce non-finite losses for "
                    "some recipe/model/device combinations. Enable only after fp32 "
                    "training produces stable losses."
                ),
            },
            "data_parallel": {
                "type": "boolean",
                "title": "Use DataParallel when multiple CUDA GPUs are available",
                "default": True,
            },
            "limit_rows": {
                "type": "integer",
                "title": "Limit rows, 0 = all",
                "default": 0,
                "minimum": 0,
            },
        },
    }

    def run(self, run: MLRunContext) -> Dict[str, Any]:
        torch, nn, optim, DataLoader, Dataset, Image, transforms = self._imports()

        params = dict(run.params)

        params.setdefault("framework", "torch")
        params.setdefault("model_title", params.get("architecture") or self.title)
        params.setdefault("optimize_metric", "val_accuracy")
        run.params.update(params)

        validation_source = str(params.get("validation_source") or "split").strip()
        test_source = str(params.get("test_source") or "none").strip()

        if validation_source not in {"split", "dataset"}:
            raise ValueError(
                "validation_source must be either `split` or `dataset`."
            )

        if test_source not in {"none", "split", "dataset"}:
            raise ValueError(
                "test_source must be one of `none`, `split`, or `dataset`."
            )

        validation_size = float(params.get("validation_size", 0.1) or 0.0)
        test_size = float(params.get("test_size", 0.0) or 0.0)
        random_state = int(params.get("random_state", 42))
        limit = int(params.get("limit_rows") or 0) or None

        if validation_source == "split" and not (0.0 < validation_size <= 0.5):
            raise ValueError(
                "Validation split must be > 0 and <= 0.5 when validation_source=split."
            )

        if validation_source == "dataset":
            validation_size = 0.0
            validation_dataset_id = str(params.get("validation_dataset_id") or "").strip()
            if not validation_dataset_id:
                raise ValueError(
                    "A validation dataset is required when validation_source=dataset."
                )
        else:
            validation_dataset_id = ""

        if test_source == "none":
            test_size = 0.0
            test_dataset_id = ""
        elif test_source == "split":
            test_dataset_id = ""
            if not (0.0 < test_size <= 0.3):
                raise ValueError(
                    "Test split must be > 0 and <= 0.3 when test_source=split. "
                    "Use test_source=none if no test set is needed."
                )
        elif test_source == "dataset":
            test_size = 0.0
            test_dataset_id = str(params.get("test_dataset_id") or "").strip()
            if not test_dataset_id:
                raise ValueError(
                    "A test dataset is required when test_source=dataset."
                )

        if validation_source == "split" and test_source == "split":
            if validation_size + test_size >= 0.9:
                raise ValueError(
                    "validation_size + test_size is too large. "
                    "Leave enough rows for training."
                )

        train_dataset_id = str(run.dataset_id or "").strip()
        if not train_dataset_id:
            raise ValueError("A selected training dataset is required.")

        source_df, source_meta = self._load_image_split_frame(
            run=run,
            dataset_id=train_dataset_id,
            params=params,
            split_name="source",
            limit=limit,
        )

        train_df, split_val_df, split_test_df = self._split_dataframe(
            source_df,
            label_column="__label_value__",
            test_size=test_size if test_source == "split" else 0.0,
            validation_size=validation_size if validation_source == "split" else 0.0,
            random_state=random_state,
        )

        train_df = train_df.copy()
        train_df["__split__"] = "train"

        if validation_source == "split":
            val_df = split_val_df.copy()
            val_df["__split__"] = "validation"
            val_meta = {
                **source_meta,
                "source": "split",
                "rows": len(val_df),
                "validation_size": validation_size,
            }
        else:
            val_df, val_meta = self._load_image_split_frame(
                run=run,
                dataset_id=validation_dataset_id,
                params=params,
                split_name="validation",
                limit=limit,
            )

        if test_source == "none":
            test_df = self._empty_split_frame()
            test_meta = None
        elif test_source == "split":
            test_df = split_test_df.copy()
            test_df["__split__"] = "test"
            test_meta = {
                **source_meta,
                "source": "split",
                "rows": len(test_df),
                "test_size": test_size,
            }
        else:
            test_df, test_meta = self._load_image_split_frame(
                run=run,
                dataset_id=test_dataset_id,
                params=params,
                split_name="test",
                limit=limit,
            )

        split_mode = "hybrid"
        split_metadata = {
            "mode": split_mode,
            "train": {
                **source_meta,
                "source": "selected_dataset",
                "rows": len(train_df),
            },
            "validation": val_meta,
            "test": test_meta,
            "validation_source": validation_source,
            "test_source": test_source,
            "validation_size": validation_size,
            "test_size": test_size,
            "random_state": random_state,
        }

        if train_df is None or train_df.empty:
            raise ValueError("The training split is empty.")

        for frame in (train_df, val_df, test_df):
            if frame is not None and not frame.empty:
                frame["__label_value__"] = (
                    frame["__label_value__"]
                    .astype(str)
                    .str.strip()
                )
                frame = frame[frame["__label_value__"] != ""]

        def _label_counts(frame):
            if frame is None or frame.empty or "__label_value__" not in frame.columns:
                return {}
            labels = (
                frame["__label_value__"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            labels = labels[labels != ""]
            return {str(k): int(v) for k, v in labels.value_counts().to_dict().items()}

        train_label_counts = _label_counts(train_df)
        val_label_counts = _label_counts(val_df)
        test_label_counts = _label_counts(test_df)

        if val_df is None or val_df.empty:
            raise ValueError(
                "A validation set is required for training. "
                "Set Validation source to `split` with validation_size > 0, "
                "or set Validation source to `dataset` and choose a validation dataset."
            )

        if not val_label_counts:
            raise ValueError(
                "The validation set has no usable labels."
            )

        if not train_label_counts:
            raise ValueError("The training split has no usable labels.")

        class_names = sorted(train_label_counts)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        if len(class_names) < 2:
            raise ValueError(
                "Image classification requires at least two classes in the training split. "
                f"Training labels found: {class_names}"
            )

        val_only = sorted(set(val_label_counts) - set(train_label_counts))
        test_only = sorted(set(test_label_counts) - set(train_label_counts))

        if val_only or test_only:
            raise ValueError(
                "Validation/test labels are not a subset of training labels. "
                f"Training labels: {class_names}. "
                f"Validation-only labels: {val_only}. "
                f"Test-only labels: {test_only}. "
                "This usually means the explicit datasets are using different target "
                "columns, different label encodings, or labels with different formatting."
            )

        for frame in (train_df, val_df, test_df):
            if frame is not None and not frame.empty:
                frame["__target_idx__"] = frame["__label_value__"].map(
                    lambda value: class_to_idx[str(value).strip()]
                )

        split_source_meta = (
            split_metadata.get("train")
            or split_metadata.get("source")
            or {}
        )

        image_column = str(
            split_source_meta.get("image_column")
            or params.get("image_column")
            or ""
        ).strip()

        target_column = str(
            split_source_meta.get("target_column")
            or params.get("target_column")
            or ""
        ).strip()

        record_id_column = str(
            split_source_meta.get("record_id_column")
            or params.get("record_id_column")
            or ""
        ).strip()

        if not image_column:
            raise ValueError(
                "Could not resolve the image column used for this recipe run. "
                "Check image_column, dataset mappings, or explicit split dataset metadata."
            )

        if not target_column:
            raise ValueError(
                "Could not resolve the target column used for this recipe run. "
                "Check target_column, dataset mappings, or explicit split dataset metadata."
            )

        run.log(
            message=(
                f"Loaded splits: train={len(train_df)}, "
                f"validation={len(val_df)}, test={len(test_df)} "
                f"across {len(class_names)} training classes."
            ),
            extra={
                "classes": class_names,
                "split_mode": split_mode,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "test_rows": len(test_df),
                "train_label_counts": train_label_counts,
                "validation_label_counts": val_label_counts,
                "test_label_counts": test_label_counts,
                "validation_only_labels": val_only,
                "test_only_labels": test_only,
                "split_metadata": split_metadata,
            },
        )

        image_size = int(params.get("image_size", 32))
        train_tf, eval_tf, transform_metadata = self._build_transforms(
            transforms=transforms,
            image_size=image_size,
            augmentation_preset=str(params.get("augmentation_preset", "cifar_standard")),
            normalization=str(params.get("normalization", "cifar10")),
        )

        recipe_self = self

        class ManifestImageDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
                self.frame = frame.reset_index(drop=True)
                self.transform = transform

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, idx: int):
                row = self.frame.iloc[idx]
                image_value = str(row["__image_value__"])
                image = recipe_self._load_image(image_value, Image)
                if self.transform is not None:
                    image = self.transform(image)
                target = int(row["__target_idx__"])
                record_id = str(row["__record_id_value__"])
                return image, target, record_id

        batch_size = int(params.get("batch_size", 128))
        num_workers = int(params.get("num_workers", 2))

        device = self._resolve_device(torch, str(params.get("device", "auto")))
        pin_memory = str(device).startswith("cuda")

        train_loader = DataLoader(
            ManifestImageDataset(train_df, train_tf),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = None
        if val_df is not None and not val_df.empty:
            val_loader = DataLoader(
                ManifestImageDataset(val_df, eval_tf),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        test_loader = None
        if test_df is not None and not test_df.empty:
            test_loader = DataLoader(
                ManifestImageDataset(test_df, eval_tf),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        train_eval_df = train_df
        if len(train_eval_df) > 2048:
            train_eval_df = train_eval_df.sample(
                n=2048,
                random_state=int(params.get("random_state", 42)),
            )

        train_eval_loader = DataLoader(
            ManifestImageDataset(train_eval_df, eval_tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = self._build_model(
            torch=torch,
            nn=nn,
            architecture=str(params.get("architecture", "torchvision.resnet18")),
            custom_model_import=str(params.get("custom_model_import") or "").strip(),
            num_classes=len(class_names),
        )

        if (
            bool(params.get("data_parallel", True))
            and str(device).startswith("cuda")
            and torch.cuda.device_count() > 1
        ):
            model = nn.DataParallel(model)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(
            optim=optim,
            model=model,
            name=str(params.get("optimizer", "sgd")),
            learning_rate=float(params.get("learning_rate", 0.1)),
            momentum=float(params.get("momentum", 0.9)),
            weight_decay=float(params.get("weight_decay", 0.0005)),
        )
        scheduler = self._build_scheduler(
            torch=torch,
            optimizer=optimizer,
            name=str(params.get("scheduler", "cosine")),
            epochs=int(params.get("epochs", 200)),
            cosine_t_max=int(params.get("cosine_t_max", params.get("epochs", 200))),
            step_size=int(params.get("step_size", 60)),
            gamma=float(params.get("gamma", 0.2)),
        )

        requested_amp = bool(params.get("amp", False))
        use_amp = requested_amp and str(device).startswith("cuda")

        if requested_amp and not use_amp:
            run.log(
                "AMP was requested but is only enabled for CUDA devices in this recipe.",
                extra={"device": str(device)},
            )        

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_acc = -1.0
        best_checkpoint = run.work_dir / "best_model.pt"
        last_checkpoint = run.work_dir / "last_model.pt"
        epochs = int(params.get("epochs", 200))
        history: List[Dict[str, Any]] = []

        image_spec_id = run.put_artifact(
            "ml.image_spec",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "dataset_id": run.dataset_id,
                "image_column": image_column,
                "target_column": target_column,
                "record_id_column": record_id_column,
                "class_names": class_names,
                "transform": transform_metadata,
            },
        )

        split_spec_id = run.put_artifact(
            "ml.split_spec",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "dataset_id": run.dataset_id,
                "record_id_column": record_id_column,
                "split_mode": split_mode,
                "split_metadata": split_metadata,
                "train_row_ids": train_df["__record_id_value__"].astype(str).tolist(),
                "validation_row_ids": (
                    val_df["__record_id_value__"].astype(str).tolist()
                    if val_df is not None and not val_df.empty
                    else []
                ),
                "test_row_ids": (
                    test_df["__record_id_value__"].astype(str).tolist()
                    if test_df is not None and not test_df.empty
                    else []
                ),
                "train_dataset_id": (
                    split_metadata.get("train", {}) or split_metadata.get("source", {})
                ).get("dataset_id"),
                "validation_dataset_id": (
                    split_metadata.get("validation", {}) or {}
                ).get("dataset_id"),
                "test_dataset_id": (
                    split_metadata.get("test", {}) or {}
                ).get("dataset_id"),
                "test_size": float(params.get("test_size", 0.2)),
                "validation_size": float(params.get("validation_size", 0.1)),
                "random_state": int(params.get("random_state", 42)),
            },
        )

        for epoch in range(1, epochs + 1):
            run.check_cancelled()

            train_metrics = self._train_one_epoch(
                torch=torch,
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                run=run,
            )

            train_eval_metrics, _, _, _ = self._evaluate(
                torch=torch,
                model=model,
                loader=train_eval_loader,
                criterion=criterion,
                device=device,
                run=run,
            )

            val_metrics = None
            if val_loader is not None:
                val_metrics, val_predictions, _, val_record_ids = self._evaluate(
                    torch=torch,
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    return_predictions=True,
                    run=run,
                )

                val_prediction_counts = {}
                for pred_idx in val_predictions:
                    label = class_names[int(pred_idx)]
                    val_prediction_counts[label] = val_prediction_counts.get(label, 0) + 1

            if scheduler is not None:
                scheduler.step()

            row = {
                "epoch": epoch,

                # User-facing training metrics:
                # evaluate the final model for this epoch on training rows using eval mode
                # and eval transforms, so this is directly comparable with validation/test.
                "train_loss": train_eval_metrics["loss"],
                "train_accuracy": train_eval_metrics["accuracy"],

                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }

            if val_metrics is not None:
                row["val_loss"] = val_metrics["loss"]
                row["val_accuracy"] = val_metrics["accuracy"]

            history.append(row)

            if val_metrics is not None:
                score_metric = "val_accuracy"
                score_value = float(val_metrics["accuracy"])
                message = (
                    f"Epoch {epoch}/{epochs}: "
                    f"train_acc={row['train_accuracy']:.4f}, "
                    f"val_acc={row['val_accuracy']:.4f}"
                )
            else:
                score_metric = "train_accuracy"
                score_value = float(train_metrics["accuracy"])
                message = (
                    f"Epoch {epoch}/{epochs}: "
                    f"train_acc={row['train_accuracy']:.4f}"
                )

            run.log(
                message=message,
                step=epoch,
                total=epochs,
                metrics=row,
                extra={
                    "epoch": epoch,
                    "val_prediction_counts": val_prediction_counts if epoch <= 5 or epoch % 10 == 0 else None,
                }
            )

            if score_value > best_val_acc:
                best_val_acc = score_value
                self._save_torch_checkpoint(
                    torch=torch,
                    path=best_checkpoint,
                    model=model,
                    params=params,
                    class_names=class_names,
                    transform_metadata=transform_metadata,
                    architecture=str(params.get("architecture", "torchvision.resnet18")),
                    custom_model_import=str(params.get("custom_model_import") or "").strip(),
                    epoch=epoch,
                    metrics={
                        "best_score": best_val_acc,
                        "best_score_metric": score_metric,
                        **row,
                    },
                )

        self._save_torch_checkpoint(
            torch=torch,
            path=last_checkpoint,
            model=model,
            params=params,
            class_names=class_names,
            transform_metadata=transform_metadata,
            architecture=str(params.get("architecture", "torchvision.resnet18")),
            custom_model_import=str(params.get("custom_model_import") or "").strip(),
            epoch=epochs,
            metrics=history[-1] if history else {},
        )

        if best_checkpoint.exists():
            self._load_model_state(torch=torch, model=model, path=best_checkpoint, device=device)

        if test_loader is not None:
            test_metrics, test_predictions, test_probabilities, test_record_ids = self._evaluate(
                torch=torch,
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                return_predictions=True,
                run=run,
            )
        else:
            test_metrics = {}
            test_predictions = []
            test_probabilities = []
            test_record_ids = []

        model_dir = run.work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        shutil.copy2(best_checkpoint if best_checkpoint.exists() else last_checkpoint, model_path)

        model_artifact_id = run.put_artifact(
            "ml.model",
            {
                "schema_version": 1,
                "kind": "torch_image_classifier",
                "framework": "torch",
                "task": "classification",
                "modality": "image",
                "run_id": run.run_id,
                "dataset_id": run.dataset_id,
                "recipe_id": run.recipe_id,
                "recipe_version": run.recipe_version,
                "model_ref": {
                    "storage": "file",
                    "path": str(model_path),
                    "format": "torch_checkpoint",
                },
                "architecture": str(params.get("architecture", "torchvision.resnet18")),
                "custom_model_import": str(params.get("custom_model_import") or "").strip(),
                "class_names": class_names,
                "input_contract": {
                    "image_column": image_column,
                    "target_column": target_column,
                    "record_id_column": record_id_column,
                    "image_size": image_size,
                    "normalization": str(params.get("normalization", "cifar10")),
                    "transform": transform_metadata,
                },
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "optimizer": str(params.get("optimizer", "sgd")),
                    "learning_rate": float(params.get("learning_rate", 0.1)),
                    "momentum": float(params.get("momentum", 0.9)),
                    "weight_decay": float(params.get("weight_decay", 0.0005)),
                    "scheduler": str(params.get("scheduler", "cosine")),
                    "augmentation_preset": str(params.get("augmentation_preset", "cifar_standard")),
                    "amp": use_amp,
                    "split_mode": split_mode,
                },
                "metrics": {
                    "best_score": best_val_acc,
                    "best_val_accuracy": best_val_acc if val_loader is not None else None,
                    "best_train_accuracy": best_val_acc if val_loader is None else None,
                    "test_loss": test_metrics.get("loss"),
                    "test_accuracy": test_metrics.get("accuracy"),
                },
                "created_at": time.time(),
            },
        )

        prediction_rows = []
        for record_id, pred_idx, probs in zip(
            test_record_ids,
            test_predictions,
            test_probabilities,
        ):
            pred_idx = int(pred_idx)
            prob_list = [float(v) for v in probs]
            max_prob = max(prob_list) if prob_list else None
            prediction_rows.append(
                {
                    "record_id": str(record_id),
                    "prediction": class_names[pred_idx],
                    "prediction_index": pred_idx,
                    "confidence": max_prob,
                    "uncertainty": None if max_prob is None else 1.0 - float(max_prob),
                    "probabilities": prob_list,
                }
            )

        predictions_artifact_id = run.put_artifact(
            "ml.predictions",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "dataset_id": run.dataset_id,
                "model_artifact_id": model_artifact_id,
                "task": "classification",
                "class_names": class_names,
                "record_id_column": record_id_column,
                "rows": prediction_rows,
            },
            row_ids=[row["record_id"] for row in prediction_rows],
        )

        evaluation_report_id = run.put_artifact(
            "ml.evaluation_report",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "dataset_id": run.dataset_id,
                "model_artifact_id": model_artifact_id,
                "image_spec_artifact_id": image_spec_id,
                "split_spec_artifact_id": split_spec_id,
                "predictions_artifact_id": predictions_artifact_id,
                "history": history,
                "metrics": {
                    "best_score": best_val_acc,
                    "best_val_accuracy": best_val_acc if val_loader is not None else None,
                    "best_train_accuracy": best_val_acc if val_loader is None else None,
                    "test_loss": test_metrics.get("loss"),
                    "test_accuracy": test_metrics.get("accuracy"),
                },
            },
        )

        final_metrics = {
            "best_score": best_val_acc,
            "best_val_accuracy": best_val_acc if val_loader is not None else None,
            "best_train_accuracy": best_val_acc if val_loader is None else None,
            "test_loss": test_metrics.get("loss"),
            "test_accuracy": test_metrics.get("accuracy"),
        }

        if test_metrics:
            final_message = (
                f"Finished image recipe. "
                f"best_score={best_val_acc:.4f}, "
                f"test_accuracy={test_metrics.get('accuracy'):.4f}"
            )
        else:
            final_message = (
                f"Finished image recipe. "
                f"best_score={best_val_acc:.4f}. No test dataset was supplied."
            )

        run.log(
            message=final_message,
            status="complete",
            metrics=final_metrics,
        )

        return json_safe(
            {
                "image_spec_artifact_id": image_spec_id,
                "split_spec_artifact_id": split_spec_id,
                "model_artifact_id": model_artifact_id,
                "predictions_artifact_id": predictions_artifact_id,
                "evaluation_report_artifact_id": evaluation_report_id,
                "training_log_artifact_id": run.training_log_artifact_id,
                "metrics": final_metrics,
                "model_path": str(model_path),
                "class_names": class_names,
                "history": history,
            }
        )

    def _imports(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms

        return torch, nn, optim, DataLoader, Dataset, Image, transforms

    def _load_image(self, value: str, Image: Any):
        if value.startswith("file://"):
            value = value[7:]
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {value}")
        return Image.open(path).convert("RGB")

    def _empty_split_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "__image_value__",
                "__label_value__",
                "__record_id_value__",
                "__source_dataset_id__",
                "__source_image_column__",
                "__source_target_column__",
                "__source_record_id_column__",
                "__split__",
            ]
        )


    def _label_counts(self, frame: Optional[pd.DataFrame]) -> Dict[str, int]:
        if frame is None or frame.empty or "__label_value__" not in frame.columns:
            return {}

        labels = (
            frame["__label_value__"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        labels = labels[labels != ""]
        counts = labels.value_counts().to_dict()
        return {str(label): int(count) for label, count in counts.items()}


    def _param_for_dataset(
        self,
        *,
        params: Mapping[str, Any],
        key: str,
        dataset_id: str,
    ) -> str:
        value = params.get(key)

        inferred_keys = {
            str(item)
            for item in (params.get("_inferred_param_keys") or [])
        }
        inferred_dataset_id = str(params.get("_inferred_dataset_id") or "").strip()

        # If image_column/target_column/record_id_column was inferred from the
        # launcher-selected dataset, do not blindly reuse it for another explicit
        # split dataset. Let that dataset resolve from its own mappings/inference.
        if key in inferred_keys and inferred_dataset_id and dataset_id != inferred_dataset_id:
            return ""

        return str(value or "").strip()

    def _resolve_image_columns_for_dataset(
        self,
        *,
        run: MLRunContext,
        dataset_id: str,
        params: Mapping[str, Any],
    ) -> Dict[str, str]:
        inferred = infer_recipe_params(run.context, dataset_id, self.spec())

        image_param = self._param_for_dataset(
            params=params,
            key="image_column",
            dataset_id=dataset_id,
        )
        target_param = self._param_for_dataset(
            params=params,
            key="target_column",
            dataset_id=dataset_id,
        )
        record_id_param = self._param_for_dataset(
            params=params,
            key="record_id_column",
            dataset_id=dataset_id,
        )

        image_column = str(
            image_param
            or inferred.get("image_column")
            or mapped_column(run.context, dataset_id, "image.path")
            or mapped_column(run.context, dataset_id, "image.uri")
            or ""
        ).strip()

        target_column = str(
            target_param
            or inferred.get("target_column")
            or mapped_column(run.context, dataset_id, "target_label")
            or ""
        ).strip()

        record_id_column = str(
            record_id_param
            or inferred.get("record_id_column")
            or mapped_column(run.context, dataset_id, "record_id")
            or ""
        ).strip()

        if not image_column:
            raise ValueError(
                f"Could not infer image column for dataset `{dataset_id}`. "
                "Set image_column or map image.path/image.uri."
            )

        if not target_column:
            raise ValueError(
                f"Could not infer target/label column for dataset `{dataset_id}`. "
                "Set target_column or map target_label."
            )

        return {
            "image_column": image_column,
            "target_column": target_column,
            "record_id_column": record_id_column,
        }


    def _load_image_split_frame(
        self,
        *,
        run: MLRunContext,
        dataset_id: str,
        params: Mapping[str, Any],
        split_name: str,
        limit: Optional[int],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        columns_meta = self._resolve_image_columns_for_dataset(
            run=run,
            dataset_id=dataset_id,
            params=params,
        )

        image_column = columns_meta["image_column"]
        target_column = columns_meta["target_column"]
        record_id_column = columns_meta["record_id_column"]

        columns = [image_column, target_column]
        if record_id_column:
            columns.append(record_id_column)

        raw = get_dataset_frame(
            run.context,
            dataset_id,
            columns=columns,
            limit=limit,
        )

        raw = raw.dropna(subset=[image_column, target_column]).reset_index(drop=True)

        if not raw.empty:
            raw[image_column] = raw[image_column].astype(str).str.strip()
            raw[target_column] = raw[target_column].astype(str).str.strip()

            raw = raw[
                (raw[image_column] != "")
                & (raw[target_column] != "")
            ].reset_index(drop=True)

        if raw.empty:
            frame = self._empty_split_frame()
        else:
            if record_id_column:
                record_ids = raw[record_id_column].astype(str).str.strip().tolist()
            else:
                record_id_column = "__rowid__"
                record_ids = [f"{dataset_id}:{i}" for i in range(len(raw))]

            frame = pd.DataFrame(
                {
                    "__image_value__": raw[image_column].astype(str).str.strip().tolist(),
                    "__label_value__": raw[target_column].astype(str).str.strip().tolist(),
                    "__record_id_value__": record_ids,
                    "__source_dataset_id__": dataset_id,
                    "__source_image_column__": image_column,
                    "__source_target_column__": target_column,
                    "__source_record_id_column__": record_id_column,
                    "__split__": split_name,
                }
            )

        meta = {
            "dataset_id": dataset_id,
            "image_column": image_column,
            "target_column": target_column,
            "record_id_column": record_id_column,
            "rows": len(frame),
        }

        return frame, meta

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        *,
        label_column: str,
        test_size: float,
        validation_size: float,
        random_state: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split

        test_size = float(test_size or 0.0)
        validation_size = float(validation_size or 0.0)

        if test_size < 0.0 or test_size >= 1.0:
            raise ValueError("test_size must be >= 0 and < 1.")

        if validation_size < 0.0 or validation_size >= 1.0:
            raise ValueError("validation_size must be >= 0 and < 1.")

        if test_size + validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be < 1.")

        if df.empty:
            return df.copy(), df.copy(), df.copy()

        train_pool = df.copy()
        test = df.iloc[0:0].copy()

        if test_size > 0.0:
            stratify = (
                df[label_column]
                if label_column in df.columns and df[label_column].nunique() > 1
                else None
            )

            train_pool, test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

        val = df.iloc[0:0].copy()

        if validation_size > 0.0:
            # validation_size is expressed as a fraction of the original selected
            # training dataset, not a fraction of the post-test remainder.
            relative_val_size = validation_size / max(1e-9, 1.0 - test_size)

            if relative_val_size <= 0.0 or relative_val_size >= 1.0:
                raise ValueError(
                    "Validation split leaves no training rows. "
                    "Reduce validation_size and/or test_size."
                )

            val_stratify = (
                train_pool[label_column]
                if label_column in train_pool.columns and train_pool[label_column].nunique() > 1
                else None
            )

            train, val = train_test_split(
                train_pool,
                test_size=relative_val_size,
                random_state=random_state,
                stratify=val_stratify,
            )
        else:
            train = train_pool

        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    def _build_transforms(
        self,
        *,
        transforms: Any,
        image_size: int,
        augmentation_preset: str,
        normalization: str,
    ):
        if normalization == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif normalization == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = None
            std = None

        train_ops = []
        eval_ops = []

        if augmentation_preset == "cifar_standard":
            if image_size != 32:
                train_ops.append(transforms.Resize((image_size, image_size)))

            train_ops.extend(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            train_ops.append(transforms.Resize((image_size, image_size)))

        if image_size != 32:
            eval_ops.append(transforms.Resize((image_size, image_size)))

        train_ops.append(transforms.ToTensor())
        eval_ops.append(transforms.ToTensor())

        if mean is not None and std is not None:
            train_ops.append(transforms.Normalize(mean, std))
            eval_ops.append(transforms.Normalize(mean, std))

        metadata = {
            "image_size": image_size,
            "augmentation_preset": augmentation_preset,
            "normalization": normalization,
            "mean": mean,
            "std": std,
        }
        return transforms.Compose(train_ops), transforms.Compose(eval_ops), metadata

    def _resolve_device(self, torch: Any, requested: str):
        requested = requested.lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "mps":
            mps_available = bool(
                getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
            )
            return torch.device("mps" if mps_available else "cpu")

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_available = bool(
            getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        )
        if mps_available:
            return torch.device("mps")

        return torch.device("cpu")

    def _build_model(
        self,
        *,
        torch: Any,
        nn: Any,
        architecture: str,
        custom_model_import: str,
        num_classes: int,
    ):
        if architecture == "custom_import":
            if not custom_model_import:
                raise ValueError("custom_model_import is required when architecture=custom_import.")
            factory = _import_object(custom_model_import)
            try:
                return factory(num_classes=num_classes)
            except TypeError:
                return factory()

        import torchvision.models as models

        if architecture == "torchvision.resnet18":
            model = models.resnet18(weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.maxpool = nn.Identity()
            return model

        if architecture == "torchvision.resnet34":
            model = models.resnet34(weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.maxpool = nn.Identity()
            return model

        if architecture == "torchvision.mobilenet_v3_small":
            return models.mobilenet_v3_small(weights=None, num_classes=num_classes)

        raise ValueError(f"Unsupported architecture: {architecture}")

    def _build_optimizer(
        self,
        *,
        optim: Any,
        model: Any,
        name: str,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ):
        name = name.lower()
        if name == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        if name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(
        self,
        *,
        torch: Any,
        optimizer: Any,
        name: str,
        epochs: int,
        cosine_t_max: int,
        step_size: int,
        gamma: float,
    ):
        name = name.lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(cosine_t_max or epochs),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        raise ValueError(f"Unsupported scheduler: {name}")

    def _train_one_epoch(
        self,
        *,
        torch: Any,
        model: Any,
        loader: Any,
        criterion: Any,
        optimizer: Any,
        scaler: Any,
        device: Any,
        use_amp: bool,
        run: MLRunContext,
    ) -> Dict[str, Any]:
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for images, targets, _record_ids in loader:
            run.check_cancelled()

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    if not bool(torch.isfinite(outputs.detach()).all().item()):
                        raise FloatingPointError(
                            "Non-finite train logits detected. Disable AMP, lower the learning rate, "
                            "or inspect image normalization."
                        )

                    if not bool(torch.isfinite(loss.detach()).item()):
                        raise FloatingPointError(
                            "Non-finite train loss detected. Disable AMP, lower the learning rate, "
                            "or inspect labels/images."
                        )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

                if not bool(torch.isfinite(outputs.detach()).all().item()):
                    raise FloatingPointError(
                        "Non-finite train logits detected. Disable AMP, lower the learning rate, "
                        "or inspect image normalization."
                    )

                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError(
                        "Non-finite train loss detected. Disable AMP, lower the learning rate, "
                        "or inspect labels/images."
                    )

                loss.backward()
                optimizer.step()

            

            batch_size = int(targets.size(0))
            total_seen += batch_size
            total_loss += float(loss.detach().item()) * batch_size

            predicted = outputs.detach().argmax(dim=1)
            total_correct += int(predicted.eq(targets).sum().item())

        return {
            "loss": None if total_seen <= 0 else total_loss / float(total_seen),
            "accuracy": None if total_seen <= 0 else total_correct / float(total_seen),
            "correct": total_correct,
            "total": total_seen,
        }

    def _evaluate(
        self,
        *,
        torch: Any,
        model: Any,
        loader: Any,
        criterion: Any,
        device: Any,
        return_predictions: bool = False,
        run: Optional[MLRunContext] = None,
    ) -> Tuple[Dict[str, Any], List[int], List[List[float]], List[str]]:
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        all_predictions: List[int] = []
        all_probabilities: List[List[float]] = []
        all_record_ids: List[str] = []

        with torch.no_grad():
            for images, targets, record_ids in loader:
                if run is not None:
                    run.check_cancelled()

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()

                outputs = model(images)
                loss = criterion(outputs, targets)
                if not bool(torch.isfinite(outputs.detach()).all().item()):
                    raise FloatingPointError(
                        "Non-finite evaluation logits detected. The model is numerically unstable."
                    )

                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError(
                        "Non-finite evaluation loss detected. The model is numerically unstable."
                    )

                batch_size = int(targets.size(0))
                total_seen += batch_size
                total_loss += float(loss.detach().item()) * batch_size

                probabilities = torch.softmax(outputs.detach(), dim=1)
                predicted = probabilities.argmax(dim=1)

                total_correct += int(predicted.eq(targets).sum().item())

                if return_predictions:
                    all_predictions.extend(int(value) for value in predicted.cpu().tolist())
                    all_probabilities.extend(
                        [
                            [float(item) for item in row]
                            for row in probabilities.cpu().tolist()
                        ]
                    )
                    all_record_ids.extend(str(value) for value in record_ids)

        metrics = {
            "loss": None if total_seen <= 0 else total_loss / float(total_seen),
            "accuracy": None if total_seen <= 0 else total_correct / float(total_seen),
            "correct": total_correct,
            "total": total_seen,
        }

        return metrics, all_predictions, all_probabilities, all_record_ids

    def _module_state_dict(self, model: Any):
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    def _save_torch_checkpoint(
        self,
        *,
        torch: Any,
        path: Path,
        model: Any,
        params: Mapping[str, Any],
        class_names: Sequence[str],
        transform_metadata: Mapping[str, Any],
        architecture: str,
        custom_model_import: str,
        epoch: int,
        metrics: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "state_dict": self._module_state_dict(model),
                "architecture": architecture,
                "custom_model_import": custom_model_import,
                "class_names": list(class_names),
                "transform": dict(transform_metadata),
                "params": dict(params),
                "epoch": int(epoch),
                "metrics": json_safe(dict(metrics)),
            },
            path,
        )

    def _load_model_state(self, *, torch: Any, model: Any, path: Path, device: Any) -> None:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state_dict)