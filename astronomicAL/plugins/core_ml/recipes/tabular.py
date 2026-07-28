from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pandas as pd

from ..harnesses.base import RunHarness
from ..protocol import TrainingComponents
from ..recipe_base import ManagedMLRecipe

_PRODUCES = [
    "ml.split_spec",
    "ml.model",
    "ml.evaluation_report",
    "ml.predictions",
    "ml.training_log",
    "ml.run",
]

_FEATURE_PROPERTIES = {
    "record_id_column": {
        "type": "string",
        "title": "Record ID column",
        "default": "",
        "x-widget": "column_select",
    },
    "target_column": {
        "type": "string",
        "title": "Target / label column",
        "default": "",
        "x-widget": "column_select",
    },
    "feature_columns": {
        "type": "array",
        "items": {"type": "string"},
        "title": "Input feature columns",
        "description": "Numeric columns used as model inputs.",
        "default": [],
        "x-widget": "column_multichoice",
    },
    "auto_feature_columns": {
        "type": "boolean",
        "title": "Auto-select features when none are chosen",
        "default": False,
    },
    "missing_value_policy": {
        "type": "string",
        "title": "Missing/non-finite feature policy",
        "description": (
            "Reject missing, NaN and infinite values, or explicitly "
            "replace them with zero. Error is recommended because zero "
            "may be a meaningful scientific value."
        ),
        "enum": ["error", "zero"],
        "default": "error",
    },
    "epochs": {"type": "integer", "default": 100, "minimum": 1},
    "batch_size": {"type": "integer", "default": 256, "minimum": 1},
    "num_workers": {"type": "integer", "default": 0, "minimum": 0},
    "learning_rate": {"type": "number", "default": 2e-4},
    "weight_decay": {"type": "number", "default": 1e-5},
    "gradient_clip_norm": {"type": "number", "default": 1.0, "minimum": 0.0},
}

def _schema(**properties):
    merged = deepcopy(_FEATURE_PROPERTIES)
    merged.update(properties)
    return {"type": "object", "required": ["feature_columns"], "properties": merged}

def _feature_count(run) -> int:
    count = len([column for column in (run.binding.input_columns or []) if column])
    if count <= 0:
        raise ValueError("The tabular recipe resolved zero input feature columns.")
    return count

def _row_identifier(run, row):
    record_id_column = getattr(
        run.binding,
        "record_id_column",
        None,
    )
    if not record_id_column:
        return "<unknown>"

    try:
        return row[record_id_column]
    except Exception:
        return "<unknown>"

def _coerce_numeric_feature(run, row, column: str) -> float:
    raw_value = row[column]

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Feature {column!r} for record "
            f"{_row_identifier(run, row)!r} is not numeric: "
            f"{raw_value!r}."
        ) from exc

    if math.isfinite(value):
        return value

    policy = str(
        run.params.get("missing_value_policy", "error")
    ).strip().lower()

    if policy == "zero":
        return 0.0

    if policy != "error":
        raise ValueError(
            "missing_value_policy must be 'error' or 'zero', "
            f"got {policy!r}."
        )

    raise ValueError(
        f"Feature {column!r} for record "
        f"{_row_identifier(run, row)!r} is non-finite: "
        f"{raw_value!r}. Set missing_value_policy='zero' only "
        "when zero replacement is scientifically appropriate."
    )

def _feature_columns(run):
    return [
        str(column)
        for column in (run.binding.input_columns or [])
        if column
    ]


def _load_numeric_sample(run, row):
    import torch

    values = [
        _coerce_numeric_feature(run, row, column)
        for column in _feature_columns(run)
    ]
    return torch.tensor(values, dtype=torch.float32)


def _encode_numeric_batch(run, frame):
    """Vectorise numeric dataframe columns into one float32 tensor.

    The feature ordering comes exclusively from ``run.binding.input_columns``.
    Invalid values are detected over the complete NumPy matrix, avoiding
    ``DataFrame.iterrows()`` and per-sample tensor construction.
    """
    import torch

    feature_columns = _feature_columns(run)
    if not feature_columns:
        raise ValueError("The tabular recipe resolved zero input feature columns.")

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "The streamed tabular batch is missing feature column(s): "
            + ", ".join(repr(column) for column in missing)
        )

    raw = frame.loc[:, feature_columns]
    try:
        matrix = raw.to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError):
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        matrix = numeric.to_numpy(dtype=np.float32, copy=True)
    invalid = ~np.isfinite(matrix)

    if invalid.any():
        policy = str(
            run.params.get("missing_value_policy", "error")
        ).strip().lower()

        if policy == "zero":
            matrix[invalid] = 0.0
        elif policy == "error":
            row_position, column_position = np.argwhere(invalid)[0]
            column = feature_columns[int(column_position)]
            record_id_column = getattr(run.binding, "record_id_column", None)
            record_id = "<unknown>"
            if record_id_column and record_id_column in frame.columns:
                try:
                    record_id = frame.iloc[int(row_position)][record_id_column]
                except Exception:
                    pass
            try:
                raw_value = raw.iloc[int(row_position), int(column_position)]
            except Exception:
                raw_value = None
            raise ValueError(
                f"Feature {column!r} for record {record_id!r} is not numeric "
                f"or non-finite: {raw_value!r}. Set "
                "missing_value_policy='zero' only when zero replacement is "
                "scientifically appropriate."
            )
        else:
            raise ValueError(
                "missing_value_policy must be 'error' or 'zero', "
                f"got {policy!r}."
            )

    return torch.from_numpy(matrix)

def _make_optimizer_and_scheduler(run, model):
    import torch.optim as optim

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(run.params.get("learning_rate", 2e-4)),
        weight_decay=float(run.params.get("weight_decay", 1e-5)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(run.params.get("epochs", 100)),
    )
    return optimizer, scheduler

def _classification_criterion(run):
    import torch.nn as nn

    return nn.CrossEntropyLoss(
        label_smoothing=float(run.params.get("label_smoothing", 0.0))
    )

def _regression_criterion(run):
    import torch.nn as nn

    loss = str(run.params.get("loss", "mse")).lower()
    if loss in {"mae", "l1"}:
        return nn.L1Loss()
    if loss in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss(beta=float(run.params.get("huber_beta", 1.0)))
    return nn.MSELoss()

def _fit_torch_tabular(
    run,
    *,
    model,
    components: TrainingComponents,
    train_loader,
    harness: RunHarness,
    classification: bool,
):
    import torch

    device = harness.device
    model.to(device)
    epochs = int(run.params.get("epochs", 100))
    clip_norm = float(run.params.get("gradient_clip_norm", 1.0))

    for epoch in run.epoch_range(epochs):
        run.check_cancelled()
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for inputs, targets, _ids in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.long() if classification else targets.float()
            components.optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            if not classification and output.dim() == 1:
                output = output.unsqueeze(1)
            loss = components.criterion(output, targets)
            loss.backward()
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            components.optimizer.step()

            batch_size = int(targets.size(0))
            loss_sum += float(loss.detach()) * batch_size
            seen += batch_size
            if classification:
                correct += int(output.argmax(1).eq(targets).sum())

        metrics = {"loss": loss_sum / max(seen, 1)}
        if classification:
            metrics["accuracy"] = correct / max(seen, 1)
        harness.report_epoch(epoch, model, train_metrics=metrics)
        if components.scheduler is not None:
            components.scheduler.step()
        harness.check_pause_boundary(epoch, model, components)

def _build_ft_transformer(
    *,
    n_features,
    d_token,
    n_blocks,
    n_heads,
    attention_dropout,
    ffn_dropout,
    d_out,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if d_token % n_heads != 0:
        raise ValueError(f"d_token={d_token} must be divisible by n_heads={n_heads}.")

    class ReGLU(nn.Module):
        def forward(self, x):
            left, gate = x.chunk(2, dim=-1)
            return left * F.relu(gate)

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_norm = nn.LayerNorm(d_token)
            self.attention = nn.MultiheadAttention(
                d_token,
                n_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(d_token)
            d_hidden = max(d_token, int(round(d_token * 4 / 3)))
            self.ffn = nn.Sequential(
                nn.Linear(d_token, d_hidden * 2),
                ReGLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(d_hidden, d_token),
                nn.Dropout(ffn_dropout),
            )

        def forward(self, x):
            normalized = self.attention_norm(x)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            x = x + attended
            return x + self.ffn(self.ffn_norm(x))

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            initialization_scale = d_token ** -0.5
            self.feature_weight = nn.Parameter(
                torch.empty(n_features, d_token).uniform_(
                    -initialization_scale, initialization_scale
                )
            )
            self.feature_bias = nn.Parameter(
                torch.empty(n_features, d_token).uniform_(
                    -initialization_scale, initialization_scale
                )
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            self.blocks = nn.ModuleList(TransformerBlock() for _ in range(n_blocks))
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.ReLU(),
                nn.Linear(d_token, d_out),
            )

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
            tokens = tokens + self.feature_bias.unsqueeze(0)
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), tokens], dim=1)
            for block in self.blocks:
                x = block(x)
            return self.head(x[:, 0])

    return FTTransformer()

def _build_tabular_resnet(*, d_in, d_main, d_hidden, n_blocks, dropout_first, dropout_second, d_out):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(d_main)
            self.linear1 = nn.Linear(d_main, d_hidden)
            self.linear2 = nn.Linear(d_hidden, d_main)
            self.dropout1 = nn.Dropout(dropout_first)
            self.dropout2 = nn.Dropout(dropout_second)
            self.activation = nn.ReLU()

        def forward(self, x):
            residual = self.linear1(self.activation(self.norm(x)))
            residual = self.dropout1(residual)
            residual = self.linear2(self.activation(residual))
            residual = self.dropout2(residual)
            return x + residual

    return nn.Sequential(
        nn.BatchNorm1d(d_in),
        nn.Linear(d_in, d_main),
        *[ResidualBlock() for _ in range(n_blocks)],
        nn.BatchNorm1d(d_main),
        nn.ReLU(),
        nn.Linear(d_main, d_out),
    )

def _build_tabular_mlp(*, d_in: int, hidden, dropout: float, d_out: int):
    """rtdl-style MLP with input BatchNorm so raw features need no external
    scaler (running stats live in the checkpoint, so train/predict agree)."""
    import torch.nn as nn
    layers = [nn.BatchNorm1d(d_in)]
    d = d_in
    for h in hidden:
        h = int(h)
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)

class _TorchTabularBase(ManagedMLRecipe):
    required_imports = ["torch", "sklearn"]
    modality = "tabular"
    framework = "torch"
    complexity = "advanced"
    author = "AstronomicAL"
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = _PRODUCES

    def train_transform(self, run):
        return None

    def eval_transform(self, run):
        return None

    def load_sample(self, run, row):
        return _load_numeric_sample(run, row)

    def encode_batch(self, run, frame, *, train: bool):
        return _encode_numeric_batch(run, frame)

# =============================================================================
# Baseline Torch tabular recipe.
# =============================================================================

class TabularMLPRegressorRecipe(ManagedMLRecipe):
    required_imports = ["torch", "sklearn"]
    id = "core.ml.tabular_mlp_regressor"
    title = "Tabular MLP regressor (rtdl baseline)"
    version = "0.2.0"
    task = "regression"
    modality = "tabular"
    framework = "torch"
    complexity = "intermediate"
    author = "AstronomicAL"
    description = (
        "Strong rtdl/pytorch-tabular-style MLP baseline for tabular regression "
        "(e.g. photo-z from photometry). Input BatchNorm normalises raw features "
        "in-model, so no external scaler is required. Protocol owns "
        "splitting/selection/test; selection defaults to val_loss (min)."
    )
    tags = ["torch", "tabular", "regression", "mlp", "photoz"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report",
                "ml.predictions", "ml.training_log", "ml.run"]

    params_schema = {
        "type": "object",
        "required": ["feature_columns"],
        "properties": {
            "record_id_column": {
                "type": "string",
                "title": "Record ID column",
                "description": "Stable row/object identifier column.",
                "default": "",
                "x-widget": "column_select",
            },
            "target_column": {
                "type": "string",
                "title": "Target / label column",
                "description": "Continuous regression target column.",
                "default": "",
                "x-widget": "column_select",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Input feature columns",
                "description": (
                    "Numeric dataset columns used as model inputs. "
                    "For this torch MLP recipe, selected values must be "
                    "convertible to float."
                ),
                "default": [],
                "x-widget": "column_multichoice",
            },
            "auto_feature_columns": {
                "type": "boolean",
                "title": "Auto-select feature columns if none are chosen",
                "description": (
                    "Fallback only. Explicit feature selection is recommended."
                ),
                "default": False,
            },
            "missing_value_policy": {
                "type": "string",
                "title": "Missing/non-finite feature policy",
                "description": (
                    "Reject missing, NaN and infinite values, or explicitly "
                    "replace them with zero."
                ),
                "enum": ["error", "zero"],
                "default": "error",
            },
            "hidden_layers": {
                "type": "array",
                "items": {"type": "integer"},
                "default": [256, 256, 128],
            },
            "dropout": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
            },
            "n_outputs": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
            },
            "loss": {
                "type": "string",
                "enum": ["mse", "mae", "huber"],
                "default": "mse",
            },
            "epochs": {
                "type": "integer",
                "default": 200,
                "minimum": 1,
            },
            "batch_size": {
                "type": "integer",
                "default": 256,
                "minimum": 1,
            },
            "num_workers": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
            },
            "learning_rate": {
                "type": "number",
                "default": 1e-3,
            },
            "weight_decay": {
                "type": "number",
                "default": 1e-5,
            },
        },
    }

    def _hidden(self, run):
        h = run.params.get("hidden_layers", [256, 256, 128])
        if isinstance(h, str):
            import re
            h = [int(x) for x in re.split(r"[,\s]+", h) if x.strip()]
        return [int(x) for x in h] or [256, 128]

    def build_model(self, run, *, num_classes: int):
        d_in = len([c for c in (run.binding.input_columns or []) if c])
        if d_in == 0:
            raise ValueError(
                "Tabular regression resolved zero feature columns. Map a "
                "record_id + target, leaving the numeric features unmapped."
            )
        return _build_tabular_mlp(
            d_in=d_in, hidden=self._hidden(run),
            dropout=float(run.params.get("dropout", 0.1)),
            d_out=int(num_classes),
        )

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.optim as optim
        p = run.params
        optimizer = optim.AdamW(model.parameters(), lr=float(p.get("learning_rate", 1e-3)),
                                weight_decay=float(p.get("weight_decay", 1e-5)))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(p.get("epochs", 200)))
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler,
                                  criterion=_regression_criterion(run))

    def train_transform(self, run):
        return None        # load_sample already returns a ready feature tensor

    def eval_transform(self, run):
        return None

    def load_sample(self, run, row):
        return _load_numeric_sample(run, row)

    def encode_batch(self, run, frame, *, train: bool):
        return _encode_numeric_batch(run, frame)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device
        model.to(device)
        for epoch in run.epoch_range(int(run.params.get("epochs", 200))):
            run.check_cancelled()
            model.train()
            loss_sum = seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()        # [B, n_outputs]
                components.optimizer.zero_grad(set_to_none=True)
                out = model(inputs)
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                loss = components.criterion(out, targets)
                loss.backward()
                components.optimizer.step()
                loss_sum += float(loss.detach()) * int(targets.size(0))
                seen += int(targets.size(0))
            harness.report_epoch(epoch, model, train_metrics={"loss": loss_sum / max(seen, 1)})
            if components.scheduler is not None:
                components.scheduler.step()
            harness.check_pause_boundary(epoch, model, components)

# =============================================================================
# Advanced Torch tabular architectures.
# =============================================================================

class FTTransformerClassifierRecipe(_TorchTabularBase):
    id = "core.ml.ft_transformer_classifier"
    title = "FT-Transformer tabular classifier"
    version = "0.2.0"
    task = "classification"
    description = (
        "Numeric-feature FT-Transformer classifier based on the strong tabular "
        "architecture from rtdl-revisiting-models."
    )
    tags = ["torch", "tabular", "classification", "transformer", "rtdl"]
    source_urls = ["https://github.com/yandex-research/rtdl-revisiting-models"]
    source_reference = "Implements the FT-Transformer feature-tokenization and Transformer encoder pattern."
    params_schema = _schema(
        d_token={"type": "integer", "default": 192, "minimum": 16},
        n_blocks={"type": "integer", "default": 3, "minimum": 1},
        n_heads={"type": "integer", "default": 8, "minimum": 1},
        attention_dropout={"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 0.9},
        ffn_dropout={"type": "number", "default": 0.1, "minimum": 0.0, "maximum": 0.9},
        label_smoothing={"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
    )

    def build_model(self, run, *, num_classes: int):
        p = run.params
        return _build_ft_transformer(
            n_features=_feature_count(run),
            d_token=int(p.get("d_token", 192)),
            n_blocks=int(p.get("n_blocks", 3)),
            n_heads=int(p.get("n_heads", 8)),
            attention_dropout=float(p.get("attention_dropout", 0.2)),
            ffn_dropout=float(p.get("ffn_dropout", 0.1)),
            d_out=int(num_classes),
        )

    def configure_training(self, run, model) -> TrainingComponents:
        optimizer, scheduler = _make_optimizer_and_scheduler(run, model)
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=_classification_criterion(run))

    def fit(self, run, *, model, components, train_loader, harness):
        return _fit_torch_tabular(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
            classification=True,
        )

class FTTransformerRegressorRecipe(FTTransformerClassifierRecipe):
    id = "core.ml.ft_transformer_regressor"
    title = "FT-Transformer tabular regressor"
    version = "0.2.0"
    task = "regression"
    description = "FT-Transformer for continuous tabular targets under the managed regression protocol."
    tags = ["torch", "tabular", "regression", "transformer", "rtdl"]
    params_schema = _schema(
        d_token={"type": "integer", "default": 192, "minimum": 16},
        n_blocks={"type": "integer", "default": 3, "minimum": 1},
        n_heads={"type": "integer", "default": 8, "minimum": 1},
        attention_dropout={"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 0.9},
        ffn_dropout={"type": "number", "default": 0.1, "minimum": 0.0, "maximum": 0.9},
        n_outputs={"type": "integer", "default": 1, "minimum": 1},
        loss={"type": "string", "enum": ["mse", "mae", "huber"], "default": "mse"},
        huber_beta={"type": "number", "default": 1.0, "minimum": 1e-6},
    )

    def configure_training(self, run, model) -> TrainingComponents:
        optimizer, scheduler = _make_optimizer_and_scheduler(run, model)
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=_regression_criterion(run))

    def fit(self, run, *, model, components, train_loader, harness):
        return _fit_torch_tabular(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
            classification=False,
        )

class TabularResNetClassifierRecipe(_TorchTabularBase):
    id = "core.ml.rtdl_resnet_classifier"
    title = "rtdl ResNet tabular classifier"
    version = "0.2.0"
    task = "classification"
    description = (
        "Residual MLP classifier with BatchNorm and skip connections, following "
        "the strong ResNet-like baseline from rtdl-revisiting-models."
    )
    tags = ["torch", "tabular", "classification", "resnet", "rtdl"]
    source_urls = ["https://github.com/yandex-research/rtdl-revisiting-models"]
    source_reference = "Implements the paper's ResNet-like tabular baseline with residual fully connected blocks."
    params_schema = _schema(
        d_main={"type": "integer", "default": 256, "minimum": 16},
        d_hidden={"type": "integer", "default": 512, "minimum": 16},
        n_blocks={"type": "integer", "default": 4, "minimum": 1},
        dropout_first={"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 0.9},
        dropout_second={"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 0.9},
        label_smoothing={"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
    )

    def build_model(self, run, *, num_classes: int):
        p = run.params
        return _build_tabular_resnet(
            d_in=_feature_count(run),
            d_main=int(p.get("d_main", 256)),
            d_hidden=int(p.get("d_hidden", 512)),
            n_blocks=int(p.get("n_blocks", 4)),
            dropout_first=float(p.get("dropout_first", 0.2)),
            dropout_second=float(p.get("dropout_second", 0.0)),
            d_out=int(num_classes),
        )

    def configure_training(self, run, model) -> TrainingComponents:
        optimizer, scheduler = _make_optimizer_and_scheduler(run, model)
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=_classification_criterion(run))

    def fit(self, run, *, model, components, train_loader, harness):
        return _fit_torch_tabular(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
            classification=True,
        )

class TabularResNetRegressorRecipe(TabularResNetClassifierRecipe):
    id = "core.ml.rtdl_resnet_regressor"
    title = "rtdl ResNet tabular regressor"
    version = "0.2.0"
    task = "regression"
    description = "Residual MLP regression baseline based on rtdl-revisiting-models."
    tags = ["torch", "tabular", "regression", "resnet", "rtdl"]
    params_schema = _schema(
        d_main={"type": "integer", "default": 256, "minimum": 16},
        d_hidden={"type": "integer", "default": 512, "minimum": 16},
        n_blocks={"type": "integer", "default": 4, "minimum": 1},
        dropout_first={"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 0.9},
        dropout_second={"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 0.9},
        n_outputs={"type": "integer", "default": 1, "minimum": 1},
        loss={"type": "string", "enum": ["mse", "mae", "huber"], "default": "mse"},
        huber_beta={"type": "number", "default": 1.0, "minimum": 1e-6},
    )

    def configure_training(self, run, model) -> TrainingComponents:
        optimizer, scheduler = _make_optimizer_and_scheduler(run, model)
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=_regression_criterion(run))

    def fit(self, run, *, model, components, train_loader, harness):
        return _fit_torch_tabular(
            run,
            model=model,
            components=components,
            train_loader=train_loader,
            harness=harness,
            classification=False,
        )
