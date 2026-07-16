from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from pathlib import Path

# Framework types now live in core_ml recipe framework. recipes.py holds ONLY
# concrete recipes — it must not redefine Partition/Partitions/RunHarness/
# MLRecipe/TrainingComponents (doing so shadows the real ones and breaks
# make_harness()).
from ..recipe_base import ManagedMLRecipe
from ..harnesses.base import RunHarness
from ..protocol import TrainingComponents

def _import_object(path: str):
    module_name, object_name = path.rsplit(".", 1)
    return getattr(import_module(module_name), object_name)

# =============================================================================
# kuangliu/pytorch-cifar, protocol-managed. Implements ONLY internals + a loop
# that delegates selection. Splitting, val/test evaluation, best-epoch
# selection, and artifact writing are the harness's — see core_ml recipe framework.
#
# What the expert keeps vs the old recipe: model, optimizer, scheduler, loss,
# augmentation, the training step. What they give up: the test()/best_acc block,
# which becomes a single harness.report_epoch() call.
# =============================================================================

class CIFARResNetRecipe(ManagedMLRecipe):
    id = "core.ml.cifar_resnet"
    title = "CIFAR ResNet (kuangliu)"
    version = "0.3.0"
    task = "classification"
    modality = "image"
    framework = "torch"                       # selects TorchClassificationHarness
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Train a CIFAR-style ResNet with a real PyTorch loop (SGD + cosine, "
        "CIFAR or RandAugment transforms). Validation, best-epoch selection and "
        "test evaluation are enforced by AstronomicAL's protocol, not this code."
    )
    tags = ["torch", "image", "classification", "cifar"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label", "image.path", "image.uri"]
    produces = [
        "ml.split_spec", "ml.model", "ml.evaluation_report",
        "ml.predictions", "ml.training_log", "ml.run",
    ]

    # Purely internals. No split sizes, no record_id/image/target columns, no
    # selection metric, no random_state — those are protocol / data-binding and
    # are collected at the run level, not here.
    params_schema = {
        "type": "object",
        "properties": {
            "architecture": {"type": "string",
                             "enum": ["resnet18", "resnet34", "custom_import"],
                             "default": "resnet18"},
            "custom_model_import": {"type": "string", "default": ""},
            "input_size": {"type": "integer", "default": 32, "minimum": 16},
            "normalization": {
                "type": "string",
                "enum": ["cifar", "imagenet", "custom", "none"],
                "default": "cifar",
                "title": "Normalisation preset",
                "description": "Use CIFAR, ImageNet, custom mean/std, or no input normalisation.",
            },
            "normalize_mean": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.4914, 0.4822, 0.4465],
                "title": "Normalisation mean",
                "description": "RGB channel means in 0..1. Used when normalisation preset is custom.",
            },
            "normalize_std": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.2023, 0.1994, 0.2010],
                "title": "Normalisation std",
                "description": "RGB channel standard deviations in 0..1. Used when normalisation preset is custom.",
            },
            "epochs": {"type": "integer", "default": 200, "minimum": 1},
            "batch_size": {"type": "integer", "default": 128, "minimum": 1},
            "num_workers": {"type": "integer", "default": 0, "minimum": 0},
            "learning_rate": {"type": "number", "default": 0.1},
            "weight_decay": {"type": "number", "default": 5e-4},
            "momentum": {"type": "number", "default": 0.9},
            "label_smoothing": {"type": "number", "default": 0.0},
            "augmentation": {"type": "string",
                             "enum": ["cifar_standard", "randaugment", "none"],
                             "default": "cifar_standard"},
        },
    }

    # --- internals: the expert's contribution -------------------------------

    def build_model(self, run, *, num_classes: int):
        import torch.nn as nn
        from torchvision import models
        arch = str(run.params.get("architecture", "resnet18"))

        if arch == "custom_import":
            path = str(run.params.get("custom_model_import") or "").strip()
            if not path:
                raise ValueError("custom_model_import is required for architecture=custom_import.")
            factory = _import_object(path)
            try:
                return factory(num_classes=num_classes)
            except TypeError:
                # kuangliu ResNet18() hard-codes 10 classes and ignores the kwarg.
                # The harness's _assert_output_dim catches the wrong-width head
                # before training, so this is a loud failure, not a silent one.
                return factory()

        ctor = models.resnet34 if arch == "resnet34" else models.resnet18
        m = ctor(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem surgery
        m.maxpool = nn.Identity()
        return m

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.nn as nn
        import torch.optim as optim
        p = run.params
        criterion = nn.CrossEntropyLoss(label_smoothing=float(p.get("label_smoothing", 0.0)))
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(p.get("learning_rate", 0.1)),
            momentum=float(p.get("momentum", 0.9)),
            weight_decay=float(p.get("weight_decay", 5e-4)),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(p.get("epochs", 200)))
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=criterion)

    def train_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params.get("input_size", 32) or 32)
        aug = str(run.params.get("augmentation", "cifar_standard"))

        ops = []
        if aug == "cifar_standard":
            ops += [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=max(1, size // 8)),
                T.RandomHorizontalFlip(),
            ]
        elif aug == "randaugment":
            ops += [
                T.Resize((size, size)),
                T.RandAugment(),
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
            ]
        else:
            ops += [T.Resize((size, size))]

        ops.append(T.ToTensor())
        _append_normalize(
            ops,
            T,
            run,
            default_mode="cifar",
            default_mean=[0.4914, 0.4822, 0.4465],
            default_std=[0.2023, 0.1994, 0.2010],
        )
        return T.Compose(ops)

    def eval_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params.get("input_size", 32) or 32)
        ops = [
            T.Resize((size, size)),
            T.ToTensor(),
        ]
        _append_normalize(
            ops,
            T,
            run,
            default_mode="cifar",
            default_mean=[0.4914, 0.4822, 0.4465],
            default_std=[0.2023, 0.1994, 0.2010],
        )
        return T.Compose(ops)

    def load_sample(self, run, row):
        """row -> raw input. The harness passes a dataframe row and handles the
        record-id keying/batching; the recipe only knows how to read one image.
        The image column is resolved by the harness (data binding), read here."""
        from PIL import Image
        binding = run.binding
        col = binding.image_column or (binding.input_columns[0] if binding.input_columns else None)
        if not col:
            raise ValueError("No image column resolved for this dataset (map image.path).")
        value = str(row[col]).strip()
        if value.startswith("file://"):
            value = value[7:]
        return Image.open(Path(value)).convert("RGB")

    # eval_forward inherits the default (model(batch_inputs)) from ManagedMLRecipe.

    # --- the loop: expert owns it; selection is delegated -------------------

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device          # harness owns device resolution
        model.to(device)
        epochs = int(run.params.get("epochs", 200))

        for epoch in range(1, epochs + 1):
            run.check_cancelled()
            model.train()
            loss_sum = 0.0
            correct = seen = 0

            for inputs, targets, _ids in train_loader:    # train_loader ONLY
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                components.optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = components.criterion(logits, targets)
                loss.backward()
                components.optimizer.step()
                loss_sum += float(loss.detach()) * int(targets.size(0))
                seen += int(targets.size(0))
                correct += int(logits.argmax(1).eq(targets).sum())

            train_metrics = {
                "loss": loss_sum / max(seen, 1),
                "accuracy": correct / max(seen, 1),
            }

            # The ONE protocol line. Harness evaluates val, records, selects the
            # best epoch, checkpoints. The recipe has no test set and does not
            # decide 'best'. Replaces kuangliu's test()/best_acc block.
            harness.report_epoch(epoch, model, train_metrics=train_metrics)

            if components.scheduler is not None:
                components.scheduler.step()
        # No return — the harness restored best-epoch weights after fit.

# =============================================================================
# Shared image model and augmentation helpers. Imports of optional ML frameworks
# remain inside methods so importing the recipe registry stays lightweight.
# =============================================================================

# -----------------------------------------------------------------------------
# Small model/augmentation factories (defined lazily so module import is torch-free)
# -----------------------------------------------------------------------------

def _build_wide_resnet(*, depth: int, widen_factor: int, dropout: float, num_classes: int):
    """Zagoruyko & Komodakis WideResNet (the WRN-28-10 family hysts trains)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _WideBasic(nn.Module):
        def __init__(self, in_planes, planes, stride, p):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
            self.dropout = nn.Dropout(p=p)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(self.dropout(F.relu(self.bn2(out))))
            return out + self.shortcut(x)

    class _WideResNet(nn.Module):
        def __init__(self, depth, k, p, num_classes):
            super().__init__()
            assert (depth - 4) % 6 == 0, "WideResNet depth must be 6n+4 (e.g. 28)."
            n = (depth - 4) // 6
            stages = [16, 16 * k, 32 * k, 64 * k]
            self.conv1 = nn.Conv2d(3, stages[0], 3, padding=1, bias=False)
            self.layer1 = self._make(stages[0], stages[1], n, 1, p)
            self.layer2 = self._make(stages[1], stages[2], n, 2, p)
            self.layer3 = self._make(stages[2], stages[3], n, 2, p)
            self.bn1 = nn.BatchNorm2d(stages[3])
            self.linear = nn.Linear(stages[3], num_classes)

        def _make(self, in_planes, planes, num_blocks, stride, p):
            strides = [stride] + [1] * (num_blocks - 1)
            layers, ip = [], in_planes
            for s in strides:
                layers.append(_WideBasic(ip, planes, s, p))
                ip = planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer3(self.layer2(self.layer1(out)))
            out = F.relu(self.bn1(out))
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
            return self.linear(out)

    return _WideResNet(depth, widen_factor, dropout, num_classes)

class _Cutout:
    """DeVries & Taylor Cutout, applied to a normalised CHW tensor."""
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img):
        import torch
        if self.size <= 0:
            return img
        _, h, w = img.shape
        cy, cx = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
        y1, y2 = max(0, cy - self.size // 2), min(h, cy + self.size // 2)
        x1, x2 = max(0, cx - self.size // 2), min(w, cx + self.size // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

def _open_image_from_row(run, row):
    from PIL import Image
    b = run.binding
    col = b.image_column or (b.input_columns[0] if b.input_columns else None)
    if not col:
        raise ValueError("No image column resolved for this dataset (map image.path).")
    value = str(row[col]).strip()
    if value.startswith("file://"):
        value = value[7:]
    return Image.open(Path(value)).convert("RGB")
def _parse_float_list(value, default, *, expected_len: int = 3):
    if value is None or value == "":
        values = list(default)
    elif isinstance(value, str):
        import json
        import re

        text = value.strip()
        if not text:
            values = list(default)
        else:
            try:
                parsed = json.loads(text)
                values = list(parsed)
            except Exception:
                values = [part for part in re.split(r"[,\s]+", text) if part.strip()]
    else:
        values = list(value)

    out = [float(item) for item in values]
    if len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} normalisation values, got {len(out)}: {out!r}")
    return out

def _normalization_params(run, *, default_mode: str, default_mean, default_std):
    params = getattr(run, "params", {}) or {}

    mode = str(
        params.get("normalization")
        or params.get("normalisation")
        or default_mode
        or "custom"
    ).strip().lower()

    if mode in {"none", "off", "false", "no", "disabled"}:
        return None

    if mode in {"imagenet", "image_net"}:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if mode in {"cifar", "cifar10", "cifar_10"}:
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

    mean = _parse_float_list(
        params.get("normalize_mean")
        or params.get("normalization_mean")
        or params.get("normalisation_mean"),
        default_mean,
    )
    std = _parse_float_list(
        params.get("normalize_std")
        or params.get("normalization_std")
        or params.get("normalisation_std"),
        default_std,
    )

    if any(float(value) <= 0 for value in std):
        raise ValueError(f"Normalisation std values must be > 0, got {std!r}")

    return mean, std

def _append_normalize(ops, T, run, *, default_mode: str, default_mean, default_std) -> None:
    resolved = _normalization_params(
        run,
        default_mode=default_mode,
        default_mean=default_mean,
        default_std=default_std,
    )
    if resolved is None:
        return

    mean, std = resolved
    ops.append(T.Normalize(mean, std))

def _timm_transform(run, *, is_training: bool, cache_attr_owner=None):
    """Deterministic timm-compatible image transform.

    The platform contract is that every image recipe returns fixed-size tensors
    before DataLoader collation. Users can choose normalisation separately from
    architecture.
    """
    from torchvision import transforms as T

    size = int(run.params.get("input_size", 224) or 224)
    resize_size = max(size, int(round(size * 1.15)))

    if is_training:
        ops = [
            T.Resize((resize_size, resize_size)),
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    else:
        ops = [
            T.Resize((resize_size, resize_size)),
            T.CenterCrop(size),
            T.ToTensor(),
        ]

    _append_normalize(
        ops,
        T,
        run,
        default_mode="imagenet",
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
    )
    return T.Compose(ops)

def _make_optimizer(model, params):
    import torch.optim as optim
    kind = str(params.get("optimizer", "adamw")).lower()
    lr = float(params.get("learning_rate", 1e-3))
    wd = float(params.get("weight_decay", 5e-4))
    if kind == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=float(params.get("momentum", 0.9)),
                         weight_decay=wd, nesterov=True)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def _regression_criterion(params):
    import torch.nn as nn
    loss = str(params.get("loss", "mse")).lower()
    if loss in {"l1", "mae"}:
        return nn.L1Loss()
    if loss in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss(beta=float(params.get("huber_beta", 1.0)))
    return nn.MSELoss()

# =============================================================================
# 1. timm image classifier  (huggingface/pytorch-image-models)
#    One recipe, the whole zoo: resnet*, convnext*, vit_*, efficientnet_*, ...
# =============================================================================

class TimmImageClassifierRecipe(ManagedMLRecipe):
    id = "core.ml.timm_classifier"
    title = "timm image classifier"
    version = "0.1.0"
    task = "classification"
    modality = "image"
    framework = "torch"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Train/fine-tune any timm architecture (ResNet, ConvNeXt, ViT, "
        "EfficientNet, Swin, ...) selected by model_name, with optional "
        "ImageNet-pretrained weights. Validation, best-epoch selection and test "
        "evaluation are enforced by the protocol, not this code."
    )
    tags = ["torch", "timm", "image", "classification", "transfer-learning"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label", "image.path", "image.uri"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report",
                "ml.predictions", "ml.training_log", "ml.run"]

    params_schema = {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "default": "resnet50"},
            "pretrained": {"type": "boolean", "default": True},
            "input_size": {"type": "integer", "default": 224, "minimum": 16},
            "normalization": {
                "type": "string",
                "enum": ["imagenet", "cifar", "custom", "none"],
                "default": "imagenet",
                "title": "Normalisation preset",
            },
            "normalize_mean": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.485, 0.456, 0.406],
                "title": "Normalisation mean",
            },
            "normalize_std": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.229, 0.224, 0.225],
                "title": "Normalisation std",
            },
            "epochs": {"type": "integer", "default": 30, "minimum": 1},
            "batch_size": {"type": "integer", "default": 64, "minimum": 1},
            "num_workers": {"type": "integer", "default": 4, "minimum": 0},
            "optimizer": {"type": "string", "enum": ["adamw", "sgd"], "default": "adamw"},
            "learning_rate": {"type": "number", "default": 3e-4},
            "weight_decay": {"type": "number", "default": 1e-4},
            "momentum": {"type": "number", "default": 0.9},
            "label_smoothing": {"type": "number", "default": 0.1},
        },
    }

    def build_model(self, run, *, num_classes: int):
        import timm
        return timm.create_model(
            str(run.params.get("model_name", "resnet50")),
            pretrained=bool(run.params.get("pretrained", True)),
            num_classes=int(num_classes),     # timm honours this for the head
        )

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.nn as nn
        import torch.optim as optim
        p = run.params
        criterion = nn.CrossEntropyLoss(label_smoothing=float(p.get("label_smoothing", 0.1)))
        optimizer = _make_optimizer(model, p)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(p.get("epochs", 30)))
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=criterion)

    def train_transform(self, run):
        return _timm_transform(run, is_training=True, cache_attr_owner=self)

    def eval_transform(self, run):
        return _timm_transform(run, is_training=False, cache_attr_owner=self)

    def load_sample(self, run, row):
        return _open_image_from_row(run, row)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device
        model.to(device)
        for epoch in range(1, int(run.params.get("epochs", 30)) + 1):
            run.check_cancelled()
            model.train()
            loss_sum = correct = seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                components.optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = components.criterion(logits, targets)
                loss.backward()
                components.optimizer.step()
                loss_sum += float(loss.detach()) * int(targets.size(0))
                seen += int(targets.size(0))
                correct += int(logits.argmax(1).eq(targets).sum())
            harness.report_epoch(epoch, model, train_metrics={
                "loss": loss_sum / max(seen, 1),
                "accuracy": correct / max(seen, 1),
            })
            if components.scheduler is not None:
                components.scheduler.step()

# =============================================================================
# 2. WideResNet on CIFAR-size cutouts  (hysts/pytorch_image_classification)
#    SGD + cosine + Cutout; the strong small-image baseline. Fully consistent
#    with TorchClassificationHarness's 32px/CIFAR-norm model metadata.
# =============================================================================

class WideResNetCIFARRecipe(ManagedMLRecipe):
    id = "core.ml.wideresnet_cifar"
    title = "WideResNet (hysts, CIFAR-style)"
    version = "0.1.0"
    task = "classification"
    modality = "image"
    framework = "torch"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "WRN-28-10 style WideResNet for 32x32 cutouts, with SGD + cosine "
        "annealing and Cutout augmentation (after hysts/pytorch_image_"
        "classification). Protocol handles validation/selection/test."
    )
    tags = ["torch", "image", "classification", "cifar", "wideresnet"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label", "image.path", "image.uri"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report",
                "ml.predictions", "ml.training_log", "ml.run"]

    params_schema = {
        "type": "object",
        "properties": {
            "depth": {"type": "integer", "default": 28, "minimum": 10},
            "widen_factor": {"type": "integer", "default": 10, "minimum": 1},
            "dropout": {"type": "number", "default": 0.3, "minimum": 0.0},
            "input_size": {"type": "integer", "default": 32, "minimum": 16},
            "normalization": {
                "type": "string",
                "enum": ["cifar", "imagenet", "custom", "none"],
                "default": "cifar",
                "title": "Normalisation preset",
            },
            "normalize_mean": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.4914, 0.4822, 0.4465],
                "title": "Normalisation mean",
            },
            "normalize_std": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.2470, 0.2435, 0.2616],
                "title": "Normalisation std",
            },
            "epochs": {"type": "integer", "default": 200, "minimum": 1},
            "batch_size": {"type": "integer", "default": 128, "minimum": 1},
            "num_workers": {"type": "integer", "default": 4, "minimum": 0},
            "learning_rate": {"type": "number", "default": 0.1},
            "weight_decay": {"type": "number", "default": 5e-4},
            "momentum": {"type": "number", "default": 0.9},
            "cutout": {"type": "boolean", "default": True},
            "cutout_size": {"type": "integer", "default": 16, "minimum": 0},
        },
    }

    _MEAN = [0.4914, 0.4822, 0.4465]
    _STD = [0.2470, 0.2435, 0.2616]

    def build_model(self, run, *, num_classes: int):
        p = run.params
        return _build_wide_resnet(
            depth=int(p.get("depth", 28)),
            widen_factor=int(p.get("widen_factor", 10)),
            dropout=float(p.get("dropout", 0.3)),
            num_classes=int(num_classes),
        )

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.nn as nn
        import torch.optim as optim
        p = run.params
        optimizer = optim.SGD(
            model.parameters(), lr=float(p.get("learning_rate", 0.1)),
            momentum=float(p.get("momentum", 0.9)),
            weight_decay=float(p.get("weight_decay", 5e-4)), nesterov=True,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(p.get("epochs", 200)))
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler,
                                  criterion=nn.CrossEntropyLoss())

    def train_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params.get("input_size", 32) or 32)
        ops = [
            T.Resize((size, size)),
            T.RandomCrop(size, padding=max(1, size // 8)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        _append_normalize(
            ops,
            T,
            run,
            default_mode="cifar",
            default_mean=self._MEAN,
            default_std=self._STD,
        )

        if bool(run.params.get("cutout", True)):
            ops.append(_Cutout(int(run.params.get("cutout_size", max(1, size // 2)))))

        return T.Compose(ops)

    def eval_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params.get("input_size", 32) or 32)
        ops = [
            T.Resize((size, size)),
            T.ToTensor(),
        ]

        _append_normalize(
            ops,
            T,
            run,
            default_mode="cifar",
            default_mean=self._MEAN,
            default_std=self._STD,
        )

        return T.Compose(ops)

    def load_sample(self, run, row):
        return _open_image_from_row(run, row)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device
        model.to(device)
        for epoch in range(1, int(run.params.get("epochs", 200)) + 1):
            run.check_cancelled()
            model.train()
            loss_sum = correct = seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                components.optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = components.criterion(logits, targets)
                loss.backward()
                components.optimizer.step()
                loss_sum += float(loss.detach()) * int(targets.size(0))
                seen += int(targets.size(0))
                correct += int(logits.argmax(1).eq(targets).sum())
            harness.report_epoch(epoch, model, train_metrics={
                "loss": loss_sum / max(seen, 1),
                "accuracy": correct / max(seen, 1),
            })
            if components.scheduler is not None:
                components.scheduler.step()

# =============================================================================
# 3. timm image REGRESSOR  (timm backbone, Zoobot-style continuous targets)
#    e.g. photometric redshift / a morphology score from a cutout.
#    Selects TorchRegressionHarness via task="regression".
# =============================================================================

class TimmImageRegressorRecipe(ManagedMLRecipe):
    id = "core.ml.timm_regressor"
    title = "timm image regressor (Zoobot-style)"
    version = "0.1.0"
    task = "regression"
    modality = "image"
    framework = "torch"
    complexity = "advanced"
    author = "AstronomicAL"
    description = (
        "Regress a continuous property (e.g. photometric redshift) from image "
        "cutouts using any timm backbone with an n_outputs-wide head. Same "
        "transfer-learning idea Zoobot uses for galaxy targets. Protocol owns "
        "splitting/selection/test; selection defaults to val_loss (min)."
    )
    tags = ["torch", "timm", "image", "regression", "photoz", "transfer-learning"]
    required_mappings = ["record_id"]
    optional_mappings = ["target_label", "image.path", "image.uri"]
    produces = ["ml.split_spec", "ml.model", "ml.evaluation_report",
                "ml.predictions", "ml.training_log", "ml.run"]

    params_schema = {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "default": "efficientnet_b0"},
            "pretrained": {"type": "boolean", "default": True},
            "input_size": {"type": "integer", "default": 224, "minimum": 16},
            "normalization": {
                "type": "string",
                "enum": ["imagenet", "cifar", "custom", "none"],
                "default": "imagenet",
                "title": "Normalisation preset",
            },
            "normalize_mean": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.485, 0.456, 0.406],
                "title": "Normalisation mean",
            },
            "normalize_std": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0.229, 0.224, 0.225],
                "title": "Normalisation std",
            },
            "n_outputs": {"type": "integer", "default": 1, "minimum": 1},
            "loss": {"type": "string", "enum": ["mse", "mae", "huber"], "default": "mse"},
            "epochs": {"type": "integer", "default": 30, "minimum": 1},
            "batch_size": {"type": "integer", "default": 64, "minimum": 1},
            "num_workers": {"type": "integer", "default": 4, "minimum": 0},
            "optimizer": {"type": "string", "enum": ["adamw", "sgd"], "default": "adamw"},
            "learning_rate": {"type": "number", "default": 3e-4},
            "weight_decay": {"type": "number", "default": 1e-4},
        },
    }

    def build_model(self, run, *, num_classes: int):
        # num_classes == TargetSpec.num_outputs == params["n_outputs"] here.
        import timm
        return timm.create_model(
            str(run.params.get("model_name", "efficientnet_b0")),
            pretrained=bool(run.params.get("pretrained", True)),
            num_classes=int(num_classes),
        )

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.optim as optim
        p = run.params
        optimizer = _make_optimizer(model, p)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(p.get("epochs", 30)))
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler,
                                  criterion=_regression_criterion(p))

    def train_transform(self, run):
        return _timm_transform(run, is_training=True, cache_attr_owner=self)

    def eval_transform(self, run):
        return _timm_transform(run, is_training=False, cache_attr_owner=self)

    def load_sample(self, run, row):
        return _open_image_from_row(run, row)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device
        model.to(device)
        for epoch in range(1, int(run.params.get("epochs", 30)) + 1):
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


# =============================================================================
# Specialised image recipes built on the core image baselines.
# =============================================================================

def _with_properties(schema, **properties):
    updated = deepcopy(schema)
    updated.setdefault("properties", {}).update(properties)
    return updated

def _cutmix_batch(inputs, targets, *, alpha: float):
    import torch

    if alpha <= 0.0 or int(inputs.size(0)) < 2:
        return inputs, targets, targets, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    height, width = int(inputs.size(-2)), int(inputs.size(-1))
    cut_ratio = float((1.0 - lam) ** 0.5)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    center_y = int(torch.randint(0, height, (1,), device=inputs.device).item())
    center_x = int(torch.randint(0, width, (1,), device=inputs.device).item())
    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)

    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[permutation, :, y1:y2, x1:x2]
    area = max(0, y2 - y1) * max(0, x2 - x1)
    adjusted_lam = 1.0 - (area / max(height * width, 1))
    return mixed, targets, targets[permutation], float(adjusted_lam)

def _set_batch_norm_running_stats(model, *, enabled: bool) -> None:
    for module in model.modules():
        if not hasattr(module, "momentum"):
            continue
        if enabled:
            if hasattr(module, "_astronomical_backup_momentum"):
                module.momentum = module._astronomical_backup_momentum
        else:
            if not hasattr(module, "_astronomical_backup_momentum"):
                module._astronomical_backup_momentum = module.momentum
            module.momentum = 0

def _make_sam_optimizer(model, params):
    import torch

    class SAM(torch.optim.Optimizer):
        def __init__(self, parameters, *, rho, adaptive, **kwargs):
            if rho < 0.0:
                raise ValueError(f"SAM rho must be non-negative, got {rho}.")
            defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
            super().__init__(parameters, defaults)
            self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(defaults)

        @torch.no_grad()
        def first_step(self, *, zero_grad: bool = False) -> None:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    multiplier = parameter.abs() if group["adaptive"] else 1.0
                    perturbation = multiplier * parameter.grad * scale.to(parameter)
                    parameter.add_(perturbation)
                    self.state[parameter]["sam_perturbation"] = perturbation
            if zero_grad:
                self.zero_grad(set_to_none=True)

        @torch.no_grad()
        def second_step(self, *, zero_grad: bool = False) -> None:
            for group in self.param_groups:
                for parameter in group["params"]:
                    perturbation = self.state[parameter].pop("sam_perturbation", None)
                    if perturbation is not None:
                        parameter.sub_(perturbation)
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad(set_to_none=True)

        def step(self, closure=None):
            if closure is None:
                raise RuntimeError("SAM.step requires a closure with a forward/backward pass.")
            closure = torch.enable_grad()(closure)
            self.first_step(zero_grad=True)
            closure()
            self.second_step(zero_grad=True)

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norms = []
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    multiplier = parameter.abs() if group["adaptive"] else 1.0
                    norms.append((multiplier * parameter.grad).norm(p=2).to(shared_device))
            if not norms:
                return torch.zeros((), device=shared_device)
            return torch.norm(torch.stack(norms), p=2)

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups

    return SAM(
        model.parameters(),
        rho=float(params.get("rho", 0.05)),
        adaptive=bool(params.get("adaptive", False)),
        lr=float(params.get("learning_rate", 0.1)),
        momentum=float(params.get("momentum", 0.9)),
        weight_decay=float(params.get("weight_decay", 5e-4)),
        nesterov=True,
    )

class CutMixTimmClassifierRecipe(TimmImageClassifierRecipe):
    id = "core.ml.cutmix_timm_classifier"
    title = "timm image classifier with CutMix"
    version = "0.1.0"
    complexity = "advanced"
    description = (
        "Fine-tune a timm image classifier with the official CutMix minibatch "
        "mixing rule. AstronomicAL still owns validation, best-epoch selection, "
        "test evaluation, and artifact production."
    )
    tags = ["torch", "timm", "image", "classification", "cutmix", "regularization"]
    source_urls = [
        "https://github.com/clovaai/CutMix-PyTorch",
        "https://github.com/huggingface/pytorch-image-models",
    ]
    source_reference = (
        "CutMix sampling and area-corrected lambda are adapted from the official "
        "CutMix-PyTorch training loop; model construction remains timm-backed."
    )
    params_schema = _with_properties(
        TimmImageClassifierRecipe.params_schema,
        cutmix_alpha={
            "type": "number",
            "default": 1.0,
            "minimum": 0.0,
            "title": "CutMix beta-distribution alpha",
        },
        cutmix_probability={
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "title": "CutMix probability",
        },
        gradient_clip_norm={
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "title": "Gradient clipping norm (0 disables)",
        },
    )

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch

        device = harness.device
        model.to(device)
        epochs = int(run.params.get("epochs", 30))
        probability = float(run.params.get("cutmix_probability", 0.5))
        alpha = float(run.params.get("cutmix_alpha", 1.0))
        clip_norm = float(run.params.get("gradient_clip_norm", 0.0))

        for epoch in range(1, epochs + 1):
            run.check_cancelled()
            model.train()
            loss_sum = 0.0
            weighted_correct = 0.0
            seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                use_cutmix = alpha > 0.0 and float(torch.rand(()).item()) < probability
                if use_cutmix:
                    mixed, target_a, target_b, lam = _cutmix_batch(inputs, targets, alpha=alpha)
                else:
                    mixed, target_a, target_b, lam = inputs, targets, targets, 1.0

                components.optimizer.zero_grad(set_to_none=True)
                logits = model(mixed)
                loss = (
                    lam * components.criterion(logits, target_a)
                    + (1.0 - lam) * components.criterion(logits, target_b)
                )
                loss.backward()
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                components.optimizer.step()

                batch_size = int(targets.size(0))
                predictions = logits.argmax(1)
                loss_sum += float(loss.detach()) * batch_size
                weighted_correct += lam * float(predictions.eq(target_a).sum())
                weighted_correct += (1.0 - lam) * float(predictions.eq(target_b).sum())
                seen += batch_size

            harness.report_epoch(
                epoch,
                model,
                train_metrics={
                    "loss": loss_sum / max(seen, 1),
                    "accuracy": weighted_correct / max(seen, 1),
                },
            )
            if components.scheduler is not None:
                components.scheduler.step()

class SAMWideResNetCIFARRecipe(WideResNetCIFARRecipe):
    id = "core.ml.sam_wideresnet_cifar"
    title = "WideResNet with SAM (CIFAR-style)"
    version = "0.1.0"
    complexity = "advanced"
    description = (
        "Train the existing CIFAR WideResNet with Sharpness-Aware Minimization "
        "using the two-step SAM update and batch-normalization handling from the "
        "reference PyTorch implementation."
    )
    tags = ["torch", "image", "classification", "cifar", "wideresnet", "sam"]
    source_urls = [
        "https://github.com/davda54/sam",
        "https://github.com/hysts/pytorch_image_classification",
    ]
    source_reference = (
        "Uses the SAM two-forward-pass optimizer pattern with rho=0.05 by default, "
        "on AstronomicAL's existing WideResNet/Cutout recipe."
    )
    params_schema = _with_properties(
        WideResNetCIFARRecipe.params_schema,
        rho={"type": "number", "default": 0.05, "minimum": 0.0, "title": "SAM rho"},
        adaptive={"type": "boolean", "default": False, "title": "Use adaptive SAM"},
        label_smoothing={
            "type": "number",
            "default": 0.1,
            "minimum": 0.0,
            "maximum": 1.0,
        },
    )

    def configure_training(self, run, model) -> TrainingComponents:
        import torch.nn as nn
        import torch.optim as optim

        optimizer = _make_sam_optimizer(model, run.params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(run.params.get("epochs", 200)),
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(run.params.get("label_smoothing", 0.1))
        )
        return TrainingComponents(optimizer=optimizer, scheduler=scheduler, criterion=criterion)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        device = harness.device
        model.to(device)
        epochs = int(run.params.get("epochs", 200))

        for epoch in range(1, epochs + 1):
            run.check_cancelled()
            model.train()
            loss_sum = 0.0
            correct = 0
            seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()

                _set_batch_norm_running_stats(model, enabled=True)
                logits = model(inputs)
                loss = components.criterion(logits, targets)
                loss.backward()
                components.optimizer.first_step(zero_grad=True)

                _set_batch_norm_running_stats(model, enabled=False)
                second_logits = model(inputs)
                second_loss = components.criterion(second_logits, targets)
                second_loss.backward()
                components.optimizer.second_step(zero_grad=True)
                _set_batch_norm_running_stats(model, enabled=True)

                batch_size = int(targets.size(0))
                loss_sum += float(loss.detach()) * batch_size
                correct += int(logits.argmax(1).eq(targets).sum())
                seen += batch_size

            harness.report_epoch(
                epoch,
                model,
                train_metrics={
                    "loss": loss_sum / max(seen, 1),
                    "accuracy": correct / max(seen, 1),
                },
            )
            if components.scheduler is not None:
                components.scheduler.step()

class ZoobotFineTuneImageRegressorRecipe(TimmImageRegressorRecipe):
    id = "core.ml.zoobot_finetune_regressor"
    title = "Zoobot-style image regressor fine-tuning"
    version = "0.1.0"
    complexity = "advanced"
    description = (
        "Fine-tune an ImageNet-pretrained timm encoder for continuous image targets "
        "with astronomy-friendly rotations/flips and an optional frozen-backbone "
        "warm-up. This follows Zoobot's reusable encoder/fine-tuning approach while "
        "retaining AstronomicAL's managed evaluation protocol."
    )
    tags = ["torch", "timm", "image", "regression", "zoobot", "astronomy", "transfer-learning"]
    source_urls = [
        "https://github.com/mwalmsley/zoobot",
        "https://github.com/huggingface/pytorch-image-models",
    ]
    source_reference = (
        "Inspired by Zoobot's pretrained-encoder fine-tuning workflow and support "
        "for regression tasks; implemented with a timm backbone inside AstronomicAL."
    )
    params_schema = _with_properties(
        TimmImageRegressorRecipe.params_schema,
        model_name={"type": "string", "default": "convnext_tiny"},
        epochs={"type": "integer", "default": 40, "minimum": 1},
        batch_size={"type": "integer", "default": 32, "minimum": 1},
        learning_rate={"type": "number", "default": 3e-4},
        freeze_backbone_epochs={
            "type": "integer",
            "default": 3,
            "minimum": 0,
            "title": "Frozen-backbone warm-up epochs",
        },
        random_rotation_degrees={
            "type": "number",
            "default": 180.0,
            "minimum": 0.0,
            "maximum": 180.0,
        },
        vertical_flip={"type": "boolean", "default": True},
    )

    def build_model(self, run, *, num_classes: int):
        model = super().build_model(run, num_classes=num_classes)
        freeze_epochs = int(run.params.get("freeze_backbone_epochs", 3))
        if freeze_epochs <= 0:
            return model
        for parameter in model.parameters():
            parameter.requires_grad = False
        classifier = model.get_classifier()
        if classifier is None:
            raise ValueError("Selected timm model does not expose a classifier head.")
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        return model

    def train_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params.get("input_size", 224) or 224)
        resize_size = max(size, int(round(size * 1.15)))
        ops = [
            T.Resize((resize_size, resize_size)),
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
        ]
        if bool(run.params.get("vertical_flip", True)):
            ops.append(T.RandomVerticalFlip())
        rotation = float(run.params.get("random_rotation_degrees", 180.0))
        if rotation > 0.0:
            ops.append(T.RandomRotation(rotation))
        ops.append(T.ToTensor())
        _append_normalize(
            ops,
            T,
            run,
            default_mode="imagenet",
            default_mean=[0.485, 0.456, 0.406],
            default_std=[0.229, 0.224, 0.225],
        )
        return T.Compose(ops)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        device = harness.device
        model.to(device)
        epochs = int(run.params.get("epochs", 40))
        freeze_epochs = int(run.params.get("freeze_backbone_epochs", 3))

        for epoch in range(1, epochs + 1):
            run.check_cancelled()
            if epoch == freeze_epochs + 1:
                for parameter in model.parameters():
                    parameter.requires_grad = True
            model.train()
            loss_sum = 0.0
            seen = 0
            for inputs, targets, _ids in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                components.optimizer.zero_grad(set_to_none=True)
                output = model(inputs)
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                loss = components.criterion(output, targets)
                loss.backward()
                components.optimizer.step()
                batch_size = int(targets.size(0))
                loss_sum += float(loss.detach()) * batch_size
                seen += batch_size

            harness.report_epoch(
                epoch,
                model,
                train_metrics={"loss": loss_sum / max(seen, 1)},
            )
            if components.scheduler is not None:
                components.scheduler.step()
