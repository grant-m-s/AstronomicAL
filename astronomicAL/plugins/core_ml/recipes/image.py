from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

# Framework types now live in core_ml recipe framework. recipes.py holds ONLY
# concrete recipes — it must not redefine Partition/Partitions/RunHarness/
# MLRecipe/TrainingComponents (doing so shadows the real ones and breaks
# make_harness()).
from ..recipe_base import MLRecipe, ManagedMLRecipe, MLRunContext
from ..harnesses.base import RunHarness
from ..protocol import TrainingComponents

def _import_object(path: str):
    module_name, object_name = path.rsplit(".", 1)
    return getattr(import_module(module_name), object_name)

# =============================================================================
# Freeform escape hatch — NOT protocol-managed. Overrides run() directly and is
# trusted to do its own thing. The launcher does not offer it the protocol UI,
# and the runner does not enforce record_id on it, precisely because it makes
# no scientific-validity promise. Use it to bridge arbitrary expert code.
# =============================================================================

class ExternalPythonRecipe(MLRecipe):
    id = "core.ml.external_python_recipe"
    title = "External Python ML recipe"
    version = "0.2.0"
    task = "custom"
    modality = "custom"
    complexity = "expert"
    execution_mode = "freeform"
    author = "AstronomicAL"
    description = (
        "Run a recipe implemented in an installed/local Python module. "
        "This is the unmanaged escape hatch: it bypasses the validation "
        "protocol and is trusted to manage its own splits and evaluation."
    )
    tags = ["expert", "python", "extension", "recipe"]
    required_mappings: list = []
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
            return dict(obj(run) or {})

        raise TypeError(
            "import_path must point to an MLRecipe subclass, MLRecipe instance, "
            "or callable accepting MLRunContext.")

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
        ops = []
        aug = str(run.params.get("augmentation", "cifar_standard"))
        if aug == "cifar_standard":
            ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        elif aug == "randaugment":
            ops += [T.RandAugment(), T.RandomHorizontalFlip()]
        ops += [T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
        return T.Compose(ops)

    def eval_transform(self, run):
        from torchvision import transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

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
# Add to recipes.py. Four protocol-managed recipes:
#
#   TimmImageClassifierRecipe   timm (huggingface/pytorch-image-models)  image / classification
#   WideResNetCIFARRecipe       hysts/pytorch_image_classification       image / classification
#   TimmImageRegressorRecipe    timm backbone, Zoobot-style targets      image / regression
#   TabularMLPRegressorRecipe   rtdl / pytorch-tabular deep baseline     tabular / regression
#
# Each implements ONLY internals (build_model / configure_training / transforms /
# load_sample / fit). Splitting, validation selection, best-epoch choice, test
# evaluation and artifact writing belong to the harness:
#   classification -> TorchClassificationHarness
#   regression     -> TorchRegressionHarness  (selected by _is_torch_regression)
#
# Torch/timm imports stay inside methods so importing recipes.py never requires
# torch (matching the existing module discipline).
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

def _timm_transform(run, *, is_training: bool, cache_attr_owner=None):
    """timm's own train/eval transform when resolvable, else an ImageNet fallback.
    Self-contained so it also works at predict time (build_model is not called
    there)."""
    name = str(run.params.get("model_name", "resnet50"))
    size = int(run.params.get("input_size", 224) or 224)
    try:
        import timm
        from timm.data import resolve_data_config, create_transform
        cfg = None
        if cache_attr_owner is not None and getattr(cache_attr_owner, "_timm_cfg", None):
            cfg = cache_attr_owner._timm_cfg
        if cfg is None:
            probe = timm.create_model(name, pretrained=False)
            cfg = resolve_data_config({}, model=probe)
            if size:
                cfg["input_size"] = (cfg["input_size"][0], size, size)
            if cache_attr_owner is not None:
                cache_attr_owner._timm_cfg = cfg
        return create_transform(**cfg, is_training=is_training)
    except Exception:
        from torchvision import transforms as T
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if is_training:
            return T.Compose([
                T.RandomResizedCrop(size), T.RandomHorizontalFlip(),
                T.ToTensor(), T.Normalize(mean, std),
            ])
        return T.Compose([
            T.Resize(int(round(size * 1.15))), T.CenterCrop(size),
            T.ToTensor(), T.Normalize(mean, std),
        ])

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
        ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
               T.ToTensor(), T.Normalize(self._MEAN, self._STD)]
        if bool(run.params.get("cutout", True)):
            ops.append(_Cutout(int(run.params.get("cutout_size", 16))))
        return T.Compose(ops)

    def eval_transform(self, run):
        from torchvision import transforms as T
        return T.Compose([T.ToTensor(), T.Normalize(self._MEAN, self._STD)])

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
# 4. Tabular MLP REGRESSOR  (rtdl baseline / pytorch-tabular family)
#    Photometry/catalogue features -> continuous target. Input BatchNorm means
#    no external scaler is needed. Selects TorchRegressionHarness.
# =============================================================================

class TabularMLPRegressorRecipe(ManagedMLRecipe):
    id = "core.ml.tabular_mlp_regressor"
    title = "Tabular MLP regressor (rtdl baseline)"
    version = "0.1.0"
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
                                  criterion=_regression_criterion(p))

    def train_transform(self, run):
        return None        # load_sample already returns a ready feature tensor

    def eval_transform(self, run):
        return None

    def load_sample(self, run, row):
        import torch
        feats = [c for c in (run.binding.input_columns or []) if c]
        return torch.tensor([float(row[c]) for c in feats], dtype=torch.float32)

    def fit(self, run, *, model, components: TrainingComponents, train_loader, harness: RunHarness):
        import torch
        device = harness.device
        model.to(device)
        for epoch in range(1, int(run.params.get("epochs", 200)) + 1):
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
