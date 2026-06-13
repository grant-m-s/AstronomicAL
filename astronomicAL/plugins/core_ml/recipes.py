from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

# Framework types now live in recipe_registry.py. recipes.py holds ONLY
# concrete recipes — it must not redefine Partition/Partitions/RunHarness/
# MLRecipe/TrainingComponents (doing so shadows the real ones and breaks
# make_harness()).
from .recipe_registry import (
    MLRecipe,            # freeform base (overrides run())
    ManagedMLRecipe,     # protocol-managed base (harness owns run())
    MLRunContext,
    RunHarness,
    TrainingComponents,
)


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
# selection, and artifact writing are the harness's — see recipe_registry.py.
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