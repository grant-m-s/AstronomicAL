# Writing `core_ml` Recipes

A `core_ml` recipe turns an existing model and training loop into a managed AstronomicAL experiment.

The key rule is:

> **The recipe owns the learning method. The harness owns the experiment.**

Your recipe defines the model, transforms, optimizer, loss, scheduler, and training step. The harness handles dataset splitting, validation, test isolation, best-model selection, artifacts, pause/resume, and cleanup.

---

## The boundary

A normal training script often does all of this:

```text
load data → split data → train → validate → test → save best model
```

A managed recipe only does:

```text
build model → configure training → train on train_loader → report each epoch
```

The harness does:

```text
bindings and splits
validation after each epoch
best-epoch selection
test evaluation
model/prediction/log artifacts
pause and resume
record-ID preservation
```

This makes different recipes comparable under the same experiment protocol.

Most importantly:

> A recipe receives the training loader. It does not receive validation or test loaders.

---

## Minimal complete recipe

```python
from pathlib import Path

from astronomicAL.plugins.core_ml.protocol import TrainingComponents
from astronomicAL.plugins.core_ml.recipe_base import ManagedMLRecipe


class SmallImageClassifierRecipe(ManagedMLRecipe):
    id = "example.small_image_classifier"
    title = "Small image classifier"
    version = "1.0.0"

    task = "classification"
    modality = "image"
    framework = "torch"

    required_imports = ["torch", "torchvision", "PIL"]
    required_mappings = ["record_id"]
    optional_mappings = [
        "target_label",
        "image.path",
        "image.uri",
    ]

    params_schema = {
        "type": "object",
        "properties": {
            "epochs": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
            },
            "batch_size": {
                "type": "integer",
                "default": 64,
                "minimum": 1,
            },
            "learning_rate": {
                "type": "number",
                "default": 0.001,
                "exclusiveMinimum": 0,
            },
            "image_size": {
                "type": "integer",
                "default": 64,
                "minimum": 16,
            },
        },
    }

    def build_model(self, run, *, num_classes: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def configure_training(self, run, model):
        import torch.nn as nn
        import torch.optim as optim

        return TrainingComponents(
            optimizer=optim.Adam(
                model.parameters(),
                lr=float(run.params["learning_rate"]),
            ),
            criterion=nn.CrossEntropyLoss(),
        )

    def train_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params["image_size"])

        return T.Compose(
            [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

    def eval_transform(self, run):
        from torchvision import transforms as T

        size = int(run.params["image_size"])

        return T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
            ]
        )

    def load_sample(self, run, row):
        from PIL import Image

        column = run.binding.image_column
        if not column:
            raise ValueError(
                "Map an image column to `image.path`."
            )

        value = str(row[column]).strip()
        if value.startswith("file://"):
            value = value[7:]

        return Image.open(Path(value)).convert("RGB")

    def fit(
        self,
        run,
        *,
        model,
        components,
        train_loader,
        harness,
    ):
        device = harness.device
        model.to(device)

        for epoch in run.epoch_range(
            int(run.params["epochs"])
        ):
            run.check_cancelled()
            model.train()

            loss_sum = 0.0
            correct = 0
            seen = 0

            for inputs, targets, _ids in train_loader:
                run.check_cancelled()

                inputs = inputs.to(device)
                targets = targets.to(device).long()

                components.optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(inputs)
                loss = components.criterion(
                    logits,
                    targets,
                )

                loss.backward()
                components.optimizer.step()

                batch_size = int(targets.size(0))
                loss_sum += float(loss.detach()) * batch_size
                correct += int(
                    logits.argmax(1).eq(targets).sum()
                )
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

            harness.check_pause_boundary(
                epoch,
                model,
                components,
            )
```

### Why it looks like this

- `build_model()` receives the resolved output width.
- `load_sample()` uses dataset bindings instead of hard-coded columns.
- `fit()` receives only `train_loader`.
- `run.epoch_range()` makes resume work.
- `harness.report_epoch()` delegates validation and best-model selection.
- `harness.check_pause_boundary()` creates a consistent resume point.

There is deliberately no split code, validation loop, test loop, or checkpoint-selection logic.

---

# Converting an external repository

This example follows the structure of a CIFAR repository such as `kuangliu/pytorch-cifar`.

A standalone script often looks like:

```python
net = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=200,
)

for epoch in range(start_epoch, 200):
    train(epoch)
    test(epoch)

    if test_accuracy > best_accuracy:
        save_checkpoint(net)

    scheduler.step()
```

## What moves where

| External code | Recipe conversion |
|---|---|
| CLI arguments | `params_schema` |
| Model factory | `build_model()` |
| Optimizer/loss/scheduler | `configure_training()` |
| Train augmentation | `train_transform()` |
| Test preprocessing | `eval_transform()` |
| Dataset and loaders | Harness |
| `train(epoch)` | `fit()` |
| `test(epoch)` | Harness |
| Best checkpoint | Harness |
| `--resume` | Managed pause/resume |

The rule is simple:

> Keep the learning algorithm. Remove the experiment protocol.

---

## 1. Import reusable model code

Do not import an external `main.py`; it may parse arguments or start training.

Import only reusable components:

```python
from third_party.kuangliu_pytorch_cifar.models.resnet import (
    ResNet18,
)
```

Keep the upstream license and required attribution with vendored code.

---

## 2. Replace hard-coded output classes

The external model may default to ten outputs.

```python
def build_model(self, run, *, num_classes: int):
    import torch.nn as nn

    from third_party.kuangliu_pytorch_cifar.models.resnet import (
        ResNet18,
    )

    model = ResNet18()

    if not hasattr(model, "linear"):
        raise TypeError(
            "Expected the imported model to expose `linear`."
        )

    model.linear = nn.Linear(
        model.linear.in_features,
        num_classes,
    )

    return model
```

The harness also checks the output width before training.

---

## 3. Move optimizer setup

```python
def configure_training(self, run, model):
    import torch.nn as nn
    import torch.optim as optim

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(run.params["learning_rate"]),
        momentum=float(run.params["momentum"]),
        weight_decay=float(
            run.params["weight_decay"]
        ),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(run.params["epochs"]),
    )

    return TrainingComponents(
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
    )
```

---

## 4. Keep train and evaluation transforms separate

```python
def train_transform(self, run):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def eval_transform(self, run):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
```

The harness applies the correct transform to each partition.

---

## 5. Replace external dataset construction

Instead of creating `CIFAR10(...)` inside the recipe, read the selected AstronomicAL dataset:

```python
def load_sample(self, run, row):
    from PIL import Image

    path = str(
        row[run.binding.image_column]
    ).strip()

    if path.startswith("file://"):
        path = path[7:]

    return Image.open(path).convert("RGB")
```

A compatible dataset might contain:

```text
object_id    image_path          class_label
1001         /data/1001.png      galaxy
1002         /data/1002.png      star
```

with mappings:

```text
record_id     -> object_id
image.path    -> image_path
target_label  -> class_label
```

---

## 6. Convert the training loop

The batch update remains almost unchanged:

```python
def fit(
    self,
    run,
    *,
    model,
    components,
    train_loader,
    harness,
):
    device = harness.device
    model.to(device)

    for epoch in run.epoch_range(
        int(run.params["epochs"])
    ):
        run.check_cancelled()
        model.train()

        loss_sum = 0.0
        correct = 0
        seen = 0

        for inputs, targets, _ids in train_loader:
            run.check_cancelled()

            inputs = inputs.to(device)
            targets = targets.to(device).long()

            components.optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(inputs)
            loss = components.criterion(
                outputs,
                targets,
            )

            loss.backward()
            components.optimizer.step()

            batch_size = int(targets.size(0))
            loss_sum += float(loss.detach()) * batch_size
            correct += int(
                outputs.argmax(1).eq(targets).sum()
            )
            seen += batch_size

        harness.report_epoch(
            epoch,
            model,
            train_metrics={
                "loss": loss_sum / max(seen, 1),
                "accuracy": correct / max(seen, 1),
            },
        )

        components.scheduler.step()

        harness.check_pause_boundary(
            epoch,
            model,
            components,
        )
```

The external `test(epoch)` and best-checkpoint block are removed.

After each report, the harness validates and updates the best state. After training, it restores that state, evaluates test once, and writes the standard artifacts.

---

## Register the recipe

For a built-in recipe:

1. place it under:

   ```text
   astronomicAL/plugins/core_ml/recipes/
   ```

2. import it from:

   ```text
   astronomicAL/plugins/core_ml/recipes/__init__.py
   ```

3. add it to the appropriate built-in recipe group.

For a recipe supplied by another plugin:

```python
registry = context.services.get(
    "core.ml.recipe_registry"
)

registry.register(
    ExternalKuangliuResNetRecipe,
    replace=False,
)
```

Recipe IDs must be unique and stable.

---

## Test the important contracts

At minimum, test:

### Registration

```python
spec = registry.register(
    ExternalKuangliuResNetRecipe
)

assert spec.framework == "torch"
assert spec.task == "classification"
```

### Output width

```python
model = recipe.build_model(
    run,
    num_classes=3,
)

output = model(
    torch.randn(2, 3, 32, 32)
)

assert output.shape == (2, 3)
```

### Tiny managed run

Use one or two epochs and verify:

- split IDs are disjoint;
- the configured validation metric appears;
- `ml.model` is produced;
- predictions preserve exact record IDs;
- pause produces `ml.resume_checkpoint`;
- resume starts at the next epoch;
- the saved model loads in the predictor.

---

<details>
<summary><strong>Common adaptations</strong></summary>

### Model returns a tuple or dictionary

```python
def eval_forward(self, model, inputs):
    output = model(inputs)

    if isinstance(output, dict):
        return output["logits"]

    if isinstance(output, tuple):
        return output[0]

    return output
```

### Metric-driven scheduler

```python
val_metrics = harness.report_epoch(
    epoch,
    model,
    train_metrics=train_metrics,
)

components.scheduler.step(
    val_metrics["loss"]
)

harness.check_pause_boundary(
    epoch,
    model,
    components,
)
```

### Additional resumable state

Put stateful objects in `TrainingComponents.extra`:

```python
return TrainingComponents(
    optimizer=optimizer,
    criterion=criterion,
    extra={
        "grad_scaler": scaler,
        "aux_optimizer": aux_optimizer,
    },
)
```

</details>

---

## Avoid these mistakes

Do not:

- split data inside the recipe;
- create validation or test loaders;
- select or save the best model in `fit()`;
- hard-code dataset column names;
- use `range(epochs)` instead of `run.epoch_range()`;
- keep the external repository's competing resume system;
- omit `harness.report_epoch()`;
- omit `harness.check_pause_boundary()`.

---