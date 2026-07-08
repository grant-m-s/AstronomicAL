# Core ML recipes

The recipe layer exists because high-performance ML cannot be represented fully
by the Model Builder panel.

The Model Builder is still useful for simple sklearn/Torch definitions. Recipes
are for setups that need real code:

- custom Dataset/DataLoader implementations
- torchvision or domain-specific transforms
- schedulers such as CosineAnnealingLR
- multiple optimizers
- AMP
- checkpointing
- callbacks
- custom losses
- custom evaluation
- active-learning loops
- segmentation/multimodal pipelines

## User model

A recipe exposes a small typed schema to the UI, but the implementation remains
Python code.

Non-ML users launch recipes from the ML Recipe Launcher.

ML developers write recipes as Python classes and register them in
`core.ml.recipe_registry`, or use the `External Python ML recipe` bridge.

## External recipe trust boundary

The `External Python ML recipe` imports and executes local Python in the same process as AstronomicAL.

Treat it like running a Python script from your shell: use only code you trust. AstronomicAL does not sandbox this code and does not enforce the managed train/validation/test protocol for it.

Use a managed recipe when you need AstronomicAL to own splitting, validation, test evaluation, provenance, and protocol checks.

## External recipe example

```python
from astronomicAL.plugins.core_ml.recipe_registry import MLRecipe, MLRunContext


class MyRecipe(MLRecipe):
    id = "my_lab.my_recipe"
    title = "My lab recipe"
    version = "0.1.0"
    task = "classification"
    modality = "image"

    params_schema = {
        "type": "object",
        "required": ["image_column", "target_column"],
        "properties": {
            "image_column": {
                "type": "string",
                "default": "",
            },
            "target_column": {
                "type": "string",
                "default": "",
            },
            "epochs": {
                "type": "integer",
                "default": 50,
            },
        },
    }

    def run(self, run: MLRunContext):
        run.log(message="Starting custom training loop.")

        # Implement arbitrary PyTorch/sklearn/domain code here.
        # Use run.context.datasets, run.put_artifact(), run.log(),
        # run.publish(), and run.check_cancelled() to integrate with
        # AstronomicAL.

        return {
            "status": "complete",
        }
```

Then launch it from the **ML Recipe Launcher** panel with:

```text
Recipe: External Python ML recipe
import_path: my_package.my_module.MyRecipe
kwargs_json: {}
```

## CIFAR-style recipe usage

The built-in `CIFAR-style Torch image classifier` recipe is intended for
high-performance image-classification setups that need real PyTorch behavior
without exposing the full training loop to non-ML users.

Example configuration:

```text
Recipe: CIFAR-style Torch image classifier
architecture: torchvision.resnet18
augmentation_preset: cifar_standard
optimizer: sgd
learning_rate: 0.1
momentum: 0.9
weight_decay: 0.0005
scheduler: cosine
cosine_t_max: 200
epochs: 200
batch_size: 128
normalization: cifar10
```

To use a custom model implementation, install or copy the model code into an
importable Python package and set:

```text
architecture: custom_import
custom_model_import: local_ml_recipes.pytorch_cifar.models.ResNet18
```

The callable at `custom_model_import` should return a `torch.nn.Module`. It may
accept `num_classes` as a keyword argument.

## Design principle

The panel should not try to represent every possible training loop.

Instead:

```text
Panel = choose/configure/launch
Recipe = real expert ML code
Platform = jobs/artifacts/events/datasets/provenance
```

This keeps expert ML implementations runnable, inspectable, reproducible,
configurable, and reusable by non-expert users.