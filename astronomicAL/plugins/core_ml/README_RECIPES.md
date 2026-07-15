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

A recipe exposes a small typed schema to the UI, while the implementation
remains Python code.

Non-ML users launch recipes from the ML Recipe Launcher.

ML developers implement recipes by subclassing `ManagedMLRecipe` and
registering the class with `core.ml.recipe_registry`.

AstronomicAL owns dataset binding, splitting, validation, model selection,
test evaluation, artifacts and provenance. The recipe owns model
construction, transforms, training components, sample loading and the
training loop.

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