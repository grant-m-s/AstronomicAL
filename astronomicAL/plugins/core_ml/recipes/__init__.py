from __future__ import annotations

from .image import (
    CIFARResNetRecipe,
    CutMixTimmClassifierRecipe,
    SAMWideResNetCIFARRecipe,
    TimmImageClassifierRecipe,
    TimmImageRegressorRecipe,
    WideResNetCIFARRecipe,
    ZoobotFineTuneImageRegressorRecipe,
)
from .sklearn import (
    IncrementalSGDClassifierRecipe,
    IncrementalSGDRegressorRecipe,
    SklearnTabularClassifierRecipe,
    SklearnTabularRegressorRecipe,
    XGBoostTabularClassifierRecipe,
    XGBoostTabularRegressorRecipe,
)
from .tabular import (
    FTTransformerClassifierRecipe,
    FTTransformerRegressorRecipe,
    TabularMLPRegressorRecipe,
    TabularResNetClassifierRecipe,
    TabularResNetRegressorRecipe,
)

IMAGE_RECIPE_CLASSES = (
    CIFARResNetRecipe,
    TimmImageClassifierRecipe,
    WideResNetCIFARRecipe,
    CutMixTimmClassifierRecipe,
    SAMWideResNetCIFARRecipe,
    TimmImageRegressorRecipe,
    ZoobotFineTuneImageRegressorRecipe,
)

TORCH_TABULAR_RECIPE_CLASSES = (
    TabularMLPRegressorRecipe,
    FTTransformerClassifierRecipe,
    FTTransformerRegressorRecipe,
    TabularResNetClassifierRecipe,
    TabularResNetRegressorRecipe,
)

SKLEARN_RECIPE_CLASSES = (
    SklearnTabularClassifierRecipe,
    SklearnTabularRegressorRecipe,
    IncrementalSGDClassifierRecipe,
    IncrementalSGDRegressorRecipe,
    XGBoostTabularClassifierRecipe,
    XGBoostTabularRegressorRecipe,
)

BUILTIN_RECIPE_CLASSES = (
    *IMAGE_RECIPE_CLASSES,
    *TORCH_TABULAR_RECIPE_CLASSES,
    *SKLEARN_RECIPE_CLASSES,
)

_BUILTIN_RECIPE_IDS = tuple(recipe_class.id for recipe_class in BUILTIN_RECIPE_CLASSES)

if len(_BUILTIN_RECIPE_IDS) != len(set(_BUILTIN_RECIPE_IDS)):
    duplicates = sorted(
        recipe_id
        for recipe_id in set(_BUILTIN_RECIPE_IDS)
        if _BUILTIN_RECIPE_IDS.count(recipe_id) > 1
    )
    raise RuntimeError(
        "Duplicate built-in ML recipe IDs: "
        + ", ".join(repr(recipe_id) for recipe_id in duplicates)
    )

__all__ = [
    "BUILTIN_RECIPE_CLASSES",
    "IMAGE_RECIPE_CLASSES",
    "SKLEARN_RECIPE_CLASSES",
    "TORCH_TABULAR_RECIPE_CLASSES",
]