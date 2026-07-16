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
    XGBoostTabularClassifierRecipe,
    XGBoostTabularRegressorRecipe,
)

BUILTIN_RECIPE_CLASSES = (
    *IMAGE_RECIPE_CLASSES,
    *TORCH_TABULAR_RECIPE_CLASSES,
    *SKLEARN_RECIPE_CLASSES,
)

__all__ = [
    "BUILTIN_RECIPE_CLASSES",
    "IMAGE_RECIPE_CLASSES",
    "SKLEARN_RECIPE_CLASSES",
    "TORCH_TABULAR_RECIPE_CLASSES",
]
