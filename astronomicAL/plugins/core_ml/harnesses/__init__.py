from __future__ import annotations

from .base import RunHarness
from .torch_classification import TorchClassificationHarness
from .torch_regression import TorchRegressionHarness

_HARNESSES = []


def register_harness(predicate, harness_cls):
    """Register a managed-run harness predicate. First match wins."""
    _HARNESSES.insert(0, (predicate, harness_cls))


def make_harness(run, recipe) -> RunHarness:
    framework = str(run.params.get("framework") or getattr(recipe, "framework", "") or "").lower()
    task = str(getattr(recipe, "task", "") or run.params.get("task", "") or "").lower()
    modality = str(getattr(recipe, "modality", "") or run.params.get("modality", "") or "").lower()

    for predicate, harness_cls in list(_HARNESSES):
        try:
            if predicate(framework=framework, task=task, modality=modality, run=run, recipe=recipe):
                return harness_cls(run, recipe)
        except TypeError:
            try:
                if predicate(framework, task, modality):
                    return harness_cls(run, recipe)
            except Exception:
                pass
        except Exception:
            pass

    if framework == "torch" and task == "classification":
        return TorchClassificationHarness(run, recipe)
    if framework == "torch" and task in {"regression", "regressor"}:
        return TorchRegressionHarness(run, recipe)

    raise ValueError(
        "No managed ML harness is available for "
        f"framework={framework!r}, task={task!r}, modality={modality!r}."
    )
