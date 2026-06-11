from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

IMAGE_SIDECAR_SCHEMA_VERSION = 1
TORCH_IMAGE_CLASSIFIER_FORMAT = "torch_image_classifier"
DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def is_torch_image_bundle(model: Any, metadata: Optional[Mapping[str, Any]] = None) -> bool:
    """Return True when a training result is the image-classifier bundle."""

    if not isinstance(model, Mapping):
        return False

    metadata = dict(metadata or {})
    modality = str(model.get("modality") or metadata.get("modality") or "").lower()
    task = str(model.get("task") or metadata.get("task") or "").lower()

    if modality == "image" and task in {"classification", "classifier", ""}:
        return model.get("torch_model") is not None

    return (
        model.get("torch_model") is not None
        and (model.get("class_names") or metadata.get("class_names"))
        and (model.get("template_id") or metadata.get("template_id"))
    )


def save_torch_image_sidecar_file(
    *,
    path: Path,
    model: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a torch image classifier in a portable/reconstructable form."""

    import torch

    metadata = dict(metadata or {})
    torch_model = model.get("torch_model") or model.get("model")
    if torch_model is None:
        raise ValueError("Image model bundle is missing `torch_model`.")

    class_names = _class_names(model, metadata)
    if not class_names:
        raise ValueError("Image model bundle is missing `class_names`.")

    template_id = str(model.get("template_id") or metadata.get("template_id") or "resnet18")
    architecture = str(
        model.get("architecture")
        or metadata.get("architecture")
        or infer_resnet_architecture(template_id)
    )
    image_size = int(model.get("image_size") or metadata.get("image_size") or 224)
    normalization = dict(
        model.get("normalization")
        or metadata.get("normalization")
        or DEFAULT_NORMALIZATION
    )

    state_dict = {
        str(key): value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in torch_model.state_dict().items()
    }

    sidecar = {
        "sidecar_type": TORCH_IMAGE_CLASSIFIER_FORMAT,
        "schema_version": IMAGE_SIDECAR_SCHEMA_VERSION,
        "architecture": architecture,
        "template_id": template_id,
        "state_dict": state_dict,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": image_size,
        "normalization": normalization,
        "channels": int(model.get("channels") or metadata.get("channels") or 3),
        "pretrained": False,
        "metadata": _json_safe({**metadata, **dict(model.get("metadata") or {})}),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, path)

    return {
        "sidecar_type": TORCH_IMAGE_CLASSIFIER_FORMAT,
        "schema_version": IMAGE_SIDECAR_SCHEMA_VERSION,
        "architecture": architecture,
        "template_id": template_id,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": image_size,
        "normalization": normalization,
        "channels": sidecar["channels"],
    }


def load_torch_image_sidecar_file(
    *,
    path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a sidecar and reconstruct the torch image classifier."""

    import torch

    metadata = dict(metadata or {})
    sidecar = torch.load(path, map_location=map_location)

    if not isinstance(sidecar, Mapping):
        raise TypeError(f"Image sidecar {path} did not contain a mapping payload.")

    sidecar_type = str(sidecar.get("sidecar_type") or metadata.get("sidecar_type") or "")
    if sidecar_type != TORCH_IMAGE_CLASSIFIER_FORMAT:
        raise ValueError(f"Unsupported image sidecar type `{sidecar_type}`.")

    class_names = [
        str(c)
        for c in sidecar.get("class_names")
        or metadata.get("class_names")
        or []
    ]
    if not class_names:
        raise ValueError(f"Image sidecar {path} is missing class names.")

    architecture = str(sidecar.get("architecture") or metadata.get("architecture") or "resnet18")
    state_dict = sidecar.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Image sidecar {path} is missing `state_dict`.")

    model = build_resnet_classifier(
        architecture=architecture,
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "torch_model": model,
        "class_names": class_names,
        "label_encoder": SimpleLabelEncoder(class_names),
        "image_size": int(sidecar.get("image_size") or metadata.get("image_size") or 224),
        "normalization": dict(
            sidecar.get("normalization")
            or metadata.get("normalization")
            or DEFAULT_NORMALIZATION
        ),
        "template_id": sidecar.get("template_id") or metadata.get("template_id"),
        "architecture": architecture,
        "channels": int(sidecar.get("channels") or metadata.get("channels") or 3),
        "sidecar": _json_safe({k: v for k, v in sidecar.items() if k not in {"state_dict"}}),
    }


def build_resnet_classifier(*, architecture: str, num_classes: int, pretrained: bool = False):
    """Rebuild the ResNet classifier architecture used by image training."""

    import torch.nn as nn
    from torchvision import models

    architecture = str(architecture or "resnet18").lower()
    if architecture not in {"resnet18", "resnet50"}:
        raise ValueError(f"Unsupported image classifier architecture `{architecture}`.")

    if architecture == "resnet50":
        weights_cls = getattr(models, "ResNet50_Weights", None)
        weights = weights_cls.DEFAULT if pretrained and weights_cls is not None else None
        try:
            model = models.resnet50(weights=weights)
        except TypeError:
            model = models.resnet50(pretrained=pretrained)
    else:
        weights_cls = getattr(models, "ResNet18_Weights", None)
        weights = weights_cls.DEFAULT if pretrained and weights_cls is not None else None
        try:
            model = models.resnet18(weights=weights)
        except TypeError:
            model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))
    return model


def image_transform(*, image_size: int, normalization: Optional[Mapping[str, Any]] = None):
    from torchvision import transforms

    normalization = dict(normalization or DEFAULT_NORMALIZATION)
    mean = [float(v) for v in normalization.get("mean", DEFAULT_NORMALIZATION["mean"])]
    std = [float(v) for v in normalization.get("std", DEFAULT_NORMALIZATION["std"])]

    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_image(value: Any):
    """Load a local/remote/bytes image and return RGB PIL Image."""

    from PIL import Image

    if value is None:
        raise ValueError("Image value is missing.")

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if hasattr(value, "read") and callable(value.read):
        return Image.open(value).convert("RGB")

    text = str(value).strip()
    if not text:
        raise ValueError("Image path/URI is empty.")

    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    path = Path(text).expanduser()
    return Image.open(path).convert("RGB")


def infer_resnet_architecture(template_id: str) -> str:
    text = str(template_id or "").lower()
    if "resnet50" in text or "resnet_50" in text:
        return "resnet50"
    return "resnet18"


class SimpleLabelEncoder:
    """Small inverse-transform-only label encoder for inference sidecars."""

    def __init__(self, classes: Sequence[Any]) -> None:
        self.classes_ = np.asarray([str(c) for c in classes])

    def inverse_transform(self, values: Sequence[int]):
        return np.asarray([self.classes_[int(v)] for v in values])

    def transform(self, values: Sequence[Any]):
        lookup = {str(value): idx for idx, value in enumerate(self.classes_)}
        return np.asarray([lookup[str(value)] for value in values], dtype=int)


def _class_names(model: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    if model.get("class_names"):
        return [str(c) for c in model["class_names"]]

    if metadata.get("class_names"):
        return [str(c) for c in metadata["class_names"]]

    encoder = model.get("label_encoder")
    classes = getattr(encoder, "classes_", None)
    if classes is not None:
        return [str(c) for c in list(classes)]

    return []


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]

    return str(value)