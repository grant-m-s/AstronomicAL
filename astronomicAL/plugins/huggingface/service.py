from __future__ import annotations

import base64
import html
import os
import re
import traceback
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from collections import Counter

import threading
from contextlib import contextmanager

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TASK_FILTER = "image-classification"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
}

GENERIC_FOLDERS = {
    "",
    ".",
    "data",
    "dataset",
    "datasets",
    "images",
    "image",
    "imgs",
    "jpg",
    "jpeg",
    "png",
    "files",
    "file",
    "raw",
    "samples",
    "sample",
    "records",
    "record",
    "examples",
    "example",
    "train",
    "training",
    "validation",
    "valid",
    "val",
    "test",
    "testing",
}


GENERIC_FOLDER_PATTERNS = [
    r"^data[_-]?\d+$",
    r"^part[_-]?\d+$",
    r"^chunk[_-]?\d+$",
    r"^shard[_-]?\d+$",
    r"^batch[_-]?\d+$",
    r"^images?[_-]?\d+$",
    r"^files?[_-]?\d+$",
    r"^records?[_-]?\d+$",
    r"^samples?[_-]?\d+$",
]


class SilentTqdm:
    """
    Minimal tqdm-compatible sink.

    Hugging Face snapshot_download can call tqdm_class.get_lock(),
    set_lock(), write(), and external_write_mode() when downloading files
    concurrently. This class intentionally swallows all progress output while
    satisfying that interface.
    """

    _lock = threading.RLock()

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total", None)
        self.n = 0
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def update(self, n: int = 1) -> None:
        self.n += n

    def close(self) -> None:
        pass

    def clear(self, *args, **kwargs) -> None:
        pass

    def reset(self, *args, **kwargs) -> None:
        self.n = 0

    def refresh(self, *args, **kwargs) -> None:
        pass

    def set_description(self, *args, **kwargs) -> None:
        pass

    def set_description_str(self, *args, **kwargs) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def set_postfix_str(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock) -> None:
        cls._lock = lock

    @classmethod
    def write(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    @contextmanager
    def external_write_mode(cls, *args, **kwargs):
        yield

@dataclass
class HFDatasetSearchResult:
    repo_id: str
    author: str = ""
    downloads: int | None = None
    likes: int | None = None
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    gated: str = ""
    private: bool = False
    card_summary: str = ""


@dataclass
class HFImageFile:
    path: str
    split: str = "train"
    label: str = ""
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "split": self.split,
            "label": self.label,
            "size": self.size,
        }


@dataclass
class HFDatasetDetails:
    repo_id: str
    configs: list[str] = field(default_factory=lambda: ["default"])
    splits_by_config: dict[str, list[str]] = field(default_factory=dict)
    features_by_config: dict[str, dict[str, str]] = field(default_factory=dict)
    image_column_candidates: list[str] = field(default_factory=lambda: ["__hf_file__"])
    label_column_candidates: list[str] = field(default_factory=lambda: ["__hf_path_label__"])
    warnings: list[str] = field(default_factory=list)
    sample_files: list[HFImageFile] = field(default_factory=list)
    scanned_file_count: int = 0
    total_repo_files_hint: int | None = None

    @property
    def error(self) -> str:
        return "\n".join(self.warnings)


class HuggingFaceDatasetService:
    """
    Hugging Face dataset helper.

    Inspect and Preview avoid datasets.load_dataset so they do not trigger
    Datasets' "Resolving data files" phase. Import uses snapshot_download for
    batch/concurrent download.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.cache_dir = Path(cache_dir or ".astronomical_cache/huggingface").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def load_dataset_builder_once(
        self,
        repo_id: str,
        *,
        config_name: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
    ):
        from datasets import load_dataset_builder

        kwargs = self._hf_kwargs(
            token=token,
            trust_remote_code=trust_remote_code,
        )

        try:
            if config_name:
                return load_dataset_builder(repo_id, name=config_name, **kwargs)
            return load_dataset_builder(repo_id, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                return load_dataset_builder(repo_id, name=config_name, **kwargs)
            return load_dataset_builder(repo_id, **kwargs)

    def search_datasets(
        self,
        *,
        query: str = "",
        task: str = DEFAULT_TASK_FILTER,
        limit: int = 25,
        sort: str = "downloads",
        token: str | None = None,
    ) -> list[HFDatasetSearchResult]:
        from huggingface_hub import HfApi

        resolved_token = token or self.token
        api = HfApi(token=resolved_token)

        requested_limit = max(int(limit or 25), 1)

        base_kwargs: dict[str, Any] = {
            "limit": requested_limit,
            "full": True,
        }
        if query:
            base_kwargs["search"] = query
        if sort:
            base_kwargs["sort"] = sort

        attempts: list[dict[str, Any]] = []

        if task:
            attempt = dict(base_kwargs)
            attempt["filter"] = task
            attempts.append(attempt)

            attempt = dict(base_kwargs)
            attempt["task_categories"] = task
            attempts.append(attempt)

        attempts.append(dict(base_kwargs))

        minimal: dict[str, Any] = {"limit": requested_limit}
        if query:
            minimal["search"] = query
        attempts.append(minimal)

        last_exc: BaseException | None = None
        infos = None

        for attempt in attempts:
            try:
                infos = list(api.list_datasets(**attempt))
                break
            except TypeError as exc:
                last_exc = exc
                continue

        if infos is None:
            if last_exc:
                raise last_exc
            infos = []

        results: list[HFDatasetSearchResult] = []
        for info in infos:
            repo_id = (
                getattr(info, "id", None)
                or getattr(info, "repo_id", None)
                or getattr(info, "name", None)
                or ""
            )
            if not repo_id:
                continue

            tags = list(getattr(info, "tags", []) or [])

            last_modified = (
                getattr(info, "lastModified", None)
                or getattr(info, "last_modified", None)
            )
            if last_modified is not None:
                last_modified = str(last_modified)

            card_data = (
                getattr(info, "cardData", None)
                or getattr(info, "card_data", None)
                or {}
            )
            summary = ""
            if isinstance(card_data, dict):
                summary = str(
                    card_data.get("summary")
                    or card_data.get("description")
                    or ""
                )

            results.append(
                HFDatasetSearchResult(
                    repo_id=str(repo_id),
                    author=str(getattr(info, "author", "") or "").strip(),
                    downloads=getattr(info, "downloads", None),
                    likes=getattr(info, "likes", None),
                    last_modified=str(last_modified or ""),
                    tags=tags,
                    gated=str(getattr(info, "gated", "") or ""),
                    private=bool(getattr(info, "private", False)),
                    card_summary=summary,
                )
            )

        if sort == "downloads":
            results.sort(key=lambda r: r.downloads or 0, reverse=True)
        elif sort == "likes":
            results.sort(key=lambda r: r.likes or 0, reverse=True)
        elif sort in {"lastModified", "last_modified"}:
            results.sort(key=lambda r: r.last_modified or "", reverse=True)

        return results[:requested_limit]

    def search_dataframe(
        self,
        *,
        query: str = "",
        task: str = DEFAULT_TASK_FILTER,
        limit: int = 25,
        sort: str = "downloads",
        token: str | None = None,
    ) -> pd.DataFrame:
        rows = []
        for result in self.search_datasets(
            query=query,
            task=task,
            limit=limit,
            sort=sort,
            token=token,
        ):
            rows.append(
                {
                    "repo_id": result.repo_id,
                    "downloads": result.downloads,
                    "likes": result.likes,
                    "last_modified": result.last_modified,
                    "gated": result.gated,
                    "private": result.private,
                    "tags": ", ".join(result.tags[:8]),
                    "summary": result.card_summary,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Lightweight inspect
    # ------------------------------------------------------------------

    def feature_text_for_column(
        self,
        details: Any,
        *,
        config: str,
        column: str,
    ) -> str:
        try:
            features = details.features_by_config.get(config, {}) or {}
            return str(features.get(column, ""))
        except Exception:
            return ""


    def feature_looks_like_shard_labels(self, feature_text: str) -> bool:
        """
        Detect ClassLabel names such as data_0, data_1, data_10, shard_0, part_0.

        These are usually storage shard folders, not semantic class labels.
        """

        text = str(feature_text or "")
        lowered = text.lower()

        if "classlabel" not in lowered and "names=" not in lowered:
            return False

        names = re.findall(r"[\"']([^\"']+)[\"']", text)
        if not names:
            return False

        if len(names) < 3:
            return False

        shard_like = 0
        for name in names:
            cleaned = str(name).strip().lower()
            if self._looks_like_non_label_folder(cleaned):
                shard_like += 1

        return shard_like / max(len(names), 1) >= 0.8


    def selected_builder_label_is_non_semantic(
        self,
        details: Any,
        *,
        config: str,
        label_column: str | None,
    ) -> bool:
        if not label_column:
            return True

        feature_text = self.feature_text_for_column(
            details,
            config=config,
            column=label_column,
        )
        return self.feature_looks_like_shard_labels(feature_text)


    def _looks_like_non_label_folder(self, folder_name: str) -> bool:
        cleaned = str(folder_name or "").strip().lower()
        if cleaned in GENERIC_FOLDERS:
            return True

        for pattern in GENERIC_FOLDER_PATTERNS:
            if re.match(pattern, cleaned):
                return True

        return False

    def _feature_candidates_from_features(
        self,
        features: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        image_candidates: list[str] = []
        label_candidates: list[str] = []

        for column, feature_type in dict(features or {}).items():
            col = str(column)
            text = str(feature_type).lower()
            col_lower = col.lower()

            if (
                "image" in text
                or col_lower in {"image", "img", "picture", "photo", "filepath", "path"}
            ):
                if col not in image_candidates:
                    image_candidates.append(col)

            if (
                "classlabel" in text
                or "class label" in text
                or "label" in text
                or col_lower in {"label", "labels", "target", "class", "category"}
            ):
                if col not in label_candidates:
                    label_candidates.append(col)

        return image_candidates, label_candidates


    def get_dataset_details_builder(
        self,
        repo_id: str,
        *,
        config_name: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> HFDatasetDetails:
        """
        Slower, authoritative inspection through Hugging Face Datasets.

        This version intentionally performs one builder call for the selected/default
        config, rather than separately calling config discovery, split discovery,
        feature discovery, and row sampling. This avoids repeated full
        "Resolving data files" passes for ImageFolder-style datasets.
        """

        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(repo_id=repo_id)

        config_key = config_name or "default"
        details.configs = [config_key]
        details.splits_by_config[config_key] = ["train"]
        details.scanned_file_count = 0

        print(
            f"[AL_DEBUG][HF][builder_inspect] START repo_id={repo_id!r} config={config_name!r}",
            flush=True,
        )

        try:
            builder = self.load_dataset_builder_once(
                repo_id,
                config_name=config_name,
                token=token,
                trust_remote_code=trust_remote_code,
            )

            actual_config = getattr(getattr(builder, "config", None), "name", None)
            if actual_config:
                config_key = str(actual_config)
                details.configs = [config_key]

            info = getattr(builder, "info", None)
            features = getattr(info, "features", None) or {}
            splits = getattr(info, "splits", None) or {}

            if splits:
                try:
                    split_names = [str(name) for name in splits.keys()]
                except Exception:
                    split_names = [str(name) for name in splits]
                details.splits_by_config[config_key] = split_names or ["train"]
            else:
                details.splits_by_config[config_key] = ["train"]

            details.features_by_config[config_key] = {
                str(name): self._feature_to_string(feature)
                for name, feature in dict(features).items()
            }

            image_candidates, label_candidates = self._feature_candidates_from_features(
                details.features_by_config[config_key]
            )

            details.image_column_candidates = image_candidates or ["image"]
            details.label_column_candidates = label_candidates

            # Strong fallback for common names.
            columns = list(details.features_by_config[config_key].keys())
            if "label" in columns and "label" not in details.label_column_candidates:
                details.label_column_candidates.insert(0, "label")
            if "labels" in columns and "labels" not in details.label_column_candidates:
                details.label_column_candidates.append("labels")

            print(
                f"[AL_DEBUG][HF][builder_inspect] DONE repo_id={repo_id!r} "
                f"config={config_key!r} splits={details.splits_by_config[config_key]!r} "
                f"columns={list(details.features_by_config[config_key].keys())!r}",
                flush=True,
            )

            return details

        except Exception as exc:
            details.warnings.append(
                "Single-call builder inspection failed. " + self._short_exception(exc)
            )
            print(
                f"[AL_DEBUG][HF][builder_inspect] ERROR repo_id={repo_id!r} error={exc!r}",
                flush=True,
            )

        # Fallback: stream one row only. This may still trigger a resolve once, but
        # avoids the previous pattern of several separate builder helper calls.
        try:
            sample_dataset = self.load_split(
                repo_id=repo_id,
                config_name=config_name,
                split="train",
                token=token,
                trust_remote_code=trust_remote_code,
                streaming=True,
            )
            first_row = dict(next(iter(sample_dataset)))
            features = {
                str(key): type(value).__name__
                for key, value in first_row.items()
            }
            details.features_by_config[config_key] = features
            details.splits_by_config[config_key] = ["train"]

            image_candidates, label_candidates = self._feature_candidates_from_features(features)
            details.image_column_candidates = image_candidates or ["image"]
            details.label_column_candidates = label_candidates

            details.warnings.append(
                "Builder metadata was unavailable; inferred columns from one streamed row."
            )
        except Exception as exc:
            details.features_by_config[config_key] = {
                "image": "Image fallback",
            }
            details.image_column_candidates = ["image"]
            details.label_column_candidates = []
            details.warnings.append(
                "Could not infer builder columns from a streamed sample. "
                + self._short_exception(exc)
            )

        return details


    def get_config_names_builder(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> list[str]:
        from datasets import get_dataset_config_names

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            configs = list(get_dataset_config_names(repo_id, **kwargs))
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            configs = list(get_dataset_config_names(repo_id, **kwargs))

        return [str(config) for config in configs] or ["default"]


    def get_split_names_builder(
        self,
        repo_id: str,
        *,
        config_name: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> list[str]:
        from datasets import get_dataset_split_names

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            if config_name:
                splits = list(get_dataset_split_names(repo_id, config_name, **kwargs))
            else:
                splits = list(get_dataset_split_names(repo_id, **kwargs))
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                splits = list(get_dataset_split_names(repo_id, config_name, **kwargs))
            else:
                splits = list(get_dataset_split_names(repo_id, **kwargs))

        return [str(split) for split in splits] or ["train"]


    def get_features_builder(
        self,
        repo_id: str,
        *,
        config_name: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> dict[str, str]:
        from datasets import load_dataset_builder

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)

        try:
            if config_name:
                builder = load_dataset_builder(repo_id, name=config_name, **kwargs)
            else:
                builder = load_dataset_builder(repo_id, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                builder = load_dataset_builder(repo_id, name=config_name, **kwargs)
            else:
                builder = load_dataset_builder(repo_id, **kwargs)

        features = getattr(getattr(builder, "info", None), "features", None) or {}
        return {
            str(name): self._feature_to_string(feature)
            for name, feature in features.items()
        }

    def preview_image_grid_html_builder(
        self,
        *,
        repo_id: str,
        config_name: str | None,
        split: str,
        image_column: str | None = None,
        label_column: str | None = None,
        id_column: str | None = None,
        limit: int = 12,
        token: str | None = None,
        trust_remote_code: bool = False,
        thumb_size: int = 180,
    ) -> str:
        """
        Preview through Hugging Face Datasets.

        This is slower than Hub-file preview, but it can show actual label columns
        when the dataset exposes them.
        """

        dataset = self.load_split(
            repo_id=repo_id,
            config_name=config_name,
            split=split,
            token=token,
            trust_remote_code=trust_remote_code,
            streaming=True,
        )

        features = getattr(dataset, "features", {}) or {}

        image_column = image_column or self.infer_image_column(features, dataset=dataset)

        # Prefer explicit UI choice, then normal inference, then common fallback.
        label_column = label_column or self.infer_label_column(features, dataset=dataset)

        if not image_column:
            image_column = "image"

        rows = []
        import itertools

        for index, row in enumerate(itertools.islice(iter(dataset), max(int(limit or 12), 1))):
            row = dict(row)

            if not label_column:
                if "label" in row:
                    label_column = "label"
                elif "labels" in row:
                    label_column = "labels"
                elif "target" in row:
                    label_column = "target"
                elif "class" in row:
                    label_column = "class"

            record_id = self.record_id_for_row(
                row,
                index=index,
                split=split,
                id_column=id_column,
            )

            label_value = row.get(label_column) if label_column else ""
            label_name = self.label_to_name(features, label_column, label_value)
            label_display = label_name if label_name != "" else label_value

            data_uri = ""
            try:
                data_uri = self.image_value_to_data_uri(
                    row.get(image_column),
                    max_size=thumb_size,
                )
            except Exception:
                data_uri = ""

            rows.append(
                {
                    "record_id": record_id,
                    "label": label_display,
                    "image_column": image_column,
                    "label_column": label_column or "",
                    "data_uri": data_uri,
                }
            )

        if not rows:
            return "<p>No preview rows were returned.</p>"

        cards = []
        for row in rows:
            img = row.get("data_uri", "")
            title = html.escape(str(row.get("record_id", "")))
            label = html.escape(str(row.get("label", "")))
            image_col = html.escape(str(row.get("image_column", "")))
            label_col = html.escape(str(row.get("label_column", "")))

            if img:
                image_html = (
                    f'<img src="{img}" style="max-width:100%; max-height:{thumb_size}px; '
                    f'object-fit:contain; display:block; margin:auto;" />'
                )
            else:
                image_html = (
                    '<div style="height:150px; display:flex; align-items:center; '
                    'justify-content:center; background:#eee; color:#777;">Preview unavailable</div>'
                )

            cards.append(
                f"""
                <div style="border:1px solid #ddd; border-radius:6px; padding:8px; background:white;">
                <div style="height:{thumb_size}px; display:flex; align-items:center; justify-content:center; background:#111;">
                    {image_html}
                </div>
                <div style="font-size:12px; margin-top:6px; word-break:break-all;">
                    <b>{title}</b><br/>
                    <span>{label}</span><br/>
                    <span style="color:#777;">image: {image_col}</span><br/>
                    <span style="color:#777;">label: {label_col}</span>
                </div>
                </div>
                """
            )

        return f"""
        <p style="font-size:12px; color:#555;">
        Builder preview may resolve dataset files, but can expose actual label columns.
        </p>
        <div style="
        display:grid;
        grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));
        gap:10px;
        width:100%;
        ">
        {''.join(cards)}
        </div>
        """

    def image_value_to_data_uri(self, value: Any, *, max_size: int = 180) -> str:
        from PIL import Image as PILImage
        from PIL import ImageOps

        raw: bytes | None = None

        if isinstance(value, dict):
            if value.get("bytes") is not None:
                raw = value["bytes"]
            elif value.get("path"):
                path_text = str(value["path"])
                if path_text.startswith("hf://"):
                    return ""
                path = Path(path_text).expanduser()
                if path.exists():
                    raw = path.read_bytes()
        elif isinstance(value, (str, Path)):
            text = str(value)
            if text.startswith("hf://"):
                return ""
            path = Path(text).expanduser()
            if path.exists():
                raw = path.read_bytes()
        elif hasattr(value, "save"):
            buffer = BytesIO()
            value.save(buffer, format="PNG")
            raw = buffer.getvalue()

        if raw is None:
            return ""

        with PILImage.open(BytesIO(raw)) as image:
            return self.image_to_preview_data_uri(
                image,
                size=int(max_size or 180),
                fill=True,
            )

    def get_dataset_details(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> HFDatasetDetails:
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("repo_id is required.")

        details = HFDatasetDetails(repo_id=repo_id)
        resolved_token = token or self.token

        try:
            info = self.dataset_info_lightweight(repo_id, token=resolved_token)
            siblings = list(getattr(info, "siblings", []) or [])
            if siblings:
                details.total_repo_files_hint = len(siblings)
        except Exception as exc:
            details.warnings.append(
                "Could not read Hub dataset metadata. " + self._short_exception(exc)
            )

        try:
            sample_files = self.list_image_files(
                repo_id,
                token=resolved_token,
                max_files=500,
            )
            details.sample_files = sample_files
            details.scanned_file_count = len(sample_files)

            splits = sorted({file.split for file in sample_files if file.split})
            if not splits:
                splits = ["train"]
            details.splits_by_config["default"] = splits

            labels = sorted({file.label for file in sample_files if file.label})
            if labels:
                details.warnings.append(
                    f"Detected {len(labels)} label folder(s) in sampled files."
                )
        except Exception as exc:
            details.splits_by_config["default"] = ["train"]
            details.warnings.append(
                "Could not sample repository image files. " + self._short_exception(exc)
            )

        details.features_by_config["default"] = {
            "__hf_file__": "Hub image files; batch downloaded with snapshot_download",
            "__hf_path_label__": "Inferred label from parent folder, if available",
        }
        details.image_column_candidates = ["__hf_file__"]
        details.label_column_candidates = ["__hf_path_label__"]

        if details.total_repo_files_hint is not None:
            details.warnings.append(
                f"Hub metadata reports {details.total_repo_files_hint} top-level/listed sibling file(s)."
            )

        return details

    def dataset_info_lightweight(self, repo_id: str, *, token: str | None = None):
        from huggingface_hub import HfApi

        api = HfApi(token=token or self.token)

        try:
            return api.dataset_info(
                repo_id,
                files_metadata=False,
                expand=[
                    "author",
                    "cardData",
                    "downloads",
                    "gated",
                    "lastModified",
                    "likes",
                    "private",
                    "siblings",
                    "sha",
                    "tags",
                ],
                token=token or self.token,
            )
        except TypeError:
            return api.dataset_info(
                repo_id,
                files_metadata=False,
                token=token or self.token,
            )

    # ------------------------------------------------------------------
    # Hub file listing
    # ------------------------------------------------------------------

    def list_image_files(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        split: str | None = None,
        max_files: int = 0,
    ) -> list[HFImageFile]:
        from huggingface_hub import HfApi

        api = HfApi(token=token or self.token)
        max_files = int(max_files or 0)
        wanted_split = self._normalise_split(split) if split else None

        files: list[HFImageFile] = []

        try:
            tree_iter = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                recursive=True,
                token=token or self.token,
            )
        except TypeError:
            tree_iter = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                recursive=True,
            )

        for item in tree_iter:
            path = (
                getattr(item, "path", None)
                or getattr(item, "rfilename", None)
                or ""
            )
            if not path:
                continue

            if not self._is_image_path(path):
                continue

            detected_split = self.infer_split_from_path(path)
            if wanted_split and detected_split != wanted_split:
                continue

            files.append(
                HFImageFile(
                    path=str(path),
                    split=detected_split,
                    label=self.infer_label_from_path(path, detected_split),
                    size=getattr(item, "size", None),
                )
            )

            if max_files and len(files) >= max_files:
                break

        return files

    # ------------------------------------------------------------------
    # Preview: download only displayed files
    # ------------------------------------------------------------------

    def preview_image_grid_html(
        self,
        *,
        repo_id: str,
        config_name: str | None,
        split: str,
        image_column: str | None = None,
        label_column: str | None = None,
        id_column: str | None = None,
        limit: int = 12,
        token: str | None = None,
        trust_remote_code: bool = False,
        thumb_size: int = 180,
    ) -> str:
        files = self.list_image_files(
            repo_id,
            token=token or self.token,
            split=split or "train",
            max_files=max(int(limit or 12), 1),
        )

        if not files:
            return (
                "<p>No image files were found for this split. "
                "Try a different split or import with the full dataset path.</p>"
            )

        cards = []
        for index, file in enumerate(files):
            record_id = f"{file.split}:{index}"
            label = file.label or "no inferred label"

            data_uri = ""
            try:
                local_path = self.download_file(
                    repo_id,
                    file.path,
                    token=token or self.token,
                )
                data_uri = self.local_image_to_data_uri(
                    local_path,
                    max_size=thumb_size,
                )
            except Exception:
                data_uri = ""

            title = html.escape(record_id)
            label_text = html.escape(str(label))
            path_text = html.escape(file.path)

            if data_uri:
                image_html = (
                    f'<img src="{data_uri}" style="max-width:100%; max-height:{thumb_size}px; '
                    f'object-fit:contain; display:block; margin:auto;" />'
                )
            else:
                image_html = (
                    '<div style="height:150px; display:flex; align-items:center; '
                    'justify-content:center; background:#eee; color:#777;">Preview unavailable</div>'
                )

            cards.append(
                f"""
                <div style="border:1px solid #ddd; border-radius:6px; padding:8px; background:white;">
                  <div style="height:{thumb_size}px; display:flex; align-items:center; justify-content:center; background:#111;">
                    {image_html}
                  </div>
                  <div style="font-size:12px; margin-top:6px; word-break:break-all;">
                    <b>{title}</b><br/>
                    <span>{label_text}</span><br/>
                    <span style="color:#777;">{path_text}</span>
                  </div>
                </div>
                """
            )

        return f"""
        <p style="font-size:12px; color:#555;">
          Preview downloaded only {len(files)} displayed file(s). Full split registration uses batch snapshot download.
        </p>
        <div style="
          display:grid;
          grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));
          gap:10px;
          width:100%;
        ">
          {''.join(cards)}
        </div>
        """
    
    def download_files_parallel(
        self,
        repo_id: str,
        file_paths: list[str],
        *,
        token: str | None = None,
        max_workers: int = 16,
        progress_callback: Any = None,
    ) -> dict[str, str]:
        """
        Download many Hub files concurrently and report file-count progress.

        Returns:
            {hub_relative_path: local_cache_path}

        This is intentionally used for full import/register, where the UI needs
        useful progress feedback. Preview still uses single-file downloads because
        it only displays a few images.
        """

        clean_paths = [str(path) for path in file_paths if str(path).strip()]
        total = len(clean_paths)

        if total == 0:
            return {}

        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except Exception:
            pass

        def emit(
            *,
            phase: str,
            completed: int,
            total: int,
            percent: int,
            message: str,
            current_file: str = "",
        ) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "phase": phase,
                            "completed": int(completed),
                            "total": int(total),
                            "percent": max(0, min(100, int(percent))),
                            "message": message,
                            "current_file": current_file,
                        }
                    )
                except Exception:
                    pass

        emit(
            phase="download",
            completed=0,
            total=total,
            percent=5,
            message=f"Starting parallel download of {total} file(s)…",
        )

        def worker(path: str) -> tuple[str, str]:
            local_path = self.download_file(
                repo_id,
                path,
                token=token or self.token,
            )
            return path, local_path

        results: dict[str, str] = {}
        completed = 0
        max_workers = max(1, int(max_workers or 16))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, path): path
                for path in clean_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                hub_path, local_path = future.result()
                results[hub_path] = local_path
                completed += 1

                # Reserve 5-85% for file download.
                percent = 5 + int((completed / total) * 80)

                emit(
                    phase="download",
                    completed=completed,
                    total=total,
                    percent=percent,
                    message=f"Downloaded {completed}/{total} file(s)…",
                    current_file=path,
                )

        emit(
            phase="download",
            completed=total,
            total=total,
            percent=85,
            message=f"Downloaded {total}/{total} file(s).",
        )

        return results

    def download_file(
        self,
        repo_id: str,
        filename: str,
        *,
        token: str | None = None,
    ) -> str:
        from huggingface_hub import hf_hub_download

        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": "dataset",
            "cache_dir": str(self.cache_dir / "hub_files"),
            "token": token or self.token,
        }

        try:
            return hf_hub_download(**kwargs)
        except TypeError:
            kwargs.pop("token", None)
            return hf_hub_download(**kwargs)

    # ------------------------------------------------------------------
    # Import: batch snapshot download
    # ------------------------------------------------------------------

    def snapshot_download_files(
        self,
        repo_id: str,
        file_paths: list[str],
        *,
        token: str | None = None,
        max_workers: int = 16,
        quiet: bool = True,
    ) -> str:
        """
        Download a set of files in one snapshot_download call.

        Returns the local snapshot directory. Local file path for a Hub path is:

            Path(snapshot_dir) / hub_relative_path
        """

        from huggingface_hub import snapshot_download

        clean_paths = [str(path) for path in file_paths if str(path).strip()]
        if not clean_paths:
            raise ValueError("No file paths supplied for snapshot download.")

        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "cache_dir": str(self.cache_dir / "snapshots"),
            "allow_patterns": clean_paths,
            "max_workers": max(int(max_workers or 16), 1),
            "token": token or self.token,
        }

        if quiet:
            kwargs["tqdm_class"] = SilentTqdm

        # Different huggingface_hub versions support slightly different kwargs.
        try:
            return snapshot_download(**kwargs)
        except TypeError as exc:
            message = str(exc)

            if "tqdm_class" in message:
                kwargs.pop("tqdm_class", None)
                return snapshot_download(**kwargs)

            if "max_workers" in message:
                kwargs.pop("max_workers", None)
                return snapshot_download(**kwargs)

            if "token" in message:
                kwargs.pop("token", None)
                return snapshot_download(**kwargs)

            raise

    def image_to_preview_data_uri(
        self,
        image,
        *,
        size: int = 180,
        fill: bool = True,
    ) -> str:
        """Convert a PIL image to a consistently sized preview data URI.

        Unlike PIL.thumbnail(), this also upscales tiny images such as CIFAR-10
        32x32 samples so they fill the preview card.
        """

        from PIL import Image as PILImage
        from PIL import ImageOps

        size = max(32, int(size or 180))

        image = ImageOps.exif_transpose(image)

        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        if fill:
            # Fill the full square preview area. This may crop non-square images.
            image = ImageOps.fit(
                image,
                (size, size),
                method=PILImage.Resampling.NEAREST,
                centering=(0.5, 0.5),
            )
        else:
            # Preserve full image without cropping, but still upscale tiny images.
            image.thumbnail(
                (size, size),
                resample=PILImage.Resampling.LANCZOS,
            )

            canvas = PILImage.new("RGB", (size, size), (17, 17, 17))
            if image.mode != "RGB":
                image = image.convert("RGB")

            x = (size - image.width) // 2
            y = (size - image.height) // 2
            canvas.paste(image, (x, y))
            image = canvas

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def local_image_to_data_uri(self, path: str | Path, *, max_size: int = 180) -> str:
        from PIL import Image as PILImage
        from PIL import ImageOps

        path = Path(path).expanduser()
        if not path.exists():
            return ""

        with PILImage.open(path) as image:
            return self.image_to_preview_data_uri(
                image,
                size=int(max_size or 180),
                fill=True,
            )

    # ------------------------------------------------------------------
    # Legacy Datasets API helpers
    # ------------------------------------------------------------------

    def load_split(
        self,
        *,
        repo_id: str,
        config_name: str | None,
        split: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        streaming: bool = False,
    ):
        from datasets import load_dataset

        kwargs = self._hf_kwargs(token=token, trust_remote_code=trust_remote_code)
        kwargs["streaming"] = bool(streaming)

        try:
            if config_name:
                return load_dataset(repo_id, name=config_name, split=split, **kwargs)
            return load_dataset(repo_id, split=split, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            if config_name:
                return load_dataset(repo_id, name=config_name, split=split, **kwargs)
            return load_dataset(repo_id, split=split, **kwargs)

    def cast_image_decode_false(self, dataset: Any, image_column: str) -> Any:
        try:
            from datasets import Image

            return dataset.cast_column(image_column, Image(decode=False))
        except Exception:
            return dataset

    def infer_image_column(self, features: Any, *, dataset: Any = None) -> str | None:
        for name, feature in dict(features or {}).items():
            if "image" in self._feature_to_string(feature).lower():
                return str(name)

        column_names = self._column_names(dataset)
        for candidate in ("image", "img", "picture", "photo", "file", "filepath", "image_path", "path"):
            if candidate in column_names:
                return candidate

        return None

    def infer_label_column(self, features: Any, *, dataset: Any = None) -> str | None:
        for name, feature in dict(features or {}).items():
            text = self._feature_to_string(feature).lower()
            if "classlabel" in text:
                return str(name)

        column_names = self._column_names(dataset)
        for candidate in ("label", "labels", "target", "class", "category"):
            if candidate in column_names:
                return candidate

        return None

    def record_id_for_row(
        self,
        row: dict[str, Any],
        *,
        index: int,
        split: str,
        id_column: str | None = None,
    ) -> str:
        if id_column and id_column in row and row.get(id_column) is not None:
            return str(row.get(id_column))
        for candidate in ("id", "record_id", "image_id", "file_name", "filename", "path"):
            if candidate in row and row.get(candidate) is not None:
                return str(row.get(candidate))
        return f"{split}:{index}"

    def label_to_name(self, features: Any, label_column: str | None, value: Any) -> str:
        if label_column is None:
            return ""
        try:
            feature = features[label_column]
        except Exception:
            return ""

        if hasattr(feature, "int2str"):
            try:
                return str(feature.int2str(int(value)))
            except Exception:
                return ""

        return ""

    # ------------------------------------------------------------------
    # Path inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_image_path(path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in IMAGE_EXTENSIONS

    @staticmethod
    def _normalise_split(value: str | None) -> str:
        if not value:
            return "train"
        cleaned = str(value).strip().lower()
        return SPLIT_ALIASES.get(cleaned, cleaned)

    def infer_split_from_path(self, path: str) -> str:
        parts = [part.lower() for part in Path(path).parts]
        stem = Path(path).stem.lower()

        for part in parts:
            cleaned = re.sub(r"[^a-z0-9]+", "", part)
            if cleaned in SPLIT_ALIASES:
                return SPLIT_ALIASES[cleaned]

        tokens = re.split(r"[^a-z0-9]+", stem)
        for token in tokens:
            if token in SPLIT_ALIASES:
                return SPLIT_ALIASES[token]

        for key, split in SPLIT_ALIASES.items():
            if re.search(rf"(^|[_\-.]){re.escape(key)}([_\-.]|$)", stem):
                return split

        return "train"

    def infer_label_from_path(self, path: str, split: str = "train") -> str:
        """
        Infer a semantic label from a Hub image path.

        Priority:
        1. Parent folder if it looks like a real class folder.
        2. Filename if it contains a label-like pattern.

        Examples:
        train/cat/0001.jpg                         -> cat
        train/dog/0001.jpg                         -> dog
        data/data_0/000001.jpg                     -> ""
        data/000000_bubble__bubble_0.9999.jpg      -> bubble
        data/test_room_testing_room_0.9997.jpg     -> test room testing room
        ImageNet-A.gif                             -> ""
        """

        parts = list(Path(path).parts)

        if len(parts) >= 2:
            parent = parts[-2]
            parent_clean = parent.strip().lower()

            if (
                not self._looks_like_non_label_folder(parent_clean)
                and self._normalise_split(parent_clean) != split
            ):
                return parent

        return self.infer_label_from_filename(path)


    def infer_label_from_filename(self, path: str) -> str:
        """
        Infer labels from filenames only when the filename looks intentionally
        label-bearing.

        This avoids turning generic names like ImageNet-A.gif or 000001.jpg into
        bogus labels, while supporting filenames such as:

            000000_bubble__bubble_0.9999.jpg
            000000_cockroach__cockroach__0.9996.jpg
            test_room_testing_room_0.9997.jpg
        """

        stem = Path(path).stem.strip()
        if not stem:
            return ""

        has_double_separator = "__" in stem
        has_score_suffix = bool(re.search(r"[_\-.][01]?\.\d+$", stem))
        has_leading_numeric_id = bool(re.match(r"^\d+[_\-.]+[A-Za-z]", stem))

        if not (has_double_separator or has_score_suffix or has_leading_numeric_id):
            return ""

        raw_candidates: list[str] = []

        if has_double_separator:
            raw_candidates.extend(part for part in stem.split("__") if part)

        raw_candidates.append(stem)

        normalised: list[str] = []
        for candidate in raw_candidates:
            label = self._normalise_filename_label_candidate(candidate)
            if self._looks_like_filename_label_candidate(label):
                normalised.append(label)

        if not normalised:
            return ""

        counts = Counter(normalised)

        # Prefer repeated labels such as:
        #   000000_bubble__bubble_0.9999 -> bubble, bubble
        # Otherwise choose the shortest plausible candidate.
        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        return ranked[0][0]


    def _normalise_filename_label_candidate(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        # Remove leading numeric IDs.
        text = re.sub(r"^\d+[_\-.]+", "", text)

        # Remove trailing confidence-like scores.
        text = re.sub(r"([_\-.]+[01]?\.\d+)$", "", text)

        # Remove common score suffix forms.
        text = re.sub(r"([_\-.]+score[_\-.]*[01]?\.\d+)$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"([_\-.]+conf[_\-.]*[01]?\.\d+)$", "", text, flags=re.IGNORECASE)

        # Strip leftover separators.
        text = re.sub(r"[_\-.]+$", "", text)
        text = re.sub(r"^[_\-.]+", "", text)

        # Convert separators to readable labels.
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text


    def _looks_like_filename_label_candidate(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False

        lowered = text.lower()

        if self._looks_like_non_label_folder(lowered):
            return False

        if lowered in {"train", "test", "validation", "valid", "val"}:
            return False

        if lowered.isdigit():
            return False

        # Avoid huge captions/paths being treated as labels.
        if len(text) > 80:
            return False

        # Require at least one alphabetic character. This intentionally rejects
        # pure MNIST digit filenames unless the label is provided elsewhere.
        if not re.search(r"[A-Za-z]", text):
            return False

        return True

    def _looks_like_non_label_folder(self, folder_name: str) -> bool:
        cleaned = str(folder_name or "").strip().lower()
        if cleaned in GENERIC_FOLDERS:
            return True

        for pattern in GENERIC_FOLDER_PATTERNS:
            if re.match(pattern, cleaned):
                return True

        return False

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------

    def _hf_kwargs(
        self,
        *,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir / "datasets_cache"),
        }

        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        resolved_token = token or self.token
        if resolved_token:
            kwargs["token"] = resolved_token

        return kwargs

    @staticmethod
    def _feature_to_string(feature: Any) -> str:
        try:
            return feature.__class__.__name__ + ": " + str(feature)
        except Exception:
            return str(feature)

    @staticmethod
    def _column_names(dataset: Any) -> list[str]:
        if dataset is None:
            return []
        try:
            return list(dataset.column_names)
        except Exception:
            pass
        try:
            features = getattr(dataset, "features", {}) or {}
            return list(features.keys())
        except Exception:
            return []

    @staticmethod
    def _short_exception(exc: BaseException, *, limit: int = 500) -> str:
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        text = " ".join(text.split())
        if len(text) > limit:
            return text[:limit] + "… [truncated]"
        return text