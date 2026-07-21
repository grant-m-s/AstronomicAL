from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Dict, Optional

from ..partition_reader import PartitionReader, create_partition_reader
from ..paths import ml_run_artifact_dir
from ..protocol import Partition, PartitionRef, Partitions, TargetSpec
from ..serialization import json_safe
from ..split_planner import create_streaming_partitions
from ..streaming_normalization import apply_partition_reader_image_normalization
from ..streaming_split_datasets import materialize_streaming_split_datasets
from .torch_classification import (
    TorchClassificationHarness as _MaterializedClassificationHarness,
)
from .torch_regression import (
    TorchRegressionHarness as _MaterializedRegressionHarness,
)

class ManifestBackedHarnessMixin:
    """Disk-backed split generation and streamed PyTorch epochs.

    Split creation scans only record ID, target, and protocol columns into a
    temporary disk index. Feature and image columns are not materialised during
    splitting, and every train/validation/test epoch reads bounded batches through
    ``PartitionReader``.
    """

    def _partition(self) -> Partitions:
        manifest_root = ml_run_artifact_dir(self.run, kind="split")
        parts = create_streaming_partitions(
            context=self.run.context,
            root=manifest_root,
            run_id=self.run.run_id,
            source_dataset_id=self.run.dataset_id,
            protocol=self.protocol,
            binding=self.binding,
            task_kind=self._task_kind(),
            batch_size=_split_scan_batch_size(self.run.params),
            cancel_check=self.run.check_cancelled,
            selected_row_ids=_selected_training_row_ids(self.run.params),
        )
        parts.materialized_split_dataset_ids = materialize_streaming_split_datasets(
            run=self.run,
            protocol=self.protocol,
            binding=self.binding,
            parts=parts,
        )
        _bind_materialized_partition_sources(parts)
        return _promote_partition_refs(parts, self)

    def _write_split_spec(self, parts: Partitions) -> Optional[str]:
        manifest = parts.split_manifest
        if manifest is None:
            return super()._write_split_spec(parts)

        payload = {
            "artifact_type": "ml.split_spec",
            "schema_version": 3,
            "run_id": self.run.run_id,
            "recipe_id": self.run.recipe_id,
            "recipe_version": self.run.recipe_version,
            "source_dataset_id": self.run.dataset_id,
            "protocol_id": parts.protocol_id,
            "strategy": parts.strategy,
            "validation_source": parts.validation_source,
            "test_source": parts.test_source,
            "group_column": parts.group_column,
            "random_state": parts.random_state,
            "record_id_column": parts.record_id_column,
            "target_column": parts.target_column,
            "train_dataset_id": parts.train_dataset_id,
            "validation_dataset_id": parts.validation_dataset_id,
            "test_dataset_id": parts.test_dataset_id,
            "split_dataset_ids": dict(parts.materialized_split_dataset_ids or {}),
            "split_manifest": manifest.to_dict(),
            "partitions": {
                role: ref.to_dict()
                for role, ref in parts.partition_refs.items()
            },
            "partition_counts": {
                role: int(ref.row_count)
                for role, ref in parts.partition_refs.items()
            },
            "split_generation": dict(parts.split_generation or {}),
            "training_selection": {
                "row_count": int(parts.train.row_count),
                "source": (
                    "action_selection"
                    if _selected_training_row_ids(self.run.params) is not None
                    else "dataset"
                ),
            },
            "classes": list(parts.train.classes),
        }
        return self.run.put_artifact(
            "ml.split_spec",
            json_safe(payload),
            params=self.run.params,
            required=True,
        )

    def _partition_signatures(self, parts: Partitions) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for role, partition in (
            ("train", parts.train),
            ("validation", parts.val),
            ("test", parts.test),
        ):
            if partition is None:
                result[role] = None
                continue
            if isinstance(partition, PartitionRef):
                result[role] = {
                    "count": int(partition.row_count),
                    "sha256": str(partition.fingerprint),
                    "manifest_sha256": str(partition.manifest.sha256),
                    "manifest_role": str(partition.role),
                }
                continue
            encoded = "\0".join(
                str(value) for value in partition.record_ids
            ).encode("utf-8")
            result[role] = {
                "count": len(partition.record_ids),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        return result

    def _ensure_train_split_normalization(
        self,
        partition: Partition | PartitionRef,
    ) -> None:
        if not isinstance(partition, PartitionRef):
            return super()._ensure_train_split_normalization(partition)
        if getattr(self, "_train_split_normalization_done", False):
            return

        from ..normalization import should_compute_train_split_normalization

        if not should_compute_train_split_normalization(self.run.params):
            self._train_split_normalization_done = True
            return

        train_partition = _resolve_partition_ref(self, partition, "train")
        if train_partition is None:
            return
        image_column = (
            self.binding.image_column
            or self.run.params.get("image_column")
            or self.run.params.get("image_path_column")
        )
        reader = create_partition_reader(
            self.run.context,
            train_partition,
            default_batch_size=_source_batch_size(self.run.params),
        )
        info = apply_partition_reader_image_normalization(
            reader=reader,
            params=self.run.params,
            image_column=image_column,
            cancel_token=getattr(self.run, "cancel_token", None),
        )
        self._train_split_normalization_info = info
        self._train_split_normalization_done = True
        if info:
            self.run.log(
                message="Calculated image mean/std from streamed training split.",
                status="running",
                extra={"phase": "normalization", "normalization": info},
            )

    def _restore_resume_state(
        self,
        model: Any,
        components: Any,
        parts: Partitions,
        target: TargetSpec,
    ):
        model, components = super()._restore_resume_state(
            model, components, parts, target
        )
        self._set_train_stream_epoch(
            int(getattr(self.run, "start_epoch", 1) or 1)
        )
        return model, components

    def report_epoch(
        self,
        epoch: int,
        model: Any,
        *,
        train_metrics: Dict[str, Any],
    ):
        result = super().report_epoch(
            epoch,
            model,
            train_metrics=train_metrics,
        )
        self._set_train_stream_epoch(int(epoch) + 1)
        return result

    def _set_train_stream_epoch(self, epoch: int) -> None:
        dataset = getattr(self, "_train_stream_dataset", None)
        setter = getattr(dataset, "set_epoch", None)
        if callable(setter):
            setter(max(1, int(epoch)))

    def _make_streaming_loader(
        self,
        partition: Partition | PartitionRef,
        *,
        train: bool,
        task: str,
    ):
        if not isinstance(partition, PartitionRef):
            return super()._make_loader(partition, train=train)

        import torch
        from torch.utils.data import DataLoader

        self._ensure_train_split_normalization(partition)
        transform = (
            self.recipe.train_transform(self.run)
            if train
            else self.recipe.eval_transform(self.run)
        )
        dataset = _make_iterable_dataset(
            harness=self,
            partition=partition,
            transform=transform,
            train=train,
            task=task,
        )
        if train:
            self._train_stream_dataset = dataset

        return DataLoader(
            dataset,
            batch_size=int(self.run.params.get("batch_size", 128)),
            shuffle=False,
            num_workers=int(self.run.params.get("num_workers", 0)),
            pin_memory=str(self._device()).startswith("cuda"),
            collate_fn=_collate_with_image_size_hint,
            # Workers are restarted for each epoch so they receive the updated
            # deterministic epoch seed. Shared epoch state can be added later
            # before enabling persistent workers.
            persistent_workers=False,
        )

class StreamingTorchClassificationHarness(
    ManifestBackedHarnessMixin,
    _MaterializedClassificationHarness,
):
    def _make_loader(self, partition, *, train: bool):
        self._eval_classes = list(partition.classes)
        return self._make_streaming_loader(
            partition,
            train=train,
            task="classification",
        )

class StreamingTorchRegressionHarness(
    ManifestBackedHarnessMixin,
    _MaterializedRegressionHarness,
):
    def _make_loader(self, partition, *, train: bool):
        return self._make_streaming_loader(
            partition,
            train=train,
            task="regression",
        )

def _bind_materialized_partition_sources(parts: Partitions) -> None:
    """Point partition references at their physical role datasets.

    The split manifest remains the durable membership identity used for resume
    validation. Once a role-specific dataset exists, however, epoch reads should
    scan that dataset directly rather than repeatedly joining manifest ID batches
    back to the original source.
    """

    dataset_ids = dict(parts.materialized_split_dataset_ids or {})
    if not dataset_ids:
        return

    refs = dict(parts.partition_refs or {})
    for role, dataset_id in dataset_ids.items():
        ref = refs.get(role)
        if ref is None:
            continue
        refs[role] = replace(
            ref,
            dataset_id=str(dataset_id),
            source="dataset",
        )

    parts.partition_refs = refs
    if "train" in dataset_ids:
        parts.train_dataset_id = str(dataset_ids["train"])
    if "validation" in dataset_ids:
        parts.validation_dataset_id = str(dataset_ids["validation"])
    if "test" in dataset_ids:
        parts.test_dataset_id = str(dataset_ids["test"])

    generation = dict(parts.split_generation or {})
    generation["physical_access"] = {
        role: {
            "dataset_id": str(dataset_id),
            "mode": "sequential_parquet_scan",
        }
        for role, dataset_id in dataset_ids.items()
    }
    parts.split_generation = generation


def _promote_partition_refs(parts: Partitions, harness: Any) -> Partitions:
    refs = dict(parts.partition_refs or {})
    if not refs:
        raise RuntimeError("Split manifest was created without partition references.")
    parts.train = refs["train"]
    parts.val = refs["validation"]
    parts.test = refs.get("test")

    harness._frame = None
    partition_frames = getattr(harness, "_partition_frames", None)
    if isinstance(partition_frames, dict):
        partition_frames.clear()
    return parts

def _require_materialized_partition(value: Any, role: str) -> Partition:
    if isinstance(value, Partition):
        return value
    raise TypeError(
        f"Expected a materialised {role} partition before manifest creation, "
        f"got {type(value).__name__}."
    )

def _resolve_partition_ref(
    harness: Any,
    current: PartitionRef,
    role: str,
) -> Optional[PartitionRef]:
    if str(current.role) == str(role):
        return current
    parts = (
        getattr(harness, "_parts", None)
        or getattr(harness, "_partitions", None)
        or getattr(harness, "partitions", None)
    )
    if parts is None:
        return None
    return parts.partition_ref(role)

def _make_iterable_dataset(
    *,
    harness: Any,
    partition: PartitionRef,
    transform: Any,
    train: bool,
    task: str,
):
    import torch
    from torch.utils.data import IterableDataset, get_worker_info

    recipe = harness.recipe
    run = harness.run
    binding = harness.binding
    class_to_idx = {value: index for index, value in enumerate(partition.classes)}
    n_outputs = harness._n_outputs() if task == "regression" else 0
    requested_columns = list(binding.input_columns or [])
    if binding.image_column:
        requested_columns.append(binding.image_column)
    if binding.target_column:
        requested_columns.append(binding.target_column)
    requested_columns = list(dict.fromkeys(requested_columns))
    source_batch_size = _source_batch_size(run.params)
    shuffle_buffer_size = _shuffle_buffer_size(run.params)

    class _ManifestDataset(IterableDataset):
        def __init__(self):
            super().__init__()
            self.epoch = max(1, int(getattr(run, "start_epoch", 1) or 1))

        def __len__(self):
            return int(partition.row_count)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = max(1, int(epoch))

        def __iter__(self) -> Iterator[Any]:
            worker = get_worker_info()
            worker_id = int(worker.id) if worker is not None else 0
            worker_count = int(worker.num_workers) if worker is not None else 1
            reader = create_partition_reader(
                run.context,
                partition,
                default_batch_size=source_batch_size,
            )
            rows = _iter_worker_rows(
                reader,
                columns=requested_columns,
                worker_id=worker_id,
                worker_count=worker_count,
                cancel_check=run.check_cancelled,
            )
            if train:
                seed = _epoch_seed(
                    int(getattr(harness.protocol, "random_state", 42)),
                    self.epoch,
                    worker_id,
                )
                rows = _bounded_shuffle(
                    rows,
                    buffer_size=shuffle_buffer_size,
                    seed=seed,
                )
            for row in rows:
                run.check_cancelled()
                x = recipe.load_sample(run, row)
                if transform is not None:
                    x = transform(x)
                _validate_sample_shape(recipe, x, row, binding.record_id_column)

                if task == "classification":
                    y = (
                        class_to_idx[str(row[binding.target_column])]
                        if binding.target_column
                        else -1
                    )
                    target = int(y)
                else:
                    values = (
                        harness._read_target(
                            row[binding.target_column], n_outputs
                        )
                        if binding.target_column
                        else [0.0] * n_outputs
                    )
                    target = torch.tensor(values, dtype=torch.float32)
                yield x, target, str(row[binding.record_id_column])

    return _ManifestDataset()

def _iter_worker_rows(
    reader: PartitionReader,
    *,
    columns: Sequence[str],
    worker_id: int,
    worker_count: int,
    cancel_check: Any,
) -> Iterator[Any]:
    for batch in reader.iter_batches(
        columns=columns,
        batch_size=reader.default_batch_size,
        strict=True,
        cancel_check=cancel_check,
        shard_index=worker_id,
        shard_count=worker_count,
    ):
        for _index, row in batch.frame.iterrows():
            yield row

def _bounded_shuffle(
    rows: Iterator[Any],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[Any]:
    rng = random.Random(int(seed))
    buffer = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer

def _validate_sample_shape(
    recipe: Any,
    sample: Any,
    row: Any,
    record_id_column: str,
) -> None:
    shape = tuple(getattr(sample, "shape", ()) or ())
    if len(shape) < 3:
        return
    height, width = int(shape[-2]), int(shape[-1])
    if height > 0 and width > 0:
        return
    raise ValueError(
        f"{recipe.id} produced an invalid image tensor shape {shape} "
        f"for row {row.get(record_id_column)!r}."
    )

def _collate_with_image_size_hint(batch):
    from torch.utils.data._utils.collate import default_collate

    try:
        return default_collate(batch)
    except RuntimeError as exc:
        if "stack expects each tensor to be equal size" not in str(exc):
            raise
        shapes = [tuple(getattr(x, "shape", ()) or ()) for x, _y, _rid in batch[:8]]
        row_ids = [str(rid) for _x, _y, rid in batch[:8]]
        raise ValueError(
            "Image tensors in this batch have different shapes, so PyTorch "
            "cannot stack them. Image recipes must enforce a fixed output "
            "size in both train_transform() and eval_transform(). "
            f"First batch shapes: {shapes}; row_ids: {row_ids}"
        ) from exc

def _selected_training_row_ids(
    params: Mapping[str, Any],
) -> Optional[list[str]]:
    raw = (
        params.get("training_row_ids")
        or params.get("selected_row_ids")
        or params.get("row_ids")
    )
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [
            part.strip()
            for part in raw.replace("\n", ",").split(",")
            if part.strip()
        ]
    else:
        values = [str(value).strip() for value in raw if str(value).strip()]
    return list(dict.fromkeys(values))


def _split_scan_batch_size(params: Mapping[str, Any]) -> int:
    value = int(params.get("split_scan_batch_size") or 65_536)
    if value <= 0:
        raise ValueError("split_scan_batch_size must be greater than zero.")
    return value

def _source_batch_size(params: Mapping[str, Any]) -> int:
    value = int(params.get("stream_source_batch_size") or 8192)
    if value <= 0:
        raise ValueError("stream_source_batch_size must be greater than zero.")
    return value

def _shuffle_buffer_size(params: Mapping[str, Any]) -> int:
    batch_size = int(params.get("batch_size") or 128)
    value = int(
        params.get("stream_shuffle_buffer_size")
        or max(4096, batch_size * 32)
    )
    if value <= 0:
        raise ValueError("stream_shuffle_buffer_size must be greater than zero.")
    return value

def _epoch_seed(base_seed: int, epoch: int, worker_id: int) -> int:
    return int(base_seed) + int(epoch) * 1_000_003 + int(worker_id) * 97_409