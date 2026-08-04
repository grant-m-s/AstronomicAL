from __future__ import annotations

import hashlib
import math
import random
import time
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
        scan_batch_size = _split_scan_batch_size(self.run.params)
        self._progress_report(
            stage="partitioning",
            message="Streaming record IDs, targets, and protocol columns into a disk-backed split manifest.",
            detail=f"Source scan batch size: `{scan_batch_size:,}` rows.",
            force=True,
        )
        parts = create_streaming_partitions(
            context=self.run.context,
            root=manifest_root,
            run_id=self.run.run_id,
            source_dataset_id=self.run.dataset_id,
            protocol=self.protocol,
            binding=self.binding,
            task_kind=self._task_kind(),
            batch_size=scan_batch_size,
            cancel_check=self.run.check_cancelled,
            selected_row_ids=_selected_training_row_ids(self.run.params),
        )
        self._progress_report(
            stage="partitioning",
            message="Split membership is ready. Materialising configured role datasets for repeated epoch scans.",
            detail=(
                "This creates bounded Parquet-backed train/validation/test sources when "
                "protocol_materialize_split_datasets is enabled."
            ),
            force=True,
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
        total_image_records = max(
            0,
            int(getattr(train_partition, "row_count", 0) or 0),
        )
        progress_reader = _NormalizationProgressReader(
            reader,
            harness=self,
            image_column=str(image_column or ""),
            total_image_records=total_image_records,
        )
        with self._progress_activity(
            stage="normalization",
            message="Reading and decoding training images to calculate channel mean and standard deviation.",
            detail=(
                f"Image column `{image_column}`; source batch size "
                f"`{_source_batch_size(self.run.params):,}` rows. "
                "Each completed source batch updates the running channel statistics."
            ),
            current=0,
            total=total_image_records or None,
            unit="image records",
            heartbeat_seconds=3.0,
        ):
            info = apply_partition_reader_image_normalization(
                reader=progress_reader,
                params=self.run.params,
                image_column=image_column,
                cancel_token=getattr(self.run, "cancel_token", None),
            )
        self._train_split_normalization_info = info
        self._train_split_normalization_done = True
        if info:
            self._progress_report(
                stage="normalization",
                message="Training-image normalisation statistics are ready.",
                detail=f"Calculated statistics: `{json_safe(info)}`",
                current=1,
                total=1,
                unit="normalisation pass",
                extra={"normalization": info},
                force=True,
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
        batch_encoded = _supports_batch_encoding(self.recipe)
        dataset = _make_iterable_dataset(
            harness=self,
            partition=partition,
            transform=transform,
            train=train,
            task=task,
        )
        if train:
            self._train_stream_dataset = dataset

        role = str(getattr(partition, "role", "train" if train else "validation"))
        encoding_detail = (
            "vectorised recipe batch encoding"
            if batch_encoded
            else "sample-level recipe encoding with DataLoader collation"
        )
        self._progress_report(
            stage="loading_data",
            message=f"Prepared streamed `{role}` data access.",
            detail=(
                f"Partition rows `{int(getattr(partition, 'row_count', 0) or 0):,}`, "
                f"training batch size `{int(self.run.params.get('batch_size', 128)):,}`, "
                f"workers `{int(self.run.params.get('num_workers', 0))}`, "
                f"encoding mode: {encoding_detail}."
            ),
            current=int(getattr(partition, "row_count", 0) or 0),
            total=int(getattr(partition, "row_count", 0) or 0),
            unit="rows configured",
            force=True,
        )

        common_kwargs = {
            "shuffle": False,
            "num_workers": int(self.run.params.get("num_workers", 0)),
            "pin_memory": str(self._device()).startswith("cuda"),
            # Workers are restarted for each epoch so they receive the updated
            # deterministic epoch seed. Shared epoch state can be added later
            # before enabling persistent workers.
            "persistent_workers": False,
        }
        if batch_encoded:
            return DataLoader(
                dataset,
                batch_size=None,
                collate_fn=_identity_batch,
                **common_kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=int(self.run.params.get("batch_size", 128)),
            collate_fn=_collate_with_image_size_hint,
            **common_kwargs,
        )

class _NormalizationProgressReader:
    """Transparent PartitionReader proxy with post-batch progress reporting.

    The normalization implementation owns image loading and pixel accumulation.
    Its batch iterator requests the next source batch only after the previous
    batch has been fully consumed, so reporting after ``yield`` represents rows
    whose image decoding/statistics work has completed rather than rows merely
    fetched from the dataset source.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        harness: Any,
        image_column: str,
        total_image_records: int,
    ) -> None:
        self._delegate = delegate
        self._harness = harness
        self._image_column = str(image_column or "")
        self._total = max(0, int(total_image_records or 0))
        self._started_at = time.time()
        self._completed = 0
        self._batch_index = 0

    def iter_batches(self, *args: Any, **kwargs: Any):
        for batch in self._delegate.iter_batches(*args, **kwargs):
            self._batch_index += 1
            batch_rows = _dataset_batch_row_count(batch)
            next_completed = self._completed + batch_rows
            self._harness._progress_report(
                stage="normalization",
                message="Decoding the next training-image batch and accumulating per-channel pixel statistics.",
                detail=(
                    f"Batch `{self._batch_index:,}` contains `{batch_rows:,}` image records "
                    f"from column `{self._image_column}`. The completed count advances "
                    "after this batch has been decoded and included in the running mean/variance."
                ),
                current=self._completed,
                total=self._total or None,
                unit="image records",
            )

            yield batch

            self._completed = next_completed
            elapsed = max(0.0, time.time() - self._started_at)
            rate = self._completed / elapsed if elapsed > 0 else 0.0
            detail = (
                f"Completed source batch `{self._batch_index:,}`. Decoded image pixels "
                "have been accumulated into the running channel sum and squared-sum."
            )
            if rate > 0:
                detail += f" Average throughput: `{rate:,.1f}` image records/s."
            self._harness._progress_report(
                stage="normalization",
                message="Calculating training-image channel mean and standard deviation.",
                detail=detail,
                current=self._completed,
                total=self._total or None,
                unit="image records",
            )

        self._harness._progress_report(
            stage="normalization",
            message="All requested training-image records have been processed; finalising channel statistics.",
            detail=(
                "Combining the accumulated pixel counts, sums, and squared sums into "
                "the final per-channel mean and standard deviation."
            ),
            current=self._completed,
            total=self._total or self._completed or None,
            unit="image records",
            force=True,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

def _dataset_batch_row_count(batch: Any) -> int:
    frame = getattr(batch, "frame", None)
    if frame is not None:
        try:
            return max(0, int(len(frame)))
        except Exception:
            pass

    for attr in ("row_count", "count", "size"):
        value = getattr(batch, attr, None)
        if value is None or callable(value):
            continue
        try:
            return max(0, int(value))
        except Exception:
            pass

    try:
        return max(0, int(len(batch)))
    except Exception:
        return 0

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
    if _supports_batch_encoding(harness.recipe):
        return _make_batch_iterable_dataset(
            harness=harness,
            partition=partition,
            train=train,
            task=task,
        )

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

def _supports_batch_encoding(recipe: Any) -> bool:
    method = getattr(recipe, "supports_batch_encoding", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    return False


def _make_batch_iterable_dataset(
    *,
    harness: Any,
    partition: PartitionRef,
    train: bool,
    task: str,
):
    """Create an IterableDataset that yields complete model-ready batches.

    DataLoader automatic batching is disabled for this dataset. Each yielded
    item is already ``(inputs, targets, record_ids)`` for one bounded batch.
    """
    from torch.utils.data import IterableDataset, get_worker_info

    recipe = harness.recipe
    run = harness.run
    binding = harness.binding
    class_to_idx = {
        str(value): index
        for index, value in enumerate(partition.classes)
    }
    n_outputs = harness._n_outputs() if task == "regression" else 0
    requested_columns = list(binding.input_columns or [])
    if binding.target_column:
        requested_columns.append(binding.target_column)
    requested_columns.append(binding.record_id_column)
    requested_columns = list(
        dict.fromkeys(str(column) for column in requested_columns if column)
    )
    source_batch_size = _source_batch_size(run.params)
    output_batch_size = int(run.params.get("batch_size", 128) or 128)
    if output_batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    shuffle_buffer_size = max(
        output_batch_size,
        _shuffle_buffer_size(run.params),
    )

    class _VectorizedManifestDataset(IterableDataset):
        def __init__(self):
            super().__init__()
            self.epoch = max(
                1,
                int(getattr(run, "start_epoch", 1) or 1),
            )

        def __len__(self):
            return int(
                math.ceil(
                    int(partition.row_count) / float(output_batch_size)
                )
            )

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
            frames = _iter_worker_frames(
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
                frames = _bounded_shuffle_frames(
                    frames,
                    output_batch_size=output_batch_size,
                    buffer_size=shuffle_buffer_size,
                    seed=seed,
                )
            else:
                frames = _rebatch_frames(
                    frames,
                    output_batch_size=output_batch_size,
                )

            for frame in frames:
                run.check_cancelled()
                inputs = recipe.encode_batch(
                    run,
                    frame,
                    train=bool(train),
                )
                _validate_encoded_batch(
                    recipe,
                    inputs,
                    expected_rows=len(frame),
                )
                targets = _encode_batch_targets(
                    harness=harness,
                    frame=frame,
                    task=task,
                    class_to_idx=class_to_idx,
                    n_outputs=n_outputs,
                )
                record_ids = _batch_record_ids(
                    frame,
                    binding.record_id_column,
                )
                yield inputs, targets, record_ids

    return _VectorizedManifestDataset()


def _iter_worker_frames(
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
        frame = batch.frame
        if frame is None or frame.empty:
            continue
        yield frame.reset_index(drop=True)


def _rebatch_frames(
    frames: Iterator[Any],
    *,
    output_batch_size: int,
) -> Iterator[Any]:
    """Re-slice source frames into fixed-size model batches without row objects."""
    import pandas as pd

    pieces = []
    piece_rows = 0
    for frame in frames:
        start = 0
        frame_rows = int(len(frame))
        while start < frame_rows:
            take = min(
                int(output_batch_size) - piece_rows,
                frame_rows - start,
            )
            pieces.append(frame.iloc[start : start + take])
            piece_rows += take
            start += take
            if piece_rows == int(output_batch_size):
                if len(pieces) == 1:
                    output = pieces[0].reset_index(drop=True).copy()
                else:
                    output = pd.concat(pieces, ignore_index=True)
                yield output
                pieces = []
                piece_rows = 0

    if pieces:
        if len(pieces) == 1:
            yield pieces[0].reset_index(drop=True).copy()
        else:
            yield pd.concat(pieces, ignore_index=True)


def _bounded_shuffle_frames(
    frames: Iterator[Any],
    *,
    output_batch_size: int,
    buffer_size: int,
    seed: int,
) -> Iterator[Any]:
    """Deterministically shuffle bounded dataframe blocks and emit full batches.

    Half of each shuffled buffer is retained so rows can mix with the next source
    block. Memory remains bounded by roughly ``buffer_size + source_batch_size``.
    """
    import numpy as np
    import pandas as pd

    output_batch_size = max(1, int(output_batch_size))
    buffer_size = max(output_batch_size, int(buffer_size))
    retain_target = max(output_batch_size, buffer_size // 2)
    rng = np.random.default_rng(int(seed))
    buffered = []
    buffered_rows = 0

    for frame in frames:
        buffered.append(frame)
        buffered_rows += int(len(frame))
        if buffered_rows < buffer_size:
            continue

        merged = (
            buffered[0].reset_index(drop=True).copy()
            if len(buffered) == 1
            else pd.concat(buffered, ignore_index=True)
        )
        order = rng.permutation(len(merged))
        merged = merged.iloc[order].reset_index(drop=True)

        emit_rows = max(0, len(merged) - retain_target)
        emit_rows -= emit_rows % output_batch_size
        for start in range(0, emit_rows, output_batch_size):
            yield merged.iloc[
                start : start + output_batch_size
            ].reset_index(drop=True).copy()

        remainder = merged.iloc[emit_rows:].reset_index(drop=True)
        buffered = [remainder] if not remainder.empty else []
        buffered_rows = int(len(remainder))

    if not buffered:
        return

    merged = (
        buffered[0].reset_index(drop=True).copy()
        if len(buffered) == 1
        else pd.concat(buffered, ignore_index=True)
    )
    order = rng.permutation(len(merged))
    merged = merged.iloc[order].reset_index(drop=True)
    for start in range(0, len(merged), output_batch_size):
        yield merged.iloc[
            start : start + output_batch_size
        ].reset_index(drop=True).copy()


def _encode_batch_targets(
    *,
    harness: Any,
    frame: Any,
    task: str,
    class_to_idx: Mapping[str, int],
    n_outputs: int,
):
    import numpy as np
    import pandas as pd
    import torch

    target_column = harness.binding.target_column
    if task == "classification":
        if not target_column:
            return torch.full(
                (len(frame),),
                -1,
                dtype=torch.long,
            )
        labels = frame[target_column].astype(str)
        mapped = labels.map(class_to_idx)
        invalid = mapped.isna()
        if bool(invalid.any()):
            position = int(np.flatnonzero(invalid.to_numpy())[0])
            record_ids = _batch_record_ids(
                frame,
                harness.binding.record_id_column,
            )
            raise ValueError(
                f"Target label {labels.iloc[position]!r} for record "
                f"{record_ids[position]!r} is not present in the partition "
                f"class list {list(class_to_idx)!r}."
            )
        return torch.from_numpy(
            mapped.to_numpy(dtype=np.int64, copy=True)
        )

    if not target_column:
        return torch.zeros(
            (len(frame), max(1, int(n_outputs))),
            dtype=torch.float32,
        )

    if int(n_outputs) == 1:
        numeric = pd.to_numeric(
            frame[target_column],
            errors="coerce",
        ).to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            position = int(np.flatnonzero(invalid)[0])
            record_ids = _batch_record_ids(
                frame,
                harness.binding.record_id_column,
            )
            raise ValueError(
                f"Regression target for record {record_ids[position]!r} "
                f"is not finite: {frame[target_column].iloc[position]!r}."
            )
        return torch.from_numpy(numeric.reshape(-1, 1))

    values = np.asarray(
        [
            harness._read_target(value, int(n_outputs))
            for value in frame[target_column].to_numpy(copy=False)
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(values)


def _batch_record_ids(frame: Any, record_id_column: str) -> list[str]:
    if not record_id_column or record_id_column not in frame.columns:
        raise ValueError(
            "The streamed batch does not contain the configured record-id "
            f"column {record_id_column!r}."
        )
    return [
        str(value)
        for value in frame[record_id_column].to_numpy(copy=False)
    ]


def _validate_encoded_batch(
    recipe: Any,
    inputs: Any,
    *,
    expected_rows: int,
) -> None:
    actual = _leading_batch_size(inputs)
    if actual is None:
        raise ValueError(
            f"{recipe.id} encode_batch() returned an object without a "
            "detectable leading batch dimension."
        )
    if int(actual) != int(expected_rows):
        raise ValueError(
            f"{recipe.id} encode_batch() returned leading dimension "
            f"{actual}, but the dataframe batch contains {expected_rows} rows."
        )


def _leading_batch_size(value: Any) -> Optional[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except Exception:
            pass
    if isinstance(value, Mapping):
        for item in value.values():
            size = _leading_batch_size(item)
            if size is not None:
                return size
    if isinstance(value, (tuple, list)) and value:
        for item in value:
            size = _leading_batch_size(item)
            if size is not None:
                return size
    return None


def _identity_batch(batch: Any) -> Any:
    return batch


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
