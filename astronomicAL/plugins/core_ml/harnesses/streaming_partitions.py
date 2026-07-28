from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from ..paths import ml_run_artifact_dir
from ..protocol import PartitionRef, Partitions
from ..serialization import json_safe
from ..split_planner import create_streaming_partitions
from ..streaming_split_datasets import materialize_streaming_split_datasets


class StreamingPartitionHarnessMixin:
    """Shared disk-backed split implementation for bounded-data harnesses.

    This mixin contains no framework imports. PyTorch, incremental sklearn, and
    external-memory XGBoost harnesses can therefore share the same manifest and
    resume identity without making optional frameworks depend on one another.
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
            batch_size=split_scan_batch_size(self.run.params),
            cancel_check=self.run.check_cancelled,
            selected_row_ids=selected_training_row_ids(self.run.params),
        )
        parts.materialized_split_dataset_ids = materialize_streaming_split_datasets(
            run=self.run,
            protocol=self.protocol,
            binding=self.binding,
            parts=parts,
        )
        bind_materialized_partition_sources(parts)
        return promote_partition_refs(parts, self)

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
                    if selected_training_row_ids(self.run.params) is not None
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


def bind_materialized_partition_sources(parts: Partitions) -> None:
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


def promote_partition_refs(parts: Partitions, harness: Any) -> Partitions:
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


def selected_training_row_ids(params: Mapping[str, Any]) -> Optional[list[str]]:
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


def split_scan_batch_size(params: Mapping[str, Any]) -> int:
    value = int(params.get("split_scan_batch_size") or 65_536)
    if value <= 0:
        raise ValueError("split_scan_batch_size must be greater than zero.")
    return value
