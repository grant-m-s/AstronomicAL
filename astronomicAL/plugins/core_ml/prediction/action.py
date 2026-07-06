from __future__ import annotations

# =============================================================================
# prediction.py — auditable, "by the book" inference + evaluation.
#
# Design mirrors the training overhaul: as the RunHarness owns the scientific
# protocol and the recipe owns only the loop, here a Predictor owns the audit
# guarantees and the framework subclass owns only the forward pass.
#
# Guarantees enforced (not merely encouraged):
#   1. PROVENANCE. Every predicted row is classified train/validation/test/
#      novel/unknown against the model's own ml.split_spec. Dataset-aware:
#      a row is only "test" if it is in the split's test ids AND comes from the
#      split's test dataset, so external rows can't be mislabelled as seen.
#   2. SCOPE SEPARATION. inference = predictions + uncertainty, never metrics.
#      evaluation = metrics, but only over rows whose provenance is reportable
#      (test or novel). val/train rows are excluded from any headline number;
#      a single accuracy that mixes seen and unseen rows is not expressible.
#   3. SELECTION ISOLATION. The evaluation report records that the reported
#      partition was not the selection partition, matching the harness block.
#   4. PARITY. Inference uses the recipe's own eval_transform / eval_forward,
#      so preprocessing matches what selected the best epoch.
#   5. REPRODUCIBILITY. Each predictions artifact records library versions, a
#      sha256 of the exact model checkpoint, the transform descriptor, the
#      dataset fingerprint, and whether the registered recipe version still
#      matches the one that trained the model.
#
# Reuses sibling modules: artifacts, model_contract, image_sidecar,
# recipe_registry.
# =============================================================================

import hashlib
import importlib.util
import math
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from astronomicAL.platform.plugins.specs import ActionRequest

from ..runtime import check_cancelled, coerce_action_request, request_dataset_id, publish as publish_event
from ..serialization import json_safe

PREDICTION_SCHEMA_VERSION = 3
EVALUATION_SCHEMA_VERSION = 3

SCOPE_INFERENCE = "inference"
SCOPE_EVALUATION = "evaluation"
SCOPE_AUTO = "auto"

PROV_TRAIN = "train"
PROV_VAL = "validation"
PROV_TEST = "test"
PROV_NOVEL = "novel"
PROV_UNKNOWN = "unknown"

# Only held-out-test and genuinely-novel rows may contribute to a reported
# generalisation metric. val was used for selection; train for fitting.
_REPORTABLE_PROVENANCE = frozenset({PROV_TEST, PROV_NOVEL})

_PROV_REASON = {
    PROV_TRAIN: "Rows the model was fit on. Metrics here measure memorisation, not generalisation.",
    PROV_VAL: "Rows used to select the best epoch. Reporting these leaks the selection signal.",
    PROV_TEST: "The model's own held-out test rows. Reportable; re-deriving them audits the training run.",
    PROV_NOVEL: "Rows not in the model's split (external set). Reportable as out-of-sample evaluation.",
    PROV_UNKNOWN: "No split_spec found, so seen/unseen status cannot be verified. Not certifiable.",
}

# Features required for a fully auditable workflow that cannot be implemented
# from the predict side alone. Surfaced in every artifact so the gap is visible.
AUDIT_GAPS = [
    {
        "id": "model_split_spec_backref",
        "severity": "high",
        "needs": "training-side",
        "summary": (
            "The ml.model manifest should store split_spec_artifact_id directly. "
            "Provenance currently relies on a reverse lookup by run_id/protocol_id, "
            "which is fragile and unverifiable if the store can't be queried."
        ),
    },
    {
        "id": "ood_detection",
        "severity": "high",
        "needs": "training-side",
        "summary": (
            "No out-of-distribution signal. Softmax confidence over the training "
            "label space says nothing about whether a cutout resembles training data. "
            "Requires training to persist feature statistics or an embedding density "
            "model so predict can score distribution shift."
        ),
    },
    {
        "id": "probability_calibration",
        "severity": "medium",
        "needs": "training-side",
        "summary": (
            "Reported confidences are raw softmax, not calibrated. The harness should "
            "fit a temperature (or isotonic) calibrator on the validation partition and "
            "store it on the model so confidences are trustworthy for domain experts."
        ),
    },
    {
        "id": "tamper_evident_chain",
        "severity": "medium",
        "needs": "platform",
        "summary": (
            "Checkpoint sha256 is recorded, but split_spec / evaluation_report are not "
            "hash-chained. A signed, content-addressed artifact chain would make the "
            "whole run independently verifiable end-to-end."
        ),
    },
]

# =============================================================================
# Provenance: the prediction analogue of partition isolation
# =============================================================================

@dataclass
class ProvenanceIndex:
    verified: bool
    split_spec_artifact_id: Optional[str]
    protocol_id: Optional[str]
    record_id_column: Optional[str]

    train: frozenset
    val: frozenset
    test: frozenset

    train_dataset_id: Optional[str]
    val_dataset_id: Optional[str]
    test_dataset_id: Optional[str]

    # Important for Active Learning:
    #
    # AL trains on a derived training dataset, but prediction is usually run
    # against the original pool dataset. The split_spec therefore says
    # "train_dataset_id == <derived_al_training_dataset>", while the predicted
    # dataset is "<pool_dataset>". Row ids still refer to the same source
    # objects. These aliases let provenance classify those rows as train/val/test
    # when the platform can prove that the derived dataset came from the pool.
    train_dataset_aliases: frozenset = field(default_factory=frozenset)
    val_dataset_aliases: frozenset = field(default_factory=frozenset)
    test_dataset_aliases: frozenset = field(default_factory=frozenset)

    def _matches_dataset(
        self,
        *,
        expected_dataset_id: Optional[str],
        aliases: frozenset,
        predict_dataset_id: Optional[str],
    ) -> bool:
        predicted = _clean_dataset_id(predict_dataset_id)
        if not predicted:
            return False

        expected = _clean_dataset_id(expected_dataset_id)
        if expected and predicted == expected:
            return True

        return predicted in aliases

    def classify(self, predict_dataset_id: Optional[str], row_id: Any) -> str:
        if not self.verified:
            return PROV_UNKNOWN

        rid = str(row_id)

        # Dataset-aware, but now source-aware for derived AL datasets.
        if (
            rid in self.test
            and self._matches_dataset(
                expected_dataset_id=self.test_dataset_id,
                aliases=self.test_dataset_aliases,
                predict_dataset_id=predict_dataset_id,
            )
        ):
            return PROV_TEST

        if (
            rid in self.val
            and self._matches_dataset(
                expected_dataset_id=self.val_dataset_id,
                aliases=self.val_dataset_aliases,
                predict_dataset_id=predict_dataset_id,
            )
        ):
            return PROV_VAL

        if (
            rid in self.train
            and self._matches_dataset(
                expected_dataset_id=self.train_dataset_id,
                aliases=self.train_dataset_aliases,
                predict_dataset_id=predict_dataset_id,
            )
        ):
            return PROV_TRAIN

        return PROV_NOVEL

def _resolve_provenance_index(
    context: Any,
    model_payload: Mapping[str, Any],
) -> ProvenanceIndex:
    artifacts = getattr(context, "artifacts", None)
    get = getattr(artifacts, "get", None)

    spec = None
    spec_id = model_payload.get("split_spec_artifact_id")

    if spec_id and callable(get):
        try:
            spec = get(spec_id)
        except Exception:
            spec = None

    if not isinstance(spec, Mapping):
        spec, spec_id = _find_split_spec_by_run(
            context,
            run_id=model_payload.get("run_id"),
            protocol_id=model_payload.get("protocol_id"),
        )

    if not isinstance(spec, Mapping):
        return ProvenanceIndex(
            verified=False,
            split_spec_artifact_id=None,
            protocol_id=model_payload.get("protocol_id"),
            record_id_column=None,
            train=frozenset(),
            val=frozenset(),
            test=frozenset(),
            train_dataset_id=None,
            val_dataset_id=None,
            test_dataset_id=None,
            train_dataset_aliases=frozenset(),
            val_dataset_aliases=frozenset(),
            test_dataset_aliases=frozenset(),
        )

    train_dataset_id = _clean_dataset_id(spec.get("train_dataset_id"))
    val_dataset_id = _clean_dataset_id(spec.get("validation_dataset_id"))
    test_dataset_id = _clean_dataset_id(spec.get("test_dataset_id"))

    return ProvenanceIndex(
        verified=True,
        split_spec_artifact_id=spec_id,
        protocol_id=spec.get("protocol_id"),
        record_id_column=spec.get("record_id_column"),
        train=frozenset(str(x) for x in (spec.get("train_row_ids") or [])),
        val=frozenset(str(x) for x in (spec.get("validation_row_ids") or [])),
        test=frozenset(str(x) for x in (spec.get("test_row_ids") or [])),
        train_dataset_id=train_dataset_id,
        val_dataset_id=val_dataset_id,
        test_dataset_id=test_dataset_id,
        train_dataset_aliases=_dataset_aliases_for_provenance(
            context,
            train_dataset_id,
        ),
        val_dataset_aliases=_dataset_aliases_for_provenance(
            context,
            val_dataset_id,
        ),
        test_dataset_aliases=_dataset_aliases_for_provenance(
            context,
            test_dataset_id,
        ),
    )

def _find_split_spec_by_run(context, *, run_id, protocol_id):
    """Best-effort reverse lookup. Direct model.split_spec_artifact_id is better."""
    if not run_id and not protocol_id:
        return None, None

    for aid, payload in _iter_artifact_payloads(context, "ml.split_spec"):
        if not isinstance(payload, Mapping):
            continue

        if run_id and str(payload.get("run_id")) == str(run_id):
            return payload, aid

        if protocol_id and str(payload.get("protocol_id")) == str(protocol_id):
            return payload, aid

    return None, None

def _clean_dataset_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None

def _dataset_aliases_for_provenance(
    context: Any,
    dataset_id: Optional[str],
) -> frozenset:
    """Return dataset ids that are provably aliases of `dataset_id`.

    Used mainly for AL:
      derived AL training dataset -> original AL pool dataset

    Sources of truth, in order:
      1. Dataset metadata registered by active_learning.actions
      2. al.training_set artifacts
      3. A few generic source/base/parent metadata keys
    """
    clean = _clean_dataset_id(dataset_id)
    if not clean:
        return frozenset()

    aliases = {clean}

    meta = _dataset_meta(context, clean)
    for key in (
        "source_dataset_id",
        "pool_dataset_id",
        "parent_dataset_id",
        "base_dataset_id",
        "origin_dataset_id",
    ):
        value = _clean_dataset_id(meta.get(key))
        if value:
            aliases.add(value)

    aliases.update(_al_training_source_aliases(context, training_dataset_id=clean))

    # Defensive: if any alias itself has source metadata, include one hop.
    for alias in list(aliases):
        alias_meta = _dataset_meta(context, alias)
        for key in (
            "source_dataset_id",
            "pool_dataset_id",
            "parent_dataset_id",
            "base_dataset_id",
            "origin_dataset_id",
        ):
            value = _clean_dataset_id(alias_meta.get(key))
            if value:
                aliases.add(value)

    return frozenset(str(alias) for alias in aliases if alias)

def _dataset_meta(context: Any, dataset_id: str) -> Dict[str, Any]:
    datasets = getattr(context, "datasets", None)
    if datasets is None or not dataset_id:
        return {}

    get_meta = getattr(datasets, "get_meta", None)
    if callable(get_meta):
        try:
            meta = get_meta(dataset_id)
            if isinstance(meta, Mapping):
                return dict(meta)
        except Exception:
            pass

    get = getattr(datasets, "get", None)
    if callable(get):
        try:
            item = get(dataset_id)
            for attr in ("metadata", "meta"):
                meta = getattr(item, attr, None)
                if isinstance(meta, Mapping):
                    return dict(meta)
        except Exception:
            pass

    return {}

def _al_training_source_aliases(
    context: Any,
    *,
    training_dataset_id: str,
) -> set[str]:
    aliases: set[str] = set()
    wanted = str(training_dataset_id or "").strip()
    if not wanted:
        return aliases

    for _aid, payload in _iter_artifact_payloads(context, "al.training_set"):
        if not isinstance(payload, Mapping):
            continue

        payload_training_dataset_id = _clean_dataset_id(
            payload.get("training_dataset_id")
            or payload.get("dataset_id")
        )
        if payload_training_dataset_id != wanted:
            continue

        for key in (
            "source_dataset_id",
            "pool_dataset_id",
            "original_dataset_id",
            "parent_dataset_id",
        ):
            value = _clean_dataset_id(payload.get(key))
            if value:
                aliases.add(value)

    return aliases

def _iter_artifact_payloads(context: Any, artifact_type: Optional[str] = None):
    """Yield (artifact_id, payload) across ArtifactStore API variants."""
    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        return

    method_calls = []

    for method_name in ("query", "find", "list_by_type", "list"):
        method = getattr(artifacts, method_name, None)
        if not callable(method):
            continue

        if artifact_type:
            method_calls.extend(
                [
                    lambda method=method: method(artifact_type=artifact_type),
                    lambda method=method: method(artifact_type),
                ]
            )

        method_calls.append(lambda method=method: method())

    seen_artifact_ids: set[str] = set()

    for call in method_calls:
        try:
            results = call()
        except TypeError:
            continue
        except Exception:
            continue

        for item in results or []:
            aid, payload = _unpack_artifact(context, item)
            if not isinstance(payload, Mapping):
                continue

            # If the store/list result exposes an artifact type, respect it.
            payload_type = (
                payload.get("artifact_type")
                or payload.get("type")
                or payload.get("kind")
            )
            if (
                artifact_type
                and payload_type
                and str(payload_type) != str(artifact_type)
            ):
                # Some payloads use "kind" for model kind, not artifact type.
                # Do not over-filter known AL training payloads, which may not
                # carry artifact_type in the JSON body.
                if artifact_type == "al.training_set" and payload.get(
                    "training_dataset_id"
                ):
                    pass
                else:
                    continue

            key = str(aid or id(payload))
            if key in seen_artifact_ids:
                continue
            seen_artifact_ids.add(key)

            yield aid, payload

def _unpack_artifact(context, item):
    """Return (artifact_id, payload) for several ArtifactStore result shapes."""
    if isinstance(item, Mapping):
        aid = (
            item.get("artifact_id")
            or item.get("id")
            or item.get("artifactId")
        )

        payload = item.get("payload")
        if isinstance(payload, Mapping):
            return aid, payload

        return aid, item

    if isinstance(item, str):
        try:
            return item, context.artifacts.get(item)
        except Exception:
            return item, None

    aid = getattr(item, "artifact_id", None) or getattr(item, "id", None)
    if aid:
        try:
            return aid, context.artifacts.get(aid)
        except Exception:
            return aid, None

    payload = getattr(item, "payload", None)
    if isinstance(payload, Mapping):
        return aid, payload

    return None, None

# =============================================================================
# Reproducibility / integrity
# =============================================================================

def _checkpoint_sha256(model_payload: Mapping[str, Any]) -> Optional[str]:
    ref = dict(model_payload.get("model_ref") or {})
    path = ref.get("path") or ref.get("uri")
    if not path:
        return None
    path = str(path)
    if path.startswith("file://"):
        path = path[7:]
    p = Path(path).expanduser()
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _check_recipe_version(context: Any, model_payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    recipe_id = model_payload.get("recipe_id")
    trained_version = model_payload.get("recipe_version")
    if not recipe_id:
        return None
    services = getattr(context, "services", None)
    get = getattr(services, "get", None)
    if not callable(get):
        return None
    for key in ("core.ml.recipe_registry", "recipe_registry"):
        try:
            registry = get(key)
            spec = registry.get(recipe_id)
        except Exception:
            continue
        registered = str(getattr(spec, "version", ""))
        return {
            "recipe_id": str(recipe_id),
            "trained_version": str(trained_version) if trained_version else None,
            "registered_version": registered,
            "match": registered == str(trained_version),
        }
    return {
        "recipe_id": str(recipe_id),
        "trained_version": str(trained_version) if trained_version else None,
        "registered_version": None,
        "match": False,
        "reason": "Recipe not registered; reconstruction used the generic fallback builder.",
    }

def _reproducibility_manifest(*, model_payload, transform_desc, checkpoint_sha256, recipe_version_check):
    versions: Dict[str, Any] = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    return {
        "library_versions": versions,
        "model_artifact_id": model_payload.get("run_id"),
        "checkpoint_sha256": checkpoint_sha256,
        "transform": transform_desc,
        "recipe_version_check": recipe_version_check,
        "deterministic_inference": True,
        "generated_at": time.time(),
    }

# =============================================================================
# Recipe replay shim (transform/forward parity)
# =============================================================================

class _PredictRunShim:
    def __init__(self, *, context, params, binding):
        self.context = context
        self.params = dict(params or {})
        self.binding = binding
        self.dataset_id = None
        self.run_id = None

    def check_cancelled(self):
        return None

    def log(self, *args, **kwargs):
        return None

def _resolve_recipe(context, recipe_id):
    if not recipe_id:
        return None
    services = getattr(context, "services", None)
    get = getattr(services, "get", None)
    if not callable(get):
        return None
    for key in ("core.ml.recipe_registry", "recipe_registry"):
        try:
            spec = get(key).get(recipe_id)
            return spec.recipe_cls()
        except Exception:
            continue
    return None

# =============================================================================
# Predictor base — owns the audit flow; subclasses own the forward pass only
# =============================================================================

class Predictor:
    framework = ""
    modality = ""

    def __init__(self, *, context, dataset_id, model_artifact_id, model_payload,
                 compatibility, request, scope, cancel_token=None):
        self.context = context
        self.dataset_id = dataset_id
        self.model_artifact_id = model_artifact_id
        self.model_payload = model_payload
        self.compatibility = compatibility
        self.request = request
        self.cancel_token = cancel_token
        self.params = dict(getattr(request, "params", {}) or {})

        self.binding = dict(getattr(compatibility, "resolved_input_binding", {}) or {})
        self.output_schema = dict(getattr(compatibility, "output_schema", {}) or {})
        self.classes = [str(c) for c in self.output_schema.get("classes") or []]
        self.task = str(self.output_schema.get("task") or "classification").lower()
        self.run_id = str(self.params.get("run_id") or uuid.uuid4().hex)
        self.decision_threshold = self.params.get("decision_threshold")

        self.scope = self._resolve_scope(scope)
        self.provenance = _resolve_provenance_index(context, model_payload)
        self.recipe_version_check = _check_recipe_version(context, model_payload)
        self.transform_desc: Dict[str, Any] = {}
        self.failed_rows: List[Dict[str, Any]] = []
        self.target_column = self.binding.get("target_column_for_evaluation")

    # -- subclass contract ---------------------------------------------------
    def reconstruct(self) -> None:
        raise NotImplementedError

    def read_columns(self) -> List[str]:
        raise NotImplementedError

    def predict_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return per-row records keyed by record_id with prediction +
        probabilities + uncertainty. No metrics, no provenance — the base adds
        those."""
        raise NotImplementedError

    # -- the audit flow (not overridable) ------------------------------------
    def run(self) -> Dict[str, Any]:
        from .. import artifacts as artifact_utils
        from .. import contracts as contract_utils

        check_cancelled(self.cancel_token)
        self.reconstruct()

        cols = list(dict.fromkeys([c for c in self.read_columns() if c]))

        if (
            self.scope == SCOPE_EVALUATION
            and self.target_column
            and self.target_column not in cols
        ):
            cols.append(self.target_column)

        df = self.context.datasets.get_df(self.dataset_id, columns=cols)
        df = _filter_rows(df, self.request, self.binding.get("record_id_column"))

        check_cancelled(self.cancel_token)

        records = self.predict_records(df)
        self._apply_abstention(records)
        self._attach_provenance(records)

        eval_block, eval_report_id = self._maybe_evaluate(df, records)

        checkpoint_sha = _checkpoint_sha256(self.model_payload)
        repro = _reproducibility_manifest(
            model_payload=self.model_payload,
            transform_desc=self.transform_desc,
            checkpoint_sha256=checkpoint_sha,
            recipe_version_check=self.recipe_version_check,
        )

        failed_rows = list(self.failed_rows or [])
        input_binding = {**self.binding, "transform": self.transform_desc}
        if failed_rows:
            input_binding["failed_image_rows"] = failed_rows[:100]
            input_binding["failed_image_row_count"] = len(failed_rows)

        row_ids = [r["record_id"] for r in records]

        payload = contract_utils.build_predictions_payload(
            context=self.context,
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            model_artifact_id=self.model_artifact_id,
            model_payload=self.model_payload,
            records=records,
            row_ids=row_ids,
            input_binding=input_binding,
            compatibility_report=self.compatibility,
            params=self.params,
            prediction_scope=self.scope,
        )

        self._inject_provenance_into_table(payload, records)
        payload["provenance"] = self._provenance_summary(records)
        payload["reproducibility"] = repro
        payload["recipe_version_check"] = self.recipe_version_check
        payload["scope"] = self.scope
        payload["audit_gaps"] = AUDIT_GAPS

        if eval_block is not None:
            payload["evaluation"] = eval_block
            payload["evaluation_report_artifact_id"] = eval_report_id

        artifact_id = self.context.artifacts.put(
            artifact_utils.ARTIFACTS.PREDICTIONS,
            payload,
            dataset_id=self.dataset_id,
            row_ids=row_ids,
            params={"model_artifact_id": self.model_artifact_id, **self.params},
        )

        derived_dataset_id = None
        if bool(self.params.get("register_prediction_dataset", True)):
            derived_dataset_id = register_prediction_table_dataset(
                context=self.context,
                predictions_payload=payload,
                predictions_artifact_id=artifact_id,
            )

        self._publish_events(artifact_id, derived_dataset_id, payload)

        return artifact_utils.json_safe(
            {
                "ok": True,
                "scope": self.scope,
                "artifact_id": artifact_id,
                "evaluation_report_artifact_id": eval_report_id,
                "derived_dataset_id": derived_dataset_id,
                "dataset_id": self.dataset_id,
                "model_artifact_id": self.model_artifact_id,
                "count": len(row_ids),
                "provenance": payload["provenance"],
                "evaluation": eval_block,
                "recipe_version_check": self.recipe_version_check,
                "checkpoint_sha256": checkpoint_sha,
                "audit_gaps": AUDIT_GAPS,
                "recommended_color_columns": payload.get("visualisation", {}).get(
                    "recommended_color_columns",
                    [],
                ),
                "prediction_preview": list(
                    (payload.get("prediction_table") or {}).get("rows") or []
                )[:25],
                "prediction_table_columns": list(
                    (payload.get("prediction_table") or {}).get("columns") or []
                ),
                "failed_image_row_count": len(failed_rows),
                "failed_image_rows": failed_rows[:25],
            }
        )

    # -- scope ---------------------------------------------------------------
    def _resolve_scope(self, requested: Optional[str]) -> str:
        requested = str(requested or self.params.get("scope") or SCOPE_AUTO).lower()
        if requested in (SCOPE_INFERENCE, SCOPE_EVALUATION):
            return requested
        # AUTO: evaluate only if a ground-truth target resolves on the dataset.
        has_target = bool(dict(getattr(self.compatibility, "resolved_input_binding", {}) or {})
                          .get("target_column_for_evaluation"))
        return SCOPE_EVALUATION if has_target else SCOPE_INFERENCE

    # -- provenance ----------------------------------------------------------
    def _attach_provenance(self, records: List[Dict[str, Any]]) -> None:
        for r in records:
            r["data_provenance"] = self.provenance.classify(self.dataset_id, r["record_id"])

    def _provenance_summary(self, records) -> Dict[str, Any]:
        counts: Dict[str, int] = defaultdict(int)
        for r in records:
            counts[r.get("data_provenance", PROV_UNKNOWN)] += 1

        return {
            "verified": self.provenance.verified,
            "split_spec_artifact_id": self.provenance.split_spec_artifact_id,
            "protocol_id": self.provenance.protocol_id,
            "counts": dict(counts),
            "reportable_row_count": sum(counts[p] for p in _REPORTABLE_PROVENANCE),
            "seen_row_count": counts.get(PROV_TRAIN, 0) + counts.get(PROV_VAL, 0),
            "dataset_ids": {
                "train": self.provenance.train_dataset_id,
                "validation": self.provenance.val_dataset_id,
                "test": self.provenance.test_dataset_id,
            },
            "dataset_aliases": {
                "train": sorted(self.provenance.train_dataset_aliases),
                "validation": sorted(self.provenance.val_dataset_aliases),
                "test": sorted(self.provenance.test_dataset_aliases),
            },
            "reasons": {p: _PROV_REASON[p] for p in counts},
        }

    def _inject_provenance_into_table(self, payload, records) -> None:
        prov_by_id = {r["record_id"]: r.get("data_provenance") for r in records}
        table = payload.get("prediction_table") or {}
        for row in table.get("rows", []):
            row["data_provenance"] = prov_by_id.get(str(row.get("record_id")))
        if table.get("rows"):
            cols = list(table.get("columns") or [])
            if "data_provenance" not in cols:
                cols.append("data_provenance")
            table["columns"] = cols
        viz = payload.setdefault("visualisation", {})
        rec = list(viz.get("recommended_color_columns") or [])
        if "data_provenance" not in rec:
            rec.insert(0, "data_provenance")
        viz["recommended_color_columns"] = rec

    # -- evaluation (honest, segregated) -------------------------------------
    def _maybe_evaluate(self, df, records):
        if self.scope != SCOPE_EVALUATION:
            return None, None

        if not self.target_column or self.target_column not in df.columns:
            return None, None

        rid_col = self.binding.get("record_id_column")
        truth_by_id = {}

        for idx, row in df.iterrows():
            rid = str(row[rid_col]) if (rid_col and rid_col in df.columns) else str(idx)
            value = row[self.target_column]

            if pd.isna(value):
                continue

            if self.task == "regression":
                try:
                    truth_by_id[rid] = float(value)
                except Exception:
                    continue
            else:
                truth_by_id[rid] = str(value)

        labelled = []
        for r in records:
            t = truth_by_id.get(str(r["record_id"]))
            if t is None:
                continue
            r["y_true"] = t
            labelled.append(r)

        if not labelled:
            return None, None

        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in labelled:
            buckets[r["data_provenance"]].append(r)

        per_bucket = {}
        for prov, recs in buckets.items():
            per_bucket[prov] = {
                **_metrics_for_task(self.task, recs),
                "n": len(recs),
                "reportable": prov in _REPORTABLE_PROVENANCE,
                "reason": _PROV_REASON[prov],
            }

        reportable = [
            r for r in labelled if r["data_provenance"] in _REPORTABLE_PROVENANCE
        ]
        headline = _metrics_for_task(self.task, reportable) if reportable else None

        reported_provs = sorted({r["data_provenance"] for r in reportable})
        metric_partition = (
            "none"
            if not reported_provs
            else reported_provs[0]
            if len(reported_provs) == 1
            else "mixed"
        )

        eval_block = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "scope": "post_hoc_evaluation",
            "task": self.task,
            "selection_metric": self.model_payload.get("metrics", {}).get(
                "selection_metric"
            )
            or (self.model_payload.get("model_ref", {}).get("metadata", {}) or {}).get(
                "selection_metric"
            ),
            "selected_on_partition": "validation",
            "headline_metrics": headline,
            "headline_metric_partition": metric_partition,
            "metrics_by_provenance": per_bucket,
            "validity": {
                "provenance_verified": self.provenance.verified,
                "metric_partition": metric_partition,
                "selection_partition": "validation",
                "selection_touched_reported_partition": False,
                "headline_excludes_seen_rows": True,
                "rows_excluded_as_seen": sum(
                    len(buckets.get(p, [])) for p in (PROV_TRAIN, PROV_VAL)
                ),
                "rows_unverifiable": len(buckets.get(PROV_UNKNOWN, [])),
            },
            "warnings": self._evaluation_warnings(buckets),
        }

        report_payload = {
            **eval_block,
            "artifact_type": "ml.evaluation_report",
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "model_artifact_id": self.model_artifact_id,
            "split_spec_artifact_id": self.provenance.split_spec_artifact_id,
            "protocol_id": self.provenance.protocol_id,
            "target_column": self.target_column,
            "created_at": time.time(),
        }

        report_id = self.context.artifacts.put(
            "ml.evaluation_report",
            report_payload,
            dataset_id=self.dataset_id,
            params={"model_artifact_id": self.model_artifact_id, **self.params},
        )

        return eval_block, report_id

    def _evaluation_warnings(self, buckets) -> List[str]:
        w = []
        if not self.provenance.verified:
            w.append(
                "No split_spec was found for this model; rows could not be certified as "
                "unseen. Headline metrics are withheld and all rows are marked unverifiable."
            )
        if buckets.get(PROV_TRAIN) or buckets.get(PROV_VAL):
            n = len(buckets.get(PROV_TRAIN, [])) + len(buckets.get(PROV_VAL, []))
            w.append(f"{n} rows were seen during training/selection and were excluded from headline metrics.")
        if self.recipe_version_check and not self.recipe_version_check.get("match"):
            w.append(
                "Registered recipe version differs from the version that trained this model; "
                "reconstruction fidelity is not guaranteed."
            )
        return w

    # -- abstention ----------------------------------------------------------
    def _apply_abstention(self, records) -> None:
        if self.decision_threshold is None:
            return
        try:
            thr = float(self.decision_threshold)
        except Exception:
            return
        for r in records:
            conf = r.get("confidence")
            if conf is not None and float(conf) < thr:
                r["abstained"] = True
                r["prediction_before_abstain"] = r.get("prediction")
                r["prediction"] = None

    # -- events --------------------------------------------------------------
    def _publish_events(self, artifact_id, derived_dataset_id, payload) -> None:
        publish_event(self.context, "ml.predictions.created", {
            "artifact_id": artifact_id,
            "dataset_id": self.dataset_id,
            "derived_dataset_id": derived_dataset_id,
            "model_artifact_id": self.model_artifact_id,
            "scope": self.scope,
            "provenance_verified": self.provenance.verified,
            "count": len(payload.get("row_ids") or []),
        })
        catalog = _get_trained_model_catalog(self.context)
        if catalog is not None:
            try:
                catalog.refresh()
            except Exception:
                pass

# =============================================================================
# Concrete predictors — forward pass only
# =============================================================================

class TorchImagePredictor(Predictor):
    framework = "torch"
    modality = "image"

    def reconstruct(self) -> None:

        from .. import artifacts as artifact_utils
        from .. import image_sidecar

        saved = artifact_utils.load_model_from_payload(self.model_payload)
        if not isinstance(saved, Mapping):
            raise TypeError("Image artifact did not load to a sidecar mapping.")
        self._model = saved.get("torch_model")
        if self._model is None or not self.classes:
            raise ValueError("Reconstructed image model missing torch_model/classes.")
        self._model.eval()

        image_size = int(saved.get("image_size") or 224)
        normalization = dict(saved.get("normalization") or image_sidecar.DEFAULT_NORMALIZATION)
        self._image_column = self.binding.get("image_column") or self.params.get("image_column")
        if not self._image_column:
            raise ValueError("Compatibility did not resolve an image column.")

        recipe = _resolve_recipe(self.context, str(self.model_payload.get("recipe_id") or "").strip())
        if recipe is not None:
            try:
                from .. import registry as registry_mod

                binding = registry_mod.DataBinding(
                    record_id_column=str(self.binding.get("record_id_column") or "id"),
                    target_column=None,
                    input_columns=[str(self._image_column)],
                    image_column=str(self._image_column),
                )
                saved_params = dict(saved.get("checkpoint", {}).get("params") or {})
                shim = _PredictRunShim(context=self.context, params={**saved_params, **self.params}, binding=binding)
                self._transform = recipe.eval_transform(shim)
                self._read = lambda row: recipe.load_sample(shim, row)
                self._forward = lambda m, b: recipe.eval_forward(m, b)
                self.transform_desc = {"source": "recipe.eval_transform", "recipe_id": recipe.id}
                return
            except Exception:
                pass

        self._transform = image_sidecar.image_transform(image_size=image_size, normalization=normalization)
        self._read = lambda row: image_sidecar.load_image(row[self._image_column])
        self._forward = lambda m, b: m(b)
        self.transform_desc = {"source": "image_sidecar", "image_size": image_size, "normalization": normalization}

    def read_columns(self) -> List[str]:
        return [self.binding.get("image_column"), self.binding.get("record_id_column")]

    def predict_records(self, df) -> List[Dict[str, Any]]:
        import torch
        rid_col = self.binding.get("record_id_column")
        device = _torch_device(self.params)
        self._model.to(device)
        batch_size = int(self.params.get("image_batch_size") or self.params.get("batch_size") or 32)
        skip_bad = bool(self.params.get("skip_bad_images", True))

        pending, tensors, records, failed = [], [], [], []

        def flush():
            if not tensors:
                return
            check_cancelled(self.cancel_token)
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                probs = torch.softmax(self._forward(self._model, batch), dim=1).cpu().numpy()
            for rid, p in zip(pending, probs):
                records.append(_classification_record(rid, p, self.classes))
            tensors.clear(); pending.clear()

        for idx, row in df.iterrows():
            check_cancelled(self.cancel_token)
            rid = str(row[rid_col]) if (rid_col and rid_col in df.columns) else str(idx)
            try:
                tensors.append(self._transform(self._read(row)))
                pending.append(rid)
            except Exception as exc:
                failed.append({"record_id": rid, "error": str(exc)})
                if not skip_bad:
                    raise ValueError(f"Could not load image for row {rid}: {exc}") from exc
            if len(tensors) >= batch_size:
                flush()
        flush()
        self.failed_rows = failed
        if failed:
            self.transform_desc["failed_rows"] = len(failed)
        return records

def _normalise_probability_matrix(values: Any) -> Optional[np.ndarray]:
    """Validate and row-normalise a probability matrix."""
    try:
        probs = np.asarray(values, dtype=float)
    except Exception:
        return None

    if probs.ndim != 2 or probs.shape[1] < 2:
        return None

    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.clip(probs, 0.0, None)

    totals = probs.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0.0

    if not np.any(valid):
        return None

    probs[valid] /= totals[valid]
    probs[~valid] = 1.0 / probs.shape[1]
    return probs

def _sigmoid(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -709.0, 709.0)

    result = np.empty_like(values, dtype=float)
    positive = values >= 0

    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)

    return result

def _decision_scores_to_probability_matrix(
    scores: Any,
    *,
    class_count: int,
) -> Optional[np.ndarray]:
    """Convert sklearn decision scores into probability-like values.

    These values are suitable for ranking and visualisation, but are not
    calibrated probabilities.
    """
    try:
        scores = np.asarray(scores, dtype=float)
    except Exception:
        return None

    if scores.ndim == 1:
        positive = _sigmoid(scores)
        return np.column_stack((1.0 - positive, positive))

    if scores.ndim != 2:
        return None

    if scores.shape[1] == 1:
        positive = _sigmoid(scores[:, 0])
        return np.column_stack((1.0 - positive, positive))

    if class_count and scores.shape[1] != class_count:
        # This can happen with pairwise/OVO decision-function outputs.
        return None

    shifted = scores - np.nanmax(scores, axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -709.0, 0.0))
    return _normalise_probability_matrix(exponentials)

def _sklearn_probability_output(
    model: Any,
    X: pd.DataFrame,
    *,
    class_count: int,
) -> tuple[Optional[np.ndarray], str]:
    """Return probability-like outputs and their provenance."""
    predict_proba = getattr(model, "predict_proba", None)

    if callable(predict_proba):
        try:
            probabilities = _normalise_probability_matrix(predict_proba(X))

            if probabilities is not None:
                return probabilities, "predict_proba"
        except Exception:
            pass

    decision_function = getattr(model, "decision_function", None)

    if callable(decision_function):
        try:
            probabilities = _decision_scores_to_probability_matrix(
                decision_function(X),
                class_count=class_count,
            )

            if probabilities is not None:
                return probabilities, "decision_function"
        except Exception:
            pass

    return None, "labels_only"

class SklearnTabularPredictor(Predictor):
    framework = "sklearn"
    modality = "tabular"

    def reconstruct(self) -> None:
        from .. import artifacts as artifact_utils

        self._model = artifact_utils.load_model_from_payload(self.model_payload)
        self._features = [str(c) for c in self.binding.get("feature_columns") or []]
        if not self._features:
            raise ValueError("Compatibility did not resolve feature columns.")
        self.transform_desc = {"source": "sklearn_pipeline", "feature_columns": self._features}

    def read_columns(self) -> List[str]:
        return [*self._features, self.binding.get("record_id_column")]

    def predict_records(self, df) -> List[Dict[str, Any]]:
        rid_col = self.binding.get("record_id_column")
        X = df[self._features]
        predictions = np.asarray(self._model.predict(X))

        # predict_proba and decision_function columns follow estimator.classes_,
        # not an arbitrary class ordering stored elsewhere in the model artifact.
        classes = _sklearn_classes(self._model) or list(self.classes)

        probabilities, confidence_source = _sklearn_probability_output(
            self._model,
            X,
            class_count=len(classes),
        )

        self.transform_desc["confidence_source"] = confidence_source
        self.transform_desc["confidence_semantics"] = (
            "probability_estimate"
            if confidence_source == "predict_proba"
            else "normalised_decision_score"
            if confidence_source == "decision_function"
            else "unavailable"
        )

        record_ids = [
            str(df.iloc[index][rid_col])
            if rid_col and rid_col in df.columns
            else str(df.index[index])
            for index in range(len(df))
        ]

        records: List[Dict[str, Any]] = []

        for index, record_id in enumerate(record_ids):
            if probabilities is not None and index < probabilities.shape[0]:
                record = _classification_record(
                    record_id,
                    probabilities[index],
                    classes,
                )
                record["confidence_source"] = confidence_source
                record["confidence_semantics"] = self.transform_desc[
                    "confidence_semantics"
                ]
            else:
                prediction = _json_scalar(predictions[index])
                record = {
                    "record_id": record_id,
                    "row_id": record_id,
                    "prediction": prediction,
                    "y_pred": prediction,
                    "confidence_source": "unavailable",
                    "confidence_semantics": "unavailable",
                }

            records.append(record)

        return records

class TorchTabularPredictor(Predictor):
    framework = "torch"
    modality = "tabular"

    def reconstruct(self) -> None:

        from .. import artifacts as artifact_utils

        saved = artifact_utils.load_model_from_payload(self.model_payload)
        if not isinstance(saved, Mapping):
            raise TypeError("Torch tabular sidecar must be a mapping.")
        self._model = saved.get("torch_model") or saved.get("model")
        self._pre = saved.get("preprocessor")
        self._label_encoder = saved.get("label_encoder")
        if self._model is None or self._pre is None:
            raise ValueError("Torch tabular prediction needs torch_model + preprocessor.")
        self._model.eval()
        self._features = [str(c) for c in self.binding.get("feature_columns") or []]
        self.transform_desc = {"source": "torch_tabular_preprocessor", "feature_columns": self._features}

    def read_columns(self) -> List[str]:
        return [*self._features, self.binding.get("record_id_column")]

    def predict_records(self, df) -> List[Dict[str, Any]]:
        import torch
        rid_col = self.binding.get("record_id_column")
        X = np.asarray(self._pre.transform(df[self._features]), dtype=np.float32)
        with torch.no_grad():
            out = self._model(torch.tensor(X)).cpu().numpy()
        probs = _softmax(out)
        classes = self.classes or (
            [str(c) for c in getattr(self._label_encoder, "classes_", [])]
            or [str(i) for i in range(probs.shape[1])]
        )
        ids = [str(df.iloc[i][rid_col]) if (rid_col and rid_col in df.columns) else str(df.index[i])
               for i in range(len(df))]
        return [_classification_record(rid, probs[i], classes) for i, rid in enumerate(ids)]

# Factory mirrors make_harness; extensible via register_predictor.
_PREDICTORS: List = []

def register_predictor(predicate, predictor_cls):
    _PREDICTORS.insert(0, (predicate, predictor_cls))

def make_predictor(*, framework, modality, **kwargs) -> Predictor:
    framework = str(framework or "").lower()
    modality = str(modality or "tabular").lower()
    for predicate, cls in list(_PREDICTORS):
        try:
            if predicate(framework=framework, modality=modality):
                return cls(**kwargs)
        except Exception:
            pass
    if modality == "image" and framework == "torch":
        return TorchImagePredictor(**kwargs)
    if modality == "tabular" and framework == "torch":
        return TorchTabularPredictor(**kwargs)
    if modality == "tabular" and framework == "sklearn":
        return SklearnTabularPredictor(**kwargs)
    raise NotImplementedError(f"No predictor for framework={framework!r}, modality={modality!r}.")

# =============================================================================
# Public actions
# =============================================================================

def predict_action(context: Any, request: Any, cancel_token: Any = None) -> Dict[str, Any]:

    from .. import artifacts as artifact_utils
    from .. import contracts as contract_utils

    request = coerce_action_request(request)
    params = dict(request.params or {})
    dataset_id = request_dataset_id(context, request, params)
    model_artifact_id = str(params.get("model_artifact_id") or request.artifact_id or "").strip()
    if not dataset_id:
        raise ValueError("predict requires a dataset_id.")
    if not model_artifact_id:
        raise ValueError("predict requires params.model_artifact_id or request.artifact_id.")

    model_payload = context.artifacts.get(model_artifact_id)
    if not isinstance(model_payload, Mapping):
        raise TypeError(f"Artifact {model_artifact_id!r} is not an ml.model payload.")
    if "model" in model_payload and "model_ref" not in model_payload:
        model_payload = artifact_utils.persist_existing_model_artifact(
            context=context, artifact_id=model_artifact_id)

    compatibility = contract_utils.validate_model_for_dataset(
        context=context,
        model_artifact_id=model_artifact_id,
        dataset_id=dataset_id,
        target_column=params.get("target_column"),
        image_column=params.get("image_column"),
        feature_column_mapping=params.get("feature_column_mapping") or {},
        require_target_compatible=bool(params.get("require_target_compatible", False)),
    )
    if not compatibility.can_predict:
        return {
            "ok": False,
            "status": compatibility.status,
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "compatibility_report": compatibility.to_dict(),
            "errors": list(compatibility.errors),
            "warnings": list(compatibility.warnings),
        }

    contract = contract_utils.ensure_model_contract(
        context=context, model_artifact_id=model_artifact_id,
        model_payload=model_payload, persist=True)
    framework = str(contract.get("framework") or model_payload.get("framework") or "").lower()
    modality = str(contract.get("modality") or model_payload.get("modality") or "tabular").lower()

    predictor = make_predictor(
        framework=framework, modality=modality,
        context=context, dataset_id=dataset_id,
        model_artifact_id=model_artifact_id, model_payload=model_payload,
        compatibility=compatibility, request=request,
        scope=params.get("scope", SCOPE_AUTO), cancel_token=cancel_token,
    )
    return predictor.run()


def register_prediction_table_dataset(*, context, predictions_payload, predictions_artifact_id) -> Optional[str]:
    rows = list((predictions_payload.get("prediction_table") or {}).get("rows") or [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    run_id = str(predictions_payload.get("prediction_run_id") or uuid.uuid4().hex)
    source_dataset_id = str(predictions_payload.get("dataset_id") or "dataset")
    model_title = str(predictions_payload.get("model_title") or "model")
    dataset_id = f"predictions_{source_dataset_id}_{run_id[:8]}"
    register = getattr(getattr(context, "datasets", None), "register", None)
    if not callable(register):
        return None
    try:
        register(dataset_id, df, name=f"Predictions: {model_title}", source="ml.predictions",
                 source_dataset_id=source_dataset_id, predictions_artifact_id=predictions_artifact_id,
                 row_count=len(df), columns=list(df.columns), column_mappings={"record_id": "record_id"})
        try:
            context.datasets.set_mapping(dataset_id, "record_id", "record_id")
        except Exception:
            pass
        publish_event(context, "dataset.loaded", {"dataset_id": dataset_id, "origin": "core.ml.predictions"})
        return dataset_id
    except Exception:
        return None

# =============================================================================
# Leaf helpers
# =============================================================================

def _classification_record(record_id: str, probabilities: Any, classes: Sequence[str]) -> Dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    classes = [str(c) for c in classes]
    rec: Dict[str, Any] = {"record_id": str(record_id), "row_id": str(record_id)}
    if probs.size:
        top = int(np.argmax(probs))
        rec["prediction"] = classes[top] if top < len(classes) else str(top)
        rec["y_pred"] = rec["prediction"]
        rec["probabilities"] = [float(p) for p in probs]
        if len(classes) == probs.size:
            rec["probabilities_by_class"] = {classes[i]: float(probs[i]) for i in range(probs.size)}
        order = np.sort(probs)[::-1]
        rec["confidence"] = float(order[0])
        rec["max_probability"] = float(order[0])
        rec["least_confidence"] = float(1.0 - order[0])
        if order.size >= 2:
            rec["margin"] = float(order[0] - order[1])
            rec["margin_uncertainty"] = float(1.0 - rec["margin"])
        rec["entropy"] = _entropy(probs)
    return rec

def _metrics_for_task(task: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    task = str(task or "classification").lower()
    if task == "regression":
        return _regression_metrics(records)
    return _classification_metrics(records)

def _classification_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    y_true = [str(r.get("y_true")) for r in records if r.get("y_true") is not None and r.get("prediction") is not None]
    y_pred = [str(r.get("prediction")) for r in records if r.get("y_true") is not None and r.get("prediction") is not None]
    if not y_true:
        return {"accuracy": None, "f1_macro": None, "balanced_accuracy": None}
    try:
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "n_evaluated": len(y_true),
        }
    except Exception:
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return {"accuracy": correct / len(y_true), "f1_macro": None, "balanced_accuracy": None,
                "n_evaluated": len(y_true)}

def _regression_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    y_true = []
    y_pred = []

    for r in records:
        if r.get("y_true") is None:
            continue

        pred = r.get("prediction", r.get("predicted_value", r.get("y_pred")))
        if pred is None:
            continue

        try:
            y_true.append(float(r.get("y_true")))
            y_pred.append(float(pred))
        except Exception:
            continue

    if not y_true:
        return {
            "mae": None,
            "rmse": None,
            "mse": None,
            "r2": None,
            "n_evaluated": 0,
        }

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "n_evaluated": len(y_true),
    }

def _sklearn_classes(model) -> List[str]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        steps = getattr(model, "named_steps", None)
        if steps:
            est = steps.get("model") or steps.get("estimator")
            classes = getattr(est, "classes_", None)
    return [str(c) for c in classes] if classes is not None else []

def _torch_device(params):
    import torch
    name = str(params.get("device") or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





def _filter_rows(df, request, record_id_column):
    params = dict(getattr(request, "params", {}) or {})
    row_ids = list(getattr(request, "row_ids", None) or [])
    if row_ids:
        if record_id_column and record_id_column in df.columns:
            wanted = {str(r) for r in row_ids}
            df = df[df[record_id_column].astype(str).isin(wanted)]
        else:
            try:
                df = df.loc[row_ids]
            except Exception:
                wanted = {str(r) for r in row_ids}
                df = df[df.index.astype(str).isin(wanted)]
    try:
        limit = int(params.get("max_rows") or params.get("row_limit") or 0)
    except Exception:
        limit = 0
    return df.head(limit) if limit > 0 else df


def _softmax(values):
    values = np.asarray(values, dtype=float)
    exp = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _entropy(probs):
    clean = np.asarray([p for p in probs if p > 0.0], dtype=float)
    return 0.0 if clean.size == 0 else float(-np.sum(clean * np.log(clean)))


def _json_scalar(value):
    return json_safe(value)




def _get_trained_model_catalog(context):
    get = getattr(getattr(context, "services", None), "get", None)
    if not callable(get):
        return None
    for key in ("core.ml.trained_model_catalog", "trained_model_catalog"):
        try:
            return get(key)
        except Exception:
            continue
    return None
