from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

REPORTABLE_PROVENANCE = frozenset({"test", "novel"})


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class ClassificationMetrics:
    confusion: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    count: int = 0
    correct: int = 0

    def update(self, y_true: Any, y_pred: Any) -> None:
        if y_true is None or y_pred is None:
            return
        truth = str(y_true)
        prediction = str(y_pred)
        self.confusion[truth][prediction] += 1
        self.count += 1
        self.correct += int(truth == prediction)

    def merge(self, other: "ClassificationMetrics") -> None:
        for truth, predicted in other.confusion.items():
            for prediction, count in predicted.items():
                self.confusion[truth][prediction] += int(count)
        self.count += int(other.count)
        self.correct += int(other.correct)

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"n_evaluated": 0}

        true_labels = sorted(self.confusion)
        labels = sorted(
            set(true_labels)
            | {
                prediction
                for predicted in self.confusion.values()
                for prediction in predicted
            }
        )
        recalls = []
        f1_values = []
        for label in labels:
            true_positive = self.confusion[label].get(label, 0)
            false_negative = sum(self.confusion[label].values()) - true_positive
            false_positive = sum(
                predicted.get(label, 0)
                for truth, predicted in self.confusion.items()
                if truth != label
            )
            recall_denominator = true_positive + false_negative
            precision_denominator = true_positive + false_positive
            recall = (
                true_positive / recall_denominator
                if recall_denominator
                else 0.0
            )
            precision = (
                true_positive / precision_denominator
                if precision_denominator
                else 0.0
            )
            if label in true_labels:
                recalls.append(recall)
            f1_values.append(
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )

        return {
            "accuracy": float(self.correct / self.count),
            "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "n_evaluated": int(self.count),
        }


@dataclass
class RegressionMetrics:
    count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    target_sum: float = 0.0
    target_squared_sum: float = 0.0

    def update(self, y_true: Any, y_pred: Any) -> None:
        truth = _finite_float(y_true)
        prediction = _finite_float(y_pred)
        if truth is None or prediction is None:
            return
        error = prediction - truth
        self.count += 1
        self.absolute_error_sum += abs(error)
        self.squared_error_sum += error * error
        self.target_sum += truth
        self.target_squared_sum += truth * truth

    def merge(self, other: "RegressionMetrics") -> None:
        self.count += int(other.count)
        self.absolute_error_sum += float(other.absolute_error_sum)
        self.squared_error_sum += float(other.squared_error_sum)
        self.target_sum += float(other.target_sum)
        self.target_squared_sum += float(other.target_squared_sum)

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"n_evaluated": 0}
        mse = self.squared_error_sum / self.count
        total_sum_squares = (
            self.target_squared_sum
            - (self.target_sum * self.target_sum / self.count)
        )
        r2 = (
            1.0 - self.squared_error_sum / total_sum_squares
            if total_sum_squares > 0.0
            else None
        )
        return {
            "mae": float(self.absolute_error_sum / self.count),
            "rmse": float(math.sqrt(mse)),
            "mse": float(mse),
            "r2": None if r2 is None else float(r2),
            "n_evaluated": int(self.count),
        }


class OnlineEvaluation:
    """Exact bounded-memory metrics grouped by row provenance."""

    def __init__(self, task: str):
        self.task = str(task or "classification").lower()
        self._by_provenance: Dict[str, Any] = {}
        self._reportable = self._new_accumulator()
        self.skipped_missing_target = 0
        self.skipped_non_reportable = 0

    def _new_accumulator(self):
        if self.task == "regression":
            return RegressionMetrics()
        return ClassificationMetrics()

    def update_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            if "y_true" not in record or record.get("y_true") is None:
                self.skipped_missing_target += 1
                continue
            provenance = str(record.get("data_provenance") or "unknown")
            accumulator = self._by_provenance.setdefault(
                provenance,
                self._new_accumulator(),
            )
            prediction = record.get(
                "prediction",
                record.get("y_pred", record.get("predicted_label")),
            )
            accumulator.update(record.get("y_true"), prediction)
            if provenance in REPORTABLE_PROVENANCE:
                self._reportable.update(record.get("y_true"), prediction)
            else:
                self.skipped_non_reportable += 1

    def finalize(
        self,
        *,
        scope: str,
        selection_metric: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if str(scope).lower() != "evaluation":
            return None
        reasons = {
            "train": "Rows the model was fit on. Metrics are not reportable as generalisation.",
            "validation": "Rows used for model selection. Metrics are excluded from headlines.",
            "test": "The model's held-out test rows.",
            "novel": "Rows outside the model's recorded split.",
            "unknown": "Split provenance could not be verified.",
        }
        by_provenance = {}
        for name, accumulator in sorted(self._by_provenance.items()):
            metrics = accumulator.finalize()
            by_provenance[name] = {
                **metrics,
                "n": int(metrics.get("n_evaluated", 0) or 0),
                "reportable": name in REPORTABLE_PROVENANCE,
                "reason": reasons.get(name, "Unknown provenance."),
            }
        headline_values = self._reportable.finalize()
        headline_count = int(headline_values.get("n_evaluated", 0) or 0)
        headline = headline_values if headline_count else None
        reportable_partitions = [
            name
            for name in ("test", "novel")
            if int(by_provenance.get(name, {}).get("n_evaluated", 0) or 0)
        ]
        metric_partition = (
            "none"
            if not reportable_partitions
            else reportable_partitions[0]
            if len(reportable_partitions) == 1
            else "mixed"
        )
        warnings = []
        if headline is None:
            warnings.append(
                "No held-out test or novel rows with valid targets were available; "
                "headline generalisation metrics were not produced."
            )
        if self.skipped_missing_target:
            warnings.append(
                f"Skipped {self.skipped_missing_target} rows without a usable target."
            )
        return {
            "schema_version": 3,
            "scope": "post_hoc_evaluation",
            "task": self.task,
            "selection_metric": selection_metric,
            "selected_on_partition": "validation",
            "headline_metrics": headline,
            "headline_metric_partition": metric_partition,
            "metrics_by_provenance": by_provenance,
            "validity": {
                "metric_partition": metric_partition,
                "selection_partition": "validation",
                "selection_touched_reported_partition": False,
                "headline_excludes_seen_rows": True,
                "rows_excluded_as_seen": sum(
                    int(by_provenance.get(name, {}).get("n_evaluated", 0) or 0)
                    for name in ("train", "validation")
                ),
                "rows_unverifiable": int(
                    by_provenance.get("unknown", {}).get("n_evaluated", 0) or 0
                ),
            },
            "skipped_missing_target": int(self.skipped_missing_target),
            "skipped_non_reportable": int(self.skipped_non_reportable),
            "warnings": warnings,
        }
