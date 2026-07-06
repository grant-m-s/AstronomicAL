from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class DataContractReport:
    """Small compatibility wrapper for old AL contract callers.

    The heavy recipe-specific validation has moved behind ``core_ml_bridge``.
    This class intentionally represents only a plain report: it can be carried
    in session metadata, displayed by panels, and checked by bridge actions.
    """

    contract: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Active-learning data contract failed: " + "; ".join(self.errors))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "contract": dict(self.contract or {}),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def report_from_payload(payload: Mapping[str, Any]) -> DataContractReport:
    return DataContractReport(
        contract=dict(payload.get("contract") or payload),
        errors=[str(item) for item in (payload.get("errors") or [])],
        warnings=[str(item) for item in (payload.get("warnings") or [])],
    )


def empty_contract(*, dataset_id: str = "", recipe_id: str = "") -> DataContractReport:
    return DataContractReport(
        contract={
            "schema_version": 2,
            "dataset": {"id": str(dataset_id or "")},
            "recipe": {"id": str(recipe_id or "")},
            "bindings": {},
        }
    )
