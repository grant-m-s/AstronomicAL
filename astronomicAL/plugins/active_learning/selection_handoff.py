from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from . import acquisition


@dataclass(frozen=True)
class ReviewSelectionResult:
    """Outcome of handing an Active Learning review batch to the platform.

    Active Learning owns the pool and batch artifacts. The platform selection
    manager owns the user-facing selection set and focused record. Because the
    source dataset and its pool partition share physical record IDs, a review
    batch may be represented against whichever of those two datasets is
    currently active without changing the active dataset itself.
    """

    applied: bool
    dataset_id: Optional[str]
    focused_row_id: Optional[str]
    row_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def session_source_dataset_id(session: Mapping[str, Any]) -> str:
    """Return the original pre-split dataset owned by an AL session."""

    direct = str(
        session.get("source_dataset_id")
        or session.get("dataset_id")
        or ""
    ).strip()
    if direct:
        return direct

    contract = session.get("contract")
    if isinstance(contract, Mapping):
        session_split = contract.get("session_split")
        if isinstance(session_split, Mapping):
            return str(session_split.get("source_dataset_id") or "").strip()
    return ""


def resolve_review_dataset_id(
    context: Any,
    *,
    source_dataset_id: str,
    pool_dataset_id: str,
) -> Optional[str]:
    """Resolve the active dataset that may safely represent a review batch.

    Review row IDs are valid in both the original source dataset and its pool
    partition. They are not assumed to be valid in validation, test, training,
    prediction, or unrelated datasets, so those active datasets are left
    untouched.
    """

    datasets = getattr(context, "datasets", None)
    active_id = getattr(datasets, "active_id", None)
    if not callable(active_id):
        return None
    try:
        active_dataset_id = str(active_id() or "").strip()
    except RuntimeError:
        return None

    allowed = {
        str(value).strip()
        for value in (source_dataset_id, pool_dataset_id)
        if str(value or "").strip()
    }
    return active_dataset_id if active_dataset_id in allowed else None


def apply_ranked_review_selection(
    context: Any,
    *,
    source_dataset_id: str,
    pool_dataset_id: str,
    row_ids: Sequence[Any],
    session_artifact_id: str,
    batch_artifact_id: str,
    strategy_id: str,
    origin: str,
) -> ReviewSelectionResult:
    """Create the ranked selection set and focus its first review record.

    The AL batch remains associated with ``pool_dataset_id``. Only the platform
    selection handoff is projected onto the active source or pool dataset.
    """

    ordered_row_ids = _stable_row_ids(row_ids)
    if not ordered_row_ids:
        return ReviewSelectionResult(
            applied=False,
            dataset_id=None,
            focused_row_id=None,
            row_count=0,
            reason="empty_review_batch",
        )

    selection_dataset_id = resolve_review_dataset_id(
        context,
        source_dataset_id=source_dataset_id,
        pool_dataset_id=pool_dataset_id,
    )
    if selection_dataset_id is None:
        return ReviewSelectionResult(
            applied=False,
            dataset_id=None,
            focused_row_id=None,
            row_count=len(ordered_row_ids),
            reason="active_dataset_is_not_session_source_or_pool",
        )

    acquisition.set_ranked_selection(
        context,
        dataset_id=selection_dataset_id,
        row_ids=ordered_row_ids,
        session_artifact_id=session_artifact_id,
        batch_artifact_id=batch_artifact_id,
        strategy_id=strategy_id,
        origin=origin,
    )

    focused_row_id = ordered_row_ids[0]
    selection = getattr(context, "selection", None)
    set_focus = getattr(selection, "set_focus", None)
    if not callable(set_focus):
        raise RuntimeError(
            "Active Learning requires SelectionManager.set_focus() to focus "
            "the first review record."
        )
    set_focus(
        dataset_id=selection_dataset_id,
        row_id=focused_row_id,
        origin=origin,
    )

    return ReviewSelectionResult(
        applied=True,
        dataset_id=selection_dataset_id,
        focused_row_id=focused_row_id,
        row_count=len(ordered_row_ids),
        reason="applied_to_active_session_dataset",
    )


def disabled_review_selection(row_ids: Sequence[Any]) -> ReviewSelectionResult:
    """Return an explicit result when selection handoff was disabled."""

    ordered_row_ids = _stable_row_ids(row_ids)
    return ReviewSelectionResult(
        applied=False,
        dataset_id=None,
        focused_row_id=None,
        row_count=len(ordered_row_ids),
        reason="selection_handoff_disabled",
    )


def _stable_row_ids(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        row_id = str(value or "").strip()
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        result.append(row_id)
    return result

