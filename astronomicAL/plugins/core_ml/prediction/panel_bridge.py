from __future__ import annotations

import sys
from typing import Any, Mapping


_PATCH_MARKER = "_streaming_prediction_columns_bridge_v1"


def install_prediction_panel_bridge() -> bool:
    """Prevent the Predictor panel from attaching the same predictions twice.

    Streaming prediction updates the selected dataset inside the action so the
    same behaviour is available to the panel, workflows, and Active Learning.
    The older panel then tries to repeat that work from the returned artifact.
    This bridge makes the already-loaded panel treat the action result as
    authoritative and keeps its display work bounded to the prediction preview.
    """

    package_root = __package__.rsplit(".", 1)[0]
    module_name = f"{package_root}.panels.predictor"
    predictor_panel = sys.modules.get(module_name)
    if predictor_panel is None:
        return False

    controller = getattr(predictor_panel, "MLPredictPanel", None)
    if controller is None or getattr(controller, _PATCH_MARKER, False):
        return controller is not None

    original_attach = getattr(controller, "_attach_predictions", None)
    original_fetch = getattr(controller, "_fetch_table", None)
    if not callable(original_attach) or not callable(original_fetch):
        return False

    def _attach_predictions(
        self: Any,
        result: Mapping[str, Any],
        rows: list[Mapping[str, Any]],
    ) -> tuple[list[str], Any]:
        if bool(result.get("prediction_columns_attached")):
            columns = [
                str(column)
                for column in (result.get("prediction_columns") or [])
                if str(column)
            ]
            return columns, None
        return original_attach(self, result, rows)

    def _fetch_table(
        self: Any,
        artifact_id: Any,
        result: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], int, list[Any]]:
        if bool(result.get("prediction_columns_attached")):
            return (
                list(result.get("prediction_preview") or []),
                int(result.get("failed_image_row_count") or 0),
                list(result.get("failed_image_rows") or []),
            )
        return original_fetch(self, artifact_id, result)

    controller._attach_predictions = _attach_predictions
    controller._fetch_table = _fetch_table
    setattr(controller, _PATCH_MARKER, True)
    return True
