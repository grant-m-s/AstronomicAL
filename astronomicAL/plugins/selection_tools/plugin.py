from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest
from astronomicAL.platform.plugins.specs import InputSpec
from astronomicAL.platform.plugins.specs import ActionResult, EventResult

from .panel import SelectionSetPanel, materialise_active_selection_as_dataset

PLUGIN_ID = "core.selection_tools"
PANEL_ID = f"{PLUGIN_ID}.selection_set"

manifest = PluginManifest(
    id=PLUGIN_ID,
    name="Selection Tools",
    version="0.1.0",
    description=(
        "Generic selection inspection tools for focused rows and active "
        "multi-row selection sets."
    ),
    capabilities=["panel", "selection", "datasets", "events", "actions"],
    tags=["core", "selection", "workflow", "review", "dataset"],
)


def register(api) -> None:
    api.register_panel(
        id="selection_set",
        title="Selection Set",
        factory=create_selection_set_panel,
        description=(
            "Inspect the active selection set, preview selected rows, clear the "
            "set, move focus through selected row IDs, and create a derived "
            "dataset from the active selection."
        ),
        category="Multi-Selection",
        icon="list-checks",
        tags=["selection", "focus", "dataset", "review"],
        optional_mappings=[
            {
                "semantic_name": "record_id",
                "display_name": "ID column",
                "description": (
                    "Optional. Used to match platform selection row IDs back to "
                    "dataframe rows. If unmapped, Selection Tools falls back to "
                    "the dataframe index."
                ),
                "aliases": [
                    "source_id",
                    "sourceid",
                    "object_id",
                    "objid",
                    "id",
                    "ID",
                    "row_id",
                ],
                "allow_index": True,
            }
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
    )

    api.register_action(
        id="selection_to_dataset",
        title="Create dataset from active selection",
        handler=selection_to_dataset_action,
        inputs=InputSpec(
            dataset=True,
            selection="optional",
            columns="none",
            optional_mappings=[
                {
                    "semantic_name": "record_id",
                    "display_name": "ID column",
                    "description": (
                        "Optional. Used to match selection row IDs to dataframe rows. "
                        "If unmapped, the dataframe index is used."
                    ),
                    "aliases": [
                        "source_id",
                        "sourceid",
                        "object_id",
                        "objid",
                        "id",
                        "ID",
                        "row_id",
                    ],
                    "allow_index": True,
                }
            ],
        ),
        outputs=["dataset.loaded", "dataset.active.changed", "selection.dataset.created"],
        params_schema={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "set_active": {"type": "boolean", "default": True},
            },
            "required": ["dataset_name"],
            "additionalProperties": False,
        },
        run_in_job=False,
        description="Register the active selection set as a new derived dataset.",
        category="Selection",
        tags=["selection", "dataset", "subset"],
    )


def create_selection_set_panel(context, data=None, **kwargs):
    controller = SelectionSetPanel(context=context, data=data)
    return controller.view, controller

def selection_to_dataset_action(context, request, **_kwargs) -> ActionResult:
    dataset_name = str(request.params.get("dataset_name", "")).strip()
    set_active = bool(request.params.get("set_active", True))

    if not dataset_name:
        raise ValueError("Please provide a dataset name.")

    result = materialise_active_selection_as_dataset(
        context,
        dataset_name=dataset_name,
        set_active=set_active,
        publish_events=False,
    )

    events = [
        EventResult(
            "dataset.loaded",
            {
                "dataset_id": result["dataset_id"],
                "name": result["name"],
                "rows": result["rows"],
                "derived_from": result["derived_from"],
                "selection_set_id": result["selection_set_id"],
                "origin": PLUGIN_ID,
            },
        ),
        EventResult(
            "selection.dataset.created",
            {
                "dataset_id": result["dataset_id"],
                "derived_from": result["derived_from"],
                "selection_set_id": result["selection_set_id"],
                "rows": result["rows"],
                "origin": PLUGIN_ID,
            },
        ),
    ]

    if set_active:
        events.append(
            EventResult(
                "dataset.active.changed",
                {
                    "dataset_id": result["dataset_id"],
                    "previous_dataset_id": result["derived_from"],
                    "origin": PLUGIN_ID,
                },
            )
        )

    return ActionResult(value=result, events=events)