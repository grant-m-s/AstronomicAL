from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence
import re

from .errors import PluginRegistrationError
from .specs import (
    ActionRegistration,
    ArtifactViewerRegistration,
    InputSpec,
    OutputSpec,
    PanelRegistration,
    ServiceRegistration,
    WorkflowRegistration,
)

from astronomicAL.platform.mapping_requirements import MappingRequirementLike


_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


class PluginAPI:
    """Author-facing registration object passed to ``plugin.register(api)``.

    Plugin authors should only need this object plus the runtime ``context`` they
    receive in handlers/factories.
    """

    def __init__(self, manager: Any, plugin_id: str) -> None:
        self.manager = manager
        self.plugin_id = plugin_id

    def register_panel(
        self,
        *,
        id: str,
        title: str,
        factory: Callable[..., Any],
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        required_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        optional_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        uses_services: Optional[Sequence[str]] = None,
        produces: Optional[Sequence[str]] = None,
        default_layout: Optional[Dict[str, Any]] = None,
        default_open_kwargs: Optional[Dict[str, Any]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
        state_version: int = 1,
        persist_layout: bool = True,
        persist_state: bool = True,
        restore_policy: str = "best_effort",
    ) -> None:
        canonical_id = self._canonical_id(id)

        self.manager._register_panel(
            PanelRegistration(
                plugin_id=self.plugin_id,
                id=canonical_id,
                title=title,
                description=description,
                category=category,
                icon=icon,
                tags=list(tags or []),
                factory=factory,
                required_mappings=list(required_mappings or []),
                optional_mappings=list(optional_mappings or []),
                uses_services=list(uses_services or []),
                produces=list(produces or []),
                default_layout=default_layout,
                default_open_kwargs=dict(default_open_kwargs or {}),
                requires=list(requires or []),
                optional_requires=list(optional_requires or []),
                state_version=int(state_version),
                persist_layout=bool(persist_layout),
                persist_state=bool(persist_state),
                restore_policy=str(restore_policy),
            )
        )

    def register_action(
        self,
        *,
        id: str,
        title: str,
        handler: Callable[..., Any],
        inputs: Optional[InputSpec | Dict[str, Any]] = None,
        outputs: Optional[Sequence[OutputSpec | Dict[str, Any] | str]] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        settings_schema: Optional[Dict[str, Any]] = None,
        run_in_job: bool = True,
        key_fn: Optional[Callable[..., str]] = None,
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> None:
        canonical_id = self._canonical_id(id)
        input_spec = inputs if isinstance(inputs, InputSpec) else InputSpec.from_dict(inputs or {})
        output_specs = [OutputSpec.from_any(o) for o in (outputs or [])]
        self.manager._register_action(
            ActionRegistration(
                plugin_id=self.plugin_id,
                id=canonical_id,
                title=title,
                description=description,
                category=category,
                icon=icon,
                tags=list(tags or []),
                handler=handler,
                inputs=input_spec,
                outputs=output_specs,
                params_schema=params_schema or {},
                settings_schema=settings_schema or {},
                run_in_job=run_in_job,
                key_fn=key_fn,
                requires=list(requires or []),
                optional_requires=list(optional_requires or []),
            )
        )

    def register_dataframe_action(
        self,
        *,
        id: str,
        title: str,
        handler: Callable[..., Any],
        output_type: Optional[str] = None,
        selection: str = "optional",
        numeric_columns: str = "none",
        columns: str = "optional",
        required_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        optional_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        settings_schema: Optional[Dict[str, Any]] = None,
        run_in_job: bool = True,
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> None:
        """Register a dataframe-oriented action with low authoring friction.

        The platform resolves the dataframe, selected rows, columns, params, and
        cancellation token, then calls the handler with only the arguments it accepts.
        """

        outputs = [output_type] if output_type else []

        def _adapter(context, request, cancel_token=None):
            dataset_id = request.dataset_id
            df = context.datasets.get_df(dataset_id) if dataset_id else context.datasets.get_df()

            if request.row_ids:
                id_column = _resolve_id_column(context, dataset_id, df)
                if id_column is not None:
                    work_df = df[df[id_column].isin(request.row_ids)]
                else:
                    work_df = df.loc[request.row_ids]
            else:
                work_df = df

            result = self.manager._call_with_supported_args(
                handler,
                context=context,
                request=request,
                df=work_df,
                dataset_id=dataset_id,
                row_ids=request.row_ids,
                columns=request.columns,
                params=request.params,
                cancel_token=cancel_token,
            )

            if output_type and result is not None and not self.manager._is_action_result_like(result):
                from .specs import ArtifactResult

                return ArtifactResult(
                    type=output_type,
                    payload=result,
                    dataset_id=dataset_id,
                    row_ids=request.row_ids,
                    params=dict(request.params),
                )

            return result

        self.register_action(
            id=id,
            title=title,
            handler=_adapter,
            inputs=InputSpec(
                dataset=True,
                selection=selection,
                numeric_columns=numeric_columns,
                columns=columns,
                required_mappings=list(required_mappings or []),
                optional_mappings=list(optional_mappings or []),
            ),
            outputs=outputs,
            params_schema=params_schema,
            settings_schema=settings_schema,
            run_in_job=run_in_job,
            description=description,
            category=category,
            icon=icon,
            tags=tags,
            requires=requires,
            optional_requires=optional_requires,
        )

    def action(
        self,
        *,
        id: str,
        title: str,
        inputs: Optional[InputSpec | Dict[str, Any]] = None,
        outputs: Optional[Sequence[OutputSpec | Dict[str, Any] | str]] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        settings_schema: Optional[Dict[str, Any]] = None,
        run_in_job: bool = True,
        key_fn: Optional[Callable[..., str]] = None,
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register_action(
                id=id,
                title=title,
                handler=func,
                inputs=inputs,
                outputs=outputs,
                params_schema=params_schema,
                settings_schema=settings_schema,
                run_in_job=run_in_job,
                key_fn=key_fn,
                description=description,
                category=category,
                icon=icon,
                tags=tags,
                requires=requires,
                optional_requires=optional_requires,
            )
            return func

        return decorator

    def dataframe_action(
        self,
        *,
        id: str,
        title: str,
        output_type: Optional[str] = None,
        selection: str = "optional",
        numeric_columns: str = "none",
        columns: str = "optional",
        required_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        optional_mappings: Optional[Sequence[MappingRequirementLike]] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        settings_schema: Optional[Dict[str, Any]] = None,
        run_in_job: bool = True,
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register_dataframe_action(
                id=id,
                title=title,
                handler=func,
                output_type=output_type,
                selection=selection,
                numeric_columns=numeric_columns,
                columns=columns,
                required_mappings=required_mappings,
                optional_mappings=optional_mappings,
                params_schema=params_schema,
                settings_schema=settings_schema,
                run_in_job=run_in_job,
                description=description,
                category=category,
                icon=icon,
                tags=tags,
                requires=requires,
                optional_requires=optional_requires,
            )
            return func

        return decorator

    def register_workflow(
        self,
        *,
        id: str,
        title: str,
        builder: Callable[..., Any],
        settings_schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        category: Optional[str] = None,
        icon: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> None:
        canonical_id = self._canonical_id(id)
        self.manager._register_workflow(
            WorkflowRegistration(
                plugin_id=self.plugin_id,
                id=canonical_id,
                title=title,
                description=description,
                category=category,
                icon=icon,
                tags=list(tags or []),
                builder=builder,
                settings_schema=settings_schema or {},
                requires=list(requires or []),
                optional_requires=list(optional_requires or []),
            )
        )

    def register_service(
        self,
        *,
        key: str,
        factory: Callable[..., Any],
        lazy: bool = True,
        replace: bool = False,
        description: str = "",
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> None:
        if not key:
            raise PluginRegistrationError("Service key cannot be empty.")
        self.manager._register_service(
            ServiceRegistration(
                plugin_id=self.plugin_id,
                key=self._canonical_service_key(key),
                factory=factory,
                lazy=lazy,
                replace=replace,
                description=description,
                requires=list(requires or []),
                optional_requires=list(optional_requires or []),
            )
        )

    def register_artifact_viewer(
        self,
        *,
        artifact_type: str,
        viewer_factory: Callable[..., Any],
        id: Optional[str] = None,
        title: Optional[str] = None,
        description: str = "",
        priority: int = 100,
        default: bool = False,
        requires: Optional[Sequence[str]] = None,
        optional_requires: Optional[Sequence[str]] = None,
    ) -> None:
        if not artifact_type:
            raise PluginRegistrationError("artifact_type cannot be empty.")
        self._validate_artifact_type(artifact_type)
        viewer_id = self._canonical_id(id) if id else None
        self.manager._register_artifact_viewer(
            ArtifactViewerRegistration(
                plugin_id=self.plugin_id,
                artifact_type=artifact_type,
                viewer_factory=viewer_factory,
                id=viewer_id,
                title=title,
                description=description,
                priority=priority,
                default=default,
                requires=list(requires or []),
                optional_requires=list(optional_requires or []),
            )
        )

    def register_settings_schema(self, schema: Dict[str, Any]) -> None:
        self.manager._register_settings_schema(self.plugin_id, schema)

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.manager.get_plugin_setting(self.plugin_id, key, default)

    def set_setting(self, key: str, value: Any) -> None:
        self.manager.set_plugin_setting(self.plugin_id, key, value)

    def _canonical_id(self, id: str) -> str:
        self._validate_id(id)
        if id == self.plugin_id or id.startswith(f"{self.plugin_id}."):
            return id
        return f"{self.plugin_id}.{id}"

    def _canonical_service_key(self, key: str) -> str:
        self._validate_id(key)
        if key == self.plugin_id or key.startswith(f"{self.plugin_id}."):
            return key
        return f"{self.plugin_id}.{key}"

    @staticmethod
    def _validate_id(id: str) -> None:
        if not id or not isinstance(id, str):
            raise PluginRegistrationError("Registration id must be a non-empty string.")
        if not _ID_RE.match(id):
            raise PluginRegistrationError(
                "Registration ids may only contain letters, numbers, '.', '_' or '-', "
                f"and must start with a letter or number. Got {id!r}."
            )

    @staticmethod
    def _validate_artifact_type(artifact_type: str) -> None:
        if not _ID_RE.match(artifact_type):
            raise PluginRegistrationError(
                "Artifact types may only contain letters, numbers, '.', '_' or '-', "
                f"and must start with a letter or number. Got {artifact_type!r}."
            )


def _resolve_id_column(context: Any, dataset_id: Optional[str], df: Any) -> Optional[str]:
    if context is not None and dataset_id is not None:
        datasets = getattr(context, "datasets", None)
        if datasets is not None:
            for method_name in ("get_mapping", "mapping", "get_column_mapping"):
                method = getattr(datasets, method_name, None)
                if callable(method):
                    for mapping_name in ("id", "row_id"):
                        try:
                            value = method(dataset_id, mapping_name)
                        except Exception:
                            value = None
                        if value and hasattr(df, "columns") and value in df.columns:
                            return value

    for candidate in ("id", "ID", "source_id", "object_id", "row_id"):
        if hasattr(df, "columns") and candidate in df.columns:
            return candidate

    return None