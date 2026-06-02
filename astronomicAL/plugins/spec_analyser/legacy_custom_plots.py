from __future__ import annotations

import holoviews as hv

import numpy as np
import os
import html
import pandas as pd
import panel as pn
import json
import param
import re
import uuid
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import concurrent.futures 
from panel.io import save
from bokeh.models import Legend, LinearAxis, NormalHead, Range1d
from astronomicAL.utils.optimise import matches_type
from astronomicAL.utils.debug import boot_print


import uuid
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Iterable

try:
    from astronomicAL.platform.events import Subscription
except Exception:  # pragma: no cover
    Subscription = Any  # type: ignore

@dataclass
class _ManagedJob:
    key: str
    handle: Any  # JobHandle from JobManager

class _PluginPanelAdapter:
    """Adapt a plugin panel to the legacy custom-plot interface.

    The legacy custom-plot shell already owns the close button. Plugin panels
    should not render it. However, legacy panels reserve vertical space for that
    shell button, so the adapter adds the same top clearance before the plugin
    view starts.
    """

    LEGACY_CLOSE_CLEARANCE_PX = 56

    def __init__(self, view, controller=None):
        self.view = view
        self.controller = controller
        self._panel = None

    def panel(self):
        if self._panel is None:
            self._panel = pn.Column(
                pn.Spacer(
                    height=self.LEGACY_CLOSE_CLEARANCE_PX,
                    min_height=self.LEGACY_CLOSE_CLEARANCE_PX,
                    max_height=self.LEGACY_CLOSE_CLEARANCE_PX,
                    sizing_mode="stretch_width",
                ),
                self.view,
                sizing_mode="stretch_both",
                margin=(0, 0, 0, 0),
                styles={
                    "overflow": "hidden",
                },
            )

        return self._panel

    def dispose(self):
        controller = self.controller

        if controller is not None and hasattr(controller, "dispose"):
            controller.dispose()
            return

        view = self.view
        if view is not None and hasattr(view, "dispose"):
            view.dispose()


def _plugin_panel_factory(panel_id):
    """Return a legacy custom-plot factory backed by PluginManager."""

    def _factory(data, close_button=None, context=None):
        if context is None or getattr(context, "plugins", None) is None:
            return _PluginPanelAdapter(
                pn.pane.Markdown(
                    "### Plugin manager is not available\n\n"
                    f"Could not create plugin panel `{panel_id}`."
                )
            )

        try:
            view, controller = context.plugins.create_panel(
                panel_id,
                context,
                data=data,
            )
        except KeyError:
            return _PluginPanelAdapter(
                pn.Column(
                    pn.pane.Markdown(
                        "## ⚠️ Plugin panel unavailable\n\n"
                        f"The plugin panel `{panel_id}` is no longer registered. "
                        "It may belong to a plugin that has been disabled."
                    ),
                    sizing_mode="stretch_both",
                    margin=(8, 12, 8, 12),
                )
            )
        except Exception as exc:
            return _PluginPanelAdapter(
                pn.Column(
                    pn.pane.Markdown(
                        "## ❌ Plugin panel failed to load\n\n"
                        f"Panel id: `{panel_id}`\n\n"
                        f"Error: `{exc}`"
                    ),
                    sizing_mode="stretch_both",
                    margin=(8, 12, 8, 12),
                )
            )

        return _PluginPanelAdapter(view, controller)

    return _factory


def get_customplot_dict(context=None):
    boot_print("custom_plots.get_customplot_dict: start")
    boot_print(f"custom_plots.get_customplot_dict: context_present={context is not None}")
    boot_print(
        "custom_plots.get_customplot_dict: plugins_present="
        f"{getattr(context, 'plugins', None) is not None if context is not None else False}"
    )
    plot_dict = {
        "spec_analyser": lambda data, close_button, context: SpecAnalyser(
            data,
            close_button,
            extra_features=[],
            context=context,
            require_settings=False,
        )
    }

    boot_print(
        "custom_plots.get_customplot_dict: legacy keys="
        f"{list(plot_dict.keys())}"
    )

    manager = getattr(context, "plugins", None)
    boot_print("custom_plots.get_customplot_dict: plugin panels found:")
    for registration in manager.list_panels():
        boot_print(
            f"  plugin panel title={registration.title!r} "
            f"id={registration.id!r} plugin_id={registration.plugin_id!r}"
        )
        plot_dict.setdefault(
            registration.title,
            _plugin_panel_factory(registration.id),
        )

    boot_print(
        "custom_plots.get_customplot_dict: final keys="
        f"{list(plot_dict.keys())}"
    )

    return plot_dict

class CustomPlotClass(param.Parameterized):

    available_stages = ["columns_selection", "plot"]

    stage = param.ObjectSelector(default="columns_selection", objects=available_stages)

    def __init__(
        self,
        data,
        close_button,
        extra_features,
        panel_name="custom_plot",
        ready_stage="plot",
        context=None,
        require_settings=True,
        **params
    ):
        super().__init__(**params)

        self.df = data
        self.extra_features = extra_features
        self.close_button = close_button
        self.context = context

        self.panel_name = panel_name
        self.panel_id: str = uuid.uuid4().hex

        # Convenience handles (None if context not provided)
        self.events = getattr(context, "events", None)
        self.jobs = getattr(context, "jobs", None)
        self.artifacts = getattr(context, "artifacts", None)
        self.datasets = getattr(context, "datasets", None)
        self.workspace = getattr(context, "workspace", None)
        self.config = getattr(context, "config", None)

        # Lifecycle tracking
        self._subscriptions: List[Subscription] = []
        self._jobs: List[_ManagedJob] = []
        self._bokeh_on_change: list[tuple[Any, str, Callable]] = []
        self._param_watchers = []

        self._disposed = False

        # Standard runtime subscription / refresh orchestration
        self._dataset_runtime_subscriptions_initialised = False
        self._selection_runtime_subscriptions_initialised = False

        self._refresh_pending = False
        self._refresh_pending_payload = None
        self._refresh_reasons: set[str] = set()

        self._refresh_inflight_signature = None
        self._last_completed_signature = None
        self._initial_refresh_requested = False

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if self.extra_features:
            self._get_unknown_columns(columns_needed=self.extra_features)
            self._change_state_if_unknown_columns(ready_stage=ready_stage)
        else:
            self.stage = ready_stage

        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.message_pane = pn.pane.Markdown("## Loading...", sizing_mode="stretch_width", height=80)

        self.plot_settings_button = pn.widgets.Button(
            name="Open Settings",
            button_type="primary",
            max_height=40,
            max_width=100,
            sizing_mode="stretch_width",
        )
        self.plot_settings_button.on_click(self._toggle_settings_panel)
        self.plot_settings_panel = pn.Column(visible=False)

        self.require_settings = require_settings

        if self.close_button is not None:
            try:
                self.close_button.on_click(lambda _e: self.dispose())
            except Exception as e:
                print("CustomPlotClass.dispose() Errored", e)
    # -----------------------
    # Jobs
    # -----------------------
    def submit_job(
        self,
        fn: Callable[..., Any],
        *,
        title: str = "Job",
        key: Optional[str] = None,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        track: bool = True,
        **kwargs: Any,
    ) -> Any:
        if key is None:
            key = f"{self.panel_id}:{title}"

        if self.jobs is None:
            try:
                res = fn(cancel_token=None, **kwargs)
                if on_done:
                    on_done(res)
                return res
            except BaseException as e:
                if on_error:
                    on_error(e)
                    return None
                raise

        handle = self.jobs.submit(
            fn,
            title=title,
            key=key,
            on_done=on_done,
            on_error=on_error,
            **kwargs,
        )

        if track:
            self._jobs.append(_ManagedJob(key=key, handle=handle))

        return handle

    def cancel_jobs(self) -> None:
        for mj in list(self._jobs):
            try:
                mj.handle.cancel()
            except Exception:
                pass
        self._jobs.clear()

    # -----------------------
    # Events
    # -----------------------
    def subscribe(self, topic: str, callback: Callable[[str, Any], None]) -> Optional[Subscription]:
        if self.events is None:
            return None

        sub = self.events.subscribe(
            topic,
            callback,
            owner_id=self.panel_id,
            owner_label=self.panel_name,
            owner_kind="panel",
        )
        self._subscriptions.append(sub)
        return sub

    def unsubscribe_all(self) -> None:
        if self.events is None:
            self._subscriptions.clear()
            return
        for sub in list(self._subscriptions):
            try:
                self.events.unsubscribe(sub)
            except Exception:
                pass
        self._subscriptions.clear()

    def publish(self, topic: str, payload: Any = None) -> None:
        if self.events is None:
            return
        try:
            self.events.publish(topic, payload)
        except Exception:
            traceback.print_exc()

    # -----------------------
    # Artifact helpers
    # -----------------------
    def put_artifact(
        self,
        type: str,
        payload: Any,
        *,
        dataset_id: str = "default",
        row_ids: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        persist: bool = False,
    ) -> Optional[str]:
        if self.artifacts is None:
            return None
        return self.artifacts.put(
            type,
            payload,
            dataset_id=dataset_id,
            row_ids=row_ids,
            params=params,
            persist=persist,
        )

    def _get_active_dataset_id(self):
        try:
            if self.context and getattr(self.context, "datasets", None):
                active = self.context.datasets.active_id()
                if active:
                    return active
        except Exception:
            pass
        return "default"

    def _get_selected_source_id(self):
        try:
            selected_id = self._get_selected_id()
            if selected_id is None:
                return None
            return str(selected_id)
        except Exception:
            return None

    def _event_matches_active_dataset(self, payload=None):
        """
        Return True when a dataset event should affect this panel.

        We accept:
        - no payload at all
        - payloads without dataset_id
        - payloads targeting the current active dataset
        """
        if not isinstance(payload, dict):
            return True

        active_id = self._get_active_dataset_id()
        payload_dataset_id = (
            payload.get("dataset_id")
            or payload.get("active_dataset_id")
            or payload.get("id")
        )

        return payload_dataset_id in (None, "", active_id)


    def _bind_dataset_runtime_subscriptions(self, *, include_loaded=False, include_mapping=False):
        """
        Opt a panel into dataset lifecycle responsiveness.

        New plugins should generally use:
            include_loaded=False
            include_mapping=True only if semantic mappings matter
        """
        if self._dataset_runtime_subscriptions_initialised:
            return

        self._dataset_runtime_subscriptions_initialised = True

        self.subscribe("dataset.active.changed", self._dataset_active_changed_cb)
        self.subscribe("dataset.updated", self._dataset_updated_cb)

        if include_loaded:
            self.subscribe("dataset.loaded", self._dataset_loaded_cb)

        if include_mapping:
            self.subscribe("dataset.mapping_updated", self._dataset_mapping_updated_cb)


    def _dataset_active_changed_cb(self, topic, payload):
        if not self._event_matches_active_dataset(payload):
            return
        self._request_refresh(reason=str(topic or "dataset.active.changed"), payload=payload)

    def _dataset_updated_cb(self, topic, payload):
        if not self._event_matches_active_dataset(payload):
            return
        self._request_refresh(reason=str(topic or "dataset.updated"), payload=payload)


    def _dataset_loaded_cb(self, topic, payload):
        if not self._event_matches_active_dataset(payload):
            return
        self._request_refresh(reason=str(topic or "dataset.loaded"), payload=payload)

    def _dataset_mapping_updated_cb(self, topic, payload):
        if not self._event_matches_active_dataset(payload):
            return
        self._request_refresh(reason=str(topic or "dataset.mapping_updated"), payload=payload)


    def _get_focus_state(self):
        if self.context is not None and getattr(self.context, "selection", None) is not None:
            try:
                return self.context.selection.get_focus()
            except Exception:
                pass
        return None


    def _bind_selection_runtime_subscriptions(self):
        """
        Standard selection watcher for new plugins.

        Focus changes request a coalesced refresh rather than launching work
        immediately.
        """
        if self._selection_runtime_subscriptions_initialised:
            return

        self._selection_runtime_subscriptions_initialised = True
        self.subscribe("selection.focus.changed", self._selection_focus_changed_cb)
        self.subscribe("selection.focus.cleared", self._selection_focus_cleared_cb)


    def _selection_focus_changed_cb(self, topic, payload):
        self._request_refresh(
            reason=str(topic or "selection.focus.changed"),
            payload=payload,
        )


    def _selection_focus_cleared_cb(self, topic, payload):
        self._request_refresh(
            reason=str(topic or "selection.focus.cleared"),
            payload=payload,
        )



    def _rebuild_layout_from_current_state(self):
        """
        Conservative fallback for panels that do not expose a dedicated refresh API.

        Rebuild the layout in place when possible so existing panel containers
        keep working.
        """
        if not hasattr(self, "get_layout"):
            return

        new_layout = self.get_layout()

        current_layout = getattr(self, "layout", None)
        if current_layout is None:
            self.layout = new_layout
            return

        try:
            if hasattr(current_layout, "objects") and hasattr(new_layout, "objects"):
                current_layout.objects = list(new_layout.objects)
            else:
                self.layout = new_layout
        except Exception:
            self.layout = new_layout


    def _refresh_for_dataset_change(self, reason=None, payload=None):
        """
        Backward-compatible entrypoint.

        Older code may still call this directly. Route it through the coalescing
        scheduler so it behaves the same way as dataset/src runtime triggers.
        """
        self._request_refresh(reason=reason or "dataset.change", payload=payload)

    def _perform_refresh(self, reason=None, payload=None, refresh_signature=None):
        """
        Default synchronous refresh hook for simple panels.

        New plugins should override this if they need custom refresh logic.
        Async plugins should call self._finish_refresh(refresh_signature) in their
        completion callback.
        """
        try:
            self.df = self._get_dataset_for_lookup()
        except Exception:
            traceback.print_exc()

        try:
            self._rebuild_layout_from_current_state()
        finally:
            self._finish_refresh(refresh_signature)

    def _request_refresh(self, reason="unknown", payload=None, verbose=False):
        """
        Coalesce repeated triggers onto the next tick.

        Dataset events, src.data changes, and initial layout should all funnel
        through this method.
        """
        if getattr(self, "_disposed", False):
            return

        if reason:
            self._refresh_reasons.add(str(reason))

        if payload is not None:
            self._refresh_pending_payload = payload

        if self._refresh_pending:
            return

        self._refresh_pending = True

        def _runner():
            self._refresh_pending = False
            merged_reason = " + ".join(sorted(self._refresh_reasons)) if self._refresh_reasons else "unknown"
            merged_payload = self._refresh_pending_payload

            self._refresh_reasons.clear()
            self._refresh_pending_payload = None

            self._run_scheduled_refresh(
                reason=merged_reason,
                payload=merged_payload,
                verbose=verbose,
            )

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_runner)
            else:
                _runner()
        except Exception:
            _runner()

    def _run_scheduled_refresh(self, reason=None, payload=None, verbose=False):
        """
        Start one refresh cycle unless an identical request is already in flight.
        """
        refresh_signature = self._begin_refresh(reason=reason, payload=payload, verbose=verbose)
        if refresh_signature is None:
            return

        try:
            self._perform_refresh(
                reason=reason,
                payload=payload,
                refresh_signature=refresh_signature,
            )
        except Exception:
            traceback.print_exc()
            self._finish_refresh(refresh_signature)


    def _build_refresh_signature(self, reason=None, payload=None):
        return (
            self._get_active_dataset_id(),
            self._get_selected_source_id(),
        )

    def _begin_refresh(self, reason=None, payload=None, verbose=False):
        """
        Mark a refresh as in-flight unless the same request is already running.
        """
        try:
            refresh_signature = self._build_refresh_signature(reason=reason, payload=payload)
        except Exception:
            traceback.print_exc()
            refresh_signature = (
                self._get_active_dataset_id(),
                self._get_selected_source_id(),
            )

        if refresh_signature == self._refresh_inflight_signature:
            if verbose:
                print(
                    f"[{self.panel_name}] refresh skipped; identical request already in flight: "
                    f"{refresh_signature} (reason={reason})"
                )
            return None

        self._refresh_inflight_signature = refresh_signature
        return refresh_signature

    def _finish_refresh(self, refresh_signature=None):
        """
        Mark the current refresh as complete.

        Async subclasses should call this in their completion callback.
        """
        if refresh_signature is None:
            refresh_signature = self._refresh_inflight_signature

        if refresh_signature is not None:
            self._last_completed_signature = refresh_signature

        if self._refresh_inflight_signature == refresh_signature:
            self._refresh_inflight_signature = None


    def _request_initial_refresh_once(self, reason="initial.layout"):
        """
        Call from get_layout() in new plugins instead of launching expensive work
        directly in get_layout().
        """
        if self._initial_refresh_requested:
            return

        self._initial_refresh_requested = True
        self._request_refresh(reason=reason)

    # -----------------------
    # Column mapping / semantic requirements
    # -----------------------
    def _get_dataset_for_lookup(self):
        """
        Prefer authoritative dataset from context.datasets when available.
        Fall back to self.df.
        """
        try:
            if self.datasets is not None:
                dataset_id = self._get_active_dataset_id()
                df = self.datasets.get_df(dataset_id)
                if df is not None:
                    return df
        except Exception:
            pass
        return self.df

    def _get_all_known_mappings(self) -> Dict[str, str]:
        """
        Best-effort collection of semantic column mappings.

        Supports:
        - context.datasets.get_mappings(dataset_id)
        - context.datasets.get_mappings()
        - context.config.settings
        """
        mappings = {}

        dataset_id = self._get_active_dataset_id()

        # Preferred: DatasetManager mappings
        try:
            if self.datasets is not None and hasattr(self.datasets, "get_mappings"):
                try:
                    ds_mappings = self.datasets.get_mappings(dataset_id)
                except TypeError:
                    ds_mappings = self.datasets.get_mappings()
                if isinstance(ds_mappings, dict):
                    mappings.update(ds_mappings)
        except Exception:
            pass

        # Back-compat: config.settings aliases
        try:
            if self.config is not None and hasattr(self.config, "settings"):
                if isinstance(self.config.settings, dict):
                    mappings.update(self.config.settings)
        except Exception:
            pass

        return mappings

    def resolve_column_name(
        self,
        requirement: str,
        *,
        df: Optional[pd.DataFrame] = None,
        allow_direct: bool = True,
    ) -> Optional[str]:
        """
        Resolve a semantic requirement like 'ra', 'dec', 'id_col', 'label_col'
        to a real dataframe column.

        Resolution order:
        1. direct column name match
        2. dataset/config mapping under exact key
        3. common aliases for semantic requirements
        """
        df = df if df is not None else self._get_dataset_for_lookup()
        if df is None:
            return None

        cols = set(getattr(df, "columns", []))
        mappings = self._get_all_known_mappings()

        if allow_direct and requirement in cols:
            return requirement

        mapped = mappings.get(requirement)
        if isinstance(mapped, str) and mapped in cols:
            return mapped

        aliases = {
            "id": ["id", "ids", "source_id", "object_id", "id_col"],
            "id_col": ["id_col", "id", "ids", "source_id", "object_id"],
            "ra": [
                "ra",
                "right_ascension",
                "right_ascension_euclid",
                "raj2000",
                "alpha",
                "ra_col",
            ],
            "dec": [
                "dec",
                "declination",
                "declination_euclid",
                "dej2000",
                "delta",
                "dec_col",
            ],
            "label": ["label", "class", "target", "label_col"],
            "label_col": ["label_col", "label", "class", "target"],
        }

        for alias in aliases.get(requirement, []):
            mapped = mappings.get(alias)
            if isinstance(mapped, str) and mapped in cols:
                return mapped
            if alias in cols:
                return alias

        return None

    def require_columns(self, *requirements: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Optional[str]]:
        """
        Return a mapping of semantic requirement -> resolved dataframe column.
        """
        df = df if df is not None else self._get_dataset_for_lookup()
        return {req: self.resolve_column_name(req, df=df) for req in requirements}

    # -----------------------
    # Lifecycle
    # -----------------------
    def _dispose_impl(self) -> None:
        return

    def dispose(self) -> None:
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        print(f"[dispose] {self.__class__.__name__} panel_id={getattr(self, 'panel_id', None)}")

        try:
            self._dispose_impl()
        except Exception:
            pass

        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        try:
            self.remove_all_param_watches()
        except Exception:
            pass

        if hasattr(self, "remove_column_selection"):
            try:
                self.remove_column_selection()
            except Exception:
                pass

        try:
            self.cancel_jobs()
        except Exception:
            pass

        try:
            self.unsubscribe_all()
        except Exception:
            pass

        ex = getattr(self, "executor", None)
        if ex is not None:
            try:
                ex.shutdown(wait=False)
                print(f"[{self.panel_id}] Thread executor shutdown.")
            except Exception:
                pass

    def run_multithread(
        self,
        fn,
        *,
        func_kwargs=None,
        callback=None,
        errback=None,
        title="Job",
        key=None,
    ):
        func_kwargs = func_kwargs or {}

        class _FutureLike:
            def __init__(self, result=None, exc=None):
                self._result = result
                self._exc = exc

            def result(self):
                if self._exc is not None:
                    raise self._exc
                return self._result

        def _runner(cancel_token=None, **kwargs):
            return fn(**kwargs)

        def _on_done(res):
            if callback:
                callback(_FutureLike(result=res))

        def _on_err(exc: BaseException):
            if errback:
                errback(exc)
            elif callback:
                callback(_FutureLike(exc=exc))

        return self.submit_job(
            _runner,
            title=title,
            key=key,
            on_done=_on_done,
            on_error=_on_err,
            **func_kwargs,
        )

    def add_param_watch(
        self,
        owner: Any,
        callback: Callable,
        what: Union[str, Iterable[str]],
        *,
        onlychanged: bool = True,
        queued: bool = False,
        precedence: int = 0,
    ):
        if owner is None or not hasattr(owner, "param"):
            return None
        try:
            w = owner.param.watch(
                callback,
                what,
                onlychanged=onlychanged,
                queued=queued,
                precedence=precedence,
            )
            self._param_watchers.append((owner, w))
            return w
        except Exception:
            return None

    def remove_all_param_watches(self) -> None:
        for owner, w in list(getattr(self, "_param_watchers", [])):
            try:
                owner.param.unwatch(w)
            except Exception:
                pass
        self._param_watchers.clear()

    def add_param_watch_many(self, owners, callback, what="value", **kw):
        for owner in owners:
            self.add_param_watch(owner, callback, what, **kw)

    def watch_bokeh(self, model: Any, attr: str, callback: Callable) -> None:
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            pass

    def unwatch_all_bokeh(self) -> None:
        for model, attr, callback in list(self._bokeh_on_change):
            try:
                model.remove_on_change(attr, callback)
            except Exception:
                pass
        self._bokeh_on_change.clear()

        print("on delete:", self._bokeh_on_change)

    def _submit_button_cb(self, event):
        for col, widget in self.select_widgets.items():
            selected_value = widget.value
            print(f"{col} --> {selected_value}")
            self.config.settings[col] = selected_value
        self.stage = "plot"

    def _skip_button_cb(self, event):
        current_index = self.available_stages.index(self.stage)
        self.stage = self.available_stages[current_index + 1]

    def _toggle_settings_panel(self, event):
        self.plot_settings_panel.visible = not self.plot_settings_panel.visible
        self.plot_settings_button.name = "Close Settings" if self.plot_settings_panel.visible else "Open Settings"

    # -----------------------
    # Selection helpers
    # -----------------------
    def get_selected_source(self):
        """
        Return the currently focused row as a one-row dataframe.

        Selection is now driven by context.selection rather than src.data.
        """
        selected_id = self._get_selected_id()
        if selected_id is None:
            return None

        base_df = self._get_dataset_for_lookup()
        if base_df is None or len(base_df) == 0:
            return None

        id_col = self.resolve_column_name("id_col", df=base_df) or self.resolve_column_name("id", df=base_df)

        try:
            if id_col is None or id_col == "Use Index":
                selected = base_df.loc[base_df.index.astype(str) == str(selected_id)]
            else:
                if id_col not in base_df.columns:
                    return None
                selected = base_df[base_df[id_col].astype(str) == str(selected_id)]

            if len(selected) == 0:
                return None

            return selected.head(1).reset_index(drop=True)
        except Exception:
            return None

    def get_value_from_df(self, column_or_requirement):
        """
        Read a value from the selected row using either:
        - a real dataframe column name
        - a semantic requirement like 'ra', 'dec', 'id_col'
        """
        selected_source = self.get_selected_source()
        print(f"running get_value_from_df in CustomPlotClass: {selected_source}")

        if selected_source is None or len(selected_source) != 1:
            return None

        col = self.resolve_column_name(column_or_requirement, df=selected_source)

        if col is None:
            return None

        try:
            return selected_source[col].iloc[0]
        except Exception:
            return None

    def get_ra_dec(self, err_message="No ra and dec available for this source"):
        """
        Preferred behavior:
        - resolve mapped RA/Dec columns semantically
        Legacy fallback:
        - use precomputed 'ra_dec' if present
        """
        selected_source = self.get_selected_source()
        print(f"running get_ra_dec in CustomPlotClass: {selected_source}")

        if selected_source is None or len(selected_source) != 1:
            print(err_message)
            return None, None

        ra_col = self.resolve_column_name("ra", df=selected_source)
        dec_col = self.resolve_column_name("dec", df=selected_source)

        if ra_col is not None and dec_col is not None:
            try:
                ra = float(selected_source[ra_col].iloc[0])
                dec = float(selected_source[dec_col].iloc[0])
                print(f"running get_ra_dec in CustomPlotClass: ({ra}, {dec})")
                return ra, dec
            except Exception:
                pass

        # Legacy fallback only
        if "ra_dec" in selected_source.columns:
            try:
                ra_dec = selected_source["ra_dec"].iloc[0]
                print(f"running get_ra_dec in CustomPlotClass legacy fallback: {ra_dec}")
                if ra_dec is not None:
                    ra_str, dec_str = str(ra_dec).split(",", 1)
                    return float(ra_str), float(dec_str)
            except Exception:
                pass

        print(err_message)
        return None, None

    def get_ra_dec_string(self, err_message="No ra and dec available for this source"):
        """
        Convenience helper for panels like spectra panels that still need 'ra,dec'
        for external query APIs. This value is derived locally and not assumed to
        already exist in the dataframe.
        """
        ra, dec = self.get_ra_dec(err_message=err_message)
        if ra is None or dec is None:
            return None
        return f"{ra},{dec}"

    def _get_selected_id(self):
        focus = self._get_focus_state()
        if focus is None:
            return None

        focus_dataset_id = getattr(focus, "dataset_id", None)
        focus_row_id = getattr(focus, "row_id", None)

        if focus_row_id is None:
            return None

        active_dataset_id = self._get_active_dataset_id()
        if focus_dataset_id not in (None, "", active_dataset_id):
            return None

        return str(focus_row_id)

    def check_required_column(self, column):
        """
        Backward-compatible check:
        returns True if the semantic requirement or explicit column can be resolved.
        """
        return self.resolve_column_name(column) is not None

    def get_column_list(
        self,
        excluded_columns=("id_col", "ra_dec", "label_col"),
        excluded_types=("object",),
        allowed_types=None,
    ):
        cols = list(getattr(self.df, "columns", []))

        for excluded_col in excluded_columns:
            col_name = self.resolve_column_name(excluded_col, df=self.df) or excluded_col
            if col_name in cols:
                cols.remove(col_name)

        if allowed_types:
            cols = [c for c in cols if matches_type(self.df[c].dtype, allowed_types)]
        if excluded_types:
            cols = [c for c in cols if not matches_type(self.df[c].dtype, excluded_types)]

        return cols

    def _get_selection_widgets_grid(
        self,
        columns_to_select,
        default_values=None,
        options=None,
        allowed_types=None
    ):
        settings_grid = pn.GridBox(ncols=3, sizing_mode="stretch_width", scroll=True)
        self.select_widgets = {}
        if options is None:
            options = self.get_column_list(
                excluded_columns=["ra_dec", "label_col"],
                excluded_types=["object"],
                allowed_types=allowed_types,
            )
        if len(columns_to_select) > 0:
            for i, col in enumerate(columns_to_select):
                select_widget = pn.widgets.Select(
                    name=col,
                    options=options,
                    max_height=120,
                    sizing_mode="stretch_width",
                )
                if (default_values is not None) and (i < len(default_values)):
                    select_widget.value = default_values[i]
                settings_grid.append(select_widget)
                self.select_widgets[col] = select_widget
        return settings_grid

    def columns_selection_panel(
        self,
        columns_to_select,
        skippable=False,
        options=None,
        allowed_types=None,
        info_text=None
    ):
        settings_grid = self._get_selection_widgets_grid(
            columns_to_select,
            options=options,
            allowed_types=allowed_types,
        )

        submit_button = pn.widgets.Button(name="Confirm", button_type="primary", max_height=120)
        submit_button.on_click(self._submit_button_cb)
        skip_button = pn.widgets.Button(name="Skip", button_type="primary", max_height=120)
        skip_button.on_click(self._skip_button_cb)
        if not skippable:
            skip_button.disabled = True
        if info_text is not None:
            card_content = pn.Column(
                pn.pane.Markdown(info_text, sizing_mode="stretch_width", margin=(15, 0, 15, 15)),
                settings_grid,
            )
        else:
            card_content = settings_grid

        toolbar = self.get_layout(submit_button=submit_button, skip_button=skip_button)
        return pn.Column(
            toolbar,
            card_content,
            sizing_mode="stretch_both",
            scroll=True,
            min_height=300,
        )

    def _get_unknown_columns(self, columns_needed, settings_key=None):
        current_cols = getattr(self.config.main_df, "columns", [])
        self.unknown_columns = []

        if settings_key is not None:
            if settings_key not in self.config.settings:
                self.config.settings[settings_key] = {}
            settings_dict = self.config.settings[settings_key]
        else:
            settings_dict = self.config.settings

        for col in columns_needed:
            if col not in settings_dict:
                print(f"{col} not in config")
                if col not in current_cols:
                    self.unknown_columns.append(col)
                else:
                    settings_dict[col] = col

    def _change_state_if_unknown_columns(self, unknown_stage="columns_selection", ready_stage="plot"):
        if hasattr(self, "unknown_columns"):
            if self.unknown_columns:
                self.stage = unknown_stage
            else:
                self.stage = ready_stage
        else:
            print("The unknown_columns attribute was not initialized, not changing Stage")

    def _save_panel(
        self,
        directory_path="data/saved_sources",
        save_fits_files=True,
        prefix=None,
    ):
        paths = {}
        if self.stage == "plot":
            try:
                paths["figure"] = self._save_figure(directory_path=directory_path, prefix=prefix)
            except AttributeError:
                print(f"{self.panel_name} has no _save_panel_method")

            if save_fits_files:
                try:
                    paths["fits_file"] = self._save_data_to_fits(directory_path=directory_path)
                except AttributeError:
                    pass
        return paths

    @staticmethod
    def get_empty_image():
        return hv.Image(np.ones((10, 10))).opts(
            active_tools=[],
            clim=(0, 1),
            toolbar=None,
            padding=0,
            border=0,
            framewise=True,
            xaxis=None,
            yaxis=None,
            cmap="grey",
        )

    def get_error_panel(self, message_1, message_2):
        message = f"# {message_1}:\n"
        message += f"## {message_2}"
        self.message_pane.object = message
        self.message_pane.visible = True
        self.figure.objects = [self.get_empty_image()]

    def remove_column_selection(self):
        if hasattr(self, "unknown_columns"):
            for col in self.unknown_columns:
                if col in self.config.settings:
                    del self.config.settings[col]
            print(f"[{self.panel_id}] unknown columns selected removed from config")

    def plot(self, N=20):
        self.message_pane.visible = True
        coords = [(i, np.random.random()) for i in range(N)]
        scatter = hv.Scatter(coords).opts(color="black", marker="+")
        self.figure.object = scatter
        self.message_pane.visible = False

    def get_layout(self):
        points_input = pn.widgets.IntInput(name="Number of points", value=20, start=1, sizing_mode="stretch_width")

        def update_points(event):
            N = points_input.value
            self.plot(N)

        self.plot_settings_panel.objects = [points_input]
        self.add_param_watch(points_input, update_points, what="value")
        self.plot(points_input.value)
        return pn.Column(
            self.message_pane,
            self.figure,
            self.plot_settings_panel,
            sizing_mode="stretch_both",
            min_height=450,
            styles={"background": "lightgreen"},
        )

    def get_toolbar(self, skip_button=None, submit_button=None):
        if self.stage == "columns_selection":
            toolbar = pn.Row(
                pn.Spacer(width=25),
                self.close_button,
                skip_button,
                submit_button,
                height=40,
            )
        else:
            if self.require_settings:
                toolbar = pn.Row(
                    pn.Spacer(width=25),
                    self.close_button,
                    self.plot_settings_button,
                    height=40,
                )
            else:
                toolbar = pn.Row(
                    pn.Spacer(width=25),
                    self.close_button,
                    height=40,
                )

        return toolbar

    def plot_panel(self):
        self.layout = self.get_layout()
        toolbar = self.get_toolbar()
        return pn.Column(toolbar, self.layout, sizing_mode="stretch_both", min_height=450)

    @param.depends("stage")
    def panel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(self.unknown_columns)
        else:
            return self.plot_panel()


# Constants and Configuration
EMISSION_LINES = {
    'Lyalpha': 1215.67,
    '[OII]': 3727.09,
    'Hbeta': 4861.32,
    '[OIII]4959': 4958.91,
    '[OIII]5007': 5006.84,
    'Halpha': 6562.80,
    '[SII]6716': 6716.44,
    '[SII]6731': 6730.82
}

CLASSIFICATION_COLOURS = {
    '[OII]': 'deep sky blue',
    '[OIII]5007': 'green',
    'Halpha': 'blue',
    'Unclear': 'orange',
    'Noisy/Bad': 'red',
    'Unclassified': 'black'
}

class AnalysisDefaults:
    SIGNAL_WINDOW_WIDTH = 50
    NOISE_OFFSET = 100
    NOISE_WINDOW_WIDTH = 150
    PEAK_HEIGHT_THRESHOLD_SIGMA = 2
    PEAK_MIN_DISTANCE = 30
    CONTINUUM_BUFFER = 20
    UPDATE_DEBOUNCE_MS = 250
    # Euclid-specific: typical resolution R~380 at 1.1-2.0 microns
    MIN_LINE_WIDTH_ANGSTROM = 2.0  # Minimum physical line width
    MAX_LINE_WIDTH_ANGSTROM = 100.0  # Maximum to catch broad lines

class SpecAnalyser(CustomPlotClass):

    EMISSION_LINES = {
        "OII": 3727.0,
        "Hbeta": 4861.0,
        "OIII": 5007.0,
        "Halpha": 6563.0,
    }

    def __init__(self, data, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            close_button=close_button,
            extra_features=extra_features,
            context=context,
            panel_name="SpecAnalyser",
            **params,
        )

        import logging

        from bokeh.events import Tap
        from bokeh.models import (
            BoxAnnotation,
            ColumnDataSource,
            CrosshairTool,
            HoverTool,
            Label,
            LabelSet,
        )
        from bokeh.plotting import figure

        from astronomicAL.plugins.spec_analyser.gui_analyser import (
            AnalysisError,
            AnalysisRegions,
            PlotState,
            ResultsManager,
            SpectrumAnalyser,
            SpectrumData,
        )

        self.logging = logging
        self.np = np
        self.pd = pd
        self.pn = pn

        self.AnalysisError = AnalysisError
        self.SpectrumData = SpectrumData

        self.subscribe("astro.spectrum.updated", self._spectra_updated)

        self.doc = pn.state.curdoc

        # -------------------------
        # State
        # -------------------------
        self._spectra_cache = {}
        self._current_spec_key = None
        self._current_spectrum_data = None
        self._regions = AnalysisRegions()
        self._plot_state = PlotState()
        self._locked_fits = []
        self._analysis_results = None
        self._continuum_model = None
        self._continuum = None
        self._corrected_flux = None
        self._analyser = SpectrumAnalyser()
        self._results_manager = ResultsManager()
        self._pending_region_click = None

        self._signal_region_defined = False
        self._noise_region_defined = False

        # -------------------------
        # Data sources
        # -------------------------
        self.raw_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))
        self.continuum_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))
        self.corrected_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))
        self.fit_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))
        self.locked_fit_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))
        self.residuals_source = ColumnDataSource(data=dict(wavelength=[], flux=[]))

        self.emission_line_source = ColumnDataSource(
            data=dict(x=[], y0=[], y1=[], y=[], label=[])
        )

        # -------------------------
        # Figures
        # -------------------------
        from bokeh.models import HoverTool, TapTool, WheelZoomTool, PanTool, ResetTool, SaveTool, CrosshairTool

        source_tap = TapTool()
        source_wheel = WheelZoomTool()

        self.source_plot = figure(
            height=280,
            title="Spectrum",
            tools=[CrosshairTool(), PanTool(), ResetTool(), SaveTool(), source_wheel, source_tap],
            active_scroll=source_wheel,
            active_tap=source_tap,
            output_backend="webgl",
            sizing_mode="stretch_width",
            toolbar_location="above",
        )

        residuals_tap = TapTool()
        residuals_wheel = WheelZoomTool()

        self.residuals_plot = figure(
            height=280,
            title="Residuals / Zoomed View",
            tools=[CrosshairTool(), PanTool(), ResetTool(), SaveTool(), residuals_wheel, residuals_tap],
            active_scroll=residuals_wheel,
            active_tap=residuals_tap,
            x_range=self.source_plot.x_range,
            output_backend="webgl",
            sizing_mode="stretch_width",
            toolbar_location="above",
        )
        self.residuals_plot.xaxis.axis_label = "Wavelength (Å)"
        self.residuals_plot.yaxis.axis_label = "Flux"

        self.spectrum_plot = self.source_plot

        # -------------------------
        # Region overlays
        # -------------------------
        self.signal_box = BoxAnnotation(fill_alpha=0.18, fill_color="green", visible=False)
        self.noise_box = BoxAnnotation(fill_alpha=0.15, fill_color="gray", visible=False)
        self.residuals_plot.add_layout(self.signal_box)
        self.residuals_plot.add_layout(self.noise_box)

        # -------------------------
        # Renderers
        # -------------------------
        self.raw_renderer = self.source_plot.line(
            "wavelength", "flux", source=self.raw_source, line_width=1, line_alpha=0.5, legend_label="Original Spectrum"
        )
        self.continuum_renderer = self.source_plot.line(
            "wavelength", "flux", source=self.continuum_source, line_width=2, line_dash="dashed", legend_label="Continuum Fit"
        )
        self.corrected_renderer = self.source_plot.line(
            "wavelength", "flux", source=self.corrected_source, line_width=1, legend_label="Continuum Subtracted"
        )
        self.locked_renderer = self.source_plot.line(
            "wavelength", "flux", source=self.locked_fit_source, line_width=2, legend_label="Locked Fits"
        )

        self.residuals_renderer = self.residuals_plot.line(
            "wavelength", "flux", source=self.residuals_source, line_width=1, legend_label="Residual / Corrected Flux"
        )
        self.fit_renderer = self.residuals_plot.line(
            "wavelength", "flux", source=self.fit_source, line_width=2, line_dash="dashed", legend_label="Fit"
        )

        self.emission_renderer = self.source_plot.segment(
            x0="x",
            y0="y0",
            x1="x",
            y1="y1",
            source=self.emission_line_source,
            line_dash="dashed",
            line_alpha=0.6,
            legend_label="Emission Lines",
        )
        self.emission_label_set = LabelSet(
            x="x",
            y="y",
            text="label",
            source=self.emission_line_source,
            angle=1.5708,
            text_font_size="8pt",
        )


        self.source_plot.add_layout(self.emission_label_set)

        self.emission_renderer.on_change("visible", self._on_emission_renderer_visible)
        self._sync_emission_label_visibility()

        for plot in (self.source_plot, self.residuals_plot):
            if plot.legend:
                plot.legend.visible = False
            plot.legend.click_policy = "hide"
            plot.add_tools(HoverTool(tooltips=[("Wavelength", "@wavelength"), ("Flux", "@flux")]))


        self.source_legend = self._make_compact_external_legend(
            self.source_plot,
            [
                ("Original Spectrum", [self.raw_renderer]),
                ("Continuum Fit", [self.continuum_renderer]),
                ("Continuum Subtracted", [self.corrected_renderer]),
                ("Locked Fits", [self.locked_renderer]),
                ("Emission Lines", [self.emission_renderer]),
            ],
        )

        self.residuals_legend = self._make_compact_external_legend(
            self.residuals_plot,
            [
                ("Residual / Corrected Flux", [self.residuals_renderer]),
                ("Fit", [self.fit_renderer]),
            ],
        )


        # Fit / status annotations
        self.fit_status_label = Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text="",
            text_color="red",
            text_font_size="11pt",
            visible=False,
        )
        self.residuals_plot.add_layout(self.fit_status_label)

        # -------------------------
        # UI
        # -------------------------

        SIDEBAR_WIDTH = 360
        FIELD_WIDTH = 320
        SMALL_FIELD_WIDTH = 120

        CONTROL_HEIGHT = 34
        BUTTON_HEIGHT = 34
        TEXTAREA_HEIGHT = 90

        LABEL_HEIGHT = 16
        LABEL_MARGIN_TOP = 2
        LABEL_MARGIN_BOTTOM = 2
        BLOCK_MARGIN_BOTTOM = 6


        def fix_height(widget, width, height=CONTROL_HEIGHT):
            widget.width = width
            widget.height = height
            widget.min_height = height
            widget.max_height = height
            widget.sizing_mode = "fixed"
            return widget

        def field_label(text):
            return pn.pane.HTML(
                f"""
                <div style="
                    font-weight: 600;
                    font-size: 13px;
                    line-height: {LABEL_HEIGHT}px;
                    padding-left: 2px;
                    margin: 0;
                ">
                    {text}
                </div>
                """,
                width=FIELD_WIDTH,
                height=LABEL_HEIGHT,
                margin=(LABEL_MARGIN_TOP, 0, LABEL_MARGIN_BOTTOM, 0),
                sizing_mode="fixed",
            )


        def field_block(text, widget, bottom=BLOCK_MARGIN_BOTTOM):
            return pn.Column(
                field_label(text),
                widget,
                width=FIELD_WIDTH,
                margin=(0, 0, bottom, 0),
                sizing_mode="fixed",
            )

        self.redshift_slider = pn.widgets.FloatSlider(
            name="",
            value=0.0,
            start=0.0,
            end=5.0,
            step=0.01,
            width=FIELD_WIDTH,
            margin=0,
            sizing_mode="fixed",
        )

        self.redshift_box = fix_height(
            pn.widgets.FloatInput(
                name="",
                value=0.0,
                start=0.0,
                end=5.0,
                step=0.001,
                margin=0,
            ),
            SMALL_FIELD_WIDTH,
        )

        self.finder_mode_checkbox = pn.widgets.Checkbox(
            name="",
            value=False,
            width=24,
            margin=0,
        )

        self.finder_mode_row = pn.Row(
            self.finder_mode_checkbox,
            pn.pane.HTML(
                "<div style='line-height:20px; padding-left:6px;'>Finder Mode</div>",
                width=FIELD_WIDTH - 30,
                height=20,
                margin=0,
            ),
            width=FIELD_WIDTH,
            height=20,
            margin=(0, 0, 8, 0),
            sizing_mode="fixed",
        )

        self.plot_settings_checkbox = pn.widgets.CheckBoxGroup(
            name="",
            value=["Show original spectrum", "Show continuum subtracted spectrum"],
            options=[
                "Show original spectrum",
                "Show continuum subtracted spectrum",
                "Show continuum fit",
            ],
            inline=False,
            width=FIELD_WIDTH,
            height=60,
            margin=0,
            sizing_mode="fixed",
        )

        self.fitting_mode_buttons = pn.widgets.RadioButtonGroup(
            name="",
            value="Single fit",
            options=["Single fit", "Multiline fit"],
            button_type="default",
            width=FIELD_WIDTH,
            height=30,
            margin=0,
            sizing_mode="fixed",
        )

        self.line_name_input = fix_height(
            pn.widgets.TextInput(
                name="",
                placeholder="Line Name",
                margin=0,
            ),
            FIELD_WIDTH,
        )

        self.line_profile_selector = fix_height(
            pn.widgets.Select(
                name="",
                options=["Gaussian", "Lorentzian", "Voigt"],
                value="Gaussian",
                margin=0,
            ),
            FIELD_WIDTH,
        )

        self.line_name_selector = fix_height(
            pn.widgets.Select(
                name="",
                options=self.EMISSION_LINES,
                value=3727.0,
                margin=0,
            ),
            FIELD_WIDTH,
        )

        self.select_region_buttons = pn.widgets.RadioButtonGroup(
            name="",
            value=None,
            options=["Signal region", "Noise region"],
            button_type="default",
            width=FIELD_WIDTH,
            height=34,
            min_height=34,
            max_height=34,
            margin=0,
            sizing_mode="fixed",
        )

        self.available_spectra = fix_height(
            pn.widgets.Select(
                name="",
                options=[],
                margin=0,
            ),
            FIELD_WIDTH,
        )

        self.spectra_number_message = pn.widgets.StaticText(
            name="",
            value="",
            width=FIELD_WIDTH,
            height=22,
            margin=0,
            sizing_mode="fixed",
        )

        ACTION_BUTTON_WIDTH = 100

        self.fit_button = pn.widgets.Button(
            name="Fit and Lock",
            button_type="success",
            width=ACTION_BUTTON_WIDTH,
            height=BUTTON_HEIGHT,
            min_height=BUTTON_HEIGHT,
            max_height=BUTTON_HEIGHT,
            margin=0,
            sizing_mode="fixed",
        )

        self.reset_button = pn.widgets.Button(
            name="Reset fit",
            button_type="warning",
            width=ACTION_BUTTON_WIDTH,
            height=BUTTON_HEIGHT,
            min_height=BUTTON_HEIGHT,
            max_height=BUTTON_HEIGHT,
            margin=0,
            sizing_mode="fixed",
        )

        self.undo_lock_button = pn.widgets.Button(
            name="Undo last lock",
            button_type="light",
            width=ACTION_BUTTON_WIDTH,
            height=BUTTON_HEIGHT,
            min_height=BUTTON_HEIGHT,
            max_height=BUTTON_HEIGHT,
            margin=0,
            sizing_mode="fixed",
        )

        self.action_buttons_row = pn.Row(
            self.fit_button,
            self.reset_button,
            self.undo_lock_button,
            width=FIELD_WIDTH,
            margin=(0, 0, 10, 0),
            sizing_mode="fixed",
        )

        self.status_message = pn.pane.HTML(
            "",
            visible=False,
            width=FIELD_WIDTH,
            height=56,
            min_height=56,
            max_height=56,
            margin=(0, 0, 8, 0),
            sizing_mode="fixed",
        )

        self.derived_properties_table = pn.pane.DataFrame(
            self.pd.DataFrame(
                columns=[
                    "Line",
                    "SNR",
                    "Flux Integral",
                    "FWHM (obs, Å)",
                    "EW (Å)",
                ]
            ),
            width=FIELD_WIDTH,
            height=190,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )

        self.comments_input = pn.widgets.TextAreaInput(
            name="",
            placeholder="Comments",
            height=TEXTAREA_HEIGHT,
            min_height=TEXTAREA_HEIGHT,
            max_height=TEXTAREA_HEIGHT,
            width=FIELD_WIDTH,
            margin=0,
            sizing_mode="fixed",
        )

        analysis_form = pn.Column(
            pn.Spacer(height=6),
            field_block("Redshift value", self.redshift_box, bottom=8),
            field_block("Redshift slider", self.redshift_slider, bottom=2),
            self.finder_mode_row,
            field_block("Plot settings", self.plot_settings_checkbox, bottom=2),
            field_block("Fitting mode", self.fitting_mode_buttons, bottom=10),
            field_block("Line name", self.line_name_input, bottom=8),
            field_block("Line profile", self.line_profile_selector, bottom=8),
            field_block("Go to line", self.line_name_selector, bottom=8),
            self.status_message,
            field_block("Region selection", self.select_region_buttons, bottom=10),
            self.action_buttons_row,
            field_label("Derived properties"),
            pn.Spacer(height=2),
            self.derived_properties_table,
            pn.Spacer(height=10),
            field_block("Comments", self.comments_input, bottom=8),
            width=SIDEBAR_WIDTH,
            sizing_mode="fixed",
        )

        self.analysis_tab = pn.Column(
            analysis_form,
            width=SIDEBAR_WIDTH,
            min_width=SIDEBAR_WIDTH,
            max_width=SIDEBAR_WIDTH,
            height=660,
            scroll=True,
            sizing_mode="fixed",
        )

        # self.settings_tabs = pn.Tabs(
        #     ("Analysis", self.analysis_tab),
        #     width=SIDEBAR_WIDTH,
        #     min_width=SIDEBAR_WIDTH,
        #     max_width=SIDEBAR_WIDTH,
        #     height=700,
        #     sizing_mode="fixed",
        #     dynamic=False,
        # )

        # -------------------------
        # Wiring
        # -------------------------
        self.available_spectra.param.watch(self._on_spectrum_selected, "value")
        self.redshift_slider.link(self.redshift_box, value="value")
        self.redshift_box.link(self.redshift_slider, value="value")
        self.redshift_slider.param.watch(self._on_redshift_changed, "value")
        self.plot_settings_checkbox.param.watch(self._on_plot_settings_changed, "value")
        self.line_name_selector.param.watch(self._on_line_selected, "value")
        self.select_region_buttons.param.watch(self._on_region_mode_changed, "value")

        self.fit_button.on_click(self._on_fit_clicked)
        self.reset_button.on_click(self._on_reset_clicked)
        self.undo_lock_button.on_click(self._on_undo_lock_clicked)

        self.residuals_plot.on_event(Tap, self._on_plot_tap)

        self._on_plot_settings_changed(None)
        self._update_region_overlays()
        self._update_results_table(None)
        self._set_status("", visible=False)

        self._refresh_spectra_from_artifacts()

        self._bind_selection_runtime_subscriptions()
        self._bind_dataset_runtime_subscriptions(
            include_loaded=True,
            include_mapping=True,
        )

    def _sync_emission_label_visibility(self):
        data = self.emission_line_source.data or {}
        has_lines = len(data.get("x", [])) > 0
        self.emission_label_set.visible = bool(self.emission_renderer.visible and has_lines)

    def _on_emission_renderer_visible(self, attr, old, new):
        self._sync_emission_label_visibility()

    def _make_compact_external_legend(self, plot, items, side="right"):
        legend = Legend(
            items=items,
            location="top_left",
            orientation="vertical",
            label_text_font_size="8pt",
            glyph_width=10,
            glyph_height=10,
            spacing=2,
            padding=4,
            margin=0,
            label_standoff=4,
            background_fill_alpha=0.0,
            border_line_alpha=0.0,
            click_policy="hide",
        )
        plot.add_layout(legend, side)
        return legend

    # ------------------------------------------------------------------
    # Data conversion
    # ------------------------------------------------------------------

    def _extract_artifact_redshift(self, spec):
        try:
            s0 = spec["spectra"][0]
        except Exception:
            return None

        z = s0.get("redshift", None)

        try:
            z = float(z)
        except Exception:
            return None

        if not self.np.isfinite(z):
            return None

        return z

    def _resolve_redshift_for_selected(self, selected):
        # 1) Prefer the selected spectrum's own redshift
        selected_spec = self._spectra_cache.get(selected)
        selected_z = self._extract_artifact_redshift(selected_spec) if selected_spec else None
        if selected_z is not None:
            return selected_z

        # Collect all valid redshifts from loaded spectra
        valid_redshifts = {}
        for key, spec in self._spectra_cache.items():
            z = self._extract_artifact_redshift(spec)
            if z is not None:
                valid_redshifts[key] = z

        # 2) If current slider value matches one of the loaded valid redshifts, preserve it
        try:
            current_z = float(self.redshift_slider.value)
            if self.np.isfinite(current_z):
                for z in valid_redshifts.values():
                    if abs(z - current_z) < 1e-10:
                        return current_z
        except Exception:
            pass

        # 3) Otherwise use any other available valid redshift
        if valid_redshifts:
            return next(iter(valid_redshifts.values()))

        # 4) Nothing available -> reset
        return 0.0

    def _artifact_to_spectrum_data(self, spec, source_name):
        s0 = spec["spectra"][0]

        wv = self.np.asarray(s0["wavelength"], dtype=float)
        flux = self.np.asarray(s0["flux"], dtype=float)

        finite = self.np.isfinite(wv) & self.np.isfinite(flux)

        flux_err = None
        if "flux_err" in s0:
            arr = self.np.asarray(s0["flux_err"], dtype=float)
            flux_err = arr[finite]

        lsf_var = None
        if "lsf_var" in s0:
            arr = self.np.asarray(s0["lsf_var"], dtype=float)
            lsf_var = arr[finite]

        return self.SpectrumData(
            wv=wv[finite],
            flux=flux[finite],
            flux_err=flux_err,
            lsf_var=lsf_var,
            source_id=s0.get("id", source_name),
            error=None,
        )

    # ------------------------------------------------------------------
    # External event handling
    # ------------------------------------------------------------------

    def _selection_focus_changed_cb(self, topic, payload):
        self._refresh_spectra_from_artifacts(
            dataset_id=self._get_active_dataset_id(),
            selected_id=self._get_selected_source_id(),
        )

    def _selection_focus_cleared_cb(self, topic, payload):
        self._refresh_spectra_from_artifacts(
            dataset_id=self._get_active_dataset_id(),
            selected_id=None,
        )

    def _get_active_dataset_id(self):
        try:
            if self.context and getattr(self.context, "datasets", None):
                active = self.context.datasets.active_id()
                if active:
                    return active
        except Exception:
            pass
        return "default"

    def _get_selected_source_id(self):
        try:
            selected_id = self._get_selected_id()
            if selected_id is None:
                return None
            return str(selected_id)
        except Exception:
            return None

    def _collect_spectra_from_artifacts(self, dataset_id=None, selected_id=None):
        dataset_id = dataset_id or self._get_active_dataset_id()
        selected_id = selected_id if selected_id is not None else self._get_selected_source_id()

        spectra_by_source = {}

        if not (self.context and getattr(self.context, "artifacts", None)):
            return spectra_by_source

        for src_name in ("DESI", "SDSS", "EuclidSpec"):
            try:
                params_subset = {"source": src_name}
                if selected_id is not None:
                    params_subset["selected_id"] = selected_id

                refs = self.context.artifacts.find(
                    type="astro.spectrum",
                    dataset_id=dataset_id,
                    params_subset=params_subset,
                )

                if refs:
                    spec = self.context.artifacts.get(refs[0].artifact_id)
                    if spec and spec.get("spectra"):
                        spectra_by_source[src_name] = spec

            except Exception as e:
                self.logging.warning(
                    "Failed loading %s spectrum artifact for selected_id=%s: %s",
                    src_name, selected_id, e
                )

        return spectra_by_source


    def _apply_spectra_models(self, spectra_by_source):
        avail_spectra = list(spectra_by_source.keys())

        self._spectra_cache = spectra_by_source
        self.available_spectra.options = avail_spectra

        if not avail_spectra:
            self.spectra_number_message.value = "No available spectra for this source."
            self._clear_all_sources()
            self._current_spec_key = None
            self._current_spectrum_data = None
            self._update_emission_lines()
            self._update_results_table(None)
            self._set_status("No available spectra for this source.", level="warning", visible=True)
            return

        self.spectra_number_message.value = (
            "Only one available spectrum for this source."
            if len(avail_spectra) == 1
            else "More than one available spectrum for this source."
        )

        selected = (
            self.available_spectra.value
            if self.available_spectra.value in avail_spectra
            else avail_spectra[0]
        )

        self.available_spectra.value = selected
        self._load_selected_spectrum(selected)
        self._set_status("", visible=False)


    def _refresh_spectra_from_artifacts(self, dataset_id=None, selected_id=None):
        spectra_by_source = self._collect_spectra_from_artifacts(
            dataset_id=dataset_id,
            selected_id=selected_id,
        )

        def _update():
            self._apply_spectra_models(spectra_by_source)

        if self.doc is not None:
            self.doc.add_next_tick_callback(_update)
        else:
            _update()
    
    def _spectra_updated(self, topic, payload):
        dataset_id = self._get_active_dataset_id()
        current_selected_id = self._get_selected_source_id()

        if payload:
            dataset_id = payload.get("dataset_id", dataset_id)
            payload_selected_id = payload.get("selected_id")

            if current_selected_id is not None and payload_selected_id is not None:
                if str(payload_selected_id) != str(current_selected_id):
                    return

        self._refresh_spectra_from_artifacts(
            dataset_id=dataset_id,
            selected_id=current_selected_id,
        )

    # ------------------------------------------------------------------
    # Source / plot state updates
    # ------------------------------------------------------------------
    def _clear_all_sources(self):
        empty = {"wavelength": [], "flux": []}
        self.raw_source.data = empty
        self.corrected_source.data = empty
        self.continuum_source.data = empty
        self.residuals_source.data = empty
        self.fit_source.data = empty
        self.locked_fit_source.data = empty
        self.emission_line_source.data = dict(x=[], y0=[], y1=[], y=[], label=[])

    def _load_selected_spectrum(self, selected):
        self._current_spec_key = selected
        spec = self._spectra_cache[selected]
        spectrum_data = self._artifact_to_spectrum_data(spec, selected)
        self._current_spectrum_data = spectrum_data

        resolved_redshift = self._resolve_redshift_for_selected(selected)
        if self.redshift_slider.value != resolved_redshift:
            self.redshift_slider.value = resolved_redshift

        raw_data = {
            "wavelength": spectrum_data.wv.tolist(),
            "flux": spectrum_data.flux.tolist(),
        }

        self.raw_source.data = raw_data
        self.residuals_source.data = raw_data.copy()
        self.corrected_source.data = {"wavelength": [], "flux": []}
        self.continuum_source.data = {"wavelength": [], "flux": []}
        self.fit_source.data = {"wavelength": [], "flux": []}
        self.locked_fit_source.data = {"wavelength": [], "flux": []}

        self._analysis_results = None
        self._continuum_model = None
        self._continuum = None
        self._corrected_flux = None
        self._locked_fits = []

        self._update_emission_lines()

        self._signal_region_defined = False
        self._noise_region_defined = False

        self._update_region_overlays()
        self._update_results_table(None)
        self._clear_fit_status()

        if spectrum_data.wv.size:
            self.residuals_plot.x_range.start = float(spectrum_data.wv.min())
            self.residuals_plot.x_range.end = float(spectrum_data.wv.max())

    def _update_region_overlays(self):
        if self._signal_region_defined:
            self.signal_box.left = float(self._regions.signal_start)
            self.signal_box.right = float(self._regions.signal_end)
            self.signal_box.visible = True
        else:
            self.signal_box.visible = False
            self.signal_box.left = None
            self.signal_box.right = None

        if self._noise_region_defined:
            self.noise_box.left = float(self._regions.noise_start)
            self.noise_box.right = float(self._regions.noise_end)
            self.noise_box.visible = True
        else:
            self.noise_box.visible = False
            self.noise_box.left = None
            self.noise_box.right = None

    def _update_emission_lines(self):
        if self._current_spectrum_data is None or self._current_spectrum_data.wv.size == 0:
            self.emission_line_source.data = dict(x=[], y0=[], y1=[], y=[], label=[])
            return

        wv = self._current_spectrum_data.wv
        flux = self._current_spectrum_data.flux
        z = float(self.redshift_slider.value)

        wv_min, wv_max = float(wv.min()), float(wv.max())
        y0 = float(self.np.nanmin(flux))
        y1 = float(self.np.nanmax(flux))
        if y0 == y1:
            y1 = y0 + 1.0
        y_text = y0 + 0.80 * (y1 - y0)

        xs, ys0, ys1, ys_text, labels = [], [], [], [], []
        for name, rest_wl in self.EMISSION_LINES.items():
            obs_wl = rest_wl * (1.0 + z)
            if wv_min <= obs_wl <= wv_max:
                xs.append(obs_wl)
                ys0.append(y0)
                ys1.append(y1)
                ys_text.append(y_text)
                labels.append(name)

        self.emission_line_source.data = dict(
            x=xs,
            y0=ys0,
            y1=ys1,
            y=ys_text,
            label=labels,
        )

        self._sync_emission_label_visibility()

    def _rebuild_locked_fit_source(self):
        if self._current_spectrum_data is None or not self._locked_fits:
            self.locked_fit_source.data = {"wavelength": [], "flux": []}
            return

        wv = self._current_spectrum_data.wv
        total = self.np.zeros_like(wv, dtype=float)

        for fit_result in self._locked_fits:
            model = fit_result.get("model")
            if model is not None:
                total += model(wv)

        self.locked_fit_source.data = {
            "wavelength": wv.tolist(),
            "flux": total.tolist(),
        }

    def _update_results_table(self, results):
        cols = [
            "Line",
            "SNR",
            "Flux Integral",
            "FWHM (obs, Å)",
            "FWHM (int, Å)",
            "FWHM (km/s)",
            "EW (Å)",
            "χ² reduced",
        ]

        if not results or results.get("fit_failed"):
            df = self.pd.DataFrame(columns=cols)
        else:
            df = self.pd.DataFrame([{
                "Line": self.line_name_input.value or "Unknown",
                "SNR": results.get("snr"),
                "Flux Integral": results.get("flux_integral"),
                "FWHM (obs, Å)": results.get("fwhm_obs_A"),
                "FWHM (int, Å)": results.get("fwhm_int_A"),
                "FWHM (km/s)": results.get("fwhm_kms"),
                "EW (Å)": results.get("ew"),
                "χ² reduced": results.get("chi_squared_red"),
            }])

        self.derived_properties_table.object = df

    def _set_status(self, text="", level="info", visible=False, field_width=320):
        styles = {
            "info": {
                "background": "#eff6ff",
                "border": "1px solid #93c5fd",
                "color": "#1d4ed8",
            },
            "success": {
                "background": "#ecfdf5",
                "border": "1px solid #86efac",
                "color": "#166534",
            },
            "warning": {
                "background": "#fffbeb",
                "border": "1px solid #fcd34d",
                "color": "#92400e",
            },
            "danger": {
                "background": "#fef2f2",
                "border": "1px solid #fca5a5",
                "color": "#991b1b",
            },
        }

        style = styles.get(level, styles["info"])

        self.status_message.object = f"""
        <div style="
            width: {field_width}px;
            height: 56px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            padding: 0 12px;
            margin: 0;
            border-radius: 4px;
            background: {style['background']};
            border: {style['border']};
            color: {style['color']};
            font-size: 13px;
            line-height: 1.35;
            overflow: hidden;
        ">
            {text}
        </div>
        """
        self.status_message.visible = visible

    def _show_fit_status(self, text):
        self.fit_status_label.text = text
        self.fit_status_label.visible = True

    def _clear_fit_status(self):
        self.fit_status_label.text = ""
        self.fit_status_label.visible = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_spectrum_selected(self, event):
        if not event.new or event.new not in self._spectra_cache:
            return
        self._load_selected_spectrum(event.new)

    def _on_plot_settings_changed(self, event):
        selected = set(self.plot_settings_checkbox.value or [])

        self._plot_state.show_original = "Show original spectrum" in selected
        self._plot_state.show_continuum_sub = "Show continuum subtracted spectrum" in selected
        self._plot_state.show_continuum = "Show continuum fit" in selected

        self.raw_renderer.visible = self._plot_state.show_original
        self.corrected_renderer.visible = self._plot_state.show_continuum_sub
        self.continuum_renderer.visible = self._plot_state.show_continuum

    def _on_redshift_changed(self, event):
        self._update_emission_lines()

    def _on_line_selected(self, event):
        if event.new is None or self._current_spectrum_data is None:
            return

        z = float(self.redshift_slider.value)
        obs = float(event.new) * (1.0 + z)
        self.residuals_plot.x_range.start = obs - 100.0
        self.residuals_plot.x_range.end = obs + 100.0

    def _on_region_mode_changed(self, event):
        self._pending_region_click = None
        if event.new:
            self._set_status(
                f"{event.new}: click two positions on the residual plot to define start and end.",
                level="info",
                visible=True,
            )
        else:
            self._set_status("", visible=False)

    def _on_plot_tap(self, event):
        if not self.select_region_buttons.value:
            self._set_status(
                "Choose 'Signal region' or 'Noise region' first, then click twice on the residual plot.",
                level="warning",
                visible=True,
            )
            return

        if event.x is None:
            self._set_status(
                "Click inside the residual plot frame to define a region.",
                level="warning",
                visible=True,
            )
            return

        x = float(event.x)

        if self._current_spectrum_data is None or self._current_spectrum_data.wv.size == 0:
            self._set_status(
                "No spectrum loaded.",
                level="warning",
                visible=True,
            )
            return

        if self._pending_region_click is None:
            self._pending_region_click = x
            self._set_status(
                f"{self.select_region_buttons.value}: first edge set at {x:.2f}. Click the second edge.",
                level="info",
                visible=True,
            )
            return

        x0 = self._pending_region_click
        x1 = x
        left, right = sorted([x0, x1])

        if self.select_region_buttons.value == "Signal region":
            self._regions.signal_start = left
            self._regions.signal_end = right
            self._signal_region_defined = True

        elif self.select_region_buttons.value == "Noise region":
            self._regions.noise_start = left
            self._regions.noise_end = right
            self._noise_region_defined = True

        self._pending_region_click = None
        chosen_mode = self.select_region_buttons.value
        self.select_region_buttons.value = None

        self._update_region_overlays()

        self._set_status(
            f"{chosen_mode} updated: start={left:.2f}, end={right:.2f}",
            level="success",
            visible=True,
        )

    # ------------------------------------------------------------------
    # Continuum / fitting
    # ------------------------------------------------------------------
    def _estimate_continuum(self, spectrum_data):

        wv = spectrum_data.wv
        flux = spectrum_data.flux

        if wv.size < 2:
            return np.poly1d([0.0, float(np.nanmedian(flux)) if flux.size else 0.0])

        mask_signal = (wv >= self._regions.signal_start) & (wv <= self._regions.signal_end)
        fit_mask = ~mask_signal

        if fit_mask.sum() < 2:
            fit_mask = np.ones_like(wv, dtype=bool)

        coeffs = np.polyfit(wv[fit_mask], flux[fit_mask], deg=1)
        return np.poly1d(coeffs)

    def _build_residual_flux(self):
        if self._current_spectrum_data is None:
            return None

        if self._corrected_flux is not None:
            residual_flux = self._corrected_flux.copy()
        else:
            residual_flux = self._current_spectrum_data.flux.copy()

        if self._locked_fits:
            total_locked = self.np.zeros_like(residual_flux)
            for fit_result in self._locked_fits:
                model = fit_result.get("model")
                if model is not None:
                    total_locked += model(self._current_spectrum_data.wv)
            residual_flux = residual_flux - total_locked

        return residual_flux

    def _on_fit_clicked(self, event):
        if self._current_spectrum_data is None:
            self._set_status("No spectrum loaded.", level="warning", visible=True)
            return

        try:
            if not self._regions.is_valid():
                raise self.AnalysisError("Invalid region definitions.")

            continuum_model = self._estimate_continuum(self._current_spectrum_data)
            continuum = continuum_model(self._current_spectrum_data.wv)
            corrected_flux = self._current_spectrum_data.flux - continuum

            self._current_spectrum_data.continuum = continuum
            self._current_spectrum_data.corrected_flux = corrected_flux

            self._continuum_model = continuum_model
            self._continuum = continuum
            self._corrected_flux = corrected_flux

            self.continuum_source.data = {
                "wavelength": self._current_spectrum_data.wv.tolist(),
                "flux": continuum.tolist(),
            }
            self.corrected_source.data = {
                "wavelength": self._current_spectrum_data.wv.tolist(),
                "flux": corrected_flux.tolist(),
            }

            results = self._analyser.analyse(
                self._current_spectrum_data,
                self._regions,
                self.line_profile_selector.value,
                continuum_model,
                corrected_flux,
            )
            self._analysis_results = results

            mask_signal = (
                (self._current_spectrum_data.wv >= self._regions.signal_start) &
                (self._current_spectrum_data.wv <= self._regions.signal_end)
            )
            x_fit = self._current_spectrum_data.wv[mask_signal]
            y_fit = results["model"](x_fit)

            self.fit_source.data = {
                "wavelength": x_fit.tolist(),
                "flux": y_fit.tolist(),
            }

            lock_record = dict(results)
            lock_record["line_name"] = self.line_name_input.value or "Unknown"
            lock_record["comment"] = self.comments_input.value
            lock_record["z"] = float(self.redshift_slider.value)
            lock_record["classification"] = "locked_fit"
            lock_record["model_name"] = self.line_profile_selector.value
            self._locked_fits.append(lock_record)

            self._results_manager.add_result({
                "source_id": results.get("source_id"),
                "classification": "locked_fit",
                "z": float(self.redshift_slider.value),
                "flux_integral": results.get("flux_integral"),
                "snr": results.get("snr"),
                "fwhm_obs_A": results.get("fwhm_obs_A"),
                "fwhm_int_A": results.get("fwhm_int_A"),
                "fwhm_kms": results.get("fwhm_kms"),
                "ew": results.get("ew"),
                "chi_squared_red": results.get("chi_squared_red"),
                "signal_range": results.get("signal_range"),
                "noise_range": results.get("noise_range"),
                "comment": self.comments_input.value,
                "line_name": self.line_name_input.value or "Unknown",
            })

            self._rebuild_locked_fit_source()

            residual_flux = self._build_residual_flux()
            self.residuals_source.data = {
                "wavelength": self._current_spectrum_data.wv.tolist(),
                "flux": residual_flux.tolist(),
            }

            self._update_results_table(results)
            self._clear_fit_status()
            self._set_status(
                f"Fit locked successfully. "
                f"SNR={results.get('snr', 0):.2f}, "
                f"FWHM={results.get('fwhm_obs_A', 0):.2f} Å, "
                f"χ²ᵣ={results.get('chi_squared_red', 0):.2f}",
                level="success",
                visible=True,
            )

        except Exception as e:
            self.logging.warning("Fitting failed: %s", e)
            self._analysis_results = {
                "fit_failed": True,
                "fail_reason": str(e),
            }
            self.fit_source.data = {"wavelength": [], "flux": []}
            self._update_results_table(None)
            self._show_fit_status(f"FIT FAILED ({e})")
            self._set_status(f"Fit failed: {e}", level="danger", visible=True)

    def _on_reset_clicked(self, event):
        if self._current_spectrum_data is None:
            return

        empty = {"wavelength": [], "flux": []}
        self.continuum_source.data = empty
        self.corrected_source.data = empty
        self.fit_source.data = empty
        self.locked_fit_source.data = empty

        self.residuals_source.data = {
            "wavelength": self._current_spectrum_data.wv.tolist(),
            "flux": self._current_spectrum_data.flux.tolist(),
        }

        self._analysis_results = None
        self._continuum_model = None
        self._continuum = None
        self._corrected_flux = None
        self._locked_fits = []

        self._signal_region_defined = False
        self._noise_region_defined = False
        self._update_region_overlays()

        self._update_results_table(None)
        self._clear_fit_status()
        self._set_status("Fit state reset.", level="info", visible=True)

    def _on_undo_lock_clicked(self, event):
        if not self._locked_fits:
            self._set_status("No locked fits to undo.", level="warning", visible=True)
            return

        self._locked_fits.pop()
        self._results_manager.undo_last()

        self._rebuild_locked_fit_source()

        if self._current_spectrum_data is not None:
            residual_flux = self._build_residual_flux()
            self.residuals_source.data = {
                "wavelength": self._current_spectrum_data.wv.tolist(),
                "flux": residual_flux.tolist(),
            }

        self._set_status("Removed last locked fit.", level="info", visible=True)

    # ------------------------------------------------------------------
    # Layout / lifecycle
    # ------------------------------------------------------------------
    def get_layout(self):
        sidebar = self.pn.Column(
            self.analysis_tab,
            width=360,
            min_width=360,
            max_width=360,
            height=700,
            sizing_mode="fixed",
            align="start",
            margin=(0, 0, 0, 12),
        )

        spectra_header = pn.Column(
            pn.Spacer(height=10),
            pn.pane.HTML(
                "<div style='font-weight:600; margin-bottom:4px;'>Available Spectra</div>",
                height=20,
                margin=(0, 0, 2, 0),
                sizing_mode="fixed",
            ),
            self.available_spectra,
            self.spectra_number_message,
            width=360,
            height=78,
            sizing_mode="fixed",
            margin=(0, 0, 6, 0),
        )

        main_area = self.pn.Column(
            spectra_header,
            self.source_plot,
            pn.Spacer(height=6),
            self.residuals_plot,
            sizing_mode="stretch_width",
            min_width=520,
            align="start",
        )

        return self.pn.Row(
            main_area,
            sidebar,
            sizing_mode="stretch_width",
            align="start",
        )

    def dispose(self) -> None:
        try:
            if hasattr(self, "_cb") and self._cb:
                self._cb.stop()
        except Exception:
            pass

        super().dispose()
###

from astronomicAL.extensions.samp_bridge import SAMPBridge


_SECTION_STYLE = {
    "padding": "12px 14px",
    "border": "1px solid #d9d9d9",
    "border-radius": "8px",
    "background": "#fafafa",
    "box-sizing": "border-box",
    "min-width": "0",
}

_STATUS_STYLE = {
    "padding": "10px 12px",
    "border": "1px solid #d9d9d9",
    "border-radius": "8px",
    "background": "#ffffff",
    "box-sizing": "border-box",
    "min-width": "0",
}

def _fit_block(*objects):
    return pn.FlexBox(
        *objects,
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles={
            "min-width": "0",
            "gap": "0px",
        },
    )

def _section(title: str, *objects):
    return pn.FlexBox(
        pn.pane.HTML(
            f"<div style='font-weight:600; margin-bottom:10px;'>{title}</div>",
            sizing_mode="stretch_width",
        ),
        _fit_block(*objects),
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles=dict(_SECTION_STYLE),
        margin=(0, 0, 12, 0),
    )

def _status_block(status_pane):
    return pn.FlexBox(
        pn.pane.HTML(
            "<div style='font-weight:600; margin-bottom:10px;'>Status</div>",
            sizing_mode="stretch_width",
        ),
        _fit_block(status_pane),
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_width",
        styles=dict(_STATUS_STYLE),
        margin=(0, 0, 12, 0),
    )

def _root_column(*objects):
    return pn.FlexBox(
        *objects,
        flex_direction="column",
        flex_wrap="nowrap",
        justify_content="flex-start",
        align_items="stretch",
        sizing_mode="stretch_both",
        styles={
            "overflow-x": "hidden",
            "overflow-y": "auto",
            "min-width": "0",
            "min-height": "0",
            "gap": "0px",
        },
    )

def _text_input(name: str, value: str = ""):
    return pn.widgets.TextInput(
        name=name,
        value=value,
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )


def _select(name: str, options=None, value=None, size=None):
    kwargs = dict(
        name=name,
        options=options or {},
        value=value,
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )
    if size is not None:
        kwargs["size"] = size
    return pn.widgets.Select(**kwargs)


def _int_input(name: str, value: int, start: int = 0, step: int = 1):
    return pn.widgets.IntInput(
        name=name,
        value=value,
        start=start,
        step=step,
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )


def _multichoice(name: str, options=None, value=None, height: int = 120):
    return pn.widgets.MultiChoice(
        name=name,
        options=options or [],
        value=value or [],
        sizing_mode="stretch_width",
        height=height,
        margin=(0, 0, 10, 0),
    )

def _multiselect(name: str, options=None, value=None, size: int = 10):
    return pn.widgets.MultiSelect(
        name=name,
        options=options or [],
        value=value or [],
        size=size,
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )

class SampSendPanel(CustomPlotClass):
    def __init__(self, data, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            close_button=close_button,
            extra_features=[],
            panel_name="samp_send_panel",
            ready_stage="plot",
            context=context,
            require_settings=False,
            **params,
        )

        self.services = getattr(context, "services", None)

        self.status = pn.pane.Markdown("Ready.")
        self.summary = pn.pane.Markdown("")

        self.dataset_select = _select("Dataset", options={})
        self.table_name = _text_input("Table name", value="AstronomicAL table")

        self.target_mode = _select(
            "Send target",
            options={
                "TOPCAT": "topcat",
                "All SAMP clients": "all",
                "Specific client": "client",
            },
            value="topcat",
        )

        self.client_select = _select("Specific client", options={})
        self.refresh_clients_button = pn.widgets.Button(
            name="Refresh clients",
            button_type="default",
            sizing_mode="stretch_width",
            height=40,
            margin=(0, 0, 10, 0),
        )

        self.all_columns = pn.widgets.Checkbox(
            name="Send all columns",
            value=True,
            margin=(0, 0, 10, 0),
        )

        self.column_help = pn.pane.Markdown(
            "To choose specific columns: hold **Ctrl** (or **Cmd** on Mac) to add/remove individual columns, "
            "or hold **Shift** to select a continuous range.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

        self.column_select = _multiselect("Columns", options=[], value=[], size=10)        
        
        self.row_limit = _int_input("Row limit (0 = all)", value=0, start=0)

        self.send_button = pn.widgets.Button(
            name="Send to SAMP",
            button_type="primary",
            sizing_mode="stretch_width",
            height=44,
            margin=(0, 0, 0, 0),
        )

        self.client_block = _fit_block(self.client_select)
        self.columns_block = _fit_block(self.column_help, self.column_select)

        self.add_param_watch(self.dataset_select, self._on_dataset_changed, "value")
        self.add_param_watch(self.target_mode, self._on_target_changed, "value")
        self.add_param_watch(self.all_columns, self._on_all_columns_changed, "value")

        self.refresh_clients_button.on_click(self._refresh_clients_clicked)
        self.send_button.on_click(self._send_clicked)

        self.subscribe("dataset.loaded", self._on_dataset_event)
        self.subscribe("dataset.active.changed", self._on_dataset_event)

        self._refresh_dataset_options()
        self._refresh_client_options()
        self._on_target_changed(None)
        self._on_all_columns_changed(None)

    def _ensure_samp_service(self) -> SAMPBridge:
        if self.services is None:
            raise RuntimeError("No ServiceRegistry available on context.")

        if not self.services.has("interop.samp"):
            self.services.set("interop.samp", SAMPBridge(client_name="AstronomicAL"))

        bridge = self.services.get("interop.samp")
        bridge.start()
        return bridge

    def _dataset_options(self) -> dict[str, str]:
        if self.datasets is None:
            return {}

        options = {}
        for dataset_id in self.datasets.list_ids():
            dataset = self.datasets.get(dataset_id)
            dataset_name = getattr(dataset, "name", None) or dataset_id
            options[f"{dataset_name} ({dataset_id})"] = dataset_id
        return options

    def _refresh_dataset_options(self) -> None:
        options = self._dataset_options()
        self.dataset_select.options = options

        if not options:
            self.dataset_select.value = None
            self.summary.object = "No datasets available."
            self.column_select.options = []
            self.column_select.value = []
            return

        if self.dataset_select.value not in options.values():
            try:
                active_id = self.datasets.active_id()
                self.dataset_select.value = active_id if active_id in options.values() else next(iter(options.values()))
            except Exception:
                self.dataset_select.value = next(iter(options.values()))

        self._refresh_column_options()
        self._refresh_summary()

    def _refresh_column_options(self) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.column_select.options = []
            self.column_select.value = []
            return

        df = self.datasets.get_df(dataset_id)
        cols = list(df.columns)
        self.column_select.options = cols

        if self.all_columns.value:
            self.column_select.value = cols
        else:
            self.column_select.value = [c for c in self.column_select.value if c in cols]

    def _refresh_summary(self) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.summary.object = "No dataset selected."
            return

        df = self.datasets.get_df(dataset_id)
        self.summary.object = (
            f"**Rows:** {len(df):,}  \n"
            f"**Columns:** {len(df.columns):,}  \n"
            f"**Dataset id:** `{dataset_id}`"
        )

        if not self.table_name.value.strip():
            self.table_name.value = dataset_id

    def _refresh_client_options(self) -> None:
        try:
            bridge = self._ensure_samp_service()
            clients = bridge.list_clients()
            options = {f"{c['name']} ({c['id']})": c["id"] for c in clients}
            self.client_select.options = options

            if options and self.client_select.value not in options.values():
                self.client_select.value = next(iter(options.values()))

            self.status.object = f"Found {len(clients)} SAMP client(s)."
        except Exception as exc:
            self.client_select.options = {}
            self.status.object = f"Failed to list SAMP clients: `{exc}`"

    def _on_dataset_event(self, _topic, _payload) -> None:
        self._refresh_dataset_options()

    def _on_dataset_changed(self, _event) -> None:
        self._refresh_column_options()
        self._refresh_summary()

    def _on_target_changed(self, _event) -> None:
        self.client_block.visible = self.target_mode.value == "client"

    def _on_all_columns_changed(self, _event) -> None:
        self.columns_block.visible = not bool(self.all_columns.value)
        if self.all_columns.value:
            self._refresh_column_options()

    def _refresh_clients_clicked(self, _event) -> None:
        self.status.object = "Refreshing SAMP clients..."
        self._refresh_client_options()

    def _send_clicked(self, _event) -> None:
        dataset_id = self.dataset_select.value
        if not dataset_id:
            self.status.object = "No dataset selected."
            return

        selected_columns = None
        if not self.all_columns.value:
            selected_columns = list(self.column_select.value)
            if not selected_columns:
                self.status.object = "Pick at least one column to send."
                return

        self.send_button.disabled = True
        self.status.object = "Sending table..."

        self.submit_job(
            self._send_job,
            title="Send table over SAMP",
            key=f"{self.panel_id}:send:{uuid.uuid4().hex}",
            on_done=self._on_send_done,
            on_error=self._on_error,
            dataset_id=dataset_id,
            table_name=self.table_name.value.strip() or dataset_id,
            target_mode=self.target_mode.value,
            target_client_id=self.client_select.value,
            selected_columns=selected_columns,
            row_limit=int(self.row_limit.value or 0),
        )

    def _send_job(
        self,
        *,
        cancel_token,
        dataset_id: str,
        table_name: str,
        target_mode: str,
        target_client_id: str | None,
        selected_columns: list[str] | None,
        row_limit: int,
    ):
        if cancel_token and cancel_token.cancelled():
            return None

        df = self.datasets.get_df(dataset_id)

        if selected_columns is not None:
            df = df.loc[:, selected_columns]

        if row_limit > 0:
            df = df.head(row_limit)

        bridge = self._ensure_samp_service()
        result = bridge.send_dataframe(
            df,
            table_name=table_name,
            target_mode=target_mode,
            target_client_id=target_client_id,
        )
        result["dataset_id"] = dataset_id
        result["columns"] = list(df.columns)
        return result

    def _on_send_done(self, result) -> None:
        self.send_button.disabled = False

        if result is None:
            self.status.object = "Send cancelled."
            return

        artifact_id = self.put_artifact(
            "interop.samp.export",
            result,
            dataset_id=result["dataset_id"],
            params={
                "target_mode": self.target_mode.value,
                "mtype": "table.load.votable",
            },
        )

        if artifact_id:
            self.publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": "interop.samp.export",
                    "dataset_id": result["dataset_id"],
                },
            )

        self.publish(
            "interop.samp.table.sent",
            {
                "dataset_id": result["dataset_id"],
                "artifact_id": artifact_id,
                "table_name": result["table_name"],
                "target_mode": self.target_mode.value,
            },
        )

        self.status.object = (
            f"Sent **{result['table_name']}** "
            f"({result['row_count']:,} rows, {result['column_count']:,} columns)."
        )

    def _on_error(self, exc: BaseException) -> None:
        self.send_button.disabled = False
        self.status.object = f"Failed: `{exc}`"

    def get_layout(self):
        return _root_column(
            _section("Dataset summary", self.summary),
            _section("Table selection", self.dataset_select, self.table_name),
            _section("Target client", self.target_mode, self.client_block, self.refresh_clients_button),
            _section("Export options", self.all_columns, self.row_limit, self.columns_block),
            _section("Send", self.send_button),
            _status_block(self.status),
        )


class SampReceivePanel(CustomPlotClass):
    def __init__(self, data, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            close_button=close_button,
            extra_features=[],
            panel_name="samp_receive_panel",
            ready_stage="plot",
            context=context,
            require_settings=False,
            **params,
        )

        self.services = getattr(context, "services", None)
        self._listener_token: str | None = None
        self._receive_count = 0
        self._received_items: list[dict[str, Any]] = []

        self.status = pn.pane.Markdown("Listening for incoming SAMP tables.")
        self.summary = pn.pane.Markdown("No received table selected.")
        self.inbox_info = pn.pane.Markdown("No received tables yet.")
        self.empty_state = pn.pane.Markdown(
            "No received tables yet. Send a table from TOPCAT or another SAMP client."
        )
        
        self.inbox_select = _select("Received tables", options={}, size=8)

        self.preview_rows = _int_input("Preview rows", value=20, start=1, step=5)
        self.preview_columns = _multichoice("Preview columns", options=[], value=[], height=100)

        self.dataset_name = _text_input("Dataset name", value="")
        self.dataset_id = _text_input("Dataset id", value="")
        self.make_active = pn.widgets.Checkbox(
            name="Make active after import",
            value=True,
            margin=(0, 0, 10, 0),
        )

        self.register_button = pn.widgets.Button(
            name="Register as dataset",
            button_type="primary",
            sizing_mode="stretch_width",
            height=40,
            margin=(0, 0, 10, 0),
        )

        self.activate_button = pn.widgets.Button(
            name="Make selected dataset active",
            button_type="default",
            sizing_mode="stretch_width",
            height=40,
            margin=(0, 0, 10, 0),
        )

        self.discard_button = pn.widgets.Button(
            name="Discard selected",
            button_type="warning",
            sizing_mode="stretch_width",
            height=40,
            margin=(0, 0, 10, 0),
        )

        self.clear_button = pn.widgets.Button(
            name="Clear inbox",
            button_type="default",
            sizing_mode="stretch_width",
            height=40,
            margin=(0, 0, 0, 0),
        )

        self.preview = pn.pane.DataFrame(
            pd.DataFrame(),
            index=False,
            sizing_mode="stretch_width",
            height=180,
            max_height=180,
        )

        self.inbox_body = _fit_block()

        self.add_param_watch(self.inbox_select, self._on_inbox_changed, "value")
        self.add_param_watch(self.preview_rows, self._on_preview_control_changed, "value")
        self.add_param_watch(self.preview_columns, self._on_preview_control_changed, "value")

        self.register_button.on_click(self._register_clicked)
        self.activate_button.on_click(self._activate_clicked)
        self.discard_button.on_click(self._discard_clicked)
        self.clear_button.on_click(self._clear_clicked)

        self._attach_listener()
        self._refresh_inbox_body()
        self._update_selection_view()

    def _ensure_samp_service(self) -> SAMPBridge:
        if self.services is None:
            raise RuntimeError("No ServiceRegistry available on context.")

        if not self.services.has("interop.samp"):
            self.services.set("interop.samp", SAMPBridge(client_name="AstronomicAL"))

        bridge = self.services.get("interop.samp")
        bridge.start()
        return bridge

    def _attach_listener(self) -> None:
        bridge = self._ensure_samp_service()
        if self._listener_token is None:
            self._listener_token = bridge.add_table_listener(self._on_table_received)

    def _detach_listener(self) -> None:
        if self._listener_token is None:
            return

        try:
            bridge = self._ensure_samp_service()
            bridge.remove_table_listener(self._listener_token)
        finally:
            self._listener_token = None

    def _run_on_ui_thread(self, fn) -> None:
        doc = getattr(pn.state, "curdoc", None)
        if doc is not None:
            doc.add_next_tick_callback(fn)
        else:
            fn()

    def _on_table_received(self, payload: dict[str, Any]) -> None:
        self._run_on_ui_thread(lambda: self._handle_received_table(payload))

    def _handle_received_table(self, payload: dict[str, Any]) -> None:
        self._receive_count += 1

        artifact_id = self.put_artifact(
            "interop.samp.import",
            payload,
            dataset_id=None,
            params={
                "mtype": payload.get("mtype"),
                "sender_id": payload.get("sender_id"),
                "url": payload.get("url"),
            },
        )

        item_id = f"samp-recv-{uuid.uuid4().hex[:10]}"
        item = {
            "id": item_id,
            "name": payload.get("name") or f"Incoming SAMP table {self._receive_count}",
            "sender_id": payload.get("sender_id") or "unknown",
            "artifact_id": artifact_id,
            "dataset_id": None,
            "payload": payload,
            "dataframe": payload["dataframe"],
            "row_count": payload.get("row_count", 0),
            "column_count": payload.get("column_count", 0),
            "columns": list(payload.get("columns", [])),
            "url": payload.get("url"),
            "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._received_items.insert(0, item)

        if artifact_id:
            self.publish(
                "artifact.created",
                {
                    "artifact_id": artifact_id,
                    "type": "interop.samp.import",
                    "dataset_id": None,
                },
            )

        self.publish(
            "interop.samp.table.received",
            {
                "artifact_id": artifact_id,
                "dataset_id": None,
                "table_name": item["name"],
                "sender_id": item["sender_id"],
                "row_count": item["row_count"],
                "column_count": item["column_count"],
            },
        )

        self._refresh_inbox_options(select_item_id=item_id)
        self.status.object = (
            f"Received **{item['name']}** from `{item['sender_id']}` "
            f"({item['row_count']:,} rows, {item['column_count']:,} columns)."
        )

    def _refresh_inbox_body(self) -> None:
        self.inbox_body.objects = [self.empty_state] if not self._received_items else [self.inbox_select]

    def _refresh_inbox_options(self, select_item_id: str | None = None) -> None:
        options = {}

        for item in self._received_items:
            dataset_part = f" → dataset `{item['dataset_id']}`" if item["dataset_id"] else ""
            label = (
                f"{item['name']} | {item['row_count']:,}x{item['column_count']:,} | "
                f"{item['sender_id']} | {item['received_at']}{dataset_part}"
            )
            options[label] = item["id"]

        self.inbox_select.options = options
        self.inbox_info.object = (
            f"**Received tables:** {len(self._received_items)}"
            if options
            else "No received tables yet."
        )

        if not options:
            self.inbox_select.value = None
        elif select_item_id is not None:
            self.inbox_select.value = select_item_id
        elif self.inbox_select.value not in options.values():
            self.inbox_select.value = next(iter(options.values()))

        self._refresh_inbox_body()
        self._update_selection_view()

    def _get_selected_item(self) -> dict[str, Any] | None:
        item_id = self.inbox_select.value
        if not item_id:
            return None

        for item in self._received_items:
            if item["id"] == item_id:
                return item

        return None

    def _build_default_dataset_name(self, item: dict[str, Any]) -> str:
        return item["name"] or f"SAMP import {self._receive_count}"

    def _build_default_dataset_id(self, item: dict[str, Any]) -> str:
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", self._build_default_dataset_name(item).strip()).strip("-").lower()
        base = base or "samp-import"
        return f"{base}-{item['id'][-4:]}"

    def _update_selection_view(self) -> None:
        item = self._get_selected_item()

        if item is None:
            self.summary.object = "No received table selected."
            self.preview.object = pd.DataFrame()
            self.preview_columns.options = []
            self.preview_columns.value = []
            self.dataset_name.value = ""
            self.dataset_id.value = ""
            self.register_button.disabled = True
            self.activate_button.disabled = True
            self.discard_button.disabled = True
            self.clear_button.disabled = len(self._received_items) == 0
            return

        self.register_button.disabled = item["dataset_id"] is not None
        self.activate_button.disabled = item["dataset_id"] is None
        self.discard_button.disabled = False
        self.clear_button.disabled = False

        if not self.dataset_name.value:
            self.dataset_name.value = self._build_default_dataset_name(item)

        if not self.dataset_id.value:
            self.dataset_id.value = self._build_default_dataset_id(item)

        cols = list(item["columns"])
        self.preview_columns.options = cols
        self.preview_columns.value = [c for c in self.preview_columns.value if c in cols]

        dataset_line = (
            f"**Registered dataset:** `{item['dataset_id']}`"
            if item["dataset_id"] is not None
            else "**Registered dataset:** not yet imported"
        )

        self.summary.object = (
            f"**Table:** {item['name']}  \n"
            f"**Sender:** `{item['sender_id']}`  \n"
            f"**Received:** {item['received_at']}  \n"
            f"**Rows:** {item['row_count']:,}  \n"
            f"**Columns:** {item['column_count']:,}  \n"
            f"**Artifact id:** `{item['artifact_id']}`  \n"
            f"{dataset_line}  \n"
            f"**Source URL:** `{item['url']}`"
        )

        self._update_preview()

    def _update_preview(self) -> None:
        item = self._get_selected_item()
        if item is None:
            self.preview.object = pd.DataFrame()
            return

        df = item["dataframe"]

        cols = list(self.preview_columns.value)
        if cols:
            df = df.loc[:, cols]

        self.preview.object = df.head(max(1, int(self.preview_rows.value or 20)))

    def _on_inbox_changed(self, _event) -> None:
        self.dataset_name.value = ""
        self.dataset_id.value = ""
        self._update_selection_view()

    def _on_preview_control_changed(self, _event) -> None:
        self._update_preview()

    def _register_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None:
            self.status.object = "No received table selected."
            return

        if item["dataset_id"] is not None:
            self.status.object = f"Selected table is already registered as `{item['dataset_id']}`."
            return

        dataset_name = self.dataset_name.value.strip() or self._build_default_dataset_name(item)
        dataset_id = self.dataset_id.value.strip() or self._build_default_dataset_id(item)

        existing_ids = set(self.datasets.list_ids())
        if dataset_id in existing_ids:
            self.status.object = f"Dataset id `{dataset_id}` already exists. Choose a different id."
            return

        self.datasets.register(
            dataset_id,
            item["dataframe"],
            name=dataset_name,
            source=item["url"],
            domain="interop.samp",
        )

        item["dataset_id"] = dataset_id

        self.publish(
            "dataset.loaded",
            {
                "dataset_id": dataset_id,
                "source": "interop.samp",
            },
        )

        if self.make_active.value:
            self.datasets.set_active(dataset_id)
            self.publish(
                "dataset.active.changed",
                {
                    "dataset_id": dataset_id,
                    "source": "interop.samp",
                },
            )

        self.publish(
            "interop.samp.table.imported",
            {
                "artifact_id": item["artifact_id"],
                "dataset_id": dataset_id,
                "table_name": item["name"],
                "sender_id": item["sender_id"],
            },
        )

        self.status.object = (
            f"Imported **{item['name']}** as dataset `{dataset_id}`."
            + (" It is now active." if self.make_active.value else "")
        )

        self._refresh_inbox_options(select_item_id=item["id"])

    def _activate_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None or item["dataset_id"] is None:
            self.status.object = "Selected table has not been registered as a dataset yet."
            return

        self.datasets.set_active(item["dataset_id"])
        self.publish(
            "dataset.active.changed",
            {
                "dataset_id": item["dataset_id"],
                "source": "interop.samp",
            },
        )

        self.status.object = f"Set dataset `{item['dataset_id']}` as active."

    def _discard_clicked(self, _event) -> None:
        item = self._get_selected_item()
        if item is None:
            self.status.object = "No received table selected."
            return

        self._received_items = [x for x in self._received_items if x["id"] != item["id"]]
        self.status.object = f"Discarded received table **{item['name']}**."
        self._refresh_inbox_options()

    def _clear_clicked(self, _event) -> None:
        self._received_items = []
        self.status.object = "Cleared received-table inbox."
        self._refresh_inbox_options()

    def get_layout(self):
        return _root_column(
            _section("Received tables", self.inbox_info, self.inbox_body),
            _section("Inbox actions", self.discard_button, self.clear_button),
            _section("Selected table", self.summary),
            _section("Preview", self.preview_rows, self.preview_columns, self.preview),
            _section("Import actions", self.dataset_name, self.dataset_id, self.make_active, self.register_button, self.activate_button),
            _status_block(self.status),
        )

    def dispose(self):
        self._detach_listener()
        try:
            super().dispose()
        except AttributeError:
            pass