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
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass, EuclidSpectraClass
from astronomicAL.extensions.astro_data_utility import VLASS_cutout, LoTSS_cutout, make_srcdoc_aladin_lite, SDSS_cutout


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
        "Euclid Cutout": lambda data, close_button, context: EuclidPlotClass(
            data,
            close_button,
            extra_features=[],
            context=context,
        ),
        "DESI Spectra": lambda data, close_button, context: SpectrumPlotClass(
            data,
            close_button,
            extra_features=[],
            dataset="DESI",
            context=context,
        ),
        "Euclid Spectra": lambda data, close_button, context: SpectrumPlotClass(
            data,
            close_button,
            extra_features=[],
            dataset="EuclidSpec",
            context=context,
        ),
        "SDSS Spectra": lambda data, close_button, context: SpectrumPlotClass(
            data,
            close_button,
            extra_features=[],
            dataset="SDSS",
            context=context,
        ),
        "BroadBand SED": lambda data, close_button, context: SEDPlotClass(
            data,
            close_button,
            extra_features=[],
            context=context,
        ),
        "Aladin Lite": lambda data, close_button, context: AladinClass(
            data,
            close_button,
            extra_features=[],
            context=context,
        ),
        "VLASS Cutout": lambda data, close_button, context: RadioClass(
            data,
            close_button,
            extra_features=[],
            dataset="VLASS",
            context=context,
        ),
        "LoTSS Cutout": lambda data, close_button, context: RadioClass(
            data,
            close_button,
            extra_features=[],
            dataset="LoTSS",
            context=context,
        ),
        "spec_analyser": lambda data, close_button, context: SpecAnalyser(
            data,
            close_button,
            extra_features=[],
            context=context,
            require_settings=False,
        ),
        "SAMP Send": lambda data, close_button, context: SampSendPanel(
            data,
            close_button,
            extra_features=[],
            context=context,
        ),
        "SAMP Receive": lambda data, close_button, context: SampReceivePanel(
            data,
            close_button,
            extra_features=[],
            context=context,
        ),
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

class EuclidPlotClass(CustomPlotClass):
    def __init__(self, data, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            close_button=close_button,
            extra_features=extra_features or [],
            panel_name="Euclid_Cutout",
            context=context,
            **params,
        )

        self._mapping_requests_sent = set()
        self.euclid_object = None

        self._widgets_initialised = False

        self._image_stream_watchers = []
        self._profile_stream_watchers = []

        self._initialize_settings_dictionary()

        self.filter = self._get_from_settings_dictionary("filter", "Color")
        self.radius = self._get_from_settings_dictionary("radius", 5.0)

    # ------------------------------------------------------------------
    # Refresh / event handling
    # ------------------------------------------------------------------

    def _clear_image_stream_watchers(self):
        for stream, watcher in list(getattr(self, "_image_stream_watchers", [])):
            try:
                stream.param.unwatch(watcher)
            except Exception:
                pass
        self._image_stream_watchers = []


    def _watch_image_stream(self, stream, callback, what="x"):
        watcher = stream.param.watch(callback, what)
        self._image_stream_watchers.append((stream, watcher))
        return watcher

    def _clear_profile_stream_watchers(self):
        for stream, watcher in list(getattr(self, "_profile_stream_watchers", [])):
            try:
                stream.param.unwatch(watcher)
            except Exception:
                pass
        self._profile_stream_watchers = []


    def _watch_profile_stream(self, stream, callback, what="x"):
        watcher = stream.param.watch(callback, what)
        self._profile_stream_watchers.append((stream, watcher))
        return watcher

    def _refresh_cutout(self, reason=None, verbose=False):
        """
        Backward-compatible helper.

        Older Euclid code may still call this directly. Route everything through the
        base-class refresh scheduler so dataset events, src.data changes, and manual
        refreshes all coalesce cleanly.
        """
        self._request_refresh(reason=reason or "euclid.refresh", verbose=verbose)

    def _dataset_mapping_updated_cb(self, topic, payload):
        if not self._event_matches_active_dataset(payload):
            return

        if isinstance(payload, dict):
            semantic_name = payload.get("semantic_name")
            column_name = payload.get("column_name")
            config_key = payload.get("config_key")

            if semantic_name is not None and column_name is not None and config_key:
                try:
                    self.config.settings[config_key] = column_name
                except Exception:
                    pass

        self._request_refresh(
            reason=str(topic or "dataset.mapping_updated"),
            payload=payload,
        )

    def _build_refresh_signature(self, reason=None, payload=None):
        """
        Euclid refreshes depend on more than dataset + selected source.

        Include filter/radius/stretching so repeated UI actions with the same state
        do not launch duplicate jobs, while real changes still trigger a new fetch.
        """
        dataset_id = self._dataset_id()
        selected_id = self._get_selected_id()

        filter_value = self.filter
        radius_value = self.radius
        stretching_value = self._get_from_settings_dictionary("stretching", "Linear")

        if getattr(self, "_widgets_initialised", False):
            try:
                filter_value = self.filter_input.value
            except Exception:
                pass

            try:
                radius_value = self.radius_input.value
            except Exception:
                pass

            try:
                stretching_value = self.stretching_input.value
            except Exception:
                pass

        return (
            str(dataset_id),
            None if selected_id is None else str(selected_id),
            str(filter_value),
            float(radius_value) if radius_value is not None else None,
            str(stretching_value),
        )

    def _perform_refresh(self, reason=None, payload=None, refresh_signature=None):
        """
        Euclid-specific refresh entrypoint used by the base-class scheduler.
        """
        try:
            initialised = self._initialise_euclid_object()
            self.stored_spectrum_coordinates = {}

            if not initialised:
                self._finish_refresh(refresh_signature)
                return

            if not self._widgets_initialised:
                self._finish_refresh(refresh_signature)
                return

            self._run_euclid(
                reason=reason,
                refresh_signature=refresh_signature,
            )

        except Exception:
            traceback.print_exc()
            self._finish_refresh(refresh_signature)

    def _subscribe_to_mapping_and_dataset_events(self):
        """
        Euclid uses dataset activation/update events plus mapping updates.

        We intentionally do not subscribe to dataset.loaded here because viewer-style
        panels should react to the active dataset, not to every load event.
        """
        self._bind_dataset_runtime_subscriptions(
            include_loaded=False,
            include_mapping=True,
        )

    # ------------------------------------------------------------------
    # Dataset / mapping helpers
    # ------------------------------------------------------------------

    def _dataset_id(self) -> str:
        if getattr(self, "context", None) is not None and getattr(self.context, "datasets", None) is not None:
            try:
                active = self.context.datasets.active_id()
                if active:
                    return active
            except Exception:
                pass
        return "main"

    def _guess_column(self, names):
        lowered = {str(col).lower(): col for col in self.df.columns}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    def _euclid_mapping_specs(self):
        columns = list(self.df.columns)

        return [
            {
                "semantic_name": "record_id",
                "config_key": "id_col",
                "display_name": "ID column",
                "description": "Needed by Euclid Cutout to resolve the currently selected source in the active dataset.",
                "required": True,
                "candidates": ["Use Index"] + columns,
                "suggested": self._guess_column(["source_id", "id", "objid", "object_id"]) or "Use Index",
            },
            {
                "semantic_name": "coords.ra",
                "config_key": "ra_col_name",
                "display_name": "RA column",
                "description": "Needed by Euclid Cutout to locate the currently selected source.",
                "required": True,
                "candidates": columns,
                "suggested": self._guess_column(["ra", "raj2000", "ra_deg", "right_ascension"]),
            },
            {
                "semantic_name": "coords.dec",
                "config_key": "dec_col_name",
                "display_name": "DEC column",
                "description": "Needed by Euclid Cutout to locate the currently selected source.",
                "required": True,
                "candidates": columns,
                "suggested": self._guess_column(["dec", "dej2000", "dec_deg", "declination"]),
            },
        ]

    def _sync_config_from_dataset_mappings(self) -> None:
        if getattr(self, "context", None) is None or getattr(self.context, "datasets", None) is None:
            return

        dataset_id = self._dataset_id()

        for spec in self._euclid_mapping_specs():
            semantic_name = spec["semantic_name"]
            config_key = spec["config_key"]

            mapped = self.context.datasets.get_mapping(dataset_id, semantic_name)
            existing = self.config.settings.get(config_key)

            if mapped is None and existing in spec["candidates"]:
                self.context.datasets.set_mapping(dataset_id, semantic_name, existing)
                mapped = existing

            if mapped is not None:
                self.config.settings[config_key] = mapped

    def _publish_mapping_request(self, spec):
        if getattr(self, "context", None) is None or getattr(self.context, "events", None) is None:
            return

        key = (self._dataset_id(), spec["semantic_name"])
        if key in self._mapping_requests_sent:
            return

        payload = {
            "source": "euclid_cutout",
            "panel_id": self.panel_id,
            "dataset_id": self._dataset_id(),
            "semantic_name": spec["semantic_name"],
            "display_name": spec["display_name"],
            "description": spec["description"],
            "required": spec["required"],
            "config_key": spec["config_key"],
            "candidates": spec["candidates"],
            "suggested": spec["suggested"],
        }

        self.context.events.publish("mapping.requested", payload)
        self._mapping_requests_sent.add(key)

    def _request_missing_mappings(self) -> bool:
        self._sync_config_from_dataset_mappings()

        if getattr(self, "context", None) is None or getattr(self.context, "datasets", None) is None:
            return False

        missing_required = False
        dataset_id = self._dataset_id()

        for spec in self._euclid_mapping_specs():
            mapped = self.context.datasets.get_mapping(dataset_id, spec["semantic_name"])
            if mapped is None:
                self._publish_mapping_request(spec)
                if spec["required"]:
                    missing_required = True

        return missing_required

    def _refresh_df_from_active_dataset(self):
        if getattr(self, "context", None) is not None and getattr(self.context, "datasets", None) is not None:
            try:
                self.df = self.context.datasets.get_df(self._dataset_id()).copy()
                if getattr(self, "config", None) is not None:
                    self.config.main_df = self.df
                return
            except Exception:
                pass

        self.df = self.config.main_df.copy()

    def _get_selected_row_from_active_df(self):
        selected_id = self._get_selected_id()
        if selected_id is None:
            return None

        id_col = self.config.settings.get("id_col", "Use Index")

        try:
            if id_col == "Use Index":
                idx = int(selected_id)
                if idx in self.df.index:
                    return self.df.loc[[idx]]
                return None

            matches = self.df[self.df[id_col].astype(str) == str(selected_id)]
            if len(matches) > 0:
                return matches.head(1)
        except Exception:
            pass

        return None

    def get_ra_dec(self, err_message="No ra and dec available for this source"):
        print("Running get_ra_dec in EuclidPlotClass")
        if self._request_missing_mappings():
            print(err_message)
            return None, None

        row = self._get_selected_row_from_active_df()
        if row is None or row.empty:
            print(err_message)
            return None, None

        ra_col = self.config.settings.get("ra_col_name")
        dec_col = self.config.settings.get("dec_col_name")

        try:
            if ra_col in row.columns and dec_col in row.columns:
                ra = float(row.iloc[0][ra_col])
                dec = float(row.iloc[0][dec_col])
                return ra, dec
        except Exception:
            print("ra_col in row.columns and dec_col in row.columns != True")
            pass

        try:
            if "ra_dec" in row.columns:
                ra_dec = str(row.iloc[0]["ra_dec"])
                ra = float(ra_dec[: ra_dec.index(",")])
                dec = float(ra_dec[ra_dec.index(",") + 1 :])
                return ra, dec
        except Exception:
            pass

        print(err_message)
        return None, None

    def _initialise_euclid_object(self):
        self._refresh_df_from_active_dataset()

        if self._request_missing_mappings():
            self.get_error_panel(
                "Euclid cutout unavailable",
                "Missing dataset mappings for RA and/or DEC",
            )
            return False

        self.ra, self.dec = self.get_ra_dec()

        if (self.ra is None) or (self.dec is None):
            self.get_error_panel("Euclid cutout unavailable", "Missing RA or DEC value")
            return False

        try:
            self.euclid_object.reset_data(self.ra, self.dec)
        except AttributeError:
            self.euclid_object = EuclidCutoutsClass(
                self.ra,
                self.dec,
                euclid_filters=["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                context=self.context,
            )
            self.overplotted_coordinates = []

        return True

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def get_layout(self):
        if not self._widgets_initialised:
            self._initialise_widgets()

            self._bind_selection_runtime_subscriptions()
            self._subscribe_to_mapping_and_dataset_events()
            self._manage_subscriptions()

            self._widgets_initialised = True

        self.figure.sizing_mode = "stretch_width"
        self.figure.min_height = 120
        self.figure.max_height = 260
        self.figure.margin = (0, 0, 10, 0)

        self.message_pane.visible = False

        # Important: do not launch expensive work directly from get_layout().
        self._request_initial_refresh_once(reason="initial.layout")

        return pn.Column(
            self.message_pane,
            self.figure,
            self.plot_settings_panel,
            sizing_mode="stretch_both",
            min_height=0,
            scroll=False,
        )

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _initialize_settings_dictionary(self):
        euclid_settings = self.config.settings.setdefault("Euclid_cutout_settings", {})

        default_values = {
            "filter": "Color",
            "radius": 5.0,
            "stretching": "Linear",
            "clipping": (0, 1),
            "scale": "minmax",
            "gamma": (1, 1, 1),
            "source_coordinates": False,
            "levels": 0,
        }
        for key, value in default_values.items():
            if key not in euclid_settings:
                self._update_settings_dictionary(key, value)

    def _get_from_settings_dictionary(self, key, default):
        euclid_settings = self.config.settings.setdefault("Euclid_cutout_settings", {})
        value = euclid_settings.get(key, default)
        if key in ("clipping", "gamma"):
            value = tuple(value)
        return value

    def _update_settings_dictionary(self, key, value):
        euclid_settings = self.config.settings.setdefault("Euclid_cutout_settings", {})
        euclid_settings[key] = value

    def _update_all_settings_dictionary(self):
        filter = self.filter_input.value
        low, high = self.contrast_scaler.value
        gamma = (self.gamma_red_input.value, self.gamma_green_input.value, self.gamma_blue_input.value)
        scale = self.scale_input.value
        source_coordinates = self.overplot_source_coords_widget.value
        levels = self.contour_levels_input.value

        self._update_settings_dictionary("scale", scale)
        self._update_settings_dictionary("filter", filter)
        self._update_settings_dictionary("gamma", gamma)
        self._update_settings_dictionary("clipping", (low, high))
        self._update_settings_dictionary("source_coordinates", source_coordinates)
        self._update_settings_dictionary("levels", levels)

    # ------------------------------------------------------------------
    # Image / save helpers
    # ------------------------------------------------------------------

    def _get_scaled_image(self):
        if self.filter != "Color":
            low, high = self.contrast_scaler.value
            return self.euclid_object.transform_image_range(
                self.filter,
                low,
                high,
                scale_method=self.scale_input.value,
            )

        gamma = (
            self.gamma_red_input.value,
            self.gamma_green_input.value,
            self.gamma_blue_input.value,
        )
        scale_by_channel = True
        low_r, high_r = self.contrast_scaler_red.value
        low_g, high_g = self.contrast_scaler_green.value
        low_b, high_b = self.contrast_scaler_blue.value
        low = (low_r, low_g, low_b)
        high = (high_r, high_g, high_b)

        if (low == (0, 0, 0)) and (high == (1, 1, 1)):
            low, high = self.contrast_scaler.value
            scale_by_channel = False

        return self.euclid_object.transform_image_range(
            self.filter,
            low,
            high,
            gamma=gamma,
            scale_method=self.scale_input.value,
            scale_by_channel=scale_by_channel,
        )

    def _save_figure(self, directory_path="data/saved_sources", prefix=None):
        try:
            fname = f"{prefix + '_' if prefix else ''}{self.panel_name}.png"
            filename = os.path.join(directory_path, fname)
            scaled_image = self._get_scaled_image()
            fig = self.get_euclid_figure(
                scaled_image,
                show_scale=True,
                show_coordinates=self.overplot_source_coords_widget.value,
                show_spectra_coordinates=self.overplot_coords_widget.value,
            )
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)
            return filename
        except FileNotFoundError:
            print(f"Could not find the saving directory: {directory_path}")
        except AttributeError as e:
            print(e)
        except KeyError as e:
            print(f"Missing filter {e} in euclid_object.plot_data")

    def _save_data_to_fits(self, directory_path="data/saved_sources"):
        try:
            self.euclid_object.export_cutouts_to_fits(
                bands_to_export=["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                directory_path=directory_path,
            )
        except AttributeError:
            pass
        except FileNotFoundError:
            print(f"Could not find the saving directory: {directory_path}")

    # ------------------------------------------------------------------
    # UI widget helpers
    # ------------------------------------------------------------------

    def _field_label(self, text, width=320):
        return pn.pane.HTML(
            f"""
            <div style="
                font-weight: 600;
                font-size: 13px;
                line-height: 16px;
                padding-left: 2px;
                margin: 0;
            ">
                {text}
            </div>
            """,
            width=width,
            height=16,
            margin=(2, 0, 2, 0),
            sizing_mode="fixed",
        )

    def _field_block(self, text, widget, width=320, bottom=8):
        return pn.Column(
            self._field_label(text, width=width),
            widget,
            width=width,
            margin=(0, 0, bottom, 0),
            sizing_mode="fixed",
        )

    def _fix_width(self, widget, width=320, height=34):
        widget.width = width
        widget.min_width = width
        widget.max_width = width
        if hasattr(widget, "height"):
            widget.height = height
            widget.min_height = height
            widget.max_height = height
        widget.sizing_mode = "fixed"
        return widget

    def _initialise_widgets(self):
        CONTROL_HEIGHT = 34

        def fix_control(widget, height=CONTROL_HEIGHT):
            widget.height = height
            widget.min_height = height
            widget.max_height = height
            widget.sizing_mode = "stretch_width"
            return widget

        def section_title(text):
            return pn.pane.HTML(
                f"""
                <div style="
                    font-weight: 600;
                    font-size: 13px;
                    line-height: 18px;
                    margin: 0;
                    padding: 0 0 2px 0;
                ">
                    {text}
                </div>
                """,
                height=18,
                margin=(0, 0, 4, 0),
                sizing_mode="stretch_width",
            )

        def field_label(text):
            return pn.pane.HTML(
                f"""
                <div style="
                    font-size: 12px;
                    line-height: 16px;
                    margin: 0;
                    padding: 0;
                ">
                    {text}
                </div>
                """,
                height=16,
                margin=(0, 0, 2, 0),
                sizing_mode="stretch_width",
            )

        def field_block(text, widget, bottom=8):
            return pn.Column(
                field_label(text),
                widget,
                margin=(0, 0, bottom, 0),
                sizing_mode="stretch_width",
            )

        self.radius_input = fix_control(
            pn.widgets.FloatInput(
                name="",
                value=self.radius,
                step=0.5,
                start=1,
                end=100,
                margin=0,
            )
        )

        self.filter_input = fix_control(
            pn.widgets.Select(
                name="",
                options={
                    "VIS": "VIS",
                    "Y": "NIR_Y",
                    "J": "NIR_J",
                    "H": "NIR_H",
                    "Color": "Color",
                },
                value=self.filter,
                margin=0,
            )
        )

        self.stretching_input = fix_control(
            pn.widgets.Select(
                name="",
                options=["Linear", "Sqrt", "Log", "Asinh", "PowerLaw"],
                value=self._get_from_settings_dictionary("stretching", "Linear"),
                margin=0,
            )
        )

        self.scale_input = fix_control(
            pn.widgets.Select(
                name="",
                options=["MinMax", "Expand"],
                value=self._get_from_settings_dictionary("scale", "MinMax"),
                margin=0,
            )
        )

        self.contrast_scaler = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=self._get_from_settings_dictionary("clipping", (0, 1)),
            margin=0,
            sizing_mode="stretch_width",
        )

        self.overplot_source_coords_widget = pn.widgets.Checkbox(
            name="Source Coordinates",
            value=self._get_from_settings_dictionary("source_coordinates", False),
            margin=0,
            sizing_mode="stretch_width",
        )

        self.overplot_coords_widget = pn.widgets.Checkbox(
            name="Spectrum Coordinates",
            value=False,
            margin=0,
            sizing_mode="stretch_width",
        )

        self.contour_levels_input = fix_control(
            pn.widgets.IntInput(
                name="",
                value=self._get_from_settings_dictionary("levels", 0),
                step=1,
                start=0,
                end=15,
                margin=0,
            )
        )

        self.contour_levels_scale_input = fix_control(
            pn.widgets.Select(
                name="",
                options={
                    "Sqrt(2)": (np.sqrt(2), 1),
                    "2": (2, 1),
                    "10": (10, 1),
                    "Exponential": (np.exp(1), 1),
                    "Gaussian": (np.exp(1), 2),
                    "de Vaucouleurs": (np.exp(1), 0.25),
                },
                value=1,
                margin=0,
            )
        )

        self.environment_input = fix_control(
            pn.widgets.Select(
                name="",
                options={
                    "Public Data Release": "PDR",
                    "Internal Data Release": "IDR",
                    "On The Fly": "OTF",
                    "REG": "REG",
                },
                value="PDR",
                disabled_options=["REG"],
                margin=0,
            )
        )

        self.user_input = fix_control(
            pn.widgets.TextInput(
                name="",
                placeholder="ESA username",
                margin=0,
            )
        )

        self.password_input = fix_control(
            pn.widgets.PasswordInput(
                name="",
                placeholder="Password",
                margin=0,
            )
        )

        self.confirm_login_button = pn.widgets.Button(
            name="Confirm login",
            button_type="primary",
            height=CONTROL_HEIGHT,
            min_height=CONTROL_HEIGHT,
            max_height=CONTROL_HEIGHT,
            margin=(0, 0, 0, 0),
            sizing_mode="stretch_width",
        )

        self.login_column = pn.Column(
            field_block("Username", self.user_input, bottom=8),
            field_block("Password", self.password_input, bottom=8),
            self.confirm_login_button,
            visible=False,
            margin=(4, 0, 0, 0),
            sizing_mode="stretch_width",
        )

        self.color_settings_column = self._initialise_color_settings()
        self.color_settings_button = pn.widgets.Button(
            name="Color settings ▾",
            button_type="default",
            height=CONTROL_HEIGHT,
            min_height=CONTROL_HEIGHT,
            max_height=CONTROL_HEIGHT,
            margin=(0, 0, 4, 0),
            sizing_mode="stretch_width",
        )

        self.add_param_watch_many(
            [
                self.contrast_scaler,
                self.scale_input,
                self.filter_input,
                self.overplot_source_coords_widget,
                self.contour_levels_input,
                self.contour_levels_scale_input,
            ],
            self._general_parameter_callback,
            what="value",
        )

        self.add_param_watch(self.radius_input, self._update_radius, "value")
        self.add_param_watch(self.stretching_input, self._update_stretching, "value")
        self.add_param_watch(self.overplot_coords_widget, self._overplot_coordinates_callback, "value")
        self.add_param_watch(self.environment_input, self._change_euclid_environment, "value")

        self.confirm_login_button.on_click(self._confirm_login_credentials_cb)
        self.color_settings_button.on_click(self._open_color_settings_cb)

        self.plot_settings_panel = pn.Column(
            section_title("Basic"),
            field_block("Radius [arcsec]", self.radius_input),
            field_block("Euclid Filter", self.filter_input),
            field_block("Stretching", self.stretching_input),
            field_block("Scaling Mode", self.scale_input),
            field_block("Image Clipping", self.contrast_scaler, bottom=10),

            section_title("Overlays"),
            pn.Column(
                self.overplot_source_coords_widget,
                pn.Spacer(height=4),
                self.overplot_coords_widget,
                margin=(0, 0, 10, 0),
                sizing_mode="stretch_width",
            ),

            section_title("Contours"),
            field_block("Contour Levels", self.contour_levels_input),
            field_block("Contours drop", self.contour_levels_scale_input, bottom=10),

            section_title("Color"),
            self.color_settings_button,
            self.color_settings_column,
            pn.Spacer(height=10),

            section_title("Archive"),
            field_block("Archive Environment", self.environment_input, bottom=6),
            self.login_column,

            visible=False,
            scroll=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
        )

    def _initialise_color_settings(self):
        CONTROL_HEIGHT = 34

        def fix_control(widget, height=CONTROL_HEIGHT):
            widget.height = height
            widget.min_height = height
            widget.max_height = height
            widget.sizing_mode = "stretch_width"
            return widget

        def field_label(text):
            return pn.pane.HTML(
                f"""
                <div style="
                    font-size: 12px;
                    line-height: 16px;
                    margin: 0;
                    padding: 0;
                ">
                    {text}
                </div>
                """,
                height=16,
                margin=(0, 0, 2, 0),
                sizing_mode="stretch_width",
            )

        def field_block(text, widget, bottom=8):
            return pn.Column(
                field_label(text),
                widget,
                margin=(0, 0, bottom, 0),
                sizing_mode="stretch_width",
            )

        gamma = self._get_from_settings_dictionary("gamma", [1, 1, 1])

        self.contrast_scaler_red = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="red",
            margin=0,
            sizing_mode="stretch_width",
        )

        self.contrast_scaler_green = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="green",
            margin=0,
            sizing_mode="stretch_width",
        )

        self.contrast_scaler_blue = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="blue",
            margin=0,
            sizing_mode="stretch_width",
        )

        self.gamma_red_input = fix_control(
            pn.widgets.FloatInput(
                name="",
                value=gamma[0],
                step=0.1,
                start=0,
                end=5,
                margin=0,
            )
        )

        self.gamma_green_input = fix_control(
            pn.widgets.FloatInput(
                name="",
                value=gamma[1],
                step=0.1,
                start=0,
                end=5,
                margin=0,
            )
        )

        self.gamma_blue_input = fix_control(
            pn.widgets.FloatInput(
                name="",
                value=gamma[2],
                step=0.1,
                start=0,
                end=5,
                margin=0,
            )
        )

        self.add_param_watch_many(
            [self.contrast_scaler_red, self.contrast_scaler_green, self.contrast_scaler_blue],
            self._color_specific_callback,
            what="value_throttled",
        )

        self.add_param_watch_many(
            [self.gamma_red_input, self.gamma_green_input, self.gamma_blue_input],
            self._color_specific_callback,
            what="value",
        )

        return pn.Column(
            field_block("Red clipping", self.contrast_scaler_red),
            field_block("Green clipping", self.contrast_scaler_green),
            field_block("Blue clipping", self.contrast_scaler_blue),
            field_block("Gamma R", self.gamma_red_input),
            field_block("Gamma G", self.gamma_green_input),
            field_block("Gamma B", self.gamma_blue_input),
            visible=False,
            scroll=False,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _update_radius(self, event):
        if event.new:
            self.radius = event.new
            self._update_settings_dictionary("radius", self.radius)

            if self.context and self.context.events:
                self.context.events.publish(
                    "astro.euclid.radius.changed",
                    {"radius": self.radius, "panel_id": self.panel_id},
                )

            self._request_refresh(reason="euclid.radius.changed")
        else:
            print("Input a valid value for radius")

    def _general_parameter_callback(self, event):
        if hasattr(self.euclid_object, "plot_data"):
            self.filter = self.filter_input.value
            self._update_all_settings_dictionary()
            scaled_image = self._get_scaled_image()
            self.get_euclid_figure_hv(
                scaled_image,
                show_coordinates=self.overplot_source_coords_widget.value,
            )
            self._update_image()

    def _update_stretching(self, event):
        stretch = self.stretching_input.value
        if not isinstance(event.new, str):
            stretch_scale = self.stretching_scale_input.value
            self._update_settings_dictionary("stretching_scale", stretch_scale)
        else:
            stretch_scale = None

        self._update_settings_dictionary("stretching", stretch)
        self.euclid_object.get_plot_data(stretch=stretch, stretch_scale=stretch_scale)
        scaled_image = self._get_scaled_image()
        self.get_euclid_figure_hv(
            scaled_image,
            show_coordinates=self.overplot_source_coords_widget.value,
        )
        self._update_image()

    def _color_specific_callback(self, event):
        if self.filter == "Color":
            if hasattr(self.euclid_object, "plot_data"):
                scaled_image = self._get_scaled_image()
                self.get_euclid_figure_hv(
                    scaled_image,
                    show_coordinates=self.overplot_source_coords_widget.value,
                )
                self._update_image()

    def _update_image(self):
        try:
            overlay = hv.Overlay(self.euclid_fig + self.overplotted_coordinates).opts(
                responsive=True,
                aspect="equal",
                toolbar=None,
                shared_axes=False,
                axiswise=True,
            )
            self.figure.object = overlay
            self.message_pane.visible = False
        except Exception as e:
            print(f"Euclid image unavailable:\n {e}")

    def _add_coordinates(self, coordinates, dataset):
        if not coordinates or "ra" not in coordinates or "dec" not in coordinates:
            print("Wrong passed coordinates")
            return

        ra, dec = coordinates["ra"], coordinates["dec"]
        if not hasattr(self, "stored_spectrum_coordinates"):
            self.stored_spectrum_coordinates = {}

        self.stored_spectrum_coordinates[dataset] = {"ra": ra, "dec": dec}
        self.overplot_coords_widget.name = "Spectrum Coordinates"

        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()

    def _show_overplot_coordinates(self):
        if hasattr(self, "stored_spectrum_coordinates"):
            self.overplot_coords_widget.name = "Spectrum Coordinates"
            if self.overplot_coords_widget.value:
                self.overplotted_coordinates = []
                for dataset in self.stored_spectrum_coordinates:
                    print(f"overplotting coordinates for {dataset}")
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N, 2))
                    marker = "+" if dataset == "DESI" else "*"
                    label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                    for i, (x, y) in enumerate(
                        self.euclid_object.world_2_pix(
                            ra=self.stored_spectrum_coordinates[dataset]["ra"],
                            dec=self.stored_spectrum_coordinates[dataset]["dec"],
                            filtro=self.filter,
                            zipped=True,
                        )
                    ):
                        if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                            points = hv.Points([(x, y)], label=label if i == 0 else "")
                            points = points.opts(color=colors(i), marker=marker, size=20)
                            self.overplotted_coordinates.append(points)

                self._update_image()

    def _overplot_coordinates_callback(self, event):
        if event.new:
            if not hasattr(self, "stored_spectrum_coordinates"):
                print("No spectrum coordinates available")
                event.obj.name = "Spectrum Coordinates [Not Currently Avaliable]"
                self.overplotted_coordinates = []
                return None

            event.obj.name = "Spectrum Coordinates"
            self._show_overplot_coordinates()

        elif not event.new:
            self.overplotted_coordinates = []

        self._update_image()

    def _change_euclid_environment(self, event):
        self.environment = event.new

        if self.environment == "PDR":
            self.login_column.visible = False
            self.euclid_object.change_environment(environment="PDR")
            return

        if self.environment in ["IDR", "OTF", "REG"]:
            if os.path.isfile("euclid_credentials.login"):
                self.login_column.visible = False
                print("I found the credential file")
                self.euclid_object.change_environment(
                    environment=self.environment,
                    user=None,
                    password=None,
                    credentials_filepath="euclid_credentials.login",
                )
                return

            user = self.config.settings.get("EuclidAccountUser", None)
            password = self.config.settings.get("EuclidAccountPassword", None)

            if (user is None) or (password is None) or (str(user).strip() == "") or (str(password).strip() == ""):
                self.login_column.visible = True
            else:
                self.login_column.visible = False
                self.euclid_object.change_environment(
                    environment=self.environment,
                    user=user,
                    password=password,
                )

    def _confirm_login_credentials_cb(self, event):
        self.login_column.visible = False
        self.config.settings["EuclidAccountUser"] = self.user_input.value
        self.config.settings["EuclidAccountPassword"] = self.password_input.value
        self.euclid_object.change_environment(
            environment=self.environment,
            user=self.config.settings["EuclidAccountUser"],
            password=self.config.settings["EuclidAccountPassword"],
        )
        svc = getattr(self.context, "services", None) if self.context else None
        if svc is not None:
            svc.set("euclid.client", self.euclid_object.client)

    def _open_color_settings_cb(self, event):
        self.color_settings_column.visible = not self.color_settings_column.visible
        self.color_settings_button.name = (
            "Color settings ▴" if self.color_settings_column.visible else "Color settings ▾"
        )

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix[self.filter]
        return bar_length_arcsecond

    def get_euclid_figure_hv(self, data, show_coordinates=False, show_scale=True):
        self.image_height, self.image_width = data.shape[:2]
        bounds = (0, 0, self.image_width, self.image_height)

        if len(data.shape) == 3:
            image = hv.RGB(data[::-1, ...], bounds=bounds).opts(
                active_tools=[],
                toolbar=None,
                padding=0,
                border=0,
                framewise=True,
                shared_axes=False,
                xaxis=None,
                yaxis=None,
            )
        else:
            image = hv.Image(data[::-1, ...], bounds=bounds).opts(
                active_tools=[],
                toolbar=None,
                padding=0,
                border=0,
                framewise=True,
                shared_axes=False,
                xaxis=None,
                yaxis=None,
                cmap="grey",
            )

        self.image_stream = hv.streams.Tap(source=image, x=np.nan, y=np.nan)
        self.add_param_watch(self.image_stream, self._light_profile_callback, what=["x"])

        self.euclid_fig = [image]

        if self.contour_levels_input.value > 0:
            N_contour_levels = self.contour_levels_input.value
            base, exponent = self.contour_levels_scale_input.value
            temp_data = self.euclid_object.data[self.filter]
            temp_img = hv.RGB(temp_data[::-1, ...], bounds=bounds) if len(temp_data.shape) == 3 else hv.Image(temp_data[::-1, ...], bounds=bounds)
            levels = np.nanmax(temp_data) / (base ** (np.arange(1, N_contour_levels + 1) * exponent))
            contours = hv.operation.contours(temp_img, levels=levels).opts(
                cmap=["red"],
                colorbar=False,
                active_tools=[],
                show_legend=False,
            )
            self.euclid_fig.append(contours)

        if show_scale:
            self.bar_length_pixels = self.image_width * 0.2
            x0, y0 = 0.1 * self.image_width, 0.1 * self.image_height
            x1 = x0 + self.bar_length_pixels
            scale_bar = hv.Curve(([x0, x1], [y0, y0])).opts(color="red", line_width=3)
            scale_text = hv.Text(
                x=(x0 + x1) / 2,
                y=y0 + y0 / 2,
                text=f'{self.get_plot_scale():.1f}"',
            ).opts(
                text_color="red",
                text_align="center",
                text_baseline="bottom",
                fontsize=14,
            )
            self.euclid_fig.extend([scale_bar, scale_text])

        if show_coordinates:
            label = f"{np.round(self.ra, 3)}, {np.round(self.dec, 3)}"
            x, y = self.euclid_object.world_2_pix(ra=self.ra, dec=self.dec, filtro=self.filter, zipped=False)
            if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                points = hv.Points([(x, y)], label=label).opts(
                    color="blue",
                    marker="+",
                    size=30,
                )
                self.euclid_fig.append(points)

        if self.overplot_coords_widget.value:
            self._show_overplot_coordinates()

    def _light_profile_callback(self, event):
        col, row = self.image_stream.x, self.image_stream.y
        if (row is None) or (col is None):
            return

        row, col = int(round(row)), int(round(col))
        row = max(0, min(row, self.image_height - 1))
        col = max(0, min(col, self.image_width - 1))

        if self.filter not in ["Color"]:
            self.get_light_profile_plot(row, col)

    def _light_profile_callback_reverse(self, event):
        self._update_image()

    def get_light_profile_plot(self, row, col):
        scaled_image = self._get_scaled_image()

        def get_curve(values, idx, xlabel, plot_psf=True, fwhm_psf=0.16, arcsec_per_pix=0.1):
            curve = hv.Curve(values, kdims="x", vdims="value").opts(
                toolbar=None,
                padding=0.0,
                border=1,
                framewise=True,
                shared_axes=False,
                axiswise=True,
                active_tools=[],
                xlabel=xlabel,
                yaxis=None,
                ylim=(min(0, np.nanmin(values)), np.nanmax(values) * 1.1),
                color="black",
            )
            line = hv.VLine(idx).opts(color="red", line_width=1, line_dash="dotted")

            if plot_psf:
                window = 15
                start = max(0, idx - window)
                end = min(len(values), idx + window)
                reduced_values = values[start:end]
                peak = np.nanmax(reduced_values)
                peak_indices = np.where(reduced_values == peak)[0]
                peak_idx = start + peak_indices[len(peak_indices) // 2]
                sigma_psf = fwhm_psf / 2.35482004503 / arcsec_per_pix
                x = np.arange(len(values))
                psf_profile = peak * np.exp(-0.5 * ((x - peak_idx) / sigma_psf) ** 2)
                psf = hv.Curve(psf_profile, kdims="x", vdims="value").opts(color="red", line_width=1, line_dash="solid")
                image = hv.Overlay([curve, line, psf]).opts(responsive=True, toolbar=None)
            else:
                image = hv.Overlay([curve, line]).opts(
                    responsive=True, 
                    toolbar=None,
                    shared_axes=False,
                    axiswise=True,)

            return image

        arcsec_per_pix = self.euclid_object.arcsec_per_pix[self.filter]
        fwhm_psf = 0.16 if self.filter == "VIS" else 0.3
        plot_psf = self._get_from_settings_dictionary("stretching", None) == "Linear"

        plot_x = get_curve(
            scaled_image[row, :],
            col,
            "X coordinate",
            plot_psf=plot_psf,
            fwhm_psf=fwhm_psf,
            arcsec_per_pix=arcsec_per_pix,
        )
        plot_y = get_curve(
            scaled_image[:, col],
            row,
            "Y coordinate",
            plot_psf=plot_psf,
            fwhm_psf=fwhm_psf,
            arcsec_per_pix=arcsec_per_pix,
        )

        layout = hv.Layout(plot_x + plot_y).cols(1).opts(
            sizing_mode="stretch_both",
            shared_axes=False,
            axiswise=True,
        )
        row_stream = hv.streams.Tap(source=plot_x, x=np.nan, y=np.nan)
        col_stream = hv.streams.Tap(source=plot_y, x=np.nan, y=np.nan)

        self.add_param_watch_many(
            [row_stream, col_stream],
            self._light_profile_callback_reverse,
            what=["x"],
        )

        self.figure.object = layout

    def _run_euclid(self, reason=None, refresh_signature=None):
        refresh_signature = refresh_signature or self._build_refresh_signature(
            reason=reason,
            payload=None,
        )

        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True

        if self.context and self.context.events:
            self.context.events.publish(
                "astro.cutout.running",
                {
                    "source": "Euclid",
                    "running": True,
                    "panel_id": self.panel_id,
                    "reason": reason,
                    "dataset_id": self._dataset_id(),
                    "selected_id": self._get_selected_id(),
                },
            )

        def callback(future_obj=None):
            try:
                if future_obj is not None:
                    future_obj.result()

                if self.euclid_object.error_tracker.has_error:
                    message = "# Euclid cutout unavailable:\n"
                    message += f"## {self.euclid_object.error_tracker.error_message}"
                    self.message_pane.object = message
                    self.message_pane.visible = True
                    self.figure.object = self.get_empty_image()
                    return

                self.overplot_coords_widget.value = False
                scaled_image = self._get_scaled_image()
                self.get_euclid_figure_hv(
                    scaled_image,
                    show_coordinates=self.overplot_source_coords_widget.value,
                )
                self._update_image()

            except Exception as e:
                self.message_pane.object = f"# Euclid cutout unavailable:\n## {e}"
                self.message_pane.visible = True
                self.figure.object = self.get_empty_image()

            finally:
                if self.context and self.context.events:
                    self.context.events.publish(
                        "astro.cutout.running",
                        {
                            "source": "Euclid",
                            "running": False,
                            "panel_id": self.panel_id,
                            "reason": reason,
                            "dataset_id": self._dataset_id(),
                            "selected_id": self._get_selected_id(),
                        },
                    )

                self._finish_refresh(refresh_signature)

        self.run_multithread(
            self.euclid_object.get_final_cutout,
            func_kwargs={
                "radius": self.radius,
                "stretch": self.stretching_input.value,
                "filtro": self.filter_input.value,
                "reference": "VIS",
                "verbose": True,
                "return_object": True,
            },
            callback=callback,
        )

    def get_euclid_figure(
        self,
        data,
        show_coordinates=False,
        show_scale=True,
        show_spectra_coordinates=False,
    ):
        image_height, image_width = data.shape[:2]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data, origin="lower", cmap="gray")

        if show_scale:
            bar_length_pixels = image_width * 0.2
            x0, y0 = 0.1 * image_width, 0.1 * image_height
            x1 = x0 + bar_length_pixels
            ax.plot([x0, x1], [y0, y0], color="red", lw=3)
            ax.text(
                x=(x0 + x1) / 2,
                y=y0 + y0 / 2,
                s=f'{self.get_plot_scale():.1f}"',
                color="red",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        if show_coordinates:
            label = f"{np.round(self.ra, 3)}, {np.round(self.dec, 3)}"
            x, y = self.euclid_object.world_2_pix(ra=self.ra, dec=self.dec, filtro=self.filter, zipped=False)
            if (0 <= x < image_width) and (0 <= y < image_height):
                ax.scatter(x, y, s=130, label=label, c="blue", marker="+")

        if show_spectra_coordinates:
            if hasattr(self, "stored_spectrum_coordinates"):
                for dataset in self.stored_spectrum_coordinates:
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N, 2))(np.arange(N))
                    marker = "+" if dataset == "DESI" else "x"
                    label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                    x, y = self.euclid_object.world_2_pix(
                        ra=self.stored_spectrum_coordinates[dataset]["ra"],
                        dec=self.stored_spectrum_coordinates[dataset]["dec"],
                        filtro=self.filter,
                        zipped=False,
                    )
                    x = np.where((0 <= x) & (x < image_width), x, np.nan)
                    y = np.where((0 <= y) & (y < image_height), y, np.nan)
                    ax.scatter(x, y, color=colors, label=label, marker=marker, s=100)

        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

        ax.axis("off")
        fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0)
        return fig

    # ------------------------------------------------------------------
    # Artifact subscriptions
    # ------------------------------------------------------------------

    def _manage_subscriptions(self):
        if not self.context or not getattr(self.context, "events", None) or not getattr(self.context, "artifacts", None):
            return

        if getattr(self, "_coords_subscription_ready", False):
            return
        self._coords_subscription_ready = True

        def _coords_updated(topic, payload):
            if not payload:
                return

            source = payload.get("source")
            artifact_id = payload.get("artifact_id")
            dataset_id = payload.get("dataset_id", self._get_active_dataset_id())
            payload_selected_id = payload.get("selected_id")

            current_selected_id = self._get_selected_source_id()

            if not source or not artifact_id:
                return

            if current_selected_id is not None and payload_selected_id is not None:
                if str(payload_selected_id) != str(current_selected_id):
                    return

            try:
                coords = self.context.artifacts.get(artifact_id)
            except Exception as e:
                print(f"Failed to load coords artifact {artifact_id}: {e}")
                return

            self._add_coordinates(coords, source)

        self.subscribe("astro.coords.updated", _coords_updated)

        active_dataset_id = self._get_active_dataset_id()
        current_selected_id = self._get_selected_source_id()

        dataset_ids_to_try = [active_dataset_id]
        if active_dataset_id != "default":
            dataset_ids_to_try.append("default")

        for src_name in ("DESI", "SDSS", "EuclidSpec"):
            for dsid in dataset_ids_to_try:
                try:
                    params_subset = {"source": src_name}
                    if current_selected_id is not None:
                        params_subset["selected_id"] = current_selected_id

                    refs = self.context.artifacts.find(
                        type="astro.coords",
                        dataset_id=dsid,
                        params_subset=params_subset,
                    )

                    if refs:
                        coords = self.context.artifacts.get(refs[0].artifact_id)
                        self._add_coordinates(coords, src_name)
                        break

                except Exception as e:
                    print(f"Late-join coords lookup failed for {src_name} in dataset {dsid}: {e}")

class SpectrumPlotClass(CustomPlotClass):

    def __init__(self, data, close_button, extra_features, dataset="DESI", context=None):
        super().__init__(
            data,
            close_button,
            extra_features,
            panel_name=f"{dataset}_spectrum",
            context=context,
        )

        if context is not None and getattr(context, "config", None) is not None:
            self.config = context.config

        self.figure = pn.Column(
            sizing_mode="stretch_both",
            min_height=0,
            margin=(5, 20),
            scroll=False,
        )
        self.dataset = dataset
        self._is_euclid_spec = self.dataset == "EuclidSpec"

        self.from_sourceId = False
        self._euclid_radius_sub = None
        self._widgets_initialised = False

        self._initialize_settings_dictionary()
        self.plot_settings_panel = pn.Column(visible=False, scroll=True)
        self.mode_options = ["Use TargetId", "Cone Search"]
        self.chosen_mode = self.mode_options[1]

        self._bind_dataset_runtime_subscriptions(
            include_loaded=False,
            include_mapping=True,
        )

    def _dispose_impl(self) -> None:
        """
        Extra cleanup for SpectrumPlotClass only.
        Keeps CustomPlotClass cleanup intact.
        """
        if getattr(self, "_euclid_radius_sub", None) is not None and self.events is not None:
            try:
                self.events.unsubscribe(self._euclid_radius_sub)
            except Exception:
                pass
            self._euclid_radius_sub = None

    def _to_list(self, x):
        """Convert numpy/array-like to plain python list safely."""
        if x is None:
            return None
        try:
            return list(x)
        except Exception:
            return x

    def _get_attr_or_key(self, obj, name, default=None):
        """Get attribute (Euclid SpectrumContainer) or dict key (DESI record)."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _build_refresh_signature(self, reason=None, payload=None):
        mode_value = self.chosen_mode
        max_sep_value = self.max_separation

        if getattr(self, "_widgets_initialised", False):
            try:
                mode_value = self.retrieve_mode_button.value
            except Exception:
                pass

            try:
                max_sep_value = self.max_separation_input.value
            except Exception:
                pass

        return (
            self._get_active_dataset_id(),
            self._get_selected_source_id(),
            str(self.dataset),
            bool(self.from_sourceId),
            str(mode_value),
            float(max_sep_value) if max_sep_value is not None else None,
        )
    
    def _perform_refresh(self, reason=None, payload=None, refresh_signature=None):
        try:
            self.df = self._get_dataset_for_lookup()

            if getattr(self, "_widgets_initialised", False):
                try:
                    current_value = getattr(self.redshift_column_selector, "value", "None")
                    options = ["None"] + self.get_column_list(allowed_types=["float"])
                    self.redshift_column_selector.options = options
                    if current_value in options:
                        self.redshift_column_selector.value = current_value
                    else:
                        self.redshift_column_selector.value = "None"
                except Exception:
                    pass

            initialized = self._initialize_spectrum_object()
            if not initialized:
                self._finish_refresh(refresh_signature)
                return

            self._run_spectrum(
                reason=reason,
                refresh_signature=refresh_signature,
            )

        except Exception:
            traceback.print_exc()
            self._finish_refresh(refresh_signature)

    def _build_spectrum_artifact_payload(self) -> dict:
        """
        Build a JSON-ish artifact payload from self.spectrum_object.spectra
        for either DESI (dict-like records) or Euclid (SpectrumContainer objects).
        """
        spectra = getattr(self.spectrum_object, "spectra", None)
        if spectra is None:
            return {"source": self.dataset, "spectra": []}

        payload = {
            "source": self.dataset,
            "ra0": getattr(self.spectrum_object, "ra", None),
            "dec0": getattr(self.spectrum_object, "dec", None),
            "spectra": [],
        }

        for sp in spectra:
            sid = (
                self._get_attr_or_key(sp, "specid", None)
                or self._get_attr_or_key(sp, "sourceId", None)
                or self._get_attr_or_key(sp, "sparcl_id", None)
            )

            rec = {
                "id": sid,
                "wavelength": self._to_list(self._get_attr_or_key(sp, "wavelength", None)),
                "flux": self._to_list(self._get_attr_or_key(sp, "flux", None)),
            }

            mask = self._get_attr_or_key(sp, "mask", None)
            if mask is not None:
                rec["mask"] = self._to_list(mask)

            model = self._get_attr_or_key(sp, "model", None)
            if model is not None:
                rec["model"] = self._to_list(model)

            redshift = self._get_attr_or_key(sp, "redshift", None)
            if redshift is not None:
                try:
                    rec["redshift"] = float(redshift)
                except Exception:
                    rec["redshift"] = redshift

            spectype = self._get_attr_or_key(sp, "spectype", None)
            if spectype is not None:
                rec["spectype"] = str(spectype)

            ra = self._get_attr_or_key(sp, "ra", None)
            dec = self._get_attr_or_key(sp, "dec", None)
            if ra is not None:
                try:
                    rec["ra"] = float(ra)
                except Exception:
                    rec["ra"] = ra
            if dec is not None:
                try:
                    rec["dec"] = float(dec)
                except Exception:
                    rec["dec"] = dec

            dr = self._get_attr_or_key(sp, "data_release", None)
            if dr is not None:
                rec.setdefault("meta", {})["data_release"] = dr

            payload["spectra"].append(rec)

        return payload

    def publish_spectrum_artifact(self, dataset_id: str = "default"):
        if self.artifacts is None:
            return

        selected_id = self._get_selected_id()
        spec_payload = self._build_spectrum_artifact_payload()
        spectrum_count = len(spec_payload.get("spectra", []))

        artifact_id = self.artifacts.put(
            type="astro.spectrum",
            payload=spec_payload,
            dataset_id=dataset_id,
            row_ids=[str(selected_id)] if selected_id is not None else None,
            params={
                "source": self.dataset,
                "selected_id": str(selected_id) if selected_id is not None else None,
            },
        )

        self.publish(
            "astro.spectrum.updated",
            {
                "source": self.dataset,
                "artifact_id": artifact_id,
                "dataset_id": dataset_id,
                "selected_id": str(selected_id) if selected_id is not None else None,
                "spectrum_count": spectrum_count,
            },
        )

    def _initialize_settings_dictionary(self):
        self.max_separation = self.config.settings.get("spectrumRadius", 5)

    def get_layout(self):
        if not self._widgets_initialised:
            self._initialize_settings_panel()
            self._bind_selection_runtime_subscriptions()
            self._widgets_initialised = True

        self._request_initial_refresh_once(reason="initial.layout")

        return pn.Column(
            self.message_pane,
            self.figure,
            self.plot_settings_panel,
            sizing_mode="stretch_both",
            min_height=0,
            scroll=False,
        )

    def _save_figure(self, directory_path="data/saved_sources", prefix=None):
        if self.spectrum_object.spectra is not None:
            if self.redshift_column_selector.value != "None":
                redshift_value = self.get_value_from_df(self.redshift_column_selector.value)
                if redshift_value is not None:
                    self.redshift_input.value = redshift_value
            plot_model = False if self._is_euclid_spec else True
            plot_lines = "class" if self.plot_lines_checkbox.value else False
            try:
                fname = f"{prefix + '_' if prefix else ''}{self.panel_name}.png"
                filename = os.path.join(directory_path, fname)
                fig = self.spectrum_object.plot_all_spectra(
                    plot_model=plot_model,
                    plot_lines=plot_lines,
                )
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
                return filename
            except FileNotFoundError:
                print(f"Could not find the saving directory: {directory_path}")

    def _save_data_to_fits(self, directory_path="data/saved_sources"):
        if self.spectrum_object.spectra is not None:
            try:
                self.spectrum_object.export_spectra_to_fits(
                    fname=self.dataset,
                    directory_path=directory_path,
                )
            except FileNotFoundError:
                print(f"Could not find the saving directory: {directory_path}")

    def _get_required_spectrum_columns(self):
        """
        Declare semantic requirements instead of assuming structural columns like 'ra_dec'.
        """
        if self.from_sourceId:
            return [f"{self.dataset}_TargetID"]
        return ["ra", "dec", "id_col"]

    def _validate_spectrum_requirements(self) -> bool:
        """
        Ensure the current mode has the required semantic columns available.
        """
        required = self._get_required_spectrum_columns()

        if self.from_sourceId:
            target_key = required[0]
            if not self.check_required_column(target_key):
                self.get_error_panel("Spectrum unavailable", "Missing column with target ID")
                return False
            return True

        resolved = self.require_columns("ra", "dec", "id_col")
        if resolved["ra"] is None or resolved["dec"] is None:
            self.get_error_panel("Spectrum unavailable", "Missing mapped RA or DEC values")
            return False
        return True

    def _initialize_spectrum_object(self):
        if not self._validate_spectrum_requirements():
            return False

        if self.from_sourceId:
            try:
                self.sourceId = int(self.get_value_from_df(f"{self.dataset}_TargetID"))
                self.ra, self.dec = None, None
            except KeyError:
                self.get_error_panel("Spectrum unavailable", "Missing column with target ID")
                return False
            except (TypeError, ValueError):
                self.get_error_panel("Spectrum unavailable", "Missing target ID")
                return False
        else:
            self.sourceId = None
            self.ra, self.dec = self.get_ra_dec()
            if self.ra is None or self.dec is None:
                self.get_error_panel("Spectrum unavailable", "Missing mapped RA or DEC values")
                return False

        try:
            self.spectrum_object.reset_data(
                ra=self.ra,
                dec=self.dec,
                max_separation=self.max_separation,
                sourceId=self.sourceId,
            )
        except AttributeError:
            if self._is_euclid_spec:
                self.spectrum_object = EuclidSpectraClass(
                    self.ra,
                    self.dec,
                    max_separation=self.max_separation,
                    sourceId=self.sourceId,
                    context=self.context,
                )
            else:
                datasets = (
                    ["DESI-DR1"] if self.dataset == "DESI"
                    else ["BOSS-DR17", "SDSS-DR17"] if self.dataset == "SDSS"
                    else None
                )

                self.spectrum_object = DESISpectraClass(
                    self.ra,
                    self.dec,
                    datasets=datasets,
                    max_separation=self.max_separation,
                    sourceId=self.sourceId,
                    context=self.context,
                )
        return True

    def _add_coordinates_to_shared(self, ra, dec):
        coords_dict = {"ra": list(ra), "dec": list(dec)}
        self.publish_coords(
            f"{self.dataset}",
            coords_dict["ra"],
            coords_dict["dec"],
            dataset_id=self._get_active_dataset_id(),
        )
        return None

    def publish_coords(self, source: str, ra: list[float], dec: list[float], dataset_id: str = "default"):
        if self.artifacts is None:
            return

        selected_id = self._get_selected_id()
        coords = {"ra": ra, "dec": dec}
        coordinate_count = min(len(ra or []), len(dec or []))

        artifact_id = self.artifacts.put(
            type="astro.coords",
            payload=coords,
            dataset_id=dataset_id,
            row_ids=[str(selected_id)] if selected_id is not None else None,
            params={
                "source": source,
                "selected_id": str(selected_id) if selected_id is not None else None,
            },
        )

        self.publish(
            "astro.coords.updated",
            {
                "source": source,
                "artifact_id": artifact_id,
                "dataset_id": dataset_id,
                "selected_id": str(selected_id) if selected_id is not None else None,
                "coordinate_count": coordinate_count,
            },
        )

    def _run_spectrum(self, max_separation=None, reason=None, refresh_signature=None):
        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True

        if max_separation is None:
            max_separation = self.max_separation

        self.publish(
            "astro.spectra.running",
            {
                "source": self.dataset,
                "running": True,
                "panel_id": self.panel_id,
                "reason": reason,
                "dataset_id": self._get_active_dataset_id(),
                "selected_id": self._get_selected_source_id(),
            },
        )

        def callback(future_result=None):
            try:
                if future_result is not None:
                    future_result.result()

                if self.spectrum_object.error_tracker.has_error:
                    message = "# Spectrum unavailable:\n"
                    message += f"## {self.spectrum_object.error_tracker.error_message}"
                    self.message_pane.object = message
                    self.message_pane.visible = True
                    self.figure.objects = [self.get_empty_image()]
                    return

                if self.redshift_column_selector.value != "None":
                    redshift_value = self.get_value_from_df(self.redshift_column_selector.value)
                    if redshift_value is not None:
                        self.redshift_input.value = redshift_value

                self._update_plot()

                ra_list, dec_list = self.spectrum_object.get_coordinates()
                self._add_coordinates_to_shared(ra_list, dec_list)

                self.publish_spectrum_artifact(dataset_id=self._get_active_dataset_id())
                self.message_pane.visible = False

            except Exception as e:
                self.message_pane.object = f"# Error updating spectrum panel\n## {e}"
                self.message_pane.visible = True
                self.figure.objects = [self.get_empty_image()]

            finally:
                self.publish(
                    "astro.spectra.running",
                    {
                        "source": self.dataset,
                        "running": False,
                        "panel_id": self.panel_id,
                        "reason": reason,
                        "dataset_id": self._get_active_dataset_id(),
                        "selected_id": self._get_selected_source_id(),
                    },
                )
                self._finish_refresh(refresh_signature)

        self.run_multithread(
            self.spectrum_object.get_spectra,
            func_kwargs={"max_separation": max_separation, "return_object": True},
            callback=callback,
        )

    def _ensure_available_spectra_attr(self):
        """
        Compatibility shim for EuclidSpectraClass paths that expect
        `available_spectra` to exist.
        """
        spectrum_object = getattr(self, "spectrum_object", None)
        if spectrum_object is None:
            return

        spectra = getattr(spectrum_object, "spectra", None)
        if spectra is None:
            return

        if not hasattr(spectrum_object, "available_spectra") or getattr(spectrum_object, "available_spectra", None) is None:
            try:
                spectrum_object.available_spectra = spectra
            except Exception:
                pass

    def _get_euclid_radius_arcsec(self, default: float = 0.5) -> float:
        try:
            settings = self.config.settings or {}
            eu = settings.get("Euclid_cutout_settings", {}) or {}
            return float(eu.get("radius", default))
        except Exception:
            return float(default)

    def _initialize_settings_panel(self):
        self.retrieve_mode_button = pn.widgets.RadioButtonGroup(
            name="How to retrieve spectrum",
            options=self.mode_options,
            value=self.chosen_mode,
            sizing_mode="stretch_both",
            max_height=40,
        )

        self.max_separation_input = pn.widgets.FloatInput(
            name="Cone Radius [arcsec]",
            value=self._get_euclid_radius_arcsec(0.5),
            step=0.5,
            start=1,
            end=100,
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )

        self.link_to_cutout_checkbox = pn.widgets.Checkbox(
            name="Use radius from Euclid cutout",
            value=False,
            align="center",
        )
        self.max_separation_input.disabled = (self.chosen_mode == self.mode_options[0])
        self.link_to_cutout_checkbox.disabled = (self.chosen_mode == self.mode_options[0])

        self.plot_lines_checkbox = pn.widgets.Checkbox(
            name="Plot Emission/Absorption Lines positions",
            value=not self._is_euclid_spec,
            align="center",
        )
        self.plot_lines_checkbox.disabled = self._is_euclid_spec

        self.plot_model_checkbox = pn.widgets.Checkbox(
            name="Plot Model",
            value=not self._is_euclid_spec,
            align="center",
        )
        self.plot_model_checkbox.disabled = self._is_euclid_spec

        self.smoothing_window_input = pn.widgets.IntInput(
            name="Smoothing Window",
            value=5,
            start=1,
            end=50,
            step=1,
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )
        self.smoothing_function_input = pn.widgets.Select(
            name="Smoothing Function",
            align="center",
            options={"Box": "Box1DKernel", "Gaussian": "Gaussian1DKernel"},
            value="Box1DKernel",
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )

        self.redshift_input = pn.widgets.FloatInput(
            name="Assign Redshift (Same for all Sources)",
            start=0.0,
            end=15,
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )
        self.query_redshift_button = pn.widgets.Button(
            name="Query Redshift",
            align="center",
            button_type="primary",
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )
        self.redshift_column_selector = pn.widgets.Select(
            name="Redshift Column",
            align="center",
            options=["None"] + self.get_column_list(allowed_types=["float"]),
            value="None",
            max_width=200,
            max_height=40,
            sizing_mode="stretch_both",
        )

        self.redshift_input.disabled = not self._is_euclid_spec
        self.query_redshift_button.disabled = not self._is_euclid_spec
        self.redshift_column_selector.disabled = not self._is_euclid_spec

        self.add_param_watch(self.retrieve_mode_button, self._retrieve_mode_cb, what="value")
        self.add_param_watch(self.max_separation_input, self._max_separation_input_cb, what="value")
        self.add_param_watch(self.link_to_cutout_checkbox, self._link_to_cutout_cb, what="value")

        self.add_param_watch_many(
            [self.plot_lines_checkbox, self.plot_model_checkbox],
            self._general_parameter_cb,
            "value",
        )

        self.add_param_watch_many(
            [self.smoothing_function_input, self.smoothing_window_input],
            self._update_smoothing_cb,
            "value",
        )

        self.query_redshift_button.on_click(self._query_redshift_cb)

        self.add_param_watch(self.redshift_input, self._redshift_input_cb, what="value")
        self.add_param_watch(self.redshift_column_selector, self._redshift_column_selector_cb, what="value")

        self.plot_settings_panel = pn.Column(
            self.retrieve_mode_button,
            pn.Row(
                self.max_separation_input,
                pn.Column(pn.Spacer(height=23), self.link_to_cutout_checkbox),
                align="center",
            ),
            pn.Row(self.plot_lines_checkbox, self.plot_model_checkbox),
            pn.Row(self.smoothing_function_input, self.smoothing_window_input, align="center"),
            pn.Row(
                self.redshift_input,
                pn.Column(pn.Spacer(height=10), self.query_redshift_button),
                self.redshift_column_selector,
                pn.Spacer(width=350),
                align="center",
            ),
            scroll=True,
            visible=False,
        )

    def _retrieve_mode_cb(self, event):
        if event.new == "Use TargetId":
            self.from_sourceId = True
            self.chosen_mode = event.new
            self.link_to_cutout_checkbox.disabled = True
            self.max_separation_input.disabled = True
            self._get_unknown_columns([f"{self.dataset}_TargetID"])
            self._change_state_if_unknown_columns()

        elif event.new == "Cone Search":
            self.from_sourceId = False
            self.chosen_mode = event.new
            self.link_to_cutout_checkbox.disabled = False
            self.max_separation_input.disabled = False
            self._request_refresh(reason="spectrum.retrieve_mode.changed")

    def _max_separation_input_cb(self, event):
        if event.new is not None:
            self.max_separation = event.new
            self._request_refresh(reason="spectrum.max_separation.changed")

    def _link_to_cutout_cb(self, event):
        if event.new:
            if not self.from_sourceId and self.events is not None:
                if self._euclid_radius_sub is None:
                    def _on_radius(topic, payload):
                        if not payload:
                            return

                        radius = payload.get("radius")
                        if radius is None:
                            return

                        try:
                            radius = float(radius)
                        except Exception:
                            return

                        if self.max_separation_input.value != radius:
                            self.max_separation_input.value = radius

                    self._euclid_radius_sub = self.events.subscribe(
                        "astro.euclid.radius.changed",
                        _on_radius,
                    )
        else:
            if self._euclid_radius_sub is not None and self.events is not None:
                try:
                    self.events.unsubscribe(self._euclid_radius_sub)
                except Exception:
                    pass
                self._euclid_radius_sub = None

    def _update_smoothing_cb(self, event):
        if getattr(self.spectrum_object, "spectra", None) is not None:
            self.spectrum_object.get_smoothed_spectra(
                kernel=self.smoothing_function_input.value,
                window=self.smoothing_window_input.value,
            )
            self._update_plot()

    def _general_parameter_cb(self, event):
        if getattr(self.spectrum_object, "spectra", None) is not None:
            self._update_plot()

    def _redshift_input_cb(self, event):
        redshift = event.new
        if redshift is not None and getattr(self.spectrum_object, "spectra", None) is not None:
            spectype = "galaxy" if redshift > 0 else "star"
            self.spectrum_object._update_info_spectra("spectype", spectype)
            self.spectrum_object._update_info_spectra("redshift", redshift)
            self.plot_lines_checkbox.disabled = False
            if self.plot_lines_checkbox.value:
                self._update_plot()

    def _query_redshift_cb(self, event):
        if self.spectrum_object.spectra is not None:
            self.query_redshift_button.name = "Query Redshift [Running...]"
            self.spectrum_object.query_specz_table(verbose=True)
            self.spectrum_object.update_info_from_query()
            self.plot_lines_checkbox.disabled = False
            if self.plot_lines_checkbox.value:
                self._update_plot()
        self.query_redshift_button.name = "Query Redshift"

    def _redshift_column_selector_cb(self, event):
        column = event.new
        if column == "None":
            return
        redshift_value = self.get_value_from_df(column)
        if redshift_value is not None:
            self.redshift_input.value = redshift_value

    def _update_plot(self):
        spectrum_object = getattr(self, "spectrum_object", None)
        if spectrum_object is None:
            return

        if getattr(spectrum_object, "spectra", None) is None:
            return

        self._ensure_available_spectra_attr()

        plot_model = self.plot_model_checkbox.value
        plot_lines = "class" if self.plot_lines_checkbox.value else False

        plot = self.spectrum_object.plot_all_spectra_hv(
            plot_model=plot_model,
            plot_lines=plot_lines,
            responsive=True,
        ).opts(
            sizing_mode="stretch_both"
        )

        self.figure.objects = [
            pn.pane.HoloViews(
                plot,
                sizing_mode="stretch_both",
                min_height=0,
            )
        ]

    def _update_max_separation(self, new_separation):
        if self.max_separation_input.value != new_separation:
            self.max_separation_input.value = new_separation

    @param.depends("stage")
    def panel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(
                self.unknown_columns,
                allowed_types=["int"],
                info_text="## Select column with TargetID",
            )
        else:
            return self.plot_panel()

class SEDPlotClass(CustomPlotClass):

    
    available_stages = ["filters_selection", "columns_selection",
                       "error_columns_selection", "units_selection", "plot"]

    stage = param.ObjectSelector(default = available_stages[0], objects=available_stages)

    def __init__(self, data, close_button, extra_features, context = None):
        super().__init__(data, close_button, extra_features, panel_name= "SED",
                         ready_stage = "filters_selection", context = context)


        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self._bind_selection_runtime_subscriptions()

        ##Any changes here requires an update in load_config (verify_SED)
        self.conversion_dictionary = {"AB magnitudes" : lambda f, e : self.mag_to_flux(f,e),
                                      "milliJy" : lambda f, e : (f * 1000, e * 1000),
                                       "microJy" : lambda f, e : (f,e),
                                       "nanoJy"  : lambda f, e : (f / 1000, e / 1000),
                                       "cgs (erg/s/Hz/cm2)" : lambda f, e : (f * 1e23, e * 1e23)
                                     }

        self._bind_dataset_runtime_subscriptions(
            include_loaded=False,
            include_mapping=True,
        )

    def _selection_focus_changed_cb(self, topic, payload):
        if self.stage == "plot":
            self._update_plot(None)

    def _selection_focus_cleared_cb(self, topic, payload):
        if self.stage == "plot":
            self.message_pane.visible = True
            self.figure.object = self.get_empty_image()
    
    
    def filters_selection_panel(self):
        self.filter_data = self.read_photometric_file()
        self._initialize_checkboxes()
        self.checkbox_pane = pn.Column(*self.checkbox_group)
        self._initialize_add_band()
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)

        toolbar = self.get_toolbar()

        body = pn.Column( pn.pane.Markdown("## Select the bands to plot in the SED"),
            pn.Row(pn.Column("## Available Bands", self.checkbox_pane, scroll = True),
                   pn.Column(self.add_band_button, self.add_band_pane, scroll = True)))

        return pn.Column(toolbar,body,
            sizing_mode="stretch_both",  scroll = True, min_height = 300 )


    @staticmethod
    def read_photometric_file(extra_path = ""):
        filepath = os.path.join(extra_path, "data/sed_data/photometric_bands.json")
        with open(filepath, 'r') as f:
            filter_data = json.load(f)
        return filter_data
    
    def write_photometric_file(self, extra_path = ""):
        filepath = os.path.join(extra_path, "data/sed_data/photometric_bands.json")
        with open(filepath, "w") as f:
            json.dump(self.filter_data, f, indent=4) 
    
    def create_checkbox_tooltip(self, band, name, wavlen, fwhm, value = False):
        checkbox = pn.widgets.Checkbox(name=band, value=value, width=150)
        tooltip_text = f"{name},  (Wavelength: {wavlen} Å, FWHM: {fwhm} Å)"
        tooltip_icon = pn.widgets.TooltipIcon(value=tooltip_text, margin=(0, 0, 0, 0))
        return checkbox, tooltip_icon

    def _initialize_checkboxes(self):
        self.checkboxes = {} #dictionary storing all the checkboxs available
        self.checkbox_group = [] #List storing all pairs of checkbox-tooltip
        for band, info in self.filter_data.items():
            value = band in self.config.settings["SED_bands"] if "SED_bands" in self.config.settings else False
            checkbox, tooltip_icon = self.create_checkbox_tooltip(band, info["name"], 
                                                                  info["wavelength"], info["FWHM"],
                                                                  value = value)
            self.checkbox_group.append(pn.Row(checkbox, tooltip_icon, align='center'))
            self.checkboxes[band] = checkbox

    def _initialize_add_band(self):
        self.short_name_input = pn.widgets.TextInput(name = "Short Filter Name", value ="")
        self.full_name_input = pn.widgets.TextInput(name = "Full Filter Name", value = "")
        self.wavelength_input = pn.widgets.FloatInput(name = "Effective Wavelength [Å]")
        self.fwhm_input = pn.widgets.FloatInput(name= "FWHM [Å]", value = 0)
        confirm_button = pn.widgets.Button(name="Confirm", button_type="primary")

        self.add_band_pane = pn.Column(self.short_name_input, self.full_name_input, 
                                       self.wavelength_input, self.fwhm_input, confirm_button, visible=False)
        self.add_band_button = pn.widgets.Button(name="Add Band ▾", button_type="success", max_height = 50)

        self.add_band_button.on_click(self._toggle_add_band_cb)
        confirm_button.on_click(self.add_new_band)

    def update_photometric_file(self, new_band, name, wavlen, fwhm):
        self.filter_data[new_band] = {"name" : name, 
                                      "wavelength" : wavlen,
                                      "FWHM" : fwhm}
        
    def _toggle_add_band_cb(self, event):
        self.add_band_pane.visible = not self.add_band_pane.visible
        self.add_band_button.name = "Add Band ▴" if self.add_band_pane.visible else "Add Band ▾"
        
    
    def add_new_band(self, event):
        try:
            new_band = self.short_name_input.value.strip()
        except AttributeError:
            self.short_name_input.value = "Insert a valid name (no empty string)"
            return
        try:
            name = self.full_name_input.value.strip()
        except AttributeError:
            self.full_name_input.value = "Insert a valid name (no empty string)"
            return
        wavlen = self.wavelength_input.value
        if wavlen <=0:
            print("Insert a valid effective wavelength (>0)")
            return 
        fwhm = self.fwhm_input.value
        if fwhm  < 0:
            print("Insert a valid full width half maximum for the filter (>=0)")
            return 

        if new_band:
            if new_band not in self.checkboxes:
                checkbox, tooltip_icon = self.create_checkbox_tooltip(new_band, name, wavlen, fwhm)
                self.checkbox_group.append(pn.Row(checkbox, tooltip_icon, align='center'))
                self.checkboxes[new_band] = checkbox
                self.checkbox_pane.objects = [*self.checkbox_group]
            self.update_photometric_file(new_band, name, wavlen, fwhm)

        self.short_name_input.value = ""
        self.full_name_input.value = ""
        self.wavelength_input.value = 0.0
        self.fwhm_input.value = 0.0
        self.add_band_pane.visible = False


    def _submit_button_cb(self, event):
        if self.stage == self.available_stages[0]:
            self._filters_selection_continue_cb()
        elif self.stage in (self.available_stages[1], self.available_stages[2]):
            self._columns_selection_continue_cb()
        elif self.stage ==  self.available_stages[3]:
            self._units_selection_continue_cb()
        else:
            self.stage = "plot"

    def _filters_selection_continue_cb(self):
        self.write_photometric_file()
        self.bands_to_plot = [band for band in self.checkboxes.keys() if  self.checkboxes[band].value]
        if self.bands_to_plot:
            self.error_bands_to_plot = [f"err_{band}" for band in self.bands_to_plot]
            self._get_unknown_columns(self.bands_to_plot+self.error_bands_to_plot, settings_key = "SED_bands")
            if any(band in self.unknown_columns for band in self.bands_to_plot):
                self.stage =  self.available_stages[1]
            elif any(err in self.unknown_columns for err in self.error_bands_to_plot): 
                self.stage =  self.available_stages[2]  
            elif self._get_unknown_units():
                self.stage =  self.available_stages[3] 
            else:                                     
                self.stage = "plot" ##awful but changing stage inside a @param.depends stage
                                    ## was not a good idea 
        else:
            print("Please Select at least one band to plot")

    def _columns_selection_continue_cb(self):
        if "SED_bands" not in self.config.settings:
            self.config.settings["SED_bands"] = {}
        for col, widget in self.select_widgets.items():
            selected_value = widget.value
            print(f"{col} --> {selected_value}")
            self.config.settings["SED_bands"][col] = selected_value

        current_idx = self.available_stages.index(self.stage)
        if current_idx == 1: 
            if any(err in self.unknown_columns for err in self.error_bands_to_plot): 
                    self.stage =  self.available_stages[2]  
            elif self._get_unknown_units():
                self.stage =  self.available_stages[3]
            else:
                self.stage = "plot"
        else:
            if self._get_unknown_units():
                self.stage =  self.available_stages[3]
            else:
                self.stage = "plot" 
        
    def _units_selection_continue_cb(self):
        if "SED_units" not in self.config.settings:
            self.config.settings["SED_units"] = {}
        self.config.settings["SED_units"].update({band: widget.value for band, widget in self.select_widgets.items()})
        self.stage = "plot"

    def get_filter_information(self):
        self.wavlen = np.array([self.filter_data[band]["wavelength"] for band in self.bands_to_plot]).flatten()
        self.fwhm = np.array([self.filter_data[band]["FWHM"] for band in self.bands_to_plot]).flatten()
        self.fwhm = np.where(np.logical_and(np.isfinite(self.fwhm), self.fwhm>0), self.fwhm, np.nan) #avoid potential issues

    def units_selection_panel(self, columns_to_select):
        available_units = list(self.conversion_dictionary.keys())
        settings_grid = self._get_selection_widgets_grid(columns_to_select, options = available_units)
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)
        skip_button = pn.widgets.Button(name='Skip', button_type='primary', max_height=120)
        skip_button.disabled = True

        def change_all_selections(event):
            value = event.new
            for col in columns_to_select:
                self.select_widgets[col].value = value

        master_select_widget = pn.widgets.Select(name= "Apply same units to all columns", options=available_units, max_height=120, sizing_mode = "stretch_width")
        
        self.add_param_watch(master_select_widget, change_all_selections, "value")

        toolbar = self.get_toolbar(skip_button=skip_button, submit_button=submit_button)

        body = pn.Column(pn.pane.Markdown("## Select the columns units", sizing_mode = "stretch_width",
                                                  margin=(15,0,15,15)),
                                master_select_widget,
                                settings_grid, scroll = True)
        return pn.Column(toolbar,body, sizing_mode="stretch_both", scroll=True, min_height = 300 )
    
    def _get_unknown_units(self):
        """Return the list of bands for which the units are not present in the config file"""
        if "SED_units" not in self.config.settings:
           return list(self.bands_to_plot)
        return [band for band in self.bands_to_plot if band not in 
                self.config.settings["SED_units"]]

    def get_fluxes_from_selected_source(self):
        selected_source = self.get_selected_source()
        flux = selected_source[[self.config.settings["SED_bands"][col] for col in self.bands_to_plot]].to_numpy().flatten()
        flux_err = []
        for col in  self.error_bands_to_plot:
            try:
                flux_err.append(selected_source[self.config.settings["SED_bands"][col]].iloc[0])
            except KeyError:
                flux_err.append(np.nan)
        return flux, np.array(flux_err).flatten()
 
    
    def get_layout(self):
        self.get_filter_information()
        self._initialize_settings_panel()
        self.flux, self.flux_err = self.get_fluxes_from_selected_source()
        self.flux, self.flux_err = self.clean_fluxes()
        y, y_err = self.convert_to_microjy(self.flux, self.flux_err)
        self.figure.object = self.plot_SED_hv(self.wavlen, y, y_err, self.fwhm)
        self.message_pane.visible = False
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel, scroll = True, sizing_mode = "stretch_both")

    def clean_fluxes(self):
        """It removes missing/stange fluxes"""
        cleaned_flux = []
        cleaned_err = []

        for band, f, e in zip(self.bands_to_plot, self.flux, self.flux_err):
            unit = self.config.settings["SED_units"][band]
            if unit == "AB magnitudes":
                if f > 40 or f < -40:
                   f, e = np.nan, np.nan
            else:
                if f < 0:
                    f, e = np.nan, np.nan
            cleaned_flux.append(f)
            cleaned_err.append(e)

        return np.array(cleaned_flux), np.array(cleaned_err)
        
    
    @staticmethod
    def plot_SED_hv(wavlen, flux, flux_err, fwhm, redshift=0, output_units = "fnu",
                    arrow_scale = 0.6):

        mask = np.logical_and(np.isfinite(wavlen), np.isfinite(flux))
        if np.sum(mask) < 1:
            return pn.pane.Markdown(f"## There are no available points to plot. All specified bands have NaN values") 
        x = wavlen[mask] / (1 + redshift)
        y = flux[mask]
        err_y = flux_err[mask]
        fwhm = fwhm[mask]
    
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
    
        scatter = hv.Scatter((x, y), kdims='wavelength', vdims="Flux").opts(
            color="red", fill_color = None, marker="o", size=9, active_tools=[])
      
        xerrbars = hv.ErrorBars((x, y, fwhm / 2, fwhm / 2), kdims= 'wavelength', vdims=["Flux", "xneg", "xpos"], horizontal = True).opts(color = "black",
                                                                                        lower_head = None, upper_head = None, active_tools =[])
                                                                                                                   
        is_upper_limit = err_y < 0
        has_larger_errors = err_y > y #These could also be considered as upper limits...
        good_measure = ~np.logical_or(is_upper_limit, has_larger_errors)
    
        ybars = hv.ErrorBars((x[good_measure ], y[good_measure ], err_y[good_measure ], err_y[good_measure ]), kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(
                color="black", line_width = 1.5, active_tools =[])
        
    
        arrow_length = arrow_scale * y[has_larger_errors] #all arrows have the same length in logy scale
        larger_errors = hv.ErrorBars(
            (x[has_larger_errors], y[has_larger_errors], np.full(np.sum(has_larger_errors), arrow_length), err_y[has_larger_errors]),
            kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(color="black",lower_head = NormalHead(size=8),
                                                                     line_width = 1.5, active_tools =[])
     
        arrow_length = arrow_scale * y[is_upper_limit] #all arrows have the same length in logy scale
        upper_limits = hv.ErrorBars(
            (x[is_upper_limit], y[is_upper_limit], np.full(np.sum(is_upper_limit), arrow_length), np.zeros(np.sum(is_upper_limit))),
            kdims='wavelength', vdims=["Flux", "yneg", "ypos"]).opts(color="black",lower_head = NormalHead(size=8),
                                                                     line_width = 1.5, active_tools =[])
        
    
        plot = scatter * xerrbars * ybars * upper_limits * larger_errors
        xlabel = 'Rest-frame Wavelength [Å]' if redshift > 0 else 'Observed-frame Wavelength (rest) [Å]'
        ylabel = "Flux [erg/s cm-2]" if output_units == "nufnu" else "Flux density [μJy]" 
        hooks = [] if output_units == "nufnu" else [SEDPlotClass.add_magnitude_axis]

        return plot.opts(
            xlabel=xlabel,
            logx = True, logy = True, 
            xlim = (xmin/2, xmax*2),
            ylim = (ymin/3, ymax*3),
            ylabel = ylabel, show_grid=True,
            active_tools =[],
            hooks = hooks
        )
    
    @staticmethod
    def add_magnitude_axis(plot, element):
        """Bokeh hook to add secondary Y-axis with magnitude scale"""
        fig = plot.state
        y_start, y_end = fig.y_range.start, fig.y_range.end
        mag_start, _ = SEDPlotClass.flux_to_mag(y_start, 0)
        mag_end, _  =  SEDPlotClass.flux_to_mag(y_end, 0)
        
        #Note mag_start and end are reversed compared to fluxes
        fig.extra_y_ranges = {"mag": Range1d(start=mag_start, end=mag_end)}
        mag_axis = LinearAxis(y_range_name="mag", axis_label="AB Magnitude",
                              major_label_text_color="black", axis_label_text_color="black")
        fig.add_layout(mag_axis, 'right')
    
    @staticmethod
    def plot_SED(wavlen, flux, flux_err, fwhm, redshift=0, output_units = "fnu",
                 arrow_scale = 0.6):
        
        mask = np.logical_and(np.isfinite(wavlen), np.isfinite(flux))
        if np.sum(mask) < 1:
            return None
        x = wavlen[mask] / (1 + redshift)
        y = flux[mask]
        err_y = flux_err[mask]
        fwhm = fwhm[mask]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        fig, ax = plt.subplots(figsize = (8,6))

        is_upper_limit = err_y < 0
        has_larger_errors = err_y > y #These could also be considered a upper limits...
        good_measure = ~np.logical_or(is_upper_limit, has_larger_errors)
    

        ax.errorbar(x[good_measure], y[good_measure], yerr=err_y[good_measure], xerr= fwhm[good_measure]/2,  
                    ls ="none",  ecolor ="k", markeredgecolor = "r" , marker = "o", markerfacecolor="none")
        
        arrow_length = arrow_scale * y[~good_measure]
        ax.errorbar(x[~good_measure], y[~good_measure], yerr = arrow_length, xerr= fwhm[~good_measure]/2,  
                    ls ="none",  ecolor ="k", markeredgecolor = "r" , marker = "o", markerfacecolor="none", uplims = True)
        ax.errorbar(x[has_larger_errors], y[has_larger_errors], 
                    yerr=np.vstack([np.zeros(np.sum(has_larger_errors)), err_y[has_larger_errors]]),  
                    ls ="none",  ecolor ="k",  marker = "none")
        
        if output_units != "nufnu":
            mag_ax = ax.secondary_yaxis("right", functions = (lambda x:  -2.5*np.log10(x) + 23.9,
                                                  lambda x:  10**((23.9 - x)/2.5)))
            mag_ax.set_ylabel("AB magnitudes", fontsize =12)
            mag_ax.invert_yaxis()
            mag_ax.yaxis.set_major_formatter(ScalarFormatter())
            mag_ax.ticklabel_format(style='plain', axis='y')

       

        xlabel = r'$\lambda_{rest} ~[{Å}] $' if redshift > 0 else r'$\lambda_{obs} ~[{Å}]$'
        ylabel = "Flux [erg/s cm-2]" if output_units == "nufnu" else "Flux density [μJy]" 
        ax.set_xlabel(xlabel, fontsize =12)
        ax.set_ylabel(ylabel, fontsize =12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin/2, xmax*2)
        ax.set_ylim(ymin/3, ymax*3)

        return fig

    @staticmethod
    def mag_to_flux(mag, err_mag):
        """Converts AB magnitudes in flux densities in microJy"""    
        flux = 10**((23.9 - mag)/2.5)
        err_flux = flux * err_mag * np.log(10)
        return flux, err_flux
    
    @staticmethod
    def flux_to_mag(flux, err_flux):
        """Converts fluxes in microJansky to AB magnitudes"""
        mag = -2.5*np.log10(flux) + 23.9
        err_mag = (err_flux/flux)/np.log(10)
        return mag, err_mag
    
    
    def convert_to_microjy(self, flux, err_flux):
        flux_converted =[]
        err_converted = []
        for band, f ,e in zip(self.bands_to_plot, flux, err_flux):
            unit = self.config.settings["SED_units"][band]
            try:
                fc, ec = self.conversion_dictionary[unit](f, e)
                flux_converted.append(fc)
                err_converted.append(ec)
            except KeyError:
                print(f"I cannot find this unit: {unit}")
                flux_converted.append(np.nan)
                err_converted.append(np.nan)

        return np.array(flux_converted), np.array(err_converted)
    
    def convert_to_output_units(self, wav, flux, flux_err, output_units):
        """wav in angstrom, flux in microJy"""
        #TODO do it more genereal
        if output_units == "nufnu": 
            flux, flux_err = flux /1e23, flux_err/ 1e23
            flux, flux_err = flux*2.998e18/wav, flux_err*2.998e18/wav
        return flux, flux_err
    

    def _initialize_settings_panel(self):     
        output_units = {"microJy" : "fnu", "erg/s/cm2" : "nufnu"}  #first one should be always microJy                                                                           
        self.unit_selector = pn.widgets.Select(name = "Output Units", options = output_units, max_width = 200, max_height = 40, 
                                               sizing_mode="stretch_both")
        
        self.add_param_watch(self.unit_selector, self._update_plot, "value")

        self.plot_settings_panel = pn.Column(self.unit_selector, visible = False)
                                                                                                                                                   

    def _update_plot(self, event):
        self.flux, self.flux_err = self.get_fluxes_from_selected_source()
        self.flux, self.flux_err = self.clean_fluxes()
        y, y_err = self.convert_to_microjy(self.flux, self.flux_err)
        y, y_err = self.convert_to_output_units(self.wavlen, y, y_err, output_units = self.unit_selector.value)
        self.figure.object = self.plot_SED_hv(self.wavlen, y, y_err, self.fwhm, output_units =self.unit_selector.value)
        self.message_pane.visible = False
        

    def _save_figure(self, directory_path = "data/saved_sources", prefix = None):
        y, y_err = self.convert_to_microjy(self.flux, self.flux_err) #Should already be clean
        y, y_err = self.convert_to_output_units(self.wavlen, y, y_err, output_units = self.unit_selector.value)
        fig = self.plot_SED(self.wavlen, y, y_err, self.fwhm, output_units = self.unit_selector.value)
        if fig is not None:
            try:
                fname = f"{prefix + '_' if prefix else ''}{self.panel_name}.png"
                filename = os.path.join(directory_path,fname)
                fig.savefig(filename, bbox_inches = "tight")
                plt.close(fig)
                return filename
            
            except FileNotFoundError:
                print(f"Cold not find the saving directory: {directory_path}")     

    def get_toolbar(self, skip_button=None, submit_button=None):

        if self.stage == self.available_stages[0]:
            toolbar = pn.Row(
                pn.Spacer(width=25),
                self.close_button,
                submit_button,
                max_height=50)
        
        if self.stage == self.available_stages[1]:
            toolbar = pn.Row(
                pn.Spacer(width=25), 
                self.close_button, 
                skip_button, 
                submit_button, 
                max_height=50)

        elif self.stage == self.available_stages[2]:
            toolbar = pn.Row(
                pn.Spacer(width=25), 
                self.close_button, 
                skip_button, 
                submit_button, 
                max_height=50)

        elif self.stage == self.available_stages[3]:
            toolbar = pn.Row(
                pn.Spacer(width=25), 
                self.close_button, 
                skip_button, 
                submit_button, 
                max_height=50)
        else:
            toolbar = pn.Row(
                pn.Spacer(width=25,),
                self.close_button,
                self.plot_settings_button,
                max_height=50)

        return toolbar

    def plot_panel(self):
        self.layout = self.get_layout()
        toolbar = self.get_toolbar()
        return pn.Column(toolbar, self.layout,
                         sizing_mode="stretch_both", min_height =450,)  


    @param.depends("stage")
    def panel(self):

        if self.stage == self.available_stages[0]:
            return self.filters_selection_panel()
        
        if self.stage == self.available_stages[1]:
            columns_to_select = [i for i in self.bands_to_plot if i in self.unknown_columns]
            return self.columns_selection_panel(columns_to_select, skippable=False, allowed_types = ["float"],
                                        info_text= "## Select columns with flux values")
        elif self.stage == self.available_stages[2]:
            columns_to_select = [i for i in self.error_bands_to_plot if i in self.unknown_columns]
            return self.columns_selection_panel(columns_to_select, skippable=True, allowed_types = ["float"],
                                    info_text= "## Select columns with flux error values")
        elif self.stage == self.available_stages[3]:
            units_to_select = self._get_unknown_units()
            return self.units_selection_panel(units_to_select)
        else:
            return self.plot_panel()




class RadioClass(CustomPlotClass):
    
    def __init__(self, data, close_button, extra_features, dataset, context = None):
        super().__init__(data, close_button, extra_features, context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        
        self.dataset = dataset
        self._initialize_source()
        self.radius = 20

        self._bind_selection_runtime_subscriptions()

        self._bind_dataset_runtime_subscriptions(
            include_loaded=False,
            include_mapping=True,
        )

    def _initialize_source(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.message_pane.visible = True
            self.message_pane.object = ["## Missing Ra and Dec"]   

    def _selection_focus_changed_cb(self, topic, payload):
        self._initialize_source()
        self._run_radio(radius=self.radius)

    def _selection_focus_cleared_cb(self, topic, payload):
        self.message_pane.visible = True
        self.message_pane.object = "## Missing RA and Dec"
        self.figure.object = self.get_empty_image()

    def get_layout(self):
        self._initialise_widgets()
        self._run_radio(radius = self.radius)
        return  pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                          scroll = True, sizing_mode = "stretch_both")
    
    def _initialise_widgets(self):

        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius, 
                                                  step = 1, start = 1, end = 100, max_width = 200,
                                                  sizing_mode="stretch_both", max_height =30)
        
        self.add_param_watch(self.radius_input, self._update_radius, what="value")
        
        self.plot_settings_panel = pn.Column(self.radius_input, 
                                             scroll = True, visible = False)
        
    def _update_radius(self, event):
        if event.new: 
            self.radius = event.new
            self._run_radio(radius = self.radius)
        else:
            print("Input a valid value for radius")

    def _run_radio(self, radius = 20):
        self.message_pane.visible = True
        if self.context and self.context.events:
            self.context.events.publish(
                "astro.radio.running",
                {"source": getattr(self, "dataset", "Radio"), "running": True, "panel_id": self.panel_id},
            )

        def callback(future_obj = None):
            if self.context and self.context.events:
                self.context.events.publish(
                    "astro.radio.running",
                    {"source": getattr(self, "dataset", "Radio"), "running": False, "panel_id": self.panel_id},
                )
            print("I am calling the radio callback ")
            result = future_obj.result() 
            if result is None:
                self.message_pane.object = f"## {self.dataset} cutout query failed"
                self.message_pane.visible = True #probably already visible
            elif result is not None:
                print(result.shape)
                self.figure.object = self.get_radio_figure(result)
                self.message_pane.visible = False

        if self.dataset == "VLASS":
            print("I am running VLASS")
            self.run_multithread(VLASS_cutout, 
                             func_kwargs = {"ra" : self.ra, "dec" : self.dec,
                                            "radius" : radius},
                             callback=callback)
        elif self.dataset == "LoTSS":
            self.run_multithread(LoTSS_cutout, 
                             func_kwargs = {"ra" : self.ra, "dec" : self.dec,
                                            "radius" : self.radius},
                             callback=callback)

    def get_radio_figure(self, data):
        self.image_height, self.image_width,  = data.shape[:2]
        bounds = (0, 0, self.image_height, self.image_width)
        image = hv.Image(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         )
        return image
    
class SDSSClass(CustomPlotClass):
    
    def __init__(self, data, close_button, extra_features, dataset, context = None):
        super().__init__(data, close_button, extra_features, context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self._bind_selection_runtime_subscriptions()

        self.dataset = dataset
        self._initialize_source()
        self.radius = 25.6

    def _initialize_source(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.message_pane.visible = True
            self.message_pane.object = ["## Missing Ra and Dec"]   

    def _selection_focus_changed_cb(self, topic, payload):
        self._initialize_source()
        self._run_sdss(radius=self.radius)

    def _selection_focus_cleared_cb(self, topic, payload):
        self.message_pane.visible = True
        self.message_pane.object = "## Missing RA and Dec"
        self.figure.object = self.get_empty_image()

    def get_layout(self):
        self._initialise_widgets()
        self._run_sdss(radius = self.radius)
        return  pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                          scroll = True, sizing_mode = "stretch_both")
    
    def _initialise_widgets(self):

        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius, 
                                                  step = 1, start = 1, end = 100, max_width = 200,
                                                  sizing_mode="stretch_both", max_height =30)
        
        self.add_param_watch(self.radius_input, self._update_radius, what="value")
        
        self.plot_settings_panel = pn.Column(self.radius_input, 
                                             scroll = True, visible = False)
        
    def _update_radius(self, event):
        if event.new: 
            self.radius = event.new
            self._run_sdss(radius = self.radius)
        else:
            print("Input a valid value for radius")

    def _run_sdss(self, radius = 25.6):
        self.message_pane.visible = True
        if self.context and self.context.events:
            self.context.events.publish(
                "astro.sdss.running",
                {"running": True, "panel_id": self.panel_id},
            )

        def callback(future_obj = None):
            print("Ended SDSS query")
            if self.context and self.context.events:
                self.context.events.publish(
                    "astro.sdss.running",
                    {"running": False, "panel_id": self.panel_id},
                )
            result = future_obj.result() 
            if result is None:
                self.message_pane.object = f"## {self.dataset} cutout query failed"
                self.message_pane.visible = True #probably already visible
            elif result is not None:
                self.figure.object = self.get_sdss_figure(result)
                self.message_pane.visible = False

        if self.dataset == "SDSS":
            self.run_multithread(SDSS_cutout, 
                             func_kwargs = {"ra" : self.ra, "dec" : self.dec,
                                            "radius" : radius},
                             callback=callback)


    def get_sdss_figure(self, data):
        print("Entering here!!!")
        #self.image_height, self.image_width = data.shape[:2]
        #bounds = (0, 0, self.image_height, self.image_width)
        image = hv.Image(data).opts(active_tools =[], toolbar=None,
                                    padding = 0,
                                    border = 0,
                                    framewise = True,
                                    xaxis=None, 
                                    yaxis=None,
                                    )
        return image



class AladinClass(CustomPlotClass):
    # Available surveys here: https://aladin.cds.unistra.fr/hips/list
    
    def __init__(self, data, close_button, extra_features, context = None):
        super().__init__(data, close_button, extra_features, panel_name = "Aladin Panel", context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self.figure = pn.pane.HTML(
            "",
            sizing_mode="stretch_both",
            margin=0,
            styles={
                "width": "100%",
                "height": "100%",
                "overflow": "hidden",
                "flex": "1 1 auto",
            },
        )

        self._bind_selection_runtime_subscriptions()

        self._bind_dataset_runtime_subscriptions(
            include_loaded=False,
            include_mapping=True,
        )
    

    def _selection_focus_changed_cb(self, topic, payload):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.get_error_panel("Aladin panel unavailable", "Missing RA or DEC value")
            return
        self._update_image(None)

    def _selection_focus_cleared_cb(self, topic, payload):
        self.get_error_panel("Aladin panel unavailable", "No selected source")

    @staticmethod
    def make_iframe_html(survey_id, ra, dec):
        srcdoc = make_srcdoc_aladin_lite(survey_id=survey_id, ra=ra, dec=dec)
        srcdoc_escaped = html.escape(srcdoc, quote=True)
        return f"""
        <div style="width:100%; height:100%; margin:0; padding:0; overflow:hidden;">
            <iframe
                srcdoc="{srcdoc_escaped}"
                style="display:block; width:100%; height:100%; border:none;"
            ></iframe>
        </div>
        """
    
    def _initialise_widgets(self):

        xray_surveys = {
            "eROSITA (rate, color)" :  	"erosita/dr1/rate/rgb",
            "eROSITA 0.2-0.6 keV (count)" : "erosita/dr1/count/021",
            "eROSITA 0.6-2.3 keV (count)" : "erosita/dr1/count/022",
            "eROSITA 2.3-5 keV (count)" :   "erosita/dr1/count/023",
            "Swift XRT (exposure)" :  	"nasa.heasarc/P/Swift/XRT/exp", 
            "XMM (color)" :  "xcatdb/P/XMM/PN/color",
            "XMM (0.5-1 keV)" : "xcatdb/P/XMM/PN/eb2",
            "XMM (1-2 keV)" : "xcatdb/P/XMM/PN/eb3",
            "XMM (2-4.5 keV)" : "xcatdb/P/XMM/PN/eb4",
            }
        
        optical_surveys = {
            "SDSS (color)" : "CDS/P/SDSS9/color",
            "DSS2 (color)" : "P/DSS2/color",
            "Pan-STARRS (color)" : "P/PanSTARRS/DR1/color-z-zg-g",
            "GALEX (color)": "P/GALEXGR6/AIS/color",
            "DESI Legacy Survey (color)" : "CDS/P/DESI-Legacy-Surveys/DR10/color",
            "DES (color)" : "CDS/P/DES-DR2/ColorIRG",
            }

        ir_surveys = {       
            "AllWISE (color)": "P/allWISE/color",
            "2MASS (color)": "P/2MASS/color",
            "Herschel SPIRE (color)" : "ESAVO/P/HERSCHEL/SPIRE-color",
            "Spitzer IRAC (color)" : "CDS/P/SPITZER/color",
            "Euclid Q1 (color)" : "CDS/P/Euclid/Q1/color",
        }
    

        self.survey_selector = pn.widgets.Select(
            name="Survey",
            value="P/DSS2/color",
            groups={
                "X-rays": xray_surveys,
                "Optical/UV": optical_surveys,
                "IR": ir_surveys,
            },
            sizing_mode="stretch_width",
            height=38,
            max_height=38,
            margin=0,
        )
        
        self.add_param_watch(self.survey_selector, self._update_image, what="value")


        self.plot_settings_panel = pn.Row(
            self.survey_selector,
            visible=False,
            sizing_mode="stretch_width",
            height=50,
            max_height=50,
            margin=0,
            styles={"flex": "0 0 auto"},
        )


    def _update_image(self, event):
        self.message_pane.visible = False
        self.figure.object = self.make_iframe_html(self.survey_selector.value, 
                                          self.ra, self.dec)

    def get_layout(self):
        self._initialise_widgets()
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.get_error_panel("Aladin panel unavailable", "Missing RA or DEC value")

        self._update_image(None)

        return pn.Column(
            self.message_pane,
            self.plot_settings_panel,
            self.figure,
            sizing_mode="stretch_both",
            margin=0,
        )

######

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

        from astronomicAL.extensions.gui_analyser import (
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