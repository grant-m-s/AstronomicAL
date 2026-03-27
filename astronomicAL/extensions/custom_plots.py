from __future__ import annotations

import holoviews as hv

import numpy as np
import os
import html
import pandas as pd
import panel as pn
import json
import param
import uuid
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import concurrent.futures 
from panel.io import save
from bokeh.models import  NormalHead
from bokeh.models import Range1d, LinearAxis
from astronomicAL.utils.optimise import matches_type
from astronomicAL.extensions.astro_data_utility import DESISpectraClass, EuclidCutoutsClass, EuclidSpectraClass
from astronomicAL.extensions.astro_data_utility import VLASS_cutout, LoTSS_cutout, make_srcdoc_aladin_lite, SDSS_cutout


import uuid
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Iterable

try:
    from astronomicAL.platform.events import Subscription
except Exception:  # pragma: no cover
    Subscription = Any  # type: ignore


@dataclass
class _ManagedJob:
    key: str
    handle: Any  # JobHandle from JobManager

def get_customplot_dict():

    plot_dict = {
        
        "Euclid Cutout" : lambda data, src, close_button, context : EuclidPlotClass(data, src, close_button,
                                                           extra_features=[], context=context),

        "DESI Spectra"  : lambda data, src, close_button, context : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="DESI", context=context), 

        "Euclid Spectra"  : lambda data, src, close_button, context : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="EuclidSpec", context=context), 

        "SDSS Spectra"  : lambda data, src, close_button, context : SpectrumPlotClass(data, src, close_button,
                                                            extra_features=[], dataset="SDSS", context=context),

        "BroadBand SED"  : lambda data, src, close_button, context : SEDPlotClass(data, src, close_button,
                                                            extra_features=[], context=context),

        "Notes Panel"  : lambda data, src, close_button, context : LogBookClass(data, src, close_button,
                                                            extra_features=[], context=context),
        
        "Aladin Lite"  : lambda data, src, close_button, context : AladinClass(data, src, close_button,
                                                            extra_features=[], context=context),                                                  
        
        "VLASS Cutout"  : lambda data, src, close_button, context : RadioClass(data, src, close_button,
                                                            extra_features=[], dataset="VLASS", context=context),
        
        "LoTSS Cutout"  : lambda data, src, close_button, context : RadioClass(data, src, close_button,
                                                            extra_features=[], dataset="LoTSS", context=context),

        "Event Monitor": lambda data, src, close_button, context : EventMonitorClass(
            data, src, close_button, extra_features=[], context=context
        ),

        "spec_analyser": lambda data, src, close_button, context : SpecAnalyser(
            data, src, close_button, extra_features=[], context=context
        )

        #"SDSS Cutout"  : lambda data, src, close_button, context : SDSSClass(data, src, close_button,
        #                                                    extra_features=[], dataset="SDSS", context=context)                                                                                                      

    }

    return plot_dict


class CustomPlotClass(param.Parameterized):

    available_stages = ["columns_selection", "plot"]

    stage = param.ObjectSelector(default = "columns_selection", objects = available_stages)
    
    def __init__(self, data, src, close_button, extra_features, 
                 panel_name = "custom_plot",
                 ready_stage = "plot",
                 context = None,
                 **params):
        super().__init__(**params)

        self.df = data
        self.src = src
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

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if self.extra_features:
            self._get_unknown_columns(columns_needed = self.extra_features)
            self._change_state_if_unknown_columns(ready_stage=ready_stage)
        else:
            self.stage = ready_stage
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.message_pane = pn.pane.Markdown("## Loading...", sizing_mode="stretch_width", height = 80)
        self.plot_settings_button = pn.widgets.Button(name="Open Settings", button_type="primary", max_height = 40, max_width=100, sizing_mode="stretch_both" )
        self.plot_settings_button.on_click(self._toggle_settings_panel)
        self.plot_settings_panel = pn.Column(visible = False)

        if self.close_button is not None:
            try:
                self.close_button.on_click(lambda _e: self.dispose())
            except Exception as e:
                # Some button types / contexts may not support on_click in tests
                print("CustomPlotClass.dispose() Errored", e)
                pass

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
        """
        Submit a background job through context.jobs.

        - If context.jobs is unavailable, runs synchronously (keeps old behavior working).
        - Adds cooperative cancellation token via kwarg `cancel_token` when using JobManager.
        - Tracks job handles so dispose() can cancel them.
        """
        # Default dedupe key scoped to this panel instance
        if key is None:
            key = f"{self.panel_id}:{title}"

        if self.jobs is None:
            # Synchronous fallback
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
        """Best-effort cancellation of all jobs started by this panel."""
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
        """
        Subscribe to an event topic through context.events.
        Tracks subscriptions so dispose() can unsubscribe.

        Callback signature: (topic, payload).
        """
        if self.events is None:
            return None
        sub = self.events.subscribe(topic, callback)
        self._subscriptions.append(sub)
        return sub

    def unsubscribe_all(self) -> None:
        """Unsubscribe all tracked subscriptions."""
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
        """Publish an event if the bus exists."""
        if self.events is None:
            return
        try:
            self.events.publish(topic, payload)
        except Exception:
            traceback.print_exc()

    # -----------------------
    # Artifact helpers (optional conveniences)
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
        """
        Store an artifact in context.artifacts and return artifact_id.
        If no artifact store exists, returns None.
        """
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


    # -----------------------
    # Lifecycle
    # -----------------------
    def _dispose_impl(self) -> None:
        """
        Subclass-specific cleanup hook.
        Override in subclasses if needed.
        """
        return

    def dispose(self) -> None:
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        print(f"[dispose] {self.__class__.__name__} panel_id={getattr(self, 'panel_id', None)}")

        # 1) subclass hook first (still has access to state)
        try:
            self._dispose_impl()
        except Exception:
            pass

        # 2) Remove any tracked bokeh callbacks (including src.on_change)
        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        try:
            self.remove_all_param_watches()
        except Exception:
            pass

        # 3) Remove column-selection callbacks / widgets if present
        if hasattr(self, "remove_column_selection"):
            try:
                self.remove_column_selection()
            except Exception:
                pass

        # 4) Cancel outstanding jobs submitted via context.jobs
        try:
            self.cancel_jobs()
        except Exception:
            pass

        # 5) Unsubscribe EventBus subscriptions
        try:
            self.unsubscribe_all()
        except Exception:
            pass

        # 6) Shut down legacy per-panel executors if any remain
        # (Eventually you should remove these entirely and rely on context.jobs)
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
        """
        Compatibility wrapper used by existing panels.

        - Runs `fn(**func_kwargs)` in context.jobs thread pool if available.
        - Calls `callback(future_like)` on success where future_like.result() returns result.
        """
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
            # Euclid code doesn't currently use cancel_token; kept for future
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
        """
        Register owner.param.watch(...) and track it for later cleanup.
        Returns the Watcher object (or None on failure).
        """
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
        """Idempotently unwatch everything added via add_param_watch()."""
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
        """
        Register a Bokeh on_change callback and track it for cleanup.
        """
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            # If this is called in a non-bokeh context/tests, fail silently
            pass

    def unwatch_all_bokeh(self) -> None:
        """
        Remove all tracked Bokeh callbacks.
        """
        for model, attr, callback in list(self._bokeh_on_change):
            try:
                # ColumnDataSource and other Bokeh Models support remove_on_change
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

    def get_selected_source(self):
        if self.src is None:
            return None
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            return pd.DataFrame(self.src.data, columns=cols, index=[0])
        return None
    
    def get_value_from_df(self, column):
        selected_source = self.get_selected_source()
        if (selected_source is not None) and self.check_required_column(column):
            return selected_source[column][0]
        return None
            
    def get_ra_dec(self, err_message = "No ra and dec available for this source"):
        ra_dec = self.get_value_from_df("ra_dec")
        if ra_dec is not None:
            ra = float(ra_dec[: ra_dec.index(",")])
            dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print(err_message)
            ra, dec = None, None
        return ra, dec
    
    def _get_selected_id(self):
        return self.get_value_from_df(self.config.settings["id_col"])

    def check_required_column(self, column):
        return column in self.df.columns
    

    def get_column_list(
        self,
        excluded_columns=("id_col", "ra_dec", "label_col"),
        excluded_types=("object",),
        allowed_types=None,
    ):
        cols = list(getattr(self.df, "columns", []))

        # remove excluded columns by name or config alias
        for excluded_col in excluded_columns:
            col_name = self.config.settings.get(excluded_col, excluded_col)
            if col_name in cols:
                cols.remove(col_name)

        # type filtering
        if allowed_types:
            cols = [c for c in cols if matches_type(self.df[c].dtype, allowed_types)]
        if excluded_types:
            cols = [c for c in cols if not matches_type(self.df[c].dtype, excluded_types)]

        return cols


    def _get_selection_widgets_grid(self, columns_to_select, default_values = None, 
                                    options = None, allowed_types = None):
        settings_grid = pn.GridBox(ncols=3, sizing_mode = "stretch_width", scroll = True)  
        self.select_widgets = {}
        if options is None:
            options = self.get_column_list(excluded_columns = ["ra_dec", "label_col"],
                              excluded_types = ["object"], allowed_types = allowed_types)
        if len(columns_to_select) > 0:
            for i, col in enumerate(columns_to_select):
                select_widget = pn.widgets.Select(name= col, options=options, max_height=120, sizing_mode = "stretch_width")
                if (default_values is not None) and (i < len(default_values)):
                    select_widget.value = default_values[i]
                settings_grid.append(select_widget)
                self.select_widgets[col] = select_widget
        return settings_grid
        

    def columns_selection_panel(self, columns_to_select, skippable = False,
                                options = None, allowed_types = None,
                                info_text = None):
        settings_grid = self._get_selection_widgets_grid(columns_to_select, options = options, 
                                                         allowed_types = allowed_types)
        
        submit_button = pn.widgets.Button(name='Confirm', button_type='primary', max_height=120)
        submit_button.on_click(self._submit_button_cb)
        skip_button = pn.widgets.Button(name='Skip', button_type='primary', max_height=120)
        skip_button.on_click(self._skip_button_cb)
        if not skippable:
            skip_button.disabled = True
        if info_text is not None:
           card_content = pn.Column(pn.pane.Markdown(info_text, sizing_mode="stretch_width", margin=(15,0,15,15)), 
                                           settings_grid)
        else:
           card_content = settings_grid

        toolbar = self.get_layout(submit_button=submit_button, skip_button=skip_button)
        return pn.Column(toolbar, card_content,
                                sizing_mode="stretch_both", scroll=True, min_height = 300 )

    
    def _get_unknown_columns(self, columns_needed, settings_key = None):
        
        """
        Check if required columns exist in config.settings or config.main_df.

        columns_needed : list
            Columns that are required.
        settings_key : str or None, optional
            If provided, checks within config.settings[settings_key].keys().
            Otherwise, checks directly against config.settings.
        """
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
                            
    def _change_state_if_unknown_columns(self,  unknown_stage  = "columns_selection",
                                          ready_stage = "plot"):
        """ Manages  the change of stage depending on the presence or not of unknown_columns.
        
        unknown_stage : str, optional
            Stage to set if unknown columns are present (default: 'columns_selection').
        ready_stage : str, optional
            Stage to set if all columns are known (default: 'plot').
        
        """
        if hasattr(self, "unknown_columns"):
            if self.unknown_columns:
                self.stage = unknown_stage
            else:
                self.stage = ready_stage
        else:
            print("The unknown_columns attribute was not initialized, not changing Stage")
    
    def _save_panel(self, directory_path = "data/saved_sources", 
                    save_fits_files = True, 
                    prefix = None, 
                    ): 
        paths = {}
        if self.stage == "plot":
            try:
               paths["figure"] = self._save_figure(directory_path = directory_path, prefix = prefix)
            except AttributeError:
                print(f"{self.panel_name} has no _save_panel_method")
            
            if save_fits_files:
                try:
                    paths["fits_file"] = self._save_data_to_fits(directory_path = directory_path)
                    #paths["fits_file"] is currently always None
                except AttributeError:
                    pass
        return paths
    
    @staticmethod
    def get_empty_image():
        """Returns a completely white image to update the previous one if the query fails"""
        return hv.Image(np.ones((10,10))).opts(active_tools =[], 
                                            clim = (0,1), toolbar=None,
                                            padding = 0,border = 0,framewise = True, xaxis=None, 
                                            yaxis=None, cmap = "grey")
    
    def get_error_panel(self, message_1, message_2):
        message = f"# {message_1}:\n"  
        message += f"## {message_2}"
        self.message_pane.object = message
        self.message_pane.visible = True 
        self.figure.objects = [self.get_empty_image()]

    def remove_src_listener(self):
        """Removes the callback to a change in the selected source"""
        if self.src is not None and hasattr(self, "_src_callback"):
            try:
                self.src.remove_on_change("data", self._src_callback)
                print(f"[{self.panel_id}] Listener removed")
            except Exception as e:
                print(f"[{self.panel_id}] Error removing src listener: {e}")

    def remove_column_selection(self):
        if hasattr(self, "unknown_columns"):
            for col in self.unknown_columns:
                if col in self.config.settings:
                    del self.config.settings[col]
            print(f"[{self.panel_id}] unknown columns selected removed from config")

    ##Example method
    def plot(self, N=20):
        self.message_pane.visible = True
        coords = [(i, np.random.random()) for i in range(N)]
        scatter = hv.Scatter(coords).opts(color='black', marker='+')
        self.figure.object = scatter
        self.message_pane.visible = False

    ##Example method
    def get_layout(self):
        points_input = pn.widgets.IntInput(name="Number of points", value=20, start=1, sizing_mode = "stretch_width" )
        def update_points(event):
            N = points_input.value
            self.plot(N)
        self.plot_settings_panel.objects = [points_input]

        self.add_param_watch(points_input, update_points, what = "value")
        self.plot(points_input.value)
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                         sizing_mode="stretch_both", min_height = 450, styles={'background': 'lightgreen'})
    

    def get_toolbar(self, skip_button=None, submit_button=None):

        if self.stage == "columns_selection":
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

        return pn.Column(toolbar, self.layout, sizing_mode="stretch_both", min_height =450,)
    
    @param.depends("stage")                
    def panel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(self.unknown_columns)
        else:
            return self.plot_panel()
        


class EuclidPlotClass(CustomPlotClass):
    def __init__(self, data, src, close_button=None, extra_features=None, context=None, **params):
        super().__init__(data=data,
                         src=src,
                         close_button=close_button,
                         extra_features=extra_features,
                         panel_name= "Euclid_Cutout",
                         context=context,
                         **params)
        
        self.context = context
        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)
        self._initialize_settings_dictionary()
        self.euclid_object = None
        self._initialise_euclid_object()
        self.euclid_pane = pn.pane.HoloViews(width=400, height=400) #euclid_pane = Euclid cutout, figure = euclid_pane+overplotted_coordinates
        self.filter = self._get_from_settings_dictionary("filter", "Color")
        self.radius = self._get_from_settings_dictionary("radius", 5.0)
        

    def _change_source_cb(self, attr, old, new):
        initialised = self._initialise_euclid_object()
        self.stored_spectrum_coordinates = {}
        if initialised:
            self._run_euclid()

    def get_layout(self):
        initialised = self._initialise_euclid_object()
        self._initialise_widgets()
        self._manage_subscriptions()
        if initialised:
            self._run_euclid()
        self.message_pane.visible = False

        return  pn.Column(self.message_pane, self.figure, self.plot_settings_panel, 
                          scroll = True, sizing_mode = "stretch_both")
    

    def _initialize_settings_dictionary(self):
        euclid_settings = self.config.settings.setdefault("Euclid_cutout_settings", {})
        
        default_values = { "filter" : "Color",
                           "radius" : 5.0,
                           "stretching" : "Linear",
                           "clipping" : (0,1),
                           "scale" : "minmax",
                           "gamma" : (1,1,1),
                           "source_coordinates" : False,
                           "levels" : 0,
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
        gamma = (self.gamma_red_input.value,  self.gamma_green_input.value, self.gamma_blue_input.value)
        scale = self.scale_input.value
        source_coordinates = self.overplot_source_coords_widget.value
        levels = self.contour_levels_input.value
        self._update_settings_dictionary("scale", scale)
        self._update_settings_dictionary("filter", filter)
        self._update_settings_dictionary("gamma", gamma)
        self._update_settings_dictionary("clipping", (low, high))
        self._update_settings_dictionary("source_coordinates", source_coordinates)
        self._update_settings_dictionary("levels", levels)

    def _get_scaled_image(self):
        if self.filter != "Color":
            low, high = self.contrast_scaler.value
            return self.euclid_object.transform_image_range(self.filter, low, high,
                                                            scale_method = self.scale_input.value)
    
        gamma = (self.gamma_red_input.value,  self.gamma_green_input.value, self.gamma_blue_input.value)
        scale_by_channel = True
        low_r, high_r = self.contrast_scaler_red.value
        low_g, high_g = self.contrast_scaler_green.value
        low_b, high_b = self.contrast_scaler_blue.value
        low = (low_r, low_g, low_b)
        high = (high_r, high_g, high_b)
        if (low == (0,0,0)) and (high == (1,1,1)):
            low, high = self.contrast_scaler.value
            scale_by_channel = False
        return self.euclid_object.transform_image_range(self.filter, low, high, gamma = gamma, 
                                                        scale_method = self.scale_input.value,
                                                        scale_by_channel = scale_by_channel)
         
    def _save_figure(self, directory_path = "data/saved_sources", prefix = None):
        try:
            fname = f"{prefix + '_' if prefix else ''}{self.panel_name}.png"
            filename = os.path.join(directory_path,fname)
            scaled_image =  self._get_scaled_image()
            fig = self.get_euclid_figure(scaled_image, show_scale = True,
                                        show_coordinates = self.overplot_source_coords_widget.value,
                                        show_spectra_coordinates = self.overplot_coords_widget.value)
            fig.savefig(filename, bbox_inches = "tight")
            plt.close(fig)
            return filename
        except FileNotFoundError:
            print(f"Could not find the saving directory: {directory_path}")
        except AttributeError as e:
            print(e)
        except KeyError as e:
            print(f"Missing filter {e} in euclid_object.plot_data")
   
  


    def _save_data_to_fits(self, directory_path = "data/saved_sources"):
        try:
            self.euclid_object.export_cutouts_to_fits(bands_to_export = ["VIS", "NIR_Y", "NIR_J", "NIR_H"], 
                                                      directory_path = directory_path)
        except AttributeError:
            pass
        except FileNotFoundError:
            print(f"Could not find the saving directory: {directory_path}")


    def _initialise_euclid_object(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.get_error_panel("Euclid cutout unavailable", "Missing RA or DEC value")
            return False
        
        try: 
            self.euclid_object.reset_data(self.ra, self.dec)
        
        except AttributeError:
            self.euclid_object = EuclidCutoutsClass(self.ra, self.dec, 
                             euclid_filters= ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                             context = self.context)
        
        self.overplotted_coordinates = []
        return True
            

    def _initialise_widgets(self):

        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius, 
                                                  step = 0.5, start = 1, end = 100, max_width = 200,
                                                  sizing_mode="stretch_both")
        
        self.stretching_input = pn.widgets.Select(name = "Stretching function", 
                                                options=  ['Linear', 'Sqrt', 'Log', 'Asinh', 'PowerLaw'],
                                                value = self._get_from_settings_dictionary("stretching", "Linear"),
                                                sizing_mode = "stretch_both")
        
        self.scale_input = pn.widgets.Select(name = "Scaling Mode", 
                                             options =  ["MinMax", "Expand"],
                                             value = self._get_from_settings_dictionary("scaling", "MinMax"),
                                            sizing_mode = "stretch_both")

        
        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image Clipping", 
                                                     start = 0, end = 1, step = 0.004, 
                                                     value = self._get_from_settings_dictionary("clipping", (0,1)),
                                                     sizing_mode = "stretch_both")
        
        self.filter_input = pn.widgets.Select(name = "Euclid Filter", 
                                                options =  {"VIS" : "VIS", 'Y' : "NIR_Y", 'J' : "NIR_J", 
                                                        'H' : "NIR_H", 'Color' : "Color"},
                                                value  = self.filter,
                                                max_width = 200,
                                                sizing_mode = "stretch_both")
        
        self.overplot_source_coords_widget = pn.widgets.Checkbox(name = "Source Coordinates",
                                                                 value = self._get_from_settings_dictionary("source_coordinates", "False"))
        
        self.overplot_coords_widget = pn.widgets.Checkbox(name = "Spectrum Coordinates")
    
        self.contour_levels_input = pn.widgets.IntInput(name = "Contour Levels", value = self._get_from_settings_dictionary("levels", 0), 
                                                  step =1, start = 0, end = 15, max_width = 200,
                                                  sizing_mode="stretch_both")
        self.contour_levels_scale_input = pn.widgets.Select(name= "Contours drop",
                                                            options= {"Sqrt(2)" : (np.sqrt(2), 1), "2" : (2, 1), "10" : (10,1),
                                                                      "Exponential" : (np.exp(1),1), "Gaussian" : (np.exp(1),2),
                                                                      "de Vaucouleurs" : (np.exp(1), 0.25)},  
                                                            value=1, sizing_mode = "stretch_both",
                                                            max_width = 150)
          

        self.environment_input = pn.widgets.Select(name = "Euclid Science Archive Environment", 
                                            options =  {"Public Data Release" : "PDR", "Internal Data Release" : "IDR", 
                                                        "On The Fly" : "OTF", "REG" : "REG"},
                                            value  = "PDR",
                                            disabled_options=["REG"],
                                            sizing_mode = "stretch_both")
        
        self.user_input = pn.widgets.TextInput(name = 'Euclid Science Archive username', 
                                               placeholder = 'Enter your Euclid Science Archive username here',
                                               sizing_mode = "stretch_both")
        self.password_input = pn.widgets.PasswordInput(name = "Password", 
                                                placeholder = 'Enter your Euclid Science Archive password here',
                                                sizing_mode = "stretch_both")
        
        self.confirm_login_button = pn.widgets.Button(name = "Confirm", sizing_mode = "stretch_both", max_height = 30, 
                                                       max_width = 80, button_type= "primary")
        
        self.login_column = pn.Column(self.user_input, self.password_input, self.confirm_login_button, visible = False)
        
        self.color_settings_column = self._initialise_color_settings()
        self.color_settings_button = pn.widgets.Button(name = "Color image settings", sizing_mode = "stretch_both", max_height = 30, 
                                                       max_width = 80, button_type= "primary")
    
        
        self.add_param_watch_many(
            [self.contrast_scaler,
            self.scale_input,
            self.filter_input,
            self.overplot_source_coords_widget,
            self.contour_levels_input,
            self.contour_levels_scale_input],
            self._general_parameter_callback,
            what="value",
        )

        self.add_param_watch(self.radius_input, self._update_radius, "value")
        self.add_param_watch(self.stretching_input, self._update_stretching, "value")
        self.add_param_watch(self.overplot_coords_widget, self._overplot_coordinates_callback, "value")
    
        self.add_param_watch(self.environment_input, self._change_euclid_environment, "value")

        self.confirm_login_button.on_click(self._confirm_login_credentials_cb)
        self.color_settings_button.on_click(self._open_color_settings_cb)


        self.plot_settings_panel = pn.Column(self.contrast_scaler, self.radius_input, 
                                             pn.Row(self.stretching_input, self.scale_input),
                                             self.filter_input,
                                             pn.Row(self.overplot_source_coords_widget,self.overplot_coords_widget),
                                             pn.Row(self.contour_levels_input, self.contour_levels_scale_input),
                                             self.color_settings_column,
                                             self.color_settings_button,
                                             self.environment_input, 
                                             self.login_column, 
                                             scroll = True, visible = False)
        

    def _initialise_color_settings(self):
        self.contrast_scaler_red = pn.widgets.RangeSlider(name = "Image Red scaling", 
                                                     start = 0, end = 1, step = 0.004, 
                                                     value = (0,1), bar_color = "red",
                                                     max_width = 400,
                                                     sizing_mode = "stretch_both")
        self.contrast_scaler_green = pn.widgets.RangeSlider(name = "Image Green scaling", 
                                                     start = 0, end = 1, step = 0.004, 
                                                     value = (0,1), bar_color = "green",
                                                     max_width = 400,
                                                     sizing_mode = "stretch_both")
        self.contrast_scaler_blue = pn.widgets.RangeSlider(name = "Image Blue scaling", 
                                                     start = 0, end = 1, step = 0.004, 
                                                     value = (0,1), bar_color = "blue",
                                                     max_width = 400,
                                                     sizing_mode = "stretch_both")
        
        self.gamma_red_input = pn.widgets.FloatInput(name = "Γ [R]", 
                                                  value = self._get_from_settings_dictionary("gamma", [1,1,1])[0],
                                                  step = 0.1, start = 0, end = 5, max_width = 100,
                                                  sizing_mode="stretch_both")
        self.gamma_green_input = pn.widgets.FloatInput(name = "Γ [G]", 
                                                  value = self._get_from_settings_dictionary("gamma", [1,1,1])[1],
                                                  step = 0.1, start = 0, end = 5, max_width = 100,
                                                  sizing_mode="stretch_both")
        self.gamma_blue_input = pn.widgets.FloatInput(name = "Γ [B]", 
                                                  value = self._get_from_settings_dictionary("gamma", [1,1,1])[2],
                                                  step = 0.1, start = 0, end = 5, max_width = 100,
                                                  sizing_mode="stretch_both")
        

        self.add_param_watch_many(
            [self.contrast_scaler_red, self.contrast_scaler_green, self.contrast_scaler_blue],
            self._color_specific_callback,
            what = "value_throttled",
            )
        
        self.add_param_watch_many(
            [self.gamma_red_input, self.gamma_green_input, self.gamma_blue_input],
            self._color_specific_callback,
            what = "value",
            )

        
        return pn.Column(pn.Column(self.contrast_scaler_red, self.contrast_scaler_green, self.contrast_scaler_blue),
                         pn.Row(self.gamma_red_input, self.gamma_green_input, self.gamma_blue_input),
                         visible = False)
                              

        
    def _update_radius(self, event):
        if event.new: 
            self.radius = event.new
            self._update_settings_dictionary("radius", self.radius)
            if self.context and self.context.events:
                self.context.events.publish(
                    "astro.euclid.radius.changed",
                    {"radius": self.radius, "panel_id": self.panel_id},
                )
            self._run_euclid()
        else:
            print("Input a valid value for radius")
    
    def _general_parameter_callback(self, event):
        if hasattr(self.euclid_object, "plot_data"):
            self.filter = self.filter_input.value
            self._update_all_settings_dictionary()
            scaled_image = self._get_scaled_image()
            self.get_euclid_figure_hv(scaled_image, show_coordinates= self.overplot_source_coords_widget.value)
            self._update_image()
        
    def _update_stretching(self, event):
        stretch = self.stretching_input.value
        #Use non default scale only if the actual scale parameter is being changed
        if not isinstance(event.new, str):
            stretch_scale = self.stretching_scale_input.value
            self._update_settings_dictionary("stretching_scale", stretch_scale)
        else:
            stretch_scale = None
        
        self._update_settings_dictionary("stretching", stretch)
        self.euclid_object.get_plot_data(stretch = stretch, stretch_scale = stretch_scale)
        scaled_image = self._get_scaled_image()
        self.get_euclid_figure_hv(scaled_image, show_coordinates = self.overplot_source_coords_widget.value)
        self._update_image()

    def _color_specific_callback(self, event):
        if self.filter == "Color":
            if hasattr(self.euclid_object, "plot_data"):
                scaled_image = self._get_scaled_image()
                self.get_euclid_figure_hv(scaled_image, show_coordinates = self.overplot_source_coords_widget.value)
                self._update_image()

    def _update_image(self): 
        try:
             self.figure.object = hv.Overlay(self.euclid_fig + self.overplotted_coordinates)
             self.message_pane.visible = False
        except Exception as e:         #too generic
            print(f"Euclid image unavailable:\n {e}")
 
    def _add_coordinates(self, coordinates, dataset):
        """Storing Coordinates from DESI/SDSS
           coordinates : dict : {"ra" : [...], "dec" : [...]} 
           dataset : string, key of the dictionary storing the coordinates
        """
        if not coordinates or "ra" not in coordinates or "dec" not in coordinates:
            print("Wrong passed coordinates")
            return
        ra, dec  = coordinates["ra"], coordinates["dec"]
        if not hasattr(self, "stored_spectrum_coordinates"):
            self.stored_spectrum_coordinates = {}
        self.stored_spectrum_coordinates[dataset] = {"ra" : ra, "dec" : dec}

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
                    colors = plt.get_cmap("gist_rainbow", max(N,2))
                    marker = "+" if dataset == "DESI" else "*" #TODO improve
                    label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                    for i, (x, y) in enumerate(self.euclid_object.world_2_pix(ra =  self.stored_spectrum_coordinates[dataset]["ra"],
                                                                              dec = self.stored_spectrum_coordinates[dataset]["dec"],
                                                                              filtro = self.filter, zipped = True)):
                        if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                            points = hv.Points([(x,y)], label = label if i == 0 else "")
                            points = points.opts(color = colors(i),
                                                marker = marker, 
                                                size = 20)
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
        if self.environment in  ["IDR", "OTF", "REG"]:
            if os.path.isfile("euclid_credentials.login"):
                print("I found the credential file")
                self.euclid_object.change_environment(environment=self.environment,
                                                      user = None, password = None, 
                                                      credentials_filepath = "euclid_credentials.login")
                
            else:
                user = self.config.settings.get("EuclidAccountUser", None)
                password = self.config.settings.get("EuclidAccountUser", None)
                if (user is None) or (password is None):
                    self.login_column.visible = True
                else:
                    self.euclid_object.change_environment(environment=self.environment,
                                                      user = user, password = password)
        else:
            self.euclid_object.change_environment(environment = self.environment)        


    def _confirm_login_credentials_cb(self, event):
        self.login_column.visible = False
        self.config.settings["EuclidAccountUser"] = self.user_input.value
        self.config.settings["EuclidAccountPassword"] = self.password_input.value
        self.euclid_object.change_environment(environment=self.environment,
                                                user = self.config.settings["EuclidAccountUser"], 
                                                password = self.config.settings["EuclidAccountPassword"])
        svc = getattr(self.context, "services", None) if self.context else None
        if svc is not None:
            svc.set("euclid.client", self.euclid_object.client)
    
    def _open_color_settings_cb(self, event):
        self.color_settings_column.visible  = not self.color_settings_column.visible


    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_object.arcsec_per_pix[self.filter]
        return bar_length_arcsecond

    
    def get_euclid_figure_hv(self, data, 
                             show_coordinates = False, 
                             show_scale = True):
        
        self.image_height, self.image_width,  = data.shape[:2]
        bounds = (0, 0, self.image_width, self.image_height)
        
        if len(data.shape) == 3:
            image = hv.RGB(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         )
        else:
            image = hv.Image(data[::-1,...], bounds=bounds).opts(
                                         active_tools =[], toolbar=None,
                                         padding = 0,
                                         border = 0,
                                         framewise = True,
                                         xaxis=None, 
                                         yaxis=None,
                                         cmap = "grey",
                                         )
        self.image_stream = hv.streams.Tap(source=image, x=np.nan, y=np.nan)

        self.add_param_watch(self.image_stream, self._light_profile_callback, what=["x"])
        
        self.euclid_fig = [image]

        if self.contour_levels_input.value > 0:
            N_contour_levels = self.contour_levels_input.value
            base, exponent = self.contour_levels_scale_input.value
            temp_data = self.euclid_object.data[self.filter]
            temp_img = hv.RGB(temp_data[::-1,...], bounds=bounds) if len(temp_data.shape) == 3 else hv.Image(temp_data[::-1,...], bounds=bounds)
            levels = np.nanmax(temp_data)/(base ** (np.arange(1, N_contour_levels+1)*exponent))
            contours = hv.operation.contours(temp_img, levels = levels).opts(cmap=['red'], colorbar=False, 
                                                                    active_tools=[], show_legend = False)
            self.euclid_fig.append(contours)
        
        if show_scale:
            self.bar_length_pixels = self.image_width * 0.2  #always shows a bar 1/5 of the plot 
            x0, y0 = 0.1*self.image_width, 0.1*self.image_height
            x1 = x0 + self.bar_length_pixels
            scale_bar = hv.Curve(([x0, x1], [y0, y0])).opts(color='red', line_width=3)
            scale_text = hv.Text(x=(x0 + x1)/2, y=y0 + y0/2,
                            text=f'{self.get_plot_scale():.1f}"').opts(
                            text_color='red', text_align='center',
                            text_baseline='bottom', fontsize=14
                            )
            self.euclid_fig.extend([scale_bar, scale_text])
        
        if show_coordinates:
            label = f"{np.round(self.ra,3)}, {np.round(self.dec,3)}"
            x, y = self.euclid_object.world_2_pix(ra = self.ra, dec = self.dec, filtro=self.filter, zipped = False)
            if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                points = hv.Points([(x,y)], label = label)
                points = points.opts(color = "blue",
                                    marker = "+", 
                                    size = 30)
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

        if self.filter not in ["Color"]: #TODO compute light profile for RGB images
            self.get_light_profile_plot(row, col)
    
    def _light_profile_callback_reverse(self, event):
        self._update_image()

    def get_light_profile_plot(self, row, col):

        scaled_image = self._get_scaled_image()
        
        def get_curve(values, idx, xlabel, plot_psf = True, fwhm_psf = 0.16, arcsec_per_pix = 0.1):
            curve =hv.Curve(values, kdims ="x", vdims ="value").opts(toolbar=None, padding = 0.0,
                                                                      border = 1, framewise = True,
                                                                      active_tools =[],
                                                                      xlabel=xlabel, 
                                                                      yaxis=None,  
                                                                      ylim = (min(0,np.nanmin(values)),np.nanmax(values)*1.1), 
                                                                      color = "black") 
            line = hv.VLine(idx).opts(color = "red", line_width=1, line_dash='dotted')
            
            if plot_psf:
                #Centering the psf around the brightest pixel in a 15 px window. if multiple maxima are found
                #it centers to the middle one
                window = 15
                start = max(0, idx - window)
                end = min(len(values), idx + window)
                reduced_values = values[start:end]
                peak = np.nanmax(reduced_values)
                peak_indices = np.where(reduced_values == peak)[0]
                peak_idx = start + peak_indices[len(peak_indices) // 2]
                sigma_psf = fwhm_psf/2.35482004503/arcsec_per_pix
                x = np.arange(len(values))
                psf_profile = peak * np.exp(-0.5*((x-peak_idx)/sigma_psf)**2)
                psf = hv.Curve(psf_profile, kdims = "x", vdims ="value").opts(color = "red", line_width=1, line_dash='solid')
                image = hv.Overlay([curve, line, psf]).opts(responsive=True, toolbar = None)
            else:
                image = hv.Overlay([curve, line]).opts(responsive=True, toolbar = None)

            return image

        arcsec_per_pix = self.euclid_object.arcsec_per_pix[self.filter]
        fwhm_psf = 0.16 if self.filter == "VIS" else 0.3 
        plot_psf = self._get_from_settings_dictionary("stretching", None) == "Linear"

        plot_x = get_curve(scaled_image[row, :], col, "X coordinate", 
                           plot_psf = plot_psf, fwhm_psf = fwhm_psf,
                           arcsec_per_pix = arcsec_per_pix
                           )   
        plot_y = get_curve(scaled_image[:, col], row, "Y coordinate",
                           plot_psf = plot_psf, fwhm_psf = fwhm_psf,
                           arcsec_per_pix = arcsec_per_pix)      
        
        layout = hv.Layout(plot_x + plot_y).cols(1).opts(sizing_mode = "stretch_both")
        row_stream = hv.streams.Tap(source=plot_x, x=np.nan, y=np.nan)
        col_stream = hv.streams.Tap(source=plot_y, x=np.nan, y=np.nan)
        
        self.add_param_watch_many(
            [row_stream, col_stream],
            self._light_profile_callback_reverse,
            what= ["x"]
        )

        self.figure.object = layout
    

    def _run_euclid(self):
        """Wrapper for multithreading"""
        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True
        if self.context and self.context.events:
            self.context.events.publish(
                "astro.cutout.running",
                {"source": "Euclid", "running": True, "panel_id": self.panel_id},
            )
 
        def callback(future_obj=None):
            result = future_obj.result()
            if self.context and self.context.events:
                self.context.events.publish(
                    "astro.cutout.running",
                    {"source": "Euclid", "running": False, "panel_id": self.panel_id},
                )
            
            if self.euclid_object.error_tracker.has_error:
                message =  f"# Euclid cutout unavailable:\n"
                message += f"## {self.euclid_object.error_tracker.error_message}"
                self.message_pane.object = message
                self.message_pane.visible = True #probably already visible
                self.figure.object = self.get_empty_image()
                return
            self.overplot_coords_widget.value = False
            scaled_image = self._get_scaled_image()
            self.get_euclid_figure_hv(scaled_image, show_coordinates = self.overplot_source_coords_widget.value)
            self._update_image()
     
        self.run_multithread(self.euclid_object.get_final_cutout,
                             func_kwargs = {"radius" : self.radius, "stretch" : self.stretching_input.value, 
                              "filtro" : self.filter_input.value,
                              "reference" : "VIS", "verbose" : True, "return_object" : True}, 
                              callback = callback)
    

    def get_euclid_figure(self, data, 
                          show_coordinates = False, 
                          show_scale = True,
                          show_spectra_coordinates = False):
        """Fuction to have the plot in matplotlib in order to be saved.
           Less general than get_euclid_figure_hv as in this case coordinates are overplotted on the same axis
           returns the fig to be saved
        """
        image_height, image_width = data.shape[:2]
        fig, ax = plt.subplots(figsize = (6,6))
        ax.imshow(data, origin = "lower", cmap = "gray")

        if show_scale:
            bar_length_pixels = image_width * 0.2  #always shows a bar 1/5 of the plot 
            x0, y0 = 0.1*image_width, 0.1*image_height
            x1 = x0 + bar_length_pixels 
            ax.plot([x0, x1], [y0, y0], color='red', lw=3)
            ax.text(x=(x0 + x1)/2, y = y0 + y0/2,
                    s = f'{self.get_plot_scale():.1f}"', color = "red",
                    ha = 'center',va = 'bottom', fontsize=14)

        if show_coordinates:
            label = f"{np.round(self.ra,3)}, {np.round(self.dec,3)}"
            x, y = self.euclid_object.world_2_pix(ra = self.ra, dec = self.dec, filtro=self.filter, zipped = False)
            if (0 <= x < image_width) and (0 <= y < image_height):
                ax.scatter(x,y, s = 130, label = label, c = "blue", marker = "+")
               
        if show_spectra_coordinates:
            if hasattr(self, "stored_spectrum_coordinates"):
                for dataset in self.stored_spectrum_coordinates:
                    N = len(self.stored_spectrum_coordinates[dataset]["ra"])
                    colors = plt.get_cmap("gist_rainbow", max(N,2))(np.arange(N))
                    marker = "+" if dataset == "DESI" else "x" #TODO improve
                    label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                    x, y = self.euclid_object.world_2_pix(ra = self.stored_spectrum_coordinates[dataset]["ra"],
                                                          dec = self.stored_spectrum_coordinates[dataset]["dec"],
                                                          filtro = self.filter, zipped = False)
                    x = np.where((0 <= x) & (x < image_width), x, np.nan)
                    y = np.where((0 <= y) & (y < image_height), y, np.nan)
                    ax.scatter(x,y, color = colors, label = label, marker = marker, s =100)
        
        _, labels = ax.get_legend_handles_labels()
        if labels:  
            ax.legend()
        ax.axis("off")
        fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0)
        return fig

    def _manage_subscriptions(self):
        if not self.context or not getattr(self.context, "events", None) or not getattr(self.context, "artifacts", None):
            return

        def _coords_updated(topic, payload):
            if not payload:
                return

            source = payload.get("source")
            artifact_id = payload.get("artifact_id")
            dataset_id = payload.get("dataset_id", "default")
            if not source or not artifact_id:
                return

            try:
                coords = self.context.artifacts.get(artifact_id)
            except Exception as e:
                print(f"Failed to load coords artifact {artifact_id}: {e}")
                return

            self._add_coordinates(coords, source)

        self.subscribe("astro.coords.updated", _coords_updated)

        # Late-join: load latest coords for each source (newest-first is guaranteed by ArtifactStore.find)
        active_dataset_id = "default"
        for src_name in ("DESI", "SDSS", "EuclidSpec"):
            refs = self.context.artifacts.find(
                type="astro.coords",
                dataset_id=active_dataset_id,
                params_subset={"source": src_name},
            )
            if refs:
                try:
                    coords = self.context.artifacts.get(refs[0].artifact_id)
                    self._add_coordinates(coords, src_name)
                except Exception:
                    pass

class SpectrumPlotClass(CustomPlotClass):
    

    def __init__(self, data, src, close_button, extra_features, dataset = "DESI", context = None):
        super().__init__(data, src, close_button, extra_features,
                         panel_name= f"{dataset}_spectrum", context = context)
        

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self.figure = pn.Column(scroll = True, sizing_mode = "stretch_both", margin =(5, 20))
        self.dataset = dataset
        self._is_euclid_spec = self.dataset == "EuclidSpec" 

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)

        self.from_sourceId = False
        self._initialize_settings_dictionary()
        self.plot_settings_panel = pn.Column(visible = False, scroll = True)
        self.mode_options = ["Use TargetId", "Cone Search"]
        self.chosen_mode = self.mode_options[1]
   
    def _to_list(self, x):
        """Convert numpy/array-like to plain python list safely."""
        if x is None:
            return None
        try:
            # numpy arrays, astropy columns, etc.
            return list(x)
        except Exception:
            return x  # scalar

    def _get_attr_or_key(self, obj, name, default=None):
        """Get attribute (Euclid SpectrumContainer) or dict key (DESI record)."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

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
            # target/source coords (the clicked object), if you want them:
            "ra0": getattr(self.spectrum_object, "ra", None),
            "dec0": getattr(self.spectrum_object, "dec", None),
            "spectra": [],
        }

        for sp in spectra:
            # identify spectrum id consistently
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

            # optional fields (common across your include list / SpectrumContainer)
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

            # DESI-specific useful metadata if present
            dr = self._get_attr_or_key(sp, "data_release", None)
            if dr is not None:
                rec.setdefault("meta", {})["data_release"] = dr

            payload["spectra"].append(rec)

        return payload

    def publish_spectrum_artifact(self, dataset_id: str = "default"):
        """
        Store a spectrum artifact + publish a spectrum.updated event.
        """
        if self.context is None:
            return
        spec_payload = self._build_spectrum_artifact_payload()

        artifact_id = self.context.artifacts.put(
            type="astro.spectrum",
            payload=spec_payload,
            dataset_id=dataset_id,
            params={"source": self.dataset},
        )

        self.context.events.publish(
            "astro.spectrum.updated",
            {"source": self.dataset, "artifact_id": artifact_id, "dataset_id": dataset_id},
        )

    def _initialize_settings_dictionary(self):
        self.max_separation = self.config.settings.get("spectrumRadius", 5)


    def get_layout(self):
        self._initialize_settings_panel()
        initialized = self._initialize_spectrum_object()
        if initialized:
            self._run_spectrum()
        return pn.Column(self.message_pane, self.figure, self.plot_settings_panel,  scroll = True)
    
    def _change_source_cb(self, attr, old, new):
        if self.stage == "plot":
            initialized = self._initialize_spectrum_object()
            if initialized:
                self._run_spectrum()

    def _save_figure(self, directory_path = "data/saved_sources", prefix = None):
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
                fig = self.spectrum_object.plot_all_spectra(plot_model = plot_model, plot_lines = plot_lines)
                fig.savefig(filename, bbox_inches = "tight")
                plt.close(fig)
                return filename
            
            except FileNotFoundError:
                print(f"Could not find the saving directory: {directory_path}")

    def _save_data_to_fits(self, directory_path = "data/saved_sources"):
        if self.spectrum_object.spectra is not None:
            try:
                self.spectrum_object.export_spectra_to_fits(fname = self.dataset, directory_path = directory_path)
            except FileNotFoundError:
                print(f"Could not find the saving directory: {directory_path}")

    def _initialize_spectrum_object(self):
        
        if self.from_sourceId:
            try:
                self.sourceId = int(self.get_value_from_df(self.config.settings[f"{self.dataset}_TargetID"]))
                self.ra, self.dec = None, None
            except KeyError:
                self.get_error_panel("Spectrum unavailable", "Missing column with target ID" )
                return False
            except ValueError:
                self.get_error_panel("Spectrum unavailable", "Missing target ID")
                return False         
        else:
            self.sourceId = None
            self.ra, self.dec = self.get_ra_dec()
            if (self.ra is None) or (self.dec is None):
                self.get_error_panel("Spectrum unavailable", "Missing Missing RA or DEC values")
                return False

        try:
            self.spectrum_object.reset_data(ra = self.ra, dec = self.dec,
                                             max_separation = self.max_separation,
                                             sourceId = self.sourceId)

        except AttributeError:
            if self._is_euclid_spec:
                self.spectrum_object = EuclidSpectraClass(
                    self.ra, self.dec,
                    max_separation=self.max_separation,
                    sourceId=self.sourceId,
                    context=self.context,
                )
            else:
                datasets = (["DESI-DR1"] if self.dataset == "DESI"
                            else ["BOSS-DR17", "SDSS-DR17"] if self.dataset == "SDSS"
                            else None)

                self.spectrum_object = DESISpectraClass(
                    self.ra, self.dec,
                    datasets=datasets,
                    max_separation=self.max_separation,
                    sourceId=self.sourceId,
                    context=self.context,
                )
        return True

    def _add_coordinates_to_shared(self, ra, dec):
        """
        ra and dec are lists
        """
        coords_dict = {"ra":list(ra), "dec":list(dec)}
        self.publish_coords(f"{self.dataset}", coords_dict["ra"], coords_dict["dec"])

        return None

    def publish_coords(self, source: str, ra: list[float], dec: list[float], dataset_id: str = "default"):
        if self.context is None:
            return

        coords = {"ra": ra, "dec": dec}

        artifact_id = self.context.artifacts.put(
            type="astro.coords",
            payload=coords,
            dataset_id=dataset_id,
            params={"source": source},
        )

        self.context.events.publish(
            "astro.coords.updated",
            {"source": source, "artifact_id": artifact_id, "dataset_id": dataset_id},
        )

    def _run_spectrum(self, max_separation=None):
        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True

        # running event
        if self.context and getattr(self.context, "events", None):
            self.context.events.publish(
                "astro.spectra.running",
                {"source": self.dataset, "running": True, "panel_id": self.panel_id},
            )

        if max_separation is None:
            max_separation = self.max_separation

        def _set_running(val: bool):
            if self.context and getattr(self.context, "events", None):
                self.context.events.publish(
                    "astro.spectra.running",
                    {"source": self.dataset, "running": val, "panel_id": self.panel_id},
                )

        def callback(future_result=None):
            try:
                _set_running(False)

                if self.spectrum_object.error_tracker.has_error:
                    message = "# Spectrum unavailable:\n"
                    message += f"## {self.spectrum_object.error_tracker.error_message}"
                    self.message_pane.object = message
                    self.message_pane.visible = True
                    self.figure.objects = [self.get_empty_image()]
                    return

                # success
                if self.redshift_column_selector.value != "None":
                    redshift_value = self.get_value_from_df(self.redshift_column_selector.value)
                    if redshift_value is not None:
                        self.redshift_input.value = redshift_value

                self._update_plot()

                # coords artifact+event (you already implemented publish_coords)
                ra_list, dec_list = self.spectrum_object.get_coordinates()
                self._add_coordinates_to_shared(ra_list, dec_list)

                # NEW: spectrum artifact+event
                self.publish_spectrum_artifact(dataset_id="default")

                self.message_pane.visible = False

            except Exception as e:
                self.message_pane.object = f"# Error updating spectrum panel\n## {e}"
                self.message_pane.visible = True
                _set_running(False)

        self.run_multithread(
            self.spectrum_object.get_spectra,
            func_kwargs={"max_separation": max_separation, "return_object": True},
            callback=callback,
        )

    def _get_euclid_radius_arcsec(self, default: float = 0.5) -> float:
        try:
            settings = self.config.settings or {}
            eu = settings.get("Euclid_cutout_settings", {}) or {}
            return float(eu.get("radius", default))
        except Exception:
            return float(default)

    def _initialize_settings_panel(self):
        self.retrieve_mode_button = pn.widgets.RadioButtonGroup(name="How to retrieve spectrum", options=self.mode_options, 
                                            value = self.chosen_mode, sizing_mode = "stretch_both", max_height = 40)
        
        self.max_separation_input = pn.widgets.FloatInput(
            name="Cone Radius [arcsec]",
            value=self._get_euclid_radius_arcsec(0.5),
            step=0.5, start=1, end=100,
            max_width=200, max_height=40,
            sizing_mode="stretch_both",
        )
        
        self.link_to_cutout_checkbox = pn.widgets.Checkbox(name = "Use radius from Euclid cutout",  value = False, align = "center")
        self.max_separation_input.disabled = (self.chosen_mode == self.mode_options[0])
        self.link_to_cutout_checkbox.disabled = (self.chosen_mode == self.mode_options[0])

        self.plot_lines_checkbox = pn.widgets.Checkbox(name = "Plot Emission/Absorption Lines positions",  value = not self._is_euclid_spec, align = "center")
        self.plot_lines_checkbox.disabled = self._is_euclid_spec
        
        self.plot_model_checkbox = pn.widgets.Checkbox(name = "Plot Model",  value = not self._is_euclid_spec, align = "center")
        self.plot_model_checkbox.disabled = self._is_euclid_spec


        self.smoothing_window_input = pn.widgets.IntInput(name = "Smoothing Window", value = 5, start = 1, end = 50, step =1,
                                                         max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.smoothing_function_input = pn.widgets.Select(name = "Smoothing Function", align = "center", 
                                                          options = {"Box" : "Box1DKernel", "Gaussian" : "Gaussian1DKernel"},
                                                          value = "Box1DKernel",
                                                          max_width = 200, max_height = 40, sizing_mode="stretch_both")

        self.redshift_input = pn.widgets.FloatInput(name = "Assign Redshift (Same for all Sources)", start = 0.0, end = 15, 
                                                    max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.query_redshift_button  = pn.widgets.Button(name = "Query Redshift", align = "center", button_type = "primary",
                                                       max_width = 200, max_height = 40, sizing_mode="stretch_both")
        self.redshift_column_selector  = pn.widgets.Select(name = "Redshift Column", align = "center", 
                                                           options = ["None"] + self.get_column_list(allowed_types=["float"]),
                                                           value = "None",
                                                           max_width = 200, max_height = 40, sizing_mode="stretch_both")
        
        self.redshift_input.disabled = not self._is_euclid_spec
        self.query_redshift_button.disabled = not self._is_euclid_spec
        self.redshift_column_selector.disabled = not self._is_euclid_spec

      
        self.add_param_watch(self.retrieve_mode_button, self._retrieve_mode_cb, what="value")
        self.add_param_watch(self.max_separation_input, self._max_separation_input_cb, what="value")
        self.add_param_watch(self.link_to_cutout_checkbox, self._link_to_cutout_cb, what="value")

        self.add_param_watch_many(
            [self.plot_lines_checkbox, self.plot_model_checkbox],
            self._general_parameter_cb,
            "value"
        )

        self.add_param_watch_many(
            [self.smoothing_function_input, self.smoothing_window_input],
            self._update_smoothing_cb,
            "value"
        )

        self.query_redshift_button.on_click(self._query_redshift_cb)

        self.add_param_watch(self.redshift_input, self._redshift_input_cb, what="value")
        self.add_param_watch(self.redshift_column_selector, self._redshift_column_selector_cb, what="value")
            
        self.plot_settings_panel = pn.Column(self.retrieve_mode_button, 
                                             pn.Row(self.max_separation_input, pn.Column(pn.Spacer(height=23), self.link_to_cutout_checkbox), align = "center"),
                                             pn.Row(self.plot_lines_checkbox, self.plot_model_checkbox),
                                             pn.Row(self.smoothing_function_input, self.smoothing_window_input, align = "center"),
                                             pn.Row(self.redshift_input,  pn.Column(pn.Spacer(height=10), self.query_redshift_button),
                                             self.redshift_column_selector, pn.Spacer(width=350), align = "center"),
                                             scroll = True, visible = False)
        
    
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
            self.get_layout()
    
    def _max_separation_input_cb(self, event):
        if event.new is not None:
            self.max_separation = event.new
            self._run_spectrum(self.max_separation)

    def _link_to_cutout_cb(self, event):
        # If linking is enabled, listen to radius change events
        if event.new:
            if not self.from_sourceId and self.context and self.context.events:
                # subscribe once; keep handle so we can disable link without closing panel
                if not hasattr(self, "_euclid_radius_sub"):
                    self._euclid_radius_sub = None

                if self._euclid_radius_sub is None:
                    def _on_radius(topic, payload):
                        if not payload:
                            return
                        radius = payload.get("radius")
                        if radius is None:
                            return
                        # update input and trigger your existing logic
                        try:
                            self.max_separation_input.value = float(radius)
                        except Exception:
                            pass
                        self._update_max_separation(float(radius))

                    self._euclid_radius_sub = self.context.events.subscribe(
                        "astro.euclid.radius.changed",
                        _on_radius
                    )
        else:
            # unlink: unsubscribe from event
            if getattr(self, "_euclid_radius_sub", None) is not None and self.context and self.context.events:
                try:
                    self.context.events.unsubscribe(self._euclid_radius_sub)
                except Exception:
                    pass
                self._euclid_radius_sub = None

    def _update_smoothing_cb(self, event):
        if self.spectrum_object.spectra is not None:
            self.spectrum_object.get_smoothed_spectra(kernel = self.smoothing_function_input.value,
                                                      window= self.smoothing_window_input.value)
            self._update_plot()

    
    def _general_parameter_cb(self, event):
        """This is the Calbback to update the plot without actually querying new data"""
        if self.spectrum_object.spectra is not None:
            self._update_plot()
    
    def _redshift_input_cb(self, event):
        redshift = event.new
        if redshift is not None:
            spectype = "galaxy" if redshift > 0 else "star"
            self.spectrum_object._update_info_spectra("spectype", spectype)
            self.spectrum_object._update_info_spectra("redshift", redshift)
            self.plot_lines_checkbox.disabled = False
            if self.plot_lines_checkbox.value:
                self._update_plot()

    def _query_redshift_cb(self, event):
        if self.spectrum_object.spectra is not None:
            self.query_redshift_button.name = "Query Redshift [Running...]"
            self.spectrum_object.query_specz_table(verbose = True)
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
        plot_model = self.plot_model_checkbox.value
        plot_lines = "class" if self.plot_lines_checkbox.value else False
        kwargs = {"aspect" : 3.8 if self.spectrum_object.available_spectra > 1 else 3.17, "responsive" : True}
        plot = self.spectrum_object.plot_all_spectra_hv(plot_model = plot_model, plot_lines = plot_lines,
                                                                **kwargs)
        self.figure.objects = [plot]

    
    def _update_max_separation(self, new_separation):
         self.max_separation_input.value = new_separation

    
    @param.depends("stage")                        
    def panel(self):
        if self.stage == "columns_selection":
            return self.columns_selection_panel(self.unknown_columns, allowed_types = ["int"],
                                                info_text= "## Select column with TargetID")
        else:
            return self.plot_panel()


class SEDPlotClass(CustomPlotClass):

    
    available_stages = ["filters_selection", "columns_selection",
                       "error_columns_selection", "units_selection", "plot"]

    stage = param.ObjectSelector(default = available_stages[0], objects=available_stages)

    def __init__(self, data, src, close_button, extra_features, context = None):
        super().__init__(data, src, close_button, extra_features, panel_name= "SED",
                         ready_stage = "filters_selection", context = context)


        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)

        ##Any changes here requires an update in load_config (verify_SED)
        self.conversion_dictionary = {"AB magnitudes" : lambda f, e : self.mag_to_flux(f,e),
                                      "milliJy" : lambda f, e : (f * 1000, e * 1000),
                                       "microJy" : lambda f, e : (f,e),
                                       "nanoJy"  : lambda f, e : (f / 1000, e / 1000),
                                       "cgs (erg/s/Hz/cm2)" : lambda f, e : (f * 1e23, e * 1e23)
                                     }

    def _change_source_cb(self, attr, old, new):
        if self.stage == "plot":
            self._update_plot(new)
    
    
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
    
    def __init__(self, data, src, close_button, extra_features, dataset, context = None):
        super().__init__(data, src, close_button, extra_features, context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)


        self.dataset = dataset
        self._initialize_source()
        self.radius = 20

    def _initialize_source(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.message_pane.visible = True
            self.message_pane.object = ["## Missing Ra and Dec"]   

    def _change_source_cb(self, attr, old, new):
        self._initialize_source()
        self._run_radio(radius = self.radius)

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
    
    def __init__(self, data, src, close_button, extra_features, dataset, context = None):
        super().__init__(data, src, close_button, extra_features, context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config


        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)

        self.dataset = dataset
        self._initialize_source()
        self.radius = 25.6

    def _initialize_source(self):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.message_pane.visible = True
            self.message_pane.object = ["## Missing Ra and Dec"]   

    def _change_source_cb(self, attr, old, new):
        self._initialize_source()
        self._run_sdss(radius = self.radius)

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
    
    def __init__(self, data, src, close_button, extra_features, context = None):
        super().__init__(data, src, close_button, extra_features, panel_name = "Aladin Panel", context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)


        self.figure = pn.pane.HTML("", sizing_mode="stretch_both")
    

    def _change_source_cb(self, attr, old, new):
        self.ra, self.dec = self.get_ra_dec()
        if (self.ra is None) or (self.dec is None):
            self.get_error_panel("Aladin panel unavailable", "Missing RA or DEC value")
        self._update_image(None)


    @staticmethod
    def make_iframe_html(survey_id, ra, dec):
        srcdoc = make_srcdoc_aladin_lite(survey_id = survey_id, ra = ra, dec =dec)
        # Escape for inclusion inside the srcdoc='' attribute
        srcdoc_escaped = html.escape(srcdoc, quote=True)
        return "<iframe width='800' height='500' style='border:none' srcdoc='{0}'></iframe>".format(srcdoc_escaped)
    
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
    
        self.survey_selector = pn.widgets.Select(name = "Survey",
                                                 value = "P/DSS2/color",
                                                 groups = {"X-rays" : xray_surveys,
                                                           "Optical/UV" : optical_surveys,
                                                           "IR" : ir_surveys})
        
        self.add_param_watch(self.survey_selector, self._update_image, what="value")


        self.plot_settings_panel = pn.Column(self.survey_selector, 
                                             scroll = True, visible = False, min_height=50,max_height=80)


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
        return  pn.Column(self.message_pane,self.plot_settings_panel, self.figure, 
                          scroll = True, sizing_mode = "stretch_both")



    
class LogBookClass(CustomPlotClass):
    
    def __init__(self, data, src, close_button, extra_features, context = None):
        super().__init__(data, src, close_button, extra_features, panel_name = "Notes Panel", context = context)

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config

        self._src_callback = self._change_source_cb
        self.watch_bokeh(self.src, "data", self._src_callback)

    def _change_source_cb(self, attr, old, new):
        self.logbook_panel.value = ""

    def get_layout(self):
        self._initialise_widgets()
        return  pn.Column(self.logbook_panel, 
                          scroll = True, 
                          sizing_mode = "stretch_both")
    
    def _initialise_widgets(self):

        self.logbook_panel = pn.widgets.TextAreaInput(name = "Logbook",
                                                      auto_grow = False, 
                                                      placeholder='Take your notes here...')

    def _save_panel(self, directory_path= "data/saved_sources", save_fits_files= False, prefix = None):
        paths = {}
        paths["text"] = self.logbook_panel.value.strip()
        return paths
                                                      


######

class EventMonitorClass(CustomPlotClass):
    """
    A lightweight UI panel to inspect EventBus publishes and subscriptions.
    Appears as a selectable panel in the menu.
    """

    def __init__(self, data, src, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            src=src,
            close_button=close_button,
            extra_features=extra_features,
            context=context,
            panel_name="Event Monitor",
            **params,
        )

        # Enable tracing if supported (safe no-op if not)
        if self.context and getattr(self.context, "events", None):
            try:
                self.context.events.enable_trace(True)
            except Exception:
                pass

        # UI
        self.refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary", width=100)
        self.trace_toggle = pn.widgets.Checkbox(name="Trace enabled", value=True)
        self.limit_input = pn.widgets.IntInput(name="Rows", value=200, start=10, end=5000, step=10, width=120)

        self.events_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["time", "topic", "payload"]),
            height=120,
            max_height=250,
            sizing_mode="stretch_both",
        )

        self.subs_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["topic", "subscribers"]),
            height=110,
            max_height=220,
            sizing_mode="stretch_both",
        )

        self.follow_toggle = pn.widgets.Checkbox(name="Follow newest", value=False)

        self._events_df = pd.DataFrame(columns=["time", "topic", "payload"])
        self._last_key = None

        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.refresh_btn.on_click(lambda _e: self.refresh())

        # periodic refresh
        self._period_ms = 1000
        self._cb = pn.state.add_periodic_callback(self.refresh, period=self._period_ms, start=True)

        # initial fill
        self.refresh()

    def refresh(self):
        if not (self.context and getattr(self.context, "events", None)):
            self.status.object = "### Event bus not available on context."
            return

        try:
            self.context.events.enable_trace(bool(self.trace_toggle.value))
        except Exception:
            pass

        n = int(self.limit_input.value or 200)

        try:
            events = self.context.events.recent_events(n)
        except Exception:
            self.status.object = "### Event tracing not implemented on EventBus. Add recent_events()/enable_trace()."
            return

        # Build rows in a stable way
        rows = []
        for (t, topic, payload) in events:
            payload_str = (str(payload)[:240] if payload is not None else "")
            rows.append({
                "t": t,
                "time": time.strftime("%H:%M:%S", time.localtime(t)),
                "topic": topic,
                "payload": payload_str,
            })

        # Find only the new rows since last refresh
        new_rows = []
        if rows:
            if self._last_key is None:
                # first fill: set once (this will scroll to top once, at startup)
                self._events_df = pd.DataFrame([{k: r[k] for k in ["time","topic","payload"]} for r in rows])
                self.events_table.value = self._events_df
                last = rows[-1]
                self._last_key = (last["t"], last["topic"], last["payload"])
            else:
                # scan from the end to find last_key
                last_t, last_topic, last_payload = self._last_key
                idx = -1
                for i in range(len(rows) - 1, -1, -1):
                    r = rows[i]
                    if (r["t"], r["topic"], r["payload"]) == (last_t, last_topic, last_payload):
                        idx = i
                        break

                if idx == -1:
                    # buffer mismatch (rollover changed / tracing restarted) -> reset table once
                    self._events_df = pd.DataFrame([{k: r[k] for k in ["time","topic","payload"]} for r in rows])
                    self.events_table.value = self._events_df
                else:
                    # append only truly new rows
                    new_rows = rows[idx + 1 :]

                    if new_rows:
                        append_df = pd.DataFrame([{k: r[k] for k in ["time","topic","payload"]} for r in new_rows])

                        # keep our buffer and enforce max size n
                        self._events_df = pd.concat([self._events_df, append_df], ignore_index=True)
                        if len(self._events_df) > n:
                            self._events_df = self._events_df.iloc[-n:].reset_index(drop=True)

                        # stream to Tabulator without resetting scroll
                        # rollover keeps Tabulator in sync too
                        self.events_table.stream(append_df, rollover=n, follow=bool(self.follow_toggle.value))

                        last = new_rows[-1]
                        self._last_key = (last["t"], last["topic"], last["payload"])

        try:
            subs = self.context.events.subscribers()
            df2 = pd.DataFrame([{"topic": k, "subscribers": v} for k, v in sorted(subs.items())])
            self.subs_table.value = df2
        except Exception:
            pass

        self.status.object = f"### Showing last {len(self._events_df)} events • {time.strftime('%H:%M:%S')}"

    def get_layout(self):
        return pn.Column(
            pn.Row(self.refresh_btn, self.trace_toggle, self.follow_toggle, self.limit_input),
            self.status,
            pn.pane.Markdown("#### Recent published events"),
            self.events_table,
            pn.pane.Markdown("#### Current subscriptions"),
            self.subs_table,
            sizing_mode="stretch_both",
            scroll=True,
        )

    def dispose(self) -> None:
        # stop periodic callback
        try:
            if self._cb:
                self._cb.stop()
        except Exception:
            pass

        # call base disposal (jobs/events/bokeh watchers)
        super().dispose()


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

    def __init__(self, data, src, close_button=None, extra_features=None, context=None, **params):
        super().__init__(
            data=data,
            src=src,
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

        for plot in (self.source_plot, self.residuals_plot):
            plot.legend.click_policy = "hide"
            plot.add_tools(HoverTool(tooltips=[("Wavelength", "@wavelength"), ("Flux", "@flux")]))

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

        SIDEBAR_WIDTH = 380
        FIELD_WIDTH = 340
        SMALL_FIELD_WIDTH = 120

        LABEL_HEIGHT = 20
        LABEL_MARGIN_TOP = 6
        LABEL_MARGIN_BOTTOM = 4
        BLOCK_MARGIN_BOTTOM = 14


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
                pn.Row(
                    widget,
                    width=FIELD_WIDTH,
                    margin=(0, 0, 0, 0),
                    sizing_mode="fixed",
                ),
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
        )

        self.redshift_box = pn.widgets.FloatInput(
            name="",
            value=0.0,
            start=0.0,
            end=5.0,
            step=0.001,
            width=SMALL_FIELD_WIDTH,
            margin=0,
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
                "<div style='line-height:24px; padding-left:6px;'>Finder Mode</div>",
                width=FIELD_WIDTH - 30,
                height=24,
                margin=0,
            ),
            width=FIELD_WIDTH,
            height=24,
            margin=(0, 0, 16, 0),
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
            margin=0,
        )

        self.fitting_mode_buttons = pn.widgets.RadioButtonGroup(
            name="",
            value="Single fit",
            options=["Single fit", "Multiline fit"],
            button_type="default",
            width=FIELD_WIDTH,
            margin=0,
        )

        self.line_name_input = pn.widgets.TextInput(
            name="",
            placeholder="Line Name",
            width=FIELD_WIDTH,
            margin=0,
        )

        self.line_profile_selector = pn.widgets.Select(
            name="",
            options=["Gaussian", "Lorentzian", "Voigt"],
            value="Gaussian",
            width=FIELD_WIDTH,
            margin=0,
        )

        self.line_name_selector = pn.widgets.Select(
            name="",
            options=self.EMISSION_LINES,
            value=3727.0,
            width=FIELD_WIDTH,
            margin=0,
        )

        self.select_region_buttons = pn.widgets.RadioButtonGroup(
            name="",
            value=None,
            options=["Signal region", "Noise region"],
            button_type="default",
            width=FIELD_WIDTH,
            margin=0,
        )

        self.available_spectra = pn.widgets.Select(
            name="",
            options=[],
            width=FIELD_WIDTH,
            margin=0,
        )

        self.spectra_number_message = pn.widgets.StaticText(
            name="",
            value="",
            width=FIELD_WIDTH,
            margin=0,
        )

        self.fit_button = pn.widgets.Button(
            name="Fit and Lock",
            button_type="success",
            width=FIELD_WIDTH,
            height=38,
            margin=(0, 0, 16, 0),
        )

        self.reset_button = pn.widgets.Button(
            name="Reset fit",
            button_type="warning",
            width=FIELD_WIDTH,
            height=38,
            margin=(0, 0,  16, 0),
        )

        self.undo_lock_button = pn.widgets.Button(
            name="Undo last lock",
            button_type="default",
            width=FIELD_WIDTH,
            height=38,
            margin=(0, 0, 16, 0),
        )

        self.status_message = pn.pane.Alert(
            "",
            alert_type="info",
            visible=False,
            width=FIELD_WIDTH,
            margin=(0, 0, 16, 0),
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
            height=120,
            width=FIELD_WIDTH,
            margin=0,
        )

        analysis_form = pn.WidgetBox(
            pn.Spacer(height=10),
            field_block("Redshift value", self.redshift_box, bottom=16),
            field_block("Redshift slider", self.redshift_slider, bottom=18),
            self.finder_mode_row,
            field_block("Plot settings", self.plot_settings_checkbox, bottom=18),
            field_block("Fitting mode", self.fitting_mode_buttons, bottom=18),
            field_block("Line name", self.line_name_input, bottom=16),
            field_block("Line profile", self.line_profile_selector, bottom=16),
            field_block("Go to line", self.line_name_selector, bottom=16),
            field_block("Region selection", self.select_region_buttons, bottom=18),
            self.fit_button,
            self.reset_button,
            self.undo_lock_button,
            field_label("Derived properties"),
            pn.Spacer(height=4),
            self.derived_properties_table,
            pn.Spacer(height=18),
            field_block("Comments", self.comments_input, bottom=16),
            self.status_message,
            width=SIDEBAR_WIDTH,
            sizing_mode="fixed",
        )

        self.analysis_tab = pn.Column(
            analysis_form,
            width=SIDEBAR_WIDTH,
            min_width=SIDEBAR_WIDTH,
            max_width=SIDEBAR_WIDTH,
            height=700,
            scroll=True,
            sizing_mode="fixed",
        )

        self.settings_tabs = pn.Tabs(
            ("Analysis", self.analysis_tab),
            width=SIDEBAR_WIDTH,
            min_width=SIDEBAR_WIDTH,
            max_width=SIDEBAR_WIDTH,
            sizing_mode="fixed",
            dynamic=False,
        )
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

    # ------------------------------------------------------------------
    # Data conversion
    # ------------------------------------------------------------------
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
    def _spectra_updated(self, topic, payload):
        active_dataset_id = "default"
        spectra_by_source = {}

        for src_name in ("DESI", "SDSS", "EuclidSpec"):
            refs = self.context.artifacts.find(
                type="astro.spectrum",
                dataset_id=active_dataset_id,
                params_subset={"source": src_name},
            )
            if refs:
                spec = self.context.artifacts.get(refs[0].artifact_id)
                spectra_by_source[src_name] = spec

        avail_spectra = list(spectra_by_source.keys())

        def update_models():
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

        self.doc.add_next_tick_callback(update_models)

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

    def _set_status(self, text="", level="info", visible=False):
        self.status_message.object = text
        self.status_message.alert_type = level
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
            self.settings_tabs,
            width=380,
            min_width=380,
            max_width=380,
            height=760,
            sizing_mode="fixed",
            align="start",
            margin=(0, 0, 0, 12),
        )

        main_area = self.pn.Column(
            pn.pane.HTML("<div style='font-weight:600; margin-bottom:4px;'>Available Spectra</div>"),
            self.available_spectra,
            self.spectra_number_message,
            pn.Spacer(height=8),
            self.source_plot,
            pn.Spacer(height=8),
            self.residuals_plot,
            sizing_mode="stretch_width",
            min_width=700,
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