from holoviews.operation.datashader import (
    datashade,
    dynspread,
)

import datashader as ds
import holoviews as hv
from holoviews import streams

import uuid
import numpy as np
import pandas as pd
import panel as pn
import param
import traceback

from astronomicAL.utils.optimise import matches_type


class BasePlotClass(param.Parameterized):

    X_variable = param.Selector(objects=["0"], default="0", doc= "Selection box for the X axis of the plot")
    log_xscale = param.Boolean(default=False, label = "log x",  doc = "Use log for x axis")
    log_yscale = param.Boolean(default=False, label = "log y", doc = "Use log for y axis")
    label_selector = param.ListSelector(default=["All"], objects=["All"], doc= "Labels to plot")
    
    selector_params = ("X_variable",)

    def __init__(self, close_button, context=None):
        super().__init__()

        self.context = context
        if context is not None and getattr(context, "config", None) is not None:
            self.config = context.config

        self.df = pd.DataFrame()
        self.update_df()

        self._disposed = False
        self._event_subs = []
        self._periodic_cbs = []
        self._bokeh_on_change = []
        self._param_watchers = []

        self._refresh_pending = False
        self._refresh_pending_payload = None
        self._refresh_reasons = set()
        self._refresh_inflight_signature = None
        self._last_completed_signature = None
        self._initial_refresh_requested = False

        self.panel_id = str(uuid.uuid4())

        self.close_button = close_button

        self.figure = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            min_height=0,
        )

        self.settings_button = pn.widgets.Button(
            name="Open Settings",
            button_type="primary",
            max_height=40,
            max_width=120,
        )
        self.settings_button.on_click(self._toggle_settings_panel)
    
    def watch_bokeh(self, model, attr: str, callback):
        """Register and track Bokeh model.on_change callbacks for unified disposal."""
        if model is None:
            return
        try:
            model.on_change(attr, callback)
            self._bokeh_on_change.append((model, attr, callback))
        except Exception:
            pass

    def unwatch_all_bokeh(self):
        """Remove all tracked Bokeh callbacks (idempotent)."""
        for model, attr, callback in list(getattr(self, "_bokeh_on_change", [])):
            try:
                model.remove_on_change(attr, callback)
            except Exception as e:
                # Ignore double-remove noise
                if "list.remove(x): x not in list" in str(e):
                    pass
            # continue regardless
        self._bokeh_on_change = []

    def watch_param(self, parameterized, param_name: str, callback):
        """
        Register and track param.watch callbacks for unified disposal.
        """
        if parameterized is None:
            return None

        try:
            watcher = parameterized.param.watch(callback, param_name)
            self._param_watchers.append((parameterized, watcher))
            return watcher
        except Exception:
            return None


    def unwatch_all_params(self):
        """
        Remove all tracked param watchers (idempotent).
        """
        for parameterized, watcher in list(getattr(self, "_param_watchers", [])):
            try:
                parameterized.param.unwatch(watcher)
            except Exception:
                pass

        self._param_watchers = []

    def subscribe_event(self, topic: str, callback):
        """
        Subscribe to EventBus and track the subscription so dispose() can unsubscribe.
        """
        if not self.context or not getattr(self.context, "events", None):
            return None

        try:
            sub = self.context.events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label=self.__class__.__name__,
                owner_kind="plot",
            )
        except TypeError:
            sub = self.context.events.subscribe(topic, callback)

        self._event_subs.append(sub)
        return sub
    
    def _dispose_impl(self):
        """Subclass-specific cleanup hook (override if needed)."""
        return

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        # 1) subclass cleanup first
        try:
            self._dispose_impl()
        except Exception:
            pass

        # 2) unwatch bokeh callbacks (including src.on_change if registered via watch_bokeh)
        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        try:
            self.unwatch_all_params()
        except Exception:
            pass

        # 3) Unsubscribe EventBus
        if self.context and getattr(self.context, "events", None):
            for sub in list(getattr(self, "_event_subs", [])):
                try:
                    self.context.events.unsubscribe(sub)
                except Exception:
                    pass
        self._event_subs = []

        # 4) Stop periodic callbacks
        for cb in list(getattr(self, "_periodic_cbs", [])):
            try:
                cb.stop()
            except Exception:
                pass
        self._periodic_cbs = []


    def update_df(self):
        """
        Refresh the plot dataframe from the active dataset if available.
        Falls back to config.main_df for compatibility with older code.
        """
        if self.context is not None and getattr(self.context, "datasets", None) is not None:
            try:
                dataset_id = self._get_active_dataset_id()
                self.df = self.context.datasets.get_df(dataset_id).copy()
                if getattr(self, "config", None) is not None:
                    self.config.main_df = self.df
                return
            except Exception:
                pass

        try:
            self.df = self.config.main_df.copy()
        except Exception:
            self.df = pd.DataFrame()

    def set_focus_selection(self, row_id, *, origin=None, selection_set_id=None):
        if not self.context or not getattr(self.context, "selection", None):
            return

        self.context.selection.set_focus(
            dataset_id=self._get_active_dataset_id(),
            row_id=str(row_id),
            origin=origin or self.__class__.__name__,
            panel_id=self.panel_id,
            selection_set_id=selection_set_id,
        )


    def set_selection_set(
        self,
        row_ids,
        *,
        origin=None,
        mode="replace",
        metadata=None,
        create_artifact=True,
        update_focus_policy="preserve_or_first",
    ):
        if not self.context or not getattr(self.context, "selection", None):
            return None

        return self.context.selection.set_selection_set(
            dataset_id=self._get_active_dataset_id(),
            row_ids=[str(r) for r in row_ids],
            origin=origin or self.__class__.__name__,
            panel_id=self.panel_id,
            mode=mode,
            metadata=metadata or {},
            create_artifact=create_artifact,
            update_focus_policy=update_focus_policy,
        )

    def _get_active_selection_set_state(self):
        if self.context is not None and getattr(self.context, "selection", None) is not None:
            try:
                return self.context.selection.get_active_set()
            except Exception:
                pass
        return None


    def _handle_selection_set_change_event(self, topic, payload):
        """
        Rerender on selection-set changes.

        We do not filter aggressively here because a changed/cleared selection set
        should remove stale overlays even if the new set belongs to another dataset.
        """
        self._request_refresh(reason=str(topic or "selection.set.changed"), payload=payload)


    def _handle_selection_set_cleared_event(self, topic, payload):
        self._request_refresh(reason=str(topic or "selection.set.cleared"), payload=payload)

    def _get_active_dataset_id(self):
        if self.context is not None and getattr(self.context, "datasets", None) is not None:
            try:
                return self.context.datasets.active_id()
            except Exception:
                pass
        return "default"
    
    def _get_label_strings_map(self):
        return dict(getattr(self.config, "settings", {}).get("labels_to_strings", {}) or {})

    def _get_strings_to_labels_map(self):
        return dict(getattr(self.config, "settings", {}).get("strings_to_labels", {}) or {})
    
    def _get_label_colours_map(self):
        return dict(getattr(self.config, "settings", {}).get("label_colours", {}) or {})
    
    def _get_label_display_name(self, raw_label):
        labels_to_strings = self._get_label_strings_map()

        if raw_label in labels_to_strings:
            return labels_to_strings[raw_label]

        raw_str = str(raw_label)
        if raw_str in labels_to_strings:
            return labels_to_strings[raw_str]

        return raw_str

    def _get_label_colour(self, raw_label, default="blue"):
        label_colours = self._get_label_colours_map()

        if raw_label in label_colours:
            return label_colours[raw_label]

        raw_str = str(raw_label)
        if raw_str in label_colours:
            return label_colours[raw_str]

        return default

    def _apply_label_settings_payload(self, payload):
        if not isinstance(payload, dict):
            return

        settings = getattr(self.config, "settings", {})

        if "label_col" in payload:
            settings["label_col"] = payload["label_col"]

        if "labels" in payload:
            settings["labels"] = list(payload["labels"])

        if "labels_to_strings" in payload:
            settings["labels_to_strings"] = dict(payload["labels_to_strings"])

        if "strings_to_labels" in payload:
            settings["strings_to_labels"] = dict(payload["strings_to_labels"])

        if "label_colours" in payload:
            settings["label_colours"] = dict(payload["label_colours"])

    def _handle_label_settings_event(self, topic, payload):
        if not self._event_targets_active_dataset(payload):
            return

        self._apply_label_settings_payload(payload)
        self._refresh_label_selector_objects()
        self._rerender_after_dataset_change()

    def _event_targets_active_dataset(self, payload):
        """
        Only react when the event is about the currently active dataset.
        """
        if not isinstance(payload, dict):
            return True

        dataset_id = (
            payload.get("dataset_id")
            or payload.get("active_dataset_id")
            or payload.get("id")
        )
        if dataset_id is None:
            return True

        return dataset_id == self._get_active_dataset_id()

    def _request_refresh(self, reason="unknown", payload=None, verbose=False):
        """
        Coalesce repeated triggers onto the next tick.
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

    def _get_focus_state(self):
        if self.context is not None and getattr(self.context, "selection", None) is not None:
            try:
                return self.context.selection.get_focus()
            except Exception:
                pass
        return None

    def _get_focus_signature(self):
        focus = self._get_focus_state()
        if focus is None:
            return (None, None)

        return (
            getattr(focus, "dataset_id", None),
            str(getattr(focus, "row_id", None)) if getattr(focus, "row_id", None) is not None else None,
        )

    def _build_refresh_signature(self, reason=None, payload=None):
        """
        Default refresh signature for plot dashboards.
        Includes focus state so selection changes coalesce correctly.
        """
        selector_values = tuple(
            getattr(self, name, None)
            for name in getattr(self, "selector_params", ())
        )

        label_values = tuple(self.label_selector) if hasattr(self, "label_selector") and self.label_selector else tuple()

        return (
            self._get_active_dataset_id(),
            selector_values,
            label_values,
            self._get_focus_signature(),
        )
    
    def _begin_refresh(self, reason=None, payload=None, verbose=False):
        try:
            refresh_signature = self._build_refresh_signature(reason=reason, payload=payload)
        except Exception:
            traceback.print_exc()
            refresh_signature = (self._get_active_dataset_id(),)

        if refresh_signature == self._refresh_inflight_signature:
            if verbose:
                print(
                    f"[{self.__class__.__name__}] refresh skipped; identical request already in flight: "
                    f"{refresh_signature} (reason={reason})"
                )
            return None

        self._refresh_inflight_signature = refresh_signature
        return refresh_signature

    def _finish_refresh(self, refresh_signature=None):
        if refresh_signature is None:
            refresh_signature = self._refresh_inflight_signature

        if refresh_signature is not None:
            self._last_completed_signature = refresh_signature

        if self._refresh_inflight_signature == refresh_signature:
            self._refresh_inflight_signature = None

    def _request_initial_refresh_once(self, reason="initial.panel"):
        if self._initial_refresh_requested:
            return

        self._initial_refresh_requested = True
        self._request_refresh(reason=reason)

    def _perform_refresh(self, reason=None, payload=None, refresh_signature=None):
        """
        Default synchronous refresh hook for plot dashboards.
        """
        try:
            self._refresh_selectors_from_current_df()
            self._rerender_after_dataset_change()
        finally:
            self._finish_refresh(refresh_signature)

    def _get_selected_row_from_current_df(self):
        """
        Resolve the currently focused row from the active dataset.

        Selection is now driven by context.selection rather than src.data.
        """
        focus = self._get_focus_state()
        if focus is None:
            return None

        focus_dataset_id = getattr(focus, "dataset_id", None)
        focus_row_id = getattr(focus, "row_id", None)

        if focus_row_id is None:
            return None

        if focus_dataset_id != self._get_active_dataset_id():
            return None

        if self.df is None or len(self.df) == 0:
            return None

        id_col = self.config.settings.get("id_col")

        try:
            if id_col == "Use Index":
                mask = self.df.index.astype(str) == str(focus_row_id)
                selected = self.df.loc[mask]
            elif id_col and id_col in self.df.columns:
                selected = self.df[self.df[id_col].astype(str) == str(focus_row_id)]
            else:
                return None

            if len(selected) > 0:
                return selected.head(1).reset_index(drop=True)
        except Exception:
            pass

        return None

    def _handle_focus_change_event(self, topic, payload):
        """
        Rerender on any focus change.

        We do not filter by active dataset here, because a focus change to a
        different dataset should clear the selected overlay in the current plot.
        """
        self._request_refresh(reason=str(topic or "selection.focus.changed"), payload=payload)

    def _handle_focus_cleared_event(self, topic, payload):
        self._request_refresh(reason=str(topic or "selection.focus.cleared"), payload=payload)

    def _get_available_columns_for_selectors(self):
        """
        Default selector columns for BasePlotClass-style plots.
        Override in subclasses if needed.
        """
        return self.get_column_list(
            excluded_columns=["id_col", "ra_dec", "label_col"],
            excluded_types=["object"],
            allowed_types=None,
        )

    def _refresh_label_selector_objects(self):
        """
        Refresh label selector options and keep only still-valid selections.

        Behaviour:
        - If no labels exist, default to ["All"].
        - If labels have just become available and the current state is empty or
        only ["All"], switch to all individual labels so coloured overlays are
        shown immediately.
        - Otherwise preserve the user's valid current choices.
        """
        label_names = list(self._get_strings_to_labels_map().keys())
        all_options = ["All"] + label_names
        self.param.label_selector.objects = all_options

        current = list(self.label_selector) if self.label_selector else []

        # Keep only still-valid values
        current = [x for x in current if x in all_options]

        if not label_names:
            self.label_selector = ["All"]
            return

        # If we were effectively in the pre-label/default state, auto-enable all labels
        if not current or current == ["All"]:
            self.label_selector = list(label_names)
            return

        self.label_selector = current

    def _coerce_selector_values_after_df_change(self):
        """
        Base version for plots with only X_variable.
        Subclasses with more selectors should override.
        """
        objects = list(getattr(self, "available_columns", []))
        if not objects:
            self.param.X_variable.objects = ["0"]
            self.X_variable = "0"
            return

        self.param.X_variable.objects = objects

        if self.X_variable not in objects:
            self.X_variable = objects[0]

    def _refresh_selectors_from_current_df(self):
        """
        Refresh dataframe-backed selector options after dataset mutation/switch.
        """
        self.update_df()
        self.available_columns = self._get_available_columns_for_selectors()

        if not self.available_columns:
            self.available_columns = ["0"]

        self._initialise_selector_options()
        self._refresh_label_selector_objects()
        self._coerce_selector_values_after_df_change()

    def _rerender_after_dataset_change(self):
        """
        Default re-render hook.
        """
        try:
            if hasattr(self, "_update_plot"):
                self._update_plot()
                return
        except Exception:
            traceback.print_exc()

        try:
            if hasattr(self, "plot"):
                self.figure.object = self.plot()
        except Exception:
            traceback.print_exc()

    def _handle_dataset_change_event(self, topic, payload):
        """
        Generic response to dataset.updated / dataset.active.changed.
        """
        if not self._event_targets_active_dataset(payload):
            return

        self._request_refresh(reason=str(topic or "dataset.change"), payload=payload)

    def _register_dataset_event_handlers(self):
        """
        Call this at the end of subclass __init__ once selectors/settings exist.
        """
        self.subscribe_event("dataset.updated", self._handle_dataset_change_event)
        self.subscribe_event("dataset.active.changed", self._handle_dataset_change_event)
        self.subscribe_event("labels.settings.updated", self._handle_label_settings_event)

        self.subscribe_event("selection.focus.changed", self._handle_focus_change_event)
        self.subscribe_event("selection.focus.cleared", self._handle_focus_cleared_event)

        self.subscribe_event("selection.set.changed", self._handle_selection_set_change_event)
        self.subscribe_event("selection.set.cleared", self._handle_selection_set_cleared_event)

    def _toggle_settings_panel(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Close Settings" if self.settings_panel.visible else "Open Settings"
    

    def _toolbar_field_label(self, text):
        return pn.pane.HTML(
            f"""
            <div style="
                font-size: 11px;
                font-weight: 600;
                color: #2f2f2f;
                margin: 0 0 4px 0;
                line-height: 1.1;
                white-space: nowrap;
            ">
                {text}
            </div>
            """,
            margin=(0, 0, 0, 0),
            sizing_mode="stretch_width",
        )


    def _toolbar_select_widget(self, parameter_name, width=150):
        widget = pn.widgets.Select.from_param(
            getattr(self.param, parameter_name),
            name="",
            width=width,
            min_width=width,
            max_width=width,
        )
        widget.margin = (0, 0, 0, 0)
        return widget


    def _toolbar_select_block(self, label, parameter_name, width=150):
        return pn.Column(
            self._toolbar_field_label(label),
            self._toolbar_select_widget(parameter_name, width=width),
            width=width,
            min_width=width,
            max_width=width,
            height=50,
            min_height=50,
            max_height=50,
            margin=(0, 0, 0, 0),
            sizing_mode="fixed",
        )

    def get_column_list(self, excluded_columns = ["id_col", "ra_dec", "label_col"],
                              excluded_types = ["object"], allowed_types = None):
        """
        Returns the list of columns used for panel.widgets.Selector according to their type
        -----
        excluded_columns : list of columns which are removed regardless of their type
        allowed_types : list ["float", "numeric", "int"], if not None, only columns with this type are kept
        excluded_types = list ["float", "object"] list, columns with this types are removed

        It is a bit tricky, in the sense that if you pass an empty/None allowed_types, all types are kept
        """

        cols = list(self.df.columns)
        
        for excluded_col in excluded_columns:
            col_name = self.config.settings.get(excluded_col, excluded_col)
            if col_name in cols:
               cols.remove(col_name)
        
        if allowed_types:
            cols = [col for col in cols if matches_type(self.df[col].dtype, allowed_types)]
        if excluded_types:
            cols = [col for col in cols if not matches_type(self.df[col].dtype, excluded_types)]
        return cols

    def get_id(self):
        id_col = self.config.settings["id_col"]
        if id_col == "Use Index":
            ids = self.df.index.values
        else:
            ids = self.df[id_col].values
        return ids
    
    def _initialise_settings_dictionary(self, key_name, default_values):
        """
        key_name = 'Histogram_plot_settings', 'Scatter_plot_settings' or 'Density_plot_settings'
        default_values = Dictionary with key-values to be used as default ones
        """
        settings_dict = self.config.settings.setdefault(key_name, {})
        for key, value in default_values.items():
            if key not in settings_dict:
                settings_dict[key] = value

    def _initialise_selector_options(self):
        """
        Initialise the available options for selector params.
        """
        objects = list(getattr(self, "available_columns", []))
        if not objects:
            objects = ["0"]

        for name in self.selector_params:
            self.param[name].objects = objects

    def _compose_overlay(self, layers, **opts):
        """
        Build an Overlay safely.

        Important for datashader/dynspread outputs, which are often DynamicMaps.
        Calling .collate() avoids the repeated HoloViews warning about nesting
        DynamicMaps inside an Overlay.

        Also drops None-valued opts so we do not pass invalid HoloViews params
        such as legend_opts=None.
        """
        clean_layers = [layer for layer in layers if layer is not None]

        if not clean_layers:
            return self._get_empty_plot("No data to display")

        overlay = hv.Overlay(clean_layers)

        try:
            overlay = overlay.collate()
        except Exception:
            pass

        clean_opts = {k: v for k, v in opts.items() if v is not None}

        if clean_opts:
            overlay = overlay.opts(**clean_opts)

        return overlay

    def _hv_plot_tools(self, *, include_hover=False, include_select=False):
        tools = ["pan", "wheel_zoom", "box_zoom", "reset", "save"]

        if include_hover:
            tools.append("hover")

        if include_select:
            tools = ["tap", "box_select"] + tools

        # preserve order while removing duplicates
        seen = set()
        out = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _hv_base_opts(
        self,
        *,
        xlabel=None,
        ylabel=None,
        logx=None,
        logy=None,
        tools=None,
        active_tools=None,
        xlim=None,
        ylim=None,
        show_grid=True,
        shared_axes=False,
        framewise=True,
        axiswise=True,
    ):
        return dict(
            xlabel=xlabel,
            ylabel=ylabel,
            logx=self.log_xscale if logx is None else logx,
            logy=self.log_yscale if logy is None else logy,
            tools=tools if tools is not None else self._hv_plot_tools(),
            active_tools=active_tools if active_tools is not None else ["wheel_zoom"],
            xlim=xlim,
            ylim=ylim,
            responsive=True,
            min_height=0,
            show_grid=show_grid,
            shared_axes=shared_axes,
            framewise=framewise,
            axiswise=axiswise,
            toolbar="right",
        )

    def _hv_overlay_opts(self, *, xlabel=None, ylabel=None):
        return dict(
            xlabel=xlabel,
            ylabel=ylabel,
            responsive=True,
            min_height=0,
            shared_axes=False,
            framewise=True,
            axiswise=True,
            active_tools=[],
            toolbar="right",
            show_grid=True,
            legend_position="right",
        )

    def _hv_selected_overlay_opts(self):
        return dict(
            marker="circle",
            size=14,
            fill_alpha=0.0,
            line_color="black",
            line_width=3,
            active_tools=[],
            logx=self.log_xscale,
            logy=self.log_yscale,
        )

    def _get_empty_plot(self, message="No data to display"):
        return hv.Text(0.5, 0.5, message).opts(
            xlim=(0, 1),
            ylim=(0, 1),
            responsive=True,
            min_height=0,
            toolbar=None,
            xaxis=None,
            yaxis=None,
            show_frame=False,
        )
    
    def _finite_xy(self, x, y=None):
        x = np.asarray(x)

        if y is None:
            mask = np.isfinite(x)
            return x[mask], mask

        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask], mask
    
    def _prepare_for_log_axis(self, values):
        values = np.asarray(values)
        if self.log_xscale or self.log_yscale:
            values = values[np.isfinite(values)]
        return values

    def _initialise_param_objects(self, **extra_params):
        """
        Initialise the param objects which govern the behaviour of the plot.
        """
        self.update_df()

        if not getattr(self, "available_columns", None):
            self.available_columns = self._get_available_columns_for_selectors()

        if not self.available_columns:
            self.available_columns = ["0"]

        self._initialise_selector_options()
        self.param.label_selector.objects = ["All"] + list(
            getattr(self.config, "settings", {}).get("strings_to_labels", {}).keys()
        )

        x_default = self._get_from_settings_dictionary("X_variable", self.available_columns[0])
        if x_default not in self.available_columns:
            x_default = self.available_columns[0]

        labels_default = self._get_from_settings_dictionary("labels", ["All"])
        if not labels_default:
            labels_default = ["All"]

        self.param.update(
            X_variable=x_default,
            label_selector=labels_default,
            log_xscale=self._get_from_settings_dictionary("log_x", False),
            log_yscale=self._get_from_settings_dictionary("log_y", False),
            **extra_params,
        )
    
    def get_toolbar(self):
        top_row_h = 42
        selector_row_h = 58
        toolbar_h = top_row_h + selector_row_h + 6

        top_row = pn.Row(
            pn.Spacer(width=16),
            self.close_button,
            self.settings_button,
            sizing_mode="stretch_width",
            height=top_row_h,
            min_height=top_row_h,
            max_height=top_row_h,
            margin=(0, 0, 6, 0),
            align="center",
        )

        selector_row = pn.Row(
            pn.Spacer(width=16),
            self._toolbar_select_block("X variable", "X_variable", width=220),
            pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            height=selector_row_h,
            min_height=selector_row_h,
            max_height=selector_row_h,
            margin=(0, 0, 0, 0),
            align="start",
        )

        return pn.Column(
            top_row,
            selector_row,
            sizing_mode="stretch_width",
            height=toolbar_h,
            min_height=toolbar_h,
            max_height=toolbar_h,
            margin=(0, 0, 0, 0),
        )

class ScatterPlotDashboard(BasePlotClass):
    """A Dashboard used for rendering dynamic scatter plots of the data.
    Parameters
    ----------

    Attributes
    ----------
    X_variable : param.Selector
        A Dropdown list of columns the user can use for the x-axis of the plot.
    Y_variable : DataFrame
        A Dropdown list of columns the user can use for the x-axis of the plot.
    df : DataFrame
        The shared dataframe which holds all the data.

    """   

    Y_variable = param.Selector(objects=["1"], default="1", doc="Selection box for the Y axis of the plot")
    plot_mode = param.Selector(default="tap", objects=["tap", "rasterized"], doc= "Plot Mode")
    selector_params = ("X_variable", "Y_variable")


    def __init__(self, close_button, context=None):
        super().__init__( close_button, context=context)

        self.context = context

        self.available_columns = self.get_column_list(
            excluded_columns=["id_col", "label_col", "ra_dec"]
        )

        if not self.available_columns:
            self.available_columns = ["0", "1"]

        defaults = self.config.settings.get("default_vars", self.available_columns[:2])
        if len(defaults) < 2:
            defaults = (
                list(self.available_columns[:2])
                if len(self.available_columns) > 1
                else [self.available_columns[0], self.available_columns[0]]
            )

        self._initialise_settings_dictionary(
            key_name="Scatter_plot_settings",
            default_values={
                "X_variable": defaults[0],
                "Y_variable": defaults[1],
                "log_x": False,
                "log_y": False,
                "labels": ["All"],
                "mode": "tap",
            },
        )

        y_default = self._get_from_settings_dictionary(
            "Y_variable",
            self.available_columns[1] if len(self.available_columns) > 1 else self.available_columns[0],
        )
        if y_default not in self.available_columns:
            y_default = self.available_columns[1] if len(self.available_columns) > 1 else self.available_columns[0]

        self._initialise_param_objects(
            Y_variable=y_default,
            plot_mode=self._get_from_settings_dictionary("mode", "tap"),
        )

        self.settings_panel = pn.Column(
            pn.Param(
                self,
                parameters=[
                    "log_xscale", "log_yscale", "label_selector", "plot_mode"
                ],
                widgets={
                    "label_selector": {"type": pn.widgets.MultiChoice, "width": 200, "height": 80},
                    "plot_mode": {"type": pn.widgets.RadioBoxGroup},
                },
                show_name=False,
                sizing_mode="stretch_width",
            ),
            visible=False,
            margin=(10, 0, 0, 0),
        )

        self._register_dataset_event_handlers()

    def _coerce_selector_values_after_df_change(self):
        """
        Scatter override: keep X and Y if still valid, otherwise repair them.
        """
        objects = list(getattr(self, "available_columns", []))
        if not objects:
            self.param.X_variable.objects = ["0"]
            self.param.Y_variable.objects = ["1"]
            self.X_variable = "0"
            self.Y_variable = "1"
            return

        self.param.X_variable.objects = objects
        self.param.Y_variable.objects = objects

        # Keep current choices where possible
        x_value = self.X_variable if self.X_variable in objects else objects[0]

        if self.Y_variable in objects:
            y_value = self.Y_variable
        else:
            y_value = objects[1] if len(objects) > 1 else objects[0]

        # Avoid X and Y collapsing to the same value when there are >=2 columns
        if len(objects) > 1 and x_value == y_value:
            for col in objects:
                if col != x_value:
                    y_value = col
                    break

        self.param.update(
            X_variable=x_value,
            Y_variable=y_value,
        )

    def plot_selected(self, x_var, y_var):
        selected = self._get_selected_row_from_current_df()
        if selected is None:
            return None

        if x_var not in selected.columns or y_var not in selected.columns:
            return None

        if selected.shape[0] > 0:
            return hv.Scatter(selected, x_var, y_var).opts(
                marker="circle",
                size=14,
                fill_alpha=0.0,
                line_color="black",
                line_width=3,
                active_tools=[],
                logx=self.log_xscale,
                logy=self.log_yscale,
            )

        return None

    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Scatter_plot_settings"].get(key, default)
        return value
    
    def _get_active_selection_set_row_ids(self):
        state = self._get_active_selection_set_state()
        if state is None:
            return []

        dataset_id = getattr(state, "dataset_id", None)
        if dataset_id != self._get_active_dataset_id():
            return []

        row_ids = list(getattr(state, "row_ids", []) or [])
        return [str(r) for r in row_ids]


    def _get_active_selection_box_bounds(self):
        state = self._get_active_selection_set_state()
        if state is None:
            return None

        dataset_id = getattr(state, "dataset_id", None)
        if dataset_id != self._get_active_dataset_id():
            return None

        metadata = getattr(state, "metadata", {}) or {}
        geometry = metadata.get("geometry", {}) or {}

        if geometry.get("kind") != "box":
            return None

        if geometry.get("x_variable") != self.X_variable:
            return None

        if geometry.get("y_variable") != self.Y_variable:
            return None

        bounds = geometry.get("bounds")
        if not bounds or len(bounds) != 4:
            return None

        try:
            x0, x1, y0, y1 = bounds
            return (float(x0), float(x1), float(y0), float(y1))
        except Exception:
            return None


    def _get_selection_set_overlay(self):
        row_ids = self._get_active_selection_set_row_ids()
        if not row_ids:
            return None

        id_col = self.config.settings.get("id_col", "Use Index")

        try:
            if id_col == "Use Index":
                mask = self.df.index.astype(str).isin(row_ids)
                selected = self.df.loc[mask]
            else:
                if id_col not in self.df.columns:
                    return None
                mask = self.df[id_col].astype(str).isin(row_ids)
                selected = self.df.loc[mask]

            if selected.empty:
                return None

            if self.X_variable not in selected.columns or self.Y_variable not in selected.columns:
                return None

            return hv.Points(
                selected,
                kdims=[self.X_variable, self.Y_variable],
            ).opts(
                marker="circle",
                size=8,
                fill_alpha=0.0,
                line_color="orange",
                line_width=2,
                active_tools=[],
                logx=self.log_xscale,
                logy=self.log_yscale,
            )
        except Exception:
            return None


    def _get_selection_box_overlay(self):
        bounds = self._get_active_selection_box_bounds()
        if bounds is None:
            return None

        x0, x1, y0, y1 = bounds
        left, right = sorted((x0, x1))
        bottom, top = sorted((y0, y1))

        return hv.Rectangles([(left, bottom, right, top)]).opts(
            fill_alpha=0.08,
            fill_color="orange",
            line_color="orange",
            line_width=2,
            active_tools=[],
            show_legend=False,
        )
    
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "Y_variable" : self.Y_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "mode" : self.plot_mode,
        }
        self.config.settings["Scatter_plot_settings"].update(new_values)

    @staticmethod
    def get_axis_limits(x_var, Nsigma = 3):
        x = x_var[np.isfinite(x_var)]
        if len(x) == 0:
            return 0, 1 
        max_x = np.max(x)
        min_x = np.min(x)
        x_sd = np.std(x)
        x_mu = np.mean(x)
        max_x = np.min([x_mu + Nsigma * x_sd, max_x])
        min_x = np.max([x_mu - Nsigma * x_sd, min_x])
        return min_x, max_x
    

    @param.depends(
        "X_variable", "Y_variable", "label_selector", "log_xscale",
        "log_yscale", "plot_mode",
        watch=True
    )
    def _update_plot(self):
        # Important: clear old stream watchers before rebuilding the interactive plot
        try:
            self.unwatch_all_params()
        except Exception:
            pass

        self._update_all_settings_dictionary()
        self.main_plot = self.plot()

        selection_box_plot = self._get_selection_box_overlay()
        selection_set_plot = self._get_selection_set_overlay()
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)

        overlay_opts = dict(
            active_tools=[],
            xlabel=self.X_variable,
            ylabel=self.Y_variable,
            responsive=True,
            min_height=0,
            shared_axes=False,
            framewise=True,
            axiswise=True,
            toolbar="right",
            show_grid=True,
            legend_position="right",
            legend_opts={"click_policy": "mute" if self.plot_mode == "tap" else "hide"},
        )

        self.figure.object = self._compose_overlay(
            [self.main_plot, selection_box_plot, selection_set_plot, selected_src_plot],
            **overlay_opts,
        )

    def _handle_scatter_selection_indices(self, indices, sourceid, bounds=None):
        if not indices:
            return

        row_ids = []
        for i in indices:
            try:
                row_ids.append(str(sourceid[i]))
            except Exception:
                continue

        if not row_ids:
            return

        # Single-point selection -> focus only
        if len(row_ids) == 1:
            self.set_focus_selection(
                row_ids[0],
                origin="scatter.tap",
            )
            return

        metadata = {
            "x_variable": self.X_variable,
            "y_variable": self.Y_variable,
            "plot_mode": self.plot_mode,
            "panel_type": "scatter",
        }

        if bounds is not None and len(bounds) == 4:
            left, bottom, right, top = bounds
            metadata["geometry"] = {
                "kind": "box",
                "x_variable": self.X_variable,
                "y_variable": self.Y_variable,
                "bounds": [left, right, bottom, top],
            }

        self.set_selection_set(
            row_ids,
            origin="scatter.box_select",
            mode="replace",
            metadata=metadata,
            create_artifact=True,
            update_focus_policy="preserve_or_first",
        )

    def get_scatter_hv(self, x, y, sourceid=None, plot_mode="tap", color="blue", label=""):
        x = np.asarray(x)
        y = np.asarray(y)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) == 0 or len(y) == 0:
            return self._get_empty_plot("No finite scatter data")

        if sourceid is not None:
            sourceid = np.asarray(sourceid)[finite].astype(str)

        min_x, max_x = self.get_axis_limits(x)
        min_y, max_y = self.get_axis_limits(y)

        if plot_mode == "tap" and sourceid is not None:
            points = hv.Points(
                (x, y, sourceid),
                kdims=["x", "y"],
                vdims=["id"],
                label=label,
            ).opts(
                size=4,
                color=color,
                alpha=0.60,
                line_alpha=0.0,
                xlim=(min_x, max_x),
                ylim=(min_y, max_y),
                tools=["tap", "box_select", "hover", "wheel_zoom", "pan", "reset", "save"],
                active_tools=["wheel_zoom"],
                selection_alpha=1.0,
                selection_color="orange",
                selection_line_color="black",
                selection_line_width=2,
                nonselection_alpha=0.18,
                nonselection_color=color,
                nonselection_line_alpha=0.0,
                muted_alpha=0.03,
                muted_fill_alpha=0.03,
                muted_line_alpha=0.0,
                muted_color=color,
                logx=self.log_xscale,
                logy=self.log_yscale,
                xlabel=self.X_variable,
                ylabel=self.Y_variable,
                responsive=True,
                min_height=0,
                show_grid=True,
                shared_axes=False,
                framewise=True,
                axiswise=True,
                toolbar="right",
            )

            sel_stream = streams.Selection1D(source=points)
            bounds_stream = streams.BoundsXY(source=points)

            _last_bounds = {"value": None}

            def bounds_callback(event):
                _last_bounds["value"] = event.new

            def selection_callback(event):
                indices = list(event.new or [])
                if not indices:
                    return

                bounds = _last_bounds["value"]
                if bounds is None:
                    try:
                        bounds = bounds_stream.bounds
                    except Exception:
                        bounds = None

                self._handle_scatter_selection_indices(indices, sourceid, bounds=bounds)

            self.watch_param(bounds_stream, "bounds", bounds_callback)
            self.watch_param(sel_stream, "index", selection_callback)
            return points

        points = hv.Points((x, y), kdims=["x", "y"], label=label).opts(
            logx=self.log_xscale,
            logy=self.log_yscale,
            xlabel=self.X_variable,
            ylabel=self.Y_variable,
        )

        shaded = datashade(
            points,
            aggregator=ds.count(),
            cmap=[color],
        ).opts(
            xlim=(min_x, max_x),
            ylim=(min_y, max_y),
            responsive=True,
            min_height=0,
            active_tools=[],
            toolbar="right",
            xlabel=self.X_variable,
            ylabel=self.Y_variable,
            show_grid=True,
        )

        return dynspread(
            shaded,
            threshold=0.75,
            how="saturate",
        )

    def plot(self, x_var=None, y_var=None):
        if self.df is None or len(self.df) == 0:
            return self._get_empty_plot("Dataset is empty")

        if x_var is None:
            x_var = self.df[self.X_variable].to_numpy()
        if y_var is None:
            y_var = self.df[self.Y_variable].to_numpy()

        strings_to_plot = list(self.label_selector) if self.label_selector else ["All"]
        sourceid = self.get_id().astype(str) if self.plot_mode == "tap" else None

        label_col = self.config.settings.get("label_col", "No Labels")
        has_label_column = (
            label_col not in [None, "No Labels"]
            and label_col in self.df.columns
            and len(self._get_strings_to_labels_map()) > 0
        )

        selected_display_labels = [s for s in strings_to_plot if s != "All"]

        overlays = []

        if has_label_column and selected_display_labels:
            labels = self.df[label_col]
            raw_labels_to_plot = [
                self._get_strings_to_labels_map().get(display_name)
                for display_name in selected_display_labels
                if display_name in self._get_strings_to_labels_map()
            ]

            for raw_label in raw_labels_to_plot:
                if raw_label is None:
                    continue

                select = labels == raw_label
                label_sourceid = sourceid[select] if sourceid is not None else None

                overlays.append(
                    self.get_scatter_hv(
                        x_var[select],
                        y_var[select],
                        sourceid=label_sourceid,
                        plot_mode=self.plot_mode,
                        color=self._get_label_colour(raw_label, default="blue"),
                        label=self._get_label_display_name(raw_label),
                    )
                )
        else:
            overlays.append(
                self.get_scatter_hv(
                    x_var,
                    y_var,
                    sourceid=sourceid,
                    plot_mode=self.plot_mode,
                    color="blue",
                    label="All",
                )
            )

        if not overlays:
            return self._get_empty_plot("No scatter data for selected filters")

        overlay_opts = dict(
            active_tools=[],
            xlabel=self.X_variable,
            ylabel=self.Y_variable,
            responsive=True,
            min_height=420,
            shared_axes=False,
            framewise=True,
            axiswise=True,
            toolbar="right",
            show_grid=True,
            legend_position="right",
            legend_opts={"click_policy": "mute" if self.plot_mode == "tap" else "hide"},
        )

        return self._compose_overlay(overlays, **overlay_opts)

    def get_toolbar(self):
        top_row_h = 42
        selector_row_h = 58
        toolbar_h = top_row_h + selector_row_h + 6

        top_row = pn.Row(
            pn.Spacer(width=16),
            self.close_button,
            self.settings_button,
            sizing_mode="stretch_width",
            height=top_row_h,
            min_height=top_row_h,
            max_height=top_row_h,
            margin=(0, 0, 6, 0),
            align="center",
        )

        selector_row = pn.Row(
            pn.Spacer(width=16),
            self._toolbar_select_block("X variable", "X_variable", width=220),
            self._toolbar_select_block("Y variable", "Y_variable", width=220),
            pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            height=selector_row_h,
            min_height=selector_row_h,
            max_height=selector_row_h,
            margin=(0, 0, 0, 0),
            align="start",
        )

        return pn.Column(
            top_row,
            selector_row,
            sizing_mode="stretch_width",
            height=toolbar_h,
            min_height=toolbar_h,
            max_height=toolbar_h,
            margin=(0, 0, 0, 0),
        )

    def panel(self):
        self._request_initial_refresh_once(reason="initial.panel")

        toolbar = self.get_toolbar()

        body = pn.Column(
            self.figure,
            self.settings_panel,
            sizing_mode="stretch_both",
            scroll=False,
            min_height=0,
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            toolbar,
            body,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )


class HistoDashboard(BasePlotClass):
    
    density = param.Boolean(default=False, doc = None )
    cumulative = param.Boolean(default=False, doc = None)
    Nbins = param.Integer(default=10, bounds=(2, 200), doc = "Number of bins")
    range_min = param.Number(default= None, bounds=(-np.inf, np.inf), allow_None= True,  doc= "Range min")
    range_max = param.Number(default= None, bounds=(-np.inf, np.inf), allow_None= True, doc= "Range max")

    def __init__(self, close_button, context=None):
        super().__init__(close_button, context=context)

        self.context = context

        self.available_columns = self.get_column_list(excluded_columns=["id_col", "ra_dec"])

        if not self.available_columns:
            self.available_columns = ["0"]

        self._initialise_settings_dictionary(
            key_name="Histogram_plot_settings",
            default_values={
                "X_variable": self.config.settings.get("default_vars", self.available_columns[:1])[0],
                "log_x": False,
                "log_y": False,
                "density": False,
                "cumulative": False,
                "Nbins": 10,
                "range": (-np.inf, np.inf),
                "labels": ["All"],
            },
        )

        self._initialise_param_objects(
            cumulative=self._get_from_settings_dictionary("cumulative", False),
            density=self._get_from_settings_dictionary("density", False),
            Nbins=self._get_from_settings_dictionary("Nbins", 10),
            range_min=self._get_from_settings_dictionary("range", (-np.inf, np.inf))[0],
            range_max=self._get_from_settings_dictionary("range", (-np.inf, np.inf))[1],
        )

        self.settings_panel = pn.Column(
            pn.Param(
                self,
                parameters=[
                    "log_xscale", "log_yscale", "density", "cumulative",
                    "Nbins", "range_min", "range_max", "label_selector"
                ],
                widgets={
                    "range_min": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "range_max": {"type": pn.widgets.FloatInput, "placeholder": "None"},
                    "Nbins": {"throttled": True},
                    "label_selector": {"type": pn.widgets.MultiChoice, "width": 200, "height": 80},
                },
                show_name=False,
                sizing_mode="stretch_width",
            ),
            visible=False,
            margin=(10, 0, 0, 0),
        )

        self._register_dataset_event_handlers()

    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Histogram_plot_settings"].get(key, default)
        return value
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "cumulative" : self.cumulative,
                       "density" : self.density,
                       "Nbins" : self.Nbins,
                       "range" : (self.range_min, self.range_max),
        }
        self.config.settings["Histogram_plot_settings"].update(new_values)


    @param.depends(
        "X_variable", "log_xscale", "log_yscale", "density", "cumulative",
        "Nbins", "range_min", "range_max", "label_selector",
        watch=True
    )
    def _update_plot(self):
        self._update_all_settings_dictionary()
        self.main_plot = self.plot_hv()
        selected_src_plot = self.plot_selected(self.X_variable)

        if selected_src_plot is not None:
            self.figure.object = hv.Overlay([self.main_plot, selected_src_plot]).opts(
                **self._hv_overlay_opts(xlabel=self.X_variable, ylabel="% of Sources" if self.density else "# Sources")
            )
        else:
            self.figure.object = self.main_plot

    @staticmethod
    def get_histogram_hv(
        x_var,
        Nbins=10,
        log_x=False,
        log_y=False,
        density=False,
        cumulative=False,
        range=(-np.inf, np.inf),
        label="",
        xlabel="x",
        ylabel="frequency",
        **kwargs,
    ):
        xmin, xmax = range
        xmin = -np.inf if xmin is None else xmin
        xmax = np.inf if xmax is None else xmax

        x = np.asarray(x_var)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            return hv.Histogram(([], []), kdims=[xlabel], vdims=[ylabel], label=label), 0, 1

        xmin = max(np.min(x), xmin)
        xmax = min(np.max(x), xmax)

        if xmin > xmax:
            xmin = xmax

        weights = np.ones_like(x) / len(x) if density else None

        if log_x:
            if xmax <= 0:
                return hv.Histogram(([], []), kdims=[xlabel], vdims=[ylabel], label=label), 0, 1
            if xmin <= 0:
                positive = x[x > 0]
                if len(positive) == 0:
                    return hv.Histogram(([], []), kdims=[xlabel], vdims=[ylabel], label=label), 0, 1
                xmin = np.min(positive)
            bins = np.geomspace(xmin, xmax, Nbins) if xmin != xmax else Nbins
        else:
            bins = np.linspace(xmin, xmax, Nbins) if xmin != xmax else Nbins

        stats, edges = np.histogram(x, bins=bins, weights=weights)

        if cumulative:
            stats = np.cumsum(stats)

        ylim = (0.2, None) if log_y else (0, None)
        if density and log_y and np.any(stats > 0):
            ylim = (np.min(stats[stats > 0]) / 5, None)

        histogram = hv.Histogram(
            (edges, stats),
            kdims=[xlabel],
            vdims=[ylabel],
            label=label,
        ).opts(
            logy=log_y,
            logx=log_x,
            ylim=ylim,
            xlim=(xmin, xmax),
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
            active_tools=["wheel_zoom"],
            responsive=True,
            min_height=0,
            show_grid=True,
            toolbar="right",
            **kwargs,
        )

        return histogram, xmin, xmax

    def plot_hv(self, x_var=None):
        if self.df is None or len(self.df) == 0:
            return self._get_empty_plot("Dataset is empty")

        if x_var is None:
            x_var_name = self.X_variable
            x_var = self.df[self.X_variable].to_numpy()
        else:
            x_var_name = self.X_variable

        strings_to_plot = self.label_selector
        if bool(strings_to_plot) and ("All" not in strings_to_plot or len(strings_to_plot) > 1):
            labels = self.df[self.config.settings["label_col"]]
            labels_to_plot = [
                self.config.settings.get("strings_to_labels", {}).get(i)
                for i in strings_to_plot
                if i != "All" and i in self.config.settings.get("strings_to_labels", {})
            ]
        else:
            labels_to_plot = []

        overlays = []
        xmin, xmax = np.inf, -np.inf

        xlabel = x_var_name
        ylabel = "% of Sources" if self.density else "# Sources"

        if "All" in strings_to_plot:
            h, xmin_temp, xmax_temp = self.get_histogram_hv(
                x_var,
                Nbins=self.Nbins,
                log_x=self.log_xscale,
                log_y=self.log_yscale,
                cumulative=self.cumulative,
                density=self.density,
                range=(self.range_min, self.range_max),
                label="All",
                xlabel=xlabel,
                ylabel=ylabel,
                fill_color="blue",
                line_color="blue",
                fill_alpha=0.5,
            )
            overlays.append(h)
            xmin = min(xmin, xmin_temp)
            xmax = max(xmax, xmax_temp)

        for i, label_to_plot in enumerate(labels_to_plot):
            h, xmin_temp, xmax_temp = self.get_histogram_hv(
                x_var[labels == label_to_plot],
                Nbins=self.Nbins,
                log_x=self.log_xscale,
                log_y=self.log_yscale,
                cumulative=self.cumulative,
                density=self.density,
                range=(self.range_min, self.range_max),
                label=self.config.settings["labels_to_strings"][str(label_to_plot)],
                xlabel=xlabel,
                ylabel=ylabel,
                fill_color=self.config.settings["label_colours"][label_to_plot] if i < 2 else "none",
                line_color=self.config.settings["label_colours"][label_to_plot],
                line_width=1.5,
                fill_alpha=0.5,
            )
            overlays.append(h)
            xmin = min(xmin, xmin_temp)
            xmax = max(xmax, xmax_temp)

        if not overlays:
            return self._get_empty_plot("No histogram data for selected filters")

        return hv.Overlay(overlays).opts(
            **self._hv_overlay_opts(xlabel=xlabel, ylabel=ylabel),
            xlim=(xmin, xmax),
        )
    
    
    def plot_selected(self, x_var):
        selected = self._get_selected_row_from_current_df()
        if selected is None:
            return None

        if x_var not in selected.columns:
            return None

        if selected.shape[0] > 0:
            return hv.VLine(selected[x_var].iloc[0]).opts(
                color="black",
                line_dash="dashed",
                line_width=1,
                active_tools=[],
            )

        return None

    def get_toolbar(self):
        top_row_h = 42
        selector_row_h = 58
        toolbar_h = top_row_h + selector_row_h + 6

        top_row = pn.Row(
            pn.Spacer(width=16),
            self.close_button,
            self.settings_button,
            sizing_mode="stretch_width",
            height=top_row_h,
            min_height=top_row_h,
            max_height=top_row_h,
            margin=(0, 0, 6, 0),
            align="center",
        )

        selector_row = pn.Row(
            pn.Spacer(width=16),
            self._toolbar_select_block("X variable", "X_variable", width=220),
            pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            height=selector_row_h,
            min_height=selector_row_h,
            max_height=selector_row_h,
            margin=(0, 0, 0, 0),
            align="start",
        )

        return pn.Column(
            top_row,
            selector_row,
            sizing_mode="stretch_width",
            height=toolbar_h,
            min_height=toolbar_h,
            max_height=toolbar_h,
            margin=(0, 0, 0, 0),
        )

    def panel(self):
        self._request_initial_refresh_once(reason="initial.panel")

        toolbar = self.get_toolbar()

        body = pn.Column(
            self.figure,
            self.settings_panel,
            sizing_mode="stretch_both",
            scroll=False,
            min_height=0,
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            toolbar,
            body,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )

class DensityPlotDashboard(BasePlotClass):

    Y_variable = param.Selector(objects=["1"], default="1", doc="Selection box for the Y axis of the plot")
    selector_params = ("X_variable", "Y_variable")
    Nbins = param.Integer(default=10, bounds=(2, 200), doc = "Number of bins per axis")
    x_range_min = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "X variable range min")
    x_range_max = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "X variable range max")
    y_range_min = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "Y variable range min")
    y_range_max = param.Number(default = None, bounds=(-np.inf, np.inf), allow_None= True, doc = "Y variable range max")
    log_zscale = param.Boolean(default=False, label = "log density",  doc = "Use log for density color")

    clim = param.Integer(default=10, bounds=(2, 1000), doc = "Number of bins per axis")

    def __init__(self, close_button, context=None):
        super().__init__(close_button, context=context)

        self.context = context

        self.available_columns = self.get_column_list(
            excluded_columns=["id_col", "label_col", "ra_dec"]
        )

        if not self.available_columns:
            self.available_columns = ["0", "1"]

        defaults = self.config.settings.get("default_vars", self.available_columns[:2])
        if len(defaults) < 2:
            defaults = list(self.available_columns[:2]) if len(self.available_columns) > 1 else [self.available_columns[0], self.available_columns[0]]

        self._initialise_settings_dictionary(
            key_name="Density_plot_settings",
            default_values={
                "X_variable": defaults[0],
                "Y_variable": defaults[1],
                "log_x": False,
                "log_y": False,
                "labels": ["All"],
                "x_range": (-np.inf, np.inf),
                "y_range": (-np.inf, np.inf),
                "Nbins": 20,
                "log_z": False,
            },
        )

        y_default = self._get_from_settings_dictionary("Y_variable", self.available_columns[1] if len(self.available_columns) > 1 else self.available_columns[0])
        if y_default not in self.available_columns:
            y_default = self.available_columns[1] if len(self.available_columns) > 1 else self.available_columns[0]

        self._initialise_param_objects(
            Y_variable=y_default,
            Nbins=self._get_from_settings_dictionary("Nbins", 10),
            log_zscale=self._get_from_settings_dictionary("log_z", False),
            x_range_min=self._get_from_settings_dictionary("x_range", (-np.inf, np.inf))[0],
            x_range_max=self._get_from_settings_dictionary("x_range", (-np.inf, np.inf))[1],
            y_range_min=self._get_from_settings_dictionary("y_range", (-np.inf, np.inf))[0],
            y_range_max=self._get_from_settings_dictionary("y_range", (-np.inf, np.inf))[1],
        )

        self.param_widgets = {
            "log_xscale": pn.widgets.Checkbox.from_param(self.param.log_xscale),
            "log_yscale": pn.widgets.Checkbox.from_param(self.param.log_yscale),
            "log_zscale": pn.widgets.Checkbox.from_param(self.param.log_zscale),
            "Nbins": pn.widgets.IntSlider.from_param(self.param.Nbins, throttled=True),
            "x_range_min": pn.widgets.FloatInput.from_param(self.param.x_range_min),
            "x_range_max": pn.widgets.FloatInput.from_param(self.param.x_range_max),
            "y_range_min": pn.widgets.FloatInput.from_param(self.param.y_range_min),
            "y_range_max": pn.widgets.FloatInput.from_param(self.param.y_range_max),
            "label_selector": pn.widgets.MultiChoice.from_param(self.param.label_selector, width=200, height=80),
        }

        self.settings_panel = pn.Column(
            pn.Row(
                self.param_widgets["log_xscale"],
                self.param_widgets["log_yscale"],
                self.param_widgets["log_zscale"],
            ),
            self.param_widgets["Nbins"],
            pn.Column(
                pn.Row(self.param_widgets["x_range_min"], self.param_widgets["x_range_max"]),
                pn.Row(self.param_widgets["y_range_min"], self.param_widgets["y_range_max"]),
            ),
            self.param_widgets["label_selector"],
            sizing_mode="stretch_width",
            visible=False,
            margin=(10, 0, 0, 0),
        )

        self._register_dataset_event_handlers()


    def _get_from_settings_dictionary(self, key, default):
        value = self.config.settings["Density_plot_settings"].get(key, default)
        return value
    
    
    def _update_all_settings_dictionary(self):
        new_values =  {"X_variable" : self.X_variable,
                       "Y_variable" : self.Y_variable,
                       "log_x" : self.log_xscale,
                       "log_y" : self.log_yscale,
                       "labels" : self.label_selector,
                       "Nbins" : self.Nbins,
                       "x_range" : (self.x_range_min, self.x_range_max),
                       "y_range" : (self.y_range_min, self.y_range_max)
        }
        self.config.settings["Density_plot_settings"].update(new_values)

    @param.depends(
        "X_variable", "Y_variable", "label_selector", "log_xscale",
        "log_yscale", "log_zscale",
        "Nbins", "x_range_min", "x_range_max", "y_range_min",
        "y_range_max",
        watch=True
    )
    def _update_plot(self):
        self._update_all_settings_dictionary()
        self.main_plot = self.plot()
        selected_src_plot = self.plot_selected(self.X_variable, self.Y_variable)

        if selected_src_plot is not None:
            self.figure.object = self._compose_overlay(
                [self.main_plot, selected_src_plot],
                xlabel=self.X_variable,
                ylabel=self.Y_variable,
                responsive=True,
                min_height=0,
                shared_axes=False,
                framewise=True,
                axiswise=True,
                toolbar="right",
                show_grid=True,
            )
        else:
            self.figure.object = self.main_plot
    

    def get_density_hv(
        self,
        x_var,
        y_var,
        log_x=False,
        log_y=False,
        x_range=(-np.inf, np.inf),
        y_range=(-np.inf, np.inf),
        log_z=False,
        Nbins=25,
        cmap="viridis",
    ):
        xmin, xmax = x_range
        xmin = -np.inf if xmin is None else xmin
        xmax = np.inf if xmax is None else xmax

        ymin, ymax = y_range
        ymin = -np.inf if ymin is None else ymin
        ymax = np.inf if ymax is None else ymax

        select = np.logical_and.reduce([
            np.isfinite(x_var),
            np.isfinite(y_var),
            x_var >= xmin,
            x_var < xmax,
            y_var >= ymin,
            y_var < ymax,
        ])

        x, y = x_var[select], y_var[select]

        if len(x) == 0 or len(y) == 0:
            return self._get_empty_plot("No finite density data")

        if log_x:
            positive = x > 0
            x = x[positive]
            y = y[positive]
            if len(x) == 0:
                return self._get_empty_plot("No positive X values for log scale")
            x = np.log10(x)

        if log_y:
            positive = y > 0
            x = x[positive]
            y = y[positive]
            if len(y) == 0:
                return self._get_empty_plot("No positive Y values for log scale")
            y = np.log10(y)

        return hv.HexTiles((x, y), kdims=["x", "y"]).opts(
            gridsize=Nbins,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
            active_tools=["wheel_zoom"],
            xlabel=self.X_variable,
            ylabel=self.Y_variable,
            xlim=(np.min(x), np.max(x)),
            ylim=(np.min(y), np.max(y)),
            logz=log_z,
            colorbar=True,
            cmap=cmap,
            responsive=True,
            min_height=0,
            show_grid=True,
            toolbar="right",
        )
   
    
    def plot(self, x_var=None, y_var=None):
        if self.df is None or len(self.df) == 0:
            return self._get_empty_plot("Dataset is empty")

        if x_var is None:
            x_var = self.df[self.X_variable].to_numpy()
        if y_var is None:
            y_var = self.df[self.Y_variable].to_numpy()

        strings_to_plot = list(self.label_selector) if self.label_selector else ["All"]

        label_col = self.config.settings.get("label_col", "No Labels")
        has_label_column = (
            label_col not in [None, "No Labels"]
            and label_col in self.df.columns
            and len(self._get_strings_to_labels_map()) > 0
        )

        # For density plots, treat label selection as a filter, not as separate overlays.
        if has_label_column and ("All" not in strings_to_plot):
            labels = self.df[label_col]
            raw_labels_to_keep = [
                self._get_strings_to_labels_map().get(display_name)
                for display_name in strings_to_plot
                if display_name in self._get_strings_to_labels_map()
            ]
            raw_labels_to_keep = [lab for lab in raw_labels_to_keep if lab is not None]

            if raw_labels_to_keep:
                select = labels.isin(raw_labels_to_keep)
                x_var = x_var[select]
                y_var = y_var[select]

        return self.get_density_hv(
            x_var,
            y_var,
            Nbins=self.Nbins,
            log_x=self.log_xscale,
            log_y=self.log_yscale,
            x_range=(self.x_range_min, self.x_range_max),
            y_range=(self.y_range_min, self.y_range_max),
            log_z=self.log_zscale,
            cmap="Viridis",
        )
    
    def plot_selected(self, x_var, y_var):
        selected = self._get_selected_row_from_current_df()
        if selected is None:
            return None

        if x_var not in selected.columns or y_var not in selected.columns:
            return None

        if selected.shape[0] > 0:
            return hv.Scatter(selected, x_var, y_var).opts(
                marker="circle",
                size=12,
                fill_alpha=0.0,
                line_color="black",
                line_width=3,
                active_tools=[],
                logx=self.log_xscale,
                logy=self.log_yscale,
            )

        return None
    
    def get_toolbar(self):
        top_row_h = 42
        selector_row_h = 58
        toolbar_h = top_row_h + selector_row_h + 6

        top_row = pn.Row(
            pn.Spacer(width=16),
            self.close_button,
            self.settings_button,
            sizing_mode="stretch_width",
            height=top_row_h,
            min_height=top_row_h,
            max_height=top_row_h,
            margin=(0, 0, 6, 0),
            align="center",
        )

        selector_row = pn.Row(
            pn.Spacer(width=16),
            self._toolbar_select_block("X variable", "X_variable", width=220),
            self._toolbar_select_block("Y variable", "Y_variable", width=220),
            pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            height=selector_row_h,
            min_height=selector_row_h,
            max_height=selector_row_h,
            margin=(0, 0, 0, 0),
            align="start",
        )

        return pn.Column(
            top_row,
            selector_row,
            sizing_mode="stretch_width",
            height=toolbar_h,
            min_height=toolbar_h,
            max_height=toolbar_h,
            margin=(0, 0, 0, 0),
        )

    def panel(self):
        self._request_initial_refresh_once(reason="initial.panel")

        toolbar = self.get_toolbar()

        body = pn.Column(
            self.figure,
            self.settings_panel,
            sizing_mode="stretch_both",
            scroll=False,
            min_height=0,
            margin=(0, 0, 0, 0),
        )

        return pn.Column(
            toolbar,
            body,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )