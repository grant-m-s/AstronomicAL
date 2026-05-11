from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
from astronomicAL.dashboard.labelling import LabellingDashboard
from astronomicAL.dashboard.menu import MenuDashboard
from astronomicAL.dashboard.selected_source import SelectedSourceDashboard
from astronomicAL.dashboard.settings_dashboard import SettingsDashboard
from astronomicAL.extensions import extension_plots, custom_plots
from bokeh.models import ColumnDataSource

import panel as pn
import param

NATIVE_CONTENTS = {
    "Menu",
    "Selected Source Info",

    # Built-in workflow/dashboard modes.
    # These are not custom plots and should not be validated against
    # get_customplot_dict() or get_plot_dict().
    "Active Learning",
    "Labelling",
    "Labelling Test Set",
    "Settings",
}


class Dashboard(param.Parameterized):
    """Top-level Dashboard which holds an instance of any other Dashboard.

    This will initialize and render any of the other instances of the
    dashboards and handles the interactions of switching from one dashboard
    view to another.

    Parameters
    ----------
    contents : param.String, default = "Menu"
        The identifier for which of the dashboard views should be initialised
        and rendered.

    Attributes
    ----------
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        Panel layout.
    df : DataFrame
        The shared dataframe which holds all the data.

    """

    contents = param.String()

    def __init__(self, src, contents="Menu", context=None):
        super(Dashboard, self).__init__()

        self.NATIVE_CONTENTS = NATIVE_CONTENTS

        self._bokeh_on_change = []
        self.src = ColumnDataSource(data={"0": [], "1": []})
        self.watch_bokeh(self.src, "data", self._update_extension_plots_cb)

        self.row = pn.Row(pn.pane.Str("loading"))
        self.context = context

        if context is not None and getattr(context, "config", None) is not None:
            self.config = context.config

        self.df = self.config.main_df
        self.current_extension_plot = None

        self._child_controllers = []
        self._disposed = False

        self._close_button = pn.widgets.Button(
            name="Close",
            width=100,
            height=28,
            min_height=28,
            max_height=28,
            sizing_mode="fixed",
            margin=(8, 0, 8, 4),
        )
        self._close_button.on_click(self._close_button_cb)

        self._submit_button = pn.widgets.Button(name="Submit Column Names")
        self._submit_button.on_click(self._submit_button_cb)

        self.plot_dict = extension_plots.get_plot_dict()
        self.cust_plot_dict = custom_plots.get_customplot_dict(context=context)
        self.contents = contents

    def _refresh_plot_registries(self):
        self.plot_dict = extension_plots.get_plot_dict()
        self.cust_plot_dict = custom_plots.get_customplot_dict(context=self.context)

    def _available_content_names(self) -> set[str]:
        """Return currently selectable content names.

        Built-in dashboard/workflow modes must always be included here. They are not
        plugin panels and are not present in custom_plots or extension_plots.

        Plugin/custom/extension entries are dynamic and may disappear when a plugin
        is disabled.
        """

        return (
            set(self.NATIVE_CONTENTS)
            | set(self.plot_dict.keys())
            | set(self.cust_plot_dict.keys())
        )

    def _fixed_toolbar(self):
        show_close = getattr(self, "contents", None) is not None

        if not show_close:
            return pn.Spacer(
                height=1,
                min_height=1,
                max_height=1,
                sizing_mode="stretch_width",
            )

        return pn.Row(
            self._close_button,
            pn.Spacer(sizing_mode="stretch_width"),
            height=52,
            min_height=52,
            max_height=52,
            sizing_mode="stretch_width",
            align="center",
            margin=(0, 0, 0, 0),
            styles={
                "flex": "0 0 52px",
                "min-height": "52px",
                "max-height": "52px",
                "overflow": "visible",
            },
        )

    def show_message(self, title: str, message: str, *, level: str = "warning") -> None:
        """Replace panel contents with a simple non-crashing message panel."""

        icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
        }.get(level, "⚠️")

        back_button = pn.widgets.Button(
            name="Back to Menu",
            button_type="primary",
            width=140,
            height=34,
        )

        message_panel = pn.Column(
            pn.Spacer(height=12),
            pn.pane.Markdown(
                f"## {icon} {title}\n\n{message}",
                sizing_mode="stretch_width",
            ),
            back_button,
            sizing_mode="stretch_both",
            margin=(8, 12, 8, 12),
        )

        back_button.on_click(lambda _event: self.set_contents("Menu"))

        self.panel_contents = message_panel
        self.row[0] = pn.Column(
            self._fixed_toolbar(),
            message_panel,
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 0),
            styles={"overflow": "hidden"},
        )

    def _dataset_is_loaded(self) -> bool:
        return getattr(self.config, "main_df", None) is not None and not self.config.main_df.empty

    def _require_loaded_dataset(self) -> bool:
        if not self._dataset_is_loaded():
            self.contents = "Menu"
            print("Please load a dataset before accessing this view.")
            return False
        return True

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
 
    def _submit_button_cb(self, event):
        self._submit_button.name = "Loading Plot..."
        self._submit_button.disabled = True
        if hasattr(self.plot_dict[self.contents], "col_selection"):
            for i, row in enumerate(self.plot_dict[self.contents].col_selection):
                if i == len(self.plot_dict[self.contents].col_selection) - 1:
                    continue
                for selector in row:
                    self.config.settings[selector.name] = selector.value
        else:
            self._submit_button.name = "Submit updated column names"
            self._submit_button.disabled = False
            self.plot_dict[self.contents]._load_file()
        self._update_contents()

    def dispose(self) -> None:
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        print(f"[dispose] Dashboard {getattr(self, 'contents', '')}")

        try:
            self.unwatch_all_bokeh()
        except Exception:
            pass

        self._dispose_current_contents_only()

        for ctrl in list(getattr(self, "_child_controllers", [])):
            try:
                if hasattr(ctrl, "dispose"):
                    ctrl.dispose()
            except Exception:
                pass
        self._child_controllers = []


    def _dispose_current_contents_only(self):
        """
        Internal: dispose whatever is currently active (panel_contents / current_extension_plot).
        Never call this from outside the class; external callers use dispose().
        """
        # Prefer disposing current_extension_plot if it exists
        cep = getattr(self, "current_extension_plot", None)
        if cep is not None:
            if hasattr(cep, "dispose"):
                try:
                    cep.dispose()
                except Exception:
                    pass
            else:
                assert False, f"{cep} missing dispose"

            self.current_extension_plot = None

        pc = getattr(self, "panel_contents", None)
        assert pc is None or hasattr(pc, "dispose"), f"{pc} missing dispose"
        if pc is not None:
            if hasattr(pc, "dispose"):
                try:
                    pc.dispose()
                except Exception:
                    pass

    def _close_button_cb(self, event):
        self._dispose_current_contents_only()
        self.contents = "Menu"
    

    def _update_extension_plots_cb(self, attr, old, new):
        if self.contents in self.plot_dict:

            ctrl = self.plot_dict[self.contents]

            self.current_extension_plot = ctrl
            self.panel_contents = ctrl

            render_fn = ctrl.plot(self._submit_button)
            render_fn(self.config.main_df, self.src)
            self.panel()
            
    @param.depends("contents", watch=True)
    def _update_contents(self):

        if self.contents == "Settings":
            self.mode = ""
            self.panel_contents = SettingsDashboard(self, self.src, context=self.context)

        elif self.contents == "Menu":
            self.panel_contents = MenuDashboard(self, context=self.context)

        elif self.contents == "Active Learning":
            if not self._require_loaded_dataset():
                return
            self.df = self.config.main_df
            self.panel_contents = ActiveLearningDashboard(self.src, self.df, context=self.context)

        elif self.contents == "Histogram Plot":
            if not self._require_loaded_dataset():
                return
            self.panel_contents = HistoDashboard(self._close_button, context=self.context)

        elif self.contents == "Basic Plot":
            if not self._require_loaded_dataset():
                return
            self.panel_contents = ScatterPlotDashboard(self._close_button, context=self.context)

        elif self.contents == "Density Plot":
            if not self._require_loaded_dataset():
                return
            self.panel_contents = DensityPlotDashboard(self._close_button, context=self.context)

        elif self.contents == "Labelling":
            if not self._require_loaded_dataset():
                return
            self.df = self.config.main_df
            self.panel_contents = LabellingDashboard(self.src, self.df, context=self.context)

        elif self.contents == "Selected Source Info":
            if not self._require_loaded_dataset():
                return
            self.panel_contents = SelectedSourceDashboard(self.src, self._close_button, context=self.context)

        elif self.contents in self.cust_plot_dict:
            if not self._require_loaded_dataset():
                return
            self.panel_contents = self.cust_plot_dict[self.contents](
                self.config.main_df,
                self._close_button,
                context=self.context,
            )

        else:
            if not self._require_loaded_dataset():
                return

            ctrl = self.plot_dict[self.contents](context=self.context)
            self.current_extension_plot = ctrl
            self.panel_contents = ctrl

            render_fn = ctrl.plot(self._submit_button)
            render_fn(self.config.main_df, self.src)

        self.panel()

    def set_contents(self, updated):
        """Update the current dashboard by setting a new `contents`."""

        self._refresh_plot_registries()

        # Built-in dashboard modes must be allowed through so they can perform their
        # own setup, such as Exploration requesting missing column mappings.
        if updated in self.NATIVE_CONTENTS:
            self.contents = updated
            return

        # Dynamic entries need validation because plugin-backed menu items can go
        # stale after a plugin is disabled.
        if updated not in self._available_content_names():
            self.show_message(
                title="Plot unavailable",
                message=(
                    f"`{updated}` is no longer available. "
                    "It may belong to a plugin that has been disabled."
                ),
                level="warning",
            )
            return

        self.contents = updated
    
    def panel(self):
        pc = getattr(self, "panel_contents", None)

        # Build body first
        if pc is not None and hasattr(pc, "panel"):
            body = pc.panel()
        else:
            body = pc

        # Detect provided toolbar
        provided_toolbar = None
        if pc is not None and hasattr(pc, "get_toolbar"):
            try:
                provided_toolbar = pc.get_toolbar()
            except Exception:
                provided_toolbar = None

        if provided_toolbar is None and pc is not None and hasattr(pc, "toolbar"):
            try:
                provided_toolbar = pc.toolbar
            except Exception:
                provided_toolbar = None

        # Decide toolbar: EITHER provided OR default, never both
        if provided_toolbar is None:
            toolbar = self._fixed_toolbar()
            self.row[0] = pn.Column(
                toolbar,
                body,
                sizing_mode="stretch_both",
                margin=(0, 0, 0, 0),
                styles={"overflow": "hidden"},
            )
        else:
            self.row[0] = body
        
        
        return self.row

