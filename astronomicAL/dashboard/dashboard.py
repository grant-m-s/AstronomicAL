from astronomicAL.dashboard.active_learning import ActiveLearningDashboard
from astronomicAL.dashboard.labelling import LabellingDashboard
from astronomicAL.dashboard.exploration import ExplorationDashboard
from astronomicAL.dashboard.menu import MenuDashboard
from astronomicAL.dashboard.plot import HistoDashboard, ScatterPlotDashboard, DensityPlotDashboard
from astronomicAL.dashboard.selected_source import SelectedSourceDashboard
from astronomicAL.dashboard.settings_dashboard import SettingsDashboard
from astronomicAL.extensions import extension_plots, custom_plots
from bokeh.models import ColumnDataSource

import panel as pn
import param


class Dashboard(param.Parameterized):
    """Top-level Dashboard which holds an instance of any other Dashboard.

    This will initialize and render any of the other instances of the
    dashboards and handles the interactions of switching from one dashboard
    view to another.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.
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

    src = ColumnDataSource(data={"0": [], "1": []})

    contents = param.String()

    def __init__(self, src, contents= "Menu", context=None):
        super(Dashboard, self).__init__()

        self.src = src
        self.watch_bokeh(self.src, "data", self._update_extension_plots_cb)
        self.row = pn.Row(pn.pane.Str("loading"))

        self.context = context

        if (context is not None and getattr(context, "config", None) is not None):
            self.config = context.config 
        self.df = self.config.main_df
        self.current_extension_plot = None

        self._child_controllers = []  # controllers that should be disposed with this dashboard

        self._disposed = False
        
        self._close_button = pn.widgets.Button(name="Close", max_width=100, max_height=40)
        self._close_button.on_click(self._close_button_cb)
        
        self._submit_button = pn.widgets.Button(name="Submit Column Names")
        self._submit_button.on_click(self._submit_button_cb)
        
        self.plot_dict = extension_plots.get_plot_dict()
        self.cust_plot_dict = custom_plots.get_customplot_dict()
        self.contents = contents

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

        # stop dashboard-owned src callback
        try:
            self.src.remove_on_change("data", self._update_extension_plots_cb)
        except Exception:
            pass

        # dispose active contents (custom plots or extension plots)
        self._dispose_current_contents_only()

        # dispose any child controllers if you are using them elsewhere
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

            self.panel_contents = MenuDashboard(self)

        elif self.contents == "Active Learning":

            self.df = self.config.main_df
            self.panel_contents = ActiveLearningDashboard(self.src, self.df, context=self.context)

        elif self.contents == "Histogram Plot":
            if not self.config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = HistoDashboard(self.src, self._close_button, context=self.context)
        
        elif self.contents == "Basic Plot":
            if not self.config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = ScatterPlotDashboard(self.src, self._close_button, context=self.context)
        
        elif self.contents == "Density Plot":
            if not self.config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = DensityPlotDashboard(self.src, self._close_button, context=self.context)

        elif self.contents == "Labelling":
            self.df = self.config.main_df
            self.panel_contents = LabellingDashboard(self.src, self.df, context=self.context)
        
        elif self.contents == "Exploring":
            self.df = self.config.main_df
            self.panel_contents = ExplorationDashboard(self.src, self.df, context=self.context)

        elif self.contents == "Selected Source Info":
            if not self.config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = SelectedSourceDashboard(self.src, self._close_button, context=self.context)
        
        elif self.contents in self.cust_plot_dict:
            if not self.config.settings["confirmed"]:
                self.contents = "Menu"
                print("Please Complete Settings before accessing this view.")
                return
            self.panel_contents = self.cust_plot_dict[self.contents](self.config.main_df, self.src, self._close_button, context=self.context)
        
        # elif self.contents in self.plot_dict:
        #     if not self.config.settings["confirmed"]:
        #         self.contents = "Menu"
        #         print("Please Complete Settings before accessing this view.")
        #         return
        #     self.panel_contents = self.plot_dict[self.contents](context=self.context)
        
        else:

            ctrl = self.plot_dict[self.contents](context=self.context)          # CustomPlot controller

            self.current_extension_plot = ctrl
            self.panel_contents = ctrl                    # MUST be controller

            render_fn = ctrl.plot(self._submit_button)
            render_fn(self.config.main_df, self.src)      # populate ctrl.row

        self.panel()

    def set_contents(self, updated):
        """Update the current dashboard by setting a new `contents`.

        Parameters
        ----------
        updated : str
            The new contents view required.

        Returns
        -------
        None

        """

        self.contents = updated

    def panel(self):
        """
        Render the current view into self.row[0].
        Always includes the dashboard toolbar (Close button etc.).
        """
        pc = getattr(self, "panel_contents", None)

        # Only show the dashboard Close button when NOT on Menu
        show_close = getattr(self, "contents", None) not in (None, "Menu", "Settings", "Active Learning", "Labelling", "Exploring")
        toolbar = pn.Row(self._close_button, max_height=50) if show_close else pn.Spacer(height=1)

        if pc is not None and hasattr(pc, "panel"):
            body = pc.panel()
            self.row[0] = pn.Column(toolbar, body, sizing_mode="stretch_both")
        else:
            body = pc
            self.row[0] = pn.Column(toolbar, body, sizing_mode="stretch_both")

        return self.row
    
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
            show_close = getattr(self, "contents", None) is not None
            toolbar = pn.Row(self._close_button, max_height=50) if show_close else pn.Spacer(height=1)
            self.row[0] = pn.Column(toolbar, body, sizing_mode="stretch_both")
        else:
            self.row[0] = body
        
        
        return self.row

