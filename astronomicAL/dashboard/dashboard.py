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
        self.src.on_change("data", self._update_extension_plots_cb)
        self.row = pn.Row(pn.pane.Str("loading"))

        self.context = context
        import astronomicAL.config as config
        self.config = context.config if (context is not None and getattr(context, "config", None) is not None) else config
        self.shared = getattr(context, "shared", None)
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

        # Always stop the src callback owned by the Dashboard itself
        try:
            self.src.remove_on_change("data", self._update_extension_plots_cb)
        except Exception:
            pass

        pc = getattr(self, "panel_contents", None)

        # Case 1: custom plot controllers (EuclidPlotClass, SpectrumPlotClass, etc.)
        if pc is not None and hasattr(pc, "dispose"):
            try:
                pc.dispose()
            except Exception:
                pass

        # Case 2: extension plot workflow (wrapper object + cleanup_panel_plot)
        else:
            # Clean up wrapper state if relevant
            try:
                self._cleanup_current_extension_plot()
            except Exception:
                pass

            # Also allow panel_contents cleanup if it exists (but no dispose)
            if pc is not None and hasattr(pc, "cleanup_panel_plot"):
                try:
                    pc.cleanup_panel_plot()
                except Exception:
                    pass

        # Optional: dispose any child controllers if you are using them elsewhere
        for ctrl in list(getattr(self, "_child_controllers", [])):
            try:
                if hasattr(ctrl, "dispose"):
                    ctrl.dispose()
            except Exception:
                pass
        self._child_controllers = []

    def _close_button_cb(self, event):
        self._cleanup_current_extension_plot()
        self.contents = "Menu"
    

    def _update_extension_plots_cb(self, attr, old, new):
        if self.contents in list(self.plot_dict.keys()):
            self.current_extension_plot = self.plot_dict[self.contents]
            self.panel_contents = self.plot_dict[self.contents].plot(self._submit_button)(
                self.config.main_df, self.src
            )
            self.panel()

    
    def _cleanup_current_extension_plot(self):
        if self.current_extension_plot and hasattr(self.current_extension_plot, 'cleanup_panel_plot'):
            self.current_extension_plot.cleanup_panel_plot()
        
        elif hasattr(self.panel_contents, "cleanup_panel_plot"):
            self.panel_contents.cleanup_panel_plot()
        self.current_extension_plot = None
            
            
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
        
        else:
            self.current_extension_plot = self.plot_dict[self.contents]
            self.panel_contents = self.current_extension_plot.plot(
                self._submit_button
            )(self.config.main_df, self.src)

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
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel contents is housed in a row which can then be
            rendered by the Panel layout.

        """
        if hasattr(self.panel_contents, "panel"):
            self.row[0] = self.panel_contents.panel()
        elif hasattr(self.panel_contents, "mypanel"):
            self.row[0] = self.panel_contents.mypanel
        else:
            toolbar = pn.Row(self._close_button, max_height=50)
            body = self.panel_contents
            self.row[0] = pn.Column(
                toolbar,body,
            )

        try:
            self.row._al_controller = self
            self.row.dispose = self.dispose
        except Exception:
            pass

        return self.row
    
