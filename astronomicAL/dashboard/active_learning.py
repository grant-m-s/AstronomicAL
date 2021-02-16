from active_learning.active_learning import ActiveLearningTab

import config
import panel as pn
import param


class ActiveLearningDashboard(param.Parameterized):
    """A dashboard for initialising and rendering panels for Active Learning.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.
    df : DataFrame
        The shared dataframe which holds all the data.
    active_learning : list of ActiveLearningTab
        List of all classifier tab views.

    Attributes
    ----------
    row : Panel Row
        The layout of the dashboard is housed in this row.

    """

    def __init__(self, src, df, **params):
        super(ActiveLearningDashboard, self).__init__(**params)

        self.df = df
        self.src = src
        self.row = pn.Row(pn.pane.Str("loading"))
        self.active_learning = []

        self.add_active_learning()

    def add_active_learning(self):
        """Initialise all required ActiveLearningTab classifiers and views.

        Returns
        -------
        None

        """
        # CHANGED :: Add to AL settings
        for label in config.settings["labels_to_train"]:
            print(f"Label is {label} with type: {type(label)}")
            raw_label = config.settings["strings_to_labels"][label]
            print(f"Raw Label is {raw_label} with type: {type(raw_label)}")
            self.active_learning.append(
                ActiveLearningTab(df=self.df, src=self.src, label=label)
            )
        self.al_tabs = pn.Tabs(dynamic=True)
        for i, al_tab in enumerate(self.active_learning):
            self.al_tabs.append((al_tab._label_alias, al_tab.panel()))

        print("Added all classifiers")
        self.panel()

    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which will can then be rendered by the
            parent Dashboard.

        """
        self.row[0] = pn.Card(self.al_tabs)
        return self.row
