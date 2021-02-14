from active_learning.active_learning import ActiveLearningTab

import config
import panel as pn
import param


class ActiveLearningDashboard(param.Parameterized):

    def __init__(self, src, df, **params):
        super(ActiveLearningDashboard, self).__init__(**params)

        self.df = df
        self.src = src
        self.row = pn.Row(pn.pane.Str("loading"))

        self.add_active_learning()

    def add_active_learning(self):

        self.active_learning = []
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
            self.al_tabs.append((al_tab.label_string, al_tab.panel()))
        self.panel()

    def panel(self):
        self.row[0] = pn.Card(self.al_tabs)
        return self.row
