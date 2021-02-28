from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_classifiers():
    classifiers = {
        "KNN": KNeighborsClassifier(3, n_jobs=-1),
        "DTree": DecisionTreeClassifier(
            random_state=0,
        ),
        "RForest": RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=1000),
        "AdaBoost": AdaBoostClassifier(random_state=0, n_estimators=500),
        "GBTrees": GradientBoostingClassifier(random_state=0, n_estimators=1000),
    }
    return classifiers
