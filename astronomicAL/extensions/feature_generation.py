from itertools import combinations

import astronomicAL.config as config
import numpy as np


def get_oper_dict():

    oper = {
        "subtract (a-b)": subtract,
        "add (a+b)": add,
        "multiply (a*b)": multiply,
        "divide (a/b)": divide,
    }

    return oper


def add(df, n):
    print("gen add features")
    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)

    generated_features = []

    for comb in combs:
        col = ""
        for i in range(n):
            col = col + f"{comb[i]}"
            if i != (n - 1):
                col = col + "+"
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] + df[comb[i]]

    return df, generated_features


def subtract(df, n):
    print("gen subtract features")
    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = ""
        for i in range(n):
            col = col + f"{comb[i]}"
            if i != (n - 1):
                col = col + "-"
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] - df[comb[i]]

    return df, generated_features


def multiply(df, n):
    print("gen multiply features")
    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = ""
        for i in range(n):
            col = col + f"{comb[i]}"
            if i != (n - 1):
                col = col + "*"
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] * df[comb[i]]

    return df, generated_features


def divide(df, n):
    print("gen divide features")
    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = ""
        for i in range(n):
            col = col + f"{comb[i]}"
            if i != (n - 1):
                col = col + "/"
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] / df[comb[i]]

    return df, generated_features


def generate_features(df):
    """Create the feature combinations that the user specified.

    Parameters
    ----------
    df : DataFrame
        A dataframe containing all of the dataset.

    Returns
    -------
    df : DataFrame
        An expanding dataframe of `df` with the inclusion of the feature
        combinations.
    df_al : DataFrame
        A dataframe containing a subset of `df` with only the required
        features for training.

    """
    print("generate_features")
    np.random.seed(0)

    # CHANGED :: Change this to selected["AL_Features"]
    bands = config.settings["features_for_training"]

    print(bands)

    features = bands + [config.settings["label_col"], config.settings["id_col"]]
    print(features[-5:])
    df_al = df[features]

    shuffled = np.random.permutation(list(df_al.index.values))

    print("shuffled...")

    df_al = df_al.reindex(shuffled)

    print("reindexed")

    df_al = df_al.reset_index()

    print("reset index")

    combs = list(combinations(bands, 2))

    print(combs)

    cols = list(df.columns)

    for i, j in combs:
        df_al[f"{i}-{j}"] = df_al[i] - df_al[j]

        if f"{i}-{j}" not in cols:
            df[f"{i}-{j}"] = df[i] - df[j]

    print("Feature generations complete")

    return df, df_al
