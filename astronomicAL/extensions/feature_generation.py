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
