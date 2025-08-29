from itertools import combinations

import astronomicAL.config as config
import numpy as np


def get_oper_dict():

    oper = {
        "subtract (a-b)": subtract,
        "add (a+b)": add,
        "color (-2.5log(a/b))" : color,
        "multiply (a*b)": multiply,
        "divide (a/b)": divide,
    }

    return oper


def add(df, n):

    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)

    generated_features = []

    for comb in combs:
        col = "+".join(comb)
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] + df[comb[i]]

    return df, generated_features


def subtract(df, n):

    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = "-".join(comb)
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] - df[comb[i]]

    return df, generated_features


def multiply(df, n):

    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = "*".join(comb)
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] * df[comb[i]]

    return df, generated_features


def divide(df, n):

    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    cols = list(df.columns)
    generated_features = []
    for comb in combs:
        col = "/".join(comb)
        generated_features.append(col)
        if col not in cols:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = df[col] / df[comb[i]]

    return df, generated_features



def color(df, n):
    if n > 2:
        print("It doesn't really make sense,,,")
    
    np.random.seed(0)

    bands = config.settings["features_for_training"]

    combs = list(combinations(bands, n))

    generated_features = []
    for comb in combs:
        col = "-".join(comb)
        generated_features.append(col)
        if col not in df.columns:
            for i in range(n):
                if i == 0:
                    df[col] = df[comb[i]]
                else:
                    df[col] = -2.5*np.log10(df[col]/df[comb[i]])

    return df, generated_features