import numpy as np
import pandas as pd
from itertools import combinations

def get_oper_dict():

    oper = {
        "subtract (a-b)": subtract,
        "add (a+b)": add,
        "color (-2.5log(a/b))" : color,
        "multiply (a*b)": multiply,
        "divide (a/b)": divide,
    }

    return oper

def _ensure_float32(arr):
    # optional: keep memory down for huge data
    return arr.astype(np.float32, copy=False) if arr.dtype == np.float64 else arr

def add(df, n, batch_size=64, context = None):
    # if (context is not None and getattr(context, "config", None) is not None):
    config = context.config
    
    bands = config.settings["features_for_training"]
    combs = list(combinations(bands, n))

    new_cols = {}
    generated = []

    existing = set(df.columns)

    for comb in combs:
        colname = "+".join(comb)
        generated.append(colname)
        if colname in existing:
            continue

        # compute directly: sum of columns
        arr = df[comb[0]].to_numpy(copy=False)
        for b in comb[1:]:
            arr = arr + df[b].to_numpy(copy=False)

        new_cols[colname] = _ensure_float32(np.asarray(arr))

        # concat in batches to avoid huge dict growth
        if len(new_cols) >= batch_size:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            new_cols.clear()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # optional: defragment blocks once at the end
    df = df.copy()
    return df, generated

def subtract(df, n, batch_size=64, context = None):
    # if (context is not None and getattr(context, "config", None) is not None):
    config = context.config
    bands = config.settings["features_for_training"]
    combs = list(combinations(bands, n))
    new_cols, generated = {}, []
    existing = set(df.columns)

    for comb in combs:
        colname = "-".join(comb)
        generated.append(colname)
        if colname in existing:
            continue

        arr = df[comb[0]].to_numpy(copy=False)
        for b in comb[1:]:
            arr = arr - df[b].to_numpy(copy=False)

        new_cols[colname] = _ensure_float32(np.asarray(arr))
        if len(new_cols) >= batch_size:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            new_cols.clear()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.copy()
    return df, generated


def multiply(df, n, batch_size=64, context = None):

    # if (context is not None and getattr(context, "config", None) is not None):
    config = context.config

    bands = config.settings["features_for_training"]
    combs = list(combinations(bands, n))
    new_cols, generated = {}, []
    existing = set(df.columns)

    for comb in combs:
        colname = "*".join(comb)
        generated.append(colname)
        if colname in existing:
            continue

        arr = df[comb[0]].to_numpy(copy=False)
        for b in comb[1:]:
            arr = arr * df[b].to_numpy(copy=False)

        new_cols[colname] = _ensure_float32(np.asarray(arr))
        if len(new_cols) >= batch_size:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            new_cols.clear()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.copy()
    return df, generated


def divide(df, n, batch_size=64, eps=0.0, context = None):
    # if (context is not None and getattr(context, "config", None) is not None):
    config = context.config

    bands = config.settings["features_for_training"]
    combs = list(combinations(bands, n))
    new_cols, generated = {}, []
    existing = set(df.columns)

    for comb in combs:
        colname = "/".join(comb)
        generated.append(colname)
        if colname in existing:
            continue

        num = df[comb[0]].to_numpy(copy=False)
        arr = num
        for b in comb[1:]:
            den = df[b].to_numpy(copy=False)
            if eps:
                den = den + eps
            arr = arr / den

        new_cols[colname] = _ensure_float32(np.asarray(arr))
        if len(new_cols) >= batch_size:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            new_cols.clear()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.copy()
    return df, generated

def color(df, n=2, batch_size=64, context = None):
    if n != 2:
        raise ValueError("color only makes sense for n=2")

    # if (context is not None and getattr(context, "config", None) is not None):
    config = context.config

    bands = config.settings["features_for_training"]
    combs = list(combinations(bands, 2))

    new_cols, generated = {}, []
    existing = set(df.columns)

    for a, b in combs:
        colname = f"{a}-{b}"
        generated.append(colname)
        if colname in existing:
            continue

        A = df[a].to_numpy(copy=False)
        B = df[b].to_numpy(copy=False)
        arr = -2.5 * np.log10(A / B)

        new_cols[colname] = _ensure_float32(np.asarray(arr))

        if len(new_cols) >= batch_size:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            new_cols.clear()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    df = df.copy()
    return df, generated