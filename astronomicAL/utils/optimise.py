import pandas as pd

# Following optimisation functions credited to:
# https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2


def optimise_floats(df):
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimise_ints(df):
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimise_objects(df):
    for col in df.select_dtypes(include=["object"]):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if float(num_unique_values) / num_total_values < 0.5:
            df[col] = df[col].astype("category")
    return df


def optimise(df):
    return optimise_floats(optimise_ints(optimise_objects(df)))
