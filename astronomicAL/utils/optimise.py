import pandas as pd
import time
# Following optimisation functions credited to:
# https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2


def optimise_floats(df):
    start = time.time()
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    end = time.time()
    print(f"optimise_floats {end - start}")
    return df


def optimise_ints(df):
    start = time.time()
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    end = time.time()
    print(f"optimise_ints {end - start}")
    return df


def optimise_objects(df):
    start = time.time()
    for col in df.select_dtypes(include=['object']):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if float(num_unique_values) / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    end = time.time()
    print(f"optimise objects {end - start}")
    return df


def optimise(df):
    return optimise_floats(optimise_ints(optimise_objects(df)))
