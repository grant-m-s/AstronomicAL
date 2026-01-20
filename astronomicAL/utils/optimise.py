import pandas as pd
import numpy as np
import pandas.api.types as pdt

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




########### Maybe this is not the best place where to define this function but here we are also dealing with column type

def matches_type(dtype, type_list):
    """Check if a dtype matches any keyword in type_list"""
    for t in type_list:
        if t == "bool" and pdt.is_bool_dtype(dtype):
            return True
        if t == "float" and (pdt.is_float_dtype(dtype)):
            return True
        if t == "int" and (pdt.is_integer_dtype(dtype)):
            return True
        if t == "number" and pdt.is_numeric_dtype(dtype) and not pdt.is_bool_dtype(dtype):
            return True
        if t == "object" and (pdt.is_object_dtype(dtype)):
            return True
    return False


def get_series_type(s: pd.Series):
    """Returns the dtype of the passed series. 'mixed' is for Series with both numbers and strings """
    s = s.dropna()
    if s.empty:
        return "empty" 
    if pdt.is_bool_dtype(s):
        return "bool"
    if pdt.is_integer_dtype(s):
        return "int"
    if pdt.is_float_dtype(s):
        return "float"
    if pdt.is_string_dtype(s):
        return "string"
    if pdt.is_object_dtype(s):
        types = set(s.map(type))
        if types == {str}:
            return "string"
        if types <= {int, bool}:
            return "int"
        if types <= {int, float, bool, np.number}:
            return "float"
        return "mixed"

    return "mixed"





