# load_data(), clean_duplicates(), remove_outliers()


import pandas as pd
from scipy.stats import zscore

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def move_class_column(df: pd.DataFrame) -> pd.DataFrame:
    class_col = df.pop("class")
    df["class"] = class_col
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def remove_outliers(df: pd.DataFrame, features, threshold=2.5):
    z = df[features].apply(zscore)
    mask = (z.abs() <= threshold).all(axis=1)
    return df[mask].reset_index(drop=True)
