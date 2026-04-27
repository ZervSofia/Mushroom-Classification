# load_data(), clean_duplicates(), remove_outliers()


import pandas as pd
from scipy.stats import zscore
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the mushroom dataset from CSV or Excel, even if mislabeled."""
    try:
        # Try CSV first
        return pd.read_csv(path)
    except Exception:
        # Fall back to Excel
        return pd.read_excel(path, engine="xlrd")




def move_class_column(df: pd.DataFrame) -> pd.DataFrame:
    class_col = df.pop("class")
    df["class"] = class_col
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def remove_outliers(df: pd.DataFrame, features: list, threshold: float = 2.5) -> pd.DataFrame:
    """
    Remove rows where any selected feature has a z-score above the threshold.
    """
    z = df[features].apply(zscore)
    mask = (z.abs() <= threshold).all(axis=1)
    return df[mask].reset_index(drop=True)
