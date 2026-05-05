import pandas as pd
from scipy.stats import zscore


def load_data(path: str) -> pd.DataFrame:
    """
    Load the mushroom dataset from CSV or Excel.
    Handles mislabeled .xls files that are actually CSV.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"Could not load file {path}: {e}")


def move_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """Move the target column 'class' to the end of the dataframe."""
    df = df.copy()
    class_col = df.pop("class")
    df["class"] = class_col
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def remove_outliers(df: pd.DataFrame, features: list, threshold: float = 2.5) -> pd.DataFrame:
    """
    Remove rows where any selected feature has a z-score above the threshold.
    Only applies to continuous features.
    """
    df = df.copy()
    z = df[features].apply(zscore)
    mask = (z.abs() <= threshold).all(axis=1)
    return df[mask].reset_index(drop=True)
