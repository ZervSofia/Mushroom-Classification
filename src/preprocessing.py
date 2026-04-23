import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates and resetting index.
    """
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y).
    """
    X = df.drop("class", axis=1)
    y = df["class"]
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    - Clean data
    - Split into X and y
    - Train/test split
    """
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return X_train, X_test, y_train, y_test