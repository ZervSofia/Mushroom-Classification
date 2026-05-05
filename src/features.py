import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm


def drop_correlated(df: pd.DataFrame, threshold=0.8):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                to_drop.add(corr.columns[i])

    return df.drop(columns=list(to_drop)), list(to_drop)


def select_features_sfs(X, y):
    """
    Perform Sequential Floating Forward Selection with a progress bar.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Wrap SFS iterations with tqdm
    tqdm._instances.clear()  # prevents duplicate bars

    sfs = SFS(
        model,
        k_features="best",
        forward=True,
        floating=True,
        scoring="accuracy",
        cv=5,
        verbose=2  # enables tqdm-friendly output
    )

    print("\nRunning Sequential Feature Selection...")
    sfs = sfs.fit(X, y)

    return list(sfs.k_feature_names_)
