# correlation_filter(), sfs_selection()


import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier

def drop_correlated(df: pd.DataFrame, threshold=0.8):
    corr = df.corr()
    to_drop = set()

    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                to_drop.add(corr.columns[i])

    return df.drop(columns=list(to_drop)), list(to_drop)

def select_features_sfs(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    sfs = SFS(
        model,
        k_features="best",
        forward=True,
        floating=True,
        scoring="accuracy",
        cv=5
    )

    sfs = sfs.fit(X, y)
    selected = list(sfs.k_feature_names_)
    return selected
