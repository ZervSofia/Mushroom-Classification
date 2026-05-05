import joblib
import os
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost using GridSearchCV.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }

    print("\nRunning GridSearchCV...")

    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss"),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2  # 👈 this is enough for progress
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def save_model(model, path="models/best_xgb_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)