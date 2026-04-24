# train_models(), tune_models(), save_model()


import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def save_model(model, path="models/best_xgb_model.pkl"):
    joblib.dump(model, path)
