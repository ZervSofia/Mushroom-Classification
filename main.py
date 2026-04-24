import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_data, move_class_column, remove_duplicates, remove_outliers
from src.transform import transform_features
from src.features import drop_correlated, select_features_sfs
from src.model import tune_xgboost, save_model
from src.evaluate import evaluate_model

def main():

    # 1. Load data
    df = load_data("data/raw/mushroom_cleaned.csv")
    df = move_class_column(df)
    df = remove_duplicates(df)

    # 2. Outlier removal
    outlier_features = ["cap-diameter", "stem-height", "cap-shape",
                        "gill-attachment", "gill-color", "stem-color"]
    df = remove_outliers(df, outlier_features)

    # 3. Transform
    df = transform_features(df)

    # 4. Split
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Drop correlated features
    X_train, dropped = drop_correlated(X_train)
    X_test = X_test.drop(columns=dropped)

    # 6. Feature selection
    selected = select_features_sfs(X_train, y_train)
    X_train = X_train[selected]
    X_test = X_test[selected]

    # 7. Train + tune model
    best_model, params, cv_score = tune_xgboost(X_train, y_train)

    # 8. Evaluate
    acc, report, cm = evaluate_model(best_model, X_test, y_test)

    print("\nBest XGBoost Params:", params)
    print("CV Accuracy:", cv_score)
    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # 9. Save model
    save_model(best_model)

if __name__ == "__main__":
    main()
