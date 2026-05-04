import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_data, move_class_column, remove_duplicates, remove_outliers
from src.transform import transform_features
from src.features import drop_correlated, select_features_sfs
from src.model import tune_xgboost, save_model
from src.evaluate import evaluate_model

from src.eda import (
    plot_class_distribution,
    plot_histograms,
    plot_boxplots,
    plot_correlation_heatmap,
    plot_pairplot
)


def main():

    # 1. Load data
    df = pd.read_csv("dataset/mushroom_cleaned.csv")
    df = move_class_column(df)
    df = remove_duplicates(df)

    # 2. Outlier removal
    outlier_features = ["cap-diameter", "stem-height", "stem-width"]
    df = remove_outliers(df, outlier_features)

    # 3. Transform
    df = transform_features(df)

    # 4. EDA (run BEFORE splitting)
    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    # plot_pairplot(df)  # optional

    # 5. Split
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Drop correlated features
    X_train, dropped = drop_correlated(X_train)
    X_test = X_test.drop(columns=dropped)

    # 7. Feature selection
    selected = select_features_sfs(X_train, y_train)
    X_train = X_train[selected]
    X_test = X_test[selected]

    # 8. Train + tune model
    best_model, params, cv_score = tune_xgboost(X_train, y_train)

    # 9. Evaluate
    acc, report, cm = evaluate_model(best_model, X_test, y_test)

    print("\nBest XGBoost Params:", params)
    print("CV Accuracy:", cv_score)
    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # 10. Save model
    save_model(best_model)


if __name__ == "__main__":
    main()
