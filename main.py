import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Data + cleaning
from src.data import (
    load_data,
    move_class_column,
    remove_duplicates,
    remove_outliers
)

# Transformations
from src.transform import transform_features

# Feature engineering
from src.features import drop_correlated, select_features_sfs

# Modeling
from src.model import tune_xgboost, save_model

# Evaluation
from src.evaluate import evaluate_model

# EDA
from src.eda import (
    plot_class_distribution,
    plot_histograms,
    plot_boxplots,
    plot_correlation_heatmap,
    # plot_pairplot
)


def run_eda(df):
    """Run all EDA visualizations on the raw dataset."""
    print("\n=== Running EDA ===")
    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)

    # Pairplt is slow — enable only if needed
    # plot_pairplot(df)



def main():
    steps = [
        "Loading data",
        "Running EDA",
        "Removing outliers",
        "Transforming features",
        "Splitting data",
        "Dropping correlated features",
        "Selecting features (SFS)",
        "Training + tuning XGBoost",
        "Evaluating model",
        "Saving model"
    ]

    pbar = tqdm(steps)

    # 1. Load data
    pbar.set_description("Loading data")
    df = load_data("dataset/mushroom_cleaned.csv")
    df = move_class_column(df)
    df = remove_duplicates(df)
    pbar.update()

    # 2. EDA
    pbar.set_description("Running EDA")
    run_eda(df)
    pbar.update()

    # 3. Outliers
    pbar.set_description("Removing outliers")
    df = remove_outliers(df, ["cap-diameter", "stem-height", "stem-width"])
    pbar.update()

    # 4. Transform
    pbar.set_description("Transforming features")
    df = transform_features(df)
    pbar.update()

    # 5. Split
    pbar.set_description("Splitting data")
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pbar.update()

    # 6. Drop correlated
    pbar.set_description("Dropping correlated features")
    X_train, dropped = drop_correlated(X_train)
    X_test = X_test.drop(columns=dropped, errors="ignore")
    pbar.update()

    # 7. SFS
    pbar.set_description("Selecting features (SFS)")
    selected = select_features_sfs(X_train, y_train)
    X_train = X_train[selected]
    X_test = X_test[selected]
    pbar.update()

    # 8. Train + tune
    pbar.set_description("Training + tuning XGBoost")
    best_model, params, cv_score = tune_xgboost(X_train, y_train)
    pbar.update()

    # 9. Evaluate
    pbar.set_description("Evaluating model")
    acc, report, cm = evaluate_model(best_model, X_test, y_test)
    pbar.update()
    
    # Print results
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # 10. Save
    pbar.set_description("Saving model")
    save_model(best_model)
    pbar.update()

    pbar.close()


if __name__ == "__main__":
    main()
