import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"

sns.set(style="whitegrid")


def plot_class_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    df["class"].value_counts().plot(kind="bar", color=["#4C72B0", "#55A868"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_histograms(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols].hist(figsize=(14, 10), bins=30, color="#4C72B0")
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col], color="#55A868")
        plt.title(col)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    plt.close()


# def plot_pairplot(df: pd.DataFrame, sample_size=500):
#     if len(df) > sample_size:
#         df_sample = df.sample(sample_size, random_state=42)
#     else:
#         df_sample = df

#     sns.pairplot(df_sample, hue="class", diag_kind="kde",
#                  palette=["#4C72B0", "#55A868"])
#     plt.show()
#     plt.close()
