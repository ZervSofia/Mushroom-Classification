# Mushroom Classification Project

## Overview

This project builds a complete machine learning pipeline to classify mushrooms based on their physical characteristics.

The pipeline includes data preprocessing, feature engineering, feature selection, and model training using **XGBoost**.

---

## Pipeline

The workflow consists of:

1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Outlier removal and feature transformation
4. Train/test split
5. Correlation-based feature removal
6. Feature selection (Sequential Feature Selection)
7. Model training (XGBoost + GridSearchCV)
8. Evaluation and model saving

---

## Results

The final model achieved strong performance on the test set:

* **Accuracy:** 97.08%
* **Precision / Recall / F1-score:** ~0.97

The model performs consistently across both classes, indicating balanced predictions.

### Confusion Matrix

```
[[4479  149]
 [ 145 5303]]
```

Only a small number of misclassifications are observed relative to the dataset size (~10,000 samples).

---

## Key Techniques

* **Model:** XGBoost Classifier
* **Feature Selection:** Sequential Floating Forward Selection (SFS)
* **Hyperparameter Tuning:** GridSearchCV
* **Data Processing:** Outlier removal + feature transformations

---

## Future Improvements

* Compare with additional models (e.g., Logistic Regression, Random Forest)
* Optimize feature selection runtime
* Add cross-validation visualization
* Deploy as an API or web application


