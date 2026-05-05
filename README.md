# Mushroom Classification Project

##  Overview

This project implements a complete machine learning pipeline to classify mushrooms based on their physical characteristics.

The workflow includes:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training and hyperparameter tuning
- Evaluation and model persistence

The final model is built using **XGBoost** and achieves high classification performance on the test dataset.

---

## Pipeline Steps

The pipeline in **main.py** performs:

1. **Load Data**
2. **Exploratory Data Analysis (EDA)**

   - Class distribution
   - Histograms
   - Boxplots
   - Correlation heatmap
3. **Remove Duplicates & Outliers**
4. **Feature Transformation**

   - Box-Cox transformation
   - Cube-root scaling
5. **Train/Test Split**
6. **Drop Correlated Features**
7. **Feature Selection**

   - Sequential Floating Forward Selection (SFS)
8. **Model Training**

   - XGBoost + GridSearchCV
9. **Evaluation**
10. **Model Saving**

---

## Results

The final model achieved strong performance on the test set:

* **Accuracy:** 97.08%
* **Precision / Recall / F1-score:** ~0.97 for both classes


### Confusion Matrix

```
[[4479  149]
 [ 145 5303]]
```

The model demonstrates excellent balance between precision and recall, with very few misclassifications across both classes.

---

### Interpretation of Results

The model achieves an accuracy of **97.08%**, indicating very strong overall performance on unseen data.

All evaluation metrics (precision, recall, and F1-score) are consistently around **0.97** for both classes. This suggests that the model performs equally well across classes and is not biased toward a specific category.

From the confusion matrix:

* **False Positives:** 149
* **False Negatives:** 145

These errors are relatively small compared to the total number of samples (~10,000), confirming that misclassifications are minimal.

The near-equal precision and recall values indicate a good balance between:

* avoiding false alarms (precision)
* correctly identifying true cases (recall)

### Why the Model Performs Well

Several factors contribute to the high performance:

* The dataset appears to be **well-structured and highly separable**
* Feature transformations (Box-Cox, cube-root) improve data distribution
* Correlated features are removed, reducing redundancy
* Sequential Feature Selection (SFS) identifies the most informative features
* Hyperparameter tuning (GridSearchCV) optimizes model performance

### Important Consideration

The consistently high metrics may indicate that the dataset is relatively easy to classify. This is common in structured datasets where features strongly differentiate classes.

Therefore, while the model performs very well, results may not generalize equally to more complex or noisy datasets.


---

## Model Details

* **Algorithm:** XGBoost Classifier
* **Feature Selection:** Sequential Floating Forward Selection (SFS)
* **Hyperparameter Tuning:** GridSearchCV
* **Evaluation Metrics:**

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix

---

## Notes

* The dataset is relatively well-structured and separable, which contributes to the high model performance.
* Feature selection significantly reduces dimensionality while maintaining accuracy.
* The pipeline is modular and easily extensible.

---

## Future Improvements

* Compare with additional models (e.g., Logistic Regression, Random Forest)
* Add cross-validation performance visualization
* Optimize feature selection runtime
* Deploy the model as an API or web application



