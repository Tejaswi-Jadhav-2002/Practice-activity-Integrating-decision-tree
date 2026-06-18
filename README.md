# Student Pass/Fail Prediction using Decision Tree Classifier

## Overview

This project demonstrates how to build and evaluate a **Decision Tree Classifier** using Python and Scikit-learn. The model predicts whether a student will pass or fail based on two features:

* Study Hours
* Previous Exam Score

Decision Trees are powerful and interpretable machine learning algorithms that can be used for both classification and regression tasks. In this project, we focus on binary classification and explore model training, evaluation, visualization, and tuning to improve performance.

---

## Objectives

* Build and train a Decision Tree Classifier.
* Predict student pass/fail outcomes.
* Evaluate model performance using classification metrics.
* Visualize the decision tree structure.
* Tune the model to reduce overfitting and improve generalization.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## Project Structure

```text
Decision-Tree-Student-Prediction/
│
├── decision_tree_classifier.py
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset contains information about students' study habits and previous exam performance.

| Study Hours | Previous Exam Score | Pass |
| ----------- | ------------------- | ---- |
| 1           | 30                  | 0    |
| 2           | 40                  | 0    |
| 3           | 45                  | 0    |
| 4           | 50                  | 0    |
| 5           | 60                  | 0    |
| 6           | 65                  | 1    |
| 7           | 70                  | 1    |
| 8           | 75                  | 1    |
| 9           | 80                  | 1    |
| 10          | 85                  | 1    |

### Features

* **StudyHours** – Number of hours spent studying.
* **PrevExamScore** – Score obtained in the previous examination.

### Target Variable

* **Pass**

  * 0 = Fail
  * 1 = Pass

---

## Model Workflow

### 1. Import Required Libraries

The project utilizes:

* NumPy for numerical operations
* Pandas for data manipulation
* Scikit-learn for machine learning
* Matplotlib for visualization

---

### 2. Load Dataset

The dataset is loaded into a Pandas DataFrame for preprocessing and analysis.

---

### 3. Split Data

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

using Scikit-learn's `train_test_split()` function.

---

### 4. Train the Decision Tree Model

Initialize and train the classifier:

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

The model learns decision rules based on study hours and previous exam scores to classify students as pass or fail.

---

### 5. Analyze Tree Structure

Important tree characteristics:

```python
print(model.get_depth())
print(model.get_n_leaves())
```

#### Tree Depth

Represents the maximum number of levels in the decision tree.

#### Number of Leaves

Represents the terminal nodes where final predictions are made.

---

### 6. Make Predictions

Generate predictions on unseen test data:

```python
y_pred = model.predict(X_test)
```

Predicted values:

* 0 → Fail
* 1 → Pass

---

## Model Evaluation

The model is evaluated using several classification metrics.

### Accuracy Score

Measures the percentage of correctly classified instances.

```python
accuracy_score(y_test, y_pred)
```

---

### Confusion Matrix

Provides detailed insights into prediction results.

```python
confusion_matrix(y_test, y_pred)
```

Example:

```text
[[4 0]
 [1 5]]
```

Where:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

help evaluate classification performance.

---

### Classification Report

Provides:

* Precision
* Recall
* F1-Score
* Support

```python
classification_report(y_test, y_pred)
```

Example:

```text
              precision    recall    f1-score

Fail             1.00      0.80      0.89
Pass             0.83      1.00      0.91

accuracy                           0.90
```

---

## Decision Tree Visualization

One of the major advantages of Decision Trees is interpretability.

The model structure can be visualized using:

```python
tree.plot_tree(
    model,
    feature_names=['StudyHours', 'PrevExamScore'],
    class_names=['Fail', 'Pass'],
    filled=True
)
```

The visualization shows:

* Decision rules used by the model
* Splitting conditions at each node
* Predicted class at leaf nodes
* Sample distribution across branches

This helps understand exactly how predictions are made.

---

## Model Tuning

Decision Trees can easily overfit training data if allowed to grow too deep.

To control complexity and improve generalization:

```python
model_tuned = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

Benefits of tuning:

* Reduces overfitting
* Improves performance on unseen data
* Creates a simpler and more interpretable model

Other tuning parameters include:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_leaf_nodes`

---

## Sample Output

```text
Training data: (8,2), (8,)
Testing data: (2,2), (2,)

Tree Depth: 3
Number of Leaves: 4

Predicted Outcomes:
[1 0]

Actual Outcomes:
[1 0]

Accuracy: 1.0

Confusion Matrix:
[[1 0]
 [0 1]]
```

*Note: Results may vary depending on train-test split and dataset size.*

---

## Key Learning Outcomes

After completing this project, you will understand:

* Decision Tree Fundamentals
* Binary Classification Techniques
* Data Splitting Strategies
* Model Training and Prediction
* Accuracy and Confusion Matrix Analysis
* Precision, Recall, and F1-Score
* Decision Tree Visualization
* Overfitting and Model Tuning

---

## Applications of Decision Trees

Decision Trees are widely used in:

* Student Performance Prediction
* Loan Approval Systems
* Customer Churn Prediction
* Medical Diagnosis
* Fraud Detection
* Credit Risk Assessment
* Employee Attrition Analysis
* Marketing and Customer Segmentation

---

## Future Enhancements

* Use larger real-world educational datasets.
* Add more predictive features such as attendance, assignments, and test scores.
* Compare Decision Trees with:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * XGBoost
* Perform hyperparameter optimization using GridSearchCV.
* Deploy the model using Flask, FastAPI, or Streamlit.

---

## Author

**Tejaswi**
Salesforce Developer | AI/ML Enthusiast | Aspiring Data Scientist

---

## License

This project is licensed under the MIT License.
