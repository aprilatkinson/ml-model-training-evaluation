
---

## 🧠 Lab Objectives

- Load and explore datasets using `sklearn`
- Split data into training and testing sets
- Train and evaluate a **K-Nearest Neighbors (KNN)** classifier
- Use evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
- Apply the ML workflow to a real-world business problem (customer churn)
- Interpret results and provide business recommendations
- Implement **bonus feature scaling** for distance-based models

---

## 🔹 Part 1: Breast Cancer Prediction (Guided)

**Dataset**
- Source: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numeric features
- Target: Malignant vs Benign

**Steps**
- Data exploration and visualization
- Train/test split with stratification
- KNN model training
- Model evaluation
- Experimentation with different K values
- **Bonus:** Feature scaling with `StandardScaler`

**Results**
- Accuracy: ~91% (baseline)
- Accuracy improved to ~97% after scaling
- Optimal K selected based on performance comparison

---

## 🔹 Part 2: Telco Customer Churn Prediction (Unguided)

**Dataset**
- Telco Customer Churn Dataset (CSV)
- Customers: ~7,000
- Target: Churn (Yes / No)
- Class imbalance: ~26% churn

**Steps**
- Data cleaning and preprocessing
- Encoding categorical variables using one-hot encoding
- Train/test split with stratification
- Baseline KNN model
- K optimization
- **Bonus:** Feature scaling with `StandardScaler`
- Scaled KNN model evaluation

**Key Findings**
- Feature scaling significantly improved **recall for churned customers**
- Scaled model identified ~55% of churners vs ~40% before scaling
- Trade-off between precision and recall aligns with business needs

---

## 📊 Evaluation Metrics Used

- **Accuracy** – Overall correctness
- **Precision** – Correctness of churn predictions
- **Recall** – Ability to catch actual churners (most important for business)
- **Confusion Matrix**
- **Classification Report**

---

## 🏢 Business Impact

The churn model can be used as an **early-warning system** to:
- Identify high-risk customers
- Trigger proactive retention campaigns
- Reduce customer attrition

Detailed analysis and recommendations are available in **`REPORT.md`**.

---

## ⚠️ Limitations & Future Work

- KNN does not scale well to very large datasets
- Sensitive to feature scaling and noise
- No direct feature importance

**Future Improvements**
- Logistic Regression for interpretability
- Random Forest for feature importance
- Class imbalance handling (SMOTE, class weights)
- ROC-AUC and cross-validation

---

## 🛠️ Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:
```bash
pip install pandas numpy scikit-learn matplotlib
