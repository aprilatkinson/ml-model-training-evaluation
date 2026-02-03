"""
Breast Cancer Prediction with KNN
Author: April Atkinson
Description: Predict whether a breast tumor is malignant or benign using KNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Step 1: Load the breast cancer dataset
# --------------------------------------------------

print("Loading breast cancer dataset...")
cancer_data = load_breast_cancer()

# The dataset is a Bunch object
print(f"\nDataset type: {type(cancer_data)}")
print(f"Number of samples: {len(cancer_data.data)}")
print(f"Number of features: {len(cancer_data.feature_names)}")
print(f"Target classes: {cancer_data.target_names}")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df["target"] = cancer_data.target

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget distribution:")
print(df["target"].value_counts())
print(f"Malignant (1): {(df['target'] == 1).sum()}")
print(f"Benign (0): {(df['target'] == 0).sum()}")

# --------------------------------------------------
# Step 2: Data Exploration
# --------------------------------------------------

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

print("\n" + "=" * 50)
print("FEATURE DISTRIBUTIONS")
print("=" * 50)

# Select representative features to visualize
key_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    axes[idx].hist(
        df[df["target"] == 0][feature],
        bins=30,
        alpha=0.5,
        label="Benign"
    )
    axes[idx].hist(
        df[df["target"] == 1][feature],
        bins=30,
        alpha=0.5,
        label="Malignant"
    )
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Frequency")
    axes[idx].legend()

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150)
print("Saved visualization to 'feature_distributions.png'")
plt.show()

# --------------------------------------------------
# Step 3: Train / Test Split
# --------------------------------------------------

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

print("\nTraining set target distribution:")
print(y_train.value_counts())
print(f"  Benign (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

print("\nTest set target distribution:")
print(y_test.value_counts())
print(f"  Benign (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Malignant (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# --------------------------------------------------
# Step 4: Train KNN Model
# --------------------------------------------------

# Create KNN classifier
# n_neighbors=5 means the model looks at the 5 nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

print("\nKNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")

# Make predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f"\nTraining predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")

# --------------------------------------------------
# Step 5: Model Evaluation
# --------------------------------------------------

# Calculate evaluation metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\nTest Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")

print("\nCONFUSION MATRIX")
print("                Predicted")
print("              Benign  Malignant")
print(f"Actual Benign    {test_confusion[0,0]:4d}      {test_confusion[0,1]:4d}")
print(f"      Malignant  {test_confusion[1,0]:4d}      {test_confusion[1,1]:4d}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=cancer_data.target_names
))

# --------------------------------------------------
# Step 6: Experiment with Different K Values
# --------------------------------------------------

print("\n" + "=" * 50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)

    y_pred_temp = knn_temp.predict(X_test)

    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec
    })

    print(
        f"K={k:2d}: "
        f"Accuracy={acc:.4f}, "
        f"Precision={prec:.4f}, "
        f"Recall={rec:.4f}"
    )

# Convert to DataFrame for easier comparison
results_df = pd.DataFrame(results)

best_k = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
best_acc = results_df["Accuracy"].max()

print(f"\nBest K value: {best_k} (Accuracy: {best_acc:.4f})")

# --------------------------------------------------
# BONUS: Feature Scaling + KNN (run AFTER best_k is known)
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=int(best_k))
knn_scaled.fit(X_train_scaled, y_train)

y_test_pred_scaled = knn_scaled.predict(X_test_scaled)

scaled_accuracy = accuracy_score(y_test, y_test_pred_scaled)
scaled_precision = precision_score(y_test, y_test_pred_scaled)
scaled_recall = recall_score(y_test, y_test_pred_scaled)

print("\n" + "=" * 50)
print("SCALED KNN PERFORMANCE (using best_k)")
print("=" * 50)
print(f"best_k:    {best_k}")
print(f"Accuracy:  {scaled_accuracy:.4f}")
print(f"Precision: {scaled_precision:.4f}")
print(f"Recall:    {scaled_recall:.4f}")
