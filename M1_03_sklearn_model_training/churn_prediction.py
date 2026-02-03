"""
Telco Customer Churn Prediction with KNN
Author: April Atkinson
Description: Predict customer churn (Yes/No) using a KNN classifier
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading churn dataset...")
df = pd.read_csv(CSV_FILE)

print(f"\nDataset loaded! Shape: {df.shape}")
print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
missing = df.isna().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values found (NaN).")

print("\nTarget distribution (Churn):")
print(df["Churn"].value_counts(dropna=False))
print("\nTarget distribution (%):")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# Step 2: Basic Preprocessing
# --------------------------------------------------

print("\n" + "=" * 50)
print("PREPROCESSING")
print("=" * 50)

# 1. Convert target variable Churn to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nChurn after mapping (0 = No, 1 = Yes):")
print(df["Churn"].value_counts())

# 2. Fix TotalCharges: convert to numeric
# Some values are blank strings -> coerce them to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges converted to numeric.")
print("Missing values in TotalCharges:", df["TotalCharges"].isna().sum())

# Drop rows with missing TotalCharges (very few)
df = df.dropna(subset=["TotalCharges"])

print("Dataset shape after dropping missing TotalCharges:", df.shape)

# 3. Drop customerID (identifier, not useful for modeling)
df = df.drop(columns=["customerID"])

print("\nDropped customerID column.")
print("Remaining columns:", list(df.columns))

# --------------------------------------------------
# Step 3: Encode Categorical Variables
# --------------------------------------------------

print("\n" + "=" * 50)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 50)

# Identify categorical columns (object type)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:")
print(categorical_cols)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataset shape after encoding:", df_encoded.shape)

# Separate features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

print("\nFinal feature matrix shape:", X.shape)
print("Final target vector shape:", y.shape)

# --------------------------------------------------
# Step 4: Train / Test Split
# --------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("TRAIN / TEST SPLIT (CHURN)")
print("=" * 50)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size:     {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

print("\nTraining set churn distribution:")
print(y_train.value_counts())
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTest set churn distribution:")
print(y_test.value_counts())
print((y_test.value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# BONUS: FEATURE SCALING (IMPORTANT FOR KNN)
# --------------------------------------------------

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling applied using StandardScaler.")

# --------------------------------------------------
# Step 5: Train Baseline KNN Model (Churn)
# --------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Create baseline KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

print("\nKNN model trained (baseline, unscaled).")

# Make predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Evaluate
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("\n" + "=" * 50)
print("BASELINE KNN PERFORMANCE (CHURN)")
print("=" * 50)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision:   {test_precision:.4f}")
print(f"Test Recall:      {test_recall:.4f}")

print("\nCONFUSION MATRIX")
print(test_confusion)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_test_pred))

# --------------------------------------------------
# Step 6: Experiment with Different K Values (Churn)
# --------------------------------------------------

print("\n" + "=" * 50)
print("EXPERIMENTING WITH DIFFERENT K VALUES (CHURN)")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11, 15]
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

results_df = pd.DataFrame(results)

best_k = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
best_acc = results_df["Accuracy"].max()

print(f"\nBest K by accuracy: {best_k} (Accuracy: {best_acc:.4f})")

print("\nFeature scaling applied.")

# --------------------------------------------------
# BONUS: SCALED KNN MODEL (CHURN)
# --------------------------------------------------

print("\n" + "=" * 50)
print("SCALED KNN PERFORMANCE (CHURN)")
print("=" * 50)

knn_scaled = KNeighborsClassifier(n_neighbors=best_k)
knn_scaled.fit(X_train_scaled, y_train)

y_test_pred_scaled = knn_scaled.predict(X_test_scaled)

scaled_accuracy = accuracy_score(y_test, y_test_pred_scaled)
scaled_precision = precision_score(y_test, y_test_pred_scaled)
scaled_recall = recall_score(y_test, y_test_pred_scaled)
scaled_confusion = confusion_matrix(y_test, y_test_pred_scaled)

print(f"Accuracy:  {scaled_accuracy:.4f}")
print(f"Precision: {scaled_precision:.4f}")
print(f"Recall:    {scaled_recall:.4f}")

print("\nCONFUSION MATRIX (SCALED)")
print(scaled_confusion)

print("\nCLASSIFICATION REPORT (SCALED)")
print(classification_report(y_test, y_test_pred_scaled))
