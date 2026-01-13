# === COPY AND PASTE THIS INTO A NEW CELL IN JUPYTER ===
# This is a self-contained cell that loads everything needed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Dataset(BankChurners)_CampusHiring_Dec2025(dataset).csv')
print(f"✓ Dataset loaded: {df.shape}")

# Define target column
target_col = 'Attrition_Flag'
print(f"✓ Target column: {target_col}")

# Prepare target variable
le = LabelEncoder()
y = le.fit_transform(df[target_col])
print(f"✓ Target encoded: {le.classes_}")

# Select features
feature_cols = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

X = df[feature_cols].fillna(0)
print(f"✓ Features prepared: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
)
model.fit(X_train_scaled, y_train)
print("✓ Model trained!")

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nKey Metrics:")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

# Save everything
print("\nSaving artifacts...")
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
joblib.dump(le, 'label_encoder.pkl')

import json
metrics_dict = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print("✓ All files saved!")
print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
