"""
Standalone Model Training Script
Run this if Jupyter notebook has issues
Usage: python3 train_model.py
"""

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
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" " * 15 + "CREDIT CARD CHURN PREDICTION")
print(" " * 20 + "Model Training Script")
print("="*70)

# Load dataset
print("\n[1/8] Loading dataset...")
try:
    df = pd.read_csv('Dataset(BankChurners)_CampusHiring_Dec2025(dataset).csv')
    print(f"      ✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("      ✗ ERROR: Dataset file not found!")
    print("      Make sure you're in the Codebase directory")
    exit(1)

# Define target column
target_col = 'Attrition_Flag'
print(f"\n[2/8] Preparing target variable...")
print(f"      Target column: {target_col}")
print(f"      Classes: {df[target_col].unique()}")

le = LabelEncoder()
y = le.fit_transform(df[target_col])
print(f"      ✓ Encoded: {dict(zip(le.classes_, [0, 1]))}")
print(f"      Distribution: {pd.Series(y).value_counts().to_dict()}")

# Select features
print(f"\n[3/8] Selecting features...")
feature_cols = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

X = df[feature_cols].fillna(0)
print(f"      ✓ Selected {len(feature_cols)} features")
print(f"      Feature matrix shape: {X.shape}")

# Split data
print(f"\n[4/8] Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      ✓ Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"      ✓ Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale features
print(f"\n[5/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"      ✓ Features scaled using StandardScaler")

# Train model
print(f"\n[6/8] Training Random Forest model...")
print(f"      Parameters: n_estimators=100, max_depth=10")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=0
)

model.fit(X_train_scaled, y_train)
print(f"      ✓ Model training completed!")

# Predict
print(f"\n[7/8] Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*70)
print(" " * 25 + "MODEL PERFORMANCE")
print("="*70)
print(f"\n  Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision:  {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall:     {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score:   {f1:.4f}  ({f1*100:.2f}%)")

print("\n" + "-"*70)
print("Classification Report:")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("-"*70)
print("Confusion Matrix:")
print("-"*70)
print(f"                    Predicted")
print(f"                    {le.classes_[0]:<20} {le.classes_[1]}")
print(f"Actual {le.classes_[0]:<20}  {cm[0][0]:<20} {cm[0][1]}")
print(f"       {le.classes_[1]:<20}  {cm[1][0]:<20} {cm[1][1]}")
print("="*70)

# Save everything
print(f"\n[8/8] Saving model artifacts...")

joblib.dump(model, 'churn_model.pkl')
print(f"      ✓ churn_model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print(f"      ✓ scaler.pkl")

joblib.dump(feature_cols, 'feature_cols.pkl')
print(f"      ✓ feature_cols.pkl")

joblib.dump(le, 'label_encoder.pkl')
print(f"      ✓ label_encoder.pkl")

metrics_dict = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"      ✓ model_metrics.json")

# Create confusion matrix visualization
print(f"\n      Creating visualizations...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Churn Prediction Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print(f"      ✓ confusion_matrix.png")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
print(f"      ✓ feature_importance.png")

print("\n" + "="*70)
print(" " * 20 + "✅ MODEL TRAINING COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  • churn_model.pkl")
print("  • scaler.pkl")
print("  • feature_cols.pkl")
print("  • label_encoder.pkl")
print("  • model_metrics.json")
print("  • confusion_matrix.png")
print("  • feature_importance.png")
print("\nNext steps:")
print("  1. Start API: uvicorn api:app --reload --port 8000")
print("  2. Start UI:  streamlit run app.py")
print("="*70)
