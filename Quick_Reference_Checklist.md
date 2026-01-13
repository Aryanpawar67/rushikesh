# Quick Reference Checklist - ML Assessment

## 📋 Pre-Assessment Setup (5 minutes)

### Environment Setup
```bash
# Create project directory
mkdir ml_assessment
cd ml_assessment

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies at once
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn pydantic streamlit joblib requests
```

### Folder Structure Setup
```bash
mkdir -p "Problem Statement"
mkdir -p "Solution Approach"
mkdir -p Codebase
```

---

## ✅ Task Completion Checklist

### **Task 1: Data Understanding** ⏱️ 15 min

- [ ] Load dataset using pandas
- [ ] Print dataset shape
- [ ] Count numeric vs categorical columns
- [ ] Calculate missing values per column
- [ ] Print basic statistics
- [ ] Note estimated model difficulty

**Quick Code:**
```python
import pandas as pd
import numpy as np

# Load
df = pd.read_csv('your_dataset.csv')

# Summary
print(f"Shape: {df.shape}")
print(f"\nColumn Types:\n{df.dtypes.value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nNumerics: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Categoricals: {df.select_dtypes(include=['object']).shape[1]}")
```

---

### **Task 2: Feature Engineering** ⏱️ 20 min

- [ ] Identify relevant features for churn
- [ ] Create conditional column (High/Low based on threshold)
- [ ] Create grouped aggregation (mean, count)
- [ ] Apply compound boolean filter
- [ ] Document transformations

**Example Code:**
```python
# 1. Conditional column
df['transaction_category'] = df['total_trans_amt'].apply(
    lambda x: 'High' if x > df['total_trans_amt'].mean() else 'Low'
)

# 2. Grouped aggregation
agg_summary = df.groupby('attrition_flag').agg({
    'total_trans_amt': ['mean', 'count'],
    'total_trans_ct': 'mean',
    'avg_utilization_ratio': 'mean'
}).round(2)
print(agg_summary)

# 3. Compound filter
inactive_high_risk = df[
    (df['total_trans_ct'] < df['total_trans_ct'].quantile(0.25)) & 
    (df['avg_utilization_ratio'] > 0.5) &
    (df['months_inactive_12_mon'] > 2)
]
print(f"High risk customers: {len(inactive_high_risk)}")
```

---

### **Task 3: Pattern Discovery** ⏱️ 15 min

- [ ] Create 3 visualizations
- [ ] Save plots as PNG files
- [ ] Write 2-3 sentence explanations for each
- [ ] Note root causes

**Quick Plots:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Pattern 1: Transaction behavior
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='attrition_flag', y='total_trans_ct')
plt.title('Transaction Count by Churn Status')
plt.savefig('Codebase/pattern1_transaction_behavior.png', dpi=100, bbox_inches='tight')
plt.close()

# Pattern 2: Utilization vs Churn
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='avg_utilization_ratio', hue='attrition_flag', bins=30)
plt.title('Credit Utilization Distribution by Churn')
plt.savefig('Codebase/pattern2_utilization.png', dpi=100, bbox_inches='tight')
plt.close()

# Pattern 3: Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Top 10
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.savefig('Codebase/pattern3_correlation.png', dpi=100, bbox_inches='tight')
plt.close()
```

**Insights Template:**
- Pattern 1: Churned customers show 40% fewer transactions on average. Root cause: Declining engagement.
- Pattern 2: High utilization (>0.8) correlates with churn. Root cause: Financial stress signals.
- Pattern 3: Transaction amount and count are highly correlated (0.8). Root cause: Customer spending patterns.

---

### **Task 4: Model Development** ⏱️ 30 min

- [ ] Prepare features (handle missing, encode)
- [ ] Split train/test
- [ ] Train model (RandomForest recommended)
- [ ] Evaluate (accuracy, precision, recall, F1)
- [ ] Save model and scaler
- [ ] Document model selection rationale

**Complete Pipeline:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Prepare target
le = LabelEncoder()
y = le.fit_transform(df['attrition_flag'])  # 0=Existing, 1=Attrited

# Select features (numeric only for MVP)
feature_cols = [
    'customer_age', 'dependent_count', 'months_on_book',
    'total_relationship_count', 'months_inactive_12_mon',
    'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
    'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
    'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
]

X = df[feature_cols].fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Existing', 'Attrited']))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('Codebase/confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Save
joblib.dump(model, 'Codebase/churn_model.pkl')
joblib.dump(scaler, 'Codebase/scaler.pkl')
joblib.dump(feature_cols, 'Codebase/feature_cols.pkl')

print("\nModel saved successfully!")
```

---

### **Task 5: API + Streamlit** ⏱️ 35 min

#### **5A. FastAPI (20 min)**

- [ ] Create api.py
- [ ] Implement /predict endpoint
- [ ] Implement /metrics endpoint (optional)
- [ ] Test API with curl/Postman

**File: Codebase/api.py**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="Churn Prediction API")

# Load artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

class CustomerInput(BaseModel):
    customer_age: float
    dependent_count: int
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "active"}

@app.post("/predict")
def predict(customer: CustomerInput):
    # Prepare features
    features = np.array([[
        customer.customer_age,
        customer.dependent_count,
        customer.months_on_book,
        customer.total_relationship_count,
        customer.months_inactive_12_mon,
        customer.contacts_count_12_mon,
        customer.credit_limit,
        customer.total_revolving_bal,
        customer.avg_open_to_buy,
        customer.total_amt_chng_q4_q1,
        customer.total_trans_amt,
        customer.total_trans_ct,
        customer.total_ct_chng_q4_q1,
        customer.avg_utilization_ratio
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        "prediction": "Attrited Customer" if prediction == 1 else "Existing Customer",
        "churn_probability": round(float(probability[1]), 4),
        "retention_probability": round(float(probability[0]), 4)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
```

**Test API:**
```bash
# Terminal 1: Start API
cd Codebase
uvicorn api:app --reload --port 8000

# Terminal 2: Test
curl http://localhost:8000/
```

---

#### **5B. Streamlit App (15 min)**

- [ ] Create app.py
- [ ] Display dataset info
- [ ] Create input widgets
- [ ] Call API on button click
- [ ] Display results
- [ ] Take screenshots

**File: Codebase/app.py**
```python
import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="Churn Prediction", page_icon="🏦", layout="wide")

st.title("🏦 Credit Card Churn Prediction System")
st.markdown("---")

# Load dataset for display
@st.cache_data
def load_data():
    return pd.read_csv('../your_dataset.csv')  # Adjust path

df = load_data()

# Sidebar - Dataset Info
with st.sidebar:
    st.header("📊 Dataset Information")
    st.write(f"**Total Records:** {df.shape[0]:,}")
    st.write(f"**Total Features:** {df.shape[1]}")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Columns:**")
    st.text("\n".join(df.columns.tolist()[:10]))  # Show first 10

# Main area - Prediction
st.header("🔮 Customer Churn Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    customer_age = st.number_input("Customer Age", 18, 100, 45)
    dependent_count = st.number_input("Dependents", 0, 10, 2)
    months_on_book = st.number_input("Months on Book", 0, 100, 36)

with col2:
    st.subheader("Account Info")
    total_relationship_count = st.number_input("Relationship Count", 1, 6, 3)
    months_inactive = st.number_input("Months Inactive (12m)", 0, 12, 1)
    contacts_count = st.number_input("Contacts (12m)", 0, 10, 2)
    credit_limit = st.number_input("Credit Limit", 1000, 50000, 10000)

with col3:
    st.subheader("Financial Metrics")
    total_revolving_bal = st.number_input("Revolving Balance", 0, 10000, 1000)
    avg_open_to_buy = st.number_input("Avg Open to Buy", 0, 50000, 5000)
    total_amt_chng = st.number_input("Amt Change Q4-Q1", 0.0, 5.0, 0.8, 0.1)
    total_trans_amt = st.number_input("Total Trans Amount", 0, 20000, 3500)
    total_trans_ct = st.number_input("Total Trans Count", 0, 150, 50)
    total_ct_chng = st.number_input("Count Change Q4-Q1", 0.0, 5.0, 0.7, 0.1)
    avg_utilization = st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.3, 0.01)

st.markdown("---")

if st.button("🚀 Predict Churn Risk", use_container_width=True):
    # Prepare payload
    payload = {
        "customer_age": customer_age,
        "dependent_count": dependent_count,
        "months_on_book": months_on_book,
        "total_relationship_count": total_relationship_count,
        "months_inactive_12_mon": months_inactive,
        "contacts_count_12_mon": contacts_count,
        "credit_limit": credit_limit,
        "total_revolving_bal": total_revolving_bal,
        "avg_open_to_buy": avg_open_to_buy,
        "total_amt_chng_q4_q1": total_amt_chng,
        "total_trans_amt": total_trans_amt,
        "total_trans_ct": total_trans_ct,
        "total_ct_chng_q4_q1": total_ct_chng,
        "avg_utilization_ratio": avg_utilization
    }
    
    try:
        with st.spinner("Analyzing customer data..."):
            response = requests.post("http://localhost:8000/predict", json=payload)
            result = response.json()
        
        st.success("✅ Prediction Complete!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", result['prediction'])
        with col2:
            churn_prob = result['churn_probability']
            st.metric("Churn Probability", f"{churn_prob:.2%}", 
                     delta=f"{(churn_prob - 0.5):.2%}")
        with col3:
            st.metric("Retention Probability", 
                     f"{result['retention_probability']:.2%}")
        
        # Risk level indicator
        if churn_prob > 0.7:
            st.error("⚠️ **HIGH RISK** - Customer likely to churn")
        elif churn_prob > 0.4:
            st.warning("⚡ **MEDIUM RISK** - Monitor customer activity")
        else:
            st.success("✅ **LOW RISK** - Customer likely to stay")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure the API is running at http://localhost:8000")

# Model Performance Section
st.markdown("---")
st.header("📈 Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

# Replace with actual metrics from your model
col1.metric("Accuracy", "87.5%")
col2.metric("Precision", "83.2%")
col3.metric("Recall", "79.8%")
col4.metric("F1 Score", "81.4%")

st.info("💡 **Tip:** Low transaction count and high utilization ratio are strong churn indicators.")
```

**Run Streamlit:**
```bash
# In Codebase directory
streamlit run app.py
```

**Take Screenshots:**
1. Full UI with inputs
2. Prediction results
3. Dataset info section

---

### **Task 6: Architecture Diagram** ⏱️ 10 min

- [ ] Open PowerPoint/LibreOffice
- [ ] Create simple flow diagram
- [ ] Include all 4 layers
- [ ] Show data flow arrows
- [ ] Label cloud services
- [ ] Export as image

**Components to Include:**

```
┌─────────────────────────────────────────────────┐
│              USER LAYER                          │
│  Web Browser → Streamlit UI (EC2/App Service)   │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│           API LAYER                              │
│  FastAPI (Docker Container on ECS/AKS)          │
│  + Load Balancer (ALB/Azure LB)                 │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│       MODEL & STORAGE LAYER                      │
│  S3/Blob Storage (Model Artifacts)              │
│  RDS/SQL Database (Training Data)               │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│         ETL/INGESTION LAYER                      │
│  AWS Glue / Azure Data Factory                  │
│  S3/Blob (Raw Data Storage)                     │
└─────────────────────────────────────────────────┘
```

---

### **Documentation** ⏱️ 20 min

- [ ] Create PPT with 8 slides
- [ ] Add personal details on cover
- [ ] Include all required sections
- [ ] Add screenshots
- [ ] Copy original problem statement

**PPT Structure:**

1. **Cover Page**
   - Full Name, Email, College, Stream

2. **Data Quality Assessment**
   - Shape, missing values, data types
   - Quality issues identified

3. **Feature Engineering**
   - Transformations applied
   - New features created
   - Aggregations performed

4. **Pattern Discovery**
   - 3 visualizations with insights
   - Root cause analysis

5. **Model Selection & Evaluation**
   - Model: Random Forest
   - Metrics table
   - Confusion matrix

6. **Non-Obvious Insights**
   - Transaction frequency strongest predictor
   - Utilization shows non-linear pattern
   - Seasonal trends in churn

7. **Cloud Architecture**
   - Deployment diagram
   - Technology choices

8. **UI Screenshots**
   - Streamlit interface
   - Prediction results

---

## 📦 Final Packaging

### Create Submission Package

```bash
# 1. Create final structure
mkdir submission
cd submission
mkdir "Problem Statement" "Solution Approach" Codebase

# 2. Copy files
cp original_problem.docx "Problem Statement/problem.docx"
cp solution.pptx "Solution Approach/solution.ppt"
cp -r ../Codebase/* Codebase/

# 3. Create requirements.txt
cat > Codebase/requirements.txt << EOF
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0
streamlit==1.27.0
joblib==1.3.2
requests==2.31.0
EOF

# 4. Create README
cat > Codebase/README.md << EOF
# Churn Prediction System

## Setup
pip install -r requirements.txt

## Run API
uvicorn api:app --reload --port 8000

## Run UI
streamlit run app.py

## Files
- notebook.ipynb: EDA and model training
- api.py: FastAPI backend
- app.py: Streamlit frontend
- *.pkl: Model artifacts
- *.png: Visualizations
EOF

# 5. Create ZIP
cd ..
zip -r yourname_batchnumber.zip submission/
```

---

## ⚡ Quick Commands Reference

```bash
# Install everything
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn pydantic streamlit joblib requests

# Start Jupyter
jupyter notebook

# Start API
uvicorn api:app --reload --port 8000

# Start Streamlit
streamlit run app.py

# Test API
curl http://localhost:8000/health
```

---

## 🎯 Final Checklist Before Submission

- [ ] All 6 tasks completed
- [ ] Notebook runs without errors
- [ ] API responds correctly
- [ ] Streamlit app works
- [ ] 3 visualization PNGs saved
- [ ] Model and scaler saved
- [ ] PPT has all 8 slides
- [ ] Personal details on cover
- [ ] Screenshots included
- [ ] Architecture diagram present
- [ ] Folder structure correct
- [ ] ZIP file created
- [ ] File named correctly: `yourname_batchnumber.zip`

---

## 🚨 Emergency Shortcuts

**If you're running out of time:**

1. **Skip fancy visualizations** - Basic plots are fine
2. **Simplify Streamlit** - Just inputs and prediction output
3. **Copy-paste model code** - Don't optimize hyperparameters
4. **Hand-draw architecture** - Take a photo if needed
5. **Minimal PPT** - Bullet points are acceptable

---

## 📊 Expected Metrics Ranges

For credit card churn:
- Accuracy: 80-90%
- Precision: 75-85%
- Recall: 70-80%
- F1: 75-82%

Don't worry if slightly lower - document why!

---

**Remember: Working > Perfect! Submit on time! 🎯**
