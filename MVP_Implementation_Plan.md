# MVP Implementation Plan - ML Intern Assessment
**Duration: 2 Hours 30 Minutes**

## Overview
This plan provides a streamlined approach to complete the ML-based problem-solving assessment within the time constraints. Focus is on meeting requirements efficiently without over-engineering.

**Problem Context:**
- **Dataset:** Credit card customer churn dataset with demographic, account, and behavioral data
- **Goal:** Build a churn prediction model with API + UI integration
- **Tech Stack:** Python, FastAPI/Flask, Streamlit
- **Focus:** Structured reasoning in ML coding

---

## Time Allocation (Total: 150 minutes)

| Task | Time | Priority |
|------|------|----------|
| Task 1: Data Understanding | 15 min | HIGH |
| Task 2: Feature Engineering | 20 min | HIGH |
| Task 3: Pattern Discovery | 15 min | MEDIUM |
| Task 4: Model Development | 30 min | HIGH |
| Task 5: API + Streamlit App | 35 min | HIGH |
| Task 6: Architecture Diagram | 10 min | MEDIUM |
| Documentation & Packaging | 20 min | HIGH |
| Buffer/Testing | 5 min | - |

---

## Task-by-Task Implementation Strategy

### **Task 1: Data Understanding (15 minutes)**

**Objective:** Quick exploratory analysis to understand data structure

**Steps:**
1. Load CSV using pandas
2. Generate basic statistics:
   - Shape, column names, dtypes
   - Missing value count per column
   - Numeric vs categorical breakdown
3. Create simple summary in notebook

**Code Approach:**
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dataset.csv')

# Summary stats
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes.value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")
```

**Output:**
- Type breakdown (numeric vs categorical)
- Missing value percentages
- Brief note on model difficulty

---

### **Task 2: Feature Engineering (20 minutes)**

**Objective:** Create meaningful features for churn prediction

**Steps:**
1. **Identify key features** for churn (transaction patterns, activity levels)
2. **Create conditional column:**
   ```python
   df['high_value'] = df['credit_limit'].apply(
       lambda x: 'High' if x > df['credit_limit'].mean() else 'Low'
   )
   ```
3. **Grouped aggregation:**
   ```python
   agg_summary = df.groupby('customer_segment').agg({
       'total_trans_amt': ['mean', 'count'],
       'avg_utilization_ratio': 'mean'
   })
   ```
4. **Compound filtering:**
   ```python
   churned_inactive = df[
       (df['attrition_flag'] == 'Attrited') & 
       (df['total_trans_ct'] < df['total_trans_ct'].median())
   ]
   ```

**Output:**
- New features added to dataframe
- Summary statistics table

---

### **Task 3: Pattern Discovery (15 minutes)**

**Objective:** Find 3 interesting patterns with visualizations

**Approach:**
Use matplotlib/seaborn for quick plots:

1. **Pattern 1:** Transaction count vs churn
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   plt.figure(figsize=(8, 5))
   sns.boxplot(data=df, x='attrition_flag', y='total_trans_ct')
   plt.title('Transaction Count by Churn Status')
   plt.savefig('pattern1.png')
   ```

2. **Pattern 2:** Credit utilization distribution
   ```python
   sns.histplot(data=df, x='avg_utilization_ratio', hue='attrition_flag', bins=30)
   ```

3. **Pattern 3:** Correlation heatmap
   ```python
   numeric_cols = df.select_dtypes(include=[np.number]).columns
   corr = df[numeric_cols].corr()
   sns.heatmap(corr, annot=False, cmap='coolwarm')
   ```

**Output:**
- 3 PNG files
- Brief explanations (2-3 sentences each)

---

### **Task 4: Model Development (30 minutes)**

**Objective:** Build and evaluate a churn prediction model

**Model Selection:** Logistic Regression or Random Forest (simple, interpretable)

**Steps:**
1. **Prepare data:**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder, StandardScaler
   
   # Encode target
   le = LabelEncoder()
   y = le.fit_transform(df['attrition_flag'])
   
   # Select features (numeric only for speed)
   feature_cols = ['customer_age', 'total_trans_amt', 'total_trans_ct', 
                   'avg_utilization_ratio', 'total_relationship_count']
   X = df[feature_cols].fillna(0)
   
   # Split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   # Scale
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Train model:**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, confusion_matrix
   
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train_scaled, y_train)
   ```

3. **Evaluate:**
   ```python
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   
   y_pred = model.predict(X_test_scaled)
   
   metrics = {
       'accuracy': accuracy_score(y_test, y_pred),
       'precision': precision_score(y_test, y_pred),
       'recall': recall_score(y_test, y_pred),
       'f1_score': f1_score(y_test, y_pred)
   }
   ```

4. **Save model:**
   ```python
   import joblib
   joblib.dump(model, 'churn_model.pkl')
   joblib.dump(scaler, 'scaler.pkl')
   ```

**Output:**
- Trained model file
- Performance metrics dictionary
- Confusion matrix image

---

### **Task 5: API + Streamlit App (35 minutes)**

**5A. FastAPI Backend (20 minutes)**

**File:** `api.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerData(BaseModel):
    customer_age: float
    total_trans_amt: float
    total_trans_ct: float
    avg_utilization_ratio: float
    total_relationship_count: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    features = np.array([[
        data.customer_age,
        data.total_trans_amt,
        data.total_trans_ct,
        data.avg_utilization_ratio,
        data.total_relationship_count
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    
    return {
        "prediction": "Churned" if prediction == 1 else "Retained",
        "churn_probability": float(probability[1])
    }

@app.get("/metrics")
def get_metrics():
    # Return pre-calculated metrics
    return {
        "accuracy": 0.85,  # Replace with actual
        "precision": 0.82,
        "recall": 0.78,
        "f1_score": 0.80
    }
```

**Run command:**
```bash
uvicorn api:app --reload --port 8000
```

---

**5B. Streamlit Frontend (15 minutes)**

**File:** `app.py`

```python
import streamlit as st
import requests
import pandas as pd

st.title("🏦 Credit Card Churn Prediction")

# Display dataset info
st.header("📊 Dataset Information")
df = pd.read_csv('dataset.csv')
st.write(f"**Dataset Shape:** {df.shape}")
st.write(f"**Columns:** {df.columns.tolist()}")

# Input form
st.header("🔮 Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
    total_trans_amt = st.number_input("Total Transaction Amount", min_value=0.0, value=5000.0)
    total_trans_ct = st.number_input("Total Transaction Count", min_value=0, value=50)

with col2:
    avg_utilization_ratio = st.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)
    total_relationship_count = st.number_input("Total Relationship Count", min_value=1, max_value=6, value=3)

if st.button("🚀 Predict Churn"):
    # Call API
    payload = {
        "customer_age": customer_age,
        "total_trans_amt": total_trans_amt,
        "total_trans_ct": total_trans_ct,
        "avg_utilization_ratio": avg_utilization_ratio,
        "total_relationship_count": total_relationship_count
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()
        
        st.success(f"**Prediction:** {result['prediction']}")
        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
    except Exception as e:
        st.error(f"Error: {e}")

# Display model metrics
st.header("📈 Model Performance")
try:
    metrics_response = requests.get("http://localhost:8000/metrics")
    metrics = metrics_response.json()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
except:
    st.warning("Could not fetch metrics")
```

**Run command:**
```bash
streamlit run app.py
```

---

### **Task 6: Cloud Architecture Diagram (10 minutes)**

**Tool:** Use PowerPoint/LibreOffice Draw

**Components to Include:**

1. **UI Layer**
   - Streamlit App (hosted on AWS EC2 / Azure App Service / GCP Cloud Run)

2. **API Layer**
   - FastAPI (containerized with Docker)
   - Load Balancer (AWS ALB / Azure Load Balancer / GCP Load Balancer)

3. **ETL/Data Ingestion**
   - AWS S3 / Azure Blob / GCP Cloud Storage (raw data)
   - AWS Glue / Azure Data Factory / GCP Dataflow (ETL pipeline)

4. **Storage**
   - Database: AWS RDS PostgreSQL / Azure SQL / Cloud SQL
   - Model Storage: S3 / Azure Blob / GCS

5. **Data Flow:**
   - User → Streamlit → API → Model → Response
   - ETL Pipeline → S3 → Database
   - Model Training → Save to S3

**Simple Layout:**
```
[User Browser] → [Load Balancer] → [Streamlit App Instances]
                                       ↓
                                  [FastAPI Service]
                                       ↓
                               [ML Model (S3/Blob)]
                                       ↑
                          [Training Pipeline (Glue/Dataflow)]
                                       ↑
                          [Raw Data Storage (S3/Blob)]
```

---

### **Documentation & Packaging (20 minutes)**

#### **Solution Approach PPT:**

**Slide 1: Cover Page**
- Full Name
- Email ID
- College Name
- Stream

**Slide 2: Data Quality**
- Dataset shape
- Missing values summary
- Data types breakdown

**Slide 3: Transformations**
- Feature engineering steps
- Conditional columns created
- Aggregations performed

**Slide 4: Patterns Discovered**
- 3 visualizations with explanations
- Key insights

**Slide 5: Model Selection & Evaluation**
- Model chosen (Random Forest)
- Performance metrics
- Confusion matrix

**Slide 6: Non-Obvious Insights**
- Transaction frequency is strongest predictor
- Utilization ratio shows non-linear relationship
- Customer age has minimal impact

**Slide 7: Architecture Diagram**
- Cloud deployment design

**Slide 8: Screenshots**
- Streamlit UI
- API response

---

## Final Folder Structure

```
submission.zip
├── Problem Statement/
│   └── problem.docx (copy of original)
├── Solution Approach/
│   └── solution.ppt (documentation)
└── Codebase/
    ├── notebook.ipynb (Tasks 1-4)
    ├── api.py (FastAPI)
    ├── app.py (Streamlit)
    ├── churn_model.pkl (trained model)
    ├── scaler.pkl (scaler)
    ├── requirements.txt (dependencies)
    ├── pattern1.png
    ├── pattern2.png
    ├── pattern3.png
    └── README.md
```

---

## Dependencies (requirements.txt)

```txt
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
```

---

## Quick Start Commands

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebook:**
   - Execute cells sequentially

3. **Start API:**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

4. **Start Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## Testing Checklist

- [ ] Data loads successfully
- [ ] Feature engineering produces expected columns
- [ ] Model trains without errors
- [ ] API endpoint responds correctly
- [ ] Streamlit app displays data and accepts input
- [ ] Predictions are returned
- [ ] All files are in correct folders
- [ ] ZIP file created with correct naming

---

## Tips for Time Management

1. **Don't overthink:** Use simple, proven approaches
2. **Test incrementally:** Run code after each task
3. **Use templates:** Copy-paste working code patterns
4. **Focus on deliverables:** Complete required items first
5. **Document as you go:** Add brief comments
6. **Keep visualizations simple:** Basic plots are sufficient

---

## Common Pitfalls to Avoid

❌ Over-engineering feature selection
❌ Training complex deep learning models
❌ Creating elaborate visualizations
❌ Writing extensive documentation
❌ Spending too long on architecture diagram

✅ Use RandomForest or Logistic Regression
✅ Basic matplotlib/seaborn plots
✅ Concise bullet points in PPT
✅ Simple architecture diagram
✅ Focus on functionality over perfection

---

## Emergency Time Savers

If running short on time:

1. **Skip Task 3:** Patterns can be minimal
2. **Simplify API:** Just one endpoint
3. **Basic Streamlit:** Just input and output, no fancy UI
4. **Hand-drawn diagram:** Acceptable if digital takes too long
5. **Minimal documentation:** Brief bullet points sufficient

---

## Success Criteria

**Must Have:**
- ✅ Working model with metrics
- ✅ API that returns predictions
- ✅ Streamlit app that calls API
- ✅ Architecture diagram
- ✅ PPT with all sections

**Nice to Have:**
- ⭐ Feature importance visualization
- ⭐ Interactive Streamlit elements
- ⭐ Detailed insights
- ⭐ Professional architecture diagram

---

## Final Submission Steps

1. Create folder structure
2. Copy all files to appropriate folders
3. Verify all deliverables present
4. Create ZIP: `yourname_batchnumber.zip`
5. Upload via form or email

---

**Good luck! Remember: Completion > Perfection**
