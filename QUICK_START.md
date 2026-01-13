# 🚀 Quick Start Guide

**All code is ready! Follow these steps to complete the assessment.**

---

## ⚡ Fast Track (5 Steps)

### 1️⃣ Install Dependencies (5 min)
```bash
cd "Codebase"
pip install -r requirements.txt
```

### 2️⃣ Add Your Dataset (2 min)
- Place your dataset CSV in the `Codebase/` folder
- Update filename in:
  - `notebook.ipynb` (line ~50): `df = pd.read_csv('YOUR_FILE.csv')`
  - `app.py` (line ~328): Update path in `load_data()` function

### 3️⃣ Train the Model (15-20 min)
```bash
jupyter notebook
```
- Open `notebook.ipynb`
- Run all cells (Cell → Run All)
- Wait for completion
- Verify files created:
  - ✅ churn_model.pkl
  - ✅ scaler.pkl
  - ✅ feature_cols.pkl
  - ✅ 3+ PNG files (patterns)

### 4️⃣ Start API & UI (5 min)

**Terminal 1 - API:**
```bash
cd Codebase
uvicorn api:app --reload --port 8000
```

**Terminal 2 - Streamlit:**
```bash
cd Codebase
streamlit run app.py
```

### 5️⃣ Test Everything (5 min)

**Terminal 3 - Run Tests:**
```bash
cd Codebase
python test_api.py
```

**Or manually:**
1. Visit http://localhost:8501 (Streamlit)
2. Check "API Status: Online" in sidebar
3. Enter customer data
4. Click "Predict Churn Risk"
5. Verify results appear

---

## 📊 Create PPT (20-30 min)

**File**: `Solution Approach/solution.ppt`

### Required Slides (8 total):

1. **Cover**: Name, Email, College, Stream

2. **Data Quality**:
   - Copy statistics from notebook output
   - Dataset shape, missing values

3. **Feature Engineering**:
   - List features created
   - Show aggregation table

4. **Patterns** (3 visualizations):
   - Insert pattern1_transaction_behavior.png
   - Insert pattern2_utilization.png
   - Insert pattern3_correlation.png
   - Add insights for each

5. **Model Performance**:
   - Accuracy, Precision, Recall, F1
   - Insert confusion_matrix.png

6. **Insights**:
   - Transaction count strongest predictor
   - Utilization ratio patterns
   - Your discoveries

7. **Architecture**:
   - Draw: User → Streamlit → FastAPI → Model
   - Include cloud services (AWS/Azure/GCP)

8. **Screenshots**:
   - Streamlit UI
   - Prediction results

---

## 📦 Create Submission (10 min)

```bash
# From project root directory
cd ..

# Create submission folder
mkdir submission
cd submission
mkdir "Problem Statement" "Solution Approach" Codebase

# Copy files
cp ../your_problem_statement.pdf "Problem Statement/"
cp ../your_presentation.ppt "Solution Approach/"

# Copy codebase
cp ../Codebase/notebook.ipynb Codebase/
cp ../Codebase/api.py Codebase/
cp ../Codebase/app.py Codebase/
cp ../Codebase/requirements.txt Codebase/
cp ../Codebase/README.md Codebase/
cp ../Codebase/*.pkl Codebase/
cp ../Codebase/*.png Codebase/
cp ../Codebase/*.json Codebase/

# Create ZIP
cd ..
zip -r yourname_batchnumber.zip submission/
```

---

## ✅ Submission Checklist

Before submitting, verify:

- [ ] All dependencies installed
- [ ] Dataset loaded successfully
- [ ] Notebook executed completely
- [ ] 5 PNG files generated
- [ ] 4 PKL files created
- [ ] API starts without errors
- [ ] Streamlit connects to API
- [ ] Can make predictions
- [ ] PPT has all 8 slides
- [ ] Personal details on cover slide
- [ ] Screenshots included in PPT
- [ ] Architecture diagram present
- [ ] All folders in submission package
- [ ] ZIP file created correctly

---

## 🎯 Expected Timeline

| Task | Time |
|------|------|
| Install deps | 5 min |
| Add dataset | 2 min |
| Train model | 20 min |
| Start servers | 5 min |
| Test system | 5 min |
| Create PPT | 30 min |
| Package & ZIP | 10 min |
| **TOTAL** | **~77 min** |

**Buffer**: 30-40 minutes for unexpected issues

**Total Time**: ~2 hours ✅ (Within 2.5 hour limit)

---

## 🔥 Pro Tips

1. **Start Early**: Don't wait until the last minute
2. **Test First**: Run test_api.py before creating PPT
3. **Screenshots**: Take them while everything is working
4. **Backup**: Save your work frequently
5. **Double-Check**: Verify ZIP contains all files before submitting

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements.txt` |
| Port in use | Kill process or change port |
| API offline | Check terminal for errors |
| Dataset error | Update CSV filename in code |
| Model not found | Run notebook first |

---

## 📞 Need Help?

Check these files:
1. **README.md** - Detailed documentation
2. **IMPLEMENTATION_STATUS.md** - Complete status and next steps
3. **MVP_Implementation_Plan.md** - Original implementation plan

---

## 🎉 You're Ready!

All code is complete. Just:
1. Add dataset
2. Run notebook
3. Start servers
4. Create PPT
5. Submit

**Good luck! 🚀**
