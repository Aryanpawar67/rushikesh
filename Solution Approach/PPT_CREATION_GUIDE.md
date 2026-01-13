# PPT Creation Guide

## 📊 How to Create Your Presentation

### Option 1: PowerPoint / Google Slides (Recommended)

**Step-by-Step:**

1. **Open PowerPoint or Google Slides**
   - PowerPoint: Microsoft Office
   - Google Slides: https://slides.google.com

2. **Choose a Professional Template**
   - For PowerPoint: File → New → Search "Professional" or "Business"
   - For Google Slides: Template Gallery → "Pitch" or "Business"
   - Recommended: Dark theme with blue/purple accents

3. **Create Slides Based on SOLUTION_REPORT_CONTENT.md**

---

## 📑 Slide-by-Slide Instructions

### SLIDE 1: Cover Page
**Layout:** Title Slide

**Content:**
```
Title: Credit Card Churn Prediction System
Subtitle: ML-Based Customer Retention Solution

[Center of slide]
Rushikesh Vilas Kadam
rushikadam1912@gmail.com
Vishwakarma University
BTech AI & DS

Date: January 2026
```

**Design Tips:**
- Use large, bold font for name
- Add a background image (credit cards, data visualization)
- Keep it clean and professional

---

### SLIDE 2: Data Quality Assessment
**Layout:** Title and Content (2 columns)

**Left Column:**
- Dataset Overview (bullet points)
- 10,127 customers, 21 features
- Data type breakdown

**Right Column:**
- Quality Summary Table
- Use checkmarks ✓ and warning symbols ⚠
- Add small bar chart showing class distribution

**Visual:**
- Pie chart: 84% Existing vs 16% Attrited

---

### SLIDE 3: Transformations Applied
**Layout:** Title and Content

**Content Structure:**
1. Feature Engineering header
2. Three sections:
   - A. Conditional Columns (with code snippet)
   - B. Aggregations (with formula)
   - C. Preprocessing (bullet points)

**Visual:**
- Small flowchart showing: Raw Data → Transform → Processed Data
- Use icons for each transformation step

---

### SLIDE 4: Pattern Discovery - Part 1
**Layout:** Two Content (side by side)

**Left Side:**
- **Pattern 1: Transaction Behavior**
- Insert: pattern1_transaction_behavior.png
- Key finding (bullet point)
- Root cause (bullet point)

**Right Side:**
- **Pattern 2: Credit Utilization**
- Insert: pattern2_utilization.png
- Key finding (bullet point)
- Root cause (bullet point)

---

### SLIDE 5: Pattern Discovery - Part 2
**Layout:** Title and Content

**Content:**
- **Pattern 3: Feature Correlation**
- Insert: pattern3_correlation.png (large, centered)
- Below image: 3-4 bullet points with key findings
- Highlight strongest correlations

---

### SLIDE 6: Model Selection & Evaluation
**Layout:** Title and Content (2 columns)

**Left Column:**
- **Why Random Forest?** (5 bullet points)
- Model Configuration (code block or formatted list)

**Right Column:**
- **Performance Metrics** (table)
  ```
  Accuracy:  87.5%
  Precision: 83.2%
  Recall:    79.8%
  F1 Score:  81.4%
  ```
- Insert: confusion_matrix.png (small)

**Visual:**
- Add green checkmarks next to metrics
- Use progress bars to show percentages

---

### SLIDE 7: Non-Obvious Insights (Part 1)
**Layout:** Title and Content

**Content:**
List top 5 insights:

1. **Transaction Frequency > Amount**
   - Brief explanation (1-2 sentences)
   - Business implication

2. **Goldilocks Zone of Utilization**
   - U-shaped relationship
   - Action items

3. **Relationship Count Paradox**
   - Sweet spot at 3-4 products
   - Don't over-sell

4. **Contact Frequency Paradox**
   - High contact = problem indicator
   - Proactive > Reactive

5. **Age is Irrelevant**
   - Behavior > Demographics
   - Universal strategies work better

**Design:** Use numbered list with icons

---

### SLIDE 8: Non-Obvious Insights (Part 2)
**Layout:** Title and Content

**Content:**
Continue with insights 6-10:

6. **Silent Churner Profile**
7. **Months on Book Non-Linearity**
8. **Compound Effect Discovery**
9. **Data Drift Implications**
10. **Missing Feature Hypothesis**

**Design:** Bullet format with key takeaways highlighted

---

### SLIDE 9: System Architecture
**Layout:** Blank (custom layout)

**Content:**
- Copy the architecture diagram from SOLUTION_REPORT_CONTENT.md
- Create visually using:
  - Shapes (rectangles for components)
  - Arrows for data flow
  - Different colors for different layers

**Layers to Show:**
1. User Layer (Top) - Light blue
2. API Layer (Middle) - Green
3. Model & Storage (Middle-bottom) - Orange
4. ETL/Ingestion (Bottom) - Purple

**Tools:**
- Use SmartArt or Shapes in PowerPoint
- Or use draw.io then screenshot

---

### SLIDE 10: UI Screenshots - Main Interface
**Layout:** Picture with Caption

**Content:**
- Insert Screenshot 1: Streamlit main interface
- Caption: "Customer Churn Prediction - Input Form"
- Add callout boxes pointing to:
  - API Status (green)
  - Input fields
  - Predict button

---

### SLIDE 11: UI Screenshots - Prediction Results
**Layout:** Picture with Caption

**Content:**
- Insert Screenshot 2: Prediction results
- Caption: "Churn Prediction with Risk Assessment"
- Highlight:
  - Churn probability gauge
  - Risk level badge
  - Recommendations section

---

### SLIDE 12: UI Screenshots - API Docs
**Layout:** Picture with Caption

**Content:**
- Insert Screenshot 3: API documentation
- Caption: "FastAPI Swagger UI - Interactive Documentation"
- Show endpoints list

---

### SLIDE 13: Results & Achievements
**Layout:** Title and Content

**Content:**
- **Key Accomplishments** (checkmarks)
  - Data Analysis ✓
  - Feature Engineering ✓
  - Model Training ✓
  - API Development ✓
  - UI Development ✓

- **Performance Summary Table**
  - Compare to benchmarks
  - Show "Exceeds" status

- **Business Impact** (3-4 bullets)

---

### SLIDE 14: Deliverables Summary
**Layout:** Title and Content

**Content:**
- **Executable Codebase** (file tree)
- **Documentation** (list)
- **GitHub Repository** (link)
- **Working Application** (URLs)

---

### SLIDE 15: Thank You / Q&A
**Layout:** Title Slide

**Content:**
```
Thank You!

Rushikesh Vilas Kadam
rushikadam1912@gmail.com

GitHub: https://github.com/Aryanpawar67/rushikesh

Questions?
```

---

## 🎨 Design Guidelines

### Color Scheme
- **Primary:** Navy Blue (#1e3a8a)
- **Secondary:** Purple (#7c3aed)
- **Accent:** Green (#10b981) for success/checkmarks
- **Warning:** Amber (#f59e0b)
- **Danger:** Red (#ef4444)

### Fonts
- **Headings:** Montserrat Bold or Arial Bold
- **Body:** Open Sans or Calibri
- **Code:** Consolas or Courier New

### Images
- All visualizations are in: `Codebase/`
  - pattern1_transaction_behavior.png
  - pattern2_utilization.png
  - pattern3_correlation.png
  - confusion_matrix.png
  - feature_importance.png

### Tips
- Keep slides clean (not too much text)
- Use bullet points (max 5-6 per slide)
- Add visuals to every slide
- Use animations sparingly
- Consistent font sizes (32pt title, 18pt body)

---

## 📸 How to Take Screenshots

### For Streamlit UI:
1. Start both servers (API + Streamlit)
2. Open http://localhost:8501
3. Fill in sample data
4. Take screenshot **before** clicking predict (Screenshot 1)
5. Click "Predict Churn Risk"
6. Take screenshot **after** results show (Screenshot 2)
7. Go to Performance tab
8. Take screenshot (Screenshot 3)

### For API Docs:
1. Open http://localhost:8000/docs
2. Scroll to show all endpoints
3. Take screenshot (Screenshot 4)

### Screenshot Tools:
- **Mac:** Cmd + Shift + 4 (select area)
- **Windows:** Windows + Shift + S
- **Linux:** Gnome Screenshot or Flameshot

---

## ✅ Final Checklist

Before submitting:
- [ ] All 15 slides created
- [ ] Personal details on cover page
- [ ] All 5 screenshots inserted
- [ ] Architecture diagram included
- [ ] No typos or grammar errors
- [ ] Consistent formatting throughout
- [ ] File saved as: `Rushikesh_Kadam_Solution_Approach.pptx`

---

## 💾 Save & Export

**PowerPoint:**
- File → Save As
- Name: `Rushikesh_Kadam_Solution_Approach.pptx`
- Location: `Solution Approach/` folder

**Google Slides:**
- File → Download → Microsoft PowerPoint (.pptx)
- Save to: `Solution Approach/` folder

**Also Save as PDF:**
- File → Export as PDF
- Name: `Rushikesh_Kadam_Solution_Approach.pdf`

---

## 📦 Alternative: Use the Markdown Content

If you prefer to work with Markdown or need to create a document instead:

**Option A: Convert to PDF**
```bash
# Install pandoc
brew install pandoc  # Mac
# or
sudo apt install pandoc  # Linux

# Convert to PDF
cd "Solution Approach"
pandoc SOLUTION_REPORT_CONTENT.md -o Rushikesh_Kadam_Report.pdf
```

**Option B: Open in VS Code and print**
- Open SOLUTION_REPORT_CONTENT.md in VS Code
- Install "Markdown PDF" extension
- Right-click → Markdown PDF: Export (pdf)

---

**Estimated Time to Create PPT:** 45-60 minutes

**Good luck! 🚀**
