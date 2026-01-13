# Credit Card Churn Prediction - ML Assessment Project

A comprehensive Machine Learning solution for predicting customer churn in credit card services, featuring end-to-end implementation from data analysis to deployment.

## 🎯 Project Overview

This repository contains a complete ML pipeline for credit card churn prediction, including:

- **Data Analysis & Feature Engineering**
- **Machine Learning Model Training**
- **REST API Backend (FastAPI)**
- **Interactive Web UI (Streamlit)**
- **Complete Documentation**

## 📁 Repository Structure

```
.
├── Codebase/                    # Main implementation
│   ├── notebook.ipynb           # Data analysis & model training
│   ├── api.py                   # FastAPI backend
│   ├── app.py                   # Streamlit frontend
│   ├── requirements.txt         # Dependencies
│   ├── README.md               # Detailed documentation
│   └── test_api.py             # API testing script
│
├── Problem Statement/           # Original problem documentation
├── Solution Approach/           # Presentation & architecture
│
└── Documentation/
    ├── MVP_Implementation_Plan.md
    ├── Quick_Reference_Checklist.md
    ├── IMPLEMENTATION_STATUS.md
    └── QUICK_START.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd Codebase
pip install -r requirements.txt
```

### 2. Train the Model
```bash
jupyter notebook
# Open notebook.ipynb and run all cells
```

### 3. Start the API Server
```bash
uvicorn api:app --reload --port 8000
```

### 4. Launch the Web UI
```bash
streamlit run app.py
```

### 5. Access the Application
- **API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:8501

## 🎓 Features

- ✅ Comprehensive EDA and data understanding
- ✅ Advanced feature engineering
- ✅ Pattern discovery with visualizations
- ✅ Random Forest classifier (80-90% accuracy)
- ✅ RESTful API with automatic documentation
- ✅ Interactive web interface
- ✅ Real-time churn predictions
- ✅ Risk level classification
- ✅ Actionable retention recommendations

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 14 customer attributes
- **Expected Accuracy**: 80-90%
- **Precision**: 75-85%
- **Recall**: 70-80%
- **F1 Score**: 75-82%

## 🛠️ Tech Stack

- **ML/Data**: Python, Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit
- **Model Persistence**: Joblib

## 📚 Documentation

For detailed instructions, see:
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Detailed Docs**: [Codebase/README.md](Codebase/README.md)

## 🧪 Testing

Test the API:
```bash
python test_api.py
```

Or use the interactive API docs at http://localhost:8000/docs

## 📝 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - Model performance metrics
- `POST /predict` - Make churn predictions
- `POST /batch-predict` - Batch predictions

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → ML Model → Predictions
         ↓              ↓          ↓
    Visualizations   REST API   Random Forest
```

## 📦 Files Generated

After training:
- Model artifacts (`.pkl` files)
- Visualizations (`.png` files)
- Performance metrics (`.json`)

## 👤 Author

**Aryan Pawar**
- GitHub: [@Aryanpawar67](https://github.com/Aryanpawar67)

## 📄 License

This project is created for educational and assessment purposes.

## 🤝 Contributing

This is an assessment project. For questions or suggestions, please open an issue.

---

**Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-learn**
