"""
FastAPI Backend for Credit Card Churn Prediction
Provides REST API endpoints for model predictions and health checks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List, Dict, Any
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Churn Prediction API",
    description="ML-powered API for predicting customer churn in credit card services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    print(f"✓ Model loaded successfully")
    print(f"✓ Using {len(feature_cols)} features")
except FileNotFoundError as e:
    print(f"⚠ Warning: Model files not found. Please train the model first using notebook.ipynb")
    model = None
    scaler = None
    feature_cols = []

# Load metrics if available
try:
    with open('model_metrics.json', 'r') as f:
        MODEL_METRICS = json.load(f)
except FileNotFoundError:
    MODEL_METRICS = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }


class CustomerInput(BaseModel):
    """Input schema for customer data"""
    Customer_Age: float = Field(..., description="Customer age in years", ge=18, le=100)
    Dependent_count: int = Field(..., description="Number of dependents", ge=0, le=10)
    Months_on_book: int = Field(..., description="Period of relationship with bank (months)", ge=0)
    Total_Relationship_Count: int = Field(..., description="Total number of products held", ge=1, le=6)
    Months_Inactive_12_mon: int = Field(..., description="Number of months inactive in last 12 months", ge=0, le=12)
    Contacts_Count_12_mon: int = Field(..., description="Number of contacts in last 12 months", ge=0)
    Credit_Limit: float = Field(..., description="Credit limit on credit card", ge=0)
    Total_Revolving_Bal: float = Field(..., description="Total revolving balance", ge=0)
    Avg_Open_To_Buy: float = Field(..., description="Average open to buy credit line", ge=0)
    Total_Amt_Chng_Q4_Q1: float = Field(..., description="Change in transaction amount Q4 vs Q1", ge=0)
    Total_Trans_Amt: float = Field(..., description="Total transaction amount in last 12 months", ge=0)
    Total_Trans_Ct: int = Field(..., description="Total transaction count in last 12 months", ge=0)
    Total_Ct_Chng_Q4_Q1: float = Field(..., description="Change in transaction count Q4 vs Q1", ge=0)
    Avg_Utilization_Ratio: float = Field(..., description="Average card utilization ratio", ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "Customer_Age": 45,
                "Dependent_count": 2,
                "Months_on_book": 36,
                "Total_Relationship_Count": 3,
                "Months_Inactive_12_mon": 1,
                "Contacts_Count_12_mon": 2,
                "Credit_Limit": 10000.0,
                "Total_Revolving_Bal": 1500.0,
                "Avg_Open_To_Buy": 8500.0,
                "Total_Amt_Chng_Q4_Q1": 0.8,
                "Total_Trans_Amt": 5000.0,
                "Total_Trans_Ct": 50,
                "Total_Ct_Chng_Q4_Q1": 0.7,
                "Avg_Utilization_Ratio": 0.3
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: str = Field(..., description="Predicted customer status")
    churn_probability: float = Field(..., description="Probability of customer churning")
    retention_probability: float = Field(..., description="Probability of customer retention")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., description="Model confidence score")


class MetricsResponse(BaseModel):
    """Response schema for model metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_type: str
    total_features: int


@app.get("/", tags=["General"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "Credit Card Churn Prediction API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make churn predictions",
            "/metrics": "GET - Get model performance metrics",
            "/health": "GET - Check API health status",
            "/features": "GET - List required features"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and scaler is not None

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "features_count": len(feature_cols),
        "api_version": "1.0.0"
    }


@app.get("/features", tags=["General"])
def get_features():
    """Get list of required features"""
    return {
        "total_features": len(feature_cols),
        "features": feature_cols,
        "note": "All features must be provided in the same order for prediction"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """
    Predict customer churn probability

    - **customer**: Customer data with all required features

    Returns prediction with churn probability and risk level
    """

    # Check if model is loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using notebook.ipynb"
        )

    try:
        # Prepare features in correct order
        features = np.array([[
            customer.Customer_Age,
            customer.Dependent_count,
            customer.Months_on_book,
            customer.Total_Relationship_Count,
            customer.Months_Inactive_12_mon,
            customer.Contacts_Count_12_mon,
            customer.Credit_Limit,
            customer.Total_Revolving_Bal,
            customer.Avg_Open_To_Buy,
            customer.Total_Amt_Chng_Q4_Q1,
            customer.Total_Trans_Amt,
            customer.Total_Trans_Ct,
            customer.Total_Ct_Chng_Q4_Q1,
            customer.Avg_Utilization_Ratio
        ]])

        # Handle case where model expects different number of features
        if features.shape[1] != len(feature_cols):
            # Pad or trim features to match expected count
            if features.shape[1] < len(feature_cols):
                padding = np.zeros((1, len(feature_cols) - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :len(feature_cols)]

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Calculate results
        churn_prob = float(probabilities[1])
        retention_prob = float(probabilities[0])
        confidence = float(max(probabilities))

        # Determine risk level
        if churn_prob > 0.7:
            risk_level = "HIGH"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Prepare response
        result = {
            "prediction": "Attrited Customer" if prediction == 1 else "Existing Customer",
            "churn_probability": round(churn_prob, 4),
            "retention_probability": round(retention_prob, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4)
        }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model Info"])
def get_model_metrics():
    """
    Get model performance metrics

    Returns accuracy, precision, recall, and F1 score
    """
    return {
        **MODEL_METRICS,
        "model_type": "Random Forest Classifier",
        "total_features": len(feature_cols)
    }


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(customers: List[CustomerInput]):
    """
    Batch prediction for multiple customers

    - **customers**: List of customer data

    Returns predictions for all customers
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    results = []

    for idx, customer in enumerate(customers):
        try:
            # Use the single prediction endpoint logic
            prediction_result = predict_churn(customer)
            results.append({
                "customer_index": idx,
                **prediction_result.dict()
            })
        except Exception as e:
            results.append({
                "customer_index": idx,
                "error": str(e)
            })

    return {
        "total_customers": len(customers),
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Starting Credit Card Churn Prediction API")
    print("="*60)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
