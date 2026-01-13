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
    customer_age: float = Field(..., description="Customer age in years", ge=18, le=100)
    dependent_count: int = Field(..., description="Number of dependents", ge=0, le=10)
    months_on_book: int = Field(..., description="Period of relationship with bank (months)", ge=0)
    total_relationship_count: int = Field(..., description="Total number of products held", ge=1, le=6)
    months_inactive_12_mon: int = Field(..., description="Number of months inactive in last 12 months", ge=0, le=12)
    contacts_count_12_mon: int = Field(..., description="Number of contacts in last 12 months", ge=0)
    credit_limit: float = Field(..., description="Credit limit on credit card", ge=0)
    total_revolving_bal: float = Field(..., description="Total revolving balance", ge=0)
    avg_open_to_buy: float = Field(..., description="Average open to buy credit line", ge=0)
    total_amt_chng_q4_q1: float = Field(..., description="Change in transaction amount Q4 vs Q1", ge=0)
    total_trans_amt: float = Field(..., description="Total transaction amount in last 12 months", ge=0)
    total_trans_ct: int = Field(..., description="Total transaction count in last 12 months", ge=0)
    total_ct_chng_q4_q1: float = Field(..., description="Change in transaction count Q4 vs Q1", ge=0)
    avg_utilization_ratio: float = Field(..., description="Average card utilization ratio", ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "customer_age": 45,
                "dependent_count": 2,
                "months_on_book": 36,
                "total_relationship_count": 3,
                "months_inactive_12_mon": 1,
                "contacts_count_12_mon": 2,
                "credit_limit": 10000.0,
                "total_revolving_bal": 1500.0,
                "avg_open_to_buy": 8500.0,
                "total_amt_chng_q4_q1": 0.8,
                "total_trans_amt": 5000.0,
                "total_trans_ct": 50,
                "total_ct_chng_q4_q1": 0.7,
                "avg_utilization_ratio": 0.3
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
