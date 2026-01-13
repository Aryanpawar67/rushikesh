"""
Streamlit Frontend for Credit Card Churn Prediction
Interactive UI for making churn predictions via FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def check_api_health():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_model_metrics():
    """Fetch model performance metrics from API"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def make_prediction(customer_data):
    """Call API to make prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure API is running at http://localhost:8000"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🏦 Credit Card Churn Prediction System</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - API Status and Dataset Info
    with st.sidebar:
        st.header("📊 System Information")

        # API Health Check
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Status: Online")
        else:
            st.error("❌ API Status: Offline")
            st.warning("⚠️ Please start the API server:\n```bash\nuvicorn api:app --reload\n```")

        st.markdown("---")

        # Dataset Information
        st.header("📁 Dataset Info")

        # Try to load dataset for display
        dataset_path = None
        for possible_path in ['Dataset(BankChurners)_CampusHiring_Dec2025(dataset).csv',
                            '../Dataset(BankChurners)_CampusHiring_Dec2025(dataset).csv',
                            'your_dataset.csv', '../your_dataset.csv', 'dataset.csv']:
            if os.path.exists(possible_path):
                dataset_path = possible_path
                break

        if dataset_path:
            try:
                df = pd.read_csv(dataset_path)
                st.write(f"**Total Records:** {df.shape[0]:,}")
                st.write(f"**Total Features:** {df.shape[1]}")
                st.write(f"**Shape:** {df.shape}")

                # Show sample data
                if st.checkbox("Show Sample Data"):
                    st.dataframe(df.head())

            except Exception as e:
                st.info("Dataset not yet loaded")
        else:
            st.info("📝 Upload your dataset to the project directory")

        st.markdown("---")

        # Model Information
        st.header("🤖 Model Info")
        metrics = get_model_metrics()
        if metrics:
            st.write(f"**Model:** {metrics.get('model_type', 'Random Forest')}")
            st.write(f"**Features:** {metrics.get('total_features', 'N/A')}")
        else:
            st.info("Model metrics unavailable")

    # Main Content Area
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Performance", "ℹ️ About"])

    # TAB 1: PREDICTION
    with tab1:
        st.header("Customer Churn Prediction")
        st.write("Enter customer information to predict churn risk")

        with st.form("prediction_form"):
            # Create 3 columns for input fields
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("📋 Demographics")
                customer_age = st.number_input(
                    "Customer Age",
                    min_value=18,
                    max_value=100,
                    value=45,
                    help="Customer's age in years"
                )
                dependent_count = st.number_input(
                    "Number of Dependents",
                    min_value=0,
                    max_value=10,
                    value=2,
                    help="Number of dependents"
                )
                months_on_book = st.number_input(
                    "Months on Book",
                    min_value=0,
                    max_value=100,
                    value=36,
                    help="Period of relationship with bank (months)"
                )
                total_relationship_count = st.number_input(
                    "Total Relationship Count",
                    min_value=1,
                    max_value=6,
                    value=3,
                    help="Total number of products held by customer"
                )

            with col2:
                st.subheader("💳 Account Activity")
                months_inactive = st.number_input(
                    "Months Inactive (12m)",
                    min_value=0,
                    max_value=12,
                    value=1,
                    help="Number of months inactive in last 12 months"
                )
                contacts_count = st.number_input(
                    "Contacts Count (12m)",
                    min_value=0,
                    max_value=20,
                    value=2,
                    help="Number of contacts in last 12 months"
                )
                total_trans_ct = st.number_input(
                    "Total Transaction Count",
                    min_value=0,
                    max_value=200,
                    value=50,
                    help="Total transaction count in last 12 months"
                )
                total_ct_chng_q4_q1 = st.number_input(
                    "Transaction Count Change (Q4-Q1)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.7,
                    step=0.1,
                    help="Change in transaction count Q4 vs Q1"
                )

            with col3:
                st.subheader("💰 Financial Metrics")
                credit_limit = st.number_input(
                    "Credit Limit",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=500,
                    help="Credit limit on credit card"
                )
                total_revolving_bal = st.number_input(
                    "Total Revolving Balance",
                    min_value=0,
                    max_value=50000,
                    value=1500,
                    step=100,
                    help="Total revolving balance on card"
                )
                avg_open_to_buy = st.number_input(
                    "Avg Open to Buy",
                    min_value=0,
                    max_value=100000,
                    value=8500,
                    step=500,
                    help="Average open to buy credit line"
                )
                total_trans_amt = st.number_input(
                    "Total Transaction Amount",
                    min_value=0,
                    max_value=50000,
                    value=5000,
                    step=100,
                    help="Total transaction amount in last 12 months"
                )
                total_amt_chng_q4_q1 = st.number_input(
                    "Transaction Amount Change (Q4-Q1)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.8,
                    step=0.1,
                    help="Change in transaction amount Q4 vs Q1"
                )
                avg_utilization = st.number_input(
                    "Avg Utilization Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.01,
                    format="%.2f",
                    help="Average card utilization ratio (0-1)"
                )

            # Submit button
            submitted = st.form_submit_button(
                "🚀 Predict Churn Risk",
                use_container_width=True
            )

        # Process prediction
        if submitted:
            if not api_status:
                st.error("❌ Cannot make prediction: API is offline")
            else:
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
                    "total_amt_chng_q4_q1": total_amt_chng_q4_q1,
                    "total_trans_amt": total_trans_amt,
                    "total_trans_ct": total_trans_ct,
                    "total_ct_chng_q4_q1": total_ct_chng_q4_q1,
                    "avg_utilization_ratio": avg_utilization
                }

                # Make prediction
                with st.spinner("🔄 Analyzing customer data..."):
                    result, error = make_prediction(payload)

                if error:
                    st.error(f"❌ {error}")
                elif result:
                    st.success("✅ Prediction Complete!")

                    # Display results
                    st.markdown("### 📊 Prediction Results")

                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Prediction",
                            result['prediction'],
                            delta=None
                        )

                    with col2:
                        churn_prob = result['churn_probability']
                        st.metric(
                            "Churn Probability",
                            f"{churn_prob:.2%}",
                            delta=f"{(churn_prob - 0.5):.2%}"
                        )

                    with col3:
                        st.metric(
                            "Retention Probability",
                            f"{result['retention_probability']:.2%}"
                        )

                    with col4:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.2%}"
                        )

                    # Risk level indicator
                    risk_level = result['risk_level']
                    if risk_level == "HIGH":
                        st.markdown(
                            '<div class="danger-box">⚠️ <b>HIGH RISK</b> - Customer is likely to churn. Immediate action recommended.</div>',
                            unsafe_allow_html=True
                        )
                    elif risk_level == "MEDIUM":
                        st.markdown(
                            '<div class="warning-box">⚡ <b>MEDIUM RISK</b> - Monitor customer activity closely.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="success-box">✅ <b>LOW RISK</b> - Customer is likely to stay.</div>',
                            unsafe_allow_html=True
                        )

                    # Probability gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=churn_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk Score", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': '#d4edda'},
                                {'range': [40, 70], 'color': '#fff3cd'},
                                {'range': [70, 100], 'color': '#f8d7da'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if risk_level == "HIGH":
                        st.markdown("""
                        - **Immediate Action Required:**
                          - Contact customer within 24-48 hours
                          - Offer personalized retention incentives
                          - Review account for service issues
                          - Consider special promotional offers
                        """)
                    elif risk_level == "MEDIUM":
                        st.markdown("""
                        - **Monitor and Engage:**
                          - Track account activity weekly
                          - Send engagement communications
                          - Offer relevant product upgrades
                          - Conduct satisfaction survey
                        """)
                    else:
                        st.markdown("""
                        - **Maintain Relationship:**
                          - Continue regular communications
                          - Reward loyalty with benefits
                          - Cross-sell relevant products
                          - Gather feedback for improvements
                        """)

    # TAB 2: PERFORMANCE
    with tab2:
        st.header("📈 Model Performance Metrics")

        metrics = get_model_metrics()

        if metrics:
            st.markdown("### Model Overview")
            st.write(f"**Model Type:** {metrics.get('model_type', 'Random Forest Classifier')}")
            st.write(f"**Total Features:** {metrics.get('total_features', 'N/A')}")

            st.markdown("---")

            # Performance metrics
            st.markdown("### Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                accuracy = metrics.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.2%}")

            with col2:
                precision = metrics.get('precision', 0)
                st.metric("Precision", f"{precision:.2%}")

            with col3:
                recall = metrics.get('recall', 0)
                st.metric("Recall", f"{recall:.2%}")

            with col4:
                f1 = metrics.get('f1_score', 0)
                st.metric("F1 Score", f"{f1:.2%}")

            # Metrics bar chart
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Score': [
                    accuracy * 100,
                    precision * 100,
                    recall * 100,
                    f1 * 100
                ]
            })

            fig = px.bar(
                metrics_df,
                x='Metric',
                y='Score',
                title='Model Performance Overview',
                color='Score',
                color_continuous_scale='Blues',
                range_y=[0, 100]
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Metrics explanation
            st.markdown("---")
            st.markdown("### 📚 Metrics Explanation")

            with st.expander("Understanding the Metrics"):
                st.markdown("""
                - **Accuracy**: Overall correctness of the model (percentage of correct predictions)
                - **Precision**: Of all predicted churns, how many actually churned (reduces false alarms)
                - **Recall**: Of all actual churns, how many did we catch (reduces missed churns)
                - **F1 Score**: Harmonic mean of precision and recall (balanced measure)

                **For Churn Prediction:**
                - High **Recall** is critical - we don't want to miss customers who will churn
                - Good **Precision** saves resources - reduces wasted retention efforts
                - **Accuracy** gives overall confidence in the model
                """)

        else:
            st.warning("⚠️ Model metrics not available. Please train the model first.")

    # TAB 3: ABOUT
    with tab3:
        st.header("ℹ️ About This System")

        st.markdown("""
        ### 🏦 Credit Card Churn Prediction System

        This application uses Machine Learning to predict customer churn in credit card services,
        helping businesses proactively identify at-risk customers and take preventive actions.

        ### 🎯 Features

        - **Real-time Predictions**: Get instant churn probability scores
        - **Risk Assessment**: Automated risk level classification (Low/Medium/High)
        - **Actionable Insights**: Personalized recommendations for customer retention
        - **Performance Monitoring**: Track model accuracy and reliability
        - **Interactive UI**: User-friendly interface for easy data entry

        ### 🔧 Technical Stack

        - **Frontend**: Streamlit (Python)
        - **Backend API**: FastAPI
        - **ML Model**: Random Forest Classifier
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Plotly, Seaborn, Matplotlib

        ### 📊 Model Features

        The model analyzes 14 key customer attributes:
        - Demographics (age, dependents)
        - Account information (tenure, relationship count)
        - Activity metrics (transactions, contacts)
        - Financial indicators (credit limit, utilization, balances)

        ### 🚀 Quick Start

        1. **Start the API**:
           ```bash
           cd Codebase
           uvicorn api:app --reload --port 8000
           ```

        2. **Run the Streamlit App**:
           ```bash
           streamlit run app.py
           ```

        3. **Make Predictions**: Enter customer data and get instant churn predictions!

        ### 📝 Notes

        - Ensure the model is trained before making predictions
        - The API must be running at `http://localhost:8000`
        - All input fields are required for accurate predictions
        - Model performance may vary based on training data quality

        ### 📧 Support

        For questions or issues, please refer to the README.md file or contact your system administrator.

        ---

        **Version**: 1.0.0 | **Last Updated**: """ + datetime.now().strftime("%B %Y") + """
        """)


if __name__ == "__main__":
    main()
