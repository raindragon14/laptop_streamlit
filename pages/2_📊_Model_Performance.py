import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.model_trainer import train_and_evaluate_model, get_feature_importance
from utils.data_loader import load_sample_data

def show_model_performance_page():
    """Renders the model performance evaluation page."""
    st.title("📊 Model Performance - Training & Evaluation") # Using st.title

    # Load data if not already in session state
    if 'df' not in st.session_state:
        with st.spinner('Loading dataset for model training...'):
            st.session_state.df = load_sample_data()
    df = st.session_state.df

    # Sidebar for model selection and parameters
    st.sidebar.header("🔧 Model Configuration")
    model_options = ["Linear Regression", "Random Forest", "XGBoost"]
    # Get persisted model_type or default to Random Forest
    default_model_type = st.session_state.get('current_model_type', "Random Forest")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        model_options,
        index=model_options.index(default_model_type if default_model_type in model_options else "Random Forest"),
        help="Choose the algorithm for price prediction."
    )
    # Get persisted test_size_percentage or default to 20
    default_test_size = st.session_state.get('current_test_size_percentage', 20)
    test_size_percentage = st.sidebar.slider(
        "Test Size (% of dataset)", 
        min_value=10, max_value=50, 
        value=default_test_size,
        step=5,
        help="Percentage of data to use for testing the model."
    )
    test_size_float = test_size_percentage / 100.0

    # Logic to determine if retraining is needed
    retrain_needed = False
    if 'model_metrics' not in st.session_state: # First time or cleared state
        retrain_needed = True
    elif st.session_state.get('current_model_type') != model_type: # Model type changed
        retrain_needed = True
    elif st.session_state.get('current_test_size_float') != test_size_float: # Test size changed
        retrain_needed = True

    if st.sidebar.button("Train Selected Model", type="primary") or retrain_needed:
        with st.spinner(f'Training {model_type} model with {test_size_percentage}% test data...'):
            metrics, X_test, y_test, trained_model = train_and_evaluate_model(df, model_type, test_size_float)
            st.session_state.model_metrics = metrics
            st.session_state.feature_importance = get_feature_importance(trained_model, X_test.columns, model_type)
            st.session_state.trained_model = trained_model
            st.session_state.current_model_type = model_type
            st.session_state.current_test_size_float = test_size_float
            st.session_state.current_test_size_percentage = test_size_percentage
            st.success(f"{model_type} model trained successfully!")
            # Clear retrain_needed flag after training
            retrain_needed = False 
    elif 'model_metrics' not in st.session_state:
        st.info("Select model parameters and click 'Train Selected Model' to view performance metrics.")
        st.stop()
    
    # Ensure metrics are loaded from session state for display
    metrics = st.session_state.get('model_metrics')
    feature_importance = st.session_state.get('feature_importance')
    current_model_display_name = st.session_state.get('current_model_type', "N/A")

    st.subheader(f"📈 Model Metrics for {current_model_display_name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.3f}", help="Coefficient of determination (higher is better, 1 is perfect)")
    with col2:
        st.metric("Mean Absolute Error (MAE)", f"${metrics['mae']:.2f}", help="Average absolute prediction error in USD (lower is better)")
    with col3:
        st.metric("Root Mean Squared Error (RMSE)", f"${metrics['rmse']:.2f}", help="Square root of mean squared error in USD (lower is better, penalizes large errors more)")

    st.markdown("---")

    st.subheader(f"🔍 Feature Importance for {current_model_display_name}")
    if feature_importance is not None and not feature_importance.empty:
        feature_importance_sorted = feature_importance.sort_values(by="Importance", ascending=True)
        fig = px.bar(feature_importance_sorted, x='Importance', y='Feature',
                    orientation='h', title=f'Feature Importance Scores',
                    labels={'Importance': 'Relative Importance Score', 'Feature': 'Laptop Feature'})
        fig.update_layout(height=max(400, len(feature_importance_sorted['Feature']) * 25 + 100))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Feature importance is not available or not applicable for {current_model_display_name}.")

    st.subheader("🔑 Key Insights & Recommendations")
    r2_score_val = metrics['r2']
    if r2_score_val > 0.85:
        insight_text = "Excellent performance! The model explains a very large portion of price variance."
    elif r2_score_val > 0.7:
        insight_text = "Good performance. The model has strong predictive power."
    elif r2_score_val > 0.5:
        insight_text = "Moderate performance. The model provides some explanatory power, but there's room for improvement."
    else:
        insight_text = "Limited performance. Consider feature engineering, trying different models, or checking data quality for improvement."

    st.markdown(f""" 
    - **R² Score Interpretation:** {insight_text}
    - **Error Metrics (MAE/RMSE):** These values indicate the typical error margin of the predictions in USD. For instance, an MAE of ${metrics['mae']:.2f} means, on average, the prediction is about ${metrics['mae']:.2f} off from the actual price.
    - **Feature Importance:** The plot above highlights which laptop features most significantly influence the price prediction for the *{current_model_display_name}* model. Accurate data for these top features is crucial for reliable predictions.
    - **Next Steps:** You can experiment with different models or adjust the test data percentage to observe changes in performance. The 'best' model often depends on the specific goals (e.g., highest R² vs. lowest MAE).
    """)

if __name__ == "__main__":
    show_model_performance_page()
