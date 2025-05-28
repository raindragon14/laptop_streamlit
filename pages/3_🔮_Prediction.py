import streamlit as st
import pandas as pd
# import numpy as np # Not strictly needed in this version
import sys
import os

# Add parent directory to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.model_trainer import predict_price, load_model_and_feature_columns
from utils.data_loader import load_sample_data # To get unique values for dropdowns

def show_prediction_page():
    """Renders the prediction page for estimating laptop prices."""
    st.title("🔮 Prediction - Laptop Price Estimation") # Using st.title

    # Load data for populating dropdowns
    if 'df' not in st.session_state:
        with st.spinner('Loading data for input options...'):
            st.session_state.df = load_sample_data()
    df = st.session_state.df

    # Load model and feature columns, with error handling
    try:
        if 'prediction_model' not in st.session_state or 'feature_columns' not in st.session_state:
            # Attempt to load from the 'models' directory as per project structure
            model_path = os.path.join(parent_dir, 'models', 'best_model.pkl')
            feature_columns_path = os.path.join(parent_dir, 'models', 'feature_columns.pkl')
            
            model, feature_cols = load_model_and_feature_columns(model_path, feature_columns_path)
            st.session_state.prediction_model = model
            st.session_state.feature_columns = feature_cols
        model = st.session_state.prediction_model
        feature_columns = st.session_state.feature_columns
    except FileNotFoundError:
        st.error("Prediction model or feature columns file not found. Please ensure 'best_model.pkl' and 'feature_columns.pkl' exist in the 'models/' directory.")
        st.info("You might need to train a model first from the 'Model Performance' page if these files are missing.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the prediction model: {e}")
        st.stop()

    st.subheader("🖥️ Enter Laptop Specifications")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            brand = st.selectbox("Brand", sorted(df['Brand'].unique()), help="Select the laptop brand")
            processor = st.selectbox("Processor", sorted(df['Processor'].unique()), help="Select the processor type")
            ram = st.selectbox("RAM (GB)", sorted(df['RAM_GB'].unique()), help="Select the RAM size")

        with col2:
            storage = st.selectbox("Storage (GB)", sorted(df['Storage_GB'].unique()), help="Select the storage size")
            gpu = st.selectbox("GPU Type", sorted(df['GPU'].unique()), help="Select the GPU type")
            screen_size = st.selectbox("Screen Size (inches)", sorted(df['Screen_Size'].unique()), help="Select the screen size")

        with col3:
            weight_options = sorted(df['Weight_kg'].astype(float).unique()) # Ensure float for slider
            default_weight = weight_options[len(weight_options)//2] if weight_options else 1.5
            weight = st.slider("Weight (kg)",
                               min_value=min(weight_options, default=0.5),
                               max_value=max(weight_options, default=5.0),
                               value=default_weight,
                               step=0.1,
                               help="Select the weight of the laptop")

            battery_options = sorted(df['Battery_Hours'].astype(float).unique()) # Ensure float for slider
            default_battery = battery_options[len(battery_options)//2] if battery_options else 6.0
            battery = st.slider("Battery Life (hours)",
                                min_value=min(battery_options, default=1.0),
                                max_value=max(battery_options, default=15.0),
                                value=default_battery,
                                step=0.5,
                                help="Select the battery life")

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        input_features = {
            'Brand': brand,
            'Processor': processor,
            'RAM_GB': ram,
            'Storage_GB': storage,
            'GPU': gpu,
            'Screen_Size': screen_size,
            'Weight_kg': weight,
            'Battery_Hours': battery
        }

        with st.spinner('Predicting price...'):
            try:
                predicted_price = predict_price(model, feature_columns, input_features)
                st.markdown("---")
                st.subheader("💰 Prediction Result")
                
                st.metric(label="Estimated Price", value=f"${predicted_price:,.2f}")
                
                st.markdown("**Based on the following inputs:**")
                input_summary_df = pd.DataFrame([input_features])
                st.table(input_summary_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.markdown("---")
    st.subheader("💡 Tips for Accurate Predictions")
    st.markdown("""
    - Ensure the specifications accurately reflect the laptop you are interested in.
    - The model's accuracy depends on the data it was trained on. Predictions for highly unusual configurations might be less reliable.
    - This prediction is an estimate, and actual market prices can vary due to factors not included in the model (e.g., condition, seller, current demand).
    """)

if __name__ == "__main__":
    show_prediction_page()
