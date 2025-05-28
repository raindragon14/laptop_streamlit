import streamlit as st
import sys
import os

# Ensure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Assuming main.py is in root

from utils.config import setup_page_config # Removed load_custom_css
from utils.data_loader import load_sample_data

def main_app_entry():
    """
    Sets up the main application page.
    Global configurations like page_config are loaded here.
    """
    # Page configuration
    setup_page_config()

    # Main header
    st.markdown('<h1 class="main-header">💻 Laptop Price Prediction System</h1>', unsafe_allow_html=True)

    # Load data once and store in session state
    if 'df' not in st.session_state:
        with st.spinner('Loading dataset...'):
            st.session_state.df = load_sample_data()

    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("---")

    # Navigation description
    st.sidebar.markdown("""
    **Select a page to explore:**
    - **Dashboard**: Dataset overview and analysis
    - **Model Performance**: ML model evaluation
    - **Prediction**: Make price predictions
    """)

    st.sidebar.markdown("---")

    # Dataset info in sidebar
    if 'df' in st.session_state:
        df = st.session_state.df
        st.sidebar.markdown("### 📊 Dataset Info")
        st.sidebar.metric("Total Records", len(df))
        st.sidebar.metric("Features", len(df.columns) - 1)
        st.sidebar.metric("Price Range", f"${df['Price_USD'].min():.0f} - ${df['Price_USD'].max():.0f}")

    # Main welcome page content
    st.title("💻 Laptop Price Prediction App")

    st.markdown("""
        Welcome to the Laptop Price Prediction application!

        This tool allows you to explore laptop data, understand model performance, and predict laptop prices based on their specifications.

        **Navigate through the app using the sidebar on the left:**
        - **🏠 Dashboard**: Get an overview of the laptop dataset and explore various data visualizations.
        - **📊 Model Performance**: Analyze the performance of different machine learning models trained to predict laptop prices.
        - **🔮 Prediction**: Input laptop specifications to get an estimated price.

        We hope you find this application insightful and easy to use!
    """)

    st.sidebar.success("Select a page above to get started.")

if __name__ == "__main__":
    main_app_entry()
