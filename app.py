import streamlit as st
import sys
import os

# Ensure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import setup_page_config

def run_main_app():
    """
    Sets up the main application page.
    Global configurations like page_config are loaded here.
    """
    # Setup page configuration (title, icon, layout)
    # This should be the very first Streamlit command in your app.
    setup_page_config()

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
    run_main_app()
