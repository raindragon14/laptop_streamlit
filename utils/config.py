import streamlit as st

def setup_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Laptop Price Prediction System",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_global_styles():
    """
    Placeholder for any global style adjustments using Streamlit's
    native capabilities if needed in the future, but without custom CSS injections.
    For now, this function does nothing as we are removing custom CSS.
    """
    pass
