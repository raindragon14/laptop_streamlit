import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
# This ensures that 'utils' can be imported when running this page directly or via app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import load_sample_data
# Removed load_custom_css import

def show_dashboard_page():
    """Renders the dashboard page with dataset overview and characteristics."""
    # Removed call to load_custom_css()

    st.title("🏠 Dashboard - Dataset Overview & Analysis") # Using st.title for consistency

    # Load data using the utility function
    if 'df' not in st.session_state:
        with st.spinner('Loading dataset...'):
            st.session_state.df = load_sample_data()
    df = st.session_state.df

    # Dataset overview metrics
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Removed metric-card div, relying on Streamlit's native styling for st.metric
        st.metric("Total Laptops", len(df), help="Total number of laptops in the dataset")

    with col2:
        st.metric("Average Price", f"${df['Price_USD'].mean():.0f}", help="Mean price across all laptops")

    with col3:
        st.metric("Price Range", f"${df['Price_USD'].min():.0f} - ${df['Price_USD'].max():.0f}",
                 help="Minimum and maximum prices")

    with col4:
        st.metric("Unique Brands", df['Brand'].nunique(), help="Number of different laptop brands")

    st.markdown("---")

    # Tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Price Analysis", "🔧 Specifications", "📊 Correlations"])

    with tab1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        col_desc1, col_desc2 = st.columns(2)
        with col_desc1:
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col_desc2:
            st.subheader("🔍 Data Quality")
            data_types_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            }).reset_index(drop=True)
            st.dataframe(data_types_df, use_container_width=True)

    with tab2:
        st.subheader("Price Analysis") # Added subheader for clarity
        col_price1, col_price2 = st.columns(2)
        with col_price1:
            fig = px.histogram(df, x='Price_USD', nbins=30, title='💰 Price Distribution',
                              labels={'Price_USD': 'Price (USD)', 'count': 'Frequency'})
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(df, x='Processor', y='Price_USD', title='💻 Price Distribution by Processor Type')
            fig.update_layout(xaxis_tickangle=45, height=400) # Retained for readability of x-axis labels
            st.plotly_chart(fig, use_container_width=True)
        with col_price2:
            avg_price_by_brand = df.groupby('Brand')['Price_USD'].mean().sort_values(ascending=True)
            fig = px.bar(avg_price_by_brand, x=avg_price_by_brand.values, y=avg_price_by_brand.index,
                        title='🏷️ Average Price by Brand', orientation='h',
                        labels={'x': 'Average Price (USD)', 'y': 'Brand'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(df, x='RAM_GB', y='Price_USD', color='Brand',
                           title='🎯 Price vs RAM by Brand', size='Storage_GB',
                           labels={'RAM_GB': 'RAM (GB)', 'Price_USD': 'Price (USD)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Specifications Analysis") # Added subheader
        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            ram_counts = df['RAM_GB'].value_counts().sort_index()
            fig = px.pie(ram_counts, values=ram_counts.values, names=[f"{x} GB" for x in ram_counts.index],
                        title='💾 RAM Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            gpu_counts = df['GPU'].value_counts()
            fig = px.bar(gpu_counts, x=gpu_counts.index, y=gpu_counts.values,
                        title='🎮 GPU Type Distribution',
                        labels={'x': 'GPU Type', 'y': 'Count'})
            fig.update_layout(xaxis_tickangle=45, height=400) # Retained for readability
            st.plotly_chart(fig, use_container_width=True)
        with col_spec2:
            storage_counts = df['Storage_GB'].value_counts().sort_index()
            fig = px.pie(storage_counts, values=storage_counts.values, names=[f"{x} GB" for x in storage_counts.index],
                        title='💽 Storage Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(df, x='Screen_Size', y='Weight_kg', color='Brand',
                           title='📐 Screen Size vs Weight by Brand',
                           labels={'Screen_Size': 'Screen Size (inches)', 'Weight_kg': 'Weight (kg)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Correlations Analysis") # Added subheader
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        col_corr1, col_corr2 = st.columns([3, 2]) # Adjusted column ratio for better layout
        with col_corr1:
            fig = px.imshow(corr_matrix, title='🔗 Feature Correlation Matrix',
                           color_continuous_scale='RdBu_r', aspect='auto', text_auto='.2f') # RdBu_r for better contrast, text_auto format
            fig.update_layout(height=500) # Removed width to allow container width to manage
            st.plotly_chart(fig, use_container_width=True)
        with col_corr2:
            price_corr = df[numeric_cols].corr()['Price_USD'].abs().sort_values(ascending=False) # Sort descending
            price_corr = price_corr[price_corr.index != 'Price_USD'] # Exclude self-correlation
            # Create a DataFrame for Plotly Express bar chart for easier labeling and sorting
            price_corr_df = price_corr.reset_index()
            price_corr_df.columns = ['Feature', 'Absolute Correlation']
            
            fig = px.bar(price_corr_df, x='Absolute Correlation', y='Feature',
                        title='📊 Feature Correlation with Price',
                        orientation='h', labels={'Absolute Correlation': 'Absolute Correlation with Price', 'Feature': 'Laptop Feature'})
            fig.update_layout(height=500, yaxis_title="") # Removed y-axis title for cleaner look if features are self-explanatory
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Key Insights from Correlations")
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        with insights_col1:
            st.markdown(f"""
            **💰 Price Insights:**
            - Most expensive brand (avg.): *{df.groupby('Brand')['Price_USD'].mean().idxmax()}*
            - Most affordable brand (avg.): *{df.groupby('Brand')['Price_USD'].mean().idxmin()}*
            - Overall price range: *${df['Price_USD'].min():.0f} - ${df['Price_USD'].max():.0f}*
            """)
        with insights_col2:
            st.markdown(f"""
            **🔧 Specification Insights:**
            - Most common RAM: *{df['RAM_GB'].mode()[0]} GB*
            - Most common GPU: *{df['GPU'].mode()[0]}*
            - Average screen size: *{df['Screen_Size'].mean():.1f} inches*
            """)
        with insights_col3:
            if not price_corr.empty:
                highest_corr_feature = price_corr.index[0] # Already sorted descending
                st.markdown(f"""
                **📈 Correlation Insights:**
                - Strongest price predictor: *{highest_corr_feature}*
                - Correlation value: *{price_corr.iloc[0]:.3f}*
                - Potential R² (univariate): *{(price_corr.iloc[0]**2) * 100:.1f}%*
                """)
            else:
                st.markdown("No correlation data to display.")

if __name__ == "__main__":
    # setup_page_config() is called in app.py
    # load_custom_css() is removed
    show_dashboard_page()
