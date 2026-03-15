import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import io
import base64
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Gold Price Regression Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Gold Price Linear Regression Analysis")
st.markdown("---")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    
    # Sample data download
    st.subheader("Sample Data")
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2006-01-01', periods=10, freq='YS'),
        'Price': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
    })
    
    # csv_sample = sample_data.to_csv(index=False)
    # st.download_button(
    #     label="📥 Download Sample CSV",
    #     data=csv_sample, 
    #     file_name="gold_price_sample.csv",
    #     mime="text/csv"
    # )

    # Kalau mau pakai path, harus dibaca dulu
    with open("static/gold_price_2006_2025.csv", "rb") as f:
        file_content = f.read()

    st.download_button(
        label="📥 Download Sample CSV",
        data=file_content,  # ← ini content, bukan path
        file_name="gold_price_2006_2025.csv",
        mime="text/csv"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="File must have 'Date' and 'Price' columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if 'Date' not in df.columns or 'Price' not in df.columns:
                st.error("CSV must have columns: Date and Price")
            else:
                st.session_state.df = df
                st.session_state.processed = False
                st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # =========================
    # 1️⃣ DATA PREPROCESSING
    # =========================
    with st.spinner("Processing data..."):
        # Clean Price column
        df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        
        # Convert Date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create Days variable
        start_date = df['Date'].min()
        df['Days'] = (df['Date'] - start_date).dt.days
        
        # =========================
        # 2️⃣ INITIAL DATA INFO
        # =========================
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Date Range", f"{df['Date'].min().year} - {df['Date'].max().year}")
        with col3:
            st.metric("Price Range", f"${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        
        # =========================
        # 3️⃣ DESCRIPTIVE STATISTICS
        # =========================
        st.subheader("📈 Descriptive Statistics")
        stats = df['Price'].describe().to_frame().T
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
        
        # =========================
        # 4️⃣ RAW DATA PREVIEW
        # =========================
        with st.expander("🔍 Raw Data Preview"):
            st.dataframe(df[['Date', 'Price']].head(10), use_container_width=True)
        
        # =========================
        # 5️⃣ VISUALIZATIONS
        # =========================
        st.subheader("📊 Price Movement Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Daily", "Monthly", "Yearly"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Date'], df['Price'], linewidth=1)
            ax.set_title("Daily Gold Price Movement")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            monthly_df = df.resample('ME', on='Date')['Price'].mean()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly_df.index, monthly_df.values, linewidth=1.5)
            ax.set_title("Monthly Gold Price Movement (Average)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            yearly_df = df.resample('YE', on='Date')['Price'].mean()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(yearly_df.index, yearly_df.values, marker='o', linestyle='-', linewidth=2, markersize=6)
            ax.set_title("Yearly Gold Price Movement (Average)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        # =========================
        # 6️⃣ LINEAR REGRESSION MODELS
        # =========================
        st.subheader("📐 Linear Regression Models")
        
        X = df[['Days']]
        y = df['Price']
        
        # Model 1: Without Split
        model_full = LinearRegression()
        model_full.fit(X, y)
        y_pred_full = model_full.predict(X)
        
        # Model 2: With 80:20 Split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model_split = LinearRegression()
        model_split.fit(X_train, y_train)
        y_test_pred = model_split.predict(X_test)
        y_pred_all = model_split.predict(X)
        
        # Display equations
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Without Split (In-Sample)**  \nPrice = {model_full.coef_[0]:.4f} × Day + {model_full.intercept_:.2f}")
        with col2:
            st.info(f"**80:20 Split Model (Out-of-Sample)**  \nPrice = {model_split.coef_[0]:.4f} × Day + {model_split.intercept_:.2f}")
        
        # =========================
        # 7️⃣ MODEL COMPARISON VISUALIZATION
        # =========================
        st.subheader("🔄 Model Comparison")
        
        comp_tab1, comp_tab2 = st.tabs(["Without Split", "80:20 Split"])
        
        with comp_tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(df['Date'], y, s=8, alpha=0.5, label="Actual Data", color='blue')
            ax.plot(df['Date'], y_pred_full, color='red', linewidth=2, label="Regression (Without Split)")
            ax.set_title("Linear Regression Model - Without Data Split (In-Sample)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
            
            mse_full = mean_squared_error(y, y_pred_full)
            r2_full = r2_score(y, y_pred_full)
            
            col1, col2 = st.columns(2)
            col1.metric("MSE", f"{mse_full:.2f}")
            col2.metric("R² Score", f"{r2_full:.4f}")
        
        with comp_tab2:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(df['Date'].iloc[:split_idx], y_train, s=8, label="Train Data (80%)", color='blue', alpha=0.6)
            ax.scatter(df['Date'].iloc[split_idx:], y_test, s=8, label="Test Data (20%)", color='orange', alpha=0.6)
            ax.plot(df['Date'], y_pred_all, color='red', linewidth=2, label="Regression (80:20 Split)")
            ax.set_title("Linear Regression Model - 80:20 Data Split (Out-of-Sample)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
            
            mse_split = mean_squared_error(y_test, y_test_pred)
            r2_split = r2_score(y_test, y_test_pred)
            
            col1, col2 = st.columns(2)
            col1.metric("MSE", f"{mse_split:.2f}")
            col2.metric("R² Score", f"{r2_split:.4f}")
        
        # =========================
        # 8️⃣ PREDICTIONS 2026-2035
        # =========================
        st.subheader("🔮 Gold Price Predictions 2026-2035")
        
        prediction_years = pd.date_range(start='2026-01-01', end='2035-01-01', freq='YS')
        pred_df = pd.DataFrame({'Date': prediction_years})
        pred_df['Days'] = (pred_df['Date'] - start_date).dt.days
        pred_df['Predicted_Price'] = model_split.predict(pred_df[['Days']])
        
        # Show predictions table
        st.dataframe(
            pred_df[['Date', 'Predicted_Price']].style.format({
                'Date': lambda x: x.strftime('%Y-%m-%d'),
                'Predicted_Price': '${:.2f}'
            }),
            use_container_width=True
        )
        
        # Full visualization with predictions
        st.subheader("📈 Complete Analysis with Predictions")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(df['Date'].iloc[:split_idx], y_train, s=8, label="Train (80%)", alpha=0.6)
        ax.scatter(df['Date'].iloc[split_idx:], y_test, s=8, label="Test (20%)", color='orange', alpha=0.6)
        ax.plot(df['Date'], y_pred_all, color='red', linewidth=2, label="Linear Regression")
        ax.plot(pred_df['Date'], pred_df['Predicted_Price'], 
                color='green', linestyle='--', marker='o', markersize=8, 
                linewidth=2, label="Prediction 2026–2035")
        
        ax.set_xlim(df['Date'].min(), pd.Timestamp('2035-12-31'))
        ax.set_title("Linear Regression and Gold Price Prediction 2026–2035")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
        # Download predictions
        csv_pred = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_pred,
            file_name="gold_price_predictions_2026_2035.csv",
            mime="text/csv"
        )
        
        st.session_state.processed = True

else:
    # No file uploaded
    st.info("👈 Please upload a CSV file using the sidebar to begin analysis")
    
    # Show example
    with st.expander("📋 CSV Format Example"):
        example_df = pd.DataFrame({
            'Date': ['2006-01-02', '2006-01-03', '2006-01-04'],
            'Price': [1521.55, 1520.30, 1525.80]
        })
        st.dataframe(example_df)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Simple Linear Regression Analysis System • Thesis • Informatics Engineering"
    "</div>",
    unsafe_allow_html=True
)