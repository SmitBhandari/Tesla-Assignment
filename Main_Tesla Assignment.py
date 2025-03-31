import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(page_title="Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

logo_icon = "Tesla Logo.png"
logo_image = "Tesla Logo_name.png"
st.logo(icon_image=logo_icon, image=logo_image)

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Forecasting Dashboard", "Comparative Analysis", "Model Validation"])

# File path to your Sample Data CSV file
file_path = "Sample Data.csv"  # Update this path if necessary

# Check if the file exists
if os.path.exists(file_path):
    # Load and preprocess the data
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    if page == "Forecasting Dashboard":
        st.title("Forecasting Dashboard")

        # Sidebar options
        st.sidebar.markdown("### Forecasting Options")
        sku_options = df["SKU_ID"].unique()
        selected_sku = st.sidebar.selectbox("Select SKU", sku_options)

        model_options = ["ARIMA", "Holt-Winters", "Linear Regression"]
        selected_model = st.sidebar.selectbox("Select Model", model_options)

        # Generate lag options dynamically (1 to 12 months)
        lag_options = [f"{i}-Month Lag" for i in range(1, 13)]
        selected_lag = st.sidebar.selectbox("Update Frequency", lag_options)

        # Filter data for the selected SKU
        df_sku = df[df["SKU_ID"] == selected_sku]

        # Define train-test split based on lag
        lag_months = int(selected_lag.split("-")[0])  # Extract the number of months from the selected lag
        train_end_date = pd.Timestamp("2024-06-30") - pd.DateOffset(months=lag_months)

        train = df_sku.loc[:train_end_date]
        test = df_sku.loc[train_end_date:]

        # Forecasting based on selected model
        def forecast_model(train, test, model_type):
            if model_type == "ARIMA":
                # Use statsmodels ARIMA
                model = ARIMA(train["Weekly_Sales"], order=(5, 1, 0))  # Replace (5, 1, 1) with your desired parameters
                fit = model.fit()
                return fit.forecast(steps=len(test))

            elif model_type == "Holt-Winters":
                model = ExponentialSmoothing(train["Weekly_Sales"], trend="add", seasonal="add", seasonal_periods=52)
                fit = model.fit()
                return fit.forecast(steps=len(test))

            elif model_type == "Linear Regression":
                train["Time"] = np.arange(len(train))
                test["Time"] = np.arange(len(train), len(train) + len(test))
                lr_model = LinearRegression()
                lr_model.fit(train[["Time"]], train["Weekly_Sales"])
                return lr_model.predict(test[["Time"]])

        # Generate forecast
        forecast = forecast_model(train, test, selected_model)

        # Ensure no negative forecast values as demand cannot be negative
        forecast = np.maximum(forecast, 0)

        # Align forecast with test index
        forecast = pd.Series(forecast, index=test.index)

        # Evaluate performance
        mad = np.mean(np.abs(test["Weekly_Sales"] - forecast))
        mse = np.mean((test["Weekly_Sales"] - forecast) ** 2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mbe = np.mean(forecast - test["Weekly_Sales"])  # Bias: Mean difference between forecasted and actual values

        # Weighted MAPE calculation
        wmape = (np.sum(np.abs(test["Weekly_Sales"] - forecast)) / np.sum(test["Weekly_Sales"])) * 100

        # Display scorecards on the left side
        col1, col2 = st.columns([1, 3])
        with col1:
            with st.container(border=True):   
                st.markdown("### Performance Metrics")
                st.metric(label="Mean Absolute Deviation (MAD)", value=f"{mad:.2f}")
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
                st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
                st.metric(label="Weighted Mean Absolute Percentage Error (WMAPE)", value=f"{wmape:.2f}%")
                st.metric(label="Mean Bias Error (MBE)", value=f"{mbe:.2f}")

        # Display the chart on the right side
        with col2:
            with st.container(border=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train["Weekly_Sales"], mode="lines", name="Train" ))
                fig.add_trace(go.Scatter(x=test.index, y=test["Weekly_Sales"], mode="lines", name="Test", line=dict(color="orange")))
                fig.add_trace(go.Scatter(x=test.index, y=forecast, mode="lines", name="Forecast", line=dict(color="red", dash="dash")))
                fig.update_layout(
                    title=f"{selected_model} Forecast for SKU {selected_sku} ({selected_lag})",
                    xaxis_title="Date",
                    yaxis_title="Weekly Sales",
                    template="plotly_dark",
                    width = 800,  # Set the figure width
                    height = 527  # Set the figure height
                )    
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Comparative Analysis":
        # Redirect to the Comparative Analysis page
        import comparative_analysis
        comparative_analysis.run(df)

    elif page == "Model Validation":
        # Redirect to the Model Validation page
        import model_validation
        model_validation.run_1(df)

else:
    st.error("The file 'Sample Data.csv' was not found in the current directory. Please make sure the file is present.")
