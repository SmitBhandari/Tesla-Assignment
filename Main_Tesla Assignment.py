import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

# Setting page configuration
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

        # Cache the ARIMA model to improve performance
        @st.cache_data
        def get_arima_model(train_data):
            """Fit and cache the ARIMA model."""
            model = ARIMA(train_data, order=(2, 1, 3))  # Replace (2, 1, 3) with your desired parameters
            return model.fit()

        # Forecasting based on selected model
        def forecast_model(train, test, model_type):
            if model_type == "ARIMA":
                # Use cached ARIMA model
                arima_model = get_arima_model(train["Weekly_Sales"])
                return arima_model.forecast(steps=len(test))

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

        # Ensuring no negative forecast values as demand cannot be negative
        forecast = np.maximum(forecast, 0)

        # Aligning forecast with test index
        forecast = pd.Series(forecast, index=test.index)

        # Evaluating performance
        mad = np.mean(np.abs(test["Weekly_Sales"] - forecast))
        mse = np.mean((test["Weekly_Sales"] - forecast) ** 2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mbe = np.mean(forecast - test["Weekly_Sales"])  # Bias: Mean difference between forecasted and actual values

        # Weighted MAPE calculation
        wmape = (np.sum(np.abs(test["Weekly_Sales"] - forecast)) / np.sum(test["Weekly_Sales"])) * 100

        # Displaying scorecards on the left side
        col1, col2 = st.columns([1, 3])
        with col1:
            with st.container(border=True):   
                st.markdown("### Performance Metrics")
                st.metric(label="Mean Absolute Deviation (MAD)", value=f"{mad:.2f}")
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
                st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
                st.metric(label="Weighted Mean Absolute Percentage Error (WMAPE)", value=f"{wmape:.2f}%")
                st.metric(label="Mean Bias Error (MBE)", value=f"{mbe:.2f}")

        # Displaying the chart on the right side
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
            # Adding a text section below the line graph
            last_forecast_date = test.index[-1] if not test.empty else None
            last_forecast_value = forecast.iloc[-1] if not forecast.empty else None
            
            # Forecasting for the upcoming week
            if last_forecast_date is not None:
                # Extend the training data to include the test data
                extended_train = pd.concat([train, test])
            
                # Define the upcoming week's date
                upcoming_week_date = last_forecast_date + pd.DateOffset(weeks=1)
            
                # Forecast for the upcoming week
                if selected_model == "ARIMA":
                    # Use the cached ARIMA model
                    arima_model = get_arima_model(extended_train["Weekly_Sales"])
                    upcoming_week_forecast = arima_model.forecast(steps=1)
            
                elif selected_model == "Holt-Winters":
                    hw_model = ExponentialSmoothing(extended_train["Weekly_Sales"], trend="add", seasonal="add", seasonal_periods=52)
                    hw_fit = hw_model.fit()
                    upcoming_week_forecast = hw_fit.forecast(steps=1)
            
                elif selected_model == "Linear Regression":
                    extended_train["Time"] = np.arange(len(extended_train))
                    future_time = np.array([[len(extended_train)]])  # Time index for the upcoming week
                    lr_model = LinearRegression()
                    lr_model.fit(extended_train[["Time"]], extended_train["Weekly_Sales"])
                    upcoming_week_forecast = lr_model.predict(future_time)
            
                # Ensure no negative values
                upcoming_week_forecast_value = np.maximum(upcoming_week_forecast[0], 0)
            else:
                upcoming_week_date = None
                upcoming_week_forecast_value = None
            
            # Displaying the forecast summary
            if last_forecast_date and last_forecast_value is not None:
                st.markdown("### Forecast Summary")
                st.success(
                    f"For SKU **{selected_sku}**, the latest weekly forecast for the date (**{last_forecast_date.strftime('%Y-%m-%d')}**) "
                    f"at the **{selected_lag}** using the **{selected_model}** model is **{last_forecast_value:.2f}**.\n\n"
                    f"The forecast for the upcoming week (**{upcoming_week_date.strftime('%Y-%m-%d')}**) is **{upcoming_week_forecast_value:.2f}**."
                )
            else:
                st.warning("Forecast data is not available for the selected SKU, lag, or model.")

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
