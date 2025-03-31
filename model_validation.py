import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import plotly.express as px
from pmdarima import auto_arima

@st.cache_data
def get_arima_model(train_data):
    """Cache the ARIMA model to improve performance."""
    return auto_arima(
        train_data,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )

def run_1(df):
    st.title("Model Validation")

    # Sidebar: Dropdown for selecting SKU_ID
    st.sidebar.markdown("### Model Validation Options")
    sku_options = df["SKU_ID"].unique()
    selected_sku = st.sidebar.selectbox("Select SKU_ID", sku_options)

    # Filter data for the selected SKU_ID
    df_sku = df[df["SKU_ID"] == selected_sku]

    # Forecasting function
    def forecast_model(train, test_length, model_type):
        if model_type == "ARIMA":
            # Use cached ARIMA model
            arima_model = get_arima_model(train["Weekly_Sales"])
            arima_model.fit(train["Weekly_Sales"])
            return arima_model.predict(n_periods=test_length)

        elif model_type == "Holt-Winters":
            model = ExponentialSmoothing(train["Weekly_Sales"], trend="add", seasonal="add", seasonal_periods=52)
            fit = model.fit()
            return fit.forecast(steps=test_length)

        elif model_type == "Linear Regression":
            train["Time"] = np.arange(len(train))
            future_time = np.arange(len(train), len(train) + test_length)
            lr_model = LinearRegression()
            lr_model.fit(train[["Time"]], train["Weekly_Sales"])
            return lr_model.predict(future_time.reshape(-1, 1))

    # Create a list to store the forecast results
    results = []

    # Precompute the test length for June 2024
    june_2024 = pd.Timestamp("2024-06-30")

    # Loop over lag values from 1 to 12
    for lag_months in range(1, 13):
        # Define train-test split based on lag
        train_end_date = june_2024 - pd.DateOffset(months=lag_months)
        train = df_sku.loc[:train_end_date]
        test_length = len(df_sku.loc[train_end_date:])

        if test_length == 0:
            # Skip if there's no test data
            results.append({
                "Lag (Months)": lag_months,
                "ARIMA Forecast": np.nan,
                "Holt-Winters Forecast": np.nan,
                "Linear Regression Forecast": np.nan,
                "Actual Demand": np.nan
            })
            continue

        # Forecast using all three models
        arima_forecast = forecast_model(train, test_length, model_type="ARIMA")
        hw_forecast = forecast_model(train, test_length, model_type="Holt-Winters")
        lr_forecast = forecast_model(train, test_length, model_type="Linear Regression")

        # Collect the forecasts for June 2024 for each model and lag
        results.append({
            "Lag (Months)": lag_months,
            "ARIMA Forecast": arima_forecast[-1] if len(arima_forecast) > 0 else np.nan,
            "Holt-Winters Forecast": hw_forecast[-1] if len(hw_forecast) > 0 else np.nan,
            "Linear Regression Forecast": lr_forecast[-1] if len(lr_forecast) > 0 else np.nan,
            "Actual Demand": df_sku.loc[june_2024, "Weekly_Sales"] if june_2024 in df_sku.index else np.nan
        })

    # Convert results to DataFrame for display in table format
    results_df = pd.DataFrame(results)

    # Display the table
    st.markdown("### Forecast Results for June 2024")
    st.dataframe(results_df)

    # Visualization: Plot the forecasts for each model across lags, including Actual Demand
    st.markdown("### Forecast Comparison Across Lags")
    fig = px.line(
        results_df,
        x="Lag (Months)",
        y=["ARIMA Forecast", "Holt-Winters Forecast", "Linear Regression Forecast", "Actual Demand"],
        markers=True,
        title=f"Forecast Comparison for SKU {selected_sku} (June 2024)",
        labels={"value": "Weekly Sales", "variable": "Model"},
    )

    # Update the "Actual Demand" line to be black and dashed
    fig.for_each_trace(
        lambda trace: trace.update(
            line=dict(color="black", dash="dash")
        ) if trace.name == "Actual Demand" else None
    )

    fig.update_layout(xaxis_title="Lag (Months)", yaxis_title="Weekly Sales", legend_title="Model")
    st.plotly_chart(fig, use_container_width=True)
