import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import plotly.express as px

@st.cache_data
def get_arima_model(train_data):
    """Cache the ARIMA model to improve performance."""
    model = ARIMA(train_data, order=(2, 1, 3))
    return model.fit()

def run_1(df):
    st.title("Monthly Forecast & Model Validation")

    # Sidebar: Dropdown for selecting SKU_ID
    st.sidebar.markdown("### Model Validation Options")
    sku_options = df["SKU_ID"].unique()
    selected_sku = st.sidebar.selectbox("Select SKU_ID", sku_options)

    # Filter data for the selected SKU_ID
    df_sku = df[df["SKU_ID"] == selected_sku]

    # Ensure the data is sorted by date
    df_sku = df_sku.sort_index()

    # Aggregate weekly data into monthly data
    df_sku_monthly = df_sku.resample('M').sum()  # Sum weekly sales to get monthly sales
    df_sku_monthly.rename(columns={"Weekly_Sales": "Monthly_Sales"}, inplace=True)  # Rename the column

    # Define forecast months
    forecast_months = {
        "January 2024": "2024-01-31",
        "February 2024": "2024-02-29",
        "March 2024": "2024-03-31",
        "April 2024": "2024-04-30",
        "May 2024": "2024-05-31",
        "June 2024": "2024-06-30"
    }

    # Forecasting function
    def forecast_model(train, test_length, model_type):
        if model_type == "ARIMA":
            arima_model = get_arima_model(train["Monthly_Sales"])
            return arima_model.forecast(steps=test_length)

        elif model_type == "Holt-Winters":
            model = ExponentialSmoothing(train["Monthly_Sales"], trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit()
            return fit.forecast(steps=test_length)

        elif model_type == "Linear Regression":
            train["Time"] = np.arange(len(train))
            future_time = np.arange(len(train), len(train) + test_length)
            lr_model = LinearRegression()
            lr_model.fit(train[["Time"]], train["Monthly_Sales"])
            return lr_model.predict(future_time.reshape(-1, 1))

    # Create a table for forecast results (January to June 2024)
    results = []
    for month_name, month_date in forecast_months.items():
        month_date = pd.Timestamp(month_date)

        # Define train-test split based on the selected month
        train_end_date = month_date - pd.DateOffset(months=1)
        train = df_sku_monthly.loc[:train_end_date]
        test = df_sku_monthly.loc[month_date:month_date]

        if test.empty or train.empty:
            # Handle missing data gracefully
            results.append({
                "Month": month_name,
                "Actual Value": "No Data",
                "ARIMA Forecast": "No Data",
                "Holt-Winters Forecast": "No Data",
                "Linear Regression Forecast": "No Data"
            })
            continue

        # Forecast using all three models
        arima_forecast = forecast_model(train, len(test), model_type="ARIMA")
        hw_forecast = forecast_model(train, len(test), model_type="Holt-Winters")
        lr_forecast = forecast_model(train, len(test), model_type="Linear Regression")

        # Collect the forecasts for the selected month
        results.append({
            "Month": month_name,
            "Actual Value": test["Monthly_Sales"].iloc[0] if not test.empty else "No Data",
            "ARIMA Forecast": arima_forecast[0] if len(arima_forecast) > 0 else "No Data",
            "Holt-Winters Forecast": hw_forecast[0] if len(hw_forecast) > 0 else "No Data",
            "Linear Regression Forecast": lr_forecast[0] if len(lr_forecast) > 0 else "No Data"
        })

    # Convert results to DataFrame for display in the table
    results_df = pd.DataFrame(results)

    # Display the table
    st.markdown("### Forecast Results (January to June 2024)")
    st.dataframe(results_df.style.format({
        "Actual Value": "{:.2f}" if results_df["Actual Value"].dtype != "object" else None,
        "ARIMA Forecast": "{:.2f}" if results_df["ARIMA Forecast"].dtype != "object" else None,
        "Holt-Winters Forecast": "{:.2f}" if results_df["Holt-Winters Forecast"].dtype != "object" else None,
        "Linear Regression Forecast": "{:.2f}" if results_df["Linear Regression Forecast"].dtype != "object" else None
    }).set_properties(**{
        "text-align": "center",
        "border-color": "black",
        "border-width": "1px",
        "border-style": "solid"
    }))

    # Create a line graph from the table
    st.markdown("### Forecast Comparison (January to June 2024)")
    if not results_df.empty:
        # Filter out rows with "No Data"
        graph_df = results_df.replace("No Data", np.nan).dropna()

        # Convert "Month" to datetime for proper plotting
        graph_df["Month"] = pd.to_datetime(graph_df["Month"], format="%B %Y")

        # Plot the line graph
        fig = px.line(
            graph_df,
            x="Month",
            y=["Actual Value", "ARIMA Forecast", "Holt-Winters Forecast", "Linear Regression Forecast"],
            title=f"Forecast Comparison for SKU {selected_sku}",
            labels={"value": "Monthly Sales", "variable": "Model"},
            markers=True
        )
        # Update the "Actual Demand" line to be green and dashed
        fig.for_each_trace(
            lambda trace: trace.update(
                line=dict(color="green", dash="dash")
            ) if trace.name == "Actual Value" else None
        )


        fig.update_layout(xaxis_title="Month", yaxis_title="Monthly Sales", legend_title="Model")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected SKU.")

    # Add the new graph for forecast comparison across lags
    st.markdown("### Forecast Comparison Across Lags (June 2024)")
    june_2024 = pd.Timestamp("2024-06-30")
    lag_results = []

    for lag_months in range(1, 13):
        train_end_date = june_2024 - pd.DateOffset(months=lag_months)
        train = df_sku_monthly.loc[:train_end_date]
        test_length = 1

        if train.empty:
            lag_results.append({
                "Lag (Months)": lag_months,
                "ARIMA Forecast": np.nan,
                "Holt-Winters Forecast": np.nan,
                "Linear Regression Forecast": np.nan,
                "Actual Demand": np.nan
            })
            continue

        arima_forecast = forecast_model(train, test_length, model_type="ARIMA")
        hw_forecast = forecast_model(train, test_length, model_type="Holt-Winters")
        lr_forecast = forecast_model(train, test_length, model_type="Linear Regression")

        lag_results.append({
            "Lag (Months)": lag_months,
            "ARIMA Forecast": arima_forecast[0] if len(arima_forecast) > 0 else np.nan,
            "Holt-Winters Forecast": hw_forecast[0] if len(hw_forecast) > 0 else np.nan,
            "Linear Regression Forecast": lr_forecast[0] if len(lr_forecast) > 0 else np.nan,
            "Actual Demand": df_sku_monthly.loc[june_2024, "Monthly_Sales"] if june_2024 in df_sku_monthly.index else np.nan
        })

    lag_results_df = pd.DataFrame(lag_results)

    # Plot the lag comparison graph
    fig_lag = px.line(
        lag_results_df,
        x="Lag (Months)",
        y=["ARIMA Forecast", "Holt-Winters Forecast", "Linear Regression Forecast", "Actual Demand"],
        markers=True,
        title=f"Forecast Comparison Across Lags for SKU {selected_sku} (June 2024)",
        labels={"value": "Monthly Sales", "variable": "Model"}
    )

    # Update the "Actual Demand" line to be green and dashed
    fig_lag.for_each_trace(
        lambda trace: trace.update(
            line=dict(color="green", dash="dash")
        ) if trace.name == "Actual Demand" else None
    )


    fig_lag.update_layout(xaxis_title="Lag (Months)", yaxis_title="Monthly Sales", legend_title="Model")
    st.plotly_chart(fig_lag, use_container_width=True)
