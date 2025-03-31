import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
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

@st.cache_data
def calculate_kpis(df, models, lags):
    """Calculate KPIs for all models and lags."""
    kpi_data = []

    for model in models:
        for lag in range(1, 13):
            # Calculate the train_end_date based on lag
            calculated_date = pd.Timestamp("2024-06-30") - pd.DateOffset(months=lag)

            # Check if the calculated date exists in the dataset
            if calculated_date not in df.index:
                # Find the closest date in the dataset
                train_end_date = df.index[np.abs(df.index - calculated_date).argmin()]
            else:
                train_end_date = calculated_date

            # Split data into train and test
            train = df.loc[:train_end_date]
            test = df.loc[train_end_date:]

            # Skip if test data is empty
            if test.empty:
                continue

            try:
                # Forecasting based on the model
                if model == "ARIMA":
                    arima_model = get_arima_model(train["Weekly_Sales"])
                    arima_model.fit(train["Weekly_Sales"])
                    forecast = arima_model.predict(n_periods=len(test))
                    forecast = arima_model.predict(n_periods=len(test))
                elif model == "Holt-Winters":
                    forecast = ExponentialSmoothing(train["Weekly_Sales"], trend="add", seasonal="add", seasonal_periods=52).fit().forecast(steps=len(test))
                elif model == "Linear Regression":
                    train["Time"] = np.arange(len(train))
                    test["Time"] = np.arange(len(train), len(train) + len(test))
                    lr_model = LinearRegression()
                    lr_model.fit(train[["Time"]], train["Weekly_Sales"])
                    forecast = lr_model.predict(test[["Time"]])

                # Ensure no negative forecast values
                forecast = np.maximum(forecast, 0)

                # Align forecast with test index
                forecast = pd.Series(forecast, index=test.index)

                # Calculate KPIs
                mad = np.mean(np.abs(test["Weekly_Sales"] - forecast))
                mbe = np.mean(forecast - test["Weekly_Sales"])  # Bias: Mean difference between forecasted and actual values

                # Manual calculation for MSE and RMSE
                mse = np.mean((test["Weekly_Sales"] - forecast) ** 2)  # Mean Squared Error
                rmse = np.sqrt(mse)  # Root Mean Squared Error

                wmape = (np.sum(np.abs(test["Weekly_Sales"] - forecast)) / np.sum(test["Weekly_Sales"])) * 100

                # Append KPI data
                kpi_data.append({
                    "Model": model,
                    "Lag": f"{lag}-Month Lag",
                    "MAD": mad,
                    "RMSE": rmse,
                    "MSE": mse,
                    "WMAPE": wmape,
                    "Bias": mbe
                })

            except Exception as e:
                st.warning(f"Forecasting failed for model {model} and lag {lag}-Month: {e}")
                continue

    # Create a DataFrame for KPIs
    kpi_df = pd.DataFrame(kpi_data)

    # Ensure the Lag column is ordered from 1 to 12 in ascending order
    kpi_df["Lag"] = pd.Categorical(kpi_df["Lag"], categories=[f"{i}-Month Lag" for i in range(1, 13)], ordered=True)

    return kpi_df


def run(df):
    st.title("Comparative Analysis")

    # Sidebar: Add options for SKU, KPI, and Lag selection
    st.sidebar.markdown("## Comparative Analysis Options")

    # SKU Selection
    st.sidebar.markdown("### Select SKU")
    sku_options = df["SKU_ID"].unique()
    selected_sku = st.sidebar.selectbox("Choose an SKU:", sku_options)

    # KPI Selection
    st.sidebar.markdown("### Select KPI")
    kpi_options = {
        "Mean Absolute Deviation (MAD)": "MAD",
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Squared Error (MSE)": "MSE",
        "Weighted Mean Absolute Percentage Error (WMAPE)": "WMAPE",
        "Mean Bias Error (Bias)": "Bias"
    }
    selected_kpi_full = st.sidebar.radio("Choose a KPI to display:", list(kpi_options.keys()))
    selected_kpi = kpi_options[selected_kpi_full]  # Map full name to column name

    # Lag Selection for Bar Chart
    st.sidebar.markdown("### Select Lag for Bar Chart")
    lag_options = [f"{i}-Month Lag" for i in range(1, 13)]
    selected_lag = st.sidebar.selectbox("Choose a Lag:", lag_options)

    # Filter data for the selected SKU
    df_sku = df[df["SKU_ID"] == selected_sku]

    # Generate lag options dynamically (1 to 12 months)
    models = ["ARIMA", "Holt-Winters", "Linear Regression"]

    # Calculate KPIs (cached)
    kpi_df = calculate_kpis(df_sku, models, lag_options)

    # Heatmap for KPI values
    st.subheader(f"Heatmap for {selected_kpi_full}")
    heatmap_fig = px.imshow(
        kpi_df.pivot_table(index="Model", columns="Lag", values=selected_kpi, aggfunc="mean"),
        labels=dict(x="Lag", y="Model", color=selected_kpi_full),
        color_continuous_scale="Reds",  # Red-to-white color scheme
        title=f"{selected_kpi_full} Heatmap",
        text_auto=".2f"  # Add data labels inside the heatmap rounded to 2 decimal points
    )
    heatmap_fig.update_layout(width=1100, height=500, margin=dict(l=10, r=10, t=40, b=10))

    # Display the heatmap
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Find the best KPI value across all models and lags
    if selected_kpi == "Bias":  # Handle MBE (Mean Bias Error)
        best_row = kpi_df.iloc[(kpi_df[selected_kpi].abs()).idxmin()]  # Find the row where MBE is closest to zero
    else:
        best_row = kpi_df.loc[kpi_df[selected_kpi].idxmin()]  # Find the row with the smallest KPI value

    best_model = best_row["Model"]
    best_lag = best_row["Lag"]
    best_kpi_value = best_row[selected_kpi]

    # Display the recommendation
    st.markdown("### Recommendation")
    if selected_kpi == "Bias":
        st.success(
            f"For SKU {selected_sku}, the **{selected_kpi_full}** is closest to zero at **{best_lag}** for the **{best_model}** model, "
            f"with a value of **{best_kpi_value:.2f}**."
        )
    else:
        st.success(
            f"For SKU {selected_sku}, the **{selected_kpi_full}** is the lowest at **{best_lag}** for the **{best_model}** model, "
            f"with a value of **{best_kpi_value:.2f}**."
        )

    # Filter the KPI DataFrame for the selected lag
    kpi_df_filtered = kpi_df[kpi_df["Lag"] == selected_lag]

    # Bar chart for accuracy comparison
    st.subheader(f"Bar Chart for {selected_kpi_full} ({selected_lag})")
    bar_chart_fig = px.bar(
        kpi_df_filtered,
        x="Model",
        y=selected_kpi,
        color="Model",
        title=f"{selected_kpi_full} Comparison Across Models ({selected_lag})",
        labels={selected_kpi: selected_kpi_full, "Model": "Forecasting Model"},
        color_discrete_map={
        "ARIMA": "#FF2400",  # Scarlet
        "Holt-Winters": "#FF6666",  # Medium red
        "Linear Regression": "#CC0000"  # Dark red
        }
    )
    bar_chart_fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # Add data labels
    bar_chart_fig.update_layout(width=1100, height=500, margin=dict(l=10, r=10, t=40, b=10))

    # Display the bar chart
    st.plotly_chart(bar_chart_fig, use_container_width=True)
