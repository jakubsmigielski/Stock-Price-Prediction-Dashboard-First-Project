import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Price Predictions", layout="wide")
st.title("Stock Price Prediction Dashboard")

st.markdown("""
This dashboard displays stock price predictions using linear regression.

**Features:**
- Actual vs predicted closing prices 
- Daily returns analysis & histogram
- Model performance metrics 
- <b>Sharpe Ratio</b> (risk-adjusted return)
- <b>Max Drawdown</b> (worst peak-to-trough drop)
- <b>Rolling volatility</b> (30-day std)
- <b>Correlation heatmap of daily returns</b>
- <b>Descriptive statistics</b> & <b>Full metrics table</b>
""", unsafe_allow_html=True)

metrics_df = pd.read_csv("results/metrics.csv")
company_option = st.selectbox("Select company:", ["AAPL", "TSLA", "GOOG", "ALL"])

@st.cache_data
def load_company_data(symbol):
    df = pd.read_csv(f"data/{symbol}.csv", skiprows=3,
                     names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df["Daily Return"] = df["Close"].pct_change()
    df["Cum Return"] = (1 + df["Daily Return"]).cumprod()
    return df

@st.cache_data
def load_predictions(symbol):
    df = pd.read_csv(f"data/{symbol}.csv", skiprows=3,
                     names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df["Next_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    model = LinearRegression()
    X = df[["Close"]]
    y = df["Next_Close"]
    model.fit(X, y)
    df["Predicted_Close"] = model.predict(X)
    return df, model

def sharpe_ratio(ret, rf=0):
    """Annualized Sharpe ratio, rf in decimal, ret daily return series"""
    mean = np.nanmean(ret)
    std = np.nanstd(ret)
    if std == 0 or np.isnan(std):
        return np.nan
    return (mean - rf/252) / std * np.sqrt(252)

def max_drawdown(cum_returns):
    """Max drawdown of cumulative returns series"""
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    return drawdown.min()

if company_option != "ALL":
    df = load_company_data(company_option)

    start_date, end_date = st.slider("Select date range:",
        min_value=df["Date"].min().date(),
        max_value=df["Date"].max().date(),
        value=(df["Date"].min().date(), df["Date"].max().date())
    )
    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    filtered_df = df[mask]

    with st.container():
        st.subheader(f"{company_option} – Stock Price")
        st.line_chart(filtered_df.set_index("Date")["Close"], use_container_width=True)

    with st.container():
        st.subheader("Prediction vs Actual")
        df_pred, model = load_predictions(company_option)
        df_pred = df_pred[(df_pred["Date"].dt.date >= start_date) & (df_pred["Date"].dt.date <= end_date)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["Close"], mode='lines', name='Actual Close'))
        fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["Predicted_Close"], mode='lines',
                                 name='Predicted Close', line=dict(dash='dash')))
        fig.update_layout(title=f"{company_option} – Price Prediction", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.subheader("Model Performance")
        row = metrics_df[metrics_df["Company"] == company_option]
        if not row.empty:
            st.metric("MAE", f"{row['MAE'].values[0]:.4f}")
            st.metric("R² Score", f"{row['R2'].values[0]:.4f}")
        else:
            st.warning("No metrics found.")
        last_close = df_pred["Close"].iloc[-1]
        next_day_pred = model.predict([[last_close]])[0]
        st.metric("Predicted Next Day Close", f"${next_day_pred:.2f}")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sharpe Ratio (annualized)", f"{sharpe_ratio(filtered_df['Daily Return']):.2f}", help="Higher = better risk-adjusted return")
        with col2:
            mdd = max_drawdown(filtered_df["Cum Return"].dropna())
            st.metric("Max Drawdown", f"{mdd*100:.2f}%", help="Largest peak-to-trough loss")

    with st.container():
        st.subheader("Daily Returns")
        st.line_chart(filtered_df.set_index("Date")["Daily Return"].dropna(), use_container_width=True)

    with st.container():
        st.subheader("Distribution of Daily Returns")
        fig_hist = px.histogram(
            filtered_df.dropna(),
            x="Daily Return",
            nbins=60,
            title="Histogram of Daily Returns",
            marginal="rug",
            opacity=0.75,
            color_discrete_sequence=["cornflowerblue"]
        )
        fig_hist.update_layout(
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            bargap=0.1
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Show descriptive statistics for daily returns", expanded=False):
        stats = filtered_df["Daily Return"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
        st.dataframe(stats)

    with st.container():
        st.subheader("Rolling Volatility (30-day std) – Daily Returns")
        filtered_df["Rolling Std"] = filtered_df["Daily Return"].rolling(window=30).std()
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=filtered_df["Date"], y=filtered_df["Rolling Std"],
            mode="lines", name="30-day rolling std"
        ))
        fig_vol.update_layout(title="30-Day Rolling Volatility", xaxis_title="Date", yaxis_title="Std (Volatility)")
        st.plotly_chart(fig_vol, use_container_width=True)

else:

    dfs = {comp: load_company_data(comp) for comp in ["AAPL", "TSLA", "GOOG"]}
    df_all = pd.DataFrame({"Date": dfs["AAPL"]["Date"]})
    for comp in ["AAPL", "TSLA", "GOOG"]:
        df_all[comp] = dfs[comp].set_index("Date")["Close"].reindex(df_all["Date"]).values
    df_returns = pd.DataFrame({"Date": dfs["AAPL"]["Date"]})
    for comp in ["AAPL", "TSLA", "GOOG"]:
        df_returns[comp] = dfs[comp].set_index("Date")["Daily Return"].reindex(df_returns["Date"]).values
    df_cumret = pd.DataFrame({"Date": dfs["AAPL"]["Date"]})
    for comp in ["AAPL", "TSLA", "GOOG"]:
        df_cumret[comp] = dfs[comp].set_index("Date")["Cum Return"].reindex(df_cumret["Date"]).values

    st.subheader("Combined Stock Prices")
    fig_all = go.Figure()
    for comp in ["AAPL", "TSLA", "GOOG"]:
        fig_all.add_trace(go.Scatter(x=dfs[comp]["Date"], y=dfs[comp]["Close"], name=comp, mode='lines'))
    fig_all.update_layout(title="Stock Prices – All Companies", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_all, use_container_width=True)

    st.subheader("Model Metrics – All Companies")
    st.dataframe(metrics_df.set_index("Company").round(4))


    st.subheader("Model Comparison – MAE and R²")
    import plotly.subplots as sp
    fig_bar = sp.make_subplots(rows=1, cols=2, subplot_titles=("MAE by Company", "R² Score by Company"))
    fig_bar.add_trace(go.Bar(x=metrics_df["Company"], y=metrics_df["MAE"], marker_color='coral', name="MAE"), row=1, col=1)
    fig_bar.add_trace(go.Bar(x=metrics_df["Company"], y=metrics_df["R2"], marker_color='skyblue', name="R²"), row=1, col=2)
    fig_bar.update_xaxes(title_text="Company", row=1, col=1)
    fig_bar.update_xaxes(title_text="Company", row=1, col=2)
    fig_bar.update_yaxes(title_text="MAE", row=1, col=1)
    fig_bar.update_yaxes(title_text="R²", row=1, col=2)
    fig_bar.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)


    st.subheader("Sharpe Ratio & Max Drawdown Comparison")
    sharpe = {}
    mdd = {}
    for comp in ["AAPL", "TSLA", "GOOG"]:
        sharpe[comp] = sharpe_ratio(df_returns[comp])
        mdd[comp] = max_drawdown(df_cumret[comp])
    comp_stats = pd.DataFrame({
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": {k: mdd[k]*100 for k in mdd}
    }).T
    st.dataframe(comp_stats)

    with st.expander("Show correlation heatmap of daily returns", expanded=True):
        corr = df_returns[["AAPL", "TSLA", "GOOG"]].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Correlation of Daily Returns"
        )
        fig_heat.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Show descriptive statistics for daily returns", expanded=False):
        stats = df_returns[["AAPL", "TSLA", "GOOG"]].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
        st.dataframe(stats)


    st.subheader("Rolling Volatility (30-day std) Comparison")
    fig_vol = go.Figure()
    for comp in ["AAPL", "TSLA", "GOOG"]:
        rolling_std = pd.Series(df_returns[comp]).rolling(window=30).std()
        fig_vol.add_trace(go.Scatter(
            x=df_returns["Date"], y=rolling_std,
            mode="lines", name=f"{comp} 30d std"
        ))
    fig_vol.update_layout(title="30-Day Rolling Volatility Comparison", xaxis_title="Date", yaxis_title="Std (Volatility)")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("All-in-One Comparison Table")
    table = pd.DataFrame()
    for comp in ["AAPL", "TSLA", "GOOG"]:
        stats = df_returns[comp].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
        row = {
            "Mean return": stats["mean"],
            "Std": stats["std"],
            "Min": stats["min"],
            "Max": stats["max"],
            "Sharpe Ratio": sharpe[comp],
            "Max Drawdown [%]": mdd[comp]*100,
            "MAE": metrics_df.set_index("Company").loc[comp, "MAE"],
            "R²": metrics_df.set_index("Company").loc[comp, "R2"],
        }
        table = table._append(pd.Series(row, name=comp))
    st.dataframe(table.round(4))

st.markdown("---")
st.subheader("About the Model")

st.markdown("""
This project uses a simple **linear regression** model to estimate the next day's closing stock price based on today's closing price.

**Sharpe ratio**: Annualized risk-adjusted return, assuming 0% risk-free rate.<br>
**Max drawdown**: Largest loss from a previous peak.<br>
**Volatility**: 30-day rolling std of daily returns.<br>

> This dashboard is for educational use only. It is not intended as financial advice.
""", unsafe_allow_html=True)









