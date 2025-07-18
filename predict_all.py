import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def load_and_prepare(path, name):
    df = pd.read_csv(path, names=["Date", "Close", "High", "Low", "Open", "Volume"], header=0)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[["Date", "Close"]].dropna()
    df["Next_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["Company"] = name
    return df


aapl = load_and_prepare("data/AAPL.csv", "AAPL")
tsla = load_and_prepare("data/TSLA.csv", "TSLA")
goog = load_and_prepare("data/GOOG.csv", "GOOG")

datasets = [aapl, tsla, goog]

import os

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

metrics = []

for df in datasets:
    company = df["Company"].iloc[0]

    X = df[["Close"]]
    y = df["Next_Close"]

    model = LinearRegression()
    model.fit(X, y)

    df["Predicted_Close"] = model.predict(X)

    mae = mean_absolute_error(y, df["Predicted_Close"])
    r2 = r2_score(y, df["Predicted_Close"])

    metrics.append({"Company": company, "MAE": mae, "R2": r2})


    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Actual Close")
    plt.plot(df["Date"], df["Predicted_Close"], linestyle="--", label="Predicted Close")
    plt.title(f"{company} – Price Prediction (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plt.savefig(f"plots/{company}_prediction.png")
    plt.close()

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("results/metrics.csv", index=False)

print("\n✅ Saved plots to 'plots/' and metrics to 'results/metrics.csv'")
