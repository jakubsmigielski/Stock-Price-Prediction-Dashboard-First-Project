import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

DATA_PATHS = {
    "AAPL": "data/AAPL.csv",
    "TSLA": "data/TSLA.csv",
    "GOOG": "data/GOOG.csv",
}

PLOT_DIR = "plots"
RESULTS_DIR = "results"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_prepare(path, name):
    df = pd.read_csv(path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[["Date", "Close"]].dropna()
    df["Next_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["Company"] = name
    return df

metrics = []

for company, path in DATA_PATHS.items():
    df = load_and_prepare(path, company)

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
    plt.savefig(f"{PLOT_DIR}/{company}_prediction.png")
    plt.close()

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
print("\n All predictions completed. Plots saved to 'plots/' and metrics to 'results/metrics.csv'")
