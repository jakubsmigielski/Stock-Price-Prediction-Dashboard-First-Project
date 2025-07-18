import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_and_prepare(path, ticker):
    df = pd.read_csv(path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df = df[["Date", "Close"]].dropna()
    df["Next_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["Company"] = ticker
    return df


aapl = load_and_prepare("data/AAPL.csv", "AAPL")
tsla = load_and_prepare("data/TSLA.csv", "TSLA")
goog = load_and_prepare("data/GOOG.csv", "GOOG")


datasets = [aapl, tsla, goog]
colors = {"AAPL": "blue", "TSLA": "orange", "GOOG": "green"}


plt.figure(figsize=(14, 7))

for df in datasets:
    company = df["Company"].iloc[0]
    X = df[["Close"]]
    y = df["Next_Close"]

    model = LinearRegression()
    model.fit(X, y)
    df["Predicted_Close"] = model.predict(X)


    plt.plot(df["Date"], df["Close"], label=f"{company} - Actual", color=colors[company], alpha=0.6)
    plt.plot(df["Date"], df["Predicted_Close"], label=f"{company} - Predicted", linestyle='--', color=colors[company])

plt.title("Actual vs Predicted Stock Prices (AAPL, TSLA, GOOG)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
