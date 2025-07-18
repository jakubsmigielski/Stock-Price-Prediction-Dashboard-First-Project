import pandas as pd
import matplotlib.pyplot as plt

column_names = ["Date", "Open", "High", "Low", "Close", "Volume"]

aapl = pd.read_csv("data/AAPL.csv", skiprows=3, names=column_names, parse_dates=["Date"], index_col="Date")
tsla = pd.read_csv("data/TSLA.csv", skiprows=3, names=column_names, parse_dates=["Date"], index_col="Date")
goog = pd.read_csv("data/GOOG.csv", skiprows=3, names=column_names, parse_dates=["Date"], index_col="Date")

plt.figure(figsize=(14, 7))
plt.plot(aapl["Close"], label="Apple (AAPL)")
plt.plot(tsla["Close"], label="Tesla (TSLA)")
plt.plot(goog["Close"], label="Google (GOOG)")
plt.title("Stock Closing Prices (2020–2024)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

aapl["Daily Return"] = aapl["Close"].pct_change()
tsla["Daily Return"] = tsla["Close"].pct_change()
goog["Daily Return"] = goog["Close"].pct_change()

returns = pd.DataFrame({
    "AAPL": aapl["Daily Return"],
    "TSLA": tsla["Daily Return"],
    "GOOG": goog["Daily Return"]
})

returns.dropna(inplace=True)

returns.plot(figsize=(14, 7), alpha=0.8)
plt.title("Daily Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

returns.hist(bins=50, figsize=(14, 6), layout=(1, 3), sharex=True)
plt.suptitle("Distribution of Daily Returns")
plt.tight_layout()
plt.show()

print("\nCorrelation matrix:\n", returns.corr())

aapl["30MA"] = aapl["Close"].rolling(window=30).mean()

plt.figure(figsize=(14, 7))
plt.plot(aapl["Close"], label="AAPL - Close")
plt.plot(aapl["30MA"], label="AAPL - 30 Day MA", linestyle="--")
plt.title("Apple (AAPL) Closing Price with 30-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n✔️ EDA finished successfully!")
